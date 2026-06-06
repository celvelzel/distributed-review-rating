#!/usr/bin/env python
"""T20: Stacking ensemble runner with Ridge meta-learner.

Combines OOF predictions from three base models:
  1. CatBoost  — 927 non-TFIDF features, OOF RMSE ≈ 0.548
  2. LightGBM  — 927 non-TFIDF features (regenerated via 5-fold CV)
  3. MLP       — 896 embedding features, OOF RMSE ≈ 1.152

Meta-learner: Ridge Regression (alpha=1.0), 5-fold CV stacking.

Outputs:
  artifacts/models/stacking_oof.npy   (3,007,439,)
  artifacts/models/stacking_test.npy  (10,000,)
  docs/changelog/stage-4-stacking.md  comparison report
"""

from __future__ import annotations

import gc
import importlib.util
import sys
import time
from pathlib import Path
from typing import List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import KFold

# ── path setup ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.utils.timer import timed, write_metrics

# Import stacking module
_spec = importlib.util.spec_from_file_location(
    "stacking", str(ROOT / "code" / "models" / "stacking.py"))
_stacking = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_stacking)
stack_models = _stacking.stack_models

# ── constants ──────────────────────────────────────────────────────────
FEAT_DIR = ROOT / "artifacts" / "features"
MODEL_DIR = ROOT / "artifacts" / "models"
ETL_DIR = ROOT / "artifacts" / "etl"
OUTPUT_DIR = ROOT / "output"

X_TRAIN_PATH = FEAT_DIR / "X_train.parquet"
X_TEST_PATH = FEAT_DIR / "X_test.parquet"
Y_TRAIN_PATH = FEAT_DIR / "y_train.npy"

# Base model artifacts
CATBOOST_OOF_PATH = MODEL_DIR / "catboost_oof.npy"
CATBOOST_TEST_PATH = MODEL_DIR / "catboost_test.npy"
MLP_OOF_PATH = MODEL_DIR / "mlp_oof.npy"
MLP_TEST_PATH = MODEL_DIR / "mlp_test.npy"

# LGB submission (for test predictions)
LGB_SUBMISSION_PATH = OUTPUT_DIR / "submission-stage2.csv"

# Stacking outputs
STACKING_OOF_PATH = MODEL_DIR / "stacking_oof.npy"
STACKING_TEST_PATH = MODEL_DIR / "stacking_test.npy"

CHANGELOG_PATH = ROOT / "docs" / "changelog" / "stage-4-stacking.md"
METRICS_PATH = ROOT / "docs" / "changelog" / "metrics.json"

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "n_estimators": 200,  # Reduced for OOF generation speed
    "verbose": -1,
    "random_seed": 42,
}

RANDOM_SEED = 42
N_FOLDS = 5


# ── LGB OOF generation ────────────────────────────────────────────────
def _select_columns(all_cols: List[str]) -> List[str]:
    """Return non-TFIDF column names (same as CatBoost training)."""
    return [c for c in all_cols if not c.startswith("tfidf_")]


def _read_parquet_cols(path: Path, columns: List[str]) -> np.ndarray:
    """Read selected columns from parquet → float32 numpy."""
    pf = pq.ParquetFile(str(path))
    n_groups = pf.metadata.num_row_groups
    chunks: List[np.ndarray] = []
    for rg in range(n_groups):
        table = pf.read_row_group(rg, columns=columns)
        df = table.to_pandas()
        arr = df.values.astype(np.float32)
        del df, table
        chunks.append(arr)
        gc.collect()
    X = np.vstack(chunks)
    del chunks
    gc.collect()
    return X


def generate_lgb_oof(
    columns: List[str],
    y_all: np.ndarray,
    n_folds: int = 5,
    n_sample: int = 500_000,
) -> Tuple[np.ndarray, List[float]]:
    """Train LightGBM with K-fold CV on a subsample for speed.

    Loads full data once, subsamples, trains K-fold, predicts full OOF.
    Returns (oof_preds, fold_rmses).
    """
    n_train = len(y_all)

    # Load full X_train once
    print(f"  Loading full X_train ({n_train:,} rows) …")
    X_full = _read_parquet_cols(X_TRAIN_PATH, columns)
    print(f"  X_full: {X_full.shape}")

    # Subsample for training
    rng = np.random.RandomState(RANDOM_SEED)
    sample_idx = np.sort(rng.choice(n_train, size=min(n_sample, n_train), replace=False))
    X_sub = X_full[sample_idx]
    y_sub = y_all[sample_idx]
    print(f"  Subsample: {X_sub.shape}")

    # K-fold on subsample
    oof_preds = np.zeros(n_train, dtype=np.float32)
    fold_rmses: List[float] = []

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_sub), 1):
        print(f"\n  ── LGB Fold {fold_idx}/{n_folds} "
              f"(train={len(tr_idx):,}, val={len(va_idx):,}) ──")

        X_tr, y_tr = X_sub[tr_idx], y_sub[tr_idx]
        X_va, y_va = X_sub[va_idx], y_sub[va_idx]

        ds_tr = lgb.Dataset(X_tr, y_tr, free_raw_data=True)
        ds_va = lgb.Dataset(X_va, y_va, free_raw_data=True)

        model = lgb.train(
            LGB_PARAMS, ds_tr,
            num_boost_round=LGB_PARAMS["n_estimators"],
            valid_sets=[ds_va],
            callbacks=[lgb.log_evaluation(50)],
        )

        # OOF on val fold (subsample indices)
        va_preds = np.clip(model.predict(X_va), 1.0, 5.0)
        oof_preds[sample_idx[va_idx]] = va_preds
        fold_rmse = float(np.sqrt(np.mean((va_preds - y_va) ** 2)))
        fold_rmses.append(fold_rmse)
        print(f"  Fold {fold_idx} RMSE: {fold_rmse:.5f}")

        del X_tr, y_tr, X_va, y_va, ds_tr, ds_va, model, va_preds
        gc.collect()

    # Predict remaining rows with full-data model
    unmask = oof_preds == 0
    if unmask.any():
        print(f"\n  Predicting {unmask.sum():,} remaining rows with full-data model …")
        ds_full = lgb.Dataset(X_sub, y_sub, free_raw_data=True)
        model_full = lgb.train(
            LGB_PARAMS, ds_full,
            num_boost_round=LGB_PARAMS["n_estimators"],
            callbacks=[lgb.log_evaluation(50)],
        )
        oof_preds[unmask] = np.clip(model_full.predict(X_full[unmask]), 1.0, 5.0)
        del ds_full, model_full
        gc.collect()

    del X_full, X_sub, y_sub
    gc.collect()
    return oof_preds, fold_rmses


def generate_lgb_test(columns: List[str]) -> np.ndarray:
    """Generate LGB test predictions by training on subsample.

    Loads X_test (small, 10K rows), trains on subsample, predicts.
    """
    print("\n  Training final LGB model on subsample for test predictions …")
    n_train = np.load(str(Y_TRAIN_PATH)).shape[0]
    rng = np.random.RandomState(RANDOM_SEED)
    sample_idx = np.sort(rng.choice(n_train, size=500_000, replace=False))

    X_train = _read_parquet_cols(X_TRAIN_PATH, columns)
    y_train_full = np.load(str(Y_TRAIN_PATH)).astype(np.float32)

    X_sub = X_train[sample_idx]
    y_sub = y_train_full[sample_idx]
    del X_train, y_train_full
    gc.collect()

    ds_full = lgb.Dataset(X_sub, y_sub, free_raw_data=True)
    model = lgb.train(
        LGB_PARAMS, ds_full,
        num_boost_round=LGB_PARAMS["n_estimators"],
        callbacks=[lgb.log_evaluation(50)],
    )

    # Load X_test
    print("  Loading X_test …")
    X_test = _read_parquet_cols(X_TEST_PATH, columns)
    test_preds = np.clip(model.predict(X_test), 1.0, 5.0).astype(np.float32)

    del X_sub, y_sub, ds_full, model, X_test
    gc.collect()
    return test_preds


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    t_start = time.perf_counter()
    print("=" * 60)
    print("T20: Stacking Ensemble with Ridge Meta-Learner")
    print("=" * 60)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load existing base model predictions ────────────────────────
    print("\n[1/5] Loading base model predictions …")
    t0 = time.perf_counter()

    # CatBoost
    catboost_oof = np.load(str(CATBOOST_OOF_PATH)).astype(np.float32)
    catboost_test = np.load(str(CATBOOST_TEST_PATH)).astype(np.float32)
    print(f"  CatBoost OOF: {catboost_oof.shape}  test: {catboost_test.shape}")

    # MLP
    mlp_oof = np.load(str(MLP_OOF_PATH)).astype(np.float32)
    mlp_test = np.load(str(MLP_TEST_PATH)).astype(np.float32)
    print(f"  MLP OOF: {mlp_oof.shape}  test: {mlp_test.shape}")

    # LGB test from submission CSV
    lgb_sub = pd.read_csv(LGB_SUBMISSION_PATH)
    lgb_test = lgb_sub["rating"].values.astype(np.float32)
    print(f"  LGB test (from submission): {lgb_test.shape}")
    del lgb_sub

    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # ── 2. Generate LGB OOF via 5-fold CV ──────────────────────────────
    print("\n[2/5] Generating LGB OOF predictions (5-fold CV) …")
    t0 = time.perf_counter()

    # Use same 927 non-TFIDF features as CatBoost
    pf = pq.ParquetFile(str(X_TRAIN_PATH))
    all_cols = pf.schema_arrow.names
    use_cols = _select_columns(all_cols)
    print(f"  Features: {len(use_cols)} / {len(all_cols)} columns "
          f"(dropped {len(all_cols) - len(use_cols)} TF-IDF)")

    y_train = np.load(str(Y_TRAIN_PATH)).astype(np.float32)

    lgb_oof, lgb_fold_rmses = generate_lgb_oof(use_cols, y_train, n_folds=N_FOLDS)
    lgb_oof_rmse = float(np.sqrt(np.mean((lgb_oof - y_train) ** 2)))
    lgb_cv_time = time.perf_counter() - t0
    print(f"\n  LGB OOF RMSE: {lgb_oof_rmse:.5f}  (CV time: {lgb_cv_time:.1f}s)")

    # ── 3. Compute individual model RMSEs ──────────────────────────────
    print("\n[3/5] Individual model OOF RMSEs …")
    catboost_rmse = float(np.sqrt(np.mean((catboost_oof - y_train) ** 2)))
    mlp_rmse = float(np.sqrt(np.mean((mlp_oof - y_train) ** 2)))
    print(f"  CatBoost OOF RMSE: {catboost_rmse:.5f}")
    print(f"  LightGBM  OOF RMSE: {lgb_oof_rmse:.5f}")
    print(f"  MLP       OOF RMSE: {mlp_rmse:.5f}")

    # ── 4. Run stacking ────────────────────────────────────────────────
    print("\n[4/5] Running stacking (Ridge meta-learner, 5-fold CV) …")
    t0 = time.perf_counter()

    oof_list = [lgb_oof, catboost_oof, mlp_oof]
    test_list = [lgb_test, catboost_test, mlp_test]
    model_names = ["LGB", "CatBoost", "MLP"]

    stacking_oof, stacking_test, coefficients, coeff_dict = stack_models(
        oof_list=oof_list,
        test_list=test_list,
        y_true=y_train,
        n_folds=N_FOLDS,
        alpha=1.0,
        random_seed=RANDOM_SEED,
        model_names=model_names,
    )

    stacking_time = time.perf_counter() - t0
    stacking_rmse = float(np.sqrt(np.mean((stacking_oof - y_train) ** 2)))
    print(f"\n  Stacking OOF RMSE: {stacking_rmse:.5f}  (time: {stacking_time:.1f}s)")

    # ── 5. Save outputs ────────────────────────────────────────────────
    print("\n[5/5] Saving outputs …")
    np.save(str(STACKING_OOF_PATH), stacking_oof)
    print(f"  OOF → {STACKING_OOF_PATH}  shape={stacking_oof.shape}")
    np.save(str(STACKING_TEST_PATH), stacking_test)
    print(f"  Test → {STACKING_TEST_PATH}  shape={stacking_test.shape}")

    # ── Update metrics.json ────────────────────────────────────────────
    total_time = time.perf_counter() - t_start
    metrics_update = {
        "stages": {
            "4": {
                "oof_rmse": round(stacking_rmse, 5),
                "lgb_oof_rmse": round(lgb_oof_rmse, 5),
                "catboost_oof_rmse": round(catboost_rmse, 5),
                "mlp_oof_rmse": round(mlp_rmse, 5),
                "ridge_coefficients": coeff_dict,
                "ridge_alpha": 1.0,
                "n_folds": N_FOLDS,
                "total_time_sec": round(total_time, 2),
                "model": "stacking_ridge",
                "base_models": model_names,
            }
        }
    }
    write_metrics(str(METRICS_PATH), metrics_update)
    print(f"  Metrics → {METRICS_PATH}")

    # ── Write changelog ────────────────────────────────────────────────
    write_changelog(
        stacking_rmse=stacking_rmse,
        lgb_oof_rmse=lgb_oof_rmse,
        catboost_rmse=catboost_rmse,
        mlp_rmse=mlp_rmse,
        coeff_dict=coeff_dict,
        lgb_fold_rmses=lgb_fold_rmses,
        lgb_cv_time=lgb_cv_time,
        stacking_time=stacking_time,
        total_time=total_time,
    )

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  CatBoost OOF RMSE: {catboost_rmse:.5f}")
    print(f"  LightGBM  OOF RMSE: {lgb_oof_rmse:.5f}")
    print(f"  MLP       OOF RMSE: {mlp_rmse:.5f}")
    print(f"  Stacking  OOF RMSE: {stacking_rmse:.5f}")
    print(f"\n  Ridge coefficients: {coeff_dict}")
    best_single = min(catboost_rmse, lgb_oof_rmse, mlp_rmse)
    delta = best_single - stacking_rmse
    print(f"  Δ vs best single ({best_single:.5f}): {delta:+.5f}  "
          f"{'✅ improved' if delta > 0 else '⚠️  worse'}")
    print(f"\n  Total time: {total_time:.1f}s")
    print("\n=== Done ===")


def write_changelog(
    stacking_rmse: float,
    lgb_oof_rmse: float,
    catboost_rmse: float,
    mlp_rmse: float,
    coeff_dict: dict,
    lgb_fold_rmses: List[float],
    lgb_cv_time: float,
    stacking_time: float,
    total_time: float,
) -> None:
    """Write the stage-4 stacking changelog."""
    CHANGELOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_single = min(catboost_rmse, lgb_oof_rmse, mlp_rmse)
    delta = best_single - stacking_rmse

    lines = [
        "# T20: Stacking Ensemble with Ridge Meta-Learner",
        "",
        "## Architecture",
        "- **Meta-learner**: Ridge Regression (α=1.0, fit_intercept=True)",
        "- **Base models**: LGB, CatBoost, MLP",
        "- **Stacking CV**: 5-fold (same seed as base models)",
        "- **Features**: Base model OOF predictions only (no raw features)",
        "",
        "## Base Model OOF RMSE",
        "",
        "| Model | OOF RMSE | Features | Notes |",
        "|-------|----------|----------|-------|",
        f"| CatBoost | {catboost_rmse:.5f} | 927 (non-TFIDF) | Best single model |",
        f"| LightGBM | {lgb_oof_rmse:.5f} | 927 (non-TFIDF) | Regenerated via 5-fold CV |",
        f"| MLP | {mlp_rmse:.5f} | 896 (embeddings) | DeBERTa + LightGCN |",
        f"| **Stacking** | **{stacking_rmse:.5f}** | — | Ridge meta-learner |",
        "",
        "## LightGBM OOF (Regenerated)",
        "",
        "| Fold | RMSE |",
        "|------|------|",
    ]
    for i, rmse in enumerate(lgb_fold_rmses, 1):
        lines.append(f"| {i} | {rmse:.5f} |")
    lines.append(f"| **Mean** | **{np.mean(lgb_fold_rmses):.5f}** |")

    lines.extend([
        "",
        "## Ridge Coefficients (Model Weights)",
        "",
        "| Model | Coefficient | Interpretation |",
        "|-------|-------------|----------------|",
    ])
    for name, coef in coeff_dict.items():
        pct = coef * 100
        lines.append(f"| {name} | {coef:.6f} | ~{pct:.1f}% relative weight |")

    lines.extend([
        "",
        "## Improvement",
        "",
        f"- Best single model RMSE: {best_single:.5f}",
        f"- Stacking OOF RMSE: {stacking_rmse:.5f}",
        f"- **Δ improvement: {delta:+.5f}** "
        f"({'✅ improved' if delta > 0 else '⚠️ worse'})",
        "",
        "## Timing",
        "",
        f"- LGB OOF generation: {lgb_cv_time:.1f}s",
        f"- Stacking CV: {stacking_time:.1f}s",
        f"- Total time: {total_time:.1f}s",
        "",
        "## Outputs",
        "",
        f"- Stacking OOF: `artifacts/models/stacking_oof.npy` ({3_007_439:,},)",
        f"- Stacking test: `artifacts/models/stacking_test.npy` ({10_000:,})",
        "",
        "## Notes",
        "",
        "- LGB OOF regenerated using 927 non-TFIDF features (same as CatBoost) via 5-fold CV",
        "- LGB test predictions loaded from `output/submission-stage2.csv` (trained on 5927 features)",
        "- Ridge Regression chosen over neural network meta-learner to prevent overfitting",
        "- Predictions clipped to [1.0, 5.0]",
    ])

    CHANGELOG_PATH.write_text("\n".join(lines) + "\n")
    print(f"  Changelog → {CHANGELOG_PATH}")


if __name__ == "__main__":
    main()

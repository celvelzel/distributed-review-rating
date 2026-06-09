#!/usr/bin/env python
"""Diverse ensemble: LightGBM + XGBoost + MLP weighted average (T22).

Combines OOF predictions from three diverse models:
  1. LightGBM  — TF-IDF 5000-dim, OOF RMSE ≈ 1.176
  2. XGBoost   — TF-IDF 5000-dim, OOF RMSE = 1.202
  3. MLP v2    — BERT 768-dim,    OOF RMSE = 1.131

Strategy: Simple weighted average ensemble.
  - Tries multiple weight combinations (equal, LGB-heavy, MLP-heavy, etc.)
  - Evaluates via OOF RMSE
  - Saves best ensemble test predictions

Diversity sources:
  - LGB vs XGBoost: same features, different algorithms (tree vs gradient boosted)
  - MLP: completely different feature space (BERT embeddings vs TF-IDF)
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

# ── path setup ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── constants ──────────────────────────────────────────────────────────
TRAIN_PATH = ROOT / "artifacts" / "etl" / "train.parquet"
TEST_PATH = ROOT / "artifacts" / "etl" / "test.parquet"
MODEL_DIR = ROOT / "artifacts" / "models"
OUTPUT_DIR = ROOT / "output"
FEAT_DIR = ROOT / "artifacts" / "features"

XGB_OOF_PATH = MODEL_DIR / "xgboost_oof.npy"
XGB_TEST_PATH = MODEL_DIR / "xgboost_test.npy"
MLP_OOF_PATH = MODEL_DIR / "mlp_oof.npy"
MLP_TEST_PATH = MODEL_DIR / "mlp_test.npy"

LGB_OOF_PATH = MODEL_DIR / "lgb_tfidf_oof.npy"
LGB_TEST_PATH = MODEL_DIR / "lgb_tfidf_test.npy"

ENSEMBLE_OOF_PATH = MODEL_DIR / "ensemble_diverse_oof.npy"
ENSEMBLE_TEST_PATH = MODEL_DIR / "ensemble_diverse_test.npy"

SUBMISSION_PATH = OUTPUT_DIR / "submission-ensemble-diverse.csv"
METRICS_PATH = ROOT / "docs" / "changelog" / "metrics.json"

# Precomputed char TF-IDF features (faster than recomputing)
CHARTFIDF_TRAIN = FEAT_DIR / "chartfidf_train.npz"
CHARTFIDF_TEST = FEAT_DIR / "chartfidf_test.npz"

RANDOM_SEED = 42
N_FOLDS = 5
MAX_FEATURES = 5000
N_SAMPLE = 200_000  # subsample for LGB training speed

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.1,
    "num_leaves": 63,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "random_seed": RANDOM_SEED,
}


# ── helpers ────────────────────────────────────────────────────────────
def _combine_text(df: pd.DataFrame) -> pd.Series:
    """Concatenate review title and comment."""
    return (df["title"].fillna("") + " " + df["comment"].fillna("")).str.strip()


def extract_tfidf_features(
    train_texts: pd.Series,
    test_texts: pd.Series,
    max_features: int = 5000,
) -> Tuple:
    """Fit TF-IDF on train and transform both splits."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_train = vectorizer.fit_transform(train_texts.fillna(""))
    X_test = vectorizer.transform(test_texts.fillna(""))
    return X_train, X_test, vectorizer


# ── LightGBM OOF generation (subsampled) ──────────────────────────────
def generate_lgb_oof(
    X_all,
    y_all: np.ndarray,
    X_test,
    params: dict,
    n_folds: int = 5,
    n_sample: int = 500_000,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Train LightGBM K-fold CV on a subsample for speed.

    Loads full data, subsamples, trains K-fold, predicts full OOF.
    Returns (oof_preds, test_preds, fold_rmses).
    """
    n_train = X_all.shape[0]
    n_test = X_test.shape[0]

    # Subsample for training speed
    rng = np.random.RandomState(RANDOM_SEED)
    sample_idx = np.sort(rng.choice(n_train, size=min(n_sample, n_train), replace=False))
    X_sub = X_all[sample_idx]
    y_sub = y_all[sample_idx]
    print(f"  Subsample: {X_sub.shape} ({len(sample_idx):,} / {n_train:,})")

    oof_preds = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(n_test, dtype=np.float32)
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
            params,
            ds_tr,
            num_boost_round=params["n_estimators"],
            valid_sets=[ds_va],
            callbacks=[lgb.log_evaluation(100)],
        )

        # OOF on val fold (subsample indices)
        va_preds = np.clip(model.predict(X_va), 1.0, 5.0)
        oof_preds[sample_idx[va_idx]] = va_preds
        fold_rmse = float(np.sqrt(np.mean((va_preds - y_va) ** 2)))
        fold_rmses.append(fold_rmse)
        print(f"  Fold {fold_idx} RMSE: {fold_rmse:.5f}")

        # Test predictions (accumulate)
        test_preds += np.clip(model.predict(X_test), 1.0, 5.0) / n_folds

        del X_tr, y_tr, X_va, y_va, ds_tr, ds_va, model, va_preds
        gc.collect()

    # Predict remaining rows with full-subsample model
    unmask = oof_preds == 0
    if unmask.any():
        print(f"\n  Predicting {unmask.sum():,} remaining rows with full-subsample model …")
        ds_full = lgb.Dataset(X_sub, y_sub, free_raw_data=True)
        model_full = lgb.train(
            params, ds_full,
            num_boost_round=params["n_estimators"],
            callbacks=[lgb.log_evaluation(100)],
        )
        oof_preds[unmask] = np.clip(model_full.predict(X_all[unmask]), 1.0, 5.0)
        del ds_full, model_full
        gc.collect()

    return oof_preds, test_preds, fold_rmses


# ── ensemble evaluation ────────────────────────────────────────────────
def evaluate_ensemble(
    oof_list: List[np.ndarray],
    weights: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Compute OOF RMSE for a given weight combination."""
    ensemble = oof_list[0] * weights[0]
    for i in range(1, len(oof_list)):
        ensemble = ensemble + oof_list[i] * weights[i]
    ensemble = np.clip(ensemble, 1.0, 5.0)
    return float(np.sqrt(np.mean((ensemble - y_true) ** 2)))


def make_test_predictions(
    test_list: List[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Compute weighted average of test predictions."""
    ensemble = np.zeros_like(test_list[0], dtype=np.float32)
    for test_pred, w in zip(test_list, weights):
        ensemble += w * test_pred
    return np.clip(ensemble, 1.0, 5.0)


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    t_start = time.perf_counter()
    print("=" * 60)
    print("T22: Diverse Ensemble (LGB + XGBoost + MLP)")
    print("=" * 60)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load existing OOF predictions ───────────────────────────────
    print("\n[1/5] Loading existing OOF predictions …")
    t0 = time.perf_counter()

    xgb_oof = np.load(str(XGB_OOF_PATH)).astype(np.float32)
    xgb_test = np.load(str(XGB_TEST_PATH)).astype(np.float32)
    print(f"  XGBoost OOF: {xgb_oof.shape}  test: {xgb_test.shape}")

    mlp_oof = np.load(str(MLP_OOF_PATH)).astype(np.float32)
    mlp_test = np.load(str(MLP_TEST_PATH)).astype(np.float32)
    print(f"  MLP      OOF: {mlp_oof.shape}  test: {mlp_test.shape}")

    # Load y_train
    y_train_path = FEAT_DIR / "y_train.npy"
    if y_train_path.exists():
        y_train = np.load(str(y_train_path)).astype(np.float32)
        print(f"  y_train (from .npy): {y_train.shape}  mean={y_train.mean():.3f}")
    else:
        train_df = pd.read_parquet(TRAIN_PATH, columns=["rating"])
        y_train = train_df["rating"].values.astype(np.float32)
        del train_df
    gc.collect()

    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # Load test IDs
    test_ids = pd.read_parquet(TEST_PATH, columns=["id"])["id"].values

    # ── 2. Generate LightGBM OOF via char TF-IDF 5-fold CV ─────────────
    if LGB_OOF_PATH.exists() and LGB_TEST_PATH.exists():
        print("\n[2/5] Loading existing LightGBM OOF …")
        lgb_oof = np.load(str(LGB_OOF_PATH)).astype(np.float32)
        lgb_test = np.load(str(LGB_TEST_PATH)).astype(np.float32)
        lgb_oof_rmse = float(np.sqrt(np.mean((lgb_oof - y_train) ** 2)))
        lgb_fold_rmses = []  # not available from cached
        print(f"  LightGBM OOF: {lgb_oof.shape}  RMSE: {lgb_oof_rmse:.5f}")
    else:
        print("\n[2/5] Generating LightGBM OOF (char TF-IDF 5-fold CV) …")
        t0 = time.perf_counter()

        # Load precomputed char TF-IDF features
        from scipy import sparse
        print("  Loading precomputed char TF-IDF features …")
        X_train_tfidf = sparse.load_npz(str(CHARTFIDF_TRAIN))
        X_test_tfidf = sparse.load_npz(str(CHARTFIDF_TEST))
        print(f"  TF-IDF: train={X_train_tfidf.shape}  test={X_test_tfidf.shape}")

        lgb_oof, lgb_test, lgb_fold_rmses = generate_lgb_oof(
            X_train_tfidf, y_train, X_test_tfidf, LGB_PARAMS,
            n_folds=N_FOLDS, n_sample=N_SAMPLE,
        )
        lgb_oof_rmse = float(np.sqrt(np.mean((lgb_oof - y_train) ** 2)))
        lgb_time = time.perf_counter() - t0
        print(f"\n  LightGBM OOF RMSE: {lgb_oof_rmse:.5f}  (time: {lgb_time:.1f}s)")
        print(f"  Fold RMSEs: {[f'{r:.5f}' for r in lgb_fold_rmses]}")

        # Save LGB OOF for future use
        np.save(str(LGB_OOF_PATH), lgb_oof)
        np.save(str(LGB_TEST_PATH), lgb_test)
        print(f"  LGB OOF → {LGB_OOF_PATH}  test → {LGB_TEST_PATH}")

        # Free TF-IDF memory
        del X_train_tfidf, X_test_tfidf
        gc.collect()

    # ── 3. Compute individual model OOF RMSEs ──────────────────────────
    print("\n[3/5] Individual model OOF RMSEs …")
    xgb_oof_rmse = float(np.sqrt(np.mean((xgb_oof - y_train) ** 2)))
    mlp_oof_rmse = float(np.sqrt(np.mean((mlp_oof - y_train) ** 2)))
    print(f"  LightGBM  OOF RMSE: {lgb_oof_rmse:.5f}")
    print(f"  XGBoost   OOF RMSE: {xgb_oof_rmse:.5f}")
    print(f"  MLP       OOF RMSE: {mlp_oof_rmse:.5f}")

    # ── 4. Try weight combinations ─────────────────────────────────────
    print("\n[4/5] Evaluating weight combinations …")

    oof_list = [lgb_oof, xgb_oof, mlp_oof]
    test_list = [lgb_test, xgb_test, mlp_test]
    model_names = ["LGB", "XGBoost", "MLP"]

    # Weight combos: (LGB, XGBoost, MLP)
    weight_configs: Dict[str, np.ndarray] = {
        "equal":        np.array([1/3, 1/3, 1/3]),
        "lgb_heavy":    np.array([0.50, 0.20, 0.30]),
        "mlp_heavy":    np.array([0.30, 0.20, 0.50]),
        "best_two":     np.array([0.00, 0.30, 0.70]),   # skip worst (XGB)
        "lgb_mlp":      np.array([0.40, 0.00, 0.60]),   # LGB + MLP only
        "mlp_dominant": np.array([0.20, 0.10, 0.70]),   # MLP gets most
        "inverse_rmse": None,  # computed below
    }

    # Compute inverse-RMSE weights (higher weight for lower RMSE)
    rmse_vals = np.array([lgb_oof_rmse, xgb_oof_rmse, mlp_oof_rmse])
    inv_weights = (1.0 / rmse_vals)
    inv_weights /= inv_weights.sum()
    weight_configs["inverse_rmse"] = inv_weights

    # Fine grid search around MLP-heavy region
    print("  Running fine grid search …")
    best_grid_rmse = float("inf")
    best_grid_weights = None
    for w_mlp_100 in range(50, 101):  # MLP weight: 0.50 to 1.00
        w_mlp = w_mlp_100 / 100.0
        for w_lgb_100 in range(0, 51):  # LGB weight: 0.00 to 0.50
            w_lgb = w_lgb_100 / 100.0
            w_xgb = 1.0 - w_mlp - w_lgb
            if w_xgb < -0.001:
                continue
            w_xgb = max(0.0, w_xgb)
            weights = np.array([w_lgb, w_xgb, w_mlp])
            rmse = evaluate_ensemble(oof_list, weights, y_train)
            if rmse < best_grid_rmse:
                best_grid_rmse = rmse
                best_grid_weights = weights.copy()

    weight_configs["grid_best"] = best_grid_weights
    print(f"  Grid best: LGB={best_grid_weights[0]:.2f}, XGB={best_grid_weights[1]:.2f}, "
          f"MLP={best_grid_weights[2]:.2f} → RMSE={best_grid_rmse:.5f}")

    # Evaluate all
    results: List[Tuple[str, np.ndarray, float]] = []
    print(f"\n  {'Config':<16s}  {'LGB':>6s}  {'XGB':>6s}  {'MLP':>6s}  {'OOF RMSE':>10s}")
    print(f"  {'─'*16}  {'─'*6}  {'─'*6}  {'─'*6}  {'─'*10}")

    for name, weights in weight_configs.items():
        rmse = evaluate_ensemble(oof_list, weights, y_train)
        results.append((name, weights, rmse))
        print(f"  {name:<16s}  {weights[0]:.2f}  {weights[1]:.2f}  {weights[2]:.2f}  {rmse:.5f}")

    # Find best
    results.sort(key=lambda x: x[2])
    best_name, best_weights, best_rmse = results[0]
    print(f"\n  ★ Best: {best_name}  weights={best_weights}  OOF RMSE={best_rmse:.5f}")

    # ── 5. Save best ensemble ──────────────────────────────────────────
    print("\n[5/5] Saving best ensemble predictions …")

    # OOF
    best_oof = np.zeros_like(oof_list[0], dtype=np.float32)
    for oof, w in zip(oof_list, best_weights):
        best_oof += w * oof
    best_oof = np.clip(best_oof, 1.0, 5.0)

    np.save(str(ENSEMBLE_OOF_PATH), best_oof)
    print(f"  OOF → {ENSEMBLE_OOF_PATH}  shape={best_oof.shape}")

    # Test
    best_test = make_test_predictions(test_list, best_weights)
    np.save(str(ENSEMBLE_TEST_PATH), best_test)
    print(f"  Test → {ENSEMBLE_TEST_PATH}  shape={best_test.shape}")

    # Submission CSV
    submission = pd.DataFrame({"id": test_ids, "rating": best_test})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"  Submission → {SUBMISSION_PATH}")

    # ── Summary ────────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  LightGBM  OOF RMSE: {lgb_oof_rmse:.5f}")
    print(f"  XGBoost   OOF RMSE: {xgb_oof_rmse:.5f}")
    print(f"  MLP       OOF RMSE: {mlp_oof_rmse:.5f}")
    print(f"  ─────────────────────────────────")
    print(f"  Best ensemble ({best_name}): {best_rmse:.5f}")
    print(f"  Weights: LGB={best_weights[0]:.2f}, XGB={best_weights[1]:.2f}, MLP={best_weights[2]:.2f}")

    best_single = min(lgb_oof_rmse, xgb_oof_rmse, mlp_oof_rmse)
    delta = best_single - best_rmse
    print(f"  Δ vs best single ({best_single:.5f}): {delta:+.5f}  "
          f"{'✅ improved' if delta > 0 else '⚠️  worse'}")

    if best_rmse < 1.10:
        print(f"\n  ✅ OOF RMSE < 1.10 target achieved!")
    else:
        print(f"\n  ⚠️  OOF RMSE > 1.10 — consider adding more diverse models")

    print(f"\n  Total time: {total_time:.1f}s")
    print("\n=== Done ===")

    # ── Update metrics.json ────────────────────────────────────────────
    try:
        from code.utils.timer import write_metrics
        metrics_update = {
            "stages": {
                "ensemble_diverse": {
                    "oof_rmse": round(best_rmse, 5),
                    "best_config": best_name,
                    "weights": {name: round(float(w), 4) for name, w in zip(model_names, best_weights)},
                    "lgb_oof_rmse": round(lgb_oof_rmse, 5),
                    "xgb_oof_rmse": round(xgb_oof_rmse, 5),
                    "mlp_oof_rmse": round(mlp_oof_rmse, 5),
                    "lgb_fold_rmses": [round(r, 5) for r in lgb_fold_rmses],
                    "total_time_sec": round(total_time, 2),
                    "model": "ensemble_weighted_avg",
                    "base_models": model_names,
                }
            }
        }
        write_metrics(str(METRICS_PATH), metrics_update)
        print(f"  Metrics → {METRICS_PATH}")
    except Exception as e:
        print(f"  (metrics update skipped: {e})")


if __name__ == "__main__":
    main()

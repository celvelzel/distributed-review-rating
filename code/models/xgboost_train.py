#!/usr/bin/env python
"""XGBoost base model for ensemble diversity (T19).

5-fold OOF XGBoost regressor on TF-IDF 5000-dim features.
Produces OOF predictions and test predictions for the stacking ensemble.

TF-IDF is leak-free (verified in leakage audit).
Uses native xgb.train() with DMatrix for efficient sparse handling.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

# ── path setup ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.models.tfidf_baseline import extract_tfidf_features
from code.utils.timer import write_metrics

# ── constants ──────────────────────────────────────────────────────────
TRAIN_PATH = ROOT / "artifacts" / "etl" / "train.parquet"
TEST_PATH = ROOT / "artifacts" / "etl" / "test.parquet"
MODEL_DIR = ROOT / "artifacts" / "models"
METRICS_PATH = ROOT / "docs" / "changelog" / "metrics.json"

OOF_PATH = MODEL_DIR / "xgboost_oof.npy"
TEST_PATH_NPY = MODEL_DIR / "xgboost_test.npy"

RANDOM_SEED = 42
N_FOLDS = 5
MAX_FEATURES = 5000
NUM_BOOST_ROUND = 150
EARLY_STOPPING_ROUNDS = 20

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.15,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
    "tree_method": "hist",
    "max_bin": 128,
    "seed": RANDOM_SEED,
    "nthread": -1,
    "verbosity": 0,
}


# ── helpers ────────────────────────────────────────────────────────────
def _combine_text(df: pd.DataFrame) -> pd.Series:
    """Concatenate review *title* and *comment* with a space separator."""
    return (df["title"].fillna("") + " " + df["comment"].fillna("")).str.strip()


# ── cross-validation + OOF ────────────────────────────────────────────
def train_xgb_oof(
    X_all,
    y_all: np.ndarray,
    X_test,
    params: dict,
    n_folds: int = 5,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 30,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Train XGBoost K-fold CV using native xgb.train() with DMatrix.

    Returns (oof_preds, test_preds, fold_rmses).
    """
    n_train = X_all.shape[0]
    n_test = X_test.shape[0]
    oof_preds = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(n_test, dtype=np.float32)
    fold_rmses: List[float] = []

    # Convert test set to DMatrix once (small, 10K rows)
    dtest = xgb.DMatrix(X_test)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(np.empty(n_train)), 1):
        print(f"\n  ── Fold {fold_idx}/{n_folds} "
              f"(train={len(tr_idx):,}, val={len(va_idx):,}) ──")

        # Create DMatrix for this fold (sparse slicing is efficient)
        dtrain = xgb.DMatrix(X_all[tr_idx], label=y_all[tr_idx])
        dval = xgb.DMatrix(X_all[va_idx], label=y_all[va_idx])

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, "val")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=50,
        )

        # OOF predictions (validation fold)
        best_iter = model.best_iteration
        va_preds = np.clip(model.predict(dval, iteration_range=(0, best_iter + 1)), 1.0, 5.0)
        oof_preds[va_idx] = va_preds
        fold_rmse = float(np.sqrt(np.mean((va_preds - y_all[va_idx]) ** 2)))
        fold_rmses.append(fold_rmse)
        print(f"  Fold {fold_idx} RMSE: {fold_rmse:.5f}  (best_iter={best_iter})")

        # Test predictions (accumulate for averaging)
        test_preds += np.clip(
            model.predict(dtest, iteration_range=(0, best_iter + 1)), 1.0, 5.0
        ) / n_folds

        del dtrain, dval, model
        import gc
        gc.collect()

    return oof_preds, test_preds, fold_rmses


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("XGBoost base model for ensemble diversity (T19)")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data ──────────────────────────────────────────────────────
    print("\n[1/4] Loading data …")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    print(f"  train: {len(train_df):,} rows  |  test: {len(test_df):,} rows")

    # 2. Build TF-IDF features ──────────────────────────────────────────
    print(f"\n[2/4] Extracting TF-IDF features (max_features={MAX_FEATURES}) …")
    train_texts = _combine_text(train_df)
    test_texts = _combine_text(test_df)
    X_train, X_test, _vectorizer = extract_tfidf_features(
        train_texts, test_texts, max_features=MAX_FEATURES,
    )
    y_train = train_df["rating"].values.astype(np.float32)

    # Free text data
    del train_df, test_df, train_texts, test_texts
    import gc
    gc.collect()

    print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")

    # 3. 5-fold CV training ─────────────────────────────────────────────
    print("\n[3/4] Training XGBoost (5-fold CV) …")
    start_train = time.perf_counter()

    oof_preds, test_preds, fold_rmses = train_xgb_oof(
        X_train, y_train, X_test, XGB_PARAMS,
        n_folds=N_FOLDS,
        num_boost_round=NUM_BOOST_ROUND,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    train_time = time.perf_counter() - start_train
    print(f"\n  Training completed in {train_time:.1f}s")

    # Compute overall OOF RMSE
    oof_preds = np.clip(oof_preds, 1.0, 5.0)
    oof_rmse = float(np.sqrt(np.mean((oof_preds - y_train) ** 2)))

    mean_fold_rmse = float(np.mean(fold_rmses))
    fold_std = float(np.std(fold_rmses))
    print(f"\n  Fold RMSEs: {[f'{r:.5f}' for r in fold_rmses]}")
    print(f"  Mean fold RMSE: {mean_fold_rmse:.5f} ± {fold_std:.5f}")
    print(f"  Overall OOF RMSE: {oof_rmse:.5f}")

    # 4. Save predictions ───────────────────────────────────────────────
    print("\n[4/4] Saving predictions …")
    np.save(str(OOF_PATH), oof_preds)
    print(f"  OOF → {OOF_PATH}  shape={oof_preds.shape}")

    np.save(str(TEST_PATH_NPY), test_preds)
    print(f"  Test → {TEST_PATH_NPY}  shape={test_preds.shape}")

    # Update metrics.json
    metrics_update = {
        "stages": {
            "xgboost": {
                "oof_rmse": round(oof_rmse, 5),
                "mean_fold_rmse": round(mean_fold_rmse, 5),
                "fold_std": round(fold_std, 5),
                "fold_rmses": [round(r, 5) for r in fold_rmses],
                "train_time_sec": round(train_time, 2),
                "model": "xgboost",
                "features": f"tfidf_{MAX_FEATURES}",
                "params": {
                    "learning_rate": XGB_PARAMS["learning_rate"],
                    "max_depth": XGB_PARAMS["max_depth"],
                    "num_boost_round": NUM_BOOST_ROUND,
                    "subsample": XGB_PARAMS["subsample"],
                    "colsample_bytree": XGB_PARAMS["colsample_bytree"],
                    "reg_alpha": XGB_PARAMS["reg_alpha"],
                    "reg_lambda": XGB_PARAMS["reg_lambda"],
                    "n_folds": N_FOLDS,
                },
            }
        }
    }
    write_metrics(str(METRICS_PATH), metrics_update)
    print(f"  Metrics → {METRICS_PATH}")

    # Summary
    print(f"\n  XGBoost OOF RMSE: {oof_rmse:.5f}")
    print(f"  Fold variance: {fold_std:.5f}")
    if oof_rmse < 1.20:
        print("  ✅ OOF RMSE < 1.20 target achieved")
    else:
        print("  ⚠️  OOF RMSE > 1.20 — consider tuning")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

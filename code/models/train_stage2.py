#!/usr/bin/env python
"""Stage 2: ALL features (TF-IDF + stats + embeddings) + LightGBM → Kaggle submission.

Loads pre-assembled feature matrices:
- X_train.parquet  (3,007,439 × ~900+ features)
- X_test.parquet   (10,000     × ~900+ features)
- y_train.npy      (3,007,439,) target ratings
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import KFold

# ── path setup ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.utils.timer import StageTimer, timed, write_metrics

# ── constants ──────────────────────────────────────────────────────────
FEAT_DIR = ROOT / "artifacts" / "features"
X_TRAIN_PATH = FEAT_DIR / "X_train.parquet"
X_TEST_PATH = FEAT_DIR / "X_test.parquet"
Y_TRAIN_PATH = FEAT_DIR / "y_train.npy"

SUBMISSION_PATH = ROOT / "output" / "submission-stage2.csv"
METRICS_PATH = ROOT / "docs" / "changelog" / "metrics.json"

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "n_estimators": 500,
    "verbose": -1,
}

RANDOM_SEED = 42


# ── data loading (memory-efficient via PyArrow) ────────────────────────
def load_features():
    """Load pre-assembled feature matrices via PyArrow, then convert to pandas.

    Returns
    -------
    X_train  : pd.DataFrame  (3M × ~900+)
    X_test   : pd.DataFrame  (10K × ~900+)
    y_train  : np.ndarray    (3M,)
    test_ids : np.ndarray    (10K,) — review ids inferred from row order
    """
    print("  Loading X_train via PyArrow …")
    table_train = pq.read_table(str(X_TRAIN_PATH))
    X_train = table_train.to_pandas()
    del table_train
    print(f"  X_train shape: {X_train.shape}  mem={X_train.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    print("  Loading X_test via PyArrow …")
    table_test = pq.read_table(str(X_TEST_PATH))
    X_test = table_test.to_pandas()
    del table_test
    print(f"  X_test shape: {X_test.shape}")

    print("  Loading y_train …")
    y_train = np.load(str(Y_TRAIN_PATH)).astype(np.float32)
    print(f"  y_train shape: {y_train.shape}")

    # Try to extract test ids from the test parquet; fall back to 0..9999
    if "id" in X_test.columns:
        test_ids = X_test["id"].values
        X_test = X_test.drop(columns=["id"])
        if "id" in X_train.columns:
            X_train = X_train.drop(columns=["id"])
    else:
        # Load raw test parquet to get ids
        raw_test_path = ROOT / "artifacts" / "etl" / "test.parquet"
        if raw_test_path.exists():
            raw = pd.read_parquet(raw_test_path, columns=["id"])
            test_ids = raw["id"].values
        else:
            test_ids = np.arange(len(X_test))

    # Drop any leftover non-feature columns
    for col in ["rating", "label", "target"]:
        if col in X_train.columns:
            X_train = X_train.drop(columns=[col])

    # Fill NaN with 0
    n_nan_train = X_train.isna().sum().sum()
    n_nan_test = X_test.isna().sum().sum()
    if n_nan_train > 0 or n_nan_test > 0:
        print(f"  Filling NaN: train={n_nan_train:,}  test={n_nan_test:,}")
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)

    # Ensure same column order and set
    common_cols = list(set(X_train.columns) & set(X_test.columns))
    if len(common_cols) < len(X_train.columns):
        train_only = set(X_train.columns) - set(X_test.columns)
        print(f"  Dropping {len(train_only)} train-only columns: {sorted(train_only)[:10]}…")
    if len(common_cols) < len(X_test.columns):
        test_only = set(X_test.columns) - set(X_train.columns)
        print(f"  Dropping {len(test_only)} test-only columns: {sorted(test_only)[:10]}…")
    common_cols.sort()
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    print(f"  Final feature count: {len(common_cols)}")
    return X_train, X_test, y_train, test_ids


# ── cross-validation ──────────────────────────────────────────────────
def cv_rmse(X_all: pd.DataFrame, y_all: np.ndarray, n_splits: int = 3) -> float:
    """Return mean RMSE across *n_splits* folds."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    rmses: List[float] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all), 1):
        X_tr = X_all.iloc[train_idx]
        X_val = X_all.iloc[val_idx]
        y_tr, y_val = y_all[train_idx], y_all[val_idx]

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_tr, y_tr)
        preds = np.clip(model.predict(X_val), 1.0, 5.0)
        rmse = float(np.sqrt(np.mean((preds - y_val) ** 2)))
        rmses.append(rmse)
        print(f"  fold {fold}: RMSE = {rmse:.5f}")

    mean_rmse = float(np.mean(rmses))
    print(f"  mean CV RMSE = {mean_rmse:.5f}")
    return mean_rmse


# ── timed helpers ──────────────────────────────────────────────────────
@timed("stage_2", "train_time_sec")
def _train_full(X_train: pd.DataFrame, y_train: np.ndarray) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_train, y_train)
    return model


@timed("stage_2", "inference_time_sec")
def _predict_and_save(
    model: lgb.LGBMRegressor,
    X_test: pd.DataFrame,
    test_ids: np.ndarray,
    output_path: str,
) -> pd.DataFrame:
    preds = np.clip(model.predict(X_test), 1.0, 5.0)
    submission = pd.DataFrame({"id": test_ids, "rating": preds})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("Stage 2: ALL features (multimodal) + LightGBM")
    print("=" * 60)

    # 1. Load features ──────────────────────────────────────────────────
    print("\n[1/5] Loading pre-assembled feature matrices …")
    t0 = time.perf_counter()
    X_train, X_test, y_train, test_ids = load_features()
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.1f}s  |  train: {len(X_train):,}  test: {len(X_test):,}")

    # 2. Cross-validation ───────────────────────────────────────────────
    print("\n[2/5] 3-fold cross-validation …")
    mean_rmse = cv_rmse(X_train, y_train, n_splits=3)

    # 3. Train full model ───────────────────────────────────────────────
    print("\n[3/5] Training full model on all training data …")
    timer = StageTimer()
    model = _train_full(X_train, y_train, stage_timer=timer)

    # 4. Predict + save submission ──────────────────────────────────────
    print("\n[4/5] Predicting on test set …")
    submission = _predict_and_save(
        model, X_test, test_ids, str(SUBMISSION_PATH), stage_timer=timer,
    )
    print(f"  Submission → {SUBMISSION_PATH}  ({len(submission):,} rows)")

    # 5. Update metrics.json ────────────────────────────────────────────
    print("\n[5/5] Updating metrics …")
    timings = timer.to_dict().get("stage_2", {})
    metrics_update = {
        "stages": {
            "2": {
                "rmse": round(mean_rmse, 5),
                "train_time_sec": round(timings.get("train_time_sec", 0.0), 2),
                "inference_time_sec": round(timings.get("inference_time_sec", 0.0), 2),
                "model": "lgb_multimodal",
                "features": [
                    "all",
                    "tfidf",
                    "user_stats",
                    "prod_stats",
                    "temporal",
                    "text_length",
                    "te_user",
                    "te_prod",
                    "bert_embeddings",
                    "price_features",
                    "category_stats",
                ],
            }
        }
    }
    write_metrics(str(METRICS_PATH), metrics_update)
    print(f"  Metrics → {METRICS_PATH}")

    # ── feature importance ─────────────────────────────────────────────
    print("\n  Top-20 Feature Importance (gain):")
    importance = model.feature_importances_
    feat_names = list(X_train.columns)
    imp_df = (
        pd.DataFrame({"feature": feat_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    for i, row in imp_df.iterrows():
        print(f"    {row['feature']:40s} {row['importance']:>12,.0f}")

    # Save importance for the report
    imp_path = ROOT / "artifacts" / "features" / "stage2_feature_importance.csv"
    imp_df_full = (
        pd.DataFrame({"feature": feat_names, "importance": importance})
        .sort_values("importance", ascending=False)
    )
    imp_df_full.to_csv(imp_path, index=False)
    print(f"  Full importance → {imp_path}")

    print(f"\n  stage_2 RMSE: {mean_rmse:.5f}  (stage_0: 1.17626  stage_1: 0.54975)")
    delta_vs_s1 = 0.54975 - mean_rmse
    print(f"  Δ vs stage_1: {delta_vs_s1:+.5f}  {'✅ improved' if delta_vs_s1 > 0 else '⚠️  worse'}")
    delta_vs_s0 = 1.17626 - mean_rmse
    print(f"  Δ vs stage_0: {delta_vs_s0:+.5f}  {'✅ improved' if delta_vs_s0 > 0 else '⚠️  worse'}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

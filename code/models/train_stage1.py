#!/usr/bin/env python
"""Stage 1: Statistical features + LightGBM → Kaggle submission.

Features used:
- user_stats   (user_id → avg_rating, num_reviews, avg_votes, purchased_rate, rating_std)
- product_stats (parent_prod_id → prod_avg_rating, prod_num_reviews, prod_price, prod_rating_number, main_category)
- temporal     (id → year, month, day, weekday, hour, is_weekend, is_holiday_season)
- text_length  (id → title_len, comment_len, title_comment_ratio, has_caps, has_exclamation)
- te_user      (id → user_te)   target-encoded user
- te_prod      (id → prod_te)   target-encoded product
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# ── path setup ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.utils.timer import StageTimer, timed, write_metrics

# ── constants ──────────────────────────────────────────────────────────
TRAIN_PATH = ROOT / "artifacts" / "etl" / "train.parquet"
TEST_PATH = ROOT / "artifacts" / "etl" / "test.parquet"

FEAT_DIR = ROOT / "artifacts" / "features"
USER_STATS_PATH = FEAT_DIR / "user_stats.parquet"
PROD_STATS_PATH = FEAT_DIR / "product_stats.parquet"
TEMPORAL_PATH = FEAT_DIR / "temporal.parquet"
TEXT_LEN_PATH = FEAT_DIR / "text_length.parquet"
TE_USER_PATH = FEAT_DIR / "te_user.parquet"
TE_PROD_PATH = FEAT_DIR / "te_prod.parquet"

SUBMISSION_PATH = ROOT / "output" / "submission-stage1.csv"
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


# ── feature loading ────────────────────────────────────────────────────
def load_features() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, np.ndarray]:
    """Load all feature parquets and join into train/test DataFrames.

    Returns
    -------
    X_train  : feature DataFrame for training
    X_test   : feature DataFrame for testing
    y_train  : target ratings
    test_ids : review ids for submission
    """
    print("  Loading base data …")
    train_df = pd.read_parquet(TRAIN_PATH, columns=[
        "id", "user_id", "parent_prod_id", "rating",
    ])
    test_df = pd.read_parquet(TEST_PATH, columns=[
        "id", "user_id", "parent_prod_id",
    ])
    y_train = train_df["rating"].values.astype(np.float32)
    test_ids = test_df["id"].values

    # ── user-level features (1 row per user) ───────────────────────────
    print("  Loading user_stats …")
    user_stats = pd.read_parquet(USER_STATS_PATH)
    user_stats = user_stats.drop_duplicates(subset="user_id")
    train_df = train_df.merge(user_stats, on="user_id", how="left")
    test_df = test_df.merge(user_stats, on="user_id", how="left")

    # ── product-level features (1 row per product) ─────────────────────
    print("  Loading product_stats …")
    prod_stats = pd.read_parquet(PROD_STATS_PATH)
    prod_stats = prod_stats.drop_duplicates(subset="parent_prod_id")
    # Label-encode main_category
    cat_col = "main_category"
    all_cats = prod_stats[cat_col].astype(str)
    cat_codes = pd.Categorical(all_cats).codes
    prod_stats[cat_col] = cat_codes
    train_df = train_df.merge(prod_stats, on="parent_prod_id", how="left")
    test_df = test_df.merge(prod_stats, on="parent_prod_id", how="left")

    # ── row-level features (1 row per review id) ───────────────────────
    for name, path in [
        ("temporal", TEMPORAL_PATH),
        ("text_length", TEXT_LEN_PATH),
        ("te_user", TE_USER_PATH),
        ("te_prod", TE_PROD_PATH),
    ]:
        print(f"  Loading {name} …")
        feat = pd.read_parquet(path)
        feat = feat.drop_duplicates(subset="id")
        train_df = train_df.merge(feat, on="id", how="left")
        test_df = test_df.merge(feat, on="id", how="left")

    # ── drop non-feature columns ───────────────────────────────────────
    drop_cols = ["id", "user_id", "parent_prod_id", "rating"]
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    # Fill any remaining NaN with 0 (e.g. cold-start users/products)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    print(f"  Feature columns ({len(X_train.columns)}): {list(X_train.columns)}")
    print(f"  X_train shape: {X_train.shape}  |  X_test shape: {X_test.shape}")
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
@timed("stage_1", "train_time_sec")
def _train_full(X_train: pd.DataFrame, y_train: np.ndarray) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_train, y_train)
    return model


@timed("stage_1", "inference_time_sec")
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
    print("Stage 1: Statistical features + LightGBM")
    print("=" * 60)

    # 1. Load features ──────────────────────────────────────────────────
    print("\n[1/5] Loading and joining features …")
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
    timings = timer.to_dict().get("stage_1", {})
    metrics_update = {
        "stages": {
            "1": {
                "rmse": round(mean_rmse, 5),
                "train_time_sec": round(timings.get("train_time_sec", 0.0), 2),
                "inference_time_sec": round(timings.get("inference_time_sec", 0.0), 2),
                "model": "lgb_stats",
                "features": [
                    "user_stats",
                    "prod_stats",
                    "temporal",
                    "text_length",
                    "te_user",
                    "te_prod",
                ],
            }
        }
    }
    write_metrics(str(METRICS_PATH), metrics_update)
    print(f"  Metrics → {METRICS_PATH}")
    print(f"\n  stage_1 RMSE: {mean_rmse:.5f}  (baseline stage_0: 1.17626)")
    delta = 1.17626 - mean_rmse
    print(f"  Δ vs baseline: {delta:+.5f}  {'✅ improved' if delta > 0 else '⚠️  worse'}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

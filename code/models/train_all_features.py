#!/usr/bin/env python
"""Train LightGBM, XGBoost, CatBoost with ALL features for ensemble.

Features used (all dense, 41 columns):
  - sentiment.parquet (VADER + TextBlob sentiment scores) — 17 cols
  - rating_deviation.parquet (User/Product/Category deviation) — 6 cols
  - product_metadata.parquet (Feature count, store features) — 8 cols
  - user_stats_kfold.parquet (K-Fold user statistics) — 5 cols
  - product_stats_kfold.parquet (K-Fold product statistics) — 5 cols

Note: TF-IDF (5000-dim) cannot be included due to 32 GB RSS cgroup limit.
      The dense feature matrix (~500 MB) fits comfortably.

Models (5-fold OOF):
  1. LightGBM  (n_estimators=1000, num_leaves=127, lr=0.05)
  2. XGBoost   (n_estimators=1000, max_depth=6, lr=0.05)
  3. CatBoost  (iterations=1000, depth=6, lr=0.05)

Outputs:
  - artifacts/models/lgb_allfeatures_oof.npy
  - artifacts/models/lgb_allfeatures_test.npy
  - artifacts/models/xgboost_allfeatures_oof.npy
  - artifacts/models/xgboost_allfeatures_test.npy
  - artifacts/models/catboost_allfeatures_oof.npy
  - artifacts/models/catboost_allfeatures_test.npy
"""

from __future__ import annotations

import gc
import sys
import time
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# ── path setup ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────
FEAT_DIR = ROOT / "artifacts" / "features"
ETL_DIR = ROOT / "artifacts" / "etl"
MODEL_DIR = ROOT / "artifacts" / "models"

RANDOM_SEED = 42
N_FOLDS = 5


# ══════════════════════════════════════════════════════════════════════
# Feature Loading
# ══════════════════════════════════════════════════════════════════════

def _compute_test_sentiment(test_df: pd.DataFrame) -> pd.DataFrame:
    """Compute sentiment features for test data (10K rows, fast)."""
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from textblob import TextBlob

    analyzer = SentimentIntensityAnalyzer()
    n = len(test_df)

    title = test_df["title"].fillna("").astype(str)
    comment = test_df["comment"].fillna("").astype(str)

    # VADER scores
    vader_cols = ["pos", "neg", "neu", "compound"]
    title_vader = np.zeros((n, 4), dtype=np.float32)
    comment_vader = np.zeros((n, 4), dtype=np.float32)

    for i in range(n):
        t_scores = analyzer.polarity_scores(title.iloc[i])
        c_scores = analyzer.polarity_scores(comment.iloc[i])
        for j, col in enumerate(vader_cols):
            title_vader[i, j] = t_scores[col]
            comment_vader[i, j] = c_scores[col]

    # TextBlob scores
    title_pol = np.zeros(n, dtype=np.float32)
    title_sub = np.zeros(n, dtype=np.float32)
    comment_pol = np.zeros(n, dtype=np.float32)
    comment_sub = np.zeros(n, dtype=np.float32)

    for i in range(n):
        t_blob = TextBlob(title.iloc[i])
        c_blob = TextBlob(comment.iloc[i])
        title_pol[i] = t_blob.sentiment.polarity
        title_sub[i] = t_blob.sentiment.subjectivity
        comment_pol[i] = c_blob.sentiment.polarity
        comment_sub[i] = c_blob.sentiment.subjectivity

    # Word counts
    pos_words = {
        "good", "great", "excellent", "love", "best", "nice", "wonderful",
        "perfect", "amazing", "awesome", "fantastic", "happy", "pleasant",
        "beautiful", "outstanding", "superb", "enjoy", "like", "favorite",
        "recommend", "comfortable", "impressive", "quality", "reliable",
        "easy", "fast", "helpful", "sturdy", "durable", "worth",
        "satisfied", "smooth", "elegant", "solid", "premium", "brilliant",
        "delighted", "pleased", "terrific", "fabulous", "magnificent",
    }
    neg_words = {
        "bad", "poor", "terrible", "worst", "horrible", "hate", "awful",
        "waste", "disappointing", "broken", "cheap", "useless", "defective",
        "ugly", "uncomfortable", "annoying", "frustrating", "difficult",
        "slow", "expensive", "flimsy", "junk", "trash", "garbage",
        "return", "refund", "complaint", "problem", "issue", "fail",
        "failed", "failure", "disappointed", "unhappy", "angry",
        "disgusting", "pathetic", "inferior", "mediocre", "lousy",
    }

    title_pos = np.zeros(n, dtype=np.int32)
    title_neg = np.zeros(n, dtype=np.int32)
    comment_pos = np.zeros(n, dtype=np.int32)
    comment_neg = np.zeros(n, dtype=np.int32)

    for i in range(n):
        t_words = set(title.iloc[i].lower().split())
        c_words = set(comment.iloc[i].lower().split())
        title_pos[i] = len(t_words & pos_words)
        title_neg[i] = len(t_words & neg_words)
        comment_pos[i] = len(c_words & pos_words)
        comment_neg[i] = len(c_words & neg_words)

    # Sentiment agreement
    sentiment_agreement = np.abs(title_vader[:, 3] - comment_vader[:, 3]).astype(np.float32)

    result = pd.DataFrame({
        "vader_title_pos": title_vader[:, 0],
        "vader_title_neg": title_vader[:, 1],
        "vader_title_neu": title_vader[:, 2],
        "vader_title_compound": title_vader[:, 3],
        "vader_comment_pos": comment_vader[:, 0],
        "vader_comment_neg": comment_vader[:, 1],
        "vader_comment_neu": comment_vader[:, 2],
        "vader_comment_compound": comment_vader[:, 3],
        "tb_title_polarity": title_pol,
        "tb_title_subjectivity": title_sub,
        "tb_comment_polarity": comment_pol,
        "tb_comment_subjectivity": comment_sub,
        "title_pos_words": title_pos,
        "title_neg_words": title_neg,
        "comment_pos_words": comment_pos,
        "comment_neg_words": comment_neg,
        "sentiment_agreement": sentiment_agreement,
    })
    return result


def _split_train_test_3m(df: pd.DataFrame, drop_cols: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a 3,017,439-row feature file into train/test.

    The file contains train (3,007,439 unique IDs) + test (10,000 rows, IDs 0-9999)
    sorted lexicographically by ID.  For duplicate IDs the first row is train,
    the second is test.
    """
    train = (df.drop_duplicates(subset="id", keep="first")
               .sort_values("id")
               .reset_index(drop=True)
               .drop(columns=drop_cols))

    test = (df[df.duplicated(subset="id", keep="first")]
              .sort_values("id")
              .reset_index(drop=True)
              .drop(columns=drop_cols))

    return train, test


def load_dense_features() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load all dense features.

    Returns:
        X_train: (3007439, 41) float32
        X_test:  (10000, 41) float32
        y_train: (3007439,) float32
    """
    log.info("=" * 60)
    log.info("Loading ALL dense features (41 cols)")
    log.info("=" * 60)

    # 1. Sentiment
    log.info("\n[1/5] Loading sentiment features …")
    sent_train = pd.read_parquet(FEAT_DIR / "sentiment.parquet").drop(columns=["id"])
    test_df = pd.read_parquet(ETL_DIR / "test.parquet", columns=["id", "title", "comment"])
    log.info("  Computing test sentiment (10K rows) …")
    sent_test = _compute_test_sentiment(test_df)
    del test_df
    log.info(f"  train: {sent_train.shape}  test: {sent_test.shape}")

    # 2. Rating deviation
    log.info("\n[2/5] Loading rating deviation …")
    rd = pd.read_parquet(FEAT_DIR / "rating_deviation.parquet")
    rd_train, rd_test = _split_train_test_3m(rd, drop_cols=["id"])
    del rd
    log.info(f"  train: {rd_train.shape}  test: {rd_test.shape}")

    # 3. Product metadata
    log.info("\n[3/5] Loading product metadata …")
    pm = pd.read_parquet(FEAT_DIR / "product_metadata.parquet")
    train_meta = pd.read_parquet(ETL_DIR / "train.parquet", columns=["parent_prod_id"])
    test_meta = pd.read_parquet(ETL_DIR / "test.parquet", columns=["parent_prod_id"])
    pm_train = train_meta.merge(pm, on="parent_prod_id", how="left").drop(columns=["parent_prod_id"]).fillna(0)
    pm_test = test_meta.merge(pm, on="parent_prod_id", how="left").drop(columns=["parent_prod_id"]).fillna(0)
    del pm, train_meta, test_meta
    log.info(f"  train: {pm_train.shape}  test: {pm_test.shape}")

    # 4. User stats kfold
    log.info("\n[4/5] Loading user stats kfold …")
    us = pd.read_parquet(FEAT_DIR / "user_stats_kfold.parquet")
    us_train, us_test = _split_train_test_3m(us, drop_cols=["id"])
    del us
    log.info(f"  train: {us_train.shape}  test: {us_test.shape}")

    # 5. Product stats kfold
    log.info("\n[5/5] Loading product stats kfold …")
    ps = pd.read_parquet(FEAT_DIR / "product_stats_kfold.parquet")
    ps_train, ps_test = _split_train_test_3m(ps, drop_cols=["id", "parent_prod_id"])
    del ps

    # Label encode main_category
    le = LabelEncoder()
    all_cats = pd.concat([ps_train["main_category"], ps_test["main_category"]]).fillna("unknown").astype(str)
    le.fit(all_cats)
    ps_train["main_category"] = le.transform(ps_train["main_category"].fillna("unknown").astype(str))
    ps_test["main_category"] = le.transform(ps_test["main_category"].fillna("unknown").astype(str))
    log.info(f"  train: {ps_train.shape}  test: {ps_test.shape}")

    # Combine
    log.info("\nCombining dense features …")
    dense_train = pd.concat([sent_train, rd_train, pm_train, us_train, ps_train], axis=1)
    dense_test = pd.concat([sent_test, rd_test, pm_test, us_test, ps_test], axis=1)
    del sent_train, sent_test, rd_train, rd_test, pm_train, pm_test
    del us_train, us_test, ps_train, ps_test
    gc.collect()

    # Ensure numeric
    for col in dense_train.columns:
        if dense_train[col].dtype == object:
            dense_train[col] = pd.to_numeric(dense_train[col], errors="coerce").fillna(0)
            dense_test[col] = pd.to_numeric(dense_test[col], errors="coerce").fillna(0)

    dense_train = dense_train.fillna(0).astype(np.float32)
    dense_test = dense_test.fillna(0).astype(np.float32)

    X_train = dense_train.values
    X_test = dense_test.values
    del dense_train, dense_test
    gc.collect()

    log.info(f"\n  X_train: {X_train.shape}  ({X_train.nbytes / 1e6:.1f} MB)")
    log.info(f"  X_test:  {X_test.shape}  ({X_test.nbytes / 1e6:.1f} MB)")

    # Load targets
    y_train = np.load(str(FEAT_DIR / "y_train.npy")).astype(np.float32)
    log.info(f"  y_train: {y_train.shape}  range: [{y_train.min():.1f}, {y_train.max():.1f}]")

    return X_train, X_test, y_train


# ══════════════════════════════════════════════════════════════════════
# Model Training
# ══════════════════════════════════════════════════════════════════════

def train_lgb_oof(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    n_folds: int = N_FOLDS,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Train LightGBM with 5-fold OOF."""
    import lightgbm as lgb

    log.info("\n" + "=" * 60)
    log.info("Training LightGBM (ALL features, 41 dense cols)")
    log.info("=" * 60)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 1000,
        "num_leaves": 127,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 5,
        "verbose": -1,
        "n_jobs": -1,
        "seed": RANDOM_SEED,
    }

    n_train = X_train.shape[0]
    oof_preds = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(X_test.shape[0], dtype=np.float32)
    fold_rmses = []

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(np.empty(n_train)), 1):
        log.info(f"\n  ── Fold {fold_idx}/{n_folds} (train={len(tr_idx):,}, val={len(va_idx):,}) ──")
        t0 = time.perf_counter()

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train[tr_idx], y_train[tr_idx],
            eval_set=[(X_train[va_idx], y_train[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
        )

        va_preds = np.clip(model.predict(X_train[va_idx]), 1.0, 5.0)
        oof_preds[va_idx] = va_preds
        fold_rmse = float(np.sqrt(np.mean((va_preds - y_train[va_idx]) ** 2)))
        fold_rmses.append(fold_rmse)

        test_preds += np.clip(model.predict(X_test), 1.0, 5.0) / n_folds

        elapsed = time.perf_counter() - t0
        log.info(f"  Fold {fold_idx} RMSE: {fold_rmse:.5f}  "
                 f"(best_iter={model.best_iteration_}, {elapsed:.1f}s)")

        del model
        gc.collect()

    return oof_preds, test_preds, fold_rmses


def train_xgb_oof(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    n_folds: int = N_FOLDS,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Train XGBoost with 5-fold OOF."""
    import xgboost as xgb

    log.info("\n" + "=" * 60)
    log.info("Training XGBoost (ALL features, 41 dense cols)")
    log.info("=" * 60)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 5,
        "tree_method": "hist",
        "seed": RANDOM_SEED,
        "nthread": -1,
        "verbosity": 0,
    }

    n_train = X_train.shape[0]
    oof_preds = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(X_test.shape[0], dtype=np.float32)
    fold_rmses = []

    dtest = xgb.DMatrix(X_test)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(np.empty(n_train)), 1):
        log.info(f"\n  ── Fold {fold_idx}/{n_folds} (train={len(tr_idx):,}, val={len(va_idx):,}) ──")
        t0 = time.perf_counter()

        dtrain = xgb.DMatrix(X_train[tr_idx], label=y_train[tr_idx])
        dval = xgb.DMatrix(X_train[va_idx], label=y_train[va_idx])

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=100,
        )

        best_iter = model.best_iteration
        va_preds = np.clip(model.predict(dval, iteration_range=(0, best_iter + 1)), 1.0, 5.0)
        oof_preds[va_idx] = va_preds
        fold_rmse = float(np.sqrt(np.mean((va_preds - y_train[va_idx]) ** 2)))
        fold_rmses.append(fold_rmse)

        test_preds += np.clip(model.predict(dtest, iteration_range=(0, best_iter + 1)), 1.0, 5.0) / n_folds

        elapsed = time.perf_counter() - t0
        log.info(f"  Fold {fold_idx} RMSE: {fold_rmse:.5f}  "
                 f"(best_iter={best_iter}, {elapsed:.1f}s)")

        del dtrain, dval, model
        gc.collect()

    del dtest
    gc.collect()

    return oof_preds, test_preds, fold_rmses


def train_catboost_oof(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    n_folds: int = N_FOLDS,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Train CatBoost with 5-fold OOF."""
    from catboost import CatBoostRegressor, Pool

    log.info("\n" + "=" * 60)
    log.info("Training CatBoost (ALL features, 41 dense cols)")
    log.info("=" * 60)

    params = {
        "iterations": 1000,
        "depth": 6,
        "learning_rate": 0.05,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "verbose": 100,
        "random_seed": RANDOM_SEED,
        "thread_count": -1,
        "early_stopping_rounds": 50,
    }

    n_train = len(y_train)
    oof_preds = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(X_test.shape[0], dtype=np.float32)
    fold_rmses = []

    test_pool = Pool(X_test)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(np.empty(n_train)), 1):
        log.info(f"\n  ── Fold {fold_idx}/{n_folds} (train={len(tr_idx):,}, val={len(va_idx):,}) ──")
        t0 = time.perf_counter()

        train_pool = Pool(X_train[tr_idx], y_train[tr_idx])
        val_pool = Pool(X_train[va_idx], y_train[va_idx])

        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        va_preds = np.clip(model.predict(val_pool), 1.0, 5.0)
        oof_preds[va_idx] = va_preds
        fold_rmse = float(np.sqrt(np.mean((va_preds - y_train[va_idx]) ** 2)))
        fold_rmses.append(fold_rmse)

        test_preds += np.clip(model.predict(test_pool), 1.0, 5.0) / n_folds

        elapsed = time.perf_counter() - t0
        log.info(f"  Fold {fold_idx} RMSE: {fold_rmse:.5f}  ({elapsed:.1f}s)")

        del train_pool, val_pool, model
        gc.collect()

    del test_pool
    gc.collect()

    return oof_preds, test_preds, fold_rmses


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 60)
    print("Train LGB / XGB / CatBoost with ALL features")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    # 1. Load features ────────────────────────────────────────────────
    X_train, X_test, y_train = load_dense_features()

    # ── 2. LightGBM ─────────────────────────────────────────────────
    t_lgb = time.perf_counter()
    lgb_oof, lgb_test, lgb_folds = train_lgb_oof(X_train, y_train, X_test)
    lgb_oof = np.clip(lgb_oof, 1.0, 5.0)
    lgb_oof_rmse = float(np.sqrt(np.mean((lgb_oof - y_train) ** 2)))
    lgb_time = time.perf_counter() - t_lgb
    log.info(f"\n  LightGBM OOF RMSE: {lgb_oof_rmse:.5f}  ({lgb_time:.1f}s)")

    np.save(str(MODEL_DIR / "lgb_allfeatures_oof.npy"), lgb_oof)
    np.save(str(MODEL_DIR / "lgb_allfeatures_test.npy"), lgb_test)
    log.info("  Saved: lgb_allfeatures_oof.npy, lgb_allfeatures_test.npy")
    del lgb_oof, lgb_test
    gc.collect()

    # ── 3. XGBoost ──────────────────────────────────────────────────
    t_xgb = time.perf_counter()
    xgb_oof, xgb_test, xgb_folds = train_xgb_oof(X_train, y_train, X_test)
    xgb_oof = np.clip(xgb_oof, 1.0, 5.0)
    xgb_oof_rmse = float(np.sqrt(np.mean((xgb_oof - y_train) ** 2)))
    xgb_time = time.perf_counter() - t_xgb
    log.info(f"\n  XGBoost OOF RMSE: {xgb_oof_rmse:.5f}  ({xgb_time:.1f}s)")

    np.save(str(MODEL_DIR / "xgboost_allfeatures_oof.npy"), xgb_oof)
    np.save(str(MODEL_DIR / "xgboost_allfeatures_test.npy"), xgb_test)
    log.info("  Saved: xgboost_allfeatures_oof.npy, xgboost_allfeatures_test.npy")
    del xgb_oof, xgb_test
    gc.collect()

    # ── 4. CatBoost ─────────────────────────────────────────────────
    t_cb = time.perf_counter()
    cb_oof, cb_test, cb_folds = train_catboost_oof(X_train, y_train, X_test)
    cb_oof = np.clip(cb_oof, 1.0, 5.0)
    cb_oof_rmse = float(np.sqrt(np.mean((cb_oof - y_train) ** 2)))
    cb_time = time.perf_counter() - t_cb
    log.info(f"\n  CatBoost OOF RMSE: {cb_oof_rmse:.5f}  ({cb_time:.1f}s)")

    np.save(str(MODEL_DIR / "catboost_allfeatures_oof.npy"), cb_oof)
    np.save(str(MODEL_DIR / "catboost_allfeatures_test.npy"), cb_test)
    log.info("  Saved: catboost_allfeatures_oof.npy, catboost_allfeatures_test.npy")
    del cb_oof, cb_test, X_train, X_test
    gc.collect()

    # ── 5. Summary ──────────────────────────────────────────────────
    total_time = time.perf_counter() - t_total

    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"  LightGBM  OOF RMSE: {lgb_oof_rmse:.5f}  "
             f"(folds: {[f'{r:.5f}' for r in lgb_folds]})  {lgb_time:.1f}s")
    log.info(f"  XGBoost   OOF RMSE: {xgb_oof_rmse:.5f}  "
             f"(folds: {[f'{r:.5f}' for r in xgb_folds]})  {xgb_time:.1f}s")
    log.info(f"  CatBoost  OOF RMSE: {cb_oof_rmse:.5f}  "
             f"(folds: {[f'{r:.5f}' for r in cb_folds]})  {cb_time:.1f}s")
    log.info(f"\n  Total time: {total_time:.1f}s")

    # Check target
    target = 1.10
    for name, rmse in [("LightGBM", lgb_oof_rmse), ("XGBoost", xgb_oof_rmse), ("CatBoost", cb_oof_rmse)]:
        status = "✅" if rmse < target else "⚠️"
        log.info(f"  {status} {name}: OOF RMSE {'<' if rmse < target else '>='} {target}")

    log.info("\n=== Done ===")


if __name__ == "__main__":
    main()

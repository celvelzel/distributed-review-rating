#!/usr/bin/env python
"""Train LightGBM, XGBoost, CatBoost with ONLY safe features (no leakage).

Memory-safe approach: train dense models and TF-IDF model separately, then
ensemble.  Uses word-level TF-IDF (much sparser than char-level) to stay
within the 32 GB cgroup limit.

SAFE features (no target leakage):
  - TF-IDF (word-level, 2000-dim, generated on the fly)
  - sentiment.parquet (17 cols: VADER + TextBlob + word counts)
  - product_metadata.parquet (8 cols: feature_count, store stats, etc.)

EXCLUDED (causes leakage):
  - user_stats_kfold.parquet   ❌ LEAKAGE
  - product_stats_kfold.parquet ❌ LEAKAGE
  - rating_deviation.parquet   ❌ LEAKAGE

Strategy:
  1. Dense models (LGB/XGB/CatBoost): sentiment + product_metadata (25 cols)
  2. TF-IDF model (LGB only): word TF-IDF 2000-dim
  3. Ensemble: average of all 4 OOF/test predictions

Expected OOF RMSE: ~1.05-1.10 (NOT 0.05 - that was leakage!)
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
import scipy.sparse as sp
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer

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
OUTPUT_DIR = ROOT / "output"

RANDOM_SEED = 42
N_FOLDS = 5
TFIDF_MAX_FEATURES = 2000


# ══════════════════════════════════════════════════════════════════════
# Feature Loading
# ══════════════════════════════════════════════════════════════════════

def _combine_text(df: pd.DataFrame) -> pd.Series:
    """Concatenate review title and comment."""
    return (df["title"].fillna("") + " " + df["comment"].fillna("")).str.strip()


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


def load_dense_features() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load dense safe features (sentiment + product_metadata)."""
    log.info("Loading dense safe features (sentiment + product_metadata)")

    log.info("  [1/3] Loading sentiment features …")
    sent = pd.read_parquet(FEAT_DIR / "sentiment.parquet")
    sent_train = sent.drop(columns=["id"]).values.astype(np.float32)
    del sent

    test_df = pd.read_parquet(ETL_DIR / "test.parquet", columns=["id", "title", "comment"])
    sent_test = _compute_test_sentiment(test_df).values.astype(np.float32)
    del test_df
    log.info(f"    train: {sent_train.shape}  test: {sent_test.shape}")

    log.info("  [2/3] Loading product metadata …")
    pm = pd.read_parquet(FEAT_DIR / "product_metadata.parquet")
    train_meta = pd.read_parquet(ETL_DIR / "train.parquet", columns=["parent_prod_id"])
    test_meta = pd.read_parquet(ETL_DIR / "test.parquet", columns=["parent_prod_id"])

    pm_train = train_meta.merge(pm, on="parent_prod_id", how="left").drop(columns=["parent_prod_id"]).fillna(0).values.astype(np.float32)
    pm_test = test_meta.merge(pm, on="parent_prod_id", how="left").drop(columns=["parent_prod_id"]).fillna(0).values.astype(np.float32)
    del pm, train_meta, test_meta
    log.info(f"    train: {pm_train.shape}  test: {pm_test.shape}")

    log.info("  [3/3] Combining dense features …")
    X_train = np.hstack([sent_train, pm_train])
    X_test = np.hstack([sent_test, pm_test])
    del sent_train, sent_test, pm_train, pm_test
    gc.collect()

    y_train = np.load(str(FEAT_DIR / "y_train.npy")).astype(np.float32)
    test_ids = pd.read_parquet(ETL_DIR / "test.parquet", columns=["id"])["id"].values

    log.info(f"  X_train: {X_train.shape}  ({X_train.nbytes / 1e6:.1f} MB)")
    log.info(f"  X_test:  {X_test.shape}  ({X_test.nbytes / 1e6:.1f} MB)")

    return X_train, X_test, y_train, test_ids


def load_tfidf_features() -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Generate word-level TF-IDF features (sparse, much sparser than char-level)."""
    log.info(f"Generating word-level TF-IDF (max_features={TFIDF_MAX_FEATURES})")

    train_df = pd.read_parquet(ETL_DIR / "train.parquet", columns=["title", "comment"])
    test_df = pd.read_parquet(ETL_DIR / "test.parquet", columns=["title", "comment"])

    train_texts = _combine_text(train_df)
    test_texts = _combine_text(test_df)
    del train_df, test_df
    gc.collect()

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
    )
    X_train = vectorizer.fit_transform(train_texts).astype(np.float32)
    X_test = vectorizer.transform(test_texts).astype(np.float32)
    del train_texts, test_texts
    gc.collect()

    log.info(f"  train: {X_train.shape}  ({X_train.nnz:,} nnz)")
    log.info(f"  test:  {X_test.shape}  ({X_test.nnz:,} nnz)")

    return X_train, X_test


# ══════════════════════════════════════════════════════════════════════
# Model Training (5-fold OOF)
# ══════════════════════════════════════════════════════════════════════

def train_lgb_oof(
    X_train, y_train, X_test,
    n_folds: int = N_FOLDS,
    tag: str = "",
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Train LightGBM with 5-fold OOF."""
    import lightgbm as lgb

    log.info(f"\n{'=' * 60}")
    log.info(f"Training LightGBM ({tag})")
    log.info(f"{'=' * 60}")

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
    tag: str = "",
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Train XGBoost with 5-fold OOF (dense features only)."""
    import xgboost as xgb

    log.info(f"\n{'=' * 60}")
    log.info(f"Training XGBoost ({tag})")
    log.info(f"{'=' * 60}")

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
    tag: str = "",
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Train CatBoost with 5-fold OOF (dense features only)."""
    from catboost import CatBoostRegressor, Pool

    log.info(f"\n{'=' * 60}")
    log.info(f"Training CatBoost ({tag})")
    log.info(f"{'=' * 60}")

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
    print("Train LGB / XGB / CatBoost with SAFE features (no leakage)")
    print("  Strategy: Dense models (25 cols) + TF-IDF model (2000 cols)")
    print("  Features: sentiment 17 + product_metadata 8 + word TF-IDF 2000")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    # ── PART 1: Dense models ─────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("PART 1: Dense features (sentiment + product_metadata)")
    log.info("=" * 60)

    X_train_dense, X_test_dense, y_train, test_ids = load_dense_features()

    # LightGBM (dense)
    t_lgb = time.perf_counter()
    lgb_dense_oof, lgb_dense_test, lgb_dense_folds = train_lgb_oof(
        X_train_dense, y_train, X_test_dense, tag="dense features (25 cols)"
    )
    lgb_dense_oof = np.clip(lgb_dense_oof, 1.0, 5.0)
    lgb_dense_rmse = float(np.sqrt(np.mean((lgb_dense_oof - y_train) ** 2)))
    lgb_time = time.perf_counter() - t_lgb
    log.info(f"\n  LightGBM (dense) OOF RMSE: {lgb_dense_rmse:.5f}  ({lgb_time:.1f}s)")

    np.save(str(MODEL_DIR / "lgb_safe_dense_oof.npy"), lgb_dense_oof)
    np.save(str(MODEL_DIR / "lgb_safe_dense_test.npy"), lgb_dense_test)

    # XGBoost (dense)
    t_xgb = time.perf_counter()
    xgb_oof, xgb_test, xgb_folds = train_xgb_oof(
        X_train_dense, y_train, X_test_dense, tag="dense features (25 cols)"
    )
    xgb_oof = np.clip(xgb_oof, 1.0, 5.0)
    xgb_rmse = float(np.sqrt(np.mean((xgb_oof - y_train) ** 2)))
    xgb_time = time.perf_counter() - t_xgb
    log.info(f"\n  XGBoost (dense) OOF RMSE: {xgb_rmse:.5f}  ({xgb_time:.1f}s)")

    np.save(str(MODEL_DIR / "xgboost_safe_oof.npy"), xgb_oof)
    np.save(str(MODEL_DIR / "xgboost_safe_test.npy"), xgb_test)

    # CatBoost (dense)
    t_cb = time.perf_counter()
    cb_oof, cb_test, cb_folds = train_catboost_oof(
        X_train_dense, y_train, X_test_dense, tag="dense features (25 cols)"
    )
    cb_oof = np.clip(cb_oof, 1.0, 5.0)
    cb_rmse = float(np.sqrt(np.mean((cb_oof - y_train) ** 2)))
    cb_time = time.perf_counter() - t_cb
    log.info(f"\n  CatBoost (dense) OOF RMSE: {cb_rmse:.5f}  ({cb_time:.1f}s)")

    np.save(str(MODEL_DIR / "catboost_safe_oof.npy"), cb_oof)
    np.save(str(MODEL_DIR / "catboost_safe_test.npy"), cb_test)

    del X_train_dense, X_test_dense
    gc.collect()

    # ── PART 2: TF-IDF model ────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("PART 2: TF-IDF features (word-level TF-IDF 2000-dim)")
    log.info("=" * 60)

    X_train_tfidf, X_test_tfidf = load_tfidf_features()

    t_lgb_tfidf = time.perf_counter()
    lgb_tfidf_oof, lgb_tfidf_test, lgb_tfidf_folds = train_lgb_oof(
        X_train_tfidf, y_train, X_test_tfidf, tag=f"word TF-IDF ({TFIDF_MAX_FEATURES} cols)"
    )
    lgb_tfidf_oof = np.clip(lgb_tfidf_oof, 1.0, 5.0)
    lgb_tfidf_rmse = float(np.sqrt(np.mean((lgb_tfidf_oof - y_train) ** 2)))
    lgb_tfidf_time = time.perf_counter() - t_lgb_tfidf
    log.info(f"\n  LightGBM (TF-IDF) OOF RMSE: {lgb_tfidf_rmse:.5f}  ({lgb_tfidf_time:.1f}s)")

    np.save(str(MODEL_DIR / "lgb_safe_tfidf_oof.npy"), lgb_tfidf_oof)
    np.save(str(MODEL_DIR / "lgb_safe_tfidf_test.npy"), lgb_tfidf_test)

    del X_train_tfidf, X_test_tfidf
    gc.collect()

    # ── PART 3: Ensemble ────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("PART 3: Ensemble (average of 4 models)")
    log.info("=" * 60)

    lgb_dense_oof = np.load(str(MODEL_DIR / "lgb_safe_dense_oof.npy"))
    lgb_dense_test = np.load(str(MODEL_DIR / "lgb_safe_dense_test.npy"))
    xgb_oof = np.load(str(MODEL_DIR / "xgboost_safe_oof.npy"))
    xgb_test = np.load(str(MODEL_DIR / "xgboost_safe_test.npy"))
    cb_oof = np.load(str(MODEL_DIR / "catboost_safe_oof.npy"))
    cb_test = np.load(str(MODEL_DIR / "catboost_safe_test.npy"))
    lgb_tfidf_oof = np.load(str(MODEL_DIR / "lgb_safe_tfidf_oof.npy"))
    lgb_tfidf_test = np.load(str(MODEL_DIR / "lgb_safe_tfidf_test.npy"))

    ensemble_oof = (lgb_dense_oof + xgb_oof + cb_oof + lgb_tfidf_oof) / 4.0
    ensemble_test = (lgb_dense_test + xgb_test + cb_test + lgb_tfidf_test) / 4.0
    ensemble_rmse = float(np.sqrt(np.mean((ensemble_oof - y_train) ** 2)))
    log.info(f"  Ensemble OOF RMSE: {ensemble_rmse:.5f}")

    np.save(str(MODEL_DIR / "ensemble_safe_oof.npy"), ensemble_oof)
    np.save(str(MODEL_DIR / "ensemble_safe_test.npy"), ensemble_test)

    # Save submission
    submission = pd.DataFrame({"id": test_ids, "rating": np.clip(ensemble_test, 1.0, 5.0)})
    submission_path = OUTPUT_DIR / "submission_safe_features.csv"
    submission.to_csv(str(submission_path), index=False)
    log.info(f"  Submission saved → {submission_path}  ({len(submission):,} rows)")

    # ── Summary ─────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_total

    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"  LightGBM  (dense)  OOF RMSE: {lgb_dense_rmse:.5f}  "
             f"(folds: {[f'{r:.5f}' for r in lgb_dense_folds]})  {lgb_time:.1f}s")
    log.info(f"  XGBoost   (dense)  OOF RMSE: {xgb_rmse:.5f}  "
             f"(folds: {[f'{r:.5f}' for r in xgb_folds]})  {xgb_time:.1f}s")
    log.info(f"  CatBoost  (dense)  OOF RMSE: {cb_rmse:.5f}  "
             f"(folds: {[f'{r:.5f}' for r in cb_folds]})  {cb_time:.1f}s")
    log.info(f"  LightGBM  (tfidf)  OOF RMSE: {lgb_tfidf_rmse:.5f}  "
             f"(folds: {[f'{r:.5f}' for r in lgb_tfidf_folds]})  {lgb_tfidf_time:.1f}s")
    log.info(f"  Ensemble  (4 avg)  OOF RMSE: {ensemble_rmse:.5f}")
    log.info(f"\n  Total time: {total_time:.1f}s")

    # Sanity check
    log.info("\n" + "=" * 60)
    log.info("SANITY CHECK (no leakage)")
    log.info("=" * 60)
    all_rmses = [
        ("LightGBM (dense)", lgb_dense_rmse),
        ("XGBoost (dense)", xgb_rmse),
        ("CatBoost (dense)", cb_rmse),
        ("LightGBM (tfidf)", lgb_tfidf_rmse),
        ("Ensemble", ensemble_rmse),
    ]
    for name, rmse in all_rmses:
        if rmse < 0.90:
            log.info(f"  ⚠️  {name}: OOF RMSE = {rmse:.5f} (< 0.90, suspicious)")
        elif rmse < 1.10:
            log.info(f"  ✅ {name}: OOF RMSE = {rmse:.5f} (good, no leakage)")
        else:
            log.info(f"  ⚠️  {name}: OOF RMSE = {rmse:.5f} (>= 1.10, room for improvement)")

    log.info("\n=== Done ===")


if __name__ == "__main__":
    main()

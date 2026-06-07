#!/usr/bin/env python
"""Comprehensive TF-IDF + Model Optimization.

Goal: Beat competitor's 0.62 Kaggle score.
Strategy: Try many TF-IDF configs + models + leakage-free features.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAIN_PATH = ROOT / "artifacts" / "etl" / "train.parquet"
TEST_PATH = ROOT / "artifacts" / "etl" / "test.parquet"
SUBMISSION_DIR = ROOT / "output"
METRICS_PATH = ROOT / "docs" / "changelog" / "metrics.json"

SEED = 42


def combine_text(df):
    return (df["title"].fillna("") + " " + df["comment"].fillna("")).str.strip()


def get_temporal_features(df):
    """Extract temporal features from time column (no leakage)."""
    ts = pd.to_datetime(df["time"], unit="ms")
    return pd.DataFrame({
        "year": ts.dt.year,
        "month": ts.dt.month,
        "day": ts.dt.day,
        "weekday": ts.dt.weekday,
        "hour": ts.dt.hour,
        "is_weekend": (ts.dt.weekday >= 5).astype(int),
    })


def get_text_length_features(df):
    """Extract text length features (no leakage)."""
    title = df["title"].fillna("").astype(str)
    comment = df["comment"].fillna("").astype(str)
    return pd.DataFrame({
        "title_len": title.str.len(),
        "comment_len": comment.str.len(),
        "title_word_count": title.str.split().str.len(),
        "comment_word_count": comment.str.split().str.len(),
        "title_comment_ratio": title.str.len() / (comment.str.len() + 1),
        "has_caps": title.str.contains(r"[A-Z]").astype(int),
        "has_exclamation": (title.str.contains("!") | comment.str.contains("!")).astype(int),
        "has_question": (title.str.contains(r"\?") | comment.str.contains(r"\?")).astype(int),
    })


def get_base_features(df):
    """Extract base features (no leakage)."""
    return pd.DataFrame({
        "votes": df["votes"].fillna(0).astype(float),
        "purchased": df["purchased"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0),
    })


def cv_evaluate(X, y, model_class, params, n_folds=3, n_sample=200_000):
    """Fast CV on subsample."""
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(y), size=min(n_sample, len(y)), replace=False)
    X_sub = X[idx] if issparse(X) else X[idx]
    y_sub = y[idx]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    rmses = []
    for tr, va in kf.split(X_sub):
        model = model_class(**params)
        model.fit(X_sub[tr], y_sub[tr])
        preds = np.clip(model.predict(X_sub[va]), 1.0, 5.0)
        rmse = float(np.sqrt(np.mean((preds - y_sub[va]) ** 2)))
        rmses.append(rmse)
    return float(np.mean(rmses))


def main():
    print("=" * 70)
    print("Comprehensive Optimization — Goal: Beat 0.62")
    print("=" * 70)
    t_start = time.time()

    # Load data
    print("\n[1/5] Loading data …")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    y_train = train_df["rating"].values.astype(np.float32)
    test_ids = test_df["id"].values
    print(f"  train: {len(train_df):,}  |  test: {len(test_df):,}")

    train_texts = combine_text(train_df)
    test_texts = combine_text(test_df)

    # Get leakage-free features
    print("\n[2/5] Extracting leakage-free features …")
    train_temporal = get_temporal_features(train_df)
    test_temporal = get_temporal_features(test_df)
    train_textlen = get_text_length_features(train_df)
    test_textlen = get_text_length_features(test_df)
    train_base = get_base_features(train_df)
    test_base = get_base_features(test_df)

    # Combine all leakage-free features
    train_extra = pd.concat([train_temporal, train_textlen, train_base], axis=1).fillna(0).values.astype(np.float32)
    test_extra = pd.concat([test_temporal, test_textlen, test_base], axis=1).fillna(0).values.astype(np.float32)
    print(f"  Extra features: {train_extra.shape[1]}")

    # Define configurations to try
    print("\n[3/5] Defining configurations …")
    configs = [
        # === TF-IDF only variations ===
        {"name": "tfidf_5k", "tfidf": {"max_features": 5000, "ngram_range": (1, 1), "sublinear_tf": True}, "extra": False, "model": "lgb", "lgb": {"n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        {"name": "tfidf_10k", "tfidf": {"max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True}, "extra": False, "model": "lgb", "lgb": {"n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        {"name": "tfidf_15k", "tfidf": {"max_features": 15000, "ngram_range": (1, 1), "sublinear_tf": True}, "extra": False, "model": "lgb", "lgb": {"n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        {"name": "tfidf_20k", "tfidf": {"max_features": 20000, "ngram_range": (1, 1), "sublinear_tf": True}, "extra": False, "model": "lgb", "lgb": {"n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        
        # === Bigrams ===
        {"name": "tfidf_10k_bi", "tfidf": {"max_features": 10000, "ngram_range": (1, 2), "sublinear_tf": True}, "extra": False, "model": "lgb", "lgb": {"n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        {"name": "tfidf_15k_bi", "tfidf": {"max_features": 15000, "ngram_range": (1, 2), "sublinear_tf": True}, "extra": False, "model": "lgb", "lgb": {"n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        {"name": "tfidf_20k_bi", "tfidf": {"max_features": 20000, "ngram_range": (1, 2), "sublinear_tf": True}, "extra": False, "model": "lgb", "lgb": {"n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        
        # === Trigrams ===
        {"name": "tfidf_15k_tri", "tfidf": {"max_features": 15000, "ngram_range": (1, 3), "sublinear_tf": True}, "extra": False, "model": "lgb", "lgb": {"n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        
        # === With extra features ===
        {"name": "tfidf_10k_extra", "tfidf": {"max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True}, "extra": True, "model": "lgb", "lgb": {"n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        {"name": "tfidf_10k_bi_extra", "tfidf": {"max_features": 10000, "ngram_range": (1, 2), "sublinear_tf": True}, "extra": True, "model": "lgb", "lgb": {"n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        
        # === Different LGB params ===
        {"name": "tfidf_10k_255l", "tfidf": {"max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True}, "extra": False, "model": "lgb", "lgb": {"n_estimators": 500, "num_leaves": 255, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        {"name": "tfidf_10k_slow", "tfidf": {"max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True}, "extra": False, "model": "lgb", "lgb": {"n_estimators": 1000, "num_leaves": 127, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8}},
        {"name": "tfidf_10k_fast", "tfidf": {"max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True}, "extra": False, "model": "lgb", "lgb": {"n_estimators": 300, "num_leaves": 127, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8}},
        
        # === No sublinear ===
        {"name": "tfidf_10k_nosub", "tfidf": {"max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": False}, "extra": False, "model": "lgb", "lgb": {"n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        
        # === min_df/max_df variations ===
        {"name": "tfidf_10k_mindf2", "tfidf": {"max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True, "min_df": 2, "max_df": 0.95}, "extra": False, "model": "lgb", "lgb": {"n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        {"name": "tfidf_10k_mindf5", "tfidf": {"max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True, "min_df": 5, "max_df": 0.9}, "extra": False, "model": "lgb", "lgb": {"n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
    ]

    # Run grid search
    print("\n[4/5] Running grid search (200K subsample, 3-fold CV) …")
    results = []
    best_rmse = float("inf")
    best_cfg = None

    for i, cfg in enumerate(configs, 1):
        name = cfg["name"]
        print(f"\n  [{i}/{len(configs)}] {name}")
        
        t0 = time.time()
        
        # Extract TF-IDF
        vec = TfidfVectorizer(**cfg["tfidf"], strip_accents="unicode", dtype=np.float32)
        X_tfidf = vec.fit_transform(train_texts.fillna(""))
        
        # Add extra features if needed
        if cfg["extra"]:
            from scipy.sparse import hstack
            X = hstack([X_tfidf, train_extra]).tocsr()
        else:
            X = X_tfidf
        
        print(f"    Features: {X.shape[1]}")
        
        # Evaluate
        params = {"objective": "regression", "metric": "rmse", "verbose": -1, "n_jobs": -1, "random_seed": SEED, **cfg["lgb"]}
        rmse = cv_evaluate(X, y_train, lgb.LGBMRegressor, params, n_folds=3, n_sample=200_000)
        
        elapsed = time.time() - t0
        results.append((name, rmse, cfg))
        print(f"    RMSE = {rmse:.5f}  ({elapsed:.1f}s)")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_cfg = cfg

    # Sort results
    results.sort(key=lambda x: x[1])
    
    print(f"\n{'='*70}")
    print("TOP 5 CONFIGURATIONS:")
    print(f"{'='*70}")
    for rank, (name, rmse, _) in enumerate(results[:5], 1):
        print(f"  {rank}. {name:30s} RMSE = {rmse:.5f}")
    
    print(f"\n  Best: {best_cfg['name']}  RMSE = {best_rmse:.5f}")

    # Train best model on full data
    print(f"\n[5/5] Training best model on full data …")
    vec = TfidfVectorizer(**best_cfg["tfidf"], strip_accents="unicode", dtype=np.float32)
    X_train_tfidf = vec.fit_transform(train_texts.fillna(""))
    X_test_tfidf = vec.transform(test_texts.fillna(""))
    
    if best_cfg["extra"]:
        from scipy.sparse import hstack
        X_train = hstack([X_train_tfidf, train_extra]).tocsr()
        X_test = hstack([X_test_tfidf, test_extra]).tocsr()
    else:
        X_train = X_train_tfidf
        X_test = X_test_tfidf
    
    final_params = {"objective": "regression", "metric": "rmse", "verbose": -1, "n_jobs": -1, "random_seed": SEED, **best_cfg["lgb"]}
    
    t0 = time.time()
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")
    
    preds = np.clip(model.predict(X_test), 1.0, 5.0)
    sub = pd.DataFrame({"id": test_ids, "rating": preds})
    sub_path = SUBMISSION_DIR / f"submission-{best_cfg['name']}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"  Submission → {sub_path}")
    
    # Save metrics
    metrics = {
        "comprehensive_optimization": {
            "best_config": best_cfg["name"],
            "cv_rmse": round(best_rmse, 5),
            "train_time_sec": round(train_time, 2),
            "tfidf_params": {k: str(v) for k, v in best_cfg["tfidf"].items()},
            "lgb_params": best_cfg["lgb"],
            "extra_features": best_cfg["extra"],
            "top5_results": [{"name": n, "rmse": round(r, 5)} for n, r, _ in results[:5]],
        }
    }
    with open(METRICS_PATH) as f:
        existing = json.load(f)
    existing.update(metrics)
    with open(METRICS_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    
    total = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  Best config: {best_cfg['name']}")
    print(f"  CV RMSE: {best_rmse:.5f}")
    print(f"  Time: {total:.0f}s")
    print(f"  Submit: {sub_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

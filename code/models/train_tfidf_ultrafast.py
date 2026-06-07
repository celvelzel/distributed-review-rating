#!/usr/bin/env python
"""Ultra-fast TF-IDF + LightGBM optimization.

Uses 200K subsample for TF-IDF + training. Fast iteration.
Goal: Beat Kaggle 0.80107.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAIN_PATH = ROOT / "artifacts" / "etl" / "train.parquet"
TEST_PATH = ROOT / "artifacts" / "etl" / "test.parquet"
SUBMISSION_PATH = ROOT / "output" / "submission-tfidf-v2.csv"

SEED = 42
N_SAMPLE = 200_000


def main():
    print("=" * 60)
    print("Ultra-fast TF-IDF + LightGBM Optimization")
    print(f"Using {N_SAMPLE:,} subsample for speed")
    print("=" * 60)
    t0 = time.time()

    # Load and subsample
    print("\n[1/3] Loading data …")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(train_df), size=N_SAMPLE, replace=False)
    sub_df = train_df.iloc[idx]
    
    y_sub = sub_df["rating"].values.astype(np.float32)
    test_ids = test_df["id"].values
    
    texts_sub = (sub_df["title"].fillna("") + " " + sub_df["comment"].fillna("")).str.strip()
    texts_test = (test_df["title"].fillna("") + " " + test_df["comment"].fillna("")).str.strip()
    
    print(f"  Subsample: {len(sub_df):,}  |  Test: {len(test_df):,}")

    # Configs to try
    configs = [
        # Baseline
        {"name": "baseline", "tfidf": {"max_features": 5000, "ngram_range": (1, 1), "sublinear_tf": True},
         "lgb": {"learning_rate": 0.05, "num_leaves": 63, "n_estimators": 500}},
        # More features
        {"name": "10k", "tfidf": {"max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True},
         "lgb": {"learning_rate": 0.05, "num_leaves": 63, "n_estimators": 500}},
        # Bigrams
        {"name": "10k_bi", "tfidf": {"max_features": 10000, "ngram_range": (1, 2), "sublinear_tf": True},
         "lgb": {"learning_rate": 0.05, "num_leaves": 63, "n_estimators": 500}},
        # More leaves
        {"name": "10k_bi_127l", "tfidf": {"max_features": 10000, "ngram_range": (1, 2), "sublinear_tf": True},
         "lgb": {"learning_rate": 0.05, "num_leaves": 127, "n_estimators": 500}},
        # Slower learning
        {"name": "10k_bi_slow", "tfidf": {"max_features": 10000, "ngram_range": (1, 2), "sublinear_tf": True},
         "lgb": {"learning_rate": 0.01, "num_leaves": 127, "n_estimators": 1000}},
        # Higher lr
        {"name": "10k_bi_fast", "tfidf": {"max_features": 10000, "ngram_range": (1, 2), "sublinear_tf": True},
         "lgb": {"learning_rate": 0.1, "num_leaves": 63, "n_estimators": 300}},
    ]

    print("\n[2/3] Testing configurations …")
    best_rmse = float("inf")
    best_cfg = None
    
    for i, cfg in enumerate(configs, 1):
        print(f"\n  [{i}/{len(configs)}] {cfg['name']}")
        
        vec = TfidfVectorizer(**cfg["tfidf"], strip_accents="unicode")
        X = vec.fit_transform(texts_sub.fillna(""))
        
        params = {"objective": "regression", "metric": "rmse", "verbose": -1, "n_jobs": -1, "random_seed": SEED, **cfg["lgb"]}
        
        kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
        rmses = []
        for tr, va in kf.split(X):
            model = lgb.LGBMRegressor(**params)
            model.fit(X[tr], y_sub[tr])
            preds = np.clip(model.predict(X[va]), 1.0, 5.0)
            rmse = float(np.sqrt(np.mean((preds - y_sub[va]) ** 2)))
            rmses.append(rmse)
        
        mean_rmse = float(np.mean(rmses))
        print(f"    RMSE = {mean_rmse:.5f}")
        
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_cfg = cfg

    print(f"\n  Best: {best_cfg['name']}  RMSE = {best_rmse:.5f}")

    # Train final on full data
    print("\n[3/3] Training final model on full data …")
    train_texts = (train_df["title"].fillna("") + " " + train_df["comment"].fillna("")).str.strip()
    y_train = train_df["rating"].values.astype(np.float32)
    
    vec = TfidfVectorizer(**best_cfg["tfidf"], strip_accents="unicode")
    X_train = vec.fit_transform(train_texts.fillna(""))
    X_test = vec.transform(texts_test.fillna(""))
    
    final_params = {"objective": "regression", "metric": "rmse", "verbose": -1, "n_jobs": -1, "random_seed": SEED, **best_cfg["lgb"]}
    
    t1 = time.time()
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X_train, y_train)
    print(f"  Training time: {time.time()-t1:.1f}s")
    
    preds = np.clip(model.predict(X_test), 1.0, 5.0)
    sub = pd.DataFrame({"id": test_ids, "rating": preds})
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"  Submission → {SUBMISSION_PATH}")
    
    total = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Best: {best_cfg['name']}")
    print(f"  Subsample RMSE: {best_rmse:.5f}")
    print(f"  Time: {total:.0f}s")
    print(f"  Submit: {SUBMISSION_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

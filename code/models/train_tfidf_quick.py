#!/usr/bin/env python
"""Quick TF-IDF + LightGBM optimization.

Uses 50K subsample for fast iteration. Tries 4 configs.
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


def main():
    print("=" * 60)
    print("Quick TF-IDF + LightGBM Optimization")
    print("=" * 60)
    t_start = time.time()

    # Load data
    print("\n[1/3] Loading data …")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    # Subsample for fast iteration
    sub_df = train_df.sample(50000, random_state=SEED)
    y_sub = sub_df["rating"].values.astype(np.float32)
    texts_sub = (sub_df["title"].fillna("") + " " + sub_df["comment"].fillna("")).str.strip()
    
    print(f"  Subsample: {len(sub_df):,}  |  Test: {len(test_df):,}")

    # Configs to try
    configs = [
        {"name": "baseline", "tfidf": {"max_features": 5000, "ngram_range": (1, 1), "sublinear_tf": True},
         "lgb": {"learning_rate": 0.05, "num_leaves": 63, "n_estimators": 200}},
        {"name": "more_trees", "tfidf": {"max_features": 5000, "ngram_range": (1, 1), "sublinear_tf": True},
         "lgb": {"learning_rate": 0.01, "num_leaves": 63, "n_estimators": 500}},
        {"name": "bigrams", "tfidf": {"max_features": 5000, "ngram_range": (1, 2), "sublinear_tf": True},
         "lgb": {"learning_rate": 0.05, "num_leaves": 63, "n_estimators": 200}},
        {"name": "more_leaves", "tfidf": {"max_features": 5000, "ngram_range": (1, 1), "sublinear_tf": True},
         "lgb": {"learning_rate": 0.05, "num_leaves": 127, "n_estimators": 200}},
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
    test_texts = (test_df["title"].fillna("") + " " + test_df["comment"].fillna("")).str.strip()
    y_train = train_df["rating"].values.astype(np.float32)
    test_ids = test_df["id"].values
    
    vec = TfidfVectorizer(**best_cfg["tfidf"], strip_accents="unicode")
    X_train = vec.fit_transform(train_texts.fillna(""))
    X_test = vec.transform(test_texts.fillna(""))
    
    # Use more trees for final model
    final_lgb = best_cfg["lgb"].copy()
    final_lgb["n_estimators"] = max(final_lgb["n_estimators"], 500)
    
    final_params = {"objective": "regression", "metric": "rmse", "verbose": -1, "n_jobs": -1, "random_seed": SEED, **final_lgb}
    
    t0 = time.time()
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X_train, y_train)
    print(f"  Training time: {time.time()-t0:.1f}s")
    
    preds = np.clip(model.predict(X_test), 1.0, 5.0)
    sub = pd.DataFrame({"id": test_ids, "rating": preds})
    sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"  Submission → {SUBMISSION_PATH}")
    
    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Best config: {best_cfg['name']}")
    print(f"  Subsample RMSE: {best_rmse:.5f}")
    print(f"  Stage 0 baseline: 1.17626 (Kaggle 0.80107)")
    print(f"  Time: {total:.0f}s")
    print(f"  Submit to Kaggle: {SUBMISSION_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

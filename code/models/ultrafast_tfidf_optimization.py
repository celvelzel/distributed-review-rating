#!/usr/bin/env python
"""Ultra-fast TF-IDF + LightGBM Optimization.

Goal: Beat 0.79012 (current best).
Strategy: Train on subsample for speed, then retrain on full data.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAIN_PATH = ROOT / "artifacts" / "etl" / "train.parquet"
TEST_PATH = ROOT / "artifacts" / "etl" / "test.parquet"
SUBMISSION_DIR = ROOT / "output"

SEED = 42


def combine_text(df):
    """Combine title and comment into single text."""
    return (df["title"].fillna("") + " " + df["comment"].fillna("")).str.strip()


def main():
    print("=" * 60)
    print("Ultra-fast TF-IDF + LightGBM Optimization")
    print("Goal: Beat 0.79012 (current best)")
    print("=" * 60)
    t_start = time.time()

    # Load data
    print("\n[1/3] Loading data …")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    
    # Use subsample for fast iteration
    SUBSAMPLE = 200000
    train_sub = train_df.sample(SUBSAMPLE, random_state=SEED)
    y_train = train_sub["rating"].values.astype(np.float32)
    test_ids = test_df["id"].values
    train_texts = combine_text(train_sub)
    test_texts = combine_text(test_df)
    full_train_texts = combine_text(train_df)
    y_full = train_df["rating"].values.astype(np.float32)
    print(f"  Subsample: {len(train_sub):,}  |  Full train: {len(train_df):,}  |  Test: {len(test_df):,}")

    # Key configurations to try
    print("\n[2/3] Testing configurations …")
    configs = [
        # Current best baseline
        {"name": "baseline_5k", "tfidf": {"max_features": 5000, "ngram_range": (1, 1), "sublinear_tf": True},
         "lgb": {"n_estimators": 300, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        
        # More features (10K)
        {"name": "tfidf_10k", "tfidf": {"max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True},
         "lgb": {"n_estimators": 300, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        
        # Bigrams (10K)
        {"name": "tfidf_10k_bi", "tfidf": {"max_features": 10000, "ngram_range": (1, 2), "sublinear_tf": True},
         "lgb": {"n_estimators": 300, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        
        # More leaves (255)
        {"name": "tfidf_10k_255l", "tfidf": {"max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True},
         "lgb": {"n_estimators": 300, "num_leaves": 255, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        
        # Stronger regularization
        {"name": "tfidf_10k_reg", "tfidf": {"max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True},
         "lgb": {"n_estimators": 300, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 0.5}},
        
        # Lower learning rate, more trees
        {"name": "tfidf_10k_slow", "tfidf": {"max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True},
         "lgb": {"n_estimators": 500, "num_leaves": 127, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8}},
    ]

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
        
        print(f"    Features: {X_tfidf.shape[1]}")
        
        # 3-fold CV
        kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
        rmses = []
        for tr, va in kf.split(X_tfidf):
            model = lgb.LGBMRegressor(objective="regression", metric="rmse", verbose=-1, 
                                       n_jobs=-1, random_seed=SEED, **cfg["lgb"])
            model.fit(X_tfidf[tr], y_train[tr])
            preds = np.clip(model.predict(X_tfidf[va]), 1.0, 5.0)
            rmse = float(np.sqrt(np.mean((preds - y_train[va]) ** 2)))
            rmses.append(rmse)
        
        rmse = float(np.mean(rmses))
        elapsed = time.time() - t0
        results.append((name, rmse, cfg))
        print(f"    RMSE = {rmse:.5f}  ({elapsed:.1f}s)")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_cfg = cfg

    # Sort results
    results.sort(key=lambda x: x[1])
    
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    for rank, (name, rmse, _) in enumerate(results, 1):
        print(f"  {rank}. {name:30s} RMSE = {rmse:.5f}")
    
    print(f"\n  Best: {best_cfg['name']}  RMSE = {best_rmse:.5f}")

    # Train best model on FULL data
    print(f"\n[3/3] Training best model on FULL data …")
    
    vec = TfidfVectorizer(**best_cfg["tfidf"], strip_accents="unicode", dtype=np.float32)
    X_train = vec.fit_transform(full_train_texts.fillna(""))
    X_test = vec.transform(test_texts.fillna(""))
    
    # Use more trees for final model
    final_lgb = best_cfg["lgb"].copy()
    final_lgb["n_estimators"] = max(final_lgb["n_estimators"], 500)
    
    final_params = {"objective": "regression", "metric": "rmse", "verbose": -1, "n_jobs": -1, "random_seed": SEED, **final_lgb}
    
    t0 = time.time()
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X_train, y_full)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")
    
    preds = np.clip(model.predict(X_test), 1.0, 5.0)
    
    # Save submission
    sub = pd.DataFrame({"id": test_ids, "rating": preds})
    sub_path = SUBMISSION_DIR / f"submission-{best_cfg['name']}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"  Submission → {sub_path}")
    
    # Summary
    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Best config: {best_cfg['name']}")
    print(f"  Subsample CV RMSE: {best_rmse:.5f}")
    print(f"  Current best Kaggle: 0.79012")
    print(f"  Time: {total:.0f}s")
    print(f"  Submit: {sub_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

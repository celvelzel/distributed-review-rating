#!/usr/bin/env python
"""Fast TF-IDF + LightGBM Optimization.

Goal: Beat 0.79012 (current best).
Strategy: Try key TF-IDF configs with optimized LightGBM.
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
METRICS_PATH = ROOT / "docs" / "changelog" / "metrics.json"

SEED = 42


def combine_text(df):
    """Combine title and comment into single text."""
    return (df["title"].fillna("") + " " + df["comment"].fillna("")).str.strip()


def cv_evaluate(X, y, model_class, params, n_folds=3, n_sample=100_000):
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
    print("=" * 60)
    print("Fast TF-IDF + LightGBM Optimization")
    print("Goal: Beat 0.79012 (current best)")
    print("=" * 60)
    t_start = time.time()

    # Load data
    print("\n[1/4] Loading data …")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    y_train = train_df["rating"].values.astype(np.float32)
    test_ids = test_df["id"].values
    train_texts = combine_text(train_df)
    test_texts = combine_text(test_df)
    print(f"  train: {len(train_df):,}  |  test: {len(test_df):,}")

    # Key configurations to try (based on history)
    print("\n[2/4] Testing configurations …")
    configs = [
        # Current best baseline (127 leaves, subsample=0.8, colsample=0.8)
        {"name": "baseline_5k", "tfidf": {"max_features": 5000, "ngram_range": (1, 1), "sublinear_tf": True},
         "lgb": {"n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        
        # More features (10K)
        {"name": "tfidf_10k", "tfidf": {"max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True},
         "lgb": {"n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        
        # Bigrams (10K)
        {"name": "tfidf_10k_bi", "tfidf": {"max_features": 10000, "ngram_range": (1, 2), "sublinear_tf": True},
         "lgb": {"n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        
        # More trees, lower learning rate
        {"name": "tfidf_10k_slow", "tfidf": {"max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True},
         "lgb": {"n_estimators": 1000, "num_leaves": 127, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8}},
        
        # More leaves (255)
        {"name": "tfidf_10k_255l", "tfidf": {"max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True},
         "lgb": {"n_estimators": 500, "num_leaves": 255, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        
        # Stronger regularization
        {"name": "tfidf_10k_reg", "tfidf": {"max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True},
         "lgb": {"n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 0.5}},
        
        # min_df filtering
        {"name": "tfidf_10k_mindf2", "tfidf": {"max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True, "min_df": 2, "max_df": 0.95},
         "lgb": {"n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
        
        # 15K features with bigrams
        {"name": "tfidf_15k_bi", "tfidf": {"max_features": 15000, "ngram_range": (1, 2), "sublinear_tf": True},
         "lgb": {"n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}},
    ]

    results = []
    best_rmse = float("inf")
    best_cfg = None
    best_vec = None

    for i, cfg in enumerate(configs, 1):
        name = cfg["name"]
        print(f"\n  [{i}/{len(configs)}] {name}")
        
        t0 = time.time()
        
        # Extract TF-IDF
        vec = TfidfVectorizer(**cfg["tfidf"], strip_accents="unicode", dtype=np.float32)
        X_tfidf = vec.fit_transform(train_texts.fillna(""))
        
        print(f"    Features: {X_tfidf.shape[1]}")
        
        # Evaluate
        params = {"objective": "regression", "metric": "rmse", "verbose": -1, "n_jobs": -1, "random_seed": SEED, **cfg["lgb"]}
        rmse = cv_evaluate(X_tfidf, y_train, lgb.LGBMRegressor, params, n_folds=3, n_sample=100_000)
        
        elapsed = time.time() - t0
        results.append((name, rmse, cfg))
        print(f"    RMSE = {rmse:.5f}  ({elapsed:.1f}s)")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_cfg = cfg
            best_vec = vec

    # Sort results
    results.sort(key=lambda x: x[1])
    
    print(f"\n{'='*60}")
    print("TOP 5 CONFIGURATIONS:")
    print(f"{'='*60}")
    for rank, (name, rmse, _) in enumerate(results[:5], 1):
        print(f"  {rank}. {name:30s} RMSE = {rmse:.5f}")
    
    print(f"\n  Best: {best_cfg['name']}  RMSE = {best_rmse:.5f}")

    # Train best model on full data
    print(f"\n[3/4] Training best model on full data …")
    
    # Re-fit on full training data
    vec = TfidfVectorizer(**best_cfg["tfidf"], strip_accents="unicode", dtype=np.float32)
    X_train = vec.fit_transform(train_texts.fillna(""))
    X_test = vec.transform(test_texts.fillna(""))
    
    final_params = {"objective": "regression", "metric": "rmse", "verbose": -1, "n_jobs": -1, "random_seed": SEED, **best_cfg["lgb"]}
    
    t0 = time.time()
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")
    
    preds = np.clip(model.predict(X_test), 1.0, 5.0)
    
    # Save submission
    print(f"\n[4/4] Saving submission …")
    sub = pd.DataFrame({"id": test_ids, "rating": preds})
    sub_path = SUBMISSION_DIR / f"submission-{best_cfg['name']}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"  Submission → {sub_path}")
    
    # Summary
    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Best config: {best_cfg['name']}")
    print(f"  CV RMSE: {best_rmse:.5f}")
    print(f"  Current best Kaggle: 0.79012")
    print(f"  Time: {total:.0f}s")
    print(f"  Submit: {sub_path}")
    print(f"{'='*60}")
    
    # Save metrics
    metrics = {
        "fast_optimization": {
            "best_config": best_cfg["name"],
            "cv_rmse": round(best_rmse, 5),
            "train_time_sec": round(train_time, 2),
            "tfidf_params": {k: str(v) for k, v in best_cfg["tfidf"].items()},
            "lgb_params": best_cfg["lgb"],
            "top5_results": [{"name": n, "rmse": round(r, 5)} for n, r, _ in results[:5]],
        }
    }
    try:
        with open(METRICS_PATH) as f:
            existing = json.load(f)
        existing.update(metrics)
        with open(METRICS_PATH, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        print(f"  Warning: Could not update metrics: {e}")


if __name__ == "__main__":
    main()

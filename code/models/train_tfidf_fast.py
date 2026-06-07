#!/usr/bin/env python
"""Fast TF-IDF + LightGBM optimization - focused search.

Goal: Beat Kaggle 0.80107 using ONLY TF-IDF features.
Strategy: Try 5 key configurations on 500K subsample, pick best, train full.
"""

from __future__ import annotations

import json
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

from code.utils.timer import write_metrics

TRAIN_PATH = ROOT / "artifacts" / "etl" / "train.parquet"
TEST_PATH = ROOT / "artifacts" / "etl" / "test.parquet"
SUBMISSION_PATH = ROOT / "output" / "submission-tfidf-optimized.csv"
METRICS_PATH = ROOT / "docs" / "changelog" / "metrics.json"

RANDOM_SEED = 42


def combine_text(df):
    return (df["title"].fillna("") + " " + df["comment"].fillna("")).str.strip()


def cv_rmse(X, y, params, n_folds=3, n_sample=500_000):
    """Quick CV on subsample."""
    rng = np.random.RandomState(RANDOM_SEED)
    idx = rng.choice(len(y), size=min(n_sample, len(y)), replace=False)
    X_sub = X[idx]
    y_sub = y[idx]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    rmses = []
    for tr, va in kf.split(X_sub):
        model = lgb.LGBMRegressor(**params)
        model.fit(X_sub[tr], y_sub[tr])
        preds = np.clip(model.predict(X_sub[va]), 1.0, 5.0)
        rmse = float(np.sqrt(np.mean((preds - y_sub[va]) ** 2)))
        rmses.append(rmse)
    return float(np.mean(rmses))


def main():
    print("=" * 60)
    print("Fast TF-IDF + LightGBM Optimization")
    print("Goal: Beat Kaggle 0.80107")
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

    # Configs to try (5 focused configs)
    configs = [
        {
            "name": "baseline_5k",
            "tfidf": {"max_features": 5000, "ngram_range": (1, 1), "sublinear_tf": True},
            "lgb": {"learning_rate": 0.05, "num_leaves": 63, "n_estimators": 500, "n_jobs": -1},
        },
        {
            "name": "tfidf_10k",
            "tfidf": {"max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True},
            "lgb": {"learning_rate": 0.05, "num_leaves": 63, "n_estimators": 500, "n_jobs": -1},
        },
        {
            "name": "tfidf_15k_bi",
            "tfidf": {"max_features": 15000, "ngram_range": (1, 2), "sublinear_tf": True},
            "lgb": {"learning_rate": 0.05, "num_leaves": 63, "n_estimators": 500, "n_jobs": -1},
        },
        {
            "name": "tfidf_15k_bi_127l",
            "tfidf": {"max_features": 15000, "ngram_range": (1, 2), "sublinear_tf": True},
            "lgb": {"learning_rate": 0.05, "num_leaves": 127, "n_estimators": 500, "n_jobs": -1},
        },
        {
            "name": "tfidf_15k_bi_slow",
            "tfidf": {"max_features": 15000, "ngram_range": (1, 2), "sublinear_tf": True},
            "lgb": {"learning_rate": 0.01, "num_leaves": 127, "n_estimators": 1000, "n_jobs": -1},
        },
    ]

    # Search
    print("\n[2/4] Grid search (500K subsample, 3-fold CV) …")
    results = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n  [{i}/{len(configs)}] {cfg['name']}")
        vec = TfidfVectorizer(**cfg["tfidf"], strip_accents="unicode", dtype=np.float32)
        X = vec.fit_transform(train_texts.fillna(""))
        print(f"    Features: {X.shape[1]}")

        params = {"objective": "regression", "metric": "rmse", "verbose": -1, **cfg["lgb"]}
        rmse = cv_rmse(X, y_train, params, n_folds=3, n_sample=500_000)
        results.append((cfg["name"], rmse, cfg))
        print(f"    RMSE = {rmse:.5f}")

    results.sort(key=lambda x: x[1])
    best_name, best_rmse, best_cfg = results[0]
    print(f"\n  Best: {best_name}  RMSE = {best_rmse:.5f}")

    # Train final on full data
    print("\n[3/4] Training final model on full data …")
    vec = TfidfVectorizer(**best_cfg["tfidf"], strip_accents="unicode", dtype=np.float32)
    X_train = vec.fit_transform(train_texts.fillna(""))
    X_test = vec.transform(test_texts.fillna(""))
    print(f"  Features: {X_train.shape[1]}")

    final_params = {
        "objective": "regression", "metric": "rmse", "verbose": -1,
        "n_jobs": -1, "random_seed": RANDOM_SEED,
        **best_cfg["lgb"],
    }

    t0 = time.time()
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Predict
    preds = np.clip(model.predict(X_test), 1.0, 5.0)
    submission = pd.DataFrame({"id": test_ids, "rating": preds})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"  Submission → {SUBMISSION_PATH}")

    # Full-data CV for metrics
    print("\n  3-fold CV on full data …")
    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    cv_rmses = []
    for fold, (tr, va) in enumerate(kf.split(X_train), 1):
        m = lgb.LGBMRegressor(**final_params)
        m.fit(X_train[tr], y_train[tr])
        p = np.clip(m.predict(X_train[va]), 1.0, 5.0)
        r = float(np.sqrt(np.mean((p - y_train[va]) ** 2)))
        cv_rmses.append(r)
        print(f"    Fold {fold}: RMSE = {r:.5f}")
    full_cv = float(np.mean(cv_rmses))
    print(f"  Full CV RMSE: {full_cv:.5f}")

    # Update metrics
    print("\n[4/4] Updating metrics …")
    metrics_update = {
        "tfidf_optimized": {
            "cv_rmse": round(full_cv, 5),
            "train_time_sec": round(train_time, 2),
            "model": "lgb_tfidf_optimized",
            "best_config": best_name,
            "tfidf_params": {k: str(v) for k, v in best_cfg["tfidf"].items()},
            "lgb_params": best_cfg["lgb"],
            "note": "TF-IDF only, no target leakage"
        }
    }
    write_metrics(str(METRICS_PATH), metrics_update)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Best config: {best_name}")
    print(f"  Full CV RMSE: {full_cv:.5f}")
    print(f"  Stage 0 baseline: 1.17626 (Kaggle 0.80107)")
    print(f"  Total time: {total:.0f}s")
    print(f"  Submit: {SUBMISSION_PATH}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

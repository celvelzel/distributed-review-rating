#!/usr/bin/env python
"""Stage 0 Optimized: TF-IDF only + LightGBM hyperparameter search.

Strategy:
  Phase 1 – Grid-search ~20 TF-IDF × LightGBM configs on a 100K subsample (3-fold CV)
  Phase 2 – Retrain top-3 configs on FULL training data with 5-fold CV
  Phase 3 – Pick the best, train on all data, predict test set

Goal: beat Kaggle 0.80107 using ONLY TF-IDF features.
"""

from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import issparse, spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

# ── project root ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.utils.timer import StageTimer, timed, write_metrics

# ── paths ──────────────────────────────────────────────────────────────
TRAIN_PATH = ROOT / "artifacts" / "etl" / "train.parquet"
TEST_PATH = ROOT / "artifacts" / "etl" / "test.parquet"
SUBMISSION_PATH = ROOT / "output" / "submission-tfidf-optimized.csv"
METRICS_PATH = ROOT / "docs" / "changelog" / "metrics.json"

SUBSAMPLE_SIZE = 100_000   # for Phase 1 grid search
RANDOM_SEED = 42


# ═══════════════════════════════════════════════════════════════════════
# Configuration grids
# ═══════════════════════════════════════════════════════════════════════

# Hand-picked configurations covering diverse combinations.
# Each dict has "tfidf" and "lgb" sub-dicts.
# We don't do a full Cartesian product (would be 1000+ combos);
# instead we pick ~20 promising configurations.
CONFIGS: List[Dict[str, Any]] = [
    # ── Baseline reproduction (Stage 0) ───────────────────────────────
    {
        "name": "baseline_5k",
        "tfidf": {"max_features": 5000, "ngram_range": (1, 1), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        "lgb": {"learning_rate": 0.05, "num_leaves": 63, "n_estimators": 500, "max_depth": -1,
                "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                "reg_alpha": 0.0, "reg_lambda": 0.0},
    },
    # ── More features ─────────────────────────────────────────────────
    {
        "name": "tfidf_10k_uni",
        "tfidf": {"max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        "lgb": {"learning_rate": 0.05, "num_leaves": 63, "n_estimators": 500, "max_depth": -1,
                "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                "reg_alpha": 0.0, "reg_lambda": 0.0},
    },
    {
        "name": "tfidf_15k_uni",
        "tfidf": {"max_features": 15000, "ngram_range": (1, 1), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        "lgb": {"learning_rate": 0.05, "num_leaves": 63, "n_estimators": 500, "max_depth": -1,
                "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                "reg_alpha": 0.0, "reg_lambda": 0.0},
    },
    {
        "name": "tfidf_20k_uni",
        "tfidf": {"max_features": 20000, "ngram_range": (1, 1), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        "lgb": {"learning_rate": 0.05, "num_leaves": 63, "n_estimators": 500, "max_depth": -1,
                "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                "reg_alpha": 0.0, "reg_lambda": 0.0},
    },
    # ── Bigrams ───────────────────────────────────────────────────────
    {
        "name": "tfidf_10k_bi",
        "tfidf": {"max_features": 10000, "ngram_range": (1, 2), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        "lgb": {"learning_rate": 0.05, "num_leaves": 63, "n_estimators": 500, "max_depth": -1,
                "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                "reg_alpha": 0.0, "reg_lambda": 0.0},
    },
    {
        "name": "tfidf_15k_bi",
        "tfidf": {"max_features": 15000, "ngram_range": (1, 2), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        "lgb": {"learning_rate": 0.05, "num_leaves": 63, "n_estimators": 500, "max_depth": -1,
                "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                "reg_alpha": 0.0, "reg_lambda": 0.0},
    },
    {
        "name": "tfidf_20k_bi",
        "tfidf": {"max_features": 20000, "ngram_range": (1, 2), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        "lgb": {"learning_rate": 0.05, "num_leaves": 63, "n_estimators": 500, "max_depth": -1,
                "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                "reg_alpha": 0.0, "reg_lambda": 0.0},
    },
    # ── Trigrams ──────────────────────────────────────────────────────
    {
        "name": "tfidf_20k_tri",
        "tfidf": {"max_features": 20000, "ngram_range": (1, 3), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        "lgb": {"learning_rate": 0.05, "num_leaves": 63, "n_estimators": 500, "max_depth": -1,
                "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                "reg_alpha": 0.0, "reg_lambda": 0.0},
    },
    # ── No sublinear_tf ──────────────────────────────────────────────
    {
        "name": "tfidf_15k_bi_nosub",
        "tfidf": {"max_features": 15000, "ngram_range": (1, 2), "sublinear_tf": False, "min_df": 1, "max_df": 1.0},
        "lgb": {"learning_rate": 0.05, "num_leaves": 63, "n_estimators": 500, "max_depth": -1,
                "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                "reg_alpha": 0.0, "reg_lambda": 0.0},
    },
    # ── min_df / max_df variations ────────────────────────────────────
    {
        "name": "tfidf_15k_bi_mindf2",
        "tfidf": {"max_features": 15000, "ngram_range": (1, 2), "sublinear_tf": True, "min_df": 2, "max_df": 0.95},
        "lgb": {"learning_rate": 0.05, "num_leaves": 63, "n_estimators": 500, "max_depth": -1,
                "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                "reg_alpha": 0.0, "reg_lambda": 0.0},
    },
    {
        "name": "tfidf_15k_bi_mindf5",
        "tfidf": {"max_features": 15000, "ngram_range": (1, 2), "sublinear_tf": True, "min_df": 5, "max_df": 0.9},
        "lgb": {"learning_rate": 0.05, "num_leaves": 63, "n_estimators": 500, "max_depth": -1,
                "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                "reg_alpha": 0.0, "reg_lambda": 0.0},
    },
    # ── LightGBM: more estimators + lower lr ─────────────────────────
    {
        "name": "tfidf_15k_bi_slow1000",
        "tfidf": {"max_features": 15000, "ngram_range": (1, 2), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        "lgb": {"learning_rate": 0.01, "num_leaves": 63, "n_estimators": 1000, "max_depth": -1,
                "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                "reg_alpha": 0.0, "reg_lambda": 0.0},
    },
    {
        "name": "tfidf_15k_bi_slow2000",
        "tfidf": {"max_features": 15000, "ngram_range": (1, 2), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        "lgb": {"learning_rate": 0.01, "num_leaves": 63, "n_estimators": 2000, "max_depth": -1,
                "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                "reg_alpha": 0.0, "reg_lambda": 0.0},
    },
    # ── LightGBM: more leaves ─────────────────────────────────────────
    {
        "name": "tfidf_15k_bi_127leaves",
        "tfidf": {"max_features": 15000, "ngram_range": (1, 2), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        "lgb": {"learning_rate": 0.05, "num_leaves": 127, "n_estimators": 500, "max_depth": -1,
                "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                "reg_alpha": 0.0, "reg_lambda": 0.0},
    },
    {
        "name": "tfidf_15k_bi_255leaves",
        "tfidf": {"max_features": 15000, "ngram_range": (1, 2), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        "lgb": {"learning_rate": 0.05, "num_leaves": 255, "n_estimators": 500, "max_depth": -1,
                "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
                "reg_alpha": 0.0, "reg_lambda": 0.0},
    },
    # ── LightGBM: subsample + colsample (regularization) ─────────────
    {
        "name": "tfidf_15k_bi_sub08_col08",
        "tfidf": {"max_features": 15000, "ngram_range": (1, 2), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        "lgb": {"learning_rate": 0.05, "num_leaves": 127, "n_estimators": 500, "max_depth": -1,
                "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
                "reg_alpha": 0.0, "reg_lambda": 0.0},
    },
    {
        "name": "tfidf_15k_bi_sub06_col06",
        "tfidf": {"max_features": 15000, "ngram_range": (1, 2), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        "lgb": {"learning_rate": 0.05, "num_leaves": 127, "n_estimators": 500, "max_depth": -1,
                "min_child_samples": 20, "subsample": 0.6, "colsample_bytree": 0.6,
                "reg_alpha": 0.0, "reg_lambda": 0.0},
    },
    # ── LightGBM: L1/L2 regularization ──────────────────────────────
    {
        "name": "tfidf_15k_bi_reg",
        "tfidf": {"max_features": 15000, "ngram_range": (1, 2), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        "lgb": {"learning_rate": 0.05, "num_leaves": 127, "n_estimators": 500, "max_depth": -1,
                "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
                "reg_alpha": 0.1, "reg_lambda": 1.0},
    },
    # ── Combined best guess: big TF-IDF + deep LGB ───────────────────
    {
        "name": "tfidf_20k_bi_deep",
        "tfidf": {"max_features": 20000, "ngram_range": (1, 2), "sublinear_tf": True, "min_df": 2, "max_df": 0.95},
        "lgb": {"learning_rate": 0.01, "num_leaves": 255, "n_estimators": 2000, "max_depth": -1,
                "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
                "reg_alpha": 0.1, "reg_lambda": 1.0},
    },
    # ── Max features push ────────────────────────────────────────────
    {
        "name": "tfidf_20k_bi_aggressive",
        "tfidf": {"max_features": 20000, "ngram_range": (1, 2), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        "lgb": {"learning_rate": 0.05, "num_leaves": 255, "n_estimators": 1000, "max_depth": -1,
                "min_child_samples": 10, "subsample": 0.8, "colsample_bytree": 0.8,
                "reg_alpha": 0.0, "reg_lambda": 0.0},
    },
]


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _combine_text(df: pd.DataFrame) -> pd.Series:
    """Concatenate review title and comment."""
    return (df["title"].fillna("") + " " + df["comment"].fillna("")).str.strip()


def _extract_tfidf(
    train_texts: pd.Series,
    test_texts: pd.Series,
    tfidf_params: Dict[str, Any],
) -> Tuple[Any, Any, TfidfVectorizer]:
    """Fit TF-IDF with given params and transform both splits."""
    vec_params = {
        "max_features": tfidf_params.get("max_features", 5000),
        "ngram_range": tfidf_params.get("ngram_range", (1, 1)),
        "sublinear_tf": tfidf_params.get("sublinear_tf", True),
        "min_df": tfidf_params.get("min_df", 1),
        "max_df": tfidf_params.get("max_df", 1.0),
        "strip_accents": "unicode",
    }
    vectorizer = TfidfVectorizer(**vec_params)
    X_train = vectorizer.fit_transform(train_texts.fillna(""))
    X_test = vectorizer.transform(test_texts.fillna("")) if test_texts is not None else None
    return X_train, X_test, vectorizer


def _train_lgb(
    X_train: Any,
    y_train: np.ndarray,
    lgb_params: Dict[str, Any],
) -> lgb.LGBMRegressor:
    """Train LightGBM with given params."""
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1,
        "learning_rate": lgb_params.get("learning_rate", 0.05),
        "num_leaves": lgb_params.get("num_leaves", 63),
        "n_estimators": lgb_params.get("n_estimators", 500),
        "max_depth": lgb_params.get("max_depth", -1),
        "min_child_samples": lgb_params.get("min_child_samples", 20),
        "subsample": lgb_params.get("subsample", 1.0),
        "colsample_bytree": lgb_params.get("colsample_bytree", 1.0),
        "reg_alpha": lgb_params.get("reg_alpha", 0.0),
        "reg_lambda": lgb_params.get("reg_lambda", 0.0),
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model


def _cv_rmse(
    X_all: Any,
    y_all: np.ndarray,
    lgb_params: Dict[str, Any],
    n_splits: int = 3,
) -> float:
    """K-fold CV RMSE."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    rmses: List[float] = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all), 1):
        X_tr, X_val = X_all[train_idx], X_all[val_idx]
        y_tr, y_val = y_all[train_idx], y_all[val_idx]
        model = _train_lgb(X_tr, y_tr, lgb_params)
        preds = np.clip(model.predict(X_val), 1.0, 5.0)
        rmse = np.sqrt(np.mean((preds - y_val) ** 2))
        rmses.append(rmse)
    return float(np.mean(rmses))


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Grid search on subsample
# ═══════════════════════════════════════════════════════════════════════

def phase1_grid_search(
    train_df: pd.DataFrame,
    n_configs: int = 20,
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """Run grid search on a subsample. Returns sorted (name, rmse, config) list."""
    print(f"\n{'='*70}")
    print(f"PHASE 1: Grid search on {SUBSAMPLE_SIZE:,} subsample × 3-fold CV")
    print(f"{'='*70}")

    # Subsample
    sub_df = train_df.sample(n=min(SUBSAMPLE_SIZE, len(train_df)), random_state=RANDOM_SEED)
    sub_texts = _combine_text(sub_df)
    y_sub = sub_df["rating"].values.astype(np.float32)

    results: List[Tuple[str, float, Dict[str, Any]]] = []
    total = min(n_configs, len(CONFIGS))

    for i, cfg in enumerate(CONFIGS[:total], 1):
        name = cfg["name"]
        tfidf_params = cfg["tfidf"]
        lgb_params = cfg["lgb"]

        print(f"\n  [{i}/{total}] {name}")
        print(f"    TF-IDF: max_feat={tfidf_params['max_features']}, "
              f"ngram={tfidf_params['ngram_range']}, sublinear={tfidf_params['sublinear_tf']}, "
              f"min_df={tfidf_params['min_df']}, max_df={tfidf_params['max_df']}")
        print(f"    LGB: lr={lgb_params['learning_rate']}, leaves={lgb_params['num_leaves']}, "
              f"n_est={lgb_params['n_estimators']}, sub={lgb_params['subsample']}, "
              f"col={lgb_params['colsample_bytree']}")

        t0 = time.perf_counter()

        # Extract TF-IDF on subsample
        X_sub, _, _ = _extract_tfidf(sub_texts, None, tfidf_params)
        print(f"    TF-IDF shape: {X_sub.shape}")

        # CV
        rmse = _cv_rmse(X_sub, y_sub, lgb_params, n_splits=3)
        elapsed = time.perf_counter() - t0

        results.append((name, rmse, cfg))
        print(f"    → RMSE = {rmse:.5f}  ({elapsed:.1f}s)")

    # Sort by RMSE ascending
    results.sort(key=lambda x: x[1])
    return results


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Retrain top-N on full data with 5-fold CV
# ═══════════════════════════════════════════════════════════════════════

def phase2_full_cv(
    train_df: pd.DataFrame,
    top_configs: List[Tuple[str, float, Dict[str, Any]]],
    n_top: int = 3,
) -> Tuple[str, float, Dict[str, Any], Any, TfidfVectorizer]:
    """Retrain top configs on full data with 5-fold CV. Returns best config + fitted vectorizer."""
    print(f"\n{'='*70}")
    print(f"PHASE 2: Retrain top-{n_top} on FULL data ({len(train_df):,}) × 5-fold CV")
    print(f"{'='*70}")

    train_texts = _combine_text(train_df)
    y_all = train_df["rating"].values.astype(np.float32)

    best_name = ""
    best_rmse = float("inf")
    best_cfg = None
    best_X = None
    best_vec = None

    for i, (name, subsample_rmse, cfg) in enumerate(top_configs[:n_top], 1):
        tfidf_params = cfg["tfidf"]
        lgb_params = cfg["lgb"]

        print(f"\n  [{i}/{n_top}] {name} (subsample RMSE: {subsample_rmse:.5f})")

        t0 = time.perf_counter()

        # Extract TF-IDF on full data
        X_full, _, vectorizer = _extract_tfidf(train_texts, None, tfidf_params)
        print(f"    TF-IDF shape: {X_full.shape}")

        # 5-fold CV
        rmse = _cv_rmse(X_full, y_all, lgb_params, n_splits=5)
        elapsed = time.perf_counter() - t0

        print(f"    → Full 5-fold RMSE = {rmse:.5f}  ({elapsed:.1f}s)")

        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_cfg = cfg
            best_X = X_full
            best_vec = vectorizer

    print(f"\n  ★ Best config: {best_name}  (5-fold RMSE: {best_rmse:.5f})")
    return best_name, best_rmse, best_cfg, best_X, best_vec


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Train on all data + predict test set
# ═══════════════════════════════════════════════════════════════════════

def phase3_train_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    best_cfg: Dict[str, Any],
    best_name: str,
) -> Tuple[float, str]:
    """Train final model on all training data and generate submission."""
    print(f"\n{'='*70}")
    print(f"PHASE 3: Train final model ({best_name}) + predict test set")
    print(f"{'='*70}")

    train_texts = _combine_text(train_df)
    test_texts = _combine_text(test_df)
    y_train = train_df["rating"].values.astype(np.float32)
    test_ids = test_df["id"].values

    tfidf_params = best_cfg["tfidf"]
    lgb_params = best_cfg["lgb"]

    # Extract TF-IDF
    print("  Extracting TF-IDF features …")
    X_train, X_test, vectorizer = _extract_tfidf(train_texts, test_texts, tfidf_params)
    print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")

    # Train
    print("  Training LightGBM on all data …")
    t0 = time.perf_counter()
    model = _train_lgb(X_train, y_train, lgb_params)
    train_time = time.perf_counter() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Predict
    print("  Predicting on test set …")
    t0 = time.perf_counter()
    preds = model.predict(X_test)
    preds = np.clip(preds, 1.0, 5.0)
    inf_time = time.perf_counter() - t0

    # Save submission
    submission = pd.DataFrame({"id": test_ids, "rating": preds})
    SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"  Submission saved → {SUBMISSION_PATH}  ({len(submission):,} rows)")
    print(f"  Inference time: {inf_time:.1f}s")
    print(f"  Prediction stats: min={preds.min():.3f}, max={preds.max():.3f}, "
          f"mean={preds.mean():.3f}, std={preds.std():.3f}")

    return train_time, inf_time, submission


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("TF-IDF Optimized LightGBM — Hyperparameter Search")
    print(f"Goal: beat Kaggle 0.80107 using ONLY TF-IDF features")
    print(f"Configs to try: {len(CONFIGS)}")
    print("=" * 70)

    t_total = time.perf_counter()

    # Load data
    print("\n[1/4] Loading data …")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    print(f"  train: {len(train_df):,} rows  |  test: {len(test_df):,} rows")

    # Phase 1: grid search on subsample
    print("\n[2/4] Phase 1: Grid search …")
    results = phase1_grid_search(train_df, n_configs=len(CONFIGS))

    # Print Phase 1 summary
    print(f"\n{'─'*70}")
    print("PHASE 1 RESULTS (sorted by RMSE):")
    print(f"{'─'*70}")
    for rank, (name, rmse, _) in enumerate(results, 1):
        marker = " ★" if rank <= 3 else ""
        print(f"  {rank:2d}. {name:40s} RMSE = {rmse:.5f}{marker}")
    print(f"{'─'*70}")

    # Phase 2: retrain top 3 on full data
    print("\n[3/4] Phase 2: Full-data validation of top-3 …")
    best_name, best_cv_rmse, best_cfg, X_full, best_vec = phase2_full_cv(
        train_df, results, n_top=3,
    )

    # Phase 3: train final + predict
    print("\n[4/4] Phase 3: Final training + prediction …")
    train_time, inf_time, submission = phase3_train_and_predict(
        train_df, test_df, best_cfg, best_name,
    )

    # ── Write metrics ──────────────────────────────────────────────────
    tfidf_params = best_cfg["tfidf"]
    lgb_params = best_cfg["lgb"]
    metrics_update = {
        "tfidf_optimized": {
            "cv_rmse_5fold": round(best_cv_rmse, 5),
            "train_time_sec": round(train_time, 2),
            "inference_time_sec": round(inf_time, 2),
            "model": "lgb_tfidf_optimized",
            "features": ["tfidf_only"],
            "best_config_name": best_name,
            "tfidf_params": {
                "max_features": tfidf_params["max_features"],
                "ngram_range": list(tfidf_params["ngram_range"]),
                "sublinear_tf": tfidf_params["sublinear_tf"],
                "min_df": tfidf_params["min_df"],
                "max_df": tfidf_params["max_df"],
            },
            "lgb_params": lgb_params,
            "submission": str(SUBMISSION_PATH.relative_to(ROOT)),
            "submission_lines": len(submission),
            "phase1_results": [
                {"name": name, "rmse": round(rmse, 5)}
                for name, rmse, _ in results[:10]
            ],
        }
    }
    write_metrics(str(METRICS_PATH), metrics_update)

    # ── Final summary ──────────────────────────────────────────────────
    total_time = time.perf_counter() - t_total
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  Best config:       {best_name}")
    print(f"  5-fold CV RMSE:    {best_cv_rmse:.5f}")
    print(f"  Stage 0 RMSE:      1.17626")
    print(f"  Δ RMSE:            {best_cv_rmse - 1.17626:+.5f}")
    print(f"  Total time:        {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"  Submission:        {SUBMISSION_PATH}")
    print(f"  Metrics:           {METRICS_PATH}")
    print(f"\n  TF-IDF params:     {json.dumps(tfidf_params, default=str)}")
    print(f"  LGB params:        {json.dumps(lgb_params)}")
    print(f"{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()

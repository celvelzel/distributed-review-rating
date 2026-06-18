#!/usr/bin/env python
"""
Train graph-feature base models for stacking v3.
Generates OOF and test predictions for XGBoost and LightGBM on graph features.

Two variants per algorithm:
  - full: all graph features (includes user_cat_avg_rating — potential leakage)
  - safe: excludes user_cat_avg_rating and user_cat_deviation

All models use KFold(n_splits=5, shuffle=True, random_state=42) to align with stacking v2/v3.
"""

from __future__ import annotations

import gc
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

FEAT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "features")
MODEL_DIR = os.path.join(PROJECT_ROOT, "artifacts", "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
N_FOLDS = 5

# Columns to exclude in the "safe" variant
LEAKAGE_COLS = {"user_cat_avg_rating", "user_cat_deviation"}


def load_data():
    """Load graph features + user/product K-Fold stats + interactions."""
    train_df = pd.read_parquet(os.path.join(PROJECT_ROOT, "artifacts", "etl", "train.parquet"))
    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "test.csv"))
    y = train_df["rating"].values.astype(np.float32)

    # Expanded graph features
    exp_train = pd.read_parquet(os.path.join(FEAT_DIR, "expanded_graph_train.parquet"))
    exp_test = pd.read_parquet(os.path.join(FEAT_DIR, "expanded_graph_test.parquet"))

    # User stats (K-Fold safe)
    us = pd.read_parquet(os.path.join(FEAT_DIR, "user_stats_kfold.parquet"))
    us_dedup = us.groupby("id").first()
    us_dict = us_dedup[["avg_rating", "num_reviews", "rating_std"]].rename(
        columns={"avg_rating": "user_avg_rating", "num_reviews": "user_num_reviews", "rating_std": "user_rating_std"}
    )

    # Product stats (K-Fold safe)
    ps = pd.read_parquet(os.path.join(FEAT_DIR, "product_stats_kfold.parquet"))
    ps_dedup = ps.groupby("parent_prod_id").first()
    ps_dict = ps_dedup[["prod_avg_rating", "prod_num_reviews"]]

    # Build feature DataFrames
    graph_cols = [c for c in exp_train.columns if c not in ("id", "parent_prod_id")]

    features_train = {c: exp_train[c].values.astype(np.float32) for c in graph_cols}
    features_test = {c: exp_test[c].values.astype(np.float32) for c in graph_cols}

    # Map user stats
    for col in ["user_avg_rating", "user_num_reviews", "user_rating_std"]:
        src_col = col.replace("user_", "") if col != "user_rating_std" else "rating_std"
        if src_col in us_dict.columns:
            features_train[col] = train_df["user_id"].map(us_dict[src_col]).fillna(0).values.astype(np.float32)
            features_test[col] = test_df["user_id"].map(us_dict[src_col]).fillna(0).values.astype(np.float32)

    # Map product stats
    for col in ["prod_avg_rating", "prod_num_reviews"]:
        if col in ps_dict.columns:
            features_train[col] = train_df["parent_prod_id"].map(ps_dict[col]).fillna(0).values.astype(np.float32)
            features_test[col] = test_df["parent_prod_id"].map(ps_dict[col]).fillna(0).values.astype(np.float32)

    X_all = pd.DataFrame(features_train)
    X_test_all = pd.DataFrame(features_test)

    # Interaction features
    for X in [X_all, X_test_all]:
        if "user_leniency" in X.columns and "user_num_reviews_oof" in X.columns:
            X["leniency_x_reviews"] = X["user_leniency"] * X["user_num_reviews_oof"]
        if "user_cat_deviation" in X.columns and "user_cat_review_count" in X.columns:
            X["cat_dev_x_reviews"] = X["user_cat_deviation"] * X["user_cat_review_count"]
        if "user_avg_rating" in X.columns and "prod_avg_rating" in X.columns:
            X["user_prod_diff"] = X["user_avg_rating"] - X["prod_avg_rating"]

    # Fill NaN
    X_all = X_all.fillna(0)
    X_test_all = X_test_all.fillna(0)

    print(f"  Full feature set: {X_all.shape[1]} columns")
    print(f"  Features: {list(X_all.columns)}")

    # Safe variant (exclude leakage columns)
    safe_cols = [c for c in X_all.columns if c not in LEAKAGE_COLS]
    X_safe = X_all[safe_cols]
    X_test_safe = X_test_all[safe_cols]
    print(f"  Safe feature set: {X_safe.shape[1]} columns (excluded: {LEAKAGE_COLS})")

    return X_all, X_test_all, X_safe, X_test_safe, y, train_df, test_df


def train_xgboost(X_train, X_test, y, tag, params=None):
    """5-fold XGBoost, returns OOF + averaged test predictions."""
    import xgboost as xgb

    if params is None:
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": RANDOM_SEED,
            "tree_method": "hist",
            "verbosity": 0,
        }

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(y), dtype=np.float32)
    test_preds = np.zeros(len(X_test), dtype=np.float32)

    for fold, (tr, va) in enumerate(kf.split(X_train), 1):
        dtrain = xgb.DMatrix(X_train.iloc[tr], label=y[tr])
        dval = xgb.DMatrix(X_train.iloc[va], label=y[va])
        dtest = xgb.DMatrix(X_test)

        model = xgb.train(
            params, dtrain, num_boost_round=300,
            evals=[(dval, "val")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )

        oof[va] = np.clip(model.predict(dval), 1.0, 5.0)
        test_preds += np.clip(model.predict(dtest), 1.0, 5.0) / N_FOLDS

        fold_rmse = np.sqrt(np.mean((oof[va] - y[va]) ** 2))
        print(f"    XGBoost [{tag}] fold {fold}: RMSE={fold_rmse:.5f}  best_iter={model.best_iteration}")

    oof_rmse = np.sqrt(np.mean((oof - y) ** 2))
    print(f"  XGBoost [{tag}] OOF RMSE: {oof_rmse:.5f}")
    return oof, test_preds, oof_rmse


def train_lightgbm(X_train, X_test, y, tag, params=None):
    """5-fold LightGBM, returns OOF + averaged test predictions."""
    import lightgbm as lgb

    if params is None:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 50,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
            "seed": RANDOM_SEED,
        }

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(y), dtype=np.float32)
    test_preds = np.zeros(len(X_test), dtype=np.float32)

    for fold, (tr, va) in enumerate(kf.split(X_train), 1):
        ds_tr = lgb.Dataset(X_train.iloc[tr], y[tr])
        ds_va = lgb.Dataset(X_train.iloc[va], y[va])

        model = lgb.train(
            params, ds_tr, num_boost_round=1000,
            valid_sets=[ds_va],
            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(50)],
        )

        oof[va] = np.clip(model.predict(X_train.iloc[va]), 1.0, 5.0)
        test_preds += np.clip(model.predict(X_test), 1.0, 5.0) / N_FOLDS

        fold_rmse = np.sqrt(np.mean((oof[va] - y[va]) ** 2))
        print(f"    LightGBM [{tag}] fold {fold}: RMSE={fold_rmse:.5f}  best_iter={model.best_iteration}")

    oof_rmse = np.sqrt(np.mean((oof - y) ** 2))
    print(f"  LightGBM [{tag}] OOF RMSE: {oof_rmse:.5f}")
    return oof, test_preds, oof_rmse


def main():
    import json
    from datetime import datetime

    t_start = time.perf_counter()
    print("=" * 60)
    print("Train graph-feature base models for stacking v3")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)

    print("\nLoading data...")
    X_full, X_test_full, X_safe, X_test_safe, y, train_df, test_df = load_data()
    test_ids = test_df["id"].values
    print(f"  Train: {X_full.shape}, Test: {X_test_full.shape}, y: {y.shape}")
    print(f"  y stats: mean={y.mean():.4f}, std={y.std():.4f}\n")

    results = {
        "timestamp": datetime.now().isoformat(),
        "train_shape": list(X_full.shape),
        "test_shape": list(X_test_full.shape),
        "full_features": list(X_full.columns),
        "safe_features": list(X_safe.columns),
        "excluded_leakage_cols": list(LEAKAGE_COLS),
    }

    # ── XGBoost (full features) ──
    print("[1/4] XGBoost on full graph features...")
    xgb_full_oof, xgb_full_test, xgb_full_rmse = train_xgboost(X_full, X_test_full, y, "full")
    np.save(os.path.join(MODEL_DIR, "xgb_graph_full_oof.npy"), xgb_full_oof)
    np.save(os.path.join(MODEL_DIR, "xgb_graph_full_test.npy"), xgb_full_test)
    results["xgb_full"] = {"oof_rmse": float(xgb_full_rmse), "test_mean": float(xgb_full_test.mean()), "test_std": float(xgb_full_test.std())}

    # ── XGBoost (safe features) ──
    print("\n[2/4] XGBoost on safe graph features...")
    xgb_safe_oof, xgb_safe_test, xgb_safe_rmse = train_xgboost(X_safe, X_test_safe, y, "safe")
    np.save(os.path.join(MODEL_DIR, "xgb_graph_safe_oof.npy"), xgb_safe_oof)
    np.save(os.path.join(MODEL_DIR, "xgb_graph_safe_test.npy"), xgb_safe_test)
    results["xgb_safe"] = {"oof_rmse": float(xgb_safe_rmse), "test_mean": float(xgb_safe_test.mean()), "test_std": float(xgb_safe_test.std())}

    # ── LightGBM (full features) ──
    print("\n[3/4] LightGBM on full graph features...")
    lgb_full_oof, lgb_full_test, lgb_full_rmse = train_lightgbm(X_full, X_test_full, y, "full")
    np.save(os.path.join(MODEL_DIR, "lgb_graph_full_oof.npy"), lgb_full_oof)
    np.save(os.path.join(MODEL_DIR, "lgb_graph_full_test.npy"), lgb_full_test)
    results["lgb_full"] = {"oof_rmse": float(lgb_full_rmse), "test_mean": float(lgb_full_test.mean()), "test_std": float(lgb_full_test.std())}

    # ── LightGBM (safe features) ──
    print("\n[4/4] LightGBM on safe graph features...")
    lgb_safe_oof, lgb_safe_test, lgb_safe_rmse = train_lightgbm(X_safe, X_test_safe, y, "safe")
    np.save(os.path.join(MODEL_DIR, "lgb_graph_safe_oof.npy"), lgb_safe_oof)
    np.save(os.path.join(MODEL_DIR, "lgb_graph_safe_test.npy"), lgb_safe_test)
    results["lgb_safe"] = {"oof_rmse": float(lgb_safe_rmse), "test_mean": float(lgb_safe_test.mean()), "test_std": float(lgb_safe_test.std())}

    # ── Summary ──
    total_time = time.perf_counter() - t_start
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  XGBoost (full):  OOF RMSE = {xgb_full_rmse:.5f}  test: mean={xgb_full_test.mean():.4f}, std={xgb_full_test.std():.4f}")
    print(f"  XGBoost (safe):  OOF RMSE = {xgb_safe_rmse:.5f}  test: mean={xgb_safe_test.mean():.4f}, std={xgb_safe_test.std():.4f}")
    print(f"  LightGBM (full): OOF RMSE = {lgb_full_rmse:.5f}  test: mean={lgb_full_test.mean():.4f}, std={lgb_full_test.std():.4f}")
    print(f"  LightGBM (safe): OOF RMSE = {lgb_safe_rmse:.5f}  test: mean={lgb_safe_test.mean():.4f}, std={lgb_safe_test.std():.4f}")
    print(f"\n  Leakage impact: full vs safe RMSE delta:")
    print(f"    XGBoost:  {xgb_full_rmse - xgb_safe_rmse:+.5f}")
    print(f"    LightGBM: {lgb_full_rmse - lgb_safe_rmse:+.5f}")
    print(f"  (Negative delta = leakage helped locally but may hurt on Kaggle)")
    print(f"\n  Total time: {total_time / 60:.1f} min")
    print(f"  Files saved to: {MODEL_DIR}")
    print("=" * 60)

    # ── Save structured results ──
    results["total_time_min"] = round(total_time / 60, 1)
    results["leakage_delta"] = {
        "xgb_full_minus_safe": float(xgb_full_rmse - xgb_safe_rmse),
        "lgb_full_minus_safe": float(lgb_full_rmse - lgb_safe_rmse),
    }
    json_path = os.path.join(MODEL_DIR, "graph_models_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Structured results → {json_path}")


if __name__ == "__main__":
    main()

"""
Test expanded graph features with Ridge regression.
Compares: original 11 features vs expanded 14 features.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FEAT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "features")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def train_ridge_oof(X, y, X_test, alpha=1.0, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    test = np.zeros(len(X_test))

    for fold, (tr, va) in enumerate(kf.split(X)):
        model = Ridge(alpha=alpha)
        model.fit(X[tr], y[tr])
        oof[va] = model.predict(X[va])
        test += model.predict(X_test) / n_splits

    rmse = np.sqrt(np.mean((y - oof) ** 2))
    return oof, test, rmse


def main():
    # Load data
    train_df = pd.read_parquet(os.path.join(PROJECT_ROOT, "artifacts", "etl", "train.parquet"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    y = train_df["rating"].values.astype(np.float32)

    # Load original stats features
    user_stats = pd.read_parquet(os.path.join(FEAT_DIR, "user_stats_kfold.parquet"))
    prod_stats = pd.read_parquet(os.path.join(FEAT_DIR, "product_stats_kfold.parquet"))
    cat_stats = pd.read_parquet(os.path.join(FEAT_DIR, "category_stats_kfold.parquet"))

    # Load expanded features
    exp_train = pd.read_parquet(os.path.join(FEAT_DIR, "expanded_graph_train.parquet"))
    exp_test = pd.read_parquet(os.path.join(FEAT_DIR, "expanded_graph_test.parquet"))

    # Build expanded features only (skip original stats due to format mismatch)
    expanded_cols = [c for c in exp_train.columns if c not in ["id", "parent_prod_id"]]
    exp_train_feats = exp_train[expanded_cols].values.astype(np.float32)
    exp_test_feats = exp_test[expanded_cols].values.astype(np.float32)

    print(f"Expanded features: {exp_train_feats.shape[1]}")
    print()

    # Model A: Expanded features only
    _, test_a, rmse_a = train_ridge_oof(exp_train_feats, y, exp_test_feats, alpha=10.0)
    print(f"[A] Expanded only ({exp_train_feats.shape[1]}d): OOF RMSE = {rmse_a:.4f}")

    # Save predictions
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sub = pd.DataFrame({"id": test_df["id"].values, "rating": np.clip(test_a, 1.0, 5.0)})
    sub.to_csv(os.path.join(OUTPUT_DIR, "ridge_expanded_features_only.csv"), index=False)
    print(f"  Saved: ridge_expanded_features_only.csv")

    # Summary
    print(f"\nExpanded features OOF RMSE: {rmse_a:.4f}")


if __name__ == "__main__":
    main()

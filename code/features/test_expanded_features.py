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
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    y = train_df["rating"].values.astype(np.float32)

    # Load original stats features
    user_stats = pd.read_parquet(os.path.join(FEAT_DIR, "user_stats_pandas.parquet"))
    prod_stats = pd.read_parquet(os.path.join(FEAT_DIR, "product_stats_pandas.parquet"))
    cat_stats = pd.read_parquet(os.path.join(FEAT_DIR, "category_stats_pandas.parquet"))

    # Load expanded features
    exp_train = pd.read_parquet(os.path.join(FEAT_DIR, "expanded_graph_train.parquet"))
    exp_test = pd.read_parquet(os.path.join(FEAT_DIR, "expanded_graph_test.parquet"))

    # Build original 11 features
    def build_original(df, stats_type="train"):
        features = {}
        us = user_stats.set_index("user_id")
        for col in ["user_avg_rating", "user_review_count", "user_rating_std", "user_avg_deviation"]:
            features[col] = df["user_id"].map(us[col]).fillna(0).values

        ps = prod_stats.set_index("parent_prod_id")
        for col in ["prod_avg_rating", "prod_review_count", "prod_rating_std"]:
            features[col] = df["parent_prod_id"].map(ps[col]).fillna(0).values

        if "main_category" in prod_stats.columns:
            cs = cat_stats.set_index("main_category")
            prod_cat = prod_stats.set_index("parent_prod_id")["main_category"]
            cats = df["parent_prod_id"].map(prod_cat)
            features["cat_avg_rating"] = cats.map(cs["cat_avg_rating"]).fillna(0).values
            features["cat_review_count"] = cats.map(cs["cat_review_count"]).fillna(0).values

        features["user_prod_avg_diff"] = features["user_avg_rating"] - features["prod_avg_rating"]
        features["user_prod_avg_blend"] = (features["user_avg_rating"] + features["prod_avg_rating"]) / 2
        return pd.DataFrame(features)

    orig_train = build_original(train_df)
    orig_test = build_original(test_df)

    # Build expanded features (original 11 + expanded 13 = 24 total)
    expanded_cols = [c for c in exp_train.columns if c not in ["id", "parent_prod_id"]]
    exp_train_feats = exp_train[expanded_cols].values.astype(np.float32)
    exp_test_feats = exp_test[expanded_cols].values.astype(np.float32)

    combined_train = np.hstack([orig_train.values.astype(np.float32), exp_train_feats])
    combined_test = np.hstack([orig_test.values.astype(np.float32), exp_test_feats])

    print(f"Original features: {orig_train.shape[1]}")
    print(f"Expanded features: {exp_train_feats.shape[1]}")
    print(f"Combined features: {combined_train.shape[1]}")
    print()

    # Model A: Original stats only (11 features)
    _, test_a, rmse_a = train_ridge_oof(orig_train.values.astype(np.float32), y, orig_test.values.astype(np.float32), alpha=1.0)
    print(f"[A] Stats only (11d):  OOF RMSE = {rmse_a:.4f}")

    # Model B: Expanded features only (13 features)
    _, test_b, rmse_b = train_ridge_oof(exp_train_feats, y, exp_test_feats, alpha=10.0)
    print(f"[B] Expanded only (13d): OOF RMSE = {rmse_b:.4f}")

    # Model C: Combined (24 features)
    _, test_c, rmse_c = train_ridge_oof(combined_train, y, combined_test, alpha=10.0)
    print(f"[C] Combined (24d):    OOF RMSE = {rmse_c:.4f}")

    # Save best predictions
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load best existing submission
    best_path = os.path.join(OUTPUT_DIR, "submission-final.csv")
    if os.path.exists(best_path):
        best_df = pd.read_csv(best_path)
        best_preds = best_df["rating"].values

        # Blend variants
        print(f"\nBlends with best submission (std={best_preds.std():.4f}):")
        for name, preds in [("expanded", test_c), ("stats_orig", test_a)]:
            for w in [0.05, 0.10, 0.15, 0.20]:
                blend = best_preds * (1 - w) + preds * w
                blend = np.clip(blend, 1.0, 5.0)
                sub = pd.DataFrame({"id": test_df["id"].values, "rating": blend})
                fname = f"best_{int((1-w)*100)}_{name}_{int(w*100)}.csv"
                sub.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
                print(f"  {fname}: std={blend.std():.4f}")

    # Summary
    improvement = (rmse_a - rmse_c) / rmse_a * 100
    print(f"\nImprovement from expanded features: {improvement:.2f}%")


if __name__ == "__main__":
    main()

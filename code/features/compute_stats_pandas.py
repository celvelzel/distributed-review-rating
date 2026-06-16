"""
Compute user/product/category statistical features using pandas (no PySpark needed).

These features serve as auxiliary signals for DeBERTa prediction calibration:
- User stats: avg_rating, review_count, rating_std, median_rating
- Product stats: avg_rating, review_count, rating_std, price
- Category stats: avg_rating, review_count

Output: artifacts/features/user_stats_pandas.parquet
        artifacts/features/product_stats_pandas.parquet
        artifacts/features/category_stats_pandas.parquet
"""

import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "features")


def compute_user_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-user aggregated statistics."""
    print("  Computing user stats...")
    t0 = time.time()

    user_stats = train_df.groupby("user_id").agg(
        user_avg_rating=("rating", "mean"),
        user_review_count=("rating", "count"),
        user_rating_std=("rating", "std"),
        user_median_rating=("rating", "median"),
        user_min_rating=("rating", "min"),
        user_max_rating=("rating", "max"),
    ).reset_index()

    # Fill NaN std (users with only 1 review)
    user_stats["user_rating_std"] = user_stats["user_rating_std"].fillna(0.0)

    # Derived features
    user_stats["user_rating_range"] = user_stats["user_max_rating"] - user_stats["user_min_rating"]
    user_stats["user_is_active"] = (user_stats["user_review_count"] >= 5).astype(int)

    print(f"    {len(user_stats):,} users computed in {time.time()-t0:.1f}s")
    return user_stats


def compute_product_stats(train_df: pd.DataFrame, prodinfo_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-product aggregated statistics."""
    print("  Computing product stats...")
    t0 = time.time()

    # Review-side aggregates
    prod_stats = train_df.groupby("parent_prod_id").agg(
        prod_avg_rating=("rating", "mean"),
        prod_review_count=("rating", "count"),
        prod_rating_std=("rating", "std"),
        prod_median_rating=("rating", "median"),
    ).reset_index()

    prod_stats["prod_rating_std"] = prod_stats["prod_rating_std"].fillna(0.0)

    # Merge with product metadata
    if prodinfo_df is not None:
        meta = prodinfo_df[["parent_prod_id", "price", "rating_number", "main_category"]].drop_duplicates("parent_prod_id")
        prod_stats = prod_stats.merge(meta, on="parent_prod_id", how="left")

    # Derived features
    prod_stats["prod_is_popular"] = (prod_stats["prod_review_count"] >= 10).astype(int)
    prod_stats["log_review_count"] = np.log1p(prod_stats["prod_review_count"])

    print(f"    {len(prod_stats):,} products computed in {time.time()-t0:.1f}s")
    return prod_stats


def compute_category_stats(train_df: pd.DataFrame, prodinfo_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-category aggregated statistics."""
    print("  Computing category stats...")
    t0 = time.time()

    # Merge category info
    merged = train_df.merge(
        prodinfo_df[["parent_prod_id", "main_category"]].drop_duplicates("parent_prod_id"),
        on="parent_prod_id",
        how="left"
    )

    cat_stats = merged.groupby("main_category").agg(
        cat_avg_rating=("rating", "mean"),
        cat_review_count=("rating", "count"),
        cat_rating_std=("rating", "std"),
    ).reset_index()

    cat_stats["cat_rating_std"] = cat_stats["cat_rating_std"].fillna(0.0)

    print(f"    {len(cat_stats):,} categories computed in {time.time()-t0:.1f}s")
    return cat_stats


def compute_user_product_interaction_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    """Compute user-product interaction features (for each review)."""
    print("  Computing user-product interaction stats...")
    t0 = time.time()

    # User's deviation from product average
    prod_avg = train_df.groupby("parent_prod_id")["rating"].mean()
    train_df = train_df.copy()
    train_df["prod_avg"] = train_df["parent_prod_id"].map(prod_avg)
    train_df["user_prod_deviation"] = train_df["rating"] - train_df["prod_avg"]

    # User's average deviation across all products
    user_dev = train_df.groupby("user_id")["user_prod_deviation"].mean().reset_index()
    user_dev.columns = ["user_id", "user_avg_deviation"]

    print(f"    Interaction stats computed in {time.time()-t0:.1f}s")
    return user_dev


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_total = time.time()

    # Load data
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    prodinfo_df = pd.read_csv(os.path.join(DATA_DIR, "prodInfo.csv"))
    print(f"  Train: {len(train_df):,} rows")
    print(f"  ProdInfo: {len(prodinfo_df):,} rows")

    # Compute stats
    print("\n" + "=" * 60)
    print("Computing statistics...")
    print("=" * 60)

    user_stats = compute_user_stats(train_df)
    prod_stats = compute_product_stats(train_df, prodinfo_df)
    cat_stats = compute_category_stats(train_df, prodinfo_df)
    user_dev = compute_user_product_interaction_stats(train_df)

    # Merge user deviation into user_stats
    user_stats = user_stats.merge(user_dev, on="user_id", how="left")
    user_stats["user_avg_deviation"] = user_stats["user_avg_deviation"].fillna(0.0)

    # Save
    print("\n" + "=" * 60)
    print("Saving...")
    print("=" * 60)

    user_path = os.path.join(OUT_DIR, "user_stats_pandas.parquet")
    prod_path = os.path.join(OUT_DIR, "product_stats_pandas.parquet")
    cat_path = os.path.join(OUT_DIR, "category_stats_pandas.parquet")

    user_stats.to_parquet(user_path, index=False)
    prod_stats.to_parquet(prod_path, index=False)
    cat_stats.to_parquet(cat_path, index=False)

    print(f"  user_stats: {len(user_stats):,} rows -> {user_path}")
    print(f"  product_stats: {len(prod_stats):,} rows -> {prod_path}")
    print(f"  category_stats: {len(cat_stats):,} rows -> {cat_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  User features: {list(user_stats.columns)}")
    print(f"  Product features: {list(prod_stats.columns)}")
    print(f"  Category features: {list(cat_stats.columns)}")
    print(f"\n  Total time: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()

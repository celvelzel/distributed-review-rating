"""
Phase 1: Expanded graph features (store metadata + rating deviation + user-category deviation).

Computes from CSV directly (no PySpark/parquet needed):
1. Store metadata: store_product_count, store_avg_rating_number, store_total_rating_number, store_has_name
2. Rating deviation (K-Fold OOF): user_rating_dev, prod_rating_dev, cat_rating_dev, user_leniency, user_harshness
3. User-category deviation: user_cat_avg_rating, user_cat_review_count, user_cat_deviation

Output: artifacts/features/expanded_graph_features.parquet
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "features")


def compute_store_features(prodinfo_df):
    """Extract store-level aggregated features."""
    print("  [store] Computing store features...")
    t0 = time.time()

    df = prodinfo_df[["parent_prod_id", "store", "rating_number"]].copy()
    df["rating_number"] = df["rating_number"].fillna(0)

    store_agg = df.groupby("store").agg(
        store_product_count=("parent_prod_id", "count"),
        store_avg_rating_number=("rating_number", "mean"),
        store_total_rating_number=("rating_number", "sum"),
    ).reset_index()

    result = df[["parent_prod_id", "store"]].merge(store_agg, on="store", how="left")
    result["store_has_name"] = result["store"].notna().astype(np.int8)
    result = result.drop(columns=["store"])

    for col in ["store_product_count", "store_avg_rating_number", "store_total_rating_number"]:
        result[col] = result[col].fillna(0).astype(np.float32)

    print(f"    Done: {result.shape} in {time.time()-t0:.1f}s")
    return result


def compute_rating_deviation_kfold(train_df, prodinfo_df, n_splits=5):
    """Compute rating deviation features with K-Fold OOF (no leakage)."""
    print("  [deviation] Computing rating deviation (K-Fold OOF)...")
    t0 = time.time()

    # Use main_category if already present, otherwise merge
    if "main_category" not in train_df.columns:
        prod_cat = prodinfo_df[["parent_prod_id", "main_category"]].drop_duplicates("parent_prod_id")
        train_with_cat = train_df.merge(prod_cat, on="parent_prod_id", how="left")
        train_with_cat["main_category"] = train_with_cat["main_category"].fillna("Unknown")
    else:
        train_with_cat = train_df.copy()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_devs = []

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(train_df)):
        other = train_with_cat.iloc[tr_idx]
        val = train_with_cat.iloc[va_idx]

        # Stats from OTHER folds only
        user_stats = other.groupby("user_id").agg(
            user_avg_rating_oof=("rating", "mean"),
            user_num_reviews_oof=("rating", "count"),
        )
        prod_stats = other.groupby("parent_prod_id").agg(
            prod_avg_rating_oof=("rating", "mean"),
        )
        cat_stats = other.groupby("main_category").agg(
            cat_avg_rating_oof=("rating", "mean"),
        )
        global_avg = other["rating"].mean()

        # Merge to validation
        val_dev = val[["id", "user_id", "parent_prod_id", "main_category", "rating"]].copy()

        val_dev = val_dev.merge(user_stats, on="user_id", how="left")
        val_dev["user_avg_rating_oof"] = val_dev["user_avg_rating_oof"].fillna(global_avg)
        val_dev["user_num_reviews_oof"] = val_dev["user_num_reviews_oof"].fillna(0)

        val_dev = val_dev.merge(prod_stats, on="parent_prod_id", how="left")
        val_dev["prod_avg_rating_oof"] = val_dev["prod_avg_rating_oof"].fillna(global_avg)

        val_dev = val_dev.merge(cat_stats, on="main_category", how="left")
        val_dev["cat_avg_rating_oof"] = val_dev["cat_avg_rating_oof"].fillna(global_avg)

        # Deviation features — ONLY use stats from OTHER folds, NOT the actual rating
        # user_leniency: how much user deviates from global average (no rating needed)
        val_dev["user_leniency"] = val_dev["user_avg_rating_oof"] - global_avg
        val_dev["user_harshness"] = val_dev["user_leniency"].abs()

        # user_num_reviews: number of reviews (already computed, useful as-is)
        # Do NOT compute: user_rating_dev, prod_rating_dev, cat_rating_dev
        # (these require the actual rating and would leak the target)

        oof_devs.append(val_dev[["id", "user_leniency", "user_harshness", "user_num_reviews_oof"]])

    result = pd.concat(oof_devs, ignore_index=True).sort_values("id").reset_index(drop=True)
    print(f"    Done: {result.shape} in {time.time()-t0:.1f}s")
    return result


def compute_rating_deviation_test(train_df, test_df, prodinfo_df):
    """Compute deviation features for test set using full training stats."""
    print("  [deviation] Computing test set deviation features...")
    t0 = time.time()

    train_with_cat = train_df.copy()
    test_with_cat = test_df.copy()

    # Full train stats
    user_stats = train_with_cat.groupby("user_id").agg(
        user_avg_rating_oof=("rating", "mean"),
        user_num_reviews_oof=("rating", "count"),
    )
    prod_stats = train_with_cat.groupby("parent_prod_id").agg(
        prod_avg_rating_oof=("rating", "mean"),
    )
    cat_stats = train_with_cat.groupby("main_category").agg(
        cat_avg_rating_oof=("rating", "mean"),
    )
    global_avg = train_with_cat["rating"].mean()

    # Merge to test
    test_dev = test_df[["id", "user_id", "parent_prod_id", "main_category"]].copy()

    test_dev = test_dev.merge(user_stats, on="user_id", how="left")
    test_dev["user_avg_rating_oof"] = test_dev["user_avg_rating_oof"].fillna(global_avg)
    test_dev["user_num_reviews_oof"] = test_dev["user_num_reviews_oof"].fillna(0)

    test_dev = test_dev.merge(prod_stats, on="parent_prod_id", how="left")
    test_dev["prod_avg_rating_oof"] = test_dev["prod_avg_rating_oof"].fillna(global_avg)

    test_dev = test_dev.merge(cat_stats, on="main_category", how="left")
    test_dev["cat_avg_rating_oof"] = test_dev["cat_avg_rating_oof"].fillna(global_avg)

    # Deviation (no actual rating for test — use relative stats only)
    test_dev["user_leniency"] = test_dev["user_avg_rating_oof"] - global_avg
    test_dev["user_harshness"] = test_dev["user_leniency"].abs()
    test_dev["user_num_reviews_oof"] = test_dev["user_num_reviews_oof"].fillna(0)

    print(f"    Done: {test_dev.shape} in {time.time()-t0:.1f}s")
    return test_dev[["id", "user_leniency", "user_harshness", "user_num_reviews_oof"]]


def compute_user_category_features(train_df, test_df, prodinfo_df):
    """Compute user-category interaction features."""
    print("  [user-cat] Computing user-category features...")
    t0 = time.time()

    prod_cat = prodinfo_df[["parent_prod_id", "main_category"]].drop_duplicates("parent_prod_id")

    # Train features
    train_merged = train_df.copy()
    if "main_category" not in train_merged.columns:
        train_merged = train_merged.merge(prod_cat, on="parent_prod_id", how="left")
        train_merged["main_category"] = train_merged["main_category"].fillna("Unknown")
    user_cat_stats = train_merged.groupby(["user_id", "main_category"]).agg(
        user_cat_avg_rating=("rating", "mean"),
        user_cat_review_count=("rating", "count"),
    ).reset_index()

    # Global user stats for comparison
    user_global = train_merged.groupby("user_id").agg(
        user_global_avg=("rating", "mean"),
    ).reset_index()

    user_cat_stats = user_cat_stats.merge(user_global, on="user_id", how="left")
    user_cat_stats["user_cat_deviation"] = user_cat_stats["user_cat_avg_rating"] - user_cat_stats["user_global_avg"]

    # Map to train
    train_key = train_df[["id", "user_id", "parent_prod_id"]].copy()
    if "main_category" not in train_key.columns:
        train_key = train_key.merge(prod_cat, on="parent_prod_id", how="left")
        train_key["main_category"] = train_key["main_category"].fillna("Unknown")
    else:
        train_key["main_category"] = train_df["main_category"].values
    train_feats = train_key.merge(user_cat_stats, on=["user_id", "main_category"], how="left")
    train_feats = train_feats.reindex(train_df["id"].values)
    train_feats.index = train_df["id"].values
    train_feats = train_feats.sort_index()

    train_result = pd.DataFrame({
        "id": train_df["id"].values,
        "user_cat_avg_rating": train_feats["user_cat_avg_rating"].fillna(0).values,
        "user_cat_review_count": train_feats["user_cat_review_count"].fillna(0).values,
        "user_cat_deviation": train_feats["user_cat_deviation"].fillna(0).values,
    })

    # Test features
    test_key = test_df[["id", "user_id", "parent_prod_id"]].copy()
    if "main_category" not in test_key.columns:
        test_key = test_key.merge(prod_cat, on="parent_prod_id", how="left")
        test_key["main_category"] = test_key["main_category"].fillna("Unknown")
    else:
        test_key["main_category"] = test_df["main_category"].values
    test_feats = test_key.merge(user_cat_stats, on=["user_id", "main_category"], how="left")

    test_result = pd.DataFrame({
        "id": test_df["id"].values,
        "user_cat_avg_rating": test_feats["user_cat_avg_rating"].fillna(0).values,
        "user_cat_review_count": test_feats["user_cat_review_count"].fillna(0).values,
        "user_cat_deviation": test_feats["user_cat_deviation"].fillna(0).values,
    })

    print(f"    Done: train {train_result.shape}, test {test_result.shape} in {time.time()-t0:.1f}s")
    return train_result, test_result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_total = time.time()

    # 1. Load data
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    prodinfo_df = pd.read_csv(os.path.join(DATA_DIR, "prodInfo.csv"))

    # Add main_category to train/test (fill missing with "Unknown")
    prod_cat = prodinfo_df[["parent_prod_id", "main_category"]].drop_duplicates("parent_prod_id")
    train_df = train_df.merge(prod_cat, on="parent_prod_id", how="left")
    test_df = test_df.merge(prod_cat, on="parent_prod_id", how="left")
    train_df["main_category"] = train_df["main_category"].fillna("Unknown")
    test_df["main_category"] = test_df["main_category"].fillna("Unknown")

    print(f"  Train: {len(train_df):,}, Test: {len(test_df):,}, ProdInfo: {len(prodinfo_df):,}")

    # 2. Store features (per product, map to train/test)
    print("\n" + "=" * 60)
    print("STEP 2: Computing features")
    print("=" * 60)

    store_feats = compute_store_features(prodinfo_df)

    # Map store features to train/test
    train_store = train_df[["id", "parent_prod_id"]].merge(store_feats, on="parent_prod_id", how="left")
    train_store = train_store.sort_values("id").reset_index(drop=True)

    test_store = test_df[["id", "parent_prod_id"]].merge(store_feats, on="parent_prod_id", how="left")
    test_store = test_store.sort_values("id").reset_index(drop=True)

    # 3. Rating deviation (K-Fold OOF for train, full for test)
    train_dev = compute_rating_deviation_kfold(train_df, prodinfo_df, n_splits=5)
    test_dev = compute_rating_deviation_test(train_df, test_df, prodinfo_df)

    # 4. User-category features
    train_ucat, test_ucat = compute_user_category_features(train_df, test_df, prodinfo_df)

    # 5. Assemble and save
    print("\n" + "=" * 60)
    print("STEP 3: Assembling and saving")
    print("=" * 60)

    # Train
    train_feats = pd.DataFrame({"id": train_df["id"].values})
    for df, prefix in [(train_store, None), (train_dev, None), (train_ucat, None)]:
        df_cols = [c for c in df.columns if c != "id"]
        for col in df_cols:
            train_feats[col] = df.set_index("id").reindex(train_feats["id"])[col].values

    # Test
    test_feats = pd.DataFrame({"id": test_df["id"].values})
    for df, prefix in [(test_store, None), (test_dev, None), (test_ucat, None)]:
        df_cols = [c for c in df.columns if c != "id"]
        for col in df_cols:
            test_feats[col] = df.set_index("id").reindex(test_feats["id"])[col].values

    # Fill NaN
    train_feats = train_feats.fillna(0)
    test_feats = test_feats.fillna(0)

    # Save
    train_out = os.path.join(OUT_DIR, "expanded_graph_train.parquet")
    test_out = os.path.join(OUT_DIR, "expanded_graph_test.parquet")

    train_feats.to_parquet(train_out, index=False)
    test_feats.to_parquet(test_out, index=False)

    print(f"  Train: {train_feats.shape} -> {train_out}")
    print(f"  Test: {test_feats.shape} -> {test_out}")
    print(f"  Features: {[c for c in train_feats.columns if c not in ['id', 'parent_prod_id']]}")

    print(f"\nDone in {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()

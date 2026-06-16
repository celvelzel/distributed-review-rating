"""
Full graph-enhanced prediction pipeline: stats + LightGCN embeddings.

Generates submissions blending:
1. Ridge on user/product/category stats (OOF ~0.81)
2. Ridge on LightGCN embeddings (captures collaborative filtering)
3. Combined: stats + GCN embeddings
4. Various blend ratios with existing best submission
"""

import os
import sys
import time
import gc

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

def mem_mb():
    """Current process RSS in MB."""
    import psutil
    return psutil.Process().memory_info().rss / 1024**2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FEAT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "features")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def load_gcn_embeddings():
    """Load LightGCN embeddings and mappings."""
    user_emb = np.load(os.path.join(FEAT_DIR, "user_emb_gcn.npy"), mmap_mode="r")
    item_emb = np.load(os.path.join(FEAT_DIR, "item_emb_gcn.npy"), mmap_mode="r")

    import json
    with open(os.path.join(FEAT_DIR, "user2idx_gcn.json")) as f:
        user2idx = json.load(f)
    with open(os.path.join(FEAT_DIR, "item2idx_gcn.json")) as f:
        item2idx = json.load(f)

    return user_emb, item_emb, user2idx, item2idx


def compute_gcn_features(df, user_emb, item_emb, user2idx, item2idx):
    """Compute GCN-based features for each sample: user_emb + item_emb + element-wise product."""
    n = len(df)
    emb_dim = user_emb.shape[1]

    uidx = df["user_id"].map(user2idx).fillna(-1).astype(int).values
    iidx = df["parent_prod_id"].map(item2idx).fillna(-1).astype(int).values

    u_emb = np.zeros((n, emb_dim), dtype=np.float32)
    i_emb = np.zeros((n, emb_dim), dtype=np.float32)

    mask_u = uidx >= 0
    mask_i = iidx >= 0
    u_emb[mask_u] = np.array(user_emb[uidx[mask_u]])
    i_emb[mask_i] = np.array(item_emb[iidx[mask_i]])

    # Features: user_emb, item_emb, dot product (compact: 129d instead of 193d)
    dot_product = np.sum(u_emb * i_emb, axis=1, keepdims=True)

    features = np.hstack([u_emb, i_emb, dot_product])
    return features


def compute_stat_features(df, user_stats, prod_stats, cat_stats):
    """Compute statistical features."""
    features = {}

    us = user_stats.set_index("user_id")
    for col in ["user_avg_rating", "user_review_count", "user_rating_std", "user_avg_deviation"]:
        if col in us.columns:
            features[col] = df["user_id"].map(us[col]).fillna(0).values

    ps = prod_stats.set_index("parent_prod_id")
    for col in ["prod_avg_rating", "prod_review_count", "prod_rating_std"]:
        if col in ps.columns:
            features[col] = df["parent_prod_id"].map(ps[col]).fillna(0).values

    if "main_category" in prod_stats.columns:
        cs = cat_stats.set_index("main_category")
        for col in ["cat_avg_rating", "cat_review_count"]:
            if col in cs.columns:
                prod_cat = prod_stats.set_index("parent_prod_id")["main_category"]
                cats = df["parent_prod_id"].map(prod_cat)
                features[col] = cats.map(cs[col]).fillna(0).values

    if "user_avg_rating" in features and "prod_avg_rating" in features:
        features["user_prod_avg_diff"] = features["user_avg_rating"] - features["prod_avg_rating"]
        features["user_prod_avg_blend"] = (features["user_avg_rating"] + features["prod_avg_rating"]) / 2

    return pd.DataFrame(features)


def train_ridge_oof(X, y, X_test, alpha=1.0, n_splits=5):
    """Train Ridge with K-Fold OOF, return oof_preds and test_preds."""
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_total = time.time()

    # 1. Load data
    print(f"[mem] start: {mem_mb():.0f} MB")
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    y = train_df["rating"].values.astype(np.float32)
    print(f"  Train: {len(train_df):,}, Test: {len(test_df):,}")
    print(f"[mem] data loaded: {mem_mb():.0f} MB")

    # 2. Load features
    print("\n" + "=" * 60)
    print("STEP 2: Loading features")
    print("=" * 60)
    user_stats = pd.read_parquet(os.path.join(FEAT_DIR, "user_stats_pandas.parquet"))
    prod_stats = pd.read_parquet(os.path.join(FEAT_DIR, "product_stats_pandas.parquet"))
    cat_stats = pd.read_parquet(os.path.join(FEAT_DIR, "category_stats_pandas.parquet"))
    user_emb, item_emb, user2idx, item2idx = load_gcn_embeddings()
    print(f"  Stats loaded, GCN embeddings: user {user_emb.shape}, item {item_emb.shape}")
    print(f"[mem] features loaded: {mem_mb():.0f} MB")

    # 3. Compute features
    print("\n" + "=" * 60)
    print("STEP 3: Computing features")
    print("=" * 60)

    t0 = time.time()
    print("  Stats features (train)...")
    stat_train = compute_stat_features(train_df, user_stats, prod_stats, cat_stats)
    print("  Stats features (test)...")
    stat_test = compute_stat_features(test_df, user_stats, prod_stats, cat_stats)
    print(f"  Stats: {time.time()-t0:.1f}s")

    t0 = time.time()
    print("  GCN features (train)...")
    gcn_train = compute_gcn_features(train_df, user_emb, item_emb, user2idx, item2idx)
    print("  GCN features (test)...")
    gcn_test = compute_gcn_features(test_df, user_emb, item_emb, user2idx, item2idx)
    print(f"  GCN: {time.time()-t0:.1f}s")

    # Free embeddings immediately
    del user_emb, item_emb, user2idx, item2idx
    gc.collect()

    # 4. Train models
    print("\n" + "=" * 60)
    print("STEP 4: Training Ridge models")
    print("=" * 60)

    # Model A: Stats only
    print("\n  [A] Stats only:")
    stat_train_np = np.nan_to_num(stat_train.values, 0).astype(np.float32)
    stat_test_np = np.nan_to_num(stat_test.values, 0).astype(np.float32)
    del stat_train, stat_test
    gc.collect()

    _, stats_test_pred, stats_rmse = train_ridge_oof(stat_train_np, y, stat_test_np, alpha=1.0)
    print(f"    OOF RMSE: {stats_rmse:.4f}")

    # Model B: GCN only
    print("\n  [B] GCN embeddings only:")
    _, gcn_test_pred, gcn_rmse = train_ridge_oof(gcn_train, y, gcn_test, alpha=10.0)
    print(f"    OOF RMSE: {gcn_rmse:.4f}")

    # Model C: Stats + GCN (free intermediates)
    print("\n  [C] Stats + GCN combined:")
    combined_train = np.hstack([stat_train_np, gcn_train]).astype(np.float32)
    combined_test = np.hstack([stat_test_np, gcn_test]).astype(np.float32)
    del stat_train_np, stat_test_np, gcn_train, gcn_test
    gc.collect()

    _, combined_test_pred, combined_rmse = train_ridge_oof(combined_train, y, combined_test, alpha=10.0)
    del combined_train, combined_test
    gc.collect()
    print(f"    OOF RMSE: {combined_rmse:.4f}")

    # 5. Generate submissions
    print("\n" + "=" * 60)
    print("STEP 5: Generating submissions")
    print("=" * 60)

    # Load existing best submission
    best_path = os.path.join(OUTPUT_DIR, "submission-final.csv")
    if os.path.exists(best_path):
        best_df = pd.read_csv(best_path)
        best_preds = best_df["rating"].values
        print(f"  Best submission: mean={best_preds.mean():.4f}, std={best_preds.std():.4f}")
    else:
        best_preds = None

    # Save standalone predictions
    for name, preds in [("ridge_stats_only", stats_test_pred),
                         ("ridge_gcn_only", gcn_test_pred),
                         ("ridge_stats_gcn_combined", combined_test_pred)]:
        preds_clipped = np.clip(preds, 1.0, 5.0)
        sub = pd.DataFrame({"id": test_df["id"].values, "rating": preds_clipped})
        sub.to_csv(os.path.join(OUTPUT_DIR, f"{name}.csv"), index=False)
        print(f"  {name}: mean={preds_clipped.mean():.4f}, std={preds_clipped.std():.4f}")

    # Blend with best submission
    if best_preds is not None:
        print("\n  Blends with best submission:")
        for name, preds in [("gcn", gcn_test_pred), ("stats_gcn", combined_test_pred)]:
            for w in [0.05, 0.10, 0.15, 0.20]:
                blend = best_preds * (1 - w) + preds * w
                blend = np.clip(blend, 1.0, 5.0)
                sub = pd.DataFrame({"id": test_df["id"].values, "rating": blend})
                fname = f"best_{int((1-w)*100)}_{name}_{int(w*100)}.csv"
                sub.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
                print(f"    {fname}: mean={blend.mean():.4f}, std={blend.std():.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Stats OOF RMSE:    {stats_rmse:.4f}")
    print(f"  GCN OOF RMSE:      {gcn_rmse:.4f}")
    print(f"  Combined OOF RMSE: {combined_rmse:.4f}")
    print(f"  Total time: {time.time()-t_total:.1f}s")
    print(f"\n  Recommend: submit 'best_90_gcn_10' or 'best_85_stats_gcn_15' to Kaggle")


if __name__ == "__main__":
    main()

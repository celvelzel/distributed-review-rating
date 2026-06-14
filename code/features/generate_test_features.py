#!/usr/bin/env python
"""Generate missing test features: SVD 512, sentiment, text stats, safe TE."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEAT_DIR = ROOT / "artifacts" / "features"
ETL_DIR = ROOT / "artifacts" / "etl"


def generate_svd_test():
    """Generate SVD 512 test features by fitting SVD on train TF-IDF and transforming test."""
    print("[1/4] Generating SVD 512 test features...")
    t0 = time.perf_counter()

    train_path = FEAT_DIR / "tfidf_50k_train.npz"
    test_path = FEAT_DIR / "tfidf_50k_test.npz"
    out_path = FEAT_DIR / "svd_512_test.npz"

    if out_path.exists():
        print(f"  Already exists: {out_path}")
        return

    print("  Loading TF-IDF 50K train...")
    X_train = sparse.load_npz(str(train_path))
    print(f"  Train: {X_train.shape}")

    print("  Fitting TruncatedSVD (n_components=512, randomized)...")
    svd = TruncatedSVD(n_components=512, random_state=42, algorithm="randomized")
    svd.fit(X_train)
    print(f"  Cumulative EVR: {svd.explained_variance_ratio_.sum():.4f}")

    print("  Loading TF-IDF 50K test...")
    X_test = sparse.load_npz(str(test_path))
    print(f"  Test: {X_test.shape}")

    print("  Transforming test...")
    X_test_svd = svd.transform(X_test).astype(np.float32)
    sparse.save_npz(str(out_path), sparse.csr_matrix(X_test_svd))
    print(f"  Saved: {out_path} shape={X_test_svd.shape}")
    print(f"  Time: {time.perf_counter() - t0:.1f}s")


def generate_sentiment_test():
    """Generate sentiment features for test data."""
    print("\n[2/4] Generating sentiment test features...")
    t0 = time.perf_counter()

    out_path = FEAT_DIR / "sentiment_test.parquet"
    if out_path.exists():
        print(f"  Already exists: {out_path}")
        return

    test_df = pd.read_parquet(ETL_DIR / "test.parquet", columns=["title", "comment"])
    print(f"  Test: {len(test_df)} rows")

    # Import sentiment functions
    sys.path.insert(0, str(ROOT / "code" / "features"))
    from sentiment import _get_vader

    analyzer = _get_vader()

    def compute_sentiment(row):
        title = str(row.get("title", "") or "")
        comment = str(row.get("comment", "") or "")
        text = title + " " + comment

        vs = analyzer.polarity_scores(text)
        return {
            "vader_pos": vs["pos"],
            "vader_neg": vs["neg"],
            "vader_neu": vs["neu"],
            "vader_compound": vs["compound"],
        }

    print("  Computing VADER sentiment...")
    sentiments = test_df.apply(compute_sentiment, axis=1, result_type="expand")
    sentiments.to_parquet(str(out_path))
    print(f"  Saved: {out_path} shape={sentiments.shape}")
    print(f"  Time: {time.perf_counter() - t0:.1f}s")


def generate_text_stats_test():
    """Generate text statistics for test data."""
    print("\n[3/4] Generating text stats test features...")
    t0 = time.perf_counter()

    out_path = FEAT_DIR / "text_stats_test.npz"
    if out_path.exists():
        print(f"  Already exists: {out_path}")
        return

    test_df = pd.read_parquet(ETL_DIR / "test.parquet", columns=["title", "comment", "votes", "purchased"])
    print(f"  Test: {len(test_df)} rows")

    title = test_df["title"].fillna("").astype(str)
    comment = test_df["comment"].fillna("").astype(str)

    features = np.column_stack([
        title.str.len().values.astype(np.float32),
        comment.str.len().values.astype(np.float32),
        (title.str.split().str.len().fillna(0)).values.astype(np.float32),
        (comment.str.split().str.len().fillna(0)).values.astype(np.float32),
        test_df["votes"].fillna(0).values.astype(np.float32),
        test_df["purchased"].fillna(0).values.astype(np.float32),
    ])

    sparse.save_npz(str(out_path), sparse.csr_matrix(features))
    print(f"  Saved: {out_path} shape={features.shape}")
    print(f"  Time: {time.perf_counter() - t0:.1f}s")


def generate_safe_te_test():
    """Generate safe target encoding for test data using precomputed stats."""
    print("\n[4/4] Generating safe TE test features...")
    t0 = time.perf_counter()

    out_path = FEAT_DIR / "safe_target_encoding_test.npz"
    if out_path.exists():
        print(f"  Already exists: {out_path}")
        return

    # Load training data to compute user/product/category stats
    train_df = pd.read_parquet(ETL_DIR / "train.parquet", columns=["user_id", "parent_prod_id", "category", "rating"])
    test_df = pd.read_parquet(ETL_DIR / "test.parquet", columns=["user_id", "parent_prod_id", "category"])

    global_mean = train_df["rating"].mean()
    K = 5
    smoothing = 10.0

    def safe_te(group_col):
        stats = train_df.groupby(group_col)["rating"].agg(["mean", "count"]).reset_index()
        stats.columns = [group_col, "mean", "count"]
        stats["te"] = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        return stats.set_index(group_col)["te"]

    user_te = safe_te("user_id")
    prod_te = safe_te("parent_prod_id")
    cat_te = safe_te("category")

    user_count = train_df.groupby("user_id").size()
    prod_count = train_df.groupby("parent_prod_id").size()

    test_user_te = test_df["user_id"].map(user_te).fillna(global_mean).values.astype(np.float32)
    test_prod_te = test_df["parent_prod_id"].map(prod_te).fillna(global_mean).values.astype(np.float32)
    test_cat_te = test_df["category"].map(cat_te).fillna(global_mean).values.astype(np.float32)
    test_user_count = test_df["user_id"].map(user_count).fillna(0).values.astype(np.float32)
    test_prod_count = test_df["parent_prod_id"].map(prod_count).fillna(0).values.astype(np.float32)

    features = np.column_stack([test_user_te, test_prod_te, test_cat_te, test_user_count, test_prod_count])
    sparse.save_npz(str(out_path), sparse.csr_matrix(features))
    print(f"  Saved: {out_path} shape={features.shape}")
    print(f"  Time: {time.perf_counter() - t0:.1f}s")


def main():
    print("=" * 60)
    print("Generating missing test features")
    print("=" * 60)

    generate_svd_test()
    generate_sentiment_test()
    generate_text_stats_test()
    generate_safe_te_test()

    print("\n=== All test features generated ===")


if __name__ == "__main__":
    main()

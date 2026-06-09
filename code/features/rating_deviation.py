"""
Rating deviation features with K-Fold cross-validation to eliminate target leakage.

For each training row, deviation stats are computed from OTHER folds only (not including
the current row's fold). This prevents the target variable from leaking into features.

For test rows, stats use the FULL training set (no leakage since test has no rating).

Output: artifacts/features/rating_deviation.parquet
  Columns: id, user_rating_dev, prod_rating_dev, cat_rating_dev,
           user_leniency, user_harshness, user_rating_dev_abs
"""

from __future__ import annotations

import os
import sys

# Ensure PySpark workers use the same Python interpreter.
os.environ.setdefault("PYSPARK_PYTHON", "/usr/bin/python3.8")
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", "/usr/bin/python3.8")

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"
OUTPUT_PATH = ARTIFACTS_DIR / "features" / "rating_deviation.parquet"


def _compute_user_stats(train_pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute user-level mean rating and count.

    Parameters
    ----------
    train_pdf : DataFrame with columns: user_id, rating

    Returns
    -------
    DataFrame indexed by user_id with: user_avg_rating, user_num_reviews
    """
    stats = train_pdf.groupby("user_id").agg(
        user_avg_rating=("rating", "mean"),
        user_num_reviews=("rating", "count"),
    )
    return stats


def _compute_product_stats(train_pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute product-level mean rating.

    Parameters
    ----------
    train_pdf : DataFrame with columns: parent_prod_id, rating

    Returns
    -------
    DataFrame indexed by parent_prod_id with: prod_avg_rating
    """
    stats = train_pdf.groupby("parent_prod_id").agg(
        prod_avg_rating=("rating", "mean"),
    )
    return stats


def _compute_category_stats(
    train_pdf: pd.DataFrame, prodinfo_pdf: pd.DataFrame
) -> pd.DataFrame:
    """Compute category-level mean rating.

    Parameters
    ----------
    train_pdf : DataFrame with columns: parent_prod_id, rating
    prodinfo_pdf : DataFrame with columns: parent_prod_id, main_category

    Returns
    -------
    DataFrame indexed by main_category with: cat_avg_rating
    """
    prod_cat = prodinfo_pdf[["parent_prod_id", "main_category"]].drop_duplicates(
        subset=["parent_prod_id"]
    )
    train_with_cat = train_pdf[["parent_prod_id", "rating"]].merge(
        prod_cat, on="parent_prod_id", how="left"
    )
    stats = train_with_cat.groupby("main_category").agg(
        cat_avg_rating=("rating", "mean"),
    )
    return stats


def _compute_deviation_features(
    row_df: pd.DataFrame,
    user_stats: pd.DataFrame,
    prod_stats: pd.DataFrame,
    cat_stats: pd.DataFrame,
    global_user_avg: float,
    global_prod_avg: float,
    global_cat_avg: float,
) -> pd.DataFrame:
    """Compute deviation features for a set of rows.

    Parameters
    ----------
    row_df : DataFrame with columns: id, user_id, parent_prod_id, main_category
    user_stats : DataFrame indexed by user_id
    prod_stats : DataFrame indexed by parent_prod_id
    cat_stats : DataFrame indexed by main_category
    global_*_avg : Global averages for fallback

    Returns
    -------
    DataFrame with deviation columns
    """
    result = row_df[["id"]].copy()

    # User deviation: rating - user_avg_rating (computed from other folds)
    user_merged = row_df[["id", "user_id"]].merge(
        user_stats, left_on="user_id", right_index=True, how="left"
    )
    result["user_avg_rating_for_dev"] = user_merged["user_avg_rating"].fillna(global_user_avg)
    result["user_num_reviews"] = user_merged["user_num_reviews"].fillna(0)

    # Product deviation: rating - prod_avg_rating (computed from other folds)
    prod_merged = row_df[["id", "parent_prod_id"]].merge(
        prod_stats, left_on="parent_prod_id", right_index=True, how="left"
    )
    result["prod_avg_rating_for_dev"] = prod_merged["prod_avg_rating"].fillna(global_prod_avg)

    # Category deviation: rating - cat_avg_rating (computed from other folds)
    if "main_category" in row_df.columns:
        cat_merged = row_df[["id", "main_category"]].merge(
            cat_stats, left_on="main_category", right_index=True, how="left"
        )
        result["cat_avg_rating_for_dev"] = cat_merged["cat_avg_rating"].fillna(global_cat_avg)
    else:
        result["cat_avg_rating_for_dev"] = global_cat_avg

    return result


def compute_rating_deviation_kfold(
    train_pdf: pd.DataFrame,
    prodinfo_pdf: pd.DataFrame,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Compute per-row rating deviation features using K-Fold to avoid target leakage.

    For each fold, compute user/product/category stats on the OTHER folds, then
    compute deviation features for the current fold's rows.

    Parameters
    ----------
    train_pdf : DataFrame with columns: id, user_id, parent_prod_id, rating
    prodinfo_pdf : DataFrame with columns: parent_prod_id, main_category
    n_splits : Number of K-Fold splits

    Returns
    -------
    DataFrame with columns: id, user_rating_dev, prod_rating_dev, cat_rating_dev,
        user_leniency, user_harshness, user_rating_dev_abs
    """
    log.info(f"Computing rating deviation with {n_splits}-Fold CV on {len(train_pdf)} rows")

    # Pre-join main_category to train for efficient splitting
    prod_cat = prodinfo_pdf[["parent_prod_id", "main_category"]].drop_duplicates(
        subset=["parent_prod_id"]
    )
    train_with_cat = train_pdf.merge(prod_cat, on="parent_prod_id", how="left")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    t0 = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_pdf)):
        fold_t0 = time.time()
        other_df = train_with_cat.iloc[train_idx]
        val_df = train_with_cat.iloc[val_idx]

        # Compute stats on other folds only (NO LEAKAGE)
        user_stats = _compute_user_stats(other_df)
        prod_stats = _compute_product_stats(other_df)
        cat_stats = _compute_category_stats(other_df, prodinfo_pdf)

        # Global averages from other folds
        global_user_avg = other_df["rating"].mean()
        global_prod_avg = other_df["rating"].mean()
        global_cat_avg = other_df["rating"].mean()

        # Compute deviation features for validation rows
        val_features = _compute_deviation_features(
            val_df, user_stats, prod_stats, cat_stats,
            global_user_avg, global_prod_avg, global_cat_avg
        )

        # Compute actual deviations using the validation row's rating
        val_ratings = val_df[["id", "rating"]].copy()
        val_features = val_features.merge(val_ratings, on="id", how="left")

        # User deviation: rating - user_avg_rating (from other folds)
        val_features["user_rating_dev"] = (
            val_features["rating"] - val_features["user_avg_rating_for_dev"]
        )
        val_features["user_rating_dev_abs"] = val_features["user_rating_dev"].abs()

        # Product deviation: rating - prod_avg_rating (from other folds)
        val_features["prod_rating_dev"] = (
            val_features["rating"] - val_features["prod_avg_rating_for_dev"]
        )

        # Category deviation: rating - cat_avg_rating (from other folds)
        val_features["cat_rating_dev"] = (
            val_features["rating"] - val_features["cat_avg_rating_for_dev"]
        )

        # User leniency/harshness: how much user deviates from global mean
        # Positive = lenient (rates higher than average), Negative = harsh
        val_features["user_leniency"] = (
            val_features["user_avg_rating_for_dev"] - global_user_avg
        )

        # User harshness: absolute deviation from global mean
        val_features["user_harshness"] = val_features["user_leniency"].abs()

        # Select output columns
        output_cols = [
            "id", "user_rating_dev", "prod_rating_dev", "cat_rating_dev",
            "user_leniency", "user_harshness", "user_rating_dev_abs"
        ]
        fold_results.append(val_features[output_cols])

        log.info(
            f"  Fold {fold_idx}: val_rows={len(val_idx)}, "
            f"other_rows={len(train_idx)}, "
            f"time={time.time() - fold_t0:.1f}s"
        )

    # Combine all fold results
    oof_stats = pd.concat(fold_results, ignore_index=True)
    oof_stats = oof_stats.sort_values("id").reset_index(drop=True)

    log.info(
        f"  Rating deviation K-Fold done in {time.time() - t0:.1f}s. "
        f"OOF rows: {len(oof_stats)}"
    )
    return oof_stats


def compute_rating_deviation_full(
    train_pdf: pd.DataFrame, prodinfo_pdf: pd.DataFrame
) -> tuple:
    """Compute deviation features on the FULL training set (for test set mapping).

    Parameters
    ----------
    train_pdf : DataFrame with columns: id, user_id, parent_prod_id, rating
    prodinfo_pdf : DataFrame with columns: parent_prod_id, main_category

    Returns
    -------
    user_stats : DataFrame indexed by user_id
    prod_stats : DataFrame indexed by parent_prod_id
    cat_stats : DataFrame indexed by main_category
    global_user_avg : float
    global_prod_avg : float
    global_cat_avg : float
    """
    prod_cat = prodinfo_pdf[["parent_prod_id", "main_category"]].drop_duplicates(
        subset=["parent_prod_id"]
    )
    train_with_cat = train_pdf.merge(prod_cat, on="parent_prod_id", how="left")

    user_stats = _compute_user_stats(train_with_cat)
    prod_stats = _compute_product_stats(train_with_cat)
    cat_stats = _compute_category_stats(train_with_cat, prodinfo_pdf)

    global_user_avg = train_with_cat["rating"].mean()
    global_prod_avg = train_with_cat["rating"].mean()
    global_cat_avg = train_with_cat["rating"].mean()

    return user_stats, prod_stats, cat_stats, global_user_avg, global_prod_avg, global_cat_avg


def main() -> None:
    t_total = time.time()

    # ── Load data ──────────────────────────────────────────────────────
    train_path = str(ARTIFACTS_DIR / "etl" / "train.parquet")
    test_path = str(ARTIFACTS_DIR / "etl" / "test.parquet")
    prodinfo_path = str(ARTIFACTS_DIR / "etl" / "prodinfo.parquet")

    log.info(f"Loading train from {train_path}")
    train_pdf = pd.read_parquet(train_path)
    log.info(f"  train: {train_pdf.shape}")

    log.info(f"Loading test from {test_path}")
    test_pdf = pd.read_parquet(test_path)
    log.info(f"  test: {test_pdf.shape}")

    log.info(f"Loading prodinfo from {prodinfo_path}")
    prodinfo_pdf = pd.read_parquet(prodinfo_path)
    log.info(f"  prodinfo: {prodinfo_pdf.shape}")

    # ── Compute K-Fold deviation features for train ────────────────────
    oof_dev = compute_rating_deviation_kfold(train_pdf, prodinfo_pdf, n_splits=5)

    # ── Compute full-train stats for test ──────────────────────────────
    log.info("Computing full-train stats for test set")
    user_stats, prod_stats, cat_stats, global_user_avg, global_prod_avg, global_cat_avg = \
        compute_rating_deviation_full(train_pdf, prodinfo_pdf)

    # Pre-join main_category to test
    prod_cat = prodinfo_pdf[["parent_prod_id", "main_category"]].drop_duplicates(
        subset=["parent_prod_id"]
    )
    test_with_cat = test_pdf.merge(prod_cat, on="parent_prod_id", how="left")

    # Compute deviation features for test rows
    log.info("Computing deviation features for test set")
    test_features = _compute_deviation_features(
        test_with_cat, user_stats, prod_stats, cat_stats,
        global_user_avg, global_prod_avg, global_cat_avg
    )

    # For test set, we don't have actual ratings, so deviations are relative to stats
    # We compute user leniency/harshness (no actual rating needed)
    test_features["user_rating_dev"] = 0.0  # No actual rating for test
    test_features["user_rating_dev_abs"] = 0.0
    test_features["prod_rating_dev"] = 0.0
    test_features["cat_rating_dev"] = 0.0
    test_features["user_leniency"] = (
        test_features["user_avg_rating_for_dev"] - global_user_avg
    )
    test_features["user_harshness"] = test_features["user_leniency"].abs()

    # Select output columns
    output_cols = [
        "id", "user_rating_dev", "prod_rating_dev", "cat_rating_dev",
        "user_leniency", "user_harshness", "user_rating_dev_abs"
    ]
    test_dev = test_features[output_cols]

    # ── Combine train OOF + test ───────────────────────────────────────
    all_dev = pd.concat([oof_dev, test_dev], ignore_index=True)
    all_dev = all_dev.sort_values("id").reset_index(drop=True)

    # Fill NaN with 0 (neutral deviation)
    for col in ["user_rating_dev", "prod_rating_dev", "cat_rating_dev",
                "user_leniency", "user_harshness", "user_rating_dev_abs"]:
        all_dev[col] = all_dev[col].fillna(0.0)

    # Ensure correct types
    all_dev["id"] = all_dev["id"].astype(int)

    # ── Save ───────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_dev.to_parquet(str(OUTPUT_PATH), index=False)
    log.info(f"Saved {OUTPUT_PATH} ({len(all_dev)} rows)")

    # ── Verify no leakage: sample check ────────────────────────────────
    log.info("\n=== Leakage Verification ===")
    # Find a user with exactly 2 reviews
    user_counts = train_pdf.groupby("user_id").size()
    two_review_users = user_counts[user_counts == 2].index
    if len(two_review_users) > 0:
        sample_user = two_review_users[0]
        sample_rows = train_pdf[train_pdf["user_id"] == sample_user].sort_values("id").reset_index(drop=True)
        sid1, sid2 = int(sample_rows.iloc[0]["id"]), int(sample_rows.iloc[1]["id"])
        r1, r2 = float(sample_rows.iloc[0]["rating"]), float(sample_rows.iloc[1]["rating"])

        dev1_series = all_dev.loc[all_dev["id"] == sid1, "user_rating_dev"]
        dev2_series = all_dev.loc[all_dev["id"] == sid2, "user_rating_dev"]

        if len(dev1_series) > 0 and len(dev2_series) > 0:
            dev1 = float(dev1_series.values[0])
            dev2 = float(dev2_series.values[0])

            # For a 2-review user, if K-Fold is correct:
            # Row 1's user_avg should be r2 (from other fold), so dev = r1 - r2
            # Row 2's user_avg should be r1 (from other fold), so dev = r2 - r1
            expected_dev1 = r1 - r2
            expected_dev2 = r2 - r1

            log.info(f"  Sample user: {sample_user}")
            log.info(f"    Row id={sid1}: rating={r1}, user_rating_dev={dev1:.4f} (expected {expected_dev1:.4f})")
            log.info(f"    Row id={sid2}: rating={r2}, user_rating_dev={dev2:.4f} (expected {expected_dev2:.4f})")

            if abs(dev1 - expected_dev1) < 1e-6 and abs(dev2 - expected_dev2) < 1e-6:
                log.info("  ✅ NO LEAKAGE: user_rating_dev uses OTHER review's rating for user_avg")
            else:
                log.warning("  ⚠️  Possible leakage detected! Check K-Fold logic.")
        else:
            log.warning(f"  Could not find deviation for ids {sid1}, {sid2} in output")
    else:
        log.info("  No 2-review users found; skipping sample check")

    log.info(f"\n✅ Done in {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()

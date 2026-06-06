"""
User-level statistical features with K-Fold cross-validation to eliminate target leakage.

For each training row, user stats are computed from OTHER folds only (not including
the current row's fold). This prevents the target variable from leaking into features.

For test rows, stats use the FULL training set (no leakage since test has no rating).

Output: artifacts/features/user_stats_kfold.parquet
  Columns: id, avg_rating, num_reviews, avg_votes, purchased_rate, rating_std
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
OUTPUT_PATH = ARTIFACTS_DIR / "features" / "user_stats_kfold.parquet"


def compute_user_stats_kfold(
    train_pdf: pd.DataFrame, n_splits: int = 5
) -> pd.DataFrame:
    """Compute per-row user stats using K-Fold to avoid target leakage.

    For each fold, compute user-level stats on the OTHER folds, then assign
    those stats to the current fold's rows.

    Parameters
    ----------
    train_pdf : DataFrame with columns: id, user_id, rating, votes, purchased
    n_splits : Number of K-Fold splits

    Returns
    -------
    DataFrame with columns: id, avg_rating, num_reviews, avg_votes,
        purchased_rate, rating_std  (one row per train sample)
    """
    log.info(f"Computing user stats with {n_splits}-Fold CV on {len(train_pdf)} rows")

    # Prepare purchased binary column
    train_pdf = train_pdf.copy()
    train_pdf["_purchased_bin"] = (
        train_pdf["purchased"]
        .map({"True": 1.0, "False": 0.0, True: 1.0, False: 0.0})
        .fillna(0.0)
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    t0 = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_pdf)):
        fold_t0 = time.time()
        other_df = train_pdf.iloc[train_idx]
        val_ids = train_pdf.iloc[val_idx][["id", "user_id"]]

        # Compute stats on other folds only (NO LEAKAGE)
        user_stats = other_df.groupby("user_id").agg(
            avg_rating=("rating", "mean"),
            num_reviews=("rating", "count"),
            avg_votes=("votes", "mean"),
            purchased_rate=("_purchased_bin", "mean"),
            rating_std=("rating", "std"),
        )
        user_stats["rating_std"] = user_stats["rating_std"].fillna(0.0)

        # Join stats to validation rows
        val_with_stats = val_ids.merge(
            user_stats, left_on="user_id", right_index=True, how="left"
        )
        fold_results.append(val_with_stats)

        log.info(
            f"  Fold {fold_idx}: val_rows={len(val_idx)}, "
            f"other_rows={len(train_idx)}, "
            f"unique_users_other={other_df['user_id'].nunique()}, "
            f"time={time.time() - fold_t0:.1f}s"
        )

    # Combine all fold results
    oof_stats = pd.concat(fold_results, ignore_index=True)
    oof_stats = oof_stats.sort_values("id").reset_index(drop=True)

    log.info(
        f"  User stats K-Fold done in {time.time() - t0:.1f}s. "
        f"OOF rows: {len(oof_stats)}"
    )
    return oof_stats


def compute_user_stats_full(train_pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute user stats on the FULL training set (for test set mapping).

    Parameters
    ----------
    train_pdf : DataFrame with columns: user_id, rating, votes, purchased

    Returns
    -------
    DataFrame indexed by user_id with: avg_rating, num_reviews, avg_votes,
    purchased_rate, rating_std
    """
    train_pdf = train_pdf.copy()
    train_pdf["_purchased_bin"] = (
        train_pdf["purchased"]
        .map({"True": 1.0, "False": 0.0, True: 1.0, False: 0.0})
        .fillna(0.0)
    )

    stats = train_pdf.groupby("user_id").agg(
        avg_rating=("rating", "mean"),
        num_reviews=("rating", "count"),
        avg_votes=("votes", "mean"),
        purchased_rate=("_purchased_bin", "mean"),
        rating_std=("rating", "std"),
    )
    stats["rating_std"] = stats["rating_std"].fillna(0.0)
    return stats


def main() -> None:
    t_total = time.time()

    # ── Load train data ────────────────────────────────────────────────
    train_path = str(ARTIFACTS_DIR / "etl" / "train.parquet")
    test_path = str(ARTIFACTS_DIR / "etl" / "test.parquet")

    log.info(f"Loading train from {train_path}")
    train_pdf = pd.read_parquet(train_path)
    log.info(f"  train: {train_pdf.shape}")

    log.info(f"Loading test from {test_path}")
    test_pdf = pd.read_parquet(test_path)
    log.info(f"  test: {test_pdf.shape}")

    # ── Compute K-Fold user stats for train ────────────────────────────
    oof_stats = compute_user_stats_kfold(train_pdf, n_splits=5)

    # ── Compute full-train user stats for test ─────────────────────────
    log.info("Computing full-train user stats for test set")
    full_stats = compute_user_stats_full(train_pdf)

    log.info("Mapping test rows to full-train user stats")
    test_user_stats = test_pdf[["id", "user_id"]].merge(
        full_stats, left_on="user_id", right_index=True, how="left"
    )

    # ── Combine train OOF + test ───────────────────────────────────────
    all_stats = pd.concat([oof_stats, test_user_stats], ignore_index=True)
    all_stats = all_stats.sort_values("id").reset_index(drop=True)

    # Fill NaN with global means (for users with no prior reviews)
    global_means = full_stats.mean(numeric_only=True)
    for col in ["avg_rating", "num_reviews", "avg_votes", "purchased_rate", "rating_std"]:
        all_stats[col] = all_stats[col].fillna(global_means[col])

    # Ensure correct types
    all_stats["id"] = all_stats["id"].astype(int)
    all_stats["num_reviews"] = all_stats["num_reviews"].astype(int)

    # Drop user_id (not needed in output)
    all_stats = all_stats.drop(columns=["user_id"])

    # ── Save ───────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_stats.to_parquet(str(OUTPUT_PATH), index=False)
    log.info(f"Saved {OUTPUT_PATH} ({len(all_stats)} rows)")

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

        stat1_series = all_stats.loc[all_stats["id"] == sid1, "avg_rating"]
        stat2_series = all_stats.loc[all_stats["id"] == sid2, "avg_rating"]

        if len(stat1_series) > 0 and len(stat2_series) > 0:
            stat1 = float(stat1_series.values[0])
            stat2 = float(stat2_series.values[0])

            log.info(f"  Sample user: {sample_user}")
            log.info(f"    Row id={sid1}: rating={r1}, avg_rating_feat={stat1:.4f} (should be {r2})")
            log.info(f"    Row id={sid2}: rating={r2}, avg_rating_feat={stat2:.4f} (should be {r1})")

            if abs(stat1 - r2) < 1e-6 and abs(stat2 - r1) < 1e-6:
                log.info("  ✅ NO LEAKAGE: avg_rating uses OTHER review's rating")
            else:
                log.warning("  ⚠️  Possible leakage detected! Check K-Fold logic.")
        else:
            log.warning(f"  Could not find stats for ids {sid1}, {sid2} in output")
    else:
        log.info("  No 2-review users found; skipping sample check")

    log.info(f"\n✅ Done in {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()

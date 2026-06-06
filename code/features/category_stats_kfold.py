"""Category-level statistical features with K-Fold to eliminate target leakage.

For each training row, category stats are computed from OTHER folds only.
For test rows, stats use the FULL training set (no leakage since test has no rating).
"""

from __future__ import annotations

import os
import sys

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
OUTPUT_PATH = ARTIFACTS_DIR / "features" / "category_stats_kfold.parquet"


def _compute_category_aggregates(
    train_pdf: pd.DataFrame, prodinfo_pdf: pd.DataFrame
) -> pd.DataFrame:
    """Compute category-level aggregates.

    Parameters
    ----------
    train_pdf : DataFrame with columns: parent_prod_id, rating
    prodinfo_pdf : DataFrame with columns: parent_prod_id, price, main_category

    Returns
    -------
    DataFrame indexed by main_category with: cat_avg_rating, cat_avg_price,
    cat_rating_std
    """
    # Join train with prodinfo to get main_category per review
    train_with_cat = train_pdf[["parent_prod_id", "rating"]].merge(
        prodinfo_pdf[["parent_prod_id", "main_category"]].drop_duplicates(
            subset=["parent_prod_id"]
        ),
        on="parent_prod_id",
        how="left",
    )

    # Rating stats from training reviews
    rating_stats = train_with_cat.groupby("main_category").agg(
        cat_avg_rating=("rating", "mean"),
        cat_rating_std=("rating", "std"),
    )
    rating_stats["cat_rating_std"] = rating_stats["cat_rating_std"].fillna(0.0)

    # Price stats from product info (distinct products to avoid duplicates)
    price_stats = (
        prodinfo_pdf[["main_category", "price"]]
        .drop_duplicates(subset=["main_category", "price"])
        .groupby("main_category")
        .agg(cat_avg_price=("price", "mean"))
    )

    result = rating_stats.join(price_stats, how="left")
    return result


def compute_category_stats_kfold(
    train_pdf: pd.DataFrame,
    prodinfo_pdf: pd.DataFrame,
    n_splits: int = 5,
) -> tuple:
    """Compute per-row category stats using K-Fold to avoid target leakage.

    For each fold, compute category-level stats on the OTHER folds, then assign
    those stats to the current fold's rows.

    Parameters
    ----------
    train_pdf : DataFrame with columns: id, parent_prod_id, rating
    prodinfo_pdf : DataFrame with columns: parent_prod_id, price, main_category
    n_splits : Number of K-Fold splits

    Returns
    -------
    oof_stats : DataFrame with columns: id, cat_avg_rating, cat_avg_price,
        cat_rating_std  (one row per train sample)
    full_stats : DataFrame indexed by main_category (for test set mapping)
    """
    log.info(f"Computing category stats with {n_splits}-Fold CV on {len(train_pdf)} rows")

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
        other_df = train_pdf.iloc[train_idx]
        val_ids = train_with_cat.iloc[val_idx][["id", "main_category"]]

        # Compute stats on other folds only
        fold_stats = _compute_category_aggregates(other_df, prodinfo_pdf)

        # Join stats to validation rows
        val_with_stats = val_ids.merge(
            fold_stats, left_on="main_category", right_index=True, how="left"
        )
        fold_results.append(val_with_stats)

        log.info(
            f"  Fold {fold_idx}: val_rows={len(val_idx)}, "
            f"other_rows={len(train_idx)}, "
            f"unique_cats_other={other_df.merge(prod_cat, on='parent_prod_id', how='left')['main_category'].nunique()}, "
            f"time={time.time() - fold_t0:.1f}s"
        )

    # Combine all fold results
    oof_stats = pd.concat(fold_results, ignore_index=True)
    oof_stats = oof_stats.sort_values("id").reset_index(drop=True)

    # Compute full train stats (for test set)
    full_stats = _compute_category_aggregates(train_pdf, prodinfo_pdf)

    log.info(
        f"  Category stats K-Fold done in {time.time() - t0:.1f}s. "
        f"OOF rows: {len(oof_stats)}, unique cats full: {len(full_stats)}"
    )
    return oof_stats, full_stats


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

    # ── Compute K-Fold category stats for train ────────────────────────
    oof_stats, full_stats = compute_category_stats_kfold(
        train_pdf, prodinfo_pdf, n_splits=5
    )

    # ── Compute full-train category stats for test ─────────────────────
    log.info("Mapping test rows to full-train category stats")
    prod_cat = prodinfo_pdf[["parent_prod_id", "main_category"]].drop_duplicates(
        subset=["parent_prod_id"]
    )
    test_with_cat = test_pdf[["id", "parent_prod_id"]].merge(
        prod_cat, on="parent_prod_id", how="left"
    )
    test_cat_stats = test_with_cat.merge(
        full_stats, left_on="main_category", right_index=True, how="left"
    )

    # ── Combine train OOF + test ───────────────────────────────────────
    all_stats = pd.concat([oof_stats, test_cat_stats], ignore_index=True)
    all_stats = all_stats.sort_values("id").reset_index(drop=True)

    # Fill NaN with global means
    global_means = _compute_category_aggregates(train_pdf, prodinfo_pdf).mean(numeric_only=True)
    for col in ["cat_avg_rating", "cat_avg_price", "cat_rating_std"]:
        all_stats[col] = all_stats[col].fillna(global_means[col])

    # Ensure correct types
    all_stats["id"] = all_stats["id"].astype(int)

    # Drop main_category if present (not needed in output)
    if "main_category" in all_stats.columns:
        all_stats = all_stats.drop(columns=["main_category"])

    # ── Save ───────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_stats.to_parquet(str(OUTPUT_PATH), index=False)
    log.info(f"Saved {OUTPUT_PATH} ({len(all_stats)} rows)")

    # ── Verify no leakage: sample check ────────────────────────────────
    log.info("\n=== Leakage Verification ===")
    # Find a category with exactly 2 reviews
    train_with_cat = train_pdf.merge(prod_cat, on="parent_prod_id", how="left")
    cat_counts = train_with_cat.groupby("main_category").size()
    two_review_cats = cat_counts[cat_counts == 2].index
    if len(two_review_cats) > 0:
        sample_cat = two_review_cats[0]
        sample_rows = train_with_cat[train_with_cat["main_category"] == sample_cat].sort_values("id").reset_index(drop=True)
        sid1, sid2 = int(sample_rows.iloc[0]["id"]), int(sample_rows.iloc[1]["id"])
        r1, r2 = float(sample_rows.iloc[0]["rating"]), float(sample_rows.iloc[1]["rating"])

        stat1_series = all_stats.loc[all_stats["id"] == sid1, "cat_avg_rating"]
        stat2_series = all_stats.loc[all_stats["id"] == sid2, "cat_avg_rating"]

        if len(stat1_series) > 0 and len(stat2_series) > 0:
            stat1 = float(stat1_series.values[0])
            stat2 = float(stat2_series.values[0])

            log.info(f"  Sample category: {sample_cat}")
            log.info(f"    Row id={sid1}: rating={r1}, cat_avg_rating_feat={stat1:.4f} (should be {r2})")
            log.info(f"    Row id={sid2}: rating={r2}, cat_avg_rating_feat={stat2:.4f} (should be {r1})")

            if abs(stat1 - r2) < 1e-6 and abs(stat2 - r1) < 1e-6:
                log.info("  ✅ NO LEAKAGE: cat_avg_rating uses OTHER review's rating")
            else:
                log.warning("  ⚠️  Possible leakage detected! Check K-Fold logic.")
        else:
            log.warning(f"  Could not find stats for ids {sid1}, {sid2} in output")
    else:
        log.info("  No 2-review categories found; skipping sample check")

    log.info(f"\n✅ Done in {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()

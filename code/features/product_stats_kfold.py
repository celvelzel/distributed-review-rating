"""Product-level statistical features with K-Fold to eliminate target leakage.

For each training row, product stats are computed from OTHER folds only.
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
OUTPUT_PATH = ARTIFACTS_DIR / "features" / "product_stats_kfold.parquet"


def _compute_product_aggregates(
    train_pdf: pd.DataFrame, prodinfo_pdf: pd.DataFrame
) -> pd.DataFrame:
    """Compute product-level aggregates.

    Parameters
    ----------
    train_pdf : DataFrame with columns: parent_prod_id, rating
    prodinfo_pdf : DataFrame with columns: parent_prod_id, price, rating_number, main_category

    Returns
    -------
    DataFrame indexed by parent_prod_id with: prod_avg_rating, prod_num_reviews,
    prod_price, prod_rating_number, main_category
    """
    # Review-side aggregates
    review_agg = train_pdf.groupby("parent_prod_id").agg(
        prod_avg_rating=("rating", "mean"),
        prod_num_reviews=("rating", "count"),
    )

    # Product metadata (one row per product)
    prod_meta = (
        prodinfo_pdf[["parent_prod_id", "price", "rating_number", "main_category"]]
        .drop_duplicates(subset=["parent_prod_id"])
        .set_index("parent_prod_id")
    )
    prod_meta = prod_meta.rename(
        columns={"price": "prod_price", "rating_number": "prod_rating_number"}
    )

    result = review_agg.join(prod_meta, how="left")
    return result


def compute_product_stats_kfold(
    train_pdf: pd.DataFrame,
    prodinfo_pdf: pd.DataFrame,
    n_splits: int = 5,
) -> tuple:
    """Compute per-row product stats using K-Fold to avoid target leakage.

    For each fold, compute product-level stats on the OTHER folds, then assign
    those stats to the current fold's rows.

    Parameters
    ----------
    train_pdf : DataFrame with columns: id, parent_prod_id, rating
    prodinfo_pdf : DataFrame with columns: parent_prod_id, price, rating_number, main_category
    n_splits : Number of K-Fold splits

    Returns
    -------
    oof_stats : DataFrame with columns: id, prod_avg_rating, prod_num_reviews,
        prod_price, prod_rating_number, main_category  (one row per train sample)
    full_stats : DataFrame indexed by parent_prod_id (for test set mapping)
    """
    log.info(f"Computing product stats with {n_splits}-Fold CV on {len(train_pdf)} rows")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    t0 = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_pdf)):
        fold_t0 = time.time()
        other_df = train_pdf.iloc[train_idx]
        val_ids = train_pdf.iloc[val_idx][["id", "parent_prod_id"]]

        # Compute stats on other folds only
        fold_stats = _compute_product_aggregates(other_df, prodinfo_pdf)

        # Join stats to validation rows
        val_with_stats = val_ids.merge(
            fold_stats, left_on="parent_prod_id", right_index=True, how="left"
        )
        fold_results.append(val_with_stats)

        log.info(
            f"  Fold {fold_idx}: val_rows={len(val_idx)}, "
            f"other_rows={len(train_idx)}, "
            f"unique_prods_other={other_df['parent_prod_id'].nunique()}, "
            f"time={time.time() - fold_t0:.1f}s"
        )

    # Combine all fold results
    oof_stats = pd.concat(fold_results, ignore_index=True)
    oof_stats = oof_stats.sort_values("id").reset_index(drop=True)

    # Compute full train stats (for test set)
    full_stats = _compute_product_aggregates(train_pdf, prodinfo_pdf)

    log.info(
        f"  Product stats K-Fold done in {time.time() - t0:.1f}s. "
        f"OOF rows: {len(oof_stats)}, unique prods full: {len(full_stats)}"
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

    # ── Compute K-Fold product stats for train ─────────────────────────
    oof_stats, full_stats = compute_product_stats_kfold(
        train_pdf, prodinfo_pdf, n_splits=5
    )

    # ── Compute full-train product stats for test ──────────────────────
    log.info("Mapping test rows to full-train product stats")
    test_prod_stats = test_pdf[["id", "parent_prod_id"]].merge(
        full_stats, left_on="parent_prod_id", right_index=True, how="left"
    )

    # ── Combine train OOF + test ───────────────────────────────────────
    all_stats = pd.concat([oof_stats, test_prod_stats], ignore_index=True)
    all_stats = all_stats.sort_values("id").reset_index(drop=True)

    # Fill NaN with global means (for products with no prior reviews)
    global_means = _compute_product_aggregates(train_pdf, prodinfo_pdf).mean(numeric_only=True)
    for col in ["prod_avg_rating", "prod_num_reviews", "prod_price", "prod_rating_number"]:
        if col in global_means.index:
            all_stats[col] = all_stats[col].fillna(global_means[col])

    # Fill main_category NaN with mode
    if all_stats["main_category"].isna().any():
        mode_cat = all_stats["main_category"].mode()
        if len(mode_cat) > 0:
            all_stats["main_category"] = all_stats["main_category"].fillna(mode_cat[0])

    # Ensure correct types
    all_stats["id"] = all_stats["id"].astype(int)
    all_stats["prod_num_reviews"] = all_stats["prod_num_reviews"].astype(int)

    # ── Save ───────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_stats.to_parquet(str(OUTPUT_PATH), index=False)
    log.info(f"Saved {OUTPUT_PATH} ({len(all_stats)} rows)")

    # ── Verify no leakage: sample check ────────────────────────────────
    log.info("\n=== Leakage Verification ===")
    # Find a product with exactly 2 reviews
    prod_counts = train_pdf.groupby("parent_prod_id").size()
    two_review_prods = prod_counts[prod_counts == 2].index
    if len(two_review_prods) > 0:
        sample_prod = two_review_prods[0]
        sample_rows = train_pdf[train_pdf["parent_prod_id"] == sample_prod].sort_values("id").reset_index(drop=True)
        sid1, sid2 = int(sample_rows.iloc[0]["id"]), int(sample_rows.iloc[1]["id"])
        r1, r2 = float(sample_rows.iloc[0]["rating"]), float(sample_rows.iloc[1]["rating"])

        stat1_series = all_stats.loc[all_stats["id"] == sid1, "prod_avg_rating"]
        stat2_series = all_stats.loc[all_stats["id"] == sid2, "prod_avg_rating"]

        if len(stat1_series) > 0 and len(stat2_series) > 0:
            stat1 = float(stat1_series.values[0])
            stat2 = float(stat2_series.values[0])

            log.info(f"  Sample product: {sample_prod}")
            log.info(f"    Row id={sid1}: rating={r1}, prod_avg_rating_feat={stat1:.4f} (should be {r2})")
            log.info(f"    Row id={sid2}: rating={r2}, prod_avg_rating_feat={stat2:.4f} (should be {r1})")

            if abs(stat1 - r2) < 1e-6 and abs(stat2 - r1) < 1e-6:
                log.info("  ✅ NO LEAKAGE: prod_avg_rating uses OTHER review's rating")
            else:
                log.warning("  ⚠️  Possible leakage detected! Check K-Fold logic.")
        else:
            log.warning(f"  Could not find stats for ids {sid1}, {sid2} in output")
    else:
        log.info("  No 2-review products found; skipping sample check")

    log.info(f"\n✅ Done in {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()

"""
Safe Target Encoding with K-Fold Cross-Validation, Smoothing, and Noise.

Prevents target leakage by computing encodings from OTHER folds only.
Adds Bayesian smoothing for rare groups and Gaussian noise for regularization.

Output: artifacts/features/safe_target_encoding_train.npz
  Columns (5 features):
    - user_te: User average rating (K-Fold + Smoothing + Noise)
    - prod_te: Product average rating (K-Fold + Smoothing + Noise)
    - cat_te:  Category average rating (K-Fold + Smoothing + Noise)
    - user_count: User review count (K-Fold, from other folds)
    - prod_count: Product review count (K-Fold, from other folds)

Parameters: K=5, Smoothing=10, Noise=0.01
"""

from __future__ import annotations

import os
import sys
import time
import logging
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
OUTPUT_PATH = ARTIFACTS_DIR / "features" / "safe_target_encoding_train.npz"

# Parameters
N_SPLITS = 5
SMOOTHING = 10.0
NOISE_STD = 0.01
RANDOM_STATE = 42


def _smoothed_mean(group_mean: float, group_count: int, global_mean: float) -> float:
    """Bayesian smoothing: shrink group mean toward global mean."""
    return (group_count * group_mean + SMOOTHING * global_mean) / (group_count + SMOOTHING)


def _compute_fold_encoding(
    other_df: pd.DataFrame,
    val_df: pd.DataFrame,
    group_col: str,
    global_mean: float,
) -> tuple:
    """Compute smoothed target encoding and count for a validation fold.

    Parameters
    ----------
    other_df : Training rows from OTHER folds (has 'rating' column)
    val_df : Validation rows to encode
    group_col : Column to group by
    global_mean : Global mean rating for smoothing fallback

    Returns
    -------
    te_values : np.ndarray of smoothed target-encoded values
    count_values : np.ndarray of group counts
    """
    # Compute stats on other folds only (NO LEAKAGE)
    group_stats = other_df.groupby(group_col)["rating"].agg(["mean", "count"])
    group_stats.columns = ["group_mean", "group_count"]

    # Bayesian smoothing
    group_stats["smoothed"] = (
        group_stats["group_count"] * group_stats["group_mean"]
        + SMOOTHING * global_mean
    ) / (group_stats["group_count"] + SMOOTHING)

    # Map to validation fold
    mapped_te = val_df[group_col].map(group_stats["smoothed"])
    mapped_count = val_df[group_col].map(group_stats["group_count"])

    # Fallback for unseen groups
    te_values = mapped_te.fillna(global_mean).values
    count_values = mapped_count.fillna(0).values

    return te_values, count_values


def main() -> None:
    t_total = time.time()

    # ── Load data ──────────────────────────────────────────────────────
    train_path = str(ARTIFACTS_DIR / "etl" / "train.parquet")
    prodinfo_path = str(ARTIFACTS_DIR / "etl" / "prodinfo.parquet")

    log.info(f"Loading train from {train_path}")
    train_df = pd.read_parquet(train_path)
    log.info(f"  train: {train_df.shape}")

    log.info(f"Loading prodinfo from {prodinfo_path}")
    prodinfo_df = pd.read_parquet(prodinfo_path)
    log.info(f"  prodinfo: {prodinfo_df.shape}")

    # main_category already exists in train.parquet, just fill NaN
    if "main_category" not in train_df.columns:
        prod_cat = prodinfo_df[["parent_prod_id", "main_category"]].drop_duplicates(
            subset=["parent_prod_id"]
        )
        train_df = train_df.merge(prod_cat, on="parent_prod_id", how="left")
    train_df["main_category"] = train_df["main_category"].fillna("unknown")

    n = len(train_df)
    log.info(f"  After merge: {n} rows, columns: {train_df.columns.tolist()}")

    # ── K-Fold Target Encoding ─────────────────────────────────────────
    global_mean = train_df["rating"].mean()
    log.info(f"  Global mean rating: {global_mean:.4f}")

    # Allocate output arrays
    user_te = np.full(n, np.nan)
    prod_te = np.full(n, np.nan)
    cat_te = np.full(n, np.nan)
    user_count = np.full(n, np.nan)
    prod_count = np.full(n, np.nan)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rng = np.random.RandomState(RANDOM_STATE)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        fold_t0 = time.time()
        other_df = train_df.iloc[train_idx]
        val_df = train_df.iloc[val_idx]

        # User target encoding
        u_te, u_cnt = _compute_fold_encoding(other_df, val_df, "user_id", global_mean)
        user_te[val_idx] = u_te
        user_count[val_idx] = u_cnt

        # Product target encoding
        p_te, p_cnt = _compute_fold_encoding(other_df, val_df, "parent_prod_id", global_mean)
        prod_te[val_idx] = p_te
        prod_count[val_idx] = p_cnt

        # Category target encoding
        c_te, _ = _compute_fold_encoding(other_df, val_df, "main_category", global_mean)
        cat_te[val_idx] = c_te

        log.info(
            f"  Fold {fold_idx}: val_rows={len(val_idx)}, "
            f"time={time.time() - fold_t0:.1f}s"
        )

    # ── Add noise injection ────────────────────────────────────────────
    log.info(f"Adding Gaussian noise (std={NOISE_STD})")
    user_te += rng.normal(0, NOISE_STD, n)
    prod_te += rng.normal(0, NOISE_STD, n)
    cat_te += rng.normal(0, NOISE_STD, n)

    # ── Verify no NaN ──────────────────────────────────────────────────
    nan_counts = {
        "user_te": np.isnan(user_te).sum(),
        "prod_te": np.isnan(prod_te).sum(),
        "cat_te": np.isnan(cat_te).sum(),
        "user_count": np.isnan(user_count).sum(),
        "prod_count": np.isnan(prod_count).sum(),
    }
    log.info(f"NaN counts: {nan_counts}")
    assert all(v == 0 for v in nan_counts.values()), f"Found NaN values: {nan_counts}"

    # ── Save ───────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(OUTPUT_PATH),
        user_te=user_te,
        prod_te=prod_te,
        cat_te=cat_te,
        user_count=user_count,
        prod_count=prod_count,
    )
    log.info(f"Saved {OUTPUT_PATH}")
    log.info(f"  Shape: ({n}, 5)")

    # ── Verification summary ───────────────────────────────────────────
    log.info("\n=== Verification ===")
    log.info(f"  user_te:  mean={user_te.mean():.4f}, std={user_te.std():.4f}, "
             f"min={user_te.min():.4f}, max={user_te.max():.4f}")
    log.info(f"  prod_te:  mean={prod_te.mean():.4f}, std={prod_te.std():.4f}, "
             f"min={prod_te.min():.4f}, max={prod_te.max():.4f}")
    log.info(f"  cat_te:   mean={cat_te.mean():.4f}, std={cat_te.std():.4f}, "
             f"min={cat_te.min():.4f}, max={cat_te.max():.4f}")
    log.info(f"  user_count: mean={user_count.mean():.1f}, max={user_count.max():.0f}")
    log.info(f"  prod_count: mean={prod_count.mean():.1f}, max={prod_count.max():.0f}")

    # OOF RMSE check: if user_te is a good predictor of rating
    oof_rmse = np.sqrt(np.mean((user_te - train_df["rating"].values) ** 2))
    log.info(f"  OOF RMSE (user_te vs rating): {oof_rmse:.4f}")

    log.info(f"\n✅ Done in {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()

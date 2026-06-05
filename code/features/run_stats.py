"""Runner for all statistical feature computations (T8)."""

from __future__ import annotations

import os
import sys

# Ensure PySpark workers use the same Python interpreter.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pathlib import Path

from code.config import ARTIFACTS_DIR
from code.utils.spark_session import get_spark
from code.utils.timer import StageTimer


def main() -> None:
    spark = get_spark("T8-Stats")
    timer = StageTimer()

    # Ensure output directory exists
    out_dir = ARTIFACTS_DIR / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_path = str(ARTIFACTS_DIR / "etl" / "train.parquet")
    prodinfo_path = str(ARTIFACTS_DIR / "etl" / "prodinfo.parquet")

    print(f"[run_stats] Loading train  from {train_path}")
    train_df = spark.read.parquet(train_path)
    print(f"[run_stats] Loading prodinfo from {prodinfo_path}")
    prodinfo_df = spark.read.parquet(prodinfo_path)

    # --- User stats ---
    from code.features.user_stats import compute_user_avg_rating, save_user_stats

    print("[run_stats] Computing user stats …")
    user_stats = compute_user_avg_rating(train_df, stage_timer=timer)
    save_user_stats(user_stats)
    print(f"[run_stats] → {ARTIFACTS_DIR / 'features' / 'user_stats.parquet'}")

    # --- Product stats ---
    from code.features.product_stats import compute_product_stats, save_product_stats

    print("[run_stats] Computing product stats …")
    product_stats = compute_product_stats(train_df, prodinfo_df, stage_timer=timer)
    save_product_stats(product_stats)
    print(f"[run_stats] → {ARTIFACTS_DIR / 'features' / 'product_stats.parquet'}")

    # --- Category stats ---
    from code.features.category_stats import compute_category_stats, save_category_stats

    print("[run_stats] Computing category stats …")
    category_stats = compute_category_stats(train_df, prodinfo_df, stage_timer=timer)
    save_category_stats(category_stats)
    print(f"[run_stats] → {ARTIFACTS_DIR / 'features' / 'category_stats.parquet'}")

    # Summary
    print("\n[run_stats] Timings:")
    for stage, metrics in timer.stages.items():
        for key, elapsed in metrics.items():
            print(f"  {stage}/{key}: {elapsed:.2f}s")

    print("[run_stats] Done.")


if __name__ == "__main__":
    main()

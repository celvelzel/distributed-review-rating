"""End-to-end ETL runner.

Usage:
    python code/etl/run_etl.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path so that ``code.*`` imports work.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from code.utils.spark_session import get_spark
from code.utils.timer import StageTimer, write_metrics
from code.etl.spark_etl import (
    clean_text,
    extract_time_features,
    impute_missing,
    join_with_prodinfo,
    load_prodinfo,
    load_test,
    load_train,
    persist_parquet,
)


def run_etl() -> None:
    """Execute the full ETL pipeline and persist outputs."""
    timer = StageTimer()

    spark = get_spark("COMP5434_ETL")
    spark.sparkContext.setLogLevel("WARN")

    etl_dir = _ROOT / "artifacts" / "etl"
    etl_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load ─────────────────────────────────────────────────────────────
    print("[1/6] Loading data ...")
    df_train = load_train(spark, path=str(_ROOT / "data" / "train.csv"),
                          stage_timer=timer)
    df_test = load_test(spark, path=str(_ROOT / "data" / "test.csv"),
                        stage_timer=timer)
    df_prodinfo = load_prodinfo(spark, path=str(_ROOT / "data" / "prodInfo.csv"),
                                stage_timer=timer)

    train_count_raw = df_train.count()
    print(f"  train (after filtering invalid ratings): {train_count_raw:,}")

    # ── 2. Clean text ──────────────────────────────────────────────────────
    print("[2/6] Cleaning text ...")
    for col_name in ("title", "comment"):
        if col_name in df_train.columns:
            df_train = clean_text(df_train, col_name, stage_timer=timer)
        if col_name in df_test.columns:
            df_test = clean_text(df_test, col_name, stage_timer=timer)

    # ── 3. Impute missing ──────────────────────────────────────────────────
    print("[3/6] Imputing missing values ...")
    df_train = impute_missing(df_train, stage_timer=timer)
    df_test = impute_missing(df_test, stage_timer=timer)
    df_prodinfo = impute_missing(df_prodinfo, stage_timer=timer)

    # ── 4. Join with product info ──────────────────────────────────────────
    print("[4/6] Joining with product info ...")
    df_train = join_with_prodinfo(df_train, df_prodinfo, stage_timer=timer)
    df_test = join_with_prodinfo(df_test, df_prodinfo, stage_timer=timer)

    # ── 5. Extract time features ───────────────────────────────────────────
    print("[5/6] Extracting time features ...")
    df_train = extract_time_features(df_train, stage_timer=timer)
    df_test = extract_time_features(df_test, stage_timer=timer)

    # ── 6. Persist ─────────────────────────────────────────────────────────
    print("[6/6] Persisting Parquet ...")
    train_path = str(etl_dir / "train.parquet")
    test_path = str(etl_dir / "test.parquet")
    prodinfo_path = str(etl_dir / "prodinfo.parquet")

    persist_parquet(df_train, train_path, stage_timer=timer)
    persist_parquet(df_test, test_path, stage_timer=timer)
    persist_parquet(df_prodinfo, prodinfo_path, stage_timer=timer)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\nETL complete.")
    print(f"  Artifacts: {etl_dir}")
    for name, path in [("train", train_path), ("test", test_path),
                       ("prodinfo", prodinfo_path)]:
        print(f"    {name}: {path}")

    # Write timing metrics
    metrics_path = str(etl_dir / "metrics.json")
    write_metrics(metrics_path, {"etl": timer.to_dict()})
    print(f"  Timing metrics: {metrics_path}")


if __name__ == "__main__":
    run_etl()

"""Temporal features extracted from the review timestamp."""

from __future__ import annotations

import os
import sys

# Ensure PySpark uses the correct Python interpreter.
os.environ.setdefault("PYSPARK_PYTHON", "/usr/bin/python3.8")
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", "/usr/bin/python3.8")

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from code.utils.spark_session import get_spark
from code.utils.timer import timed


@timed("features", "temporal")
def extract_temporal(df: DataFrame) -> DataFrame:
    """Extract time-based features from a review DataFrame.

    Expects the DataFrame to contain:
      - ``id``  — review identifier (string)
      - ``time`` — Unix millisecond timestamp (long or castable)

    Returns a DataFrame with columns:
      id, year, month, day, weekday, hour, is_weekend, is_holiday_season

    Also writes the result to ``artifacts/features/temporal.parquet``.
    """
    # Cast time to long if needed (handles string inputs)
    df = df.withColumn("time_ms", F.col("time").cast(T.LongType()))

    temporal_df = (
        df.select(
            F.col("id"),
            F.year(F.from_unixtime(F.col("time_ms") / 1000)).alias("year"),
            F.month(F.from_unixtime(F.col("time_ms") / 1000)).alias("month"),
            F.dayofmonth(F.from_unixtime(F.col("time_ms") / 1000)).alias("day"),
            F.dayofweek(F.from_unixtime(F.col("time_ms") / 1000)).alias("weekday"),
            F.hour(F.from_unixtime(F.col("time_ms") / 1000)).alias("hour"),
            # dayofweek: 1=Sun, 7=Sat in Spark
            F.when(
                F.dayofweek(F.from_unixtime(F.col("time_ms") / 1000)).isin(1, 7), 1
            ).otherwise(0).alias("is_weekend"),
            # Holiday season: November (11) and December (12)
            F.when(
                F.month(F.from_unixtime(F.col("time_ms") / 1000)).isin(11, 12), 1
            ).otherwise(0).alias("is_holiday_season"),
        )
    )

    # Persist to disk
    out_path = "artifacts/features/temporal.parquet"
    temporal_df.write.mode("overwrite").parquet(out_path)
    print(f"[temporal] wrote {out_path}")

    return temporal_df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    spark = get_spark("features-temporal")
    train_df = spark.read.parquet("artifacts/etl/train.parquet")
    result = extract_temporal(train_df)
    result.show(10, truncate=False)
    print(f"[temporal] rows = {result.count()}")
    spark.stop()

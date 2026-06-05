"""Text-length features extracted from review title and comment."""

from __future__ import annotations

import os
import sys

# Ensure PySpark uses the correct Python interpreter.
os.environ.setdefault("PYSPARK_PYTHON", "/usr/bin/python3.8")
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", "/usr/bin/python3.8")

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import DataFrame

from code.utils.spark_session import get_spark
from code.utils.timer import timed


@timed("features", "text_length")
def extract_length(df: DataFrame) -> DataFrame:
    """Extract text-length features from a review DataFrame.

    Expects the DataFrame to contain:
      - ``id``      — review identifier (string)
      - ``title``   — review title (string, nullable)
      - ``comment`` — review body text (string, nullable)

    Returns a DataFrame with columns:
      id, title_len, comment_len, title_comment_ratio, has_caps, has_exclamation

    Also writes the result to ``artifacts/features/text_length.parquet``.
    """
    # Coalesce nulls to empty strings for safe length computation
    title_safe = F.coalesce(F.col("title"), F.lit(""))
    comment_safe = F.coalesce(F.col("comment"), F.lit(""))

    df = df.withColumn("title_len", F.length(title_safe).cast(T.IntegerType()))
    df = df.withColumn("comment_len", F.length(comment_safe).cast(T.IntegerType()))

    # title_comment_ratio = title_len / (comment_len + 1)
    df = df.withColumn(
        "title_comment_ratio",
        F.col("title_len").cast(T.DoubleType()) / (F.col("comment_len") + 1),
    )

    # has_caps: 1 if any uppercase letter in title+comment
    combined = F.concat(title_safe, comment_safe)
    df = df.withColumn(
        "has_caps",
        F.when(F.regexp_extract(combined, r"[A-Z]", 0) != "", 1).otherwise(0).cast(T.IntegerType()),
    )

    # has_exclamation: 1 if '!' in title+comment
    df = df.withColumn(
        "has_exclamation",
        F.when(F.regexp_extract(combined, r"[!]", 0) != "", 1).otherwise(0).cast(T.IntegerType()),
    )

    result = df.select(
        "id",
        "title_len",
        "comment_len",
        "title_comment_ratio",
        "has_caps",
        "has_exclamation",
    )

    # Persist to disk
    out_path = "artifacts/features/text_length.parquet"
    result.write.mode("overwrite").parquet(out_path)
    print(f"[text_length] wrote {out_path}")

    return result


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    spark = get_spark("features-text-length")
    train_df = spark.read.parquet("artifacts/etl/train.parquet")
    result = extract_length(train_df)
    result.show(10, truncate=False)
    print(f"[text_length] rows = {result.count()}")
    spark.stop()

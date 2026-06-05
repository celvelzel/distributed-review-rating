"""Price-based features extracted from product information."""

from __future__ import annotations

import os
import sys

# Ensure PySpark uses the correct Python interpreter.
os.environ.setdefault("PYSPARK_PYTHON", "/usr/bin/python3.8")
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", "/usr/bin/python3.8")

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import DataFrame, Window

from code.utils.spark_session import get_spark
from code.utils.timer import timed


@timed("features", "price_features")
def extract_price(prodinfo_df: DataFrame) -> DataFrame:
    """Extract price-based features from the product-info DataFrame.

    Expects the DataFrame to contain:
      - ``parent_prod_id`` — product identifier (string)
      - ``main_category``  — product category (string)
      - ``price``          — price as double (nullable)

    Returns a DataFrame with columns:
      parent_prod_id, log_price, price_rank_in_category, price_bucket

    Also writes the result to ``artifacts/features/price_features.parquet``.
    """
    # Cast price to double if needed (handles string inputs)
    df = prodinfo_df.withColumn("price_d", F.col("price").cast(T.DoubleType()))

    # --- log_price = log(1 + price) ---
    df = df.withColumn("log_price", F.log1p(F.col("price_d")))

    # --- price_rank_in_category ---
    # Rank within main_category (dense_rank, ascending by price)
    cat_window = Window.partitionBy("main_category").orderBy(F.col("price_d").asc_nulls_last())
    df = df.withColumn(
        "price_rank_in_category",
        F.dense_rank().over(cat_window).cast(T.DoubleType()),
    )

    # --- price_bucket: 0=low (<$20), 1=mid ($20-$100), 2=high (>$100) ---
    df = df.withColumn(
        "price_bucket",
        F.when(F.col("price_d") < 20, 0)
        .when(F.col("price_d") <= 100, 1)
        .otherwise(2)
        .cast(T.IntegerType()),
    )

    result = df.select(
        "parent_prod_id",
        "log_price",
        "price_rank_in_category",
        "price_bucket",
    )

    # Persist to disk
    out_path = "artifacts/features/price_features.parquet"
    result.write.mode("overwrite").parquet(out_path)
    print(f"[price_features] wrote {out_path}")

    return result


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    spark = get_spark("features-price")
    prodinfo_df = spark.read.parquet("artifacts/etl/prodinfo.parquet")
    result = extract_price(prodinfo_df)
    result.show(10, truncate=False)
    print(f"[price_features] rows = {result.count()}")
    spark.stop()

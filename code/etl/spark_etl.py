"""Spark ETL functions for data loading, cleaning, joining, and persisting."""

from __future__ import annotations

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DoubleType

from code.utils.timer import timed


# ---------------------------------------------------------------------------
# 1. Loading functions
# ---------------------------------------------------------------------------

@timed("etl", "load_train_sec")
def load_train(spark: SparkSession, *, path: str = "data/train.csv",
               stage_timer=None) -> DataFrame:
    """Read training CSV.  All columns read as strings to avoid schema
    inference issues with multi-line comment overflows in the rating column.
    Rating and votes are cast to double and invalid rows are filtered out.
    """
    df = (
        spark.read
        .option("header", "true")
        .option("multiLine", "true")
        .option("escape", '"')
        .csv(path)
    )
    df = df.withColumn("rating", F.col("rating").cast(DoubleType()))
    df = df.withColumn("votes", F.col("votes").cast(DoubleType()))
    df = df.withColumn("time", F.col("time").cast("long"))
    # Filter out rows where rating could not be parsed (text pollution)
    df = df.filter(F.col("rating").isNotNull())
    return df


@timed("etl", "load_test_sec")
def load_test(spark: SparkSession, *, path: str = "data/test.csv",
              stage_timer=None) -> DataFrame:
    """Read test CSV.  Cast votes to double and time to long."""
    df = (
        spark.read
        .option("header", "true")
        .option("multiLine", "true")
        .option("escape", '"')
        .csv(path)
    )
    df = df.withColumn("votes", F.col("votes").cast(DoubleType()))
    df = df.withColumn("time", F.col("time").cast("long"))
    return df


@timed("etl", "load_prodinfo_sec")
def load_prodinfo(spark: SparkSession, *, path: str = "data/prodInfo.csv",
                  stage_timer=None) -> DataFrame:
    """Read product info CSV.  Cast price and rating_number to double."""
    df = (
        spark.read
        .option("header", "true")
        .option("multiLine", "true")
        .option("escape", '"')
        .csv(path)
    )
    df = df.withColumn("price", F.col("price").cast(DoubleType()))
    df = df.withColumn("rating_number", F.col("rating_number").cast(DoubleType()))
    return df


# ---------------------------------------------------------------------------
# 2. Cleaning functions
# ---------------------------------------------------------------------------

@timed("etl", "clean_text_sec")
def clean_text(df: DataFrame, col_name: str, *,
               stage_timer=None) -> DataFrame:
    """Remove HTML tags, URLs, and special characters from *col_name*,
    then lowercase the result.
    """
    cleaned = F.col(col_name)
    # Remove HTML tags
    cleaned = F.regexp_replace(cleaned, r"<[^>]+>", " ")
    # Remove URLs
    cleaned = F.regexp_replace(cleaned, r"https?://\S+", " ")
    # Remove special characters (keep alphanumeric, spaces, basic punctuation)
    cleaned = F.regexp_replace(cleaned, r"[^a-zA-Z0-9\s.,!?']", " ")
    # Collapse whitespace and lowercase
    cleaned = F.lower(F.trim(F.regexp_replace(cleaned, r"\s+", " ")))
    return df.withColumn(col_name, cleaned)


# ---------------------------------------------------------------------------
# 3. Imputation
# ---------------------------------------------------------------------------

@timed("etl", "impute_missing_sec")
def impute_missing(df: DataFrame, *,
                   stage_timer=None) -> DataFrame:
    """Fill missing values:
    - title / comment → 'unknown'
    - votes → 0
    - price → median of non-null prices (if price column exists)
    """
    for col_name in ("title", "comment"):
        if col_name in df.columns:
            df = df.fillna({col_name: "unknown"})

    if "votes" in df.columns:
        df = df.fillna({"votes": 0})

    if "price" in df.columns:
        median_price = df.filter(F.col("price").isNotNull()) \
                         .approxQuantile("price", [0.5], 0.01)
        if median_price:
            df = df.fillna({"price": median_price[0]})

    return df


# ---------------------------------------------------------------------------
# 4. Join
# ---------------------------------------------------------------------------

@timed("etl", "join_with_prodinfo_sec")
def join_with_prodinfo(df: DataFrame, df_prodinfo: DataFrame, *,
                       stage_timer=None) -> DataFrame:
    """Broadcast-join *df* with product info on ``parent_prod_id``."""
    from pyspark.sql.functions import broadcast
    return df.join(broadcast(df_prodinfo), on="parent_prod_id", how="left")


# ---------------------------------------------------------------------------
# 5. Feature extraction
# ---------------------------------------------------------------------------

@timed("etl", "extract_time_features_sec")
def extract_time_features(df: DataFrame, *,
                          stage_timer=None) -> DataFrame:
    """Extract year, month, weekday, hour, and is_weekend from the ``time``
    column (Unix timestamp in milliseconds).
    """
    ts = F.col("time") / 1000  # convert ms → seconds
    df = df.withColumn("review_year", F.year(F.from_unixtime(ts)))
    df = df.withColumn("review_month", F.month(F.from_unixtime(ts)))
    df = df.withColumn("review_weekday", F.dayofweek(F.from_unixtime(ts)))
    df = df.withColumn("review_hour", F.hour(F.from_unixtime(ts)))
    df = df.withColumn("is_weekend", F.when(
        F.col("review_weekday").isin(1, 7), 1).otherwise(0))
    return df


# ---------------------------------------------------------------------------
# 6. Persistence
# ---------------------------------------------------------------------------

@timed("etl", "persist_parquet_sec")
def persist_parquet(df: DataFrame, output_path: str, *,
                    stage_timer=None) -> None:
    """Write DataFrame to Parquet (overwrite mode)."""
    df.write.mode("overwrite").parquet(output_path)

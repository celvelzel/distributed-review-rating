"""Product-level statistical features."""

from __future__ import annotations

from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from code.config import ARTIFACTS_DIR
from code.utils.timer import timed

OUTPUT_PATH = ARTIFACTS_DIR / "features" / "product_stats.parquet"


@timed("features", "product_stats")
def compute_product_stats(train_df: DataFrame, prodinfo_df: DataFrame, **kwargs) -> DataFrame:
    """Compute per-product aggregated statistics.

    Combines review-side aggregates (from *train_df*) with product metadata
    (from *prodinfo_df*) keyed by ``parent_prod_id``.

    Parameters
    ----------
    train_df : DataFrame
        Training data with columns: parent_prod_id, rating.
    prodinfo_df : DataFrame
        Product info with columns: parent_prod_id, price, rating_number, main_category.

    Returns
    -------
    DataFrame
        Columns: parent_prod_id, prod_avg_rating, prod_num_reviews,
        prod_price, prod_rating_number, main_category
    """
    # Aggregates from training reviews
    train_agg = (
        train_df.groupBy("parent_prod_id")
        .agg(
            F.avg("rating").alias("prod_avg_rating"),
            F.count("*").alias("prod_num_reviews"),
        )
    )

    # Distinct product metadata (one row per parent_prod_id)
    prod_meta = (
        prodinfo_df.select("parent_prod_id", "price", "rating_number", "main_category")
        .dropDuplicates(["parent_prod_id"])
    )

    result = train_agg.join(prod_meta, on="parent_prod_id", how="left")
    result = result.withColumnRenamed("price", "prod_price").withColumnRenamed("rating_number", "prod_rating_number")
    return result


def save_product_stats(df: DataFrame) -> Path:
    output = Path(OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.write.mode("overwrite").parquet(str(output))
    return output

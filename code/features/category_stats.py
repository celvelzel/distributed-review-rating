"""Category-level statistical features."""

from __future__ import annotations

from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from code.config import ARTIFACTS_DIR
from code.utils.timer import timed

OUTPUT_PATH = ARTIFACTS_DIR / "features" / "category_stats.parquet"


@timed("features", "category_stats")
def compute_category_stats(train_df: DataFrame, prodinfo_df: DataFrame, **kwargs) -> DataFrame:
    """Compute per-category aggregated statistics.

    Uses review ratings from *train_df* and prices from *prodinfo_df*,
    grouped by ``main_category``.

    Parameters
    ----------
    train_df : DataFrame
        Training data with columns: main_category, rating.
    prodinfo_df : DataFrame
        Product info with columns: main_category, price.

    Returns
    -------
    DataFrame
        Columns: main_category, cat_avg_rating, cat_avg_price, cat_rating_std
    """
    # Rating stats from training set
    rating_stats = (
        train_df.groupBy("main_category")
        .agg(
            F.avg("rating").alias("cat_avg_rating"),
            F.stddev_pop("rating").alias("cat_rating_std"),
        )
    )

    # Price stats from product info (distinct products to avoid counting duplicates)
    price_stats = (
        prodinfo_df.select("main_category", "price")
        .dropDuplicates(["main_category", "price"])  # dedup exact price per category
        .groupBy("main_category")
        .agg(F.avg("price").alias("cat_avg_price"))
    )

    result = rating_stats.join(price_stats, on="main_category", how="left")
    return result


def save_category_stats(df: DataFrame) -> Path:
    output = Path(OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.write.mode("overwrite").parquet(str(output))
    return output

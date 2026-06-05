"""User-level statistical features."""

from __future__ import annotations

from pathlib import Path

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from code.config import ARTIFACTS_DIR
from code.utils.timer import timed

OUTPUT_PATH = ARTIFACTS_DIR / "features" / "user_stats.parquet"


@timed("features", "user_stats")
def compute_user_avg_rating(df: DataFrame, **kwargs) -> DataFrame:
    """Compute per-user aggregated statistics from the training set.

    Parameters
    ----------
    df : DataFrame
        Training DataFrame with columns: user_id, rating, votes, purchased.

    Returns
    -------
    DataFrame
        Columns: user_id, avg_rating, num_reviews, avg_votes, purchased_rate, rating_std
    """
    stats = (
        df.groupBy("user_id")
        .agg(
            F.avg("rating").alias("avg_rating"),
            F.count("*").alias("num_reviews"),
            F.avg("votes").alias("avg_votes"),
            F.avg(F.when(F.col("purchased") == "True", 1.0).otherwise(0.0)).alias("purchased_rate"),
            F.stddev_pop("rating").alias("rating_std"),
        )
    )
    return stats


def save_user_stats(df: DataFrame) -> Path:
    output = Path(OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.write.mode("overwrite").parquet(str(output))
    return output

"""Unit tests for code.etl.spark_etl module."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure workers use the same Python as the driver.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

# Add code/ to sys.path so ``etl.spark_etl`` is importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from etl.spark_etl import (
    clean_text,
    extract_time_features,
    impute_missing,
    join_with_prodinfo,
    persist_parquet,
)


def _make_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("test_etl")
        .master("local[2]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.memory", "1g")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )


# Shared SparkSession (created once per test process).
_spark: SparkSession = _make_spark()


def _sample_train_df(n: int = 100):
    """Create a sample train DataFrame with *n* rows."""
    rows = []
    for i in range(n):
        rows.append((
            str(i),                     # id
            f"u_{i}",                   # user_id
            f"p_{i % 10}",             # prod_id
            f"pp_{i % 5}",             # parent_prod_id
            f"<b>Title {i}</b>",       # title  (contains HTML)
            f"Visit http://ex.com/{i} great product!",  # comment
            1541207203590 + i * 86400000,  # time (ms)
            float(i % 3),              # votes
            "True" if i % 2 == 0 else "False",  # purchased
            float((i % 5) + 1),        # rating 1.0-5.0
        ))
    schema = StructType([
        StructField("id", StringType()),
        StructField("user_id", StringType()),
        StructField("prod_id", StringType()),
        StructField("parent_prod_id", StringType()),
        StructField("title", StringType()),
        StructField("comment", StringType()),
        StructField("time", LongType()),
        StructField("votes", DoubleType()),
        StructField("purchased", StringType()),
        StructField("rating", DoubleType()),
    ])
    return _spark.createDataFrame(rows, schema)


def _sample_prodinfo_df(n: int = 5):
    """Create a small product-info DataFrame."""
    rows = []
    for i in range(n):
        rows.append((
            str(i),           # id
            f"pp_{i}",        # parent_prod_id
            "Electronics",    # main_category
            float(10 + i),    # price
            f"Product {i}",   # title
            "[]",             # features
            "StoreA",         # store
            float(100 + i),   # rating_number
        ))
    schema = StructType([
        StructField("id", StringType()),
        StructField("parent_prod_id", StringType()),
        StructField("main_category", StringType()),
        StructField("price", DoubleType()),
        StructField("title", StringType()),
        StructField("features", StringType()),
        StructField("store", StringType()),
        StructField("rating_number", DoubleType()),
    ])
    return _spark.createDataFrame(rows, schema)


class TestCleanText(unittest.TestCase):
    """Tests for clean_text()."""

    def test_removes_html_tags(self):
        df = _sample_train_df(10)
        result = clean_text(df, "title")
        titles = [r.title for r in result.select("title").collect()]
        for t in titles:
            self.assertNotIn("<b>", t)
            self.assertNotIn("</b>", t)

    def test_removes_urls_and_lowercases(self):
        df = _sample_train_df(10)
        result = clean_text(df, "comment")
        comments = [r.comment for r in result.select("comment").collect()]
        for c in comments:
            self.assertNotIn("http://", c)
            self.assertEqual(c, c.lower())


class TestImputeMissing(unittest.TestCase):
    """Tests for impute_missing()."""

    def test_fills_null_title_and_comment(self):
        rows = [
            ("1", "u1", "p1", "pp1", None, None, 1541207203590,
             0.0, "True", 5.0),
        ]
        schema = StructType([
            StructField("id", StringType()),
            StructField("user_id", StringType()),
            StructField("prod_id", StringType()),
            StructField("parent_prod_id", StringType()),
            StructField("title", StringType()),
            StructField("comment", StringType()),
            StructField("time", LongType()),
            StructField("votes", DoubleType()),
            StructField("purchased", StringType()),
            StructField("rating", DoubleType()),
        ])
        df = _spark.createDataFrame(rows, schema)
        result = impute_missing(df)
        row = result.first()
        self.assertEqual(row.title, "unknown")
        self.assertEqual(row.comment, "unknown")

    def test_fills_null_votes_with_zero(self):
        rows = [
            ("1", "u1", "p1", "pp1", "t", "c", 1541207203590,
             None, "True", 5.0),
        ]
        schema = StructType([
            StructField("id", StringType()),
            StructField("user_id", StringType()),
            StructField("prod_id", StringType()),
            StructField("parent_prod_id", StringType()),
            StructField("title", StringType()),
            StructField("comment", StringType()),
            StructField("time", LongType()),
            StructField("votes", DoubleType()),
            StructField("purchased", StringType()),
            StructField("rating", DoubleType()),
        ])
        df = _spark.createDataFrame(rows, schema)
        result = impute_missing(df)
        row = result.first()
        self.assertEqual(row.votes, 0.0)


class TestExtractTimeFeatures(unittest.TestCase):
    """Tests for extract_time_features()."""

    def test_adds_expected_columns(self):
        df = _sample_train_df(20)
        result = extract_time_features(df)
        expected_cols = {"review_year", "review_month", "review_weekday",
                         "review_hour", "is_weekend"}
        self.assertTrue(expected_cols.issubset(set(result.columns)))

    def test_values_are_plausible(self):
        df = _sample_train_df(20)
        result = extract_time_features(df)
        row = result.first()
        # 1541207203590 ms → 2018-11-02 ~17:06 UTC
        self.assertEqual(row.review_year, 2018)
        self.assertIn(row.review_month, range(1, 13))
        self.assertIn(row.review_weekday, range(1, 8))
        self.assertIn(row.review_hour, range(0, 24))
        self.assertIn(row.is_weekend, (0, 1))


class TestJoinWithProdinfo(unittest.TestCase):
    """Tests for join_with_prodinfo()."""

    def test_join_adds_prodinfo_columns(self):
        df_train = _sample_train_df(20)
        df_prod = _sample_prodinfo_df(5)
        result = join_with_prodinfo(df_train, df_prod)
        self.assertIn("main_category", result.columns)
        self.assertIn("price", result.columns)
        # Row count should match train (left join)
        self.assertEqual(result.count(), df_train.count())


class TestPersistParquet(unittest.TestCase):
    """Tests for persist_parquet()."""

    def test_writes_readable_parquet(self):
        df = _sample_train_df(10)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "test.parquet")
            persist_parquet(df, out)
            loaded = _spark.read.parquet(out)
            self.assertEqual(loaded.count(), 10)
            self.assertEqual(set(loaded.columns), set(df.columns))


if __name__ == "__main__":
    unittest.main()

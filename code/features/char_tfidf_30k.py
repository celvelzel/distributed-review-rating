"""Character-level TF-IDF features with 30K dimensions.

Distributed data loading via PySpark, then sklearn TfidfVectorizer for
char-level TF-IDF extraction (char_wb analyzer, ngram_range=(3,5), 30K features).

Character-level n-grams on 3M rows are too memory-intensive for PySpark ML
(NGram explosion causes JVM OOM), so we use sklearn which handles this much
more efficiently with its optimized C implementation.

Parameters
----------
max_features : 30000
ngram_range  : (3, 5)
analyzer     : char_wb

Output
------
- ``artifacts/features/char_tfidf_30k_train.npz`` — sparse CSR matrix (train)
- ``artifacts/features/char_tfidf_30k_test.npz``  — sparse CSR matrix (test)
- ``artifacts/features/char_tfidf_30k_meta.json``  — feature metadata
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARTIFACT_DIR = Path("artifacts/features")
ETL_DIR = Path("artifacts/etl")

MAX_FEATURES = 30000
NGRAM_RANGE = (3, 5)
ANALYZER = "char_wb"
TEXT_COLS = ("title", "comment")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_spark():
    """Get or create SparkSession (local mode)."""
    from pyspark.sql import SparkSession

    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    spark = (
        SparkSession.builder
        .appName("CharTFIDF_30K")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "100")
        .config("spark.driver.memory", "8g")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.broadcastTimeout", "600")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )
    return spark


def prepare_text(spark, parquet_path):
    """Read parquet and create combined, cleaned text column."""
    from pyspark.sql.functions import col, concat_ws, lower, regexp_replace

    df = spark.read.parquet(parquet_path)
    title_col, comment_col = TEXT_COLS

    df = df.withColumn(
        "text",
        lower(regexp_replace(
            concat_ws(" ",
                      col(title_col).cast("string"),
                      col(comment_col).cast("string")),
            "[^a-z0-9\\s]", " ",
        ))
    )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(etl_dir=ETL_DIR, out_dir=ARTIFACT_DIR,
        max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE,
        analyzer=ANALYZER):
    """Run char-level TF-IDF 30K feature extraction.

    Pipeline:
    1. PySpark: load ETL parquet, combine title+comment, clean text
    2. Collect text to driver as pandas Series
    3. sklearn TfidfVectorizer: char_wb, ngram_range=(3,5), max_features=30000
    4. Save sparse matrices as .npz
    """
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer

    etl_dir = Path(etl_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Char-level TF-IDF 30K Feature Extraction")
    print(f"  max_features={max_features}, ngram_range={ngram_range}")
    print(f"  analyzer={analyzer}")
    print("=" * 60)

    t_total = time.perf_counter()

    # ---- Step 1: Load data with PySpark ----
    print("\n--- Step 1: Loading ETL data with PySpark ---")
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    print(f"  Spark {spark.version}, master={spark.sparkContext.master}")

    t0 = time.perf_counter()
    train_df = prepare_text(spark, str(etl_dir / "train.parquet"))
    test_df = prepare_text(spark, str(etl_dir / "test.parquet"))

    # Collect text columns to driver
    print("  Collecting train text...")
    train_text_rows = train_df.select("text").collect()
    train_texts = pd.Series([row["text"] or "" for row in train_text_rows])
    print(f"  Train: {len(train_texts):,} rows ({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    print("  Collecting test text...")
    test_text_rows = test_df.select("text").collect()
    test_texts = pd.Series([row["text"] or "" for row in test_text_rows])
    print(f"  Test:  {len(test_texts):,} rows ({time.perf_counter() - t0:.1f}s)")

    # Free Spark memory
    spark.stop()
    print("  Spark stopped.")

    # ---- Step 2: Fit TF-IDF with sklearn ----
    print("\n--- Step 2: Fitting char-level TF-IDF (sklearn) ---")
    vec = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )

    t0 = time.perf_counter()
    X_train = vec.fit_transform(train_texts)
    print(f"  Train: {X_train.shape}  ({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    X_test = vec.transform(test_texts)
    print(f"  Test:  {X_test.shape}  ({time.perf_counter() - t0:.1f}s)")

    feature_names = vec.get_feature_names_out().tolist()
    print(f"  Features: {len(feature_names)}")
    print(f"  Sample: {feature_names[:10]}")

    # ---- Step 3: Save ----
    print("\n--- Step 3: Saving ---")
    train_path = out_dir / "char_tfidf_30k_train.npz"
    sparse.save_npz(str(train_path), X_train)
    print(f"  Train -> {train_path}  {X_train.shape}")

    test_path = out_dir / "char_tfidf_30k_test.npz"
    sparse.save_npz(str(test_path), X_test)
    print(f"  Test  -> {test_path}  {X_test.shape}")

    meta = {
        "max_features": max_features,
        "ngram_range": list(ngram_range),
        "analyzer": analyzer,
        "sublinear_tf": True,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "train_nnz": int(X_train.nnz),
        "test_nnz": int(X_test.nnz),
        "train_density": float(X_train.nnz / np.prod(X_train.shape)),
        "test_density": float(X_test.nnz / np.prod(X_test.shape)),
        "n_features": len(feature_names),
        "feature_names_sample": feature_names[:50],
    }
    meta_path = out_dir / "char_tfidf_30k_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta  -> {meta_path}")

    # ---- Summary ----
    print("\n--- Summary ---")
    print(f"  Train: {X_train.shape}, nnz={X_train.nnz:,}")
    print(f"  Test:  {X_test.shape}, nnz={X_test.nnz:,}")
    print(f"  Train density: {X_train.nnz / np.prod(X_train.shape):.6f}")
    print(f"  Test  density: {X_test.nnz / np.prod(X_test.shape):.6f}")
    print(f"\nDone in {time.perf_counter() - t_total:.1f}s")

    return X_train, X_test


if __name__ == "__main__":
    run()

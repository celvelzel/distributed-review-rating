"""Word-level TF-IDF features with 50K dimensions using PySpark.

Distributed TF-IDF extraction using PySpark ML pipeline:
  RegexTokenizer → NGram(n=2,3) → concat → HashingTF(50K) → IDF

Parameters
----------
max_features : 50000
ngram_range  : (1, 3)

Output
------
- ``artifacts/features/tfidf_50k_train.npz`` — sparse CSR matrix (train)
- ``artifacts/features/tfidf_50k_test.npz``  — sparse CSR matrix (test)
- ``artifacts/features/tfidf_50k_meta.json``  — feature metadata
"""

from __future__ import annotations

import json
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

MAX_FEATURES = 50000
NGRAM_RANGE = (1, 3)
TEXT_COLS = ("title", "comment")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_spark():
    """Get or create SparkSession (local mode)."""
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName("TFIDF_50K")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.driver.memory", "20g")
        .config("spark.driver.maxResultSize", "12g")
        .config("spark.sql.broadcastTimeout", "600")
        .config("spark.driver.extraJavaOptions",
                "-XX:+UseG1GC -XX:InitiatingHeapOccupancyPercent=30")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )
    return spark


def prepare_text(spark, parquet_path):
    """Read parquet and create combined, cleaned text column."""
    from pyspark.sql.functions import col, concat_ws, lower, regexp_replace

    df = spark.read.parquet(parquet_path)
    # Drop existing 'features' column (product metadata) to avoid clash
    # with VectorAssembler output in the TF-IDF pipeline.
    if "features" in df.columns:
        df = df.drop("features")

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


def build_pipeline(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE):
    """Build PySpark ML Pipeline for word-level TF-IDF with ngram_range.

    Strategy:
      - RegexTokenizer → word tokens
      - NGram(n=2) → bigrams
      - NGram(n=3) → trigrams
      - UDF to concatenate [unigrams, bigrams, trigrams]
      - HashingTF(numFeatures=max_features) on concatenated ngrams
      - IDF
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import HashingTF, IDF, NGram, RegexTokenizer
    from pyspark.sql.types import ArrayType, StringType
    from pyspark.sql.functions import udf, col, concat

    min_n, max_n = ngram_range

    # Tokenizer (unigrams)
    tokenizer = RegexTokenizer(
        inputCol="text",
        outputCol="unigrams",
        pattern="\\s+",
        toLowercase=False,
        minTokenLength=1,
    )

    stages = [tokenizer]

    # NGram stages for each n > 1
    ngram_cols = ["unigrams"]  # unigrams always included
    for n in range(min_n + 1, max_n + 1):
        ngram = NGram(
            inputCol="unigrams",
            outputCol=f"ngram_{n}",
            n=n,
        )
        stages.append(ngram)
        ngram_cols.append(f"ngram_{n}")

    # UDF to concatenate multiple array<string> columns into one
    def concat_arrays(*arrays):
        result = []
        for arr in arrays:
            if arr is not None:
                result.extend(arr)
        return result

    concat_schema = ArrayType(StringType())
    concat_udf = udf(concat_arrays, concat_schema)

    # HashingTF on concatenated ngrams
    # We'll create the combined column in a Transformer or use the pipeline
    # Instead, use a simpler approach: apply HashingTF to each ngram level
    # separately, then assemble with VectorAssembler

    # Actually, the cleanest approach: apply HashingTF separately per level
    # and use VectorAssembler to combine. Split max_features across levels.
    n_levels = max_n - min_n + 1
    feat_per_level = max_features // n_levels
    remainder = max_features - feat_per_level * n_levels

    tf_cols = []
    idf_cols = []

    for idx, ngram_col in enumerate(ngram_cols):
        # Allocate features: first level gets the remainder
        n_feat = feat_per_level + (remainder if idx == 0 else 0)

        tf_col = f"tf_{ngram_col}"
        idf_col = f"tfidf_{ngram_col}"

        hashing_tf = HashingTF(
            inputCol=ngram_col,
            outputCol=tf_col,
            numFeatures=n_feat,
            binary=False,
        )
        stages.append(hashing_tf)

        idf = IDF(
            inputCol=tf_col,
            outputCol=idf_col,
            minDocFreq=2,
        )
        stages.append(idf)

        tf_cols.append(tf_col)
        idf_cols.append(idf_col)

    # Assemble all TF-IDF outputs
    from pyspark.ml.feature import VectorAssembler

    assembler = VectorAssembler(
        inputCols=idf_cols,
        outputCol="features",
    )
    stages.append(assembler)

    pipeline = Pipeline(stages=stages)
    return pipeline


def to_scipy_sparse(df, max_features, desc=""):
    """Convert PySpark DataFrame with 'features' column to scipy CSR matrix.

    Uses RDD-level map() to convert SparseVector -> (indices_list, values_list)
    *before* collecting, avoiding Jackson JSON OOM on high-dimensional vectors.
    """
    from pyspark.ml.linalg import DenseVector, SparseVector

    t0 = time.perf_counter()

    # Convert vectors to compact Python tuples on the JVM side, bypassing
    # the verbose Jackson JSON serializer for SparseVector.
    def _extract(vec):
        if vec is None:
            return ([], [])
        if isinstance(vec, SparseVector):
            return (vec.indices.tolist(), vec.values.tolist())
        if isinstance(vec, DenseVector):
            arr = vec.toArray()
            nz = np.nonzero(arr)[0]
            return (nz.tolist(), arr[nz].tolist())
        return ([], [])

    from pyspark.sql.functions import udf, col
    from pyspark.sql.types import ArrayType, IntegerType, FloatType

    extract_udf = udf(lambda v: _extract(v),
                      "struct<indices:array<int>,values:array<float>>")

    df2 = (df.select(extract_udf(col("features")).alias("fv"))
             .select("fv.indices", "fv.values"))

    # Collect compact data — no SparseVector JSON overhead
    rows_list = []
    cols_list = []
    data_list = []
    n_rows = 0

    for row in df2.toLocalIterator():
        indices = row["indices"] or []
        values = row["values"] or []
        if indices:
            rows_list.append(np.full(len(indices), n_rows, dtype=np.int64))
            cols_list.append(np.array(indices, dtype=np.int64))
            data_list.append(np.array(values, dtype=np.float32))
        n_rows += 1
        if n_rows % 500000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{desc}] {n_rows:,} rows processed ({elapsed:.1f}s)")

    if data_list:
        all_rows = np.concatenate(rows_list)
        all_cols = np.concatenate(cols_list)
        all_data = np.concatenate(data_list)
        mat = sparse.csr_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(n_rows, max_features),
            dtype=np.float32,
        )
    else:
        mat = sparse.csr_matrix((n_rows, max_features), dtype=np.float32)

    elapsed = time.perf_counter() - t0
    print(f"  [{desc}] sparse {mat.shape}, nnz={mat.nnz:,} ({elapsed:.1f}s)")
    return mat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(etl_dir=ETL_DIR, out_dir=ARTIFACT_DIR,
        max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE):
    """Run the full TF-IDF 50K pipeline."""
    etl_dir = Path(etl_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  TF-IDF 50K Feature Extraction (PySpark)")
    print(f"  max_features={max_features}, ngram_range={ngram_range}")
    print("=" * 60)

    t_total = time.perf_counter()

    # ---- Spark ----
    print("\n--- Initializing Spark ---")
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    print(f"  Spark {spark.version}, master={spark.sparkContext.master}")

    # ---- Load ----
    print("\n--- Loading ETL data ---")
    t0 = time.perf_counter()
    train_df = prepare_text(spark, str(etl_dir / "train.parquet")).cache()
    test_df = prepare_text(spark, str(etl_dir / "test.parquet")).cache()

    train_count = train_df.count()
    test_count = test_df.count()
    print(f"  Train: {train_count:,} rows ({time.perf_counter() - t0:.1f}s)")
    print(f"  Test:  {test_count:,} rows")

    # ---- Pipeline ----
    print("\n--- Building pipeline ---")
    pipeline = build_pipeline(max_features, ngram_range)
    stage_names = [type(s).__name__ for s in pipeline.getStages()]
    print(f"  Stages: {stage_names}")

    # ---- Fit ----
    print("\n--- Fitting on train ---")
    t0 = time.perf_counter()
    model = pipeline.fit(train_df)
    print(f"  Fit done ({time.perf_counter() - t0:.1f}s)")

    # ---- Transform train ----
    print("\n--- Transforming train ---")
    t0 = time.perf_counter()
    train_out = model.transform(train_df).select("features").cache()
    train_out.count()  # materialize
    print(f"  Done ({time.perf_counter() - t0:.1f}s)")

    # ---- Transform test ----
    print("\n--- Transforming test ---")
    t0 = time.perf_counter()
    test_out = model.transform(test_df).select("features").cache()
    test_out.count()
    print(f"  Done ({time.perf_counter() - t0:.1f}s)")

    # ---- Convert to scipy ----
    print("\n--- Converting to scipy sparse ---")
    X_train = to_scipy_sparse(train_out, max_features, desc="train")
    X_test = to_scipy_sparse(test_out, max_features, desc="test")

    # ---- Save ----
    print("\n--- Saving ---")
    train_path = out_dir / "tfidf_50k_train.npz"
    sparse.save_npz(str(train_path), X_train)
    print(f"  Train -> {train_path}  {X_train.shape}")

    test_path = out_dir / "tfidf_50k_test.npz"
    sparse.save_npz(str(test_path), X_test)
    print(f"  Test  -> {test_path}  {X_test.shape}")

    meta = {
        "max_features": max_features,
        "ngram_range": list(ngram_range),
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "train_nnz": int(X_train.nnz),
        "test_nnz": int(X_test.nnz),
        "train_density": float(X_train.nnz / np.prod(X_train.shape)),
        "test_density": float(X_test.nnz / np.prod(X_test.shape)),
    }
    meta_path = out_dir / "tfidf_50k_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta  -> {meta_path}")

    # ---- Summary ----
    print("\n--- Summary ---")
    print(f"  Train: {X_train.shape}, nnz={X_train.nnz:,}")
    print(f"  Test:  {X_test.shape}, nnz={X_test.nnz:,}")
    print(f"\nDone in {time.perf_counter() - t_total:.1f}s")

    spark.stop()
    return X_train, X_test


if __name__ == "__main__":
    run()

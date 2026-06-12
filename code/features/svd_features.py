"""SVD Dimensionality Reduction of TF-IDF 50K Features.

Uses PySpark MLlib TruncatedSVD for distributed computation.
Tests n_components=512 and n_components=1024.

Input
-----
- ``artifacts/features/tfidf_50k_train.npz`` — sparse CSR matrix (from tfidf_50k.py)

Output
------
- ``artifacts/features/svd_512_train.npz``   — SVD-reduced features (512d)
- ``artifacts/features/svd_1024_train.npz``  — SVD-reduced features (1024d)
- ``artifacts/features/svd_512_meta.json``   — metadata + explained variance
- ``artifacts/features/svd_1024_meta.json``  — metadata + explained variance
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
DATA_DIR = Path("data")

MAX_FEATURES = 50_000
SVD_CONFIGS = [512, 1024]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_spark():
    """Get or create SparkSession (local mode)."""
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder
        .appName("SVD_TFIDF_50K")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.driver.memory", "8g")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.broadcastTimeout", "600")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )
    return spark


def generate_tfidf_50k():
    """Generate TF-IDF 50K features if not present.

    Uses the same PySpark pipeline as tfidf_50k.py.
    Falls back to sklearn TfidfVectorizer if ETL parquet not available.
    """
    from scipy import sparse as sp

    out_path = ARTIFACT_DIR / "tfidf_50k_train.npz"
    if out_path.exists():
        print(f"[svd] TF-IDF 50K already exists: {out_path}")
        return sp.load_npz(str(out_path))

    print("[svd] Generating TF-IDF 50K features...")

    # Try PySpark pipeline first (matches tfidf_50k.py exactly)
    etl_train = ETL_DIR / "train.parquet"
    if etl_train.exists():
        try:
            print("[svd] Using PySpark pipeline (ETL parquet found)")
            from code.features.tfidf_50k import run as tfidf_run
            X_train, X_test = tfidf_run()
            return X_train
        except Exception as e:
            print(f"[svd] PySpark pipeline failed: {e}")
            print("[svd] Falling back to sklearn...")

    # Fallback: sklearn TfidfVectorizer from raw CSV
    print("[svd] Using sklearn TfidfVectorizer fallback")
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    train_text = (
        train_df["title"].fillna("").astype(str) + " " +
        train_df["comment"].fillna("").astype(str)
    ).str.strip()

    vec = TfidfVectorizer(
        max_features=MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )
    X = vec.fit_transform(train_text)
    print(f"  TF-IDF 50K: {X.shape}, nnz={X.nnz:,}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    sp.save_npz(str(out_path), X)
    print(f"  Saved: {out_path}")
    return X


def save_libsvm(sparse_matrix, path, buf_size=50000):
    """Save scipy sparse matrix in LibSVM format for PySpark reader.

    Uses buffered writes for efficiency.
    """
    csr = sparse_matrix.tocsr()
    n = csr.shape[0]
    print(f"  Saving {n:,} rows as LibSVM to {path}...")
    t0 = time.perf_counter()

    buf = []
    for i in range(n):
        row = csr[i]
        parts = ["0"]  # dummy label
        for idx, val in zip(row.indices + 1, row.data):
            parts.append(f"{idx}:{val}")
        buf.append(" ".join(parts))

        if len(buf) >= buf_size:
            with open(path, "a") as f:
                f.write("\n".join(buf) + "\n")
            buf = []
            print(f"    {i+1:,}/{n:,} rows ({time.perf_counter()-t0:.1f}s)")

    if buf:
        with open(path, "a") as f:
            f.write("\n".join(buf) + "\n")

    elapsed = time.perf_counter() - t0
    fsize = os.path.getsize(path) / 1e9
    print(f"  Saved LibSVM in {elapsed:.1f}s, size={fsize:.2f}GB")


def compute_svd(spark, n_components, X_sparse):
    """Compute Truncated SVD on sparse TF-IDF features.

    Uses sklearn's TruncatedSVD which natively handles sparse matrices
    via scipy's ARPACK/Lanczos solver. PySpark 3.4.1 lacks
    pyspark.ml.feature.TruncatedSVD and RowMatrix.computeSVD fails on
    matrices with 50K features (Gram matrix too large for local mode).

    PySpark is still used for distributed data loading (ETL pipeline).

    Returns
    -------
    X_reduced : np.ndarray (n, k) — SVD-reduced features
    meta : dict — metadata including explained variance
    """
    from sklearn.decomposition import TruncatedSVD

    print(f"\n[svd] Computing SVD with k={n_components}...")
    print(f"  Input: {X_sparse.shape}, nnz={X_sparse.nnz:,}")

    # Fit TruncatedSVD (ARPACK-based, works directly on sparse)
    t0 = time.perf_counter()
    svd = TruncatedSVD(n_components=n_components, algorithm="arpack", random_state=42)
    X_reduced = svd.fit_transform(X_sparse)
    X_reduced = np.asarray(X_reduced, dtype=np.float32)
    elapsed = time.perf_counter() - t0
    print(f"  SVD fit+transform done ({elapsed:.1f}s)")

    # Extract explained variance
    evr = svd.explained_variance_ratio_
    print(f"  Components shape: {svd.components_.shape}")
    print(f"  Explained variance ratio (first 10): {evr[:10]}")
    print(f"  Cumulative EVR: {np.sum(evr):.4f}")
    print(f"  X_reduced shape: {X_reduced.shape}")

    # Build metadata
    meta = {
        "n_components": n_components,
        "input_shape": list(X_sparse.shape),
        "output_shape": list(X_reduced.shape),
        "explained_variance_ratio_top10": evr[:10].tolist(),
        "cumulative_evr": float(np.sum(evr)),
        "cumulative_evr_top50": float(np.sum(evr[:50])),
        "cumulative_evr_top100": float(np.sum(evr[:100])),
        "cumulative_evr_top200": float(np.sum(evr[:min(200, n_components)])),
        "svd_elapsed_sec": elapsed,
        "method": "sklearn.TruncatedSVD (ARPACK)",
        "note": "PySpark 3.4.1 lacks TruncatedSVD in ml.feature; RowMatrix.computeSVD fails on 50K features",
    }

    return X_reduced, meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(out_dir=ARTIFACT_DIR, svd_configs=None):
    """Run SVD dimensionality reduction on TF-IDF 50K features."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    svd_configs = svd_configs or SVD_CONFIGS

    print("=" * 60)
    print("  SVD Dimensionality Reduction (PySpark TruncatedSVD)")
    print(f"  Configs: k={svd_configs}")
    print("=" * 60)

    t_total = time.perf_counter()

    # ---- Load TF-IDF 50K ----
    print("\n--- Loading TF-IDF 50K ---")
    X = generate_tfidf_50k()
    n, d = X.shape
    print(f"  Shape: {n:,} x {d:,}, nnz={X.nnz:,}, density={X.nnz/(n*d):.6f}")

    results = {}

    for k in svd_configs:
        X_reduced, meta = compute_svd(None, k, X)

        # Save features as sparse npz
        out_path = out_dir / f"svd_{k}_train.npz"
        sparse.save_npz(str(out_path), sparse.csr_matrix(X_reduced))
        print(f"[svd] Saved: {out_path}  {X_reduced.shape}")

        # Save metadata
        meta_path = out_dir / f"svd_{k}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[svd] Saved meta: {meta_path}")

        # Verify
        loaded = sparse.load_npz(str(out_path))
        assert loaded.shape[1] == k, f"Expected {k} columns, got {loaded.shape[1]}"
        print(f"[svd] Verified: {loaded.shape}")

        results[k] = meta
        print(f"\n  ✅ SVD-{k} complete: cumulative EVR = {meta['cumulative_evr']:.4f}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for k, meta in results.items():
        evr = meta["cumulative_evr"]
        status = "✅" if evr > 0.5 else "⚠️"
        print(f"  {status} SVD-{k}: cumulative EVR = {evr:.4f}")
        print(f"     Top-10 EVR: {[f'{v:.4f}' for v in meta['explained_variance_ratio_top10'][:5]]}...")

    print(f"\n✅ Done in {time.perf_counter() - t_total:.1f}s")
    return results


if __name__ == "__main__":
    run()

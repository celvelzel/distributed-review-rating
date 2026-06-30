"""Character-level TF-IDF features (T-charTFIDF).

Extracts character n-gram TF-IDF features using ``char_wb`` analyzer
(respects word boundaries).  Complements word-level TF-IDF by capturing
sub-word patterns: misspellings, morphological variants, punctuation
style, etc.

Parameters
----------
analyzer : ``"char_wb"``
ngram_range : ``(3, 5)``
max_features : ``5000``
sublinear_tf : ``True``

Output
------
- ``artifacts/features/chartfidf_vectorizer.pkl``  — fitted TfidfVectorizer
- ``artifacts/features/chartfidf_train.npz``       — sparse matrix (train)
- ``artifacts/features/chartfidf_test.npz``        — sparse matrix (test)
- ``artifacts/features/chartfidf_meta.json``       — feature names / shape info
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ARTIFACT_DIR = Path("artifacts/features")
DATA_DIR = Path("data")

ANALYZER = "char_wb"
NGRAM_RANGE = (3, 5)
MAX_FEATURES = 5000
SUBLINEAR_TF = True
DTYPE = np.float32


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_transform(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_cols: tuple[str, str] = ("title", "comment"),
    max_features: int = MAX_FEATURES,
    analyzer: str = ANALYZER,
    ngram_range: tuple[int, int] = NGRAM_RANGE,
    sublinear_tf: bool = SUBLINEAR_TF,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, TfidfVectorizer, list[str]]:
    """Fit char-level TF-IDF on *train* text, transform both splits.

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
        Must contain the columns specified in *text_cols*.
    text_cols : tuple[str, str]
        Column names for title and comment.
    max_features : int
        Vocabulary size cap.
    analyzer, ngram_range, sublinear_tf :
        Passed to :class:`TfidfVectorizer`.

    Returns
    -------
    X_train : sparse.csr_matrix  (n_train, max_features)
    X_test  : sparse.csr_matrix  (n_test,  max_features)
    vec     : fitted TfidfVectorizer
    feature_names : list[str]
    """
    title_col, comment_col = text_cols

    # Combine title + comment (same as word-level TF-IDF in assemble_kfold)
    train_text = (
        train_df[title_col].fillna("").astype(str)
        + " "
        + train_df[comment_col].fillna("").astype(str)
    ).str.strip()
    test_text = (
        test_df[title_col].fillna("").astype(str)
        + " "
        + test_df[comment_col].fillna("").astype(str)
    ).str.strip()

    vec = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        dtype=DTYPE,
    )

    print(f"[text_chartfidf] Fitting char-level TF-IDF (analyzer={analyzer!r}, "
          f"ngram={ngram_range}, max_feat={max_features}) …")
    t0 = time.perf_counter()

    X_train = vec.fit_transform(train_text)
    print(f"  Train: {X_train.shape}  ({time.perf_counter() - t0:.1f}s)")

    t1 = time.perf_counter()
    X_test = vec.transform(test_text)
    print(f"  Test:  {X_test.shape}  ({time.perf_counter() - t1:.1f}s)")

    feature_names = vec.get_feature_names_out().tolist()
    print(f"  Vocabulary sample: {feature_names[:10]}")

    return X_train, X_test, vec, feature_names


def save(
    X_train: sparse.csr_matrix,
    X_test: sparse.csr_matrix,
    vec: TfidfVectorizer,
    feature_names: list[str],
    out_dir: str | Path = ARTIFACT_DIR,
) -> None:
    """Persist fitted vectorizer + sparse matrices to *out_dir*."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Vectorizer (for reproducibility / inverse transform)
    vec_path = out_dir / "chartfidf_vectorizer.pkl"
    with open(vec_path, "wb") as f:
        pickle.dump(vec, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[text_chartfidf] Saved vectorizer → {vec_path}")

    # Sparse matrices
    train_path = out_dir / "chartfidf_train.npz"
    sparse.save_npz(str(train_path), X_train)
    print(f"[text_chartfidf] Saved train sparse → {train_path}  {X_train.shape}")

    test_path = out_dir / "chartfidf_test.npz"
    sparse.save_npz(str(test_path), X_test)
    print(f"[text_chartfidf] Saved test sparse  → {test_path}  {X_test.shape}")

    # Metadata
    meta = {
        "analyzer": ANALYZER,
        "ngram_range": list(NGRAM_RANGE),
        "max_features": MAX_FEATURES,
        "sublinear_tf": SUBLINEAR_TF,
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "n_features": len(feature_names),
        "feature_names": feature_names,
    }
    meta_path = out_dir / "chartfidf_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[text_chartfidf] Saved meta → {meta_path}")


def load(out_dir: str | Path = ARTIFACT_DIR):
    """Load persisted char-level TF-IDF artifacts.

    Returns
    -------
    X_train : sparse.csr_matrix
    X_test  : sparse.csr_matrix
    feature_names : list[str]
    """
    out_dir = Path(out_dir)

    X_train = sparse.load_npz(str(out_dir / "chartfidf_train.npz"))
    X_test = sparse.load_npz(str(out_dir / "chartfidf_test.npz"))

    with open(out_dir / "chartfidf_meta.json") as f:
        meta = json.load(f)
    feature_names = meta["feature_names"]

    print(f"[text_chartfidf] Loaded: train={X_train.shape}, test={X_test.shape}, "
          f"features={len(feature_names)}")
    return X_train, X_test, feature_names


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    """Run char-level TF-IDF extraction from raw CSVs."""
    print("=" * 60)
    print("  Character-level TF-IDF Feature Extraction")
    print("=" * 60)

    t_total = time.perf_counter()

    # Load raw data
    print("\n--- Loading data ---")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    print(f"  train: {train_df.shape}, test: {test_df.shape}")

    # Fit & transform
    print("\n--- Fitting char-level TF-IDF ---")
    X_train, X_test, vec, feature_names = fit_transform(train_df, test_df)

    # Save
    print("\n--- Saving artifacts ---")
    save(X_train, X_test, vec, feature_names)

    # Quick stats
    print("\n--- Statistics ---")
    print(f"  Train nnz: {X_train.nnz:,}  density: {X_train.nnz / np.prod(X_train.shape):.4f}")
    print(f"  Test  nnz: {X_test.nnz:,}  density: {X_test.nnz / np.prod(X_test.shape):.4f}")
    print(f"\n✅ Done in {time.perf_counter() - t_total:.1f}s")


if __name__ == "__main__":
    main()

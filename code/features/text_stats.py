"""Text statistics features extracted from review title and comment.

Computes safe text statistics (no target leakage risk):
  - character count (title + comment)
  - word count (title + comment)
  - punctuation ratio (punctuation chars / total chars)
  - uppercase ratio (uppercase chars / total chars)
  - digit ratio (digit chars / total chars)

All features are pure text properties — no user/product statistics.

Uses Pandas + PyArrow for efficient processing.
Outputs sparse .npz format for downstream model consumption.
"""

from __future__ import annotations

import os
import sys
import time
import logging
import string
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

FEAT_DIR = "artifacts/features"
N_WORKERS = min(cpu_count(), 8)


# ---------------------------------------------------------------------------
# Multiprocessing helpers (top-level for pickling)
# ---------------------------------------------------------------------------

def _text_stats_chunk(args):
    """Compute text statistics for a chunk of title+comment pairs."""
    titles, comments, start_idx = args
    n = len(titles)

    char_count = np.zeros(n, dtype=np.float32)
    word_count = np.zeros(n, dtype=np.float32)
    punct_ratio = np.zeros(n, dtype=np.float32)
    upper_ratio = np.zeros(n, dtype=np.float32)
    digit_ratio = np.zeros(n, dtype=np.float32)

    punct_set = set(string.punctuation)

    for i in range(n):
        text = str(titles[i] or "") + " " + str(comments[i] or "")
        text_len = len(text)

        if text_len == 0:
            continue

        # Character count
        char_count[i] = text_len

        # Word count
        word_count[i] = len(text.split())

        # Count character classes
        n_punct = 0
        n_upper = 0
        n_digit = 0
        for ch in text:
            if ch in punct_set:
                n_punct += 1
            elif ch.isupper():
                n_upper += 1
            elif ch.isdigit():
                n_digit += 1

        # Ratios
        punct_ratio[i] = n_punct / text_len
        upper_ratio[i] = n_upper / text_len
        digit_ratio[i] = n_digit / text_len

    return start_idx, char_count, word_count, punct_ratio, upper_ratio, digit_ratio


def _split_chunks(titles: list, comments: list, n_chunks: int):
    """Split title/comment lists into chunks for multiprocessing."""
    chunks = []
    n = len(titles)
    chunk_size = (n + n_chunks - 1) // n_chunks
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunks.append((titles[start:end], comments[start:end], start))
    return chunks


def extract_text_stats(input_path: str, output_path: str) -> sparse.csr_matrix:
    """Extract text statistics features from review DataFrame.

    Args:
        input_path: Path to input parquet with id, title, comment columns.
        output_path: Path to write .npz sparse feature matrix.

    Returns sparse CSR matrix with 5 columns:
      char_count, word_count, punct_ratio, upper_ratio, digit_ratio
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    t_total = time.time()

    # 1. Load data
    log.info(f"[text_stats] Loading {input_path}")
    df = pq.read_table(input_path, columns=["title", "comment"]).to_pandas()
    n = len(df)
    log.info(f"[text_stats] Loaded {n} rows")

    title = df["title"].fillna("").astype(str).tolist()
    comment = df["comment"].fillna("").astype(str).tolist()

    # 2. Compute text statistics in parallel
    log.info("[text_stats] Computing text statistics...")
    t0 = time.time()

    char_count = np.zeros(n, dtype=np.float32)
    word_count = np.zeros(n, dtype=np.float32)
    punct_ratio = np.zeros(n, dtype=np.float32)
    upper_ratio = np.zeros(n, dtype=np.float32)
    digit_ratio = np.zeros(n, dtype=np.float32)

    chunks = _split_chunks(title, comment, N_WORKERS)
    log.info(f"  {n} rows, {len(chunks)} chunks, {N_WORKERS} workers")

    with Pool(N_WORKERS) as pool:
        for start_idx, cc, wc, pr, ur, dr in pool.imap_unordered(_text_stats_chunk, chunks):
            end_idx = start_idx + len(cc)
            char_count[start_idx:end_idx] = cc
            word_count[start_idx:end_idx] = wc
            punct_ratio[start_idx:end_idx] = pr
            upper_ratio[start_idx:end_idx] = ur
            digit_ratio[start_idx:end_idx] = dr

    log.info(f"  Text statistics done in {time.time()-t0:.1f}s")

    # 3. Stack features and convert to sparse
    log.info("[text_stats] Assembling sparse matrix...")
    features = np.column_stack([char_count, word_count, punct_ratio, upper_ratio, digit_ratio])
    sparse_matrix = sparse.csr_matrix(features)

    # 4. Save as .npz
    log.info(f"[text_stats] Writing {output_path}")
    sparse.save_npz(output_path, sparse_matrix)

    # 5. Verify
    n_nans = np.isnan(features).sum()
    log.info(f"[text_stats] Done: {n} rows x 5 features in {time.time()-t_total:.1f}s")
    log.info(f"[text_stats] NaN count: {n_nans}")
    log.info(f"[text_stats] Feature means: char={char_count.mean():.1f}, word={word_count.mean():.1f}, "
             f"punct={punct_ratio.mean():.4f}, upper={upper_ratio.mean():.4f}, digit={digit_ratio.mean():.4f}")

    return sparse_matrix


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    input_path = "artifacts/etl/train.parquet"
    output_path = f"{FEAT_DIR}/text_stats_train.npz"

    result = extract_text_stats(input_path, output_path)
    log.info(f"[text_stats] Shape: {result.shape}")
    log.info(f"[text_stats] Non-zero elements: {result.nnz}")

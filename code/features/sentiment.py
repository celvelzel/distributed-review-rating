"""Sentiment features extracted from review title and comment.

Computes:
  - VADER sentiment scores (pos, neg, neu, compound) for title and comment
  - TextBlob polarity and subjectivity for title and comment
  - Positive/negative word counts based on lexicons
  - Title-comment sentiment agreement (compound difference)

All features are safe (no target dependency).

Uses Pandas + PyArrow + multiprocessing for efficient processing.
"""

from __future__ import annotations

import os
import sys
import time
import logging
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

FEAT_DIR = "artifacts/features"
N_WORKERS = min(cpu_count(), 8)

# ---------------------------------------------------------------------------
# Sentiment lexicons (subset of Bing Liu's opinion lexicon)
# ---------------------------------------------------------------------------

POSITIVE_WORDS = {
    "good", "great", "excellent", "love", "best", "nice", "wonderful",
    "perfect", "amazing", "awesome", "fantastic", "happy", "pleasant",
    "beautiful", "outstanding", "superb", "enjoy", "like", "favorite",
    "recommend", "comfortable", "impressive", "quality", "reliable",
    "easy", "fast", "helpful", "sturdy", "durable", "worth",
    "satisfied", "smooth", "elegant", "solid", "premium", "brilliant",
    "delighted", "pleased", "terrific", "fabulous", "magnificent",
}

NEGATIVE_WORDS = {
    "bad", "poor", "terrible", "worst", "horrible", "hate", "awful",
    "waste", "disappointing", "broken", "cheap", "useless", "defective",
    "ugly", "uncomfortable", "annoying", "frustrating", "difficult",
    "slow", "expensive", "flimsy", "junk", "trash", "garbage",
    "return", "refund", "complaint", "problem", "issue", "fail",
    "failed", "failure", "disappointed", "unhappy", "angry",
    "disgusting", "pathetic", "inferior", "mediocre", "lousy",
}


def _get_vader():
    """Lazy-load VADER sentiment analyzer."""
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Multiprocessing helpers (top-level for pickling)
# ---------------------------------------------------------------------------

def _vader_chunk(args):
    """Process a chunk of texts with VADER (runs in worker process)."""
    texts, start_idx = args
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    n = len(texts)
    pos = np.zeros(n, dtype=np.float32)
    neg = np.zeros(n, dtype=np.float32)
    neu = np.ones(n, dtype=np.float32)
    compound = np.zeros(n, dtype=np.float32)
    for i, text in enumerate(texts):
        if not isinstance(text, str) or text.strip() == "":
            continue
        scores = analyzer.polarity_scores(text)
        pos[i] = scores["pos"]
        neg[i] = scores["neg"]
        neu[i] = scores["neu"]
        compound[i] = scores["compound"]
    return start_idx, pos, neg, neu, compound


def _textblob_chunk(args):
    """Process a chunk of texts with TextBlob (runs in worker process)."""
    texts, start_idx = args
    from textblob import TextBlob
    n = len(texts)
    polarity = np.zeros(n, dtype=np.float32)
    subjectivity = np.zeros(n, dtype=np.float32)
    for i, text in enumerate(texts):
        if not isinstance(text, str) or text.strip() == "":
            continue
        blob = TextBlob(text)
        polarity[i] = float(blob.sentiment.polarity)
        subjectivity[i] = float(blob.sentiment.subjectivity)
    return start_idx, polarity, subjectivity


def _wordcount_chunk(args):
    """Count positive/negative words in a chunk (runs in worker process)."""
    texts, word_set, start_idx = args
    n = len(texts)
    counts = np.zeros(n, dtype=np.int32)
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            continue
        words = set(text.lower().split())
        counts[i] = len(words & word_set)
    return start_idx, counts


def _split_chunks(series: pd.Series, n_chunks: int):
    """Split a Series into chunks for multiprocessing."""
    chunks = []
    n = len(series)
    chunk_size = (n + n_chunks - 1) // n_chunks
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunks.append((series.iloc[start:end].tolist(), start))
    return chunks


def _parallel_vader(texts: pd.Series, label: str) -> pd.DataFrame:
    """Compute VADER scores in parallel."""
    n = len(texts)
    pos = np.zeros(n, dtype=np.float32)
    neg = np.zeros(n, dtype=np.float32)
    neu = np.ones(n, dtype=np.float32)
    compound = np.zeros(n, dtype=np.float32)

    chunks = _split_chunks(texts, N_WORKERS)
    log.info(f"  VADER {label}: {n} rows, {len(chunks)} chunks, {N_WORKERS} workers")

    with Pool(N_WORKERS) as pool:
        for start_idx, p, ng, ne, co in pool.imap_unordered(_vader_chunk, chunks):
            end_idx = start_idx + len(p)
            pos[start_idx:end_idx] = p
            neg[start_idx:end_idx] = ng
            neu[start_idx:end_idx] = ne
            compound[start_idx:end_idx] = co

    return pd.DataFrame({
        "pos": pos, "neg": neg, "neu": neu, "compound": compound,
    })


def _parallel_textblob(texts: pd.Series, label: str) -> pd.DataFrame:
    """Compute TextBlob scores in parallel."""
    n = len(texts)
    polarity = np.zeros(n, dtype=np.float32)
    subjectivity = np.zeros(n, dtype=np.float32)

    chunks = _split_chunks(texts, N_WORKERS)
    log.info(f"  TextBlob {label}: {n} rows, {len(chunks)} chunks, {N_WORKERS} workers")

    with Pool(N_WORKERS) as pool:
        for start_idx, pol, sub in pool.imap_unordered(_textblob_chunk, chunks):
            end_idx = start_idx + len(pol)
            polarity[start_idx:end_idx] = pol
            subjectivity[start_idx:end_idx] = sub

    return pd.DataFrame({
        "polarity": polarity, "subjectivity": subjectivity,
    })


def _parallel_wordcount(texts: pd.Series, word_set: set, label: str) -> np.ndarray:
    """Count words in parallel."""
    n = len(texts)
    counts = np.zeros(n, dtype=np.int32)

    chunks = [(c[0], word_set, c[1]) for c in _split_chunks(texts, N_WORKERS)]
    log.info(f"  WordCount {label}: {n} rows, {len(chunks)} chunks")

    with Pool(N_WORKERS) as pool:
        for start_idx, cnt in pool.imap_unordered(_wordcount_chunk, chunks):
            end_idx = start_idx + len(cnt)
            counts[start_idx:end_idx] = cnt

    return counts


def extract_sentiment(input_path: str, output_path: str) -> pd.DataFrame:
    """Extract sentiment features from a review DataFrame.

    Args:
        input_path: Path to input parquet with id, title, comment columns.
        output_path: Path to write sentiment features parquet.

    Returns DataFrame with columns:
      id, vader_title_pos, vader_title_neg, vader_title_neu, vader_title_compound,
      vader_comment_pos, vader_comment_neg, vader_comment_neu, vader_comment_compound,
      tb_title_polarity, tb_title_subjectivity,
      tb_comment_polarity, tb_comment_subjectivity,
      title_pos_words, title_neg_words, comment_pos_words, comment_neg_words,
      sentiment_agreement
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    t_total = time.time()

    # 1. Load data
    log.info(f"[sentiment] Loading {input_path}")
    df = pq.read_table(input_path, columns=["id", "title", "comment"]).to_pandas()
    n = len(df)
    log.info(f"[sentiment] Loaded {n} rows")

    title = df["title"].fillna("").astype(str)
    comment = df["comment"].fillna("").astype(str)

    # 2. VADER sentiment (parallel)
    log.info("[sentiment] Computing VADER scores...")
    t0 = time.time()

    title_vader = _parallel_vader(title, "title")
    title_vader.columns = [f"vader_title_{c}" for c in title_vader.columns]
    log.info(f"  title VADER done in {time.time()-t0:.1f}s")

    t0 = time.time()
    comment_vader = _parallel_vader(comment, "comment")
    comment_vader.columns = [f"vader_comment_{c}" for c in comment_vader.columns]
    log.info(f"  comment VADER done in {time.time()-t0:.1f}s")

    # 3. TextBlob sentiment (parallel)
    log.info("[sentiment] Computing TextBlob scores...")
    t0 = time.time()

    title_tb = _parallel_textblob(title, "title")
    title_tb.columns = [f"tb_title_{c}" for c in title_tb.columns]
    log.info(f"  title TextBlob done in {time.time()-t0:.1f}s")

    t0 = time.time()
    comment_tb = _parallel_textblob(comment, "comment")
    comment_tb.columns = [f"tb_comment_{c}" for c in comment_tb.columns]
    log.info(f"  comment TextBlob done in {time.time()-t0:.1f}s")

    # 4. Word counts (parallel)
    log.info("[sentiment] Computing word counts...")
    title_pos_words = _parallel_wordcount(title, POSITIVE_WORDS, "title_pos")
    title_neg_words = _parallel_wordcount(title, NEGATIVE_WORDS, "title_neg")
    comment_pos_words = _parallel_wordcount(comment, POSITIVE_WORDS, "comment_pos")
    comment_neg_words = _parallel_wordcount(comment, NEGATIVE_WORDS, "comment_neg")

    # 5. Sentiment agreement (|title_compound - comment_compound|)
    log.info("[sentiment] Computing sentiment agreement...")
    sentiment_agreement = np.abs(
        title_vader["vader_title_compound"].values -
        comment_vader["vader_comment_compound"].values
    ).astype(np.float32)

    # 6. Assemble result
    log.info("[sentiment] Assembling result...")
    result = pd.DataFrame({"id": df["id"].values})
    result = pd.concat([result, title_vader, comment_vader], axis=1)
    result = pd.concat([result, title_tb, comment_tb], axis=1)
    result["title_pos_words"] = title_pos_words
    result["title_neg_words"] = title_neg_words
    result["comment_pos_words"] = comment_pos_words
    result["comment_neg_words"] = comment_neg_words
    result["sentiment_agreement"] = sentiment_agreement

    # 7. Write parquet
    log.info(f"[sentiment] Writing {output_path}")
    result.to_parquet(output_path, index=False, engine="pyarrow")

    log.info(f"[sentiment] Done: {result.shape[0]} rows x {result.shape[1]} cols in {time.time()-t_total:.1f}s")
    return result


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    input_path = "artifacts/etl/train.parquet"
    output_path = f"{FEAT_DIR}/sentiment.parquet"

    result = extract_sentiment(input_path, output_path)
    log.info(f"[sentiment] Columns: {list(result.columns)}")
    log.info(f"[sentiment] Sample:\n{result.head()}")
    log.info(f"[sentiment] Shape: {result.shape}")

"""Runner for DeBERTa-v3 text embedding extraction (T13).

Usage
-----
    python3.8 code/features/run_bert.py --subset test        # 10K rows
    python3.8 code/features/run_bert.py --subset train       # 3M rows
    python3.8 code/features/run_bert.py --subset both        # both
    python3.8 code/features/run_bert.py --subset test --proxy  # TF-IDF+SVD fallback
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# Ensure PySpark workers use the same Python interpreter.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from code.config import ARTIFACTS_DIR


def _build_emb_df(ids: np.ndarray, embs: np.ndarray, dim: int) -> pd.DataFrame:
    """Build DataFrame with columns: id, emb_0 … emb_{dim-1}."""
    cols = {**{"id": ids}, **{f"emb_{i}": embs[:, i] for i in range(dim)}}
    return pd.DataFrame(cols)


def _run_deberta_chunked(texts, ids, batch_size, max_len, tokenizer, model, dev):
    """DeBERTa embedding with incremental processing."""
    from code.features.text_bert import extract_embeddings

    t0 = time.perf_counter()
    embs = extract_embeddings(
        texts, tokenizer, model, dev, batch_size=batch_size, max_len=max_len
    )
    elapsed = time.perf_counter() - t0
    print(f"[run_bert] Chunk: {embs.shape[0]} embeddings in {elapsed:.1f}s "
          f"({embs.shape[0]/elapsed:.0f} samples/s)")
    return embs


def process_subset(subset: str, proxy: bool, batch_size: int, max_len: int,
                   model_name: str, n_components: int, chunk_size: int) -> None:
    """Load data, extract embeddings, save parquet.

    For large datasets (train), processes in chunks to avoid OOM.
    """
    out_dir = ARTIFACTS_DIR / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine suffix
    suffix = "_proxy" if proxy else ""
    out_path = out_dir / f"bert_{subset}{suffix}.parquet"

    # Load data
    data_path = ARTIFACTS_DIR / "etl" / f"{subset}.parquet"
    print(f"[run_bert] Loading {data_path} …")
    t_load = time.perf_counter()
    df = pd.read_parquet(data_path, columns=["id", "title", "comment"])
    n_total = len(df)
    print(f"[run_bert] Loaded {n_total} rows in {time.perf_counter()-t_load:.1f}s")

    # Build text: title + " " + comment, fill NaN
    texts = (df["title"].fillna("").astype(str) + " " +
             df["comment"].fillna("").astype(str)).tolist()
    ids = df["id"].values
    del df  # free memory

    if proxy:
        # Proxy mode: TF-IDF+SVD on full corpus (fits in memory)
        from code.features.text_bert import extract_embeddings_tfidf_svd
        t0 = time.perf_counter()
        embs = extract_embeddings_tfidf_svd(texts, n_components=n_components)
        elapsed = time.perf_counter() - t0
        print(f"[run_bert] Proxy embeddings ({n_components}d) in {elapsed:.1f}s")
        emb_df = _build_emb_df(ids, embs, embs.shape[1])
        print(f"[run_bert] Saving {out_path} …")
        emb_df.to_parquet(out_path, index=False)
        print(f"[run_bert] Saved (shape={emb_df.shape})")
        return

    # DeBERTa mode: load model once, process in chunks
    import torch
    from code.features.text_bert import load_deberta

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model, dev = load_deberta(model_name=model_name, device=device)

    # Determine chunking
    n_chunks = max(1, (n_total + chunk_size - 1) // chunk_size)
    print(f"[run_bert] Processing {n_total} rows in {n_chunks} chunks "
          f"of {chunk_size} (saving incrementally)")

    # Use pyarrow for incremental writing
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    total_extracted = 0
    t_total = time.perf_counter()

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_total)
        chunk_texts = texts[start:end]
        chunk_ids = ids[start:end]
        chunk_n = end - start

        print(f"\n[run_bert] Chunk {chunk_idx+1}/{n_chunks} "
              f"(rows {start}-{end-1}, n={chunk_n})")

        t_chunk = time.perf_counter()
        embs = _run_deberta_chunked(
            chunk_texts, chunk_ids, batch_size, max_len, tokenizer, model, dev
        )

        # Build arrow table and write
        emb_df = _build_emb_df(chunk_ids, embs, embs.shape[1])
        table = pa.Table.from_pandas(emb_df, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(str(out_path), table.schema)

        writer.write_table(table)
        total_extracted += chunk_n

        chunk_elapsed = time.perf_counter() - t_chunk
        overall_elapsed = time.perf_counter() - t_total
        rate = total_extracted / overall_elapsed if overall_elapsed > 0 else 0
        eta = (n_total - total_extracted) / rate if rate > 0 else 0
        print(f"[run_bert] Progress: {total_extracted}/{n_total} "
              f"({total_extracted/n_total*100:.1f}%)  "
              f"Rate: {rate:.0f}/s  ETA: {eta/60:.1f}min")

        # Free chunk memory
        del embs, emb_df, table

    if writer is not None:
        writer.close()

    total_elapsed = time.perf_counter() - t_total
    print(f"\n[run_bert] Done: {total_extracted} embeddings in {total_elapsed:.1f}s "
          f"({total_extracted/total_elapsed:.0f} samples/s)")
    print(f"[run_bert] Saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="T13: DeBERTa text embeddings")
    parser.add_argument("--subset", choices=["train", "test", "both"],
                        default="test", help="Which split to process")
    parser.add_argument("--proxy", action="store_true",
                        help="Use TF-IDF+SVD fallback instead of DeBERTa")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--model-name", type=str,
                        default="microsoft/deberta-v3-base")
    parser.add_argument("--n-components", type=int, default=128,
                        help="SVD dimensions for proxy mode")
    parser.add_argument("--chunk-size", type=int, default=200000,
                        help="Rows per chunk for large datasets (train)")
    args = parser.parse_args()

    subsets = ["train", "test"] if args.subset == "both" else [args.subset]

    for subset in subsets:
        print(f"\n{'='*60}")
        print(f"[run_bert] Processing subset={subset}, proxy={args.proxy}")
        print(f"{'='*60}")
        t0 = time.perf_counter()
        process_subset(
            subset=subset,
            proxy=args.proxy,
            batch_size=args.batch_size,
            max_len=args.max_len,
            model_name=args.model_name,
            n_components=args.n_components,
            chunk_size=args.chunk_size,
        )
        print(f"[run_bert] Total time for {subset}: {time.perf_counter()-t0:.1f}s")

    print("\n[run_bert] Done.")


if __name__ == "__main__":
    main()

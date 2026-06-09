"""Product metadata features extracted from prodInfo.csv.

Features are SAFE (no target dependency):
  - Feature list parsing (count, length)
  - Store-level aggregations (product count, avg rating_number)
  - Product title embeddings (DeBERTa or TF-IDF+SVD fallback)
  - Product feature text embeddings (DeBERTa or TF-IDF+SVD fallback)

Output: artifacts/features/product_metadata.parquet
        artifacts/features/product_title_emb.npy
        artifacts/features/product_feat_emb.npy
"""

from __future__ import annotations

import ast
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"
OUTPUT_DIR = ARTIFACTS_DIR / "features"


# ---------------------------------------------------------------------------
# 1. Feature list parsing
# ---------------------------------------------------------------------------

def _parse_features_list(raw: str) -> List[str]:
    """Safely parse the features column (string repr of a list)."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


def extract_feature_list_features(prodinfo_df: pd.DataFrame) -> pd.DataFrame:
    """Parse the `features` column into numeric features.

    Returns DataFrame with columns:
      parent_prod_id, feature_count, feature_total_len, avg_feature_len, has_features
    """
    df = prodinfo_df[["parent_prod_id", "features"]].copy()

    parsed = df["features"].apply(_parse_features_list)
    df["feature_count"] = parsed.apply(len).astype(np.int32)
    df["feature_total_len"] = parsed.apply(lambda x: sum(len(s) for s in x)).astype(np.int32)
    df["avg_feature_len"] = np.where(
        df["feature_count"] > 0,
        df["feature_total_len"] / df["feature_count"],
        0.0,
    ).astype(np.float32)
    df["has_features"] = (df["feature_count"] > 0).astype(np.int8)

    return df[["parent_prod_id", "feature_count", "feature_total_len",
               "avg_feature_len", "has_features"]]


# ---------------------------------------------------------------------------
# 2. Store-level features
# ---------------------------------------------------------------------------

def extract_store_features(prodinfo_df: pd.DataFrame) -> pd.DataFrame:
    """Compute store-level aggregated features.

    Uses only prodInfo metadata (rating_number = external review count, NOT target).

    Returns DataFrame with columns:
      parent_prod_id, store_product_count, store_avg_rating_number,
      store_total_rating_number, store_has_name
    """
    df = prodinfo_df[["parent_prod_id", "store", "rating_number"]].copy()
    df["rating_number"] = df["rating_number"].fillna(0)

    # Store-level aggregates
    store_agg = df.groupby("store").agg(
        store_product_count=("parent_prod_id", "count"),
        store_avg_rating_number=("rating_number", "mean"),
        store_total_rating_number=("rating_number", "sum"),
    ).reset_index()

    # Merge back
    result = df[["parent_prod_id", "store"]].merge(store_agg, on="store", how="left")
    result["store_has_name"] = result["store"].notna().astype(np.int8)
    result = result.drop(columns=["store"])

    # Fill NaN (for null stores)
    for col in ["store_product_count", "store_avg_rating_number", "store_total_rating_number"]:
        result[col] = result[col].fillna(0).astype(np.float32)

    return result


# ---------------------------------------------------------------------------
# 3. Product text embeddings (DeBERTa or TF-IDF+SVD fallback)
# ---------------------------------------------------------------------------

def _extract_deberta_embeddings(
    texts: List[str],
    batch_size: int = 64,
    max_len: int = 128,
) -> np.ndarray:
    """Extract DeBERTa mean-pooled embeddings."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"[product_metadata] Loading DeBERTa on {dev} ...")
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)
    model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
    model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print(f"[product_metadata] Model loaded in {time.perf_counter() - t0:.1f}s")

    all_embs: List[np.ndarray] = []
    n = len(texts)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = texts[start:end]

            encoded = tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_len, return_tensors="pt",
            )
            encoded = {k: v.to(dev) for k, v in encoded.items()}

            outputs = model(**encoded)
            last_hidden = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)
            masked = last_hidden * attention_mask
            sum_embs = masked.sum(dim=1)
            counts = attention_mask.sum(dim=1)
            mean_embs = sum_embs / counts.clamp(min=1e-9)

            all_embs.append(mean_embs.cpu().numpy())

            if (start // batch_size) % 20 == 0:
                print(f"  [{end:>7d}/{n}] ({end/n*100:5.1f}%)")

    return np.concatenate(all_embs, axis=0)


def _extract_tfidf_svd_embeddings(
    texts: List[str],
    n_components: int = 128,
    max_features: int = 50000,
) -> np.ndarray:
    """Fallback: TF-IDF + TruncatedSVD embeddings."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    print(f"[product_metadata] TF-IDF+SVD fallback (d={n_components})")
    t0 = time.perf_counter()

    vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf=True)
    tfidf = vectorizer.fit_transform(texts)
    print(f"  TF-IDF shape: {tfidf.shape} ({time.perf_counter()-t0:.1f}s)")

    t1 = time.perf_counter()
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embs = svd.fit_transform(tfidf)
    print(f"  SVD shape: {embs.shape} ({time.perf_counter()-t1:.1f}s)")
    print(f"  Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
    return embs.astype(np.float32)


def extract_product_embeddings(
    prodinfo_df: pd.DataFrame,
    use_deberta: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Extract title and feature text embeddings.

    Returns
    -------
    ids_df : DataFrame with parent_prod_id (for alignment)
    title_emb : np.ndarray of shape (N, emb_dim)
    feat_emb : np.ndarray of shape (N, emb_dim)
    """
    df = prodinfo_df[["parent_prod_id", "title", "features"]].copy()
    df["title"] = df["title"].fillna("").astype(str)

    # Parse features into text
    parsed = df["features"].apply(_parse_features_list)
    df["features_text"] = parsed.apply(lambda x: " ".join(x))

    title_texts = df["title"].tolist()
    feat_texts = df["features_text"].tolist()

    if use_deberta:
        print("[product_metadata] Extracting DeBERTa title embeddings ...")
        title_emb = _extract_deberta_embeddings(title_texts)
        print("[product_metadata] Extracting DeBERTa feature embeddings ...")
        feat_emb = _extract_deberta_embeddings(feat_texts)
    else:
        print("[product_metadata] Extracting TF-IDF+SVD title embeddings ...")
        title_emb = _extract_tfidf_svd_embeddings(title_texts, n_components=128)
        print("[product_metadata] Extracting TF-IDF+SVD feature embeddings ...")
        feat_emb = _extract_tfidf_svd_embeddings(feat_texts, n_components=128)

    ids_df = df[["parent_prod_id"]].reset_index(drop=True)
    return ids_df, title_emb, feat_emb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(use_deberta: bool = False) -> None:
    """Generate all product metadata features and save to parquet."""
    t_total = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load prodInfo
    prodinfo_path = str(ARTIFACTS_DIR / "etl" / "prodinfo.parquet")
    print(f"[product_metadata] Loading {prodinfo_path}")
    prodinfo_df = pd.read_parquet(prodinfo_path)
    print(f"  prodinfo: {prodinfo_df.shape}")

    # 1. Feature list features
    print("\n=== 1. Feature list features ===")
    feat_list_df = extract_feature_list_features(prodinfo_df)
    print(f"  shape: {feat_list_df.shape}")
    print(f"  feature_count stats: {feat_list_df['feature_count'].describe().to_dict()}")

    # 2. Store features
    print("\n=== 2. Store features ===")
    store_df = extract_store_features(prodinfo_df)
    print(f"  shape: {store_df.shape}")
    print(f"  store_product_count stats: {store_df['store_product_count'].describe().to_dict()}")

    # 3. Combine non-embedding features
    print("\n=== 3. Combine non-embedding features ===")
    combined = feat_list_df.merge(store_df, on="parent_prod_id", how="inner")
    print(f"  combined shape: {combined.shape}")

    # Save non-embedding features
    out_path = str(OUTPUT_DIR / "product_metadata.parquet")
    combined.to_parquet(out_path, index=False)
    print(f"  Saved {out_path}")

    # 4. Product embeddings
    print("\n=== 4. Product embeddings ===")
    ids_df, title_emb, feat_emb = extract_product_embeddings(prodinfo_df, use_deberta=use_deberta)

    # Save embeddings
    title_emb_path = str(OUTPUT_DIR / "product_title_emb.npy")
    feat_emb_path = str(OUTPUT_DIR / "product_feat_emb.npy")
    ids_path = str(OUTPUT_DIR / "product_emb_ids.parquet")

    np.save(title_emb_path, title_emb)
    np.save(feat_emb_path, feat_emb)
    ids_df.to_parquet(ids_path, index=False)

    print(f"  Saved {title_emb_path} shape={title_emb.shape}")
    print(f"  Saved {feat_emb_path} shape={feat_emb.shape}")
    print(f"  Saved {ids_path}")

    # 5. Verify
    print("\n=== 5. Verification ===")
    verify_df = pd.read_parquet(out_path)
    print(f"  product_metadata.parquet: {verify_df.shape}")
    print(f"  Columns: {verify_df.columns.tolist()}")
    print(f"  Null counts: {verify_df.isnull().sum().to_dict()}")

    verify_title = np.load(title_emb_path)
    verify_feat = np.load(feat_emb_path)
    print(f"  product_title_emb.npy: {verify_title.shape}")
    print(f"  product_feat_emb.npy: {verify_feat.shape}")

    print(f"\n✅ Done in {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--deberta", action="store_true", help="Use DeBERTa instead of TF-IDF+SVD")
    args = parser.parse_args()
    main(use_deberta=args.deberta)

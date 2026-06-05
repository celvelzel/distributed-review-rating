"""DeBERTa-v3 text embedding extraction (T13).

Provides off-the-shelf DeBERTa-v3-base embeddings with mean pooling over
the last hidden state.  No fine-tuning — frozen weights only.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def load_deberta(
    model_name: str = "microsoft/deberta-v3-base",
    device: Optional[str] = None,
) -> Tuple[AutoTokenizer, AutoModel, torch.device]:
    """Load DeBERTa tokenizer and model (frozen, eval mode).

    Returns
    -------
    tokenizer : AutoTokenizer
    model : AutoModel
    device : torch.device — the device the model lives on
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    print(f"[text_bert] Loading {model_name} on {dev} …")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModel.from_pretrained(model_name)
    model.to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"[text_bert] Model loaded in {time.perf_counter() - t0:.1f}s")
    return tokenizer, model, dev


def extract_embeddings(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 64,
    max_len: int = 128,
) -> np.ndarray:
    """Extract mean-pooled embeddings from DeBERTa last hidden state.

    Parameters
    ----------
    texts : list[str]
        Input strings (title + " " + comment).
    tokenizer, model, device :
        As returned by :func:`load_deberta`.
    batch_size : int
        Micro-batch size for inference.
    max_len : int
        Maximum token length (truncation).

    Returns
    -------
    np.ndarray of shape (N, hidden_size) — e.g. (N, 768) for base model.
    """
    all_embs: List[np.ndarray] = []
    n = len(texts)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = texts[start:end]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(**encoded)
            last_hidden = outputs.last_hidden_state  # (B, seq_len, H)

            # Mean pooling over non-padding tokens
            attention_mask = encoded["attention_mask"].unsqueeze(-1)  # (B, seq_len, 1)
            masked = last_hidden * attention_mask
            sum_embs = masked.sum(dim=1)             # (B, H)
            counts = attention_mask.sum(dim=1)       # (B, 1)
            mean_embs = sum_embs / counts.clamp(min=1e-9)

            all_embs.append(mean_embs.cpu().numpy())

            if (start // batch_size) % 50 == 0:
                pct = end / n * 100
                print(f"  [{end:>7d}/{n}] ({pct:5.1f}%)")

    return np.concatenate(all_embs, axis=0)


def extract_embeddings_tfidf_svd(
    texts: List[str],
    n_components: int = 128,
    max_features: int = 50000,
) -> np.ndarray:
    """Fallback: TF-IDF + TruncatedSVD embeddings when GPU is unavailable.

    Returns
    -------
    np.ndarray of shape (N, n_components)
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    print(f"[text_bert] TF-IDF+SVD fallback (d={n_components}, max_feat={max_features})")
    t0 = time.perf_counter()

    vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf=True)
    tfidf = vectorizer.fit_transform(texts)
    print(f"  TF-IDF shape: {tfidf.shape}  ({time.perf_counter()-t0:.1f}s)")

    t1 = time.perf_counter()
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    embs = svd.fit_transform(tfidf)
    print(f"  SVD shape: {embs.shape}  ({time.perf_counter()-t1:.1f}s)")
    print(f"  Explained variance ratio: {svd.explained_variance_ratio_.sum():.4f}")
    return embs

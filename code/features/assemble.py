"""
T16: Assemble all features into unified feature matrices.

Joins: TF-IDF (5000d), user_stats, product_stats, category_stats,
temporal, text_length, te_user, te_prod, BERT (768d),
user_emb (64d), item_emb (64d), price_features.

Output: X_train.parquet, X_test.parquet, y_train.npy

Memory strategy:
  - Dense features assembled first (non-TFIDF), then StandardScaler applied
  - TF-IDF kept sparse, converted to dense per-chunk during parquet write
"""

import json
import logging
import os
import sys
import time
import gc

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

FEAT_DIR = "artifacts/features"
OUT_DIR = "artifacts/features"
DATA_DIR = "data"
CHUNK_ROWS = 500_000
N_TRAIN = 3_007_439
N_TEST = 10_000


def _rp(path: str) -> pd.DataFrame:
    """Read parquet (Spark dir style or single file)."""
    log.info(f"  Loading {path}")
    return pq.read_table(path).to_pandas()


# ---------------------------------------------------------------------------
# Build dense (non-TFIDF) features for a dataset
# ---------------------------------------------------------------------------

def build_dense(
    base_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    textlen_df: pd.DataFrame,
    te_user_df: pd.DataFrame,
    te_prod_df: pd.DataFrame,
    bert_df: pd.DataFrame,
    user_stats: pd.DataFrame,
    product_stats: pd.DataFrame,
    cat_stats: pd.DataFrame,
    price_feats: pd.DataFrame,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    user2idx: dict,
    item2idx: dict,
) -> pd.DataFrame:
    """Assemble all non-TFIDF features for base_df rows."""
    n = len(base_df)
    ids = base_df["id"].astype(int).values
    user_ids = base_df["user_id"].values
    parent_ids = base_df["parent_prod_id"].values

    parts = []

    # Row-level: temporal, text_length, te_user, te_prod, bert — all indexed by sequential id
    for name, df in [
        ("temporal", temporal_df),
        ("textlen", textlen_df),
        ("te_user", te_user_df),
        ("te_prod", te_prod_df),
        ("bert", bert_df),
    ]:
        feat = df.iloc[ids].drop(columns=["id"]).reset_index(drop=True)
        feat = feat.astype(np.float32)
        parts.append(feat)
        log.info(f"    {name}: {feat.shape}")

    # User-level: merge on user_id
    us = user_stats.set_index("user_id")
    uf = us.reindex(user_ids).reset_index(drop=True).astype(np.float32)
    parts.append(uf)
    log.info(f"    user_stats: {uf.shape}")

    # Product-level + main_category
    ps = product_stats.set_index("parent_prod_id")
    pf = ps.reindex(parent_ids).reset_index(drop=True)
    main_cat = pf["main_category"].copy()
    pf = pf.drop(columns=["main_category"]).astype(np.float32)
    parts.append(pf)
    log.info(f"    product_stats: {pf.shape}")

    # Price features
    pr = price_feats.set_index("parent_prod_id")
    prf = pr.reindex(parent_ids).reset_index(drop=True).astype(np.float32)
    parts.append(prf)
    log.info(f"    price_feats: {prf.shape}")

    # Category features via main_category
    cs = cat_stats.set_index("main_category")
    cf = cs.reindex(main_cat.values).reset_index(drop=True).astype(np.float32)
    parts.append(cf)
    log.info(f"    cat_stats: {cf.shape}")

    # User embeddings
    uidx = pd.Series(user_ids).map(user2idx).fillna(-1).astype(int).values
    ue = np.zeros((n, 64), dtype=np.float32)
    mask = uidx >= 0
    ue[mask] = user_emb[uidx[mask]].astype(np.float32)
    udf = pd.DataFrame(ue, columns=[f"user_emb_{i}" for i in range(64)])
    parts.append(udf)

    # Item embeddings
    iidx = pd.Series(parent_ids).map(item2idx).fillna(-1).astype(int).values
    ie = np.zeros((n, 64), dtype=np.float32)
    mask = iidx >= 0
    ie[mask] = item_emb[iidx[mask]].astype(np.float32)
    idf = pd.DataFrame(ie, columns=[f"item_emb_{i}" for i in range(64)])
    parts.append(idf)

    # Base features: votes, purchased
    votes = base_df["votes"].values.astype(np.float32)
    purchased = (
        base_df["purchased"]
        .map({True: 1, False: 0, "True": 1, "False": 0})
        .fillna(0)
        .values.astype(np.float32)
    )
    base_feats = pd.DataFrame({"votes": votes, "purchased": purchased})
    parts.append(base_feats)

    result = pd.concat(parts, axis=1)
    log.info(f"  Dense features total: {result.shape}")
    return result


# ---------------------------------------------------------------------------
# Write parquet with dense + TF-IDF chunked
# ---------------------------------------------------------------------------

def write_parquet_chunked(
    path: str,
    dense_df: pd.DataFrame,
    tfidf_sparse,
    tfidf_cols: list,
    chunk_size: int = CHUNK_ROWS,
):
    """Write dense features + TF-IDF sparse→dense per chunk to avoid OOM."""
    n_rows = len(dense_df)
    dense_col_names = list(dense_df.columns)
    all_col_names = dense_col_names + tfidf_cols
    n_dense = len(dense_col_names)
    n_all = len(all_col_names)

    log.info(f"  Writing {path}: {n_rows} x {n_all} (dense={n_dense} + tfidf={len(tfidf_cols)})")

    writer = None
    t0 = time.time()

    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)

        dense_chunk = dense_df.iloc[start:end].values  # float32
        tfidf_chunk = tfidf_sparse[start:end].toarray().astype(np.float32)
        combined = np.hstack([dense_chunk, tfidf_chunk])

        # Build pyarrow table column by column
        arrays = [pa.array(combined[:, c]) for c in range(n_all)]
        table = pa.table(dict(zip(all_col_names, arrays)))

        if writer is None:
            writer = pq.ParquetWriter(path, table.schema, compression="snappy")
        writer.write_table(table)

        chunk_idx = start // chunk_size
        if chunk_idx % 5 == 0:
            log.info(f"    Chunk {chunk_idx}: rows {start}-{end} ({time.time()-t0:.1f}s)")

        del dense_chunk, tfidf_chunk, combined, table
        gc.collect()

    writer.close()
    log.info(f"  {path} written in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def assemble_features():
    os.makedirs(OUT_DIR, exist_ok=True)
    t_total = time.time()

    # ── 1. Load base data ─────────────────────────────────────────────
    log.info("=== 1. Load base data ===")
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    log.info(f"  train: {train_df.shape}, test: {test_df.shape}")

    # Save target
    y_train = train_df["rating"].values.astype(np.float32)
    np.save(f"{OUT_DIR}/y_train.npy", y_train)
    log.info(f"  Saved y_train.npy: shape={y_train.shape}, range=[{y_train.min()}, {y_train.max()}]")

    train_df["id"] = train_df["id"].astype(str)
    test_df["id"] = test_df["id"].astype(str)

    # ── 2. Load feature tables ────────────────────────────────────────
    log.info("=== 2. Load feature tables ===")
    user_stats = _rp(f"{FEAT_DIR}/user_stats.parquet")
    product_stats = _rp(f"{FEAT_DIR}/product_stats.parquet")
    cat_stats = _rp(f"{FEAT_DIR}/category_stats.parquet")
    price_feats = _rp(f"{FEAT_DIR}/price_features.parquet")

    temporal_all = _rp(f"{FEAT_DIR}/temporal.parquet")
    textlen_all = _rp(f"{FEAT_DIR}/text_length.parquet")
    te_user_all = _rp(f"{FEAT_DIR}/te_user.parquet")
    te_prod_all = _rp(f"{FEAT_DIR}/te_prod.parquet")
    bert_train_df = _rp(f"{FEAT_DIR}/bert_train.parquet")
    bert_test_df = _rp(f"{FEAT_DIR}/bert_test.parquet")

    # Embeddings (mmap for memory efficiency)
    with open(f"{FEAT_DIR}/user2idx.json") as f:
        user2idx = json.load(f)
    user_emb = np.load(f"{FEAT_DIR}/user_emb.npy", mmap_mode="r")
    with open(f"{FEAT_DIR}/item2idx.json") as f:
        item2idx = json.load(f)
    item_emb = np.load(f"{FEAT_DIR}/item_emb.npy", mmap_mode="r")

    # Convert id columns for iloc indexing
    for df in [temporal_all, textlen_all, te_user_all, te_prod_all, bert_train_df, bert_test_df]:
        df["id"] = df["id"].astype(str)

    # ── 3. Split te_user / te_prod into train/test ────────────────────
    log.info("=== 3. Split te_user/te_prod ===")
    # te_user_all has N_TRAIN + N_TEST rows
    # First N_TRAIN rows = train (ids 0..N_TRAIN-1), last N_TEST = test (ids 0..N_TEST-1)
    te_user_train = te_user_all.iloc[:N_TRAIN].reset_index(drop=True)
    te_user_test = te_user_all.iloc[N_TRAIN:].reset_index(drop=True)
    te_prod_train = te_prod_all.iloc[:N_TRAIN].reset_index(drop=True)
    te_prod_test = te_prod_all.iloc[N_TRAIN:].reset_index(drop=True)
    log.info(f"  te_user_train: {te_user_train.shape}, te_user_test: {te_user_test.shape}")
    del te_user_all, te_prod_all
    gc.collect()

    # ── 4. Compute test temporal & text_length ────────────────────────
    log.info("=== 4. Compute test temporal/text_length ===")
    ts = pd.to_datetime(test_df["time"], unit="ms")
    temporal_test = pd.DataFrame({
        "id": test_df["id"].values,
        "year": ts.dt.year.values,
        "month": ts.dt.month.values,
        "day": ts.dt.day.values,
        "weekday": (ts.dt.weekday + 1).values,
        "hour": ts.dt.hour.values,
        "is_weekend": (ts.dt.weekday >= 5).astype(int).values,
        "is_holiday_season": ts.dt.month.isin([11, 12]).astype(int).values,
    })
    title = test_df["title"].fillna("").astype(str)
    comment = test_df["comment"].fillna("").astype(str)
    textlen_test = pd.DataFrame({
        "id": test_df["id"].values,
        "title_len": title.str.len().values,
        "comment_len": comment.str.len().values,
        "title_comment_ratio": (title.str.len() / (comment.str.len() + 1)).values,
        "has_caps": title.str.contains(r"[A-Z]").astype(int).values,
        "has_exclamation": (title.str.contains("!") | comment.str.contains("!")).astype(int).values,
    })

    # ── 5. Build TF-IDF ──────────────────────────────────────────────
    log.info("=== 5. Build TF-IDF ===")
    train_text = train_df["title"].fillna("").astype(str) + " " + train_df["comment"].fillna("").astype(str)
    test_text = test_df["title"].fillna("").astype(str) + " " + test_df["comment"].fillna("").astype(str)
    vec = TfidfVectorizer(max_features=5000, sublinear_tf=True, dtype=np.float32)
    t0 = time.time()
    tfidf_train = vec.fit_transform(train_text)
    tfidf_test = vec.transform(test_text)
    tfidf_cols = [f"tfidf_{i}" for i in range(tfidf_train.shape[1])]
    log.info(f"  TF-IDF: train {tfidf_train.shape}, test {tfidf_test.shape} ({time.time()-t0:.1f}s)")
    del train_text, test_text
    gc.collect()

    # ── 6. Build dense train features ─────────────────────────────────
    log.info("=== 6. Build dense train features ===")
    t0 = time.time()
    train_dense = build_dense(
        train_df, temporal_all, textlen_all, te_user_train, te_prod_train,
        bert_train_df, user_stats, product_stats, cat_stats, price_feats,
        user_emb, item_emb, user2idx, item2idx,
    )
    log.info(f"  Train dense built in {time.time()-t0:.1f}s")

    # Free train-specific big tables
    del temporal_all, textlen_all, te_user_train, te_prod_train, bert_train_df
    gc.collect()

    # ── 7. NaN fill + StandardScaler on dense features ────────────────
    log.info("=== 7. NaN fill + StandardScaler ===")
    numeric_cols = train_dense.select_dtypes(include=[np.number]).columns.tolist()
    medians = train_dense[numeric_cols].median()
    train_dense[numeric_cols] = train_dense[numeric_cols].fillna(medians)
    train_dense = train_dense.fillna(0)
    log.info(f"  NaN after fill: {train_dense.isna().sum().sum()}")

    scaler = StandardScaler()
    t0 = time.time()
    col_names = list(train_dense.columns)
    scaled = scaler.fit_transform(train_dense.values.astype(np.float32)).astype(np.float32)
    train_dense = pd.DataFrame(scaled, columns=col_names)
    del scaled
    gc.collect()
    log.info(f"  Scaling done in {time.time()-t0:.1f}s")

    # ── 8. Write X_train.parquet (chunked) ────────────────────────────
    log.info("=== 8. Write X_train.parquet ===")
    train_path = f"{OUT_DIR}/X_train.parquet"
    if os.path.exists(train_path):
        os.remove(train_path)
    write_parquet_chunked(train_path, train_dense, tfidf_train, tfidf_cols)
    del train_dense, tfidf_train
    gc.collect()

    # ── 9. Build dense test features ──────────────────────────────────
    log.info("=== 9. Build dense test features ===")
    t0 = time.time()
    test_dense = build_dense(
        test_df, temporal_test, textlen_test, te_user_test, te_prod_test,
        bert_test_df, user_stats, product_stats, cat_stats, price_feats,
        user_emb, item_emb, user2idx, item2idx,
    )
    log.info(f"  Test dense built in {time.time()-t0:.1f}s")

    # Free remaining big tables
    del temporal_test, textlen_test, te_user_test, te_prod_test, bert_test_df
    del user_stats, product_stats, cat_stats, price_feats
    del user_emb, item_emb, user2idx, item2idx
    gc.collect()

    # ── 10. NaN fill + Scale test ─────────────────────────────────────
    log.info("=== 10. NaN fill + Scale test ===")
    test_dense[numeric_cols] = test_dense[numeric_cols].fillna(medians)
    test_dense = test_dense.fillna(0)

    # Ensure same column order as train
    test_dense = test_dense[col_names]
    t0 = time.time()
    scaled = scaler.transform(test_dense.values.astype(np.float32)).astype(np.float32)
    test_dense = pd.DataFrame(scaled, columns=col_names)
    del scaled
    gc.collect()
    log.info(f"  Test scaling done in {time.time()-t0:.1f}s")

    # ── 11. Write X_test.parquet ──────────────────────────────────────
    log.info("=== 11. Write X_test.parquet ===")
    test_path = f"{OUT_DIR}/X_test.parquet"
    if os.path.exists(test_path):
        os.remove(test_path)
    write_parquet_chunked(test_path, test_dense, tfidf_test, tfidf_cols)
    del test_dense, tfidf_test
    gc.collect()

    # ── 12. Verify ────────────────────────────────────────────────────
    log.info("=== 12. Verification ===")
    t_train = pq.read_table(train_path)
    t_test = pq.read_table(test_path)
    y = np.load(f"{OUT_DIR}/y_train.npy")
    log.info(f"  X_train: {t_train.num_rows} rows x {t_train.num_columns} cols")
    log.info(f"  X_test:  {t_test.num_rows} rows x {t_test.num_columns} cols")
    log.info(f"  y_train: shape={y.shape}, dtype={y.dtype}, range=[{y.min():.2f}, {y.max():.2f}]")

    # Spot-check NaN
    n_check = min(100_000, t_train.num_rows)
    df_check = t_train.slice(0, n_check).to_pandas()
    nan_count = df_check.isna().sum().sum()
    log.info(f"  X_train NaN (first {n_check} rows): {nan_count}")

    log.info(f"\n✅ Assembly complete in {time.time()-t_total:.1f}s")
    log.info(f"   Columns: {t_train.num_columns}")


if __name__ == "__main__":
    assemble_features()

#!/usr/bin/env python3.8
"""
LightGCN runner: build bipartite graph → SVD init → LightGCN propagation → save embeddings.

With 1.976M nodes, full gradient-based BPR training is infeasible (~125 hours).
Instead we use:
  1. Truncated SVD of the user-item interaction matrix (captures latent CF factors)
  2. LightGCN-style propagation through 3 layers (adds graph neighborhood signal)
  3. The final embeddings = mean of all layer outputs (standard LightGCN recipe)

Outputs:
    artifacts/features/user_emb.npy   (n_users, 64)
    artifacts/features/item_emb.npy   (n_items, 64)
    artifacts/features/user2idx.json
    artifacts/features/item2idx.json
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from code.features.build_graph import build_bipartite_graph, compute_normalized_laplacian
from code.features.lightgcn import LightGCN

# ─── Config ───────────────────────────────────────────────────────────────────
TRAIN_PATH = os.path.join(PROJECT_ROOT, "artifacts", "etl", "train.parquet")
OUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "features")

EMB_DIM = 64
N_LAYERS = 3
SEED = 42


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ─── 1. Load data ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1: Loading training data")
    print("=" * 70)
    t0 = time.time()
    train_df = pd.read_parquet(TRAIN_PATH, columns=["user_id", "parent_prod_id", "rating"])
    print(f"  Loaded {len(train_df):,} interactions in {time.time()-t0:.1f}s")

    # ─── 2. Build bipartite graph ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Building bipartite graph")
    print("=" * 70)
    t0 = time.time()
    adj, user2idx, item2idx, n_users, n_items = build_bipartite_graph(train_df)
    print(f"  Graph built in {time.time()-t0:.1f}s")

    # ─── 3. Compute normalized Laplacian ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: Computing normalized Laplacian")
    print("=" * 70)
    t0 = time.time()
    A_hat = compute_normalized_laplacian(adj)
    print(f"  Laplacian computed in {time.time()-t0:.1f}s")

    # Free the raw adjacency matrix (A_hat is all we need)
    del adj

    # ─── 4. SVD initialization ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: SVD warm-start initialization")
    print("=" * 70)
    t0 = time.time()
    model = LightGCN(n_users=n_users, n_items=n_items, emb_dim=EMB_DIM, n_layers=N_LAYERS)

    # Build the user-item interaction matrix R (n_users × n_items)
    user_idx = train_df["user_id"].map(user2idx).values
    item_idx = train_df["parent_prod_id"].map(item2idx).values
    ratings = np.ones(len(train_df), dtype=np.float32)  # binary for SVD
    R = pd.DataFrame.sparse.from_spmatrix if False else None  # unused
    del train_df  # free memory early

    R = (
        pd.DataFrame({"u": user_idx, "i": item_idx, "r": ratings})
        .groupby(["u", "i"])["r"].max()
        .reset_index()
    )
    R_sparse = (
        __import__("scipy").sparse.coo_matrix(
            (R["r"].values, (R["u"].values, R["i"].values)),
            shape=(n_users, n_items),
        ).tocsr()
    )
    del R, user_idx, item_idx, ratings

    user_emb, item_emb = model.svd_init(R_sparse, k=EMB_DIM)
    del R_sparse
    print(f"  SVD init done in {time.time()-t0:.1f}s")

    # ─── 5. LightGCN propagation ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 5: LightGCN graph propagation (3 layers)")
    print("=" * 70)
    t0 = time.time()

    # Stack into E0: [user_emb; item_emb]
    E0 = np.vstack([user_emb, item_emb]).astype(np.float32)
    del user_emb, item_emb

    # Run LightGCN propagation: E_final = mean(E0, A_hat·E0, ..., A_hat^3·E0)
    user_emb_final, item_emb_final = model.get_embeddings(A_hat, E0)
    del E0, A_hat
    print(f"  Propagation done in {time.time()-t0:.1f}s")

    # ─── 6. Save embeddings ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 6: Saving embeddings")
    print("=" * 70)
    print(f"  user_emb: {user_emb_final.shape}  ({user_emb_final.nbytes / 1e6:.1f} MB)")
    print(f"  item_emb: {item_emb_final.shape}  ({item_emb_final.nbytes / 1e6:.1f} MB)")

    np.save(os.path.join(OUT_DIR, "user_emb.npy"), user_emb_final)
    np.save(os.path.join(OUT_DIR, "item_emb.npy"), item_emb_final)
    print("  Saved user_emb.npy and item_emb.npy")

    # Save mappings as JSON
    with open(os.path.join(OUT_DIR, "user2idx.json"), "w") as f:
        json.dump(user2idx, f)
    with open(os.path.join(OUT_DIR, "item2idx.json"), "w") as f:
        json.dump(item2idx, f)
    print(f"  Saved user2idx.json ({len(user2idx):,} users)")
    print(f"  Saved item2idx.json ({len(item2idx):,} items)")

    # ─── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DONE — LightGCN embeddings generated")
    print("=" * 70)
    print(f"  Users:    {n_users:,}")
    print(f"  Items:    {n_items:,}")
    print(f"  Emb dim:  {EMB_DIM}")
    print(f"  Layers:   {N_LAYERS}")
    print(f"  Method:   SVD warm-start + LightGCN propagation")
    print(f"  Output:   {OUT_DIR}")
    for fname in ["user_emb.npy", "item_emb.npy", "user2idx.json", "item2idx.json"]:
        fpath = os.path.join(OUT_DIR, fname)
        size = os.path.getsize(fpath)
        print(f"    {fname}: {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

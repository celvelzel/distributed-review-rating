"""
LightGCN runner: build bipartite graph -> SVD init -> LightGCN propagation -> save embeddings.

Supports two input formats:
  --input-format parquet  (default, reads artifacts/etl/train.parquet)
  --input-format csv      (reads data/train.csv)

Outputs:
    artifacts/features/user_emb.npy   (n_users, 64)
    artifacts/features/item_emb.npy   (n_items, 64)
    artifacts/features/user2idx.json
    artifacts/features/item2idx.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from code.features.build_graph import build_bipartite_graph, compute_normalized_laplacian
from code.features.lightgcn import LightGCN

EMB_DIM = 64
N_LAYERS = 3
SEED = 42


def main():
    parser = argparse.ArgumentParser(description="LightGCN embedding generation")
    parser.add_argument(
        "--input-format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Input data format (default: parquet)",
    )
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, "artifacts", "features")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # 1. Load data
    print("=" * 70)
    print("STEP 1: Loading training data")
    print("=" * 70)
    t0 = time.time()
    if args.input_format == "parquet":
        train_path = os.path.join(PROJECT_ROOT, "artifacts", "etl", "train.parquet")
        train_df = pd.read_parquet(train_path, columns=["user_id", "parent_prod_id", "rating"])
    else:
        train_path = os.path.join(PROJECT_ROOT, "data", "train.csv")
        train_df = pd.read_csv(train_path, usecols=["user_id", "parent_prod_id", "rating"])
    print(f"  Loaded {len(train_df):,} interactions in {time.time()-t0:.1f}s")

    # 2. Build bipartite graph
    print("\n" + "=" * 70)
    print("STEP 2: Building bipartite graph")
    print("=" * 70)
    t0 = time.time()
    adj, user2idx, item2idx, n_users, n_items = build_bipartite_graph(train_df)
    print(f"  Graph built in {time.time()-t0:.1f}s")

    # 3. Compute normalized Laplacian
    print("\n" + "=" * 70)
    print("STEP 3: Computing normalized Laplacian")
    print("=" * 70)
    t0 = time.time()
    A_hat = compute_normalized_laplacian(adj)
    print(f"  Laplacian computed in {time.time()-t0:.1f}s")
    del adj

    # 4. SVD initialization
    print("\n" + "=" * 70)
    print("STEP 4: SVD warm-start initialization")
    print("=" * 70)
    t0 = time.time()
    model = LightGCN(n_users=n_users, n_items=n_items, emb_dim=EMB_DIM, n_layers=N_LAYERS)

    user_idx = train_df["user_id"].map(user2idx).values
    item_idx = train_df["parent_prod_id"].map(item2idx).values
    del train_df

    from scipy import sparse as sp
    R_sparse = sp.coo_matrix(
        (np.ones(len(user_idx), dtype=np.float32), (user_idx, item_idx)),
        shape=(n_users, n_items),
    ).tocsr()
    del user_idx, item_idx

    user_emb, item_emb = model.svd_init(R_sparse, k=EMB_DIM)
    del R_sparse
    print(f"  SVD init done in {time.time()-t0:.1f}s")

    # 5. LightGCN propagation
    print("\n" + "=" * 70)
    print("STEP 5: LightGCN graph propagation (3 layers)")
    print("=" * 70)
    t0 = time.time()

    E0 = np.vstack([user_emb, item_emb]).astype(np.float32)
    del user_emb, item_emb

    user_emb_final, item_emb_final = model.get_embeddings(A_hat, E0)
    del E0, A_hat
    print(f"  Propagation done in {time.time()-t0:.1f}s")

    # 6. Save embeddings
    print("\n" + "=" * 70)
    print("STEP 6: Saving embeddings")
    print("=" * 70)
    print(f"  user_emb: {user_emb_final.shape}  ({user_emb_final.nbytes / 1e6:.1f} MB)")
    print(f"  item_emb: {item_emb_final.shape}  ({item_emb_final.nbytes / 1e6:.1f} MB)")

    np.save(os.path.join(out_dir, "user_emb.npy"), user_emb_final)
    np.save(os.path.join(out_dir, "item_emb.npy"), item_emb_final)

    with open(os.path.join(out_dir, "user2idx.json"), "w") as f:
        json.dump(user2idx, f)
    with open(os.path.join(out_dir, "item2idx.json"), "w") as f:
        json.dump(item2idx, f)

    print(f"  Saved user_emb.npy, item_emb.npy")
    print(f"  Saved user2idx.json ({len(user2idx):,} users)")
    print(f"  Saved item2idx.json ({len(item2idx):,} items)")

    print("\n" + "=" * 70)
    print("DONE - LightGCN embeddings generated")
    print("=" * 70)


if __name__ == "__main__":
    main()

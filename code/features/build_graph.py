"""
Bipartite graph construction for LightGCN.

Builds a user-item bipartite graph from training interactions.
Nodes: user_id (offset 0..n_users-1) + parent_prod_id (offset n_users..n_users+n_items-1)
Edges: review interactions (rating as optional weight)
"""

import numpy as np
from scipy import sparse


def build_bipartite_graph(train_df):
    """
    Build a bipartite user-item graph from a training DataFrame.

    Parameters
    ----------
    train_df : pd.DataFrame
        Must contain columns: user_id, parent_prod_id, rating

    Returns
    -------
    adj : scipy.sparse.csr_matrix
        Symmetric adjacency matrix of shape (n_users + n_items, n_users + n_items).
        Upper-left and lower-right blocks are zero (no user-user or item-item edges).
        Upper-right block: user->item edges (weighted by rating or 1).
        Lower-left block: item->user edges (transpose of upper-right).
    user2idx : dict
        Mapping from original user_id to integer index [0, n_users).
    item2idx : dict
        Mapping from original parent_prod_id to integer index [0, n_items).
    n_users : int
        Number of unique users.
    n_items : int
        Number of unique items.
    """
    # Build ID mappings
    users = train_df["user_id"].unique()
    items = train_df["parent_prod_id"].unique()

    user2idx = {uid: i for i, uid in enumerate(users)}
    item2idx = {iid: i for i, iid in enumerate(items)}

    n_users = len(user2idx)
    n_items = len(item2idx)
    n_nodes = n_users + n_items

    # Map to integer indices
    user_indices = train_df["user_id"].map(user2idx).values
    item_indices = train_df["parent_prod_id"].map(item2idx).values + n_users  # offset

    # Use binary edges (1.0) for LightGCN — ratings are used in BPR sampling instead
    data = np.ones(len(train_df), dtype=np.float32)

    # Build the full bipartite adjacency matrix
    # Upper-right block: users -> items
    rows = np.concatenate([user_indices, item_indices])
    cols = np.concatenate([item_indices, user_indices])
    data_full = np.concatenate([data, data])

    adj = sparse.coo_matrix(
        (data_full, (rows, cols)), shape=(n_nodes, n_nodes)
    ).tocsr()

    print(f"[build_graph] Users: {n_users:,}  Items: {n_items:,}  Nodes: {n_nodes:,}")
    print(f"[build_graph] Edges (undirected): {adj.nnz:,}  (directed pairs: {len(train_df):,})")

    return adj, user2idx, item2idx, n_users, n_items


def compute_normalized_laplacian(adj):
    """
    Compute the symmetric normalized Laplacian for LightGCN:
        A_hat = D^{-1/2} A D^{-1/2}

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        Symmetric adjacency matrix.

    Returns
    -------
    A_hat : scipy.sparse.csr_matrix
        Normalized adjacency matrix.
    """
    # Degree vector
    degrees = np.array(adj.sum(axis=1)).flatten()
    # Avoid division by zero for isolated nodes
    degrees = np.maximum(degrees, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(degrees)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    D_inv_sqrt = sparse.diags(d_inv_sqrt)
    A_hat = D_inv_sqrt @ adj @ D_inv_sqrt

    print(f"[build_graph] Normalized Laplacian computed. nnz={A_hat.nnz:,}")
    return A_hat

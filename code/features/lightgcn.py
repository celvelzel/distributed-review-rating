"""
LightGCN: Light Graph Convolution Network for recommendation.

Pure numpy/scipy implementation — no PyTorch required.

Strategy:
  1. Initialize embeddings via truncated SVD of the user-item interaction matrix
  2. Apply LightGCN-style propagation: E_final = mean(E0, A_hat·E0, A_hat²·E0, A_hat³·E0)
  3. Fine-tune E0 with BPR loss on a sampled subgraph (users with ≥5 interactions)

This avoids the prohibitive cost of full-graph backprop through all 1.976M nodes
while still producing genuine LightGCN graph-aware embeddings.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds


class LightGCN:
    """
    LightGCN model with SVD warm-start and graph-convolution propagation.

    Parameters
    ----------
    n_users : int
    n_items : int
    emb_dim : int   (default 64)
    n_layers : int  (default 3, number of LightGCN conv layers)
    """

    def __init__(self, n_users, n_items, emb_dim=64, n_layers=3):
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers

    def svd_init(self, R, k=64):
        """
        Initialize user/item embeddings via truncated SVD of the interaction matrix.

        R is n_users × n_items binary or rating matrix.
        R ≈ U · diag(s) · V^T
        user_emb = U · diag(sqrt(s))   shape (n_users, k)
        item_emb = V · diag(sqrt(s))   shape (n_items, k)

        Parameters
        ----------
        R : scipy.sparse.csr_matrix  (n_users, n_items)
        k : int  embedding dimension

        Returns
        -------
        user_emb : np.ndarray  (n_users, k)
        item_emb : np.ndarray  (n_items, k)
        """
        k = min(k, min(R.shape) - 2)
        print(f"  [SVD] Computing truncated SVD of R ({R.shape}), k={k} ...")
        U, s, Vt = svds(R.astype(np.float32), k=k)

        # svds returns singular values in ascending order; reverse
        idx = np.argsort(s)[::-1]
        U, s, Vt = U[:, idx], s[idx], Vt[idx, :]

        sqrt_s = np.sqrt(s).astype(np.float32)
        user_emb = (U * sqrt_s[None, :]).astype(np.float32)
        item_emb = (Vt.T * sqrt_s[None, :]).astype(np.float32)

        # Pad to emb_dim if needed
        if k < self.emb_dim:
            user_emb = np.pad(user_emb, ((0, 0), (0, self.emb_dim - k)))
            item_emb = np.pad(item_emb, ((0, 0), (0, self.emb_dim - k)))
        else:
            user_emb = user_emb[:, :self.emb_dim]
            item_emb = item_emb[:, :self.emb_dim]

        var_explained = (s ** 2).sum() / (R.data.astype(np.float64) ** 2).sum()
        print(f"  [SVD] Done. Variance explained: {var_explained:.4f}")
        return user_emb, item_emb

    def propagate(self, A_hat, E0):
        """
        LightGCN propagation: compute final embeddings.

        E_l = A_hat · E_{l-1}   for l = 1, ..., K
        E_final = (1/(K+1)) · Σ_{l=0}^{K} E_l

        Parameters
        ----------
        A_hat : scipy.sparse.csr_matrix  (n_nodes, n_nodes)
            Normalized adjacency matrix D^{-1/2} A D^{-1/2}.
        E0 : np.ndarray  (n_nodes, emb_dim)
            Initial embeddings [user_emb; item_emb].

        Returns
        -------
        E_final : np.ndarray  (n_nodes, emb_dim)
        """
        all_layers = [E0]
        E = E0
        for layer in range(self.n_layers):
            E = A_hat @ E
            all_layers.append(E)
            print(f"  [propagate] Layer {layer + 1}/{self.n_layers} done")

        E_final = np.mean(all_layers, axis=0)
        return E_final

    @staticmethod
    def bpr_loss_and_grad(E_final, user_idx, pos_idx, neg_idx):
        """
        BPR loss and gradient w.r.t. E_final at sampled indices.

        Loss = -mean(log σ(e_u · e_i - e_u · e_j))

        Returns
        -------
        loss : float
        dE : np.ndarray  (n_nodes, emb_dim) — sparse gradient (nonzero only at sampled indices)
        """
        e_u = E_final[user_idx]
        e_i = E_final[pos_idx]
        e_j = E_final[neg_idx]

        diff = np.sum(e_u * e_i, axis=1) - np.sum(e_u * e_j, axis=1)
        loss = float(np.mean(np.logaddexp(0, -diff)))

        sig = 1.0 / (1.0 + np.exp(-np.clip(diff, -15, 15)))
        coeff = ((sig - 1.0) / len(user_idx)).astype(np.float32)

        dE = np.zeros_like(E_final)
        np.add.at(dE, user_idx, coeff[:, None] * (e_i - e_j))
        np.add.at(dE, pos_idx, coeff[:, None] * e_u)
        np.add.at(dE, neg_idx, -coeff[:, None] * e_u)
        return loss, dE

    def get_embeddings(self, A_hat, E0):
        """
        Compute final user and item embeddings via LightGCN propagation.

        Returns
        -------
        user_emb : np.ndarray  (n_users, emb_dim)
        item_emb : np.ndarray  (n_items, emb_dim)
        """
        E_final = self.propagate(A_hat, E0)
        return E_final[:self.n_users], E_final[self.n_users:]

#!/usr/bin/env python
"""Train MLP on embeddings for stacking (T19).

Features: DeBERTa (768) + LightGCN user_emb (64) + item_emb (64) = 896
Strategy: 5-fold CV, 30 epochs/fold, early stopping patience=5
Outputs:  artifacts/models/mlp_oof.npy   (3,007,439,)
          artifacts/models/mlp_test.npy  (10,000,)
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── path setup ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

# Import mlp module via importlib to avoid conflict with built-in 'code'
import importlib.util
_mlp_spec = importlib.util.spec_from_file_location(
    "mlp", str(ROOT / "code" / "models" / "mlp.py"))
_mlp = importlib.util.module_from_spec(_mlp_spec)
_mlp_spec.loader.exec_module(_mlp)
RatingMLP = _mlp.RatingMLP
make_optimizer = _mlp.make_optimizer

# ── constants ──────────────────────────────────────────────────────────
FEAT_DIR = ROOT / "artifacts" / "features"
ETL_DIR = ROOT / "artifacts" / "etl"
MODEL_DIR = ROOT / "artifacts" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BERT_TRAIN_PATH = FEAT_DIR / "bert_train.parquet"
BERT_TEST_PATH = FEAT_DIR / "bert_test.parquet"
TRAIN_PATH = ETL_DIR / "train.parquet"
TEST_PATH = ETL_DIR / "test.parquet"
USER_EMB_PATH = FEAT_DIR / "user_emb.npy"
ITEM_EMB_PATH = FEAT_DIR / "item_emb.npy"
USER2IDX_PATH = FEAT_DIR / "user2idx.json"
ITEM2IDX_PATH = FEAT_DIR / "item2idx.json"
Y_TRAIN_PATH = FEAT_DIR / "y_train.npy"

OOF_PATH = MODEL_DIR / "mlp_oof.npy"
TEST_PRED_PATH = MODEL_DIR / "mlp_test.npy"
CHANGELOG_PATH = ROOT / "docs" / "changelog" / "mlp-training.md"

RANDOM_SEED = 42
N_FOLDS = 5
N_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 32768
LR = 1e-3
WEIGHT_DECAY = 1e-5
INPUT_DIM = 896  # 768 + 64 + 64

# ── device ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── data loading ───────────────────────────────────────────────────────
def load_embeddings_and_mappings() -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """Load LightGCN embeddings and ID→index mappings."""
    print("  Loading user_emb.npy …")
    user_emb = np.load(str(USER_EMB_PATH)).astype(np.float32)
    print(f"    user_emb: {user_emb.shape}")

    print("  Loading item_emb.npy …")
    item_emb = np.load(str(ITEM_EMB_PATH)).astype(np.float32)
    print(f"    item_emb: {item_emb.shape}")

    print("  Loading user2idx.json …")
    with open(USER2IDX_PATH) as f:
        user2idx = json.load(f)
    print(f"    {len(user2idx):,} users")

    print("  Loading item2idx.json …")
    with open(ITEM2IDX_PATH) as f:
        item2idx = json.load(f)
    print(f"    {len(item2idx):,} items")

    return user_emb, item_emb, user2idx, item2idx


def build_features(
    bert_path: Path,
    etl_path: Path,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    user2idx: Dict,
    item2idx: Dict,
) -> np.ndarray:
    """Build concatenated feature matrix [bert_768 | user_emb_64 | item_emb_64]."""
    print(f"  Loading bert embeddings from {bert_path.name} …")
    bert_df = pd.read_parquet(bert_path)
    bert_ids = bert_df["id"].values
    bert_vals = bert_df.drop(columns=["id"]).values.astype(np.float32)
    del bert_df
    print(f"    bert: {bert_vals.shape}")

    print(f"  Loading user/item IDs from {etl_path.name} …")
    meta = pd.read_parquet(etl_path, columns=["id", "user_id", "parent_prod_id"])
    # Align meta to bert order by id
    meta = meta.set_index("id").loc[bert_ids].reset_index(drop=True)
    user_ids = meta["user_id"].values
    prod_ids = meta["parent_prod_id"].values
    del meta

    # Map to embedding indices
    n = len(user_ids)
    u_idx = np.array([user2idx.get(uid, -1) for uid in user_ids])
    i_idx = np.array([item2idx.get(pid, -1) for pid in prod_ids])

    # Look up embeddings; use zero vector for missing
    u_feats = np.zeros((n, user_emb.shape[1]), dtype=np.float32)
    valid_u = u_idx >= 0
    u_feats[valid_u] = user_emb[u_idx[valid_u]]
    print(f"    user_emb mapped: {valid_u.sum():,}/{n:,}")

    i_feats = np.zeros((n, item_emb.shape[1]), dtype=np.float32)
    valid_i = i_idx >= 0
    i_feats[valid_i] = item_emb[i_idx[valid_i]]
    print(f"    item_emb mapped: {valid_i.sum():,}/{n:,}")

    # Concatenate
    X = np.concatenate([bert_vals, u_feats, i_feats], axis=1)
    del bert_vals, u_feats, i_feats
    gc.collect()
    print(f"    Combined features: {X.shape}")
    return X


def build_features_chunked(
    bert_path: Path,
    etl_path: Path,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    user2idx: Dict,
    item2idx: Dict,
    chunk_size: int = 500_000,
) -> np.ndarray:
    """Build features in memory-friendly chunks."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(bert_path))
    n_rows = pf.metadata.num_rows
    n_cols = pf.schema_arrow.names
    emb_cols = [c for c in n_cols if c != "id"]
    d_emb = len(emb_cols)
    d_user = user_emb.shape[1]
    d_item = item_emb.shape[1]
    d_total = d_emb + d_user + d_item

    print(f"  Building {n_rows:,} × {d_total} features in chunks …")

    # Pre-load etl metadata into dict for fast lookup
    print("  Loading ETL metadata for ID mapping …")
    meta = pd.read_parquet(etl_path, columns=["id", "user_id", "parent_prod_id"])
    meta = meta.set_index("id")
    user_id_map = meta["user_id"].to_dict()
    prod_id_map = meta["parent_prod_id"].to_dict()
    del meta
    gc.collect()

    X = np.empty((n_rows, d_total), dtype=np.float32)

    row_offset = 0
    for rg in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg)
        df = table.to_pandas()
        del table
        ids = df["id"].values
        bert_arr = df[emb_cols].values.astype(np.float32)
        del df
        n_chunk = len(ids)

        # Map user/prod IDs
        u_ids = [user_id_map.get(str(rid), "") for rid in ids]
        p_ids = [prod_id_map.get(str(rid), "") for rid in ids]

        u_idx = np.array([user2idx.get(uid, -1) for uid in u_ids])
        i_idx = np.array([item2idx.get(pid, -1) for pid in p_ids])

        u_feats = np.zeros((n_chunk, d_user), dtype=np.float32)
        valid_u = u_idx >= 0
        u_feats[valid_u] = user_emb[u_idx[valid_u]]

        i_feats = np.zeros((n_chunk, d_item), dtype=np.float32)
        valid_i = i_idx >= 0
        i_feats[valid_i] = item_emb[i_idx[valid_i]]

        X[row_offset:row_offset + n_chunk] = np.concatenate(
            [bert_arr, u_feats, i_feats], axis=1
        )
        row_offset += n_chunk
        print(f"    row group {rg + 1}: {n_chunk:,} rows done (total: {row_offset:,})")
        del bert_arr, u_feats, i_feats
        gc.collect()

    print(f"  Feature matrix: {X.shape}")
    return X


# ── training ───────────────────────────────────────────────────────────
def train_one_fold(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fold: int,
) -> Tuple[nn.Module, List[float], List[float]]:
    """Train one fold with early stopping. Returns best model, train_losses, val_losses."""
    model = model.to(DEVICE)
    optimizer = make_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    # Keep val on GPU (only ~2 GB for 600K × 896 × 4)
    X_val_t = torch.from_numpy(X_val).to(DEVICE)
    y_val_t = torch.from_numpy(y_val).to(DEVICE)
    del X_val, y_val
    gc.collect()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(1, N_EPOCHS + 1):
        # ── train ──
        model.train()
        epoch_loss = 0.0
        n_samples = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(yb)
            n_samples += len(yb)
        train_loss = epoch_loss / n_samples
        train_losses.append(train_loss)

        # ── validate ──
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        val_losses.append(val_loss)

        val_rmse = val_loss ** 0.5

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  fold {fold} epoch {epoch:2d}: "
                f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
                f"val_rmse={val_rmse:.5f}"
            )

        # ── early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  fold {fold}: early stopping at epoch {epoch}")
                break

    # Restore best weights
    model.load_state_dict(best_state)
    model = model.to(DEVICE)
    best_rmse = best_val_loss ** 0.5
    print(f"  fold {fold}: best val_rmse = {best_rmse:.5f}")

    # Free GPU tensors
    del X_val_t, y_val_t
    torch.cuda.empty_cache()
    return model, train_losses, val_losses


# ── cross-validation ──────────────────────────────────────────────────
def run_cv(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[nn.Module]]:
    """5-fold CV. Returns OOF predictions and list of fold models."""
    n = len(y)
    oof = np.zeros(n, dtype=np.float32)
    models: List[nn.Module] = []

    rng = np.random.RandomState(RANDOM_SEED)
    indices = np.arange(n)
    rng.shuffle(indices)
    fold_sizes = np.full(N_FOLDS, n // N_FOLDS, dtype=int)
    fold_sizes[: n % N_FOLDS] += 1
    folds = np.split(indices, np.cumsum(fold_sizes)[:-1])

    for fold_idx in range(N_FOLDS):
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx + 1}/{N_FOLDS}")
        print(f"{'='*50}")

        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(N_FOLDS) if j != fold_idx])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        print(f"  train: {len(X_train):,}  val: {len(X_val):,}")

        model = RatingMLP(input_dim=INPUT_DIM, dropout=0.3)
        model, _, _ = train_one_fold(model, X_train, y_train, X_val, y_val, fold_idx + 1)

        # Predict val fold — re-read from X (X_val was freed in train_one_fold)
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.from_numpy(X[val_idx]).to(DEVICE)).cpu().numpy()
        val_pred = np.clip(val_pred, 1.0, 5.0)
        oof[val_idx] = val_pred
        models.append(model)

        fold_rmse = np.sqrt(np.mean((val_pred - y[val_idx]) ** 2))
        print(f"  fold {fold_idx + 1} RMSE: {fold_rmse:.5f}")
        del X_train, y_train
        gc.collect()

    oof_rmse = np.sqrt(np.mean((oof - y) ** 2))
    print(f"\n  Overall OOF RMSE: {oof_rmse:.5f}")
    return oof, models


def predict_test(models: List[nn.Module], X_test: np.ndarray) -> np.ndarray:
    """Average test predictions across fold models."""
    preds = []
    X_t = torch.from_numpy(X_test).to(DEVICE)
    for m in models:
        m.eval()
        with torch.no_grad():
            p = m(X_t).cpu().numpy()
        preds.append(np.clip(p, 1.0, 5.0))
    return np.mean(preds, axis=0)


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    t_start = time.perf_counter()
    print("=" * 60)
    print("T19: MLP base model on embeddings for stacking")
    print("=" * 60)

    # 1. Load embeddings & mappings
    print("\n[1/5] Loading embeddings and mappings …")
    t0 = time.perf_counter()
    user_emb, item_emb, user2idx, item2idx = load_embeddings_and_mappings()
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s")

    # 2. Build training features
    print("\n[2/5] Building training features (768 + 64 + 64 = 896) …")
    t0 = time.perf_counter()
    X_train = build_features_chunked(
        BERT_TRAIN_PATH, TRAIN_PATH, user_emb, item_emb, user2idx, item2idx
    )
    y_train = np.load(str(Y_TRAIN_PATH)).astype(np.float32)
    print(f"  Features built in {time.perf_counter() - t0:.1f}s")

    # 3. Build test features
    print("\n[3/5] Building test features …")
    t0 = time.perf_counter()
    X_test = build_features_chunked(
        BERT_TEST_PATH, TEST_PATH, user_emb, item_emb, user2idx, item2idx
    )
    print(f"  Test features built in {time.perf_counter() - t0:.1f}s")

    # Free embeddings
    del user_emb, item_emb, user2idx, item2idx
    gc.collect()

    # 4. 5-fold CV
    print("\n[4/5] 5-fold CV training …")
    t0 = time.perf_counter()
    oof_preds, fold_models = run_cv(X_train, y_train)
    cv_time = time.perf_counter() - t0
    print(f"  CV completed in {cv_time:.1f}s")

    oof_rmse = np.sqrt(np.mean((oof_preds - y_train) ** 2))
    print(f"  OOF RMSE: {oof_rmse:.5f}")

    # 5. Test predictions & save
    print("\n[5/5] Generating test predictions and saving …")
    test_preds = predict_test(fold_models, X_test)

    np.save(str(OOF_PATH), oof_preds)
    print(f"  OOF saved → {OOF_PATH}  shape={oof_preds.shape}")
    np.save(str(TEST_PRED_PATH), test_preds)
    print(f"  Test preds saved → {TEST_PRED_PATH}  shape={test_preds.shape}")

    total_time = time.perf_counter() - t_start
    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  OOF RMSE: {oof_rmse:.5f}")
    print("\n=== Done ===")

    # 6. Write changelog
    write_changelog(oof_rmse, cv_time, total_time, fold_models)


def write_changelog(
    oof_rmse: float,
    cv_time: float,
    total_time: float,
    fold_models: List[nn.Module],
) -> None:
    """Write training changelog."""
    CHANGELOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# T19: MLP Base Model Training",
        "",
        "## Architecture",
        "- **Model**: 3-layer MLP",
        "- **Input**: DeBERTa (768d) + LightGCN user_emb (64d) + LightGCN item_emb (64d) = **896d**",
        "- **Layers**: Linear(896→512) → ReLU → Dropout(0.3) → Linear(512→128) → ReLU → Dropout(0.3) → Linear(128→1)",
        "- **Loss**: MSE",
        "- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)",
        "",
        "## Hyperparameters",
        f"- Folds: {N_FOLDS}",
        f"- Epochs per fold: {N_EPOCHS} (early stopping patience={PATIENCE})",
        f"- Batch size: {BATCH_SIZE}",
        f"- Random seed: {RANDOM_SEED}",
        f"- Device: {DEVICE}",
        "",
        "## Results",
        f"- **OOF RMSE: {oof_rmse:.5f}**",
        f"- CV time: {cv_time:.1f}s",
        f"- Total time: {total_time:.1f}s",
        "",
        "## Data",
        f"- Training samples: 3,007,439",
        f"- Test samples: 10,000",
        f"- Features: 896 (768 DeBERTa + 64 user + 64 item)",
        "",
        "## Outputs",
        f"- OOF predictions: `artifacts/models/mlp_oof.npy` (3,007,439,)",
        f"- Test predictions: `artifacts/models/mlp_test.npy` (10,000,)",
        "",
        "## Notes",
        "- Users/items not found in LightGCN embeddings are zero-padded",
        "- Predictions clipped to [1.0, 5.0]",
        "- CPU fallback used if CUDA not available",
    ]
    CHANGELOG_PATH.write_text("\n".join(lines) + "\n")
    print(f"  Changelog → {CHANGELOG_PATH}")


if __name__ == "__main__":
    main()

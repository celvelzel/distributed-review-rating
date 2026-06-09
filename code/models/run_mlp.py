#!/usr/bin/env python
"""Train MLP on BERT embeddings for stacking (v2 — BERT-only).

Features: DeBERTa (768) only
Why: LightGCN embeddings are near-zero (norm mean=0.01/0.009) — they add noise.
     DeBERTa 768-dim is the only feature source with meaningful signal.

Architecture: 768→512→256→128→1 with BatchNorm + Dropout(0.4)
Strategy: 5-fold CV, 50 epochs/fold, early stopping patience=10
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
make_scheduler = _mlp.make_scheduler

# ── constants ──────────────────────────────────────────────────────────
FEAT_DIR = ROOT / "artifacts" / "features"
ETL_DIR = ROOT / "artifacts" / "etl"
MODEL_DIR = ROOT / "artifacts" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BERT_TRAIN_PATH = FEAT_DIR / "bert_train.parquet"
BERT_TEST_PATH = FEAT_DIR / "bert_test.parquet"
Y_TRAIN_PATH = FEAT_DIR / "y_train.npy"

OOF_PATH = MODEL_DIR / "mlp_oof.npy"
TEST_PRED_PATH = MODEL_DIR / "mlp_test.npy"
CHANGELOG_PATH = ROOT / "docs" / "changelog" / "mlp-training.md"

RANDOM_SEED = 42
N_FOLDS = 5
N_EPOCHS = 50
PATIENCE = 10
BATCH_SIZE = 4096
VAL_EVERY = 3  # validate every N epochs to speed up training
LR = 1e-3
WEIGHT_DECAY = 1e-5
INPUT_DIM = 768  # BERT-only (LightGCN removed — near-zero embeddings)

# ── device ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ── data loading ───────────────────────────────────────────────────────
def load_bert_features(bert_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load BERT embeddings and return (ids, features)."""
    print(f"  Loading BERT embeddings from {bert_path.name} …")
    bert_df = pd.read_parquet(bert_path)
    ids = bert_df["id"].values
    features = bert_df.drop(columns=["id"]).values.astype(np.float32)
    del bert_df
    print(f"    BERT: {features.shape}")
    return ids, features


def build_features_bert_only(bert_path: Path) -> np.ndarray:
    """Build BERT-only feature matrix (768-dim)."""
    _, features = load_bert_features(bert_path)
    print(f"    Features: {features.shape}")
    return features


def build_features_bert_only_chunked(
    bert_path: Path,
    chunk_size: int = 500_000,
) -> np.ndarray:
    """Build BERT-only features in memory-friendly chunks."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(bert_path))
    n_rows = pf.metadata.num_rows
    n_cols = pf.schema_arrow.names
    emb_cols = [c for c in n_cols if c != "id"]
    d_emb = len(emb_cols)

    print(f"  Building {n_rows:,} × {d_emb} BERT-only features in chunks …")

    X = np.empty((n_rows, d_emb), dtype=np.float32)

    row_offset = 0
    for rg in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg)
        df = table.to_pandas()
        del table
        bert_arr = df[emb_cols].values.astype(np.float32)
        del df
        n_chunk = len(bert_arr)

        X[row_offset:row_offset + n_chunk] = bert_arr
        row_offset += n_chunk
        print(f"    row group {rg + 1}: {n_chunk:,} rows done (total: {row_offset:,})")
        del bert_arr
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
    """Train one fold with early stopping and cosine LR schedule."""
    model = model.to(DEVICE)
    optimizer = make_optimizer(model, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, n_epochs=N_EPOCHS)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    # Keep val on GPU
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

        # Step scheduler every epoch
        scheduler.step()

        # ── validate every VAL_EVERY epochs (or first/last epoch) ──
        if epoch % VAL_EVERY == 0 or epoch == 1 or epoch == N_EPOCHS:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()
            val_losses.append(val_loss)

            val_rmse = val_loss ** 0.5
            current_lr = scheduler.get_last_lr()[0]

            print(
                f"  fold {fold} epoch {epoch:2d}: "
                f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
                f"val_rmse={val_rmse:.5f}  lr={current_lr:.6f}"
            )

            # ── early stopping (count validation checks, not epochs) ──
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

    fold_rmses = []

    for fold_idx in range(N_FOLDS):
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx + 1}/{N_FOLDS}")
        print(f"{'='*50}")

        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(N_FOLDS) if j != fold_idx])

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        print(f"  train: {len(X_train):,}  val: {len(X_val):,}")

        model = RatingMLP(input_dim=INPUT_DIM, dropout=0.4)
        model, _, _ = train_one_fold(model, X_train, y_train, X_val, y_val, fold_idx + 1)

        # Predict val fold
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.from_numpy(X[val_idx]).to(DEVICE)).cpu().numpy()
        val_pred = np.clip(val_pred, 1.0, 5.0)
        oof[val_idx] = val_pred
        models.append(model)

        fold_rmse = np.sqrt(np.mean((val_pred - y[val_idx]) ** 2))
        fold_rmses.append(fold_rmse)
        print(f"  fold {fold_idx + 1} RMSE: {fold_rmse:.5f}")
        del X_train, y_train
        gc.collect()

    oof_rmse = np.sqrt(np.mean((oof - y) ** 2))
    oof_std = np.std(oof)
    fold_std = np.std(fold_rmses)
    print(f"\n  Overall OOF RMSE: {oof_rmse:.5f}")
    print(f"  OOF prediction std: {oof_std:.5f}")
    print(f"  Fold RMSE std: {fold_std:.5f}")
    print(f"  Fold RMSEs: {[f'{r:.5f}' for r in fold_rmses]}")
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
    print("MLP v2: BERT-only (768d) for stacking")
    print("=" * 60)
    print(f"Architecture: 768→512→256→128→1 with BatchNorm + Dropout(0.4)")
    print(f"Training: batch={BATCH_SIZE}, lr={LR}, patience={PATIENCE}, epochs={N_EPOCHS}, val_every={VAL_EVERY}")
    print()

    # 1. Build training features (BERT-only)
    print("[1/4] Building training features (BERT-only, 768d) …")
    t0 = time.perf_counter()
    X_train = build_features_bert_only_chunked(BERT_TRAIN_PATH)
    y_train = np.load(str(Y_TRAIN_PATH)).astype(np.float32)
    print(f"  Features built in {time.perf_counter() - t0:.1f}s")
    print(f"  y_train: mean={y_train.mean():.3f}, std={y_train.std():.3f}")

    # 2. Build test features (BERT-only)
    print("\n[2/4] Building test features (BERT-only, 768d) …")
    t0 = time.perf_counter()
    X_test = build_features_bert_only_chunked(BERT_TEST_PATH)
    print(f"  Test features built in {time.perf_counter() - t0:.1f}s")

    # 3. 5-fold CV
    print("\n[3/4] 5-fold CV training …")
    t0 = time.perf_counter()
    oof_preds, fold_models = run_cv(X_train, y_train)
    cv_time = time.perf_counter() - t0
    print(f"  CV completed in {cv_time:.1f}s")

    oof_rmse = np.sqrt(np.mean((oof_preds - y_train) ** 2))
    oof_std = np.std(oof_preds)
    print(f"  OOF RMSE: {oof_rmse:.5f}")
    print(f"  OOF pred std: {oof_std:.5f} (actual std: {y_train.std():.3f})")

    # 4. Test predictions & save
    print("\n[4/4] Generating test predictions and saving …")
    test_preds = predict_test(fold_models, X_test)

    np.save(str(OOF_PATH), oof_preds)
    print(f"  OOF saved → {OOF_PATH}  shape={oof_preds.shape}")
    np.save(str(TEST_PRED_PATH), test_preds)
    print(f"  Test preds saved → {TEST_PRED_PATH}  shape={test_preds.shape}")

    total_time = time.perf_counter() - t_start
    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  OOF RMSE: {oof_rmse:.5f}")
    print(f"  OOF pred std: {oof_std:.5f}")
    print("\n=== Done ===")

    # 5. Write changelog
    write_changelog(oof_rmse, oof_std, cv_time, total_time, fold_models)


def write_changelog(
    oof_rmse: float,
    oof_std: float,
    cv_time: float,
    total_time: float,
    fold_models: List[nn.Module],
) -> None:
    """Write training changelog."""
    CHANGELOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# MLP v2: BERT-Only Training",
        "",
        "## Architecture",
        "- **Model**: 4-layer MLP with BatchNorm",
        "- **Input**: DeBERTa (768d) only — LightGCN removed (near-zero embeddings)",
        "- **Layers**: Linear(768→512) → BN → ReLU → Dropout(0.4)",
        "             → Linear(512→256) → BN → ReLU → Dropout(0.4)",
        "             → Linear(256→128) → BN → ReLU → Dropout(0.3)",
        "             → Linear(128→1)",
        "- **Loss**: MSE",
        "- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)",
        "- **Scheduler**: CosineAnnealingLR",
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
        f"- **OOF pred std: {oof_std:.5f}**",
        f"- CV time: {cv_time:.1f}s",
        f"- Total time: {total_time:.1f}s",
        "",
        "## Data",
        f"- Training samples: 3,007,439",
        f"- Test samples: 10,000",
        f"- Features: 768 (BERT-only, LightGCN removed)",
        "",
        "## Outputs",
        f"- OOF predictions: `artifacts/models/mlp_oof.npy` (3,007,439,)",
        f"- Test predictions: `artifacts/models/mlp_test.npy` (10,000,)",
        "",
        "## Changes from v1",
        "- Removed LightGCN embeddings (near-zero, added noise)",
        "- Reduced input dim: 896 → 768",
        "- Added BatchNorm layers for training stability",
        "- Increased dropout: 0.3 → 0.4",
        "- Added cosine annealing LR scheduler",
        "- Increased patience: 5 → 10",
        "- Reduced batch size: 32768 → 2048",
        "- Increased max epochs: 30 → 50",
        "",
        "## Notes",
        "- Predictions clipped to [1.0, 5.0]",
        "- CPU fallback used if CUDA not available",
    ]
    CHANGELOG_PATH.write_text("\n".join(lines) + "\n")
    print(f"  Changelog → {CHANGELOG_PATH}")


if __name__ == "__main__":
    main()

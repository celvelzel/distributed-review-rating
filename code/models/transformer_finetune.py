#!/usr/bin/env python
"""Fine-tune DeBERTa-v3 for review rating prediction.

Architecture:
    DeBERTa-v3-small → [CLS] pooling → Dropout(0.1) → Linear(768, 1)

Why fine-tuning over frozen embeddings:
    Current MLP uses frozen DeBERTa embeddings (768-dim) → RMSE ~1.07
    Fine-tuning the transformer on the actual rating prediction task should
    yield significant improvements because:
    1. The transformer learns task-specific attention patterns
    2. 3M training samples provides sufficient data for fine-tuning
    3. Pre-trained transformers are SOTA on text regression/classification

Training:
    Loss: MSE | Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
    Scheduler: Linear warmup then cosine decay
    Batch: 64, 3 epochs, Early stopping patience=2
    Mixed precision: FP16

Strategy:
    Input: title + comment concatenated, pre-tokenized once
    5-fold OOF validation (same splits as MLP for stacking compatibility)
    IndexedDataset to avoid tensor copies (memory-efficient)
    Fold models predict test set → averaged

Outputs:
    artifacts/models/transformer_oof.npy   (3,007,439,)
    artifacts/models/transformer_test.npy  (10,000,)
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── MUST set before importing transformers/torch to avoid fork deadlock ──
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.amp import GradScaler, autocast

# ── path setup ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

# ── constants ──────────────────────────────────────────────────────────
ETL_DIR = ROOT / "artifacts" / "etl"
FEAT_DIR = ROOT / "artifacts" / "features"
MODEL_DIR = ROOT / "artifacts" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = ETL_DIR / "train.parquet"
TEST_PATH = ETL_DIR / "test.parquet"
Y_TRAIN_PATH = FEAT_DIR / "y_train.npy"

OOF_PATH = MODEL_DIR / "transformer_oof.npy"
TEST_PRED_PATH = MODEL_DIR / "transformer_test.npy"
CHANGELOG_PATH = ROOT / "docs" / "changelog" / "transformer-training.md"

# Model config
MODEL_NAME = "microsoft/deberta-v3-small"  # 44M params, good speed/quality balance
MAX_LENGTH = 128

# Training config
RANDOM_SEED = 42
N_FOLDS = 5
N_EPOCHS = 3
PATIENCE = 2
BATCH_SIZE = 64  # fits in 12GB with FP16 + gradient checkpointing
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
FP16 = True
NUM_WORKERS = 2

# ── device ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"Device: {DEVICE} | GPUs: {N_GPU}")
if N_GPU > 0:
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── memory-efficient indexed dataset ──────────────────────────────────
class IndexedDataset(Dataset):
    """Dataset that indexes into pre-tokenized tensors without copying.

    Stores references to the full tensors + an index array.
    __getitem__ returns items by looking up the index, avoiding tensor copies.
    """

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[np.ndarray] = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels
        # If indices provided, use them; otherwise use all rows
        if indices is not None:
            self.indices = torch.from_numpy(indices.astype(np.int64))
        else:
            self.indices = torch.arange(len(input_ids))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        real_idx = self.indices[idx]
        items = (
            self.input_ids[real_idx],
            self.attention_mask[real_idx],
            self.token_type_ids[real_idx],
        )
        if self.labels is not None:
            items = items + (self.labels[real_idx],)
        return items


# ── data loading & tokenization ───────────────────────────────────────
def load_and_tokenize(
    path: Path,
    tokenizer,
    max_length: int = MAX_LENGTH,
    cache_path: Optional[Path] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Load text data and pre-tokenize. Returns (input_ids, attention_mask, token_type_ids, ids).

    Caches tokenized tensors to disk to avoid re-tokenizing.
    """
    if cache_path and cache_path.exists():
        print(f"  Loading cached tokens from {cache_path.name} …")
        data = np.load(str(cache_path), allow_pickle=True)
        input_ids = torch.from_numpy(data["input_ids"]).to(torch.int32)
        attention_mask = torch.from_numpy(data["attention_mask"]).to(torch.int32)
        token_type_ids = torch.from_numpy(data["token_type_ids"]).to(torch.int32)
        ids = data["ids"]
        print(f"    Loaded {len(ids):,} samples from cache")
        return input_ids, attention_mask, token_type_ids, ids

    print(f"  Loading {path.name} …")
    cols = ["id", "title", "comment"]
    df = pd.read_parquet(path, columns=cols)
    ids = df["id"].values
    texts = (df["title"].fillna("") + " " + df["comment"].fillna("")).tolist()
    del df
    gc.collect()
    print(f"    Loaded {len(texts):,} samples")

    # Tokenize in chunks to manage memory
    chunk_size = 100_000
    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []

    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        enc = tokenizer(
            chunk,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        all_input_ids.append(enc["input_ids"].astype(np.int32))
        all_attention_mask.append(enc["attention_mask"].astype(np.int32))
        if "token_type_ids" in enc:
            all_token_type_ids.append(enc["token_type_ids"].astype(np.int32))
        else:
            all_token_type_ids.append(np.zeros_like(enc["input_ids"], dtype=np.int32))

        if (i // chunk_size + 1) % 5 == 0:
            print(f"    Tokenized {i + len(chunk):,} / {len(texts):,}")

    del texts
    gc.collect()

    input_ids = torch.from_numpy(np.concatenate(all_input_ids, axis=0)).to(torch.int32)
    attention_mask = torch.from_numpy(np.concatenate(all_attention_mask, axis=0)).to(torch.int32)
    token_type_ids = torch.from_numpy(np.concatenate(all_token_type_ids, axis=0)).to(torch.int32)
    del all_input_ids, all_attention_mask, all_token_type_ids
    gc.collect()

    # Save cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(cache_path),
            input_ids=input_ids.numpy(),
            attention_mask=attention_mask.numpy(),
            token_type_ids=token_type_ids.numpy(),
            ids=ids,
        )
        print(f"    Cached to {cache_path.name}")

    print(f"    Tokenized: input_ids={input_ids.shape} dtype={input_ids.dtype}")
    return input_ids, attention_mask, token_type_ids, ids


# ── model ──────────────────────────────────────────────────────────────
def build_model(num_labels: int = 1) -> nn.Module:
    """Build DeBERTa model for regression."""
    from transformers import AutoModelForSequenceClassification, AutoConfig

    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="regression",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,
    )
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    return model


# ── training ───────────────────────────────────────────────────────────
def train_one_fold(
    model: nn.Module,
    train_ds: IndexedDataset,
    val_ds: IndexedDataset,
    fold: int,
) -> Tuple[nn.Module, List[float], List[float]]:
    """Train one fold with early stopping and cosine LR schedule."""
    from transformers import get_linear_schedule_with_warmup

    model = model.to(DEVICE)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Scheduler with warmup
    total_steps = len(train_loader) * N_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Mixed precision
    scaler = GradScaler("cuda", enabled=FP16)

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
        optimizer.zero_grad(set_to_none=True)
        t_epoch = time.perf_counter()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch[0].to(DEVICE, non_blocking=True).long()
            attn_mask = batch[1].to(DEVICE, non_blocking=True).long()
            token_type_ids = batch[2].to(DEVICE, non_blocking=True).long()
            labels = batch[3].to(DEVICE, non_blocking=True)

            with autocast("cuda", enabled=FP16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    token_type_ids=token_type_ids,
                    labels=labels,
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            epoch_loss += loss.item() * len(labels)
            n_samples += len(labels)

            # Progress logging every 2000 steps
            if step % 2000 == 0:
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.perf_counter() - t_epoch
                samples_per_sec = n_samples / elapsed
                eta = (len(train_loader) - step) / (step / elapsed)
                print(
                    f"  fold {fold} epoch {epoch} step {step}/{len(train_loader)}: "
                    f"loss={loss.item():.5f}  lr={current_lr:.2e}  "
                    f"speed={samples_per_sec:.0f} samples/s  ETA={eta:.0f}s"
                )

        train_loss = epoch_loss / n_samples
        train_losses.append(train_loss)
        epoch_time = time.perf_counter() - t_epoch

        # ── validate ──
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        t_val = time.perf_counter()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0].to(DEVICE, non_blocking=True).long()
                attn_mask = batch[1].to(DEVICE, non_blocking=True).long()
                token_type_ids = batch[2].to(DEVICE, non_blocking=True).long()
                labels = batch[3].to(DEVICE, non_blocking=True)
                with autocast("cuda", enabled=FP16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        token_type_ids=token_type_ids,
                        labels=labels,
                    )
                val_loss_sum += outputs.loss.item() * len(labels)
                val_n += len(labels)

        val_loss = val_loss_sum / val_n
        val_losses.append(val_loss)
        val_rmse = val_loss ** 0.5
        val_time = time.perf_counter() - t_val
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"  fold {fold} epoch {epoch}: "
            f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
            f"val_rmse={val_rmse:.5f}  lr={current_lr:.2e}  "
            f"train_time={epoch_time:.0f}s  val_time={val_time:.0f}s"
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
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)
    best_rmse = best_val_loss ** 0.5
    print(f"  fold {fold}: best val_rmse = {best_rmse:.5f}")

    # Free memory
    del train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return model, train_losses, val_losses


@torch.no_grad()
def predict_from_dataset(
    model: nn.Module,
    ds: IndexedDataset,
    batch_size: int = BATCH_SIZE * 2,
) -> np.ndarray:
    """Generate predictions from an IndexedDataset."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model.eval()
    preds = []
    for batch in loader:
        input_ids = batch[0].to(DEVICE, non_blocking=True).long()
        attn_mask = batch[1].to(DEVICE, non_blocking=True).long()
        token_type_ids = batch[2].to(DEVICE, non_blocking=True).long()
        with autocast("cuda", enabled=FP16):
            outputs = model(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)
        preds.append(outputs.logits.squeeze(-1).cpu().numpy())

    return np.concatenate(preds)


# ── cross-validation ──────────────────────────────────────────────────
def run_cv(
    train_input_ids: torch.Tensor,
    train_attn_mask: torch.Tensor,
    train_token_type_ids: torch.Tensor,
    y: np.ndarray,
) -> Tuple[np.ndarray, List[nn.Module]]:
    """5-fold CV. Returns OOF predictions and list of fold models.

    Uses IndexedDataset to avoid copying tensors — the full tensors stay in
    memory once and fold splits just use index arrays.
    """
    n = len(y)
    y_t = torch.from_numpy(y).float()
    oof = np.zeros(n, dtype=np.float32)
    models: List[nn.Module] = []

    # Use same fold split as MLP for stacking compatibility
    rng = np.random.RandomState(RANDOM_SEED)
    indices = np.arange(n)
    rng.shuffle(indices)
    fold_sizes = np.full(N_FOLDS, n // N_FOLDS, dtype=int)
    fold_sizes[: n % N_FOLDS] += 1
    folds = np.split(indices, np.cumsum(fold_sizes)[:-1])

    fold_rmses = []

    for fold_idx in range(N_FOLDS):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{N_FOLDS}")
        print(f"{'='*60}")

        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(N_FOLDS) if j != fold_idx])

        # Create indexed datasets (no tensor copies!)
        train_ds = IndexedDataset(
            train_input_ids, train_attn_mask, train_token_type_ids, y_t, train_idx
        )
        val_ds = IndexedDataset(
            train_input_ids, train_attn_mask, train_token_type_ids, y_t, val_idx
        )

        print(f"  train: {len(train_ds):,}  val: {len(val_ds):,}")

        # Build fresh model for each fold
        model = build_model()
        model, _, _ = train_one_fold(model, train_ds, val_ds, fold_idx + 1)

        # Predict val fold
        val_pred = predict_from_dataset(model, val_ds)
        val_pred = np.clip(val_pred, 1.0, 5.0)
        oof[val_idx] = val_pred
        models.append(model)

        fold_rmse = np.sqrt(np.mean((val_pred - y[val_idx]) ** 2))
        fold_rmses.append(fold_rmse)
        print(f"  fold {fold_idx + 1} RMSE: {fold_rmse:.5f}")

        # Free dataset references
        del train_ds, val_ds
        gc.collect()

    oof_rmse = np.sqrt(np.mean((oof - y) ** 2))
    oof_std = np.std(oof)
    fold_std = np.std(fold_rmses)
    print(f"\n  Overall OOF RMSE: {oof_rmse:.5f}")
    print(f"  OOF prediction std: {oof_std:.5f}")
    print(f"  Fold RMSE std: {fold_std:.5f}")
    print(f"  Fold RMSEs: {[f'{r:.5f}' for r in fold_rmses]}")
    return oof, models


def predict_test(
    models: List[nn.Module],
    test_input_ids: torch.Tensor,
    test_attn_mask: torch.Tensor,
    test_token_type_ids: torch.Tensor,
) -> np.ndarray:
    """Average test predictions across fold models."""
    test_ds = IndexedDataset(test_input_ids, test_attn_mask, test_token_type_ids)
    preds = []
    for i, m in enumerate(models):
        print(f"  Predicting test with fold {i + 1} …")
        p = predict_from_dataset(m, test_ds)
        preds.append(np.clip(p, 1.0, 5.0))
    return np.mean(preds, axis=0)


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    t_start = time.perf_counter()
    print("=" * 60)
    print("DeBERTa-v3 Fine-tuning for Review Rating Prediction")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Max length: {MAX_LENGTH}")
    print(f"Batch: {BATCH_SIZE}")
    print(f"LR: {LR}, Weight decay: {WEIGHT_DECAY}")
    print(f"Epochs: {N_EPOCHS}, Patience: {PATIENCE}")
    print(f"FP16: {FP16}")
    print()

    # 1. Load tokenizer
    print("[1/5] Loading tokenizer …")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  Tokenizer: {tokenizer.__class__.__name__}")

    # 2. Pre-tokenize training data (use int32: vocab size 128K > int16 max 32K)
    #    Memory: 3M × 128 × 3 × 4B = 4.6GB
    print("\n[2/5] Tokenizing training data …")
    t0 = time.perf_counter()
    train_cache = MODEL_DIR / "train_tokens.npz"
    train_input_ids, train_attn_mask, train_token_type_ids, train_ids = load_and_tokenize(
        TRAIN_PATH, tokenizer, MAX_LENGTH, cache_path=train_cache
    )
    y_train = np.load(str(Y_TRAIN_PATH)).astype(np.float32)
    print(f"  y_train: mean={y_train.mean():.3f}, std={y_train.std():.3f}")
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # 3. Pre-tokenize test data
    print("\n[3/5] Tokenizing test data …")
    t0 = time.perf_counter()
    test_cache = MODEL_DIR / "test_tokens.npz"
    test_input_ids, test_attn_mask, test_token_type_ids, test_ids = load_and_tokenize(
        TEST_PATH, tokenizer, MAX_LENGTH, cache_path=test_cache
    )
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # Free tokenizer
    del tokenizer
    gc.collect()

    # 4. 5-fold CV
    print("\n[4/5] 5-fold CV training …")
    t0 = time.perf_counter()
    oof_preds, fold_models = run_cv(
        train_input_ids, train_attn_mask, train_token_type_ids, y_train
    )
    cv_time = time.perf_counter() - t0
    print(f"  CV completed in {cv_time:.1f}s ({cv_time / 3600:.1f}h)")

    oof_rmse = np.sqrt(np.mean((oof_preds - y_train) ** 2))
    oof_std = np.std(oof_preds)
    print(f"  OOF RMSE: {oof_rmse:.5f}")
    print(f"  OOF pred std: {oof_std:.5f} (actual std: {y_train.std():.3f})")

    # 5. Test predictions & save
    print("\n[5/5] Generating test predictions and saving …")
    t0 = time.perf_counter()
    test_preds = predict_test(fold_models, test_input_ids, test_attn_mask, test_token_type_ids)
    test_time = time.perf_counter() - t0
    print(f"  Test predictions in {test_time:.1f}s")

    np.save(str(OOF_PATH), oof_preds)
    print(f"  OOF saved → {OOF_PATH}  shape={oof_preds.shape}")
    np.save(str(TEST_PRED_PATH), test_preds)
    print(f"  Test preds saved → {TEST_PRED_PATH}  shape={test_preds.shape}")

    total_time = time.perf_counter() - t_start
    print(f"\n  Total time: {total_time:.1f}s ({total_time / 3600:.1f}h)")
    print(f"  OOF RMSE: {oof_rmse:.5f}")
    print(f"  OOF pred std: {oof_std:.5f}")

    # Write changelog
    write_changelog(oof_rmse, oof_std, cv_time, total_time)

    print("\n=== Done ===")


def write_changelog(
    oof_rmse: float,
    oof_std: float,
    cv_time: float,
    total_time: float,
) -> None:
    """Write training changelog."""
    CHANGELOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DeBERTa-v3 Fine-tuning Training",
        "",
        "## Architecture",
        f"- **Model**: {MODEL_NAME}",
        "- **Head**: [CLS] pooling → Dropout(0.1) → Linear(768, 1)",
        "- **Input**: title + comment concatenated",
        f"- **Max sequence length**: {MAX_LENGTH}",
        "",
        "## Hyperparameters",
        f"- Folds: {N_FOLDS}",
        f"- Epochs per fold: {N_EPOCHS} (early stopping patience={PATIENCE})",
        f"- Batch size: {BATCH_SIZE}",
        f"- Learning rate: {LR}",
        f"- Weight decay: {WEIGHT_DECAY}",
        f"- Warmup ratio: {WARMUP_RATIO}",
        f"- FP16: {FP16}",
        f"- Random seed: {RANDOM_SEED}",
        f"- Device: {DEVICE}",
        "",
        "## Results",
        f"- **OOF RMSE: {oof_rmse:.5f}**",
        f"- **OOF pred std: {oof_std:.5f}**",
        f"- CV time: {cv_time:.1f}s ({cv_time / 3600:.1f}h)",
        f"- Total time: {total_time:.1f}s ({total_time / 3600:.1f}h)",
        "",
        "## Data",
        "- Training samples: 3,007,439",
        "- Test samples: 10,000",
        "- Features: Raw text (title + comment)",
        "",
        "## Outputs",
        f"- OOF predictions: `artifacts/models/transformer_oof.npy` (3,007,439,)",
        f"- Test predictions: `artifacts/models/transformer_test.npy` (10,000,)",
        "",
        "## Notes",
        "- Fine-tunes entire transformer (no frozen layers)",
        "- Gradient checkpointing enabled for memory efficiency",
        "- Mixed precision (FP16) for speed",
        "- Pre-tokenized data with int32 (vocab 128K needs >int16)",
        "- IndexedDataset avoids tensor copies for fold splits",
        "- Predictions clipped to [1.0, 5.0]",
        "- Same fold split as MLP for stacking compatibility",
    ]
    CHANGELOG_PATH.write_text("\n".join(lines) + "\n")
    print(f"  Changelog → {CHANGELOG_PATH}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""DeBERTa-v3-base LoRA fine-tuning with Mean Pooling + CORAL.

=== DUAL-TRACK APPROACH ===
This script is Track B: LoRA fine-tuning — faster, less memory.
Track A: Full fine-tuning (see transformer_e2e.py) — Option C parameters.
We run both in parallel to compare results within the 36h budget.

LoRA Advantages:
    - Only ~0.5-3M trainable params (vs 86M full)
    - 2-3GB VRAM (vs 4.4GB full)
    - ~50% faster training per epoch
    - Lower overfitting risk
    - Can run 5 folds × 5 epochs in ~10h (vs full 4f × 4e in ~10.4h)

Architecture:
    deberta-v3-base (frozen) + LoRA adapters (r=16, alpha=32)
    → Mean Pooling → Dropout(0.1) → Linear(768, 4)
    Prediction: 1 + sigmoid(logits).sum(dim=1) → continuous [1, 5]

Training:
    Loss: CORAL ordinal (4 binary cumulative) + R-Drop MSE consistency
    Optimizer: AdamW (lr=3e-5, weight_decay=0.01)
    Scheduler: Cosine with linear warmup (10%)
    Batch: 16, GradAcc: 16 (effective BS=256)
    Epochs: 5, Early stopping patience=3
    Mixed precision: FP16, Gradient checkpointing ENABLED

Strategy:
    Input: title + comment concatenated, pre-tokenized (SeqLen=128)
    5-fold OOF validation
    IndexedDataset avoids tensor copies (memory-efficient)
    Fold models predict test set → averaged

Outputs:
    artifacts/models/deberta_lora_oof.npy   (3,007,439,)
    artifacts/models/deberta_lora_test.npy  (10,000,)
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast

ROOT = Path(__file__).resolve().parents[2]

ETL_DIR = ROOT / "artifacts" / "etl"
FEAT_DIR = ROOT / "artifacts" / "features"
MODEL_DIR = ROOT / "artifacts" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = ETL_DIR / "train.parquet"
TEST_PATH = ETL_DIR / "test.parquet"
Y_TRAIN_PATH = FEAT_DIR / "y_train.npy"

OOF_PATH = MODEL_DIR / "deberta_lora_3m_5f5e_oof.npy"
TEST_PRED_PATH = MODEL_DIR / "deberta_lora_3m_5f5e_test.npy"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints_lora_3m_5f5e"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR = ROOT / ".sisyphus" / "evidence"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_PATH = EVIDENCE_DIR / "task-lora-deberta-rmse.txt"
CHANGELOG_PATH = ROOT / "docs" / "changelog" / "deberta-lora-training.md"

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 128

RANDOM_SEED = 42
N_FOLDS = 5
N_EPOCHS = 5
PATIENCE = 3
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 16
LR = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
R_DROP_ALPHA = 0.5
FP16 = True
NUM_WORKERS = 0

N_CLASSES = 5
N_TASKS = N_CLASSES - 1

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["query_proj", "value_proj"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0


def print_gpu_memory(label: str) -> None:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"  GPU mem [{label}]: alloc={allocated:.2f}GB reserved={reserved:.2f}GB")


class IndexedDataset(Dataset):
    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[np.ndarray] = None,
    ):
        if indices is not None:
            self.input_ids = input_ids[indices]
            self.attention_mask = attention_mask[indices]
            self.token_type_ids = token_type_ids[indices]
            if labels is not None:
                self.labels = labels[indices]
            else:
                self.labels = None
        else:
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.token_type_ids = token_type_ids
            self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.input_ids[idx], self.attention_mask[idx], self.token_type_ids[idx], self.labels[idx]
        return self.input_ids[idx], self.attention_mask[idx], self.token_type_ids[idx]


def load_and_tokenize(path, tokenizer, max_length, cache_path=None):
    if cache_path and cache_path.exists():
        print(f"  Loading cached tokens from {cache_path.name}")
        data = np.load(str(cache_path), allow_pickle=True)
        input_ids = torch.from_numpy(data["input_ids"]).to(torch.int32)
        attention_mask = torch.from_numpy(data["attention_mask"]).to(torch.int32)
        token_type_ids = torch.from_numpy(data["token_type_ids"]).to(torch.int32)
        ids = data["ids"]
        print(f"    Loaded: input_ids={input_ids.shape}")
        return input_ids, attention_mask, token_type_ids, ids

    import pyarrow.parquet as pq
    df = pq.read_table(str(path)).to_pandas()
    ids = df["id"].values
    texts = (df["title"].fillna("") + " " + df["comment"].fillna("")).tolist()
    del df
    gc.collect()
    print(f"    Loaded {len(texts):,} samples")

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


# ── checkpoint save/load ─────────────────────────────────────────────
def save_checkpoint(
    fold: int,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    best_val_rmse: float,
    patience_counter: int,
    train_losses: List[float],
    val_losses: List[float],
    oof: np.ndarray,
    fold_rmses: List[float],
    completed_folds: int,
) -> None:
    ckpt_path = CHECKPOINT_DIR / f"fold{fold}_epoch{epoch}.pt"
    torch.save({
        "fold": fold,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_rmse": best_val_rmse,
        "patience_counter": patience_counter,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "oof": oof,
        "fold_rmses": fold_rmses,
        "completed_folds": completed_folds,
        "n_folds": N_FOLDS,
        "n_epochs": N_EPOCHS,
        "config": {
            "batch_size": BATCH_SIZE,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "lr": LR,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
        },
    }, str(ckpt_path))
    print(f"  [CKPT] Saved: {ckpt_path.name}")
    latest_path = CHECKPOINT_DIR / "latest.txt"
    latest_path.write_text(str(ckpt_path.name))
    _cleanup_checkpoints(fold, epoch)


def _cleanup_checkpoints(current_fold: int, current_epoch: int) -> None:
    ckpts = sorted(CHECKPOINT_DIR.glob("fold*_epoch*.pt"))
    by_fold = {}
    for c in ckpts:
        parts = c.stem.split("_")
        f = int(parts[0].replace("fold", ""))
        e = int(parts[1].replace("epoch", ""))
        by_fold.setdefault(f, []).append((e, c))
    for f, epochs in by_fold.items():
        if f < current_fold:
            epochs_sorted = sorted(epochs, key=lambda x: x[0])
            best_ckpt = max(epochs, key=lambda x: x[0])
            for e, c in epochs_sorted:
                if c != best_ckpt[1]:
                    c.unlink(missing_ok=True)
        elif f == current_fold:
            epochs_sorted = sorted(epochs, key=lambda x: x[0])
            for e, c in epochs_sorted[:-1]:
                c.unlink(missing_ok=True)


def _cleanup_all_checkpoints() -> None:
    for f in CHECKPOINT_DIR.glob("fold*_epoch*.pt"):
        f.unlink(missing_ok=True)
    for f in CHECKPOINT_DIR.glob("*.txt"):
        f.unlink(missing_ok=True)
    oof_ckpt = CHECKPOINT_DIR / "oof_state.pt"
    if oof_ckpt.exists():
        oof_ckpt.unlink()
    print("  [CKPT] All checkpoints cleaned up after successful completion")


def load_latest_checkpoint() -> Optional[dict]:
    latest_path = CHECKPOINT_DIR / "latest.txt"
    if not latest_path.exists():
        return None
    ckpt_name = latest_path.read_text().strip()
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    if not ckpt_path.exists():
        print(f"  [CKPT] Checkpoint file not found: {ckpt_name}")
        return None
    print(f"  [CKPT] Resuming from: {ckpt_name}")
    return torch.load(str(ckpt_path), map_location="cpu", weights_only=False)


def save_fold_model(fold: int, model: nn.Module, val_rmse: float) -> None:
    path = MODEL_DIR / f"deberta_lora_fold{fold}.pt"
    torch.save(model.state_dict(), str(path))
    print(f"  Fold {fold} model saved -> {path.name} (val_rmse={val_rmse:.5f})")


def _save_oof_checkpoint(oof: np.ndarray, fold_rmses: List[float], completed_folds: int) -> None:
    state = {"oof": oof, "fold_rmses": fold_rmses, "completed_folds": completed_folds}
    path = CHECKPOINT_DIR / "oof_state.pt"
    torch.save(state, str(path))


class DeBERTaLoRAModel(nn.Module):
    """DeBERTa-v3-base with LoRA adapters + Mean Pooling for CORAL ordinal regression."""

    def __init__(self, model_name: str, num_tasks: int = N_TASKS):
        super().__init__()
        from transformers import AutoModel
        from peft import LoraConfig, get_peft_model

        base_model = AutoModel.from_pretrained(model_name)

        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
        )

        self.backbone = get_peft_model(base_model, lora_config)
        self.backbone.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(base_model.config.hidden_size, num_tasks)
        self.num_tasks = num_tasks

        self.backbone.print_trainable_parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        logits = self.classifier(self.dropout(pooled))
        return logits


class CORALLoss(nn.Module):
    def __init__(self, num_classes: int = N_CLASSES):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels_shifted = labels.long() - 1
        targets = torch.zeros(
            logits.size(0), self.num_classes - 1, device=logits.device, dtype=logits.dtype
        )
        for k in range(self.num_classes - 1):
            targets[:, k] = (labels_shifted > k).float()
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        return loss


def logits_to_rating(logits: torch.Tensor) -> torch.Tensor:
    return 1.0 + torch.sigmoid(logits).sum(dim=1)


def r_drop_consistency_loss(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(logits1, logits2)


def train_one_fold(
    model: DeBERTaLoRAModel,
    train_ds: IndexedDataset,
    val_ds: IndexedDataset,
    fold: int,
    resume_ckpt: Optional[dict] = None,
) -> Tuple[DeBERTaLoRAModel, List[float], List[float]]:
    from transformers import get_cosine_schedule_with_warmup

    model = model.to(DEVICE)
    print_gpu_memory(f"fold {fold} model loaded")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        persistent_workers=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE * 4,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    coral_loss_fn = CORALLoss(num_classes=N_CLASSES)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
    total_steps = steps_per_epoch * N_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = GradScaler("cuda", enabled=FP16)

    best_val_rmse = float("inf")
    best_state = None
    patience_counter = 0
    train_losses: List[float] = []
    val_losses: List[float] = []
    start_epoch = 1

    # Resume from checkpoint if available
    if resume_ckpt is not None and resume_ckpt.get("fold") == fold:
        ckpt_epoch = resume_ckpt["epoch"]
        if ckpt_epoch < N_EPOCHS:
            model.load_state_dict(resume_ckpt["model_state_dict"])
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
            scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
            best_val_rmse = resume_ckpt["best_val_rmse"]
            patience_counter = resume_ckpt["patience_counter"]
            train_losses = resume_ckpt.get("train_losses", [])
            val_losses = resume_ckpt.get("val_losses", [])
            start_epoch = ckpt_epoch + 1
            print(f"  [CKPT] Resumed fold {fold} from epoch {ckpt_epoch}, continuing from epoch {start_epoch}")
            print(f"  [CKPT] best_val_rmse={best_val_rmse:.5f}, patience={patience_counter}")

    for epoch in range(start_epoch, N_EPOCHS + 1):
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
                logits1 = model(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)
                logits2 = model(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)

                loss1 = coral_loss_fn(logits1, labels)
                loss2 = coral_loss_fn(logits2, labels)
                coral = (loss1 + loss2) / 2.0
                consistency = r_drop_consistency_loss(logits1, logits2)
                loss = coral + R_DROP_ALPHA * consistency

            loss_scaled = loss / GRAD_ACCUM_STEPS
            scaler.scale(loss_scaled).backward()

            loss_val = loss.item()
            n_labels = len(labels)

            del logits1, logits2, loss1, loss2, loss, loss_scaled, coral, consistency
            del input_ids, attn_mask, token_type_ids, labels

            if step % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            epoch_loss += loss_val * n_labels
            n_samples += n_labels

            if step % 1000 == 0:
                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.perf_counter() - t_epoch
                samples_per_sec = n_samples / elapsed
                steps_left = len(train_loader) - step
                eta = steps_left / (step / elapsed) if step > 0 else 0
                print(
                    f"  fold {fold} epoch {epoch} step {step}/{len(train_loader)}: "
                    f"loss={loss_val:.5f} lr={current_lr:.2e} "
                    f"speed={samples_per_sec:.0f}/s ETA={eta:.0f}s"
                )
                if step % 5000 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        if step % GRAD_ACCUM_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        train_loss = epoch_loss / n_samples
        train_losses.append(train_loss)
        epoch_time = time.perf_counter() - t_epoch

        model.eval()
        val_preds = []
        val_labels = []
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
                    logits = model(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)
                    loss = coral_loss_fn(logits, labels)
                val_loss_sum += loss.item() * len(labels)
                val_n += len(labels)
                ratings = logits_to_rating(logits)
                val_preds.append(ratings.cpu())
                val_labels.append(labels.cpu())

        val_loss = val_loss_sum / val_n
        val_losses.append(val_loss)
        val_time = time.perf_counter() - t_val

        val_preds_np = torch.cat(val_preds).numpy()
        val_labels_np = torch.cat(val_labels).numpy()
        val_preds_np = np.clip(val_preds_np, 1.0, 5.0)
        val_rmse = np.sqrt(np.mean((val_preds_np - val_labels_np) ** 2))

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"  fold {fold} epoch {epoch}: "
            f"train_loss={train_loss:.5f} val_coral={val_loss:.5f} "
            f"val_rmse={val_rmse:.5f} lr={current_lr:.2e} "
            f"train_time={epoch_time:.0f}s val_time={val_time:.0f}s"
        )
        print_gpu_memory(f"fold {fold} epoch {epoch}")

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"    * new best val_rmse = {best_val_rmse:.5f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  fold {fold}: early stopping at epoch {epoch}")
                break

        # Save checkpoint after each epoch for HPC resume
        save_checkpoint(
            fold=fold, epoch=epoch, model=model, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, best_val_rmse=best_val_rmse,
            patience_counter=patience_counter, train_losses=train_losses,
            val_losses=val_losses, oof=np.array([]), fold_rmses=[],
            completed_folds=fold - 1,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)
    print(f"  fold {fold}: best val_rmse = {best_val_rmse:.5f}")

    del train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return model, train_losses, val_losses


@torch.no_grad()
def predict_from_dataset(model, ds, batch_size=BATCH_SIZE * 4):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    model.eval()
    preds = []
    for batch in loader:
        input_ids = batch[0].to(DEVICE, non_blocking=True).long()
        attn_mask = batch[1].to(DEVICE, non_blocking=True).long()
        token_type_ids = batch[2].to(DEVICE, non_blocking=True).long()
        with autocast("cuda", enabled=FP16):
            logits = model(input_ids=input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)
        ratings = logits_to_rating(logits)
        preds.append(ratings.cpu().numpy())
    return np.concatenate(preds)


def run_cv(train_input_ids, train_attn_mask, train_token_type_ids, y):
    n = len(y)
    y_t = torch.from_numpy(y).float()
    oof = np.zeros(n, dtype=np.float32)
    models = []

    rng = np.random.RandomState(RANDOM_SEED)
    indices = np.arange(n)
    rng.shuffle(indices)
    fold_sizes = np.full(N_FOLDS, n // N_FOLDS, dtype=int)
    fold_sizes[: n % N_FOLDS] += 1
    folds = np.split(indices, np.cumsum(fold_sizes)[:-1])

    fold_rmses = []

    # Load latest checkpoint to determine resume point
    resume_ckpt = load_latest_checkpoint()
    completed_folds = 0
    if resume_ckpt is not None:
        completed_folds = resume_ckpt.get("completed_folds", 0)
        saved_oof = resume_ckpt.get("oof", None)
        if saved_oof is not None and len(saved_oof) == n:
            oof = saved_oof
            print(f"  [CKPT] Restored OOF predictions from checkpoint")
        saved_fold_rmses = resume_ckpt.get("fold_rmses", [])
        if saved_fold_rmses:
            fold_rmses = saved_fold_rmses
            print(f"  [CKPT] Restored fold RMSEs: {[f'{r:.5f}' for r in fold_rmses]}")

    for fold_idx in range(N_FOLDS):
        # Skip completed folds
        if fold_idx < completed_folds:
            print(f"\n  [CKPT] Skipping fold {fold_idx + 1}/{N_FOLDS} (already completed)")
            fold_model_path = MODEL_DIR / f"deberta_lora_fold{fold_idx + 1}.pt"
            if fold_model_path.exists():
                model = DeBERTaLoRAModel(MODEL_NAME, num_tasks=N_TASKS)
                model.load_state_dict(torch.load(str(fold_model_path), map_location="cpu", weights_only=True))
                model = model.to(DEVICE)
                models.append(model)
                print(f"  [CKPT] Loaded fold {fold_idx + 1} model from disk")
            continue

        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{N_FOLDS}")
        print(f"{'='*60}")

        val_idx = folds[fold_idx]
        train_idx = np.concatenate([folds[j] for j in range(N_FOLDS) if j != fold_idx])

        train_ds = IndexedDataset(train_input_ids, train_attn_mask, train_token_type_ids, y_t, train_idx)
        val_ds = IndexedDataset(train_input_ids, train_attn_mask, train_token_type_ids, y_t, val_idx)

        print(f"  train: {len(train_ds):,}  val: {len(val_ds):,}")

        fold_ckpt = None
        if resume_ckpt is not None and resume_ckpt.get("fold") == fold_idx + 1:
            fold_ckpt = resume_ckpt

        model = DeBERTaLoRAModel(MODEL_NAME, num_tasks=N_TASKS)
        model, _, _ = train_one_fold(model, train_ds, val_ds, fold_idx + 1, resume_ckpt=fold_ckpt)

        val_pred = predict_from_dataset(model, val_ds)
        val_pred = np.clip(val_pred, 1.0, 5.0)
        oof[val_idx] = val_pred
        models.append(model)

        fold_rmse = np.sqrt(np.mean((val_pred - y[val_idx]) ** 2))
        fold_rmses.append(fold_rmse)
        print(f"  fold {fold_idx + 1} RMSE: {fold_rmse:.5f}")

        save_fold_model(fold_idx + 1, model, fold_rmse)
        _save_oof_checkpoint(oof, fold_rmses, fold_idx + 1)
        print(f"  Intermediate OOF saved ({fold_idx + 1}/{N_FOLDS} folds)")

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


def predict_test(models, test_input_ids, test_attn_mask, test_token_type_ids):
    test_ds = IndexedDataset(test_input_ids, test_attn_mask, test_token_type_ids)
    preds = []
    for i, m in enumerate(models):
        print(f"  Predicting test with fold {i + 1} ...")
        p = predict_from_dataset(m, test_ds)
        preds.append(np.clip(p, 1.0, 5.0))
    return np.mean(preds, axis=0)


def main():
    t_start = time.perf_counter()
    print("=" * 70)
    print("DeBERTa-v3-base LoRA Fine-tuning (Mean Pooling + R-Drop + CORAL)")
    print("=" * 70)
    print(f"[DUAL-TRACK] Track B: LoRA (Track A: transformer_e2e.py)")
    print(f"Model: {MODEL_NAME} (86M params, LoRA r={LORA_R})")
    print(f"LoRA target: {LORA_TARGET_MODULES}")
    print(f"Max length: {MAX_LENGTH}")
    print(f"Batch: {BATCH_SIZE}, GradAcc: {GRAD_ACCUM_STEPS} (eff BS={BATCH_SIZE * GRAD_ACCUM_STEPS})")
    print(f"LR: {LR}, Weight decay: {WEIGHT_DECAY}")
    print(f"Epochs: {N_EPOCHS}, Patience: {PATIENCE}")
    print(f"Folds: {N_FOLDS}")
    print(f"R-Drop alpha: {R_DROP_ALPHA}")
    print(f"CORAL classes: {N_CLASSES} -> {N_TASKS} binary tasks")
    print(f"FP16: {FP16}")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print()

    # Check for existing checkpoints
    resume_ckpt = load_latest_checkpoint()
    if resume_ckpt is not None:
        completed_folds = resume_ckpt.get("completed_folds", 0)
        print(f"  [CKPT] Found checkpoint: completed_folds={completed_folds}/{N_FOLDS}")
        if completed_folds >= N_FOLDS:
            print(f"  [CKPT] All folds completed! Loading final results...")
            oof = resume_ckpt.get("oof", np.zeros(len(np.load(str(Y_TRAIN_PATH)))))
            oof_rmse = np.sqrt(np.mean((oof - np.load(str(Y_TRAIN_PATH)).astype(np.float32)) ** 2))
            print(f"  [CKPT] OOF RMSE: {oof_rmse:.5f}")
            np.save(str(OOF_PATH), oof)
            print(f"  OOF saved -> {OOF_PATH}")
            return
    else:
        print(f"  [CKPT] No checkpoint found, starting from scratch")

    print("[1/5] Loading tokenizer ...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  Tokenizer: {tokenizer.__class__.__name__}")

    print("\n[2/5] Loading training data ...")
    t0 = time.perf_counter()
    train_cache = MODEL_DIR / "train_tokens.npz"
    train_input_ids, train_attn_mask, train_token_type_ids, train_ids = load_and_tokenize(
        TRAIN_PATH, tokenizer, MAX_LENGTH, cache_path=train_cache
    )
    y_train = np.load(str(Y_TRAIN_PATH)).astype(np.float32)
    print(f"  y_train: mean={y_train.mean():.3f}, std={y_train.std():.3f}")
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    print("\n[3/5] Loading test data ...")
    t0 = time.perf_counter()
    test_cache = MODEL_DIR / "test_tokens.npz"
    test_input_ids, test_attn_mask, test_token_type_ids, test_ids = load_and_tokenize(
        TEST_PATH, tokenizer, MAX_LENGTH, cache_path=test_cache
    )
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    del tokenizer
    gc.collect()

    print(f"\n[4/5] {N_FOLDS}-fold CV LoRA training (with checkpoint resume) ...")
    t0 = time.perf_counter()
    oof_preds, fold_models = run_cv(
        train_input_ids, train_attn_mask, train_token_type_ids, y_train
    )
    cv_time = time.perf_counter() - t0
    print(f"  CV completed in {cv_time:.1f}s ({cv_time / 3600:.1f}h)")

    oof_rmse = np.sqrt(np.mean((oof_preds - y_train) ** 2))
    oof_std = np.std(oof_preds)
    print(f"  OOF RMSE: {oof_rmse:.5f}")
    print(f"  OOF pred std: {oof_std:.5f}")

    print("\n[5/5] Generating test predictions and saving ...")
    t0 = time.perf_counter()
    test_preds = predict_test(fold_models, test_input_ids, test_attn_mask, test_token_type_ids)
    test_time = time.perf_counter() - t0
    print(f"  Test predictions in {test_time:.1f}s")

    np.save(str(OOF_PATH), oof_preds)
    print(f"  OOF saved -> {OOF_PATH}")
    np.save(str(TEST_PRED_PATH), test_preds)
    print(f"  Test preds saved -> {TEST_PRED_PATH}")

    total_time = time.perf_counter() - t_start
    print(f"\n  Total time: {total_time:.1f}s ({total_time / 3600:.1f}h)")
    print(f"  OOF RMSE: {oof_rmse:.5f}")

    save_evidence(oof_rmse, oof_std, fold_models, cv_time, total_time)
    write_changelog(oof_rmse, oof_std, cv_time, total_time)

    # Clean up checkpoints after successful completion
    _cleanup_all_checkpoints()

    print("\n=== Done ===")


def save_evidence(oof_rmse, oof_std, fold_models, cv_time, total_time):
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "Task LoRA: DeBERTa-v3-base LoRA Fine-tuning (Mean Pooling + R-Drop + CORAL)",
        "=" * 60,
        "",
        "[DUAL-TRACK] Track B: LoRA (Track A: transformer_e2e.py)",
        "",
        f"Model: {MODEL_NAME} (86M params, LoRA r={LORA_R}, alpha={LORA_ALPHA})",
        f"LoRA target modules: {LORA_TARGET_MODULES}",
        f"Trainable params: ~0.5-3M (vs 86M full)",
        "",
        f"OOF RMSE: {oof_rmse:.5f}",
        f"OOF pred std: {oof_std:.5f}",
        f"CV time: {cv_time:.1f}s ({cv_time / 3600:.1f}h)",
        f"Total time: {total_time:.1f}s ({total_time / 3600:.1f}h)",
        "",
        "Training config:",
        f"  LoRA rank: {LORA_R}",
        f"  LoRA alpha: {LORA_ALPHA}",
        f"  LoRA dropout: {LORA_DROPOUT}",
        f"  Batch size: {BATCH_SIZE}",
        f"  Grad accum steps: {GRAD_ACCUM_STEPS}",
        f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}",
        f"  Learning rate: {LR}",
        f"  Max epochs: {N_EPOCHS}",
        f"  Folds: {N_FOLDS}",
        f"  Patience: {PATIENCE}",
        f"  Sequence length: {MAX_LENGTH}",
        f"  FP16: {FP16}",
        f"  R-Drop alpha: {R_DROP_ALPHA}",
        "",
        f"Status: {'PASS - OOF RMSE < 1.05' if oof_rmse < 1.05 else 'NEEDS COMPARISON with Track A'}",
    ]
    EVIDENCE_PATH.write_text("\n".join(lines) + "\n")
    print(f"  Evidence -> {EVIDENCE_PATH}")


def write_changelog(oof_rmse, oof_std, cv_time, total_time):
    CHANGELOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DeBERTa-v3-base LoRA Fine-tuning",
        "",
        "## Dual-Track Approach",
        "- **Track A**: Full fine-tuning (transformer_e2e.py) — 4 folds × 4 epochs",
        "- **Track B**: LoRA fine-tuning (deberta_lora.py) — 5 folds × 5 epochs",
        "- Both tracks run in parallel to compare within 36h budget",
        "",
        "## Architecture",
        f"- **Model**: {MODEL_NAME} (86M params, LoRA r={LORA_R})",
        f"- **LoRA modules**: {LORA_TARGET_MODULES}",
        "- **Head**: Mean Pooling -> Dropout(0.1) -> Linear(768, 4) [CORAL]",
        "- **Prediction**: 1 + sigmoid(logits).sum() -> continuous [1, 5]",
        "",
        "## LoRA Config",
        f"- Rank: {LORA_R}",
        f"- Alpha: {LORA_ALPHA}",
        f"- Dropout: {LORA_DROPOUT}",
        f"- Target: {LORA_TARGET_MODULES}",
        "",
        "## Hyperparameters",
        f"- Folds: {N_FOLDS}",
        f"- Epochs: {N_EPOCHS} (patience={PATIENCE})",
        f"- Batch size: {BATCH_SIZE}",
        f"- Gradient accumulation: {GRAD_ACCUM_STEPS} (effective BS={BATCH_SIZE * GRAD_ACCUM_STEPS})",
        f"- Learning rate: {LR}",
        f"- Weight decay: {WEIGHT_DECAY}",
        f"- Warmup ratio: {WARMUP_RATIO}",
        f"- R-Drop alpha: {R_DROP_ALPHA}",
        f"- FP16: {FP16}",
        "",
        "## Results",
        f"- **OOF RMSE: {oof_rmse:.5f}**",
        f"- **OOF pred std: {oof_std:.5f}**",
        f"- CV time: {cv_time:.1f}s ({cv_time / 3600:.1f}h)",
        f"- Total time: {total_time:.1f}s ({total_time / 3600:.1f}h)",
        "",
        "## Outputs",
        f"- OOF: `artifacts/models/deberta_lora_oof.npy`",
        f"- Test: `artifacts/models/deberta_lora_test.npy`",
    ]
    CHANGELOG_PATH.write_text("\n".join(lines) + "\n")
    print(f"  Changelog -> {CHANGELOG_PATH}")


if __name__ == "__main__":
    main()

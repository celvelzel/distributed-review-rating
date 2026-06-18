#!/usr/bin/env python3.8
"""Fix 3M OOF bug by re-generating OOF from checkpoints."""

import os
import sys
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
from sklearn.model_selection import KFold

ROOT = "/hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating"
MODEL_DIR = os.path.join(ROOT, "artifacts", "models")
FEAT_DIR = os.path.join(ROOT, "artifacts", "features")
CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints_base_full")

MODEL_NAME = "microsoft/deberta-v3-base"
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET = ["query_proj", "value_proj"]
N_CLASSES, N_TASKS = 5, 4
N_FOLDS, N_EPOCHS = 3, 3
BATCH_SIZE = 256
FP16 = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DS(Dataset):
    def __init__(self, *tensors, idx=None):
        self.t = [t[idx] if idx is not None else t for t in tensors]
    def __len__(self): return len(self.t[0])
    def __getitem__(self, i): return tuple(t[i] for t in self.t)


class DeBERTaLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        from peft import LoraConfig, get_peft_model
        base = AutoModel.from_pretrained(MODEL_NAME)
        cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET,
                         lora_dropout=LORA_DROPOUT, bias="none")
        self.backbone = get_peft_model(base, cfg)
        self.backbone.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(base.config.hidden_size, N_TASKS)

    def forward(self, ids, mask, ttids):
        h = self.backbone(input_ids=ids, attention_mask=mask, token_type_ids=ttids).last_hidden_state
        m = mask.unsqueeze(-1).float()
        p = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.classifier(self.dropout(p))


def to_rating(logits):
    return 1.0 + torch.sigmoid(logits).sum(1)


def predict_fold(model, ds, bs=64):
    """Generate predictions for a fold."""
    model.eval()
    preds = []
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    with torch.no_grad():
        for b in dl:
            ids, mask, tt = [x.to(DEVICE).long() for x in b[:3]]
            with autocast("cuda", enabled=FP16):
                preds.append(to_rating(model(ids, mask, tt)).cpu().numpy())
    return np.concatenate(preds)


def main():
    t_start = time.time()
    print("=" * 60, flush=True)
    print("Fix 3M OOF - Re-generate from checkpoints", flush=True)
    print("=" * 60, flush=True)

    # Load tokens
    print("Loading tokens...", flush=True)
    td = np.load(os.path.join(MODEL_DIR, "train_tokens.npz"), allow_pickle=True)
    input_ids = torch.from_numpy(td["input_ids"]).to(torch.int32)
    attn_mask = torch.from_numpy(td["attention_mask"]).to(torch.int32)
    ttids = torch.from_numpy(td["token_type_ids"]).to(torch.int32)
    y_train = np.load(os.path.join(FEAT_DIR, "y_train.npy")).astype(np.float32)

    print(f"Train: {input_ids.shape}", flush=True)
    print(f"y_train: {y_train.shape}", flush=True)

    # Initialize OOF array
    oof = np.zeros(len(y_train), dtype=np.float32)
    fold_rmses = []

    # Process each fold
    print("Creating KFold split...", flush=True)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fi, (tr_idx, va_idx) in enumerate(kf.split(input_ids), 1):
        print(f"\n{'='*60}\nFold {fi}/{N_FOLDS}\n{'='*60}", flush=True)
        print(f"Val indices: {len(va_idx)} samples", flush=True)

        # Load best checkpoint for this fold (epoch 3 or last available)
        best_ckpt_path = os.path.join(CKPT_DIR, f"fold{fi}_epoch{N_EPOCHS}.pt")
        if not os.path.exists(best_ckpt_path):
            # Try to find the latest epoch for this fold
            for ep in range(N_EPOCHS, 0, -1):
                path = os.path.join(CKPT_DIR, f"fold{fi}_epoch{ep}.pt")
                if os.path.exists(path):
                    best_ckpt_path = path
                    break

        print(f"Loading checkpoint: {best_ckpt_path}", flush=True)
        ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        print(f"Checkpoint loaded. Keys: {list(ckpt.keys())}", flush=True)

        # Create model and load state
        print("Creating model...", flush=True)
        model = DeBERTaLoRA().to(DEVICE)
        print(f"Loading state dict...", flush=True)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

        # Create validation dataset
        print(f"Creating validation dataset...", flush=True)
        val_ds = DS(input_ids, attn_mask, ttids, torch.from_numpy(y_train), idx=va_idx)
        print(f"Val dataset size: {len(val_ds)}", flush=True)

        # Generate OOF predictions for this fold
        print(f"Generating OOF predictions for fold {fi}...", flush=True)
        val_preds = predict_fold(model, val_ds, bs=BATCH_SIZE)
        oof[va_idx] = np.clip(val_preds, 1.0, 5.0)

        # Calculate fold RMSE
        fold_rmse = np.sqrt(np.mean((oof[va_idx] - y_train[va_idx])**2))
        fold_rmses.append(fold_rmse)
        print(f"Fold {fi} OOF RMSE: {fold_rmse:.5f}", flush=True)

        # Cleanup
        del model, ckpt
        gc.collect()
        torch.cuda.empty_cache()

    # Calculate overall OOF RMSE
    oof_rmse = np.sqrt(np.mean((oof - y_train)**2))
    print(f"\n{'='*60}", flush=True)
    print(f"Overall OOF RMSE: {oof_rmse:.5f}", flush=True)
    print(f"Fold RMSEs: {[f'{r:.5f}' for r in fold_rmses]}", flush=True)
    print(f"Time: {(time.time()-t_start)/60:.1f} min", flush=True)

    # Save fixed OOF
    output_path = os.path.join(MODEL_DIR, "deberta_base_full_oof_fixed.npy")
    np.save(output_path, oof)
    print(f"\nSaved fixed OOF to: {output_path}", flush=True)

    # Verify
    non_zero = np.count_nonzero(oof)
    print(f"Non-zero predictions: {non_zero} / {len(oof)}", flush=True)
    print(f"Mean: {oof.mean():.4f}, Std: {oof.std():.4f}", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()

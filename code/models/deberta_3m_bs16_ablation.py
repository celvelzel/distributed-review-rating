#!/usr/bin/env python
"""3M BS Ablation: 3M data + 1M config (BS=16/GradAcc=16) to isolate batch size effect.

Hypothesis: 3M model's poor performance (0.681 vs 1M's 0.617) is due to
batch config difference (32×8 vs 16×16), not data size.

This script trains fold1 epoch1 only, with:
  - 3M data (train_tokens.npz + y_train.npy)
  - BS=16, GradAcc=16 (copied from 1M config)
  - Gradient checkpointing enabled
  - All other params identical to deberta_lora_1m.py
"""

import os, sys, time, gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from sklearn.model_selection import KFold

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, "artifacts", "models")
FEAT_DIR = os.path.join(ROOT, "artifacts", "features")
CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints_3m_bs16_ablation")
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "microsoft/deberta-v3-base"
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET = ["query_proj", "value_proj"]
N_CLASSES, N_TASKS = 5, 4
N_FOLDS, N_EPOCHS = 1, 1
BATCH_SIZE, GRAD_ACCUM = 16, 16
LR, WEIGHT_DECAY, WARMUP_RATIO = 3e-5, 0.01, 0.1
R_DROP_ALPHA, FP16, PATIENCE = 0.5, True, 3
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
        cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET, lora_dropout=LORA_DROPOUT, bias="none")
        self.backbone = get_peft_model(base, cfg)
        self.backbone.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(base.config.hidden_size, N_TASKS)

    def forward(self, ids, mask, ttids):
        h = self.backbone(input_ids=ids, attention_mask=mask, token_type_ids=ttids).last_hidden_state
        m = mask.unsqueeze(-1).float()
        p = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.classifier(self.dropout(p))


class CORAL(nn.Module):
    def forward(self, logits, labels):
        t = torch.zeros(logits.size(0), N_TASKS, device=logits.device, dtype=logits.dtype)
        for k in range(N_TASKS):
            t[:, k] = (labels.long() - 1 > k).float()
        return F.binary_cross_entropy_with_logits(logits, t, reduction="mean")


def to_rating(logits):
    return 1.0 + torch.sigmoid(logits).sum(1)


def predict(model, ds, bs=64):
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
    print("=" * 60, flush=True)
    print("3M BS Ablation: 3M data + BS=16/GradAcc=16", flush=True)
    print(f"Model: {MODEL_NAME}, LoRA r={LORA_R}", flush=True)
    print(f"Batch: {BATCH_SIZE}, GradAcc: {GRAD_ACCUM}, EffBS: {BATCH_SIZE*GRAD_ACCUM}", flush=True)
    print(f"Folds: {N_FOLDS}, Epochs: {N_EPOCHS}", flush=True)
    print("=" * 60, flush=True)

    # Load 3M data
    print("Loading 3M data...", flush=True)
    td = np.load(os.path.join(MODEL_DIR, "train_tokens.npz"), allow_pickle=True)
    input_ids = torch.from_numpy(td["input_ids"]).to(torch.int32)
    attn_mask = torch.from_numpy(td["attention_mask"]).to(torch.int32)
    ttids = torch.from_numpy(td["token_type_ids"]).to(torch.int32)
    y_train = np.load(os.path.join(FEAT_DIR, "y_train.npy")).astype(np.float32)

    td2 = np.load(os.path.join(MODEL_DIR, "test_tokens.npz"), allow_pickle=True)
    test_ids = td2["ids"]
    t_ids = torch.from_numpy(td2["input_ids"]).to(torch.int32)
    t_mask = torch.from_numpy(td2["attention_mask"]).to(torch.int32)
    t_tt = torch.from_numpy(td2["token_type_ids"]).to(torch.int32)
    print(f"Train: {input_ids.shape}, Test: {t_ids.shape}", flush=True)

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(kf.split(input_ids))
    fi = 1
    tr_idx, va_idx = folds[0]

    steps_per_epoch = len(tr_idx) // BATCH_SIZE // GRAD_ACCUM
    print(f"\nFold 1 train: {len(tr_idx):,} samples, val: {len(va_idx):,} samples", flush=True)
    print(f"Steps per epoch: {steps_per_epoch:,}", flush=True)

    model = DeBERTaLoRA().to(DEVICE)
    print(f"GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    train_ds = DS(input_ids, attn_mask, ttids, torch.from_numpy(y_train), idx=tr_idx)
    val_ds = DS(input_ids, attn_mask, ttids, torch.from_numpy(y_train), idx=va_idx)
    test_ds = DS(t_ids, t_mask, t_tt)

    tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    vl = DataLoader(val_ds, batch_size=BATCH_SIZE*4, shuffle=False, num_workers=0)

    coral_fn = CORAL()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    from transformers import get_cosine_schedule_with_warmup
    steps = len(tl) // GRAD_ACCUM
    sched = get_cosine_schedule_with_warmup(opt, int(steps*N_EPOCHS*WARMUP_RATIO), steps*N_EPOCHS)
    scaler = GradScaler("cuda", enabled=FP16)

    # Train 1 epoch
    ep = 1
    model.train()
    opt.zero_grad(set_to_none=True)
    t0 = time.time()
    n_samp = 0

    for si, batch in enumerate(tl, 1):
        ids = batch[0].to(DEVICE).long()
        mask = batch[1].to(DEVICE).long()
        tt = batch[2].to(DEVICE).long()
        lab = batch[3].to(DEVICE)

        with autocast("cuda", enabled=FP16):
            l1 = model(ids, mask, tt)
            l2 = model(ids, mask, tt)
            loss = (coral_fn(l1, lab) + coral_fn(l2, lab)) / 2 + R_DROP_ALPHA * F.mse_loss(l1, l2)

        scaler.scale(loss / GRAD_ACCUM).backward()
        del l1, l2, ids, mask, tt, lab

        if si % GRAD_ACCUM == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()

        n_samp += len(batch[0])
        if si % 1000 == 0:
            eta = (len(tl) - si) / (si / (time.time() - t0))
            print(f"  step {si}/{len(tl)}: loss={loss.item():.5f} ETA={eta:.0f}s", flush=True)
        if si % 5000 == 0:
            gc.collect(); torch.cuda.empty_cache()

    if si % GRAD_ACCUM != 0:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()

    train_time = time.time() - t0
    print(f"\n  Training done in {train_time:.0f}s ({train_time/60:.1f}min)", flush=True)

    # Validate
    model.eval()
    vp, vlabs = [], []
    with torch.no_grad():
        for batch in vl:
            ids = batch[0].to(DEVICE).long()
            mask = batch[1].to(DEVICE).long()
            tt = batch[2].to(DEVICE).long()
            with autocast("cuda", enabled=FP16):
                vp.append(to_rating(model(ids, mask, tt)).cpu())
            vlabs.append(batch[3])
    vp = torch.cat(vp).numpy()
    vlabs = torch.cat(vlabs).numpy()
    vrmse = float(np.sqrt(np.mean((vp - vlabs)**2)))
    print(f"  Fold 1 Epoch 1: val_rmse={vrmse:.5f}", flush=True)

    # OOF
    oof = np.zeros(len(y_train), dtype=np.float32)
    oof[va_idx] = np.clip(vp, 1.0, 5.0)
    oof_rmse = np.sqrt(np.mean((oof[va_idx] - y_train[va_idx])**2))
    print(f"  OOF RMSE (fold1): {oof_rmse:.5f}", flush=True)

    # Test predictions
    test_preds = np.clip(predict(model, test_ds), 1.0, 5.0)
    print(f"  Test: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}", flush=True)

    # Save
    np.save(os.path.join(MODEL_DIR, "deberta_3m_bs16_ablation_fold1e1_oof.npy"), oof)
    np.save(os.path.join(MODEL_DIR, "deberta_3m_bs16_ablation_fold1e1_test.npy"), test_preds)

    # Save checkpoint
    ckpt = {"fold": 1, "epoch": 1, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(), "scheduler_state_dict": sched.state_dict(),
            "scaler_state_dict": scaler.state_dict(), "best_val_rmse": vrmse,
            "patience_counter": 0, "completed_folds": 0}
    torch.save(ckpt, os.path.join(CKPT_DIR, "fold1_epoch1.pt"))

    # Generate submission with VE + stacking blend
    import pandas as pd

    target_std = y_train.std()
    target_mean = y_train.mean()
    pred_std = test_preds.std()
    pred_mean = test_preds.mean()
    ve_test = np.clip((test_preds - pred_mean) / pred_std * target_std + target_mean, 1.0, 5.0)

    ridge_test = np.load(os.path.join(MODEL_DIR, "stacking_v2_test.npy")).astype(np.float32)

    for w in [90, 95]:
        blend = np.clip(w/100 * ve_test + (100-w)/100 * ridge_test, 1.0, 5.0)
        path = os.path.join(OUTPUT_DIR, f"submission-3m-bs16-ablation-ve{w}-r{100-w}.csv")
        pd.DataFrame({"id": test_ids, "rating": blend}).to_csv(path, index=False)
        print(f"  Saved: submission-3m-bs16-ablation-ve{w}-r{100-w}.csv", flush=True)

    # Also save raw (no VE)
    path = os.path.join(OUTPUT_DIR, "submission-3m-bs16-ablation-raw.csv")
    pd.DataFrame({"id": test_ids, "rating": test_preds}).to_csv(path, index=False)

    print(f"\n{'='*60}", flush=True)
    print(f"3M BS Ablation Result", flush=True)
    print(f"  val_rmse={vrmse:.5f}", flush=True)
    print(f"  Test pred: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}", flush=True)
    print(f"  VE pred:   mean={ve_test.mean():.4f}, std={ve_test.std():.4f}", flush=True)
    print(f"  Training time: {train_time:.0f}s ({train_time/60:.1f}min)", flush=True)
    print(f"  GPU peak: {torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
    print("=" * 60, flush=True)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()

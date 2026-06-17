#!/usr/bin/env python
"""Retrain fold 2 epoch 3 of DeBERTa-v3-base on GPU. Pause v3-large, run this, restart v3-large."""

import os, sys, time, gc
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
from sklearn.model_selection import KFold

ROOT = "/hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating"
MODEL_DIR = os.path.join(ROOT, "artifacts", "models")
FEAT_DIR = os.path.join(ROOT, "artifacts", "features")
CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints_base_full")
os.makedirs(CKPT_DIR, exist_ok=True)

MODEL_NAME = "microsoft/deberta-v3-base"
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET = ["query_proj", "value_proj"]
N_TASKS = 4
N_FOLDS, N_EPOCHS = 3, 3
BATCH_SIZE, GRAD_ACCUM = 32, 8
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
        cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET,
                         lora_dropout=LORA_DROPOUT, bias="none")
        self.backbone = get_peft_model(base, cfg)
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
    t_start = time.time()
    print("=" * 60, flush=True)
    print("Retrain fold 2 epoch 3 of DeBERTa-v3-base on GPU", flush=True)
    print("=" * 60, flush=True)

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

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fi, (tr_idx, va_idx) in enumerate(kf.split(input_ids), 1):
        if fi != 2:
            continue

        print(f"\nFold 2 — Loading from fold2_epoch2 checkpoint...", flush=True)
        model = DeBERTaLoRA().to(DEVICE)
        print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

        ckpt = torch.load(os.path.join(CKPT_DIR, "fold2_epoch2.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded fold2_epoch2 weights", flush=True)

        train_ds = DS(input_ids, attn_mask, ttids, torch.from_numpy(y_train), idx=tr_idx)
        val_ds = DS(input_ids, attn_mask, ttids, torch.from_numpy(y_train), idx=va_idx)
        test_ds = DS(t_ids, t_mask, t_tt)

        tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
        vl = DataLoader(val_ds, batch_size=BATCH_SIZE*4, shuffle=False, num_workers=0)

        coral_fn = CORAL()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        from transformers import get_cosine_schedule_with_warmup
        steps = len(tl) // GRAD_ACCUM
        sched = get_cosine_schedule_with_warmup(opt, int(steps * WARMUP_RATIO), steps)
        scaler = GradScaler("cuda", enabled=FP16)

        opt.load_state_dict(ckpt["optimizer_state_dict"])
        sched.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"  Loaded optimizer + scheduler state", flush=True)

        ep = 3
        model.train()
        opt.zero_grad(set_to_none=True)
        t0 = time.time()

        print(f"\nFold 2 Epoch 3 — {len(tl)} batches, {steps} grad steps", flush=True)

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

            if si % 1000 == 0:
                elapsed = time.time() - t0
                speed = si / elapsed
                eta = (len(tl) - si) / speed
                print(f"  f2e3 step {si}/{len(tl)}: loss={loss.item():.5f} ETA={eta:.0f}s ({speed:.1f} steps/s)", flush=True)
            if si % 5000 == 0:
                gc.collect(); torch.cuda.empty_cache()

        if si % GRAD_ACCUM != 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()

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
        print(f"  Fold 2 Epoch 3: val_rmse={vrmse:.5f} ({time.time()-t0:.1f}s)", flush=True)

        ckpt_out = {"fold": fi, "epoch": ep, "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": opt.state_dict(), "scheduler_state_dict": sched.state_dict(),
                     "scaler_state_dict": scaler.state_dict(), "best_val_rmse": vrmse,
                     "patience_counter": 0, "completed_folds": fi - 1}
        cp = os.path.join(CKPT_DIR, f"fold{fi}_epoch{ep}.pt")
        torch.save(ckpt_out, cp)
        with open(os.path.join(CKPT_DIR, "latest.txt"), "w") as f:
            f.write(f"fold{fi}_epoch{ep}.pt")
        print(f"  Saved: fold{fi}_epoch{ep}.pt", flush=True)

        test_preds = np.clip(predict(model, test_ds), 1.0, 5.0)
        np.save(os.path.join(MODEL_DIR, "deberta_base_fold2_test.npy"), test_preds)
        print(f"  Saved fold2 test predictions", flush=True)

        with open(os.path.join(CKPT_DIR, "latest.txt"), "w") as f:
            f.write("fold3_epoch3.pt")
        print("Restored latest.txt to fold3_epoch3.pt", flush=True)

        del model; gc.collect(); torch.cuda.empty_cache()

    print(f"\nFold 2 epoch 3 complete! Total: {(time.time()-t_start)/3600:.1f}h", flush=True)


if __name__ == "__main__":
    main()

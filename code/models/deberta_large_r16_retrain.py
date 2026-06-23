#!/usr/bin/env python3
"""
Retrain DeBERTa-v3-large fold2 and fold3 with LoRA r=16 (same as fold1)
"""

import os, sys, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.model_selection import KFold

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")
FEAT_DIR = os.path.join(ARTIFACTS_DIR, "features")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints_large_full_r16")

os.makedirs(CKPT_DIR, exist_ok=True)

MODEL_NAME = "microsoft/deberta-v3-large"
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.02  # Changed from 32 to 16
LORA_TARGET = ["query_proj", "value_proj"]  # Same as fold1
N_CLASSES, N_TASKS = 5, 4
N_FOLDS, N_EPOCHS = 3, 3
BATCH_SIZE, GRAD_ACCUM = 64, 4
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
        self.backbone.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(base.config.hidden_size, N_TASKS)
        self.backbone.print_trainable_parameters()

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


def predict(model, ds, bs=16):
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
    print("DeBERTa-v3-large LoRA r=16 Retrain (fold2 & fold3)", flush=True)
    print(f"GPU: {torch.cuda.get_device_name()}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB", flush=True)
    print("=" * 60, flush=True)

    # Load tokens
    print("Loading tokens...", flush=True)
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

    # Load fold1 predictions (already done with r=16)
    fold1_test = np.load(os.path.join(MODEL_DIR, "deberta_large_fold1_test.npy"))
    print(f"Loaded fold1 predictions: mean={fold1_test.mean():.4f}, std={fold1_test.std():.4f}", flush=True)

    # Prepare
    oof = np.zeros(len(y_train), dtype=np.float32)
    test_preds_list = [fold1_test]  # Start with fold1
    fold_rmses = []

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fi, (tr_idx, va_idx) in enumerate(kf.split(input_ids), 1):
        if fi == 1:
            print(f"Skipping fold 1 (already done with r=16)", flush=True)
            continue

        print(f"\n{'='*60}\nFold {fi}/{N_FOLDS}\n{'='*60}", flush=True)
        model = DeBERTaLoRA().to(DEVICE)
        print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB", flush=True)

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

        best_rmse = float("inf")
        best_state = None
        patience = 0

        for ep in range(1, N_EPOCHS + 1):
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
                    gpu = torch.cuda.memory_allocated()/1e9
                    print(f"  f{fi}e{ep} step {si}/{len(tl)}: loss={loss.item():.5f} ETA={eta:.0f}s GPU={gpu:.1f}GB", flush=True)
                if si % 5000 == 0:
                    gc.collect(); torch.cuda.empty_cache()

            if si % GRAD_ACCUM != 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()

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
            print(f"  Fold {fi} Epoch {ep}: val_rmse={vrmse:.5f} ({time.time()-t0:.1f}s)", flush=True)

            # Save checkpoint
            ckpt = {"fold": fi, "epoch": ep, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(), "scheduler_state_dict": sched.state_dict(),
                    "scaler_state_dict": scaler.state_dict(), "best_val_rmse": best_rmse,
                    "patience_counter": patience, "completed_folds": fi - 1}
            cp = os.path.join(CKPT_DIR, f"fold{fi}_epoch{ep}.pt")
            torch.save(ckpt, cp)
            with open(os.path.join(CKPT_DIR, "latest.txt"), "w") as f:
                f.write(os.path.basename(cp))
            print(f"  Saved: {os.path.basename(cp)}", flush=True)

            if vrmse < best_rmse:
                best_rmse = vrmse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE:
                    print(f"  Early stopping", flush=True)
                    break

        # OOF with best model
        model.load_state_dict(best_state)
        model = model.to(DEVICE)
        model.eval()
        vp = []
        with torch.no_grad():
            for batch in vl:
                ids = batch[0].to(DEVICE).long()
                mask = batch[1].to(DEVICE).long()
                tt = batch[2].to(DEVICE).long()
                with autocast("cuda", enabled=FP16):
                    vp.append(to_rating(model(ids, mask, tt)).cpu().numpy())
        oof[va_idx] = np.clip(np.concatenate(vp), 1.0, 5.0)
        frmse = np.sqrt(np.mean((oof[va_idx] - y_train[va_idx])**2))
        fold_rmses.append(frmse)
        print(f"  Fold {fi} OOF RMSE: {frmse:.5f}", flush=True)

        test_preds_list.append(np.clip(predict(model, test_ds), 1.0, 5.0))
        np.save(os.path.join(MODEL_DIR, f"deberta_large_r16_fold{fi}_test.npy"), test_preds_list[-1])
        del model, best_state; gc.collect(); torch.cuda.empty_cache()

    # Results
    avg_test = np.clip(np.mean(test_preds_list, axis=0), 1.0, 5.0)
    oof_rmse = np.sqrt(np.mean((oof - y_train)**2))
    print(f"\n{'='*60}", flush=True)
    print(f"DeBERTa-v3-large r=16 (FULL 3M, {N_FOLDS}f x {N_EPOCHS}e)", flush=True)
    print(f"OOF RMSE: {oof_rmse:.5f}", flush=True)
    print(f"Fold RMSEs: {[f'{r:.5f}' for r in fold_rmses]}", flush=True)
    print(f"Test: mean={avg_test.mean():.4f}, std={avg_test.std():.4f}", flush=True)
    print(f"Time: {(time.time()-t_start)/3600:.1f}h", flush=True)

    # Save
    np.save(os.path.join(MODEL_DIR, "deberta_large_r16_oof.npy"), oof)
    np.save(os.path.join(MODEL_DIR, "deberta_large_r16_test.npy"), avg_test)

    import pandas as pd
    pd.DataFrame({"id": test_ids, "rating": avg_test}).to_csv(
        os.path.join(OUTPUT_DIR, "submission-deberta-large-r16.csv"), index=False)

    print("Done!", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""DeBERTa-v3-base LoRA on 1M subsample with checkpoint resume.

Best single model in the COMP5434 ensemble pipeline. Achieves the strongest
Kaggle leaderboard score (0.61734) when blended with stacking predictions.

Training techniques:
  - LoRA (Low-Rank Adaptation): Injects trainable low-rank matrices into the
    attention query/value projections, reducing trainable params from ~140M
    to ~1.2M while preserving DeBERTa-v3-base representation quality.
  - CORAL (Consistent Rank Logits): Ordinal regression loss that decomposes
    the 5-class rating into 4 binary thresholds, capturing ordinal structure.
  - R-Drop regularisation: Forward-passes each batch twice and penalises the
    KL divergence between the two output distributions, improving robustness.
  - Mixed-precision (FP16) training with gradient accumulation (effective
    batch size = 16 × 16 = 256).

Cross-validation: 3-fold KFold (shuffle=True, seed=42) on a 1M-row subsample.
Each fold trains for up to 3 epochs with early stopping (patience=3).

Checkpoint resume: After every epoch, saves full training state (model,
optimizer, scheduler, scaler) so training can resume after interruptions.
"""

import os, sys, time, gc, json
# Disable HuggingFace tokenizers parallelism to avoid fork-safety warnings
# when running under DataLoader with num_workers > 0.
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
CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints_v3base_1m")
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "microsoft/deberta-v3-base"

# LoRA configuration: rank=16 balances expressiveness and parameter savings.
# alpha=32 (= 2× rank) is the standard scaling factor. dropout=0.05 for
# lightweight regularisation on the adapter weights.
# Target modules: query_proj and value_proj in the attention layer — the
# two projections where LoRA is most effective per the original paper.
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET = ["query_proj", "value_proj"]

# N_TASKS = N_CLASSES - 1: CORAL reduces 5 ordinal classes to 4 binary
# thresholds (rating > 1, > 2, > 3, > 4).
N_CLASSES, N_TASKS = 5, 4

# Cross-validation and training schedule
N_FOLDS, N_EPOCHS = 3, 3

# Effective batch size = BATCH_SIZE × GRAD_ACCUM = 16 × 16 = 256.
# Large effective batch stabilises training on the 1M subsample.
BATCH_SIZE, GRAD_ACCUM = 16, 16

# AdamW with cosine schedule and 10% warmup steps to avoid early instability.
LR, WEIGHT_DECAY, WARMUP_RATIO = 3e-5, 0.01, 0.1

# R_DROP_ALPHA controls the MSE penalty between two forward passes (R-Drop).
# FP16 enables mixed-precision for memory savings and speed.
# PATIENCE=3: early-stop if val RMSE does not improve for 3 consecutive epochs.
R_DROP_ALPHA, FP16, PATIENCE = 0.5, True, 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DS(Dataset):
    """Generic tensor dataset supporting optional index-based subsetting.

    Used to create train/val splits from the same underlying tensors without
    copying data — passing idx=tr_idx or idx=va_idx selects fold rows in-place.
    """

    def __init__(self, *tensors, idx=None):
        # If idx is provided, each tensor is sliced to those indices (for KFold splits).
        self.t = [t[idx] if idx is not None else t for t in tensors]

    def __len__(self): return len(self.t[0])

    def __getitem__(self, i): return tuple(t[i] for t in self.t)


class DeBERTaLoRA(nn.Module):
    """DeBERTa-v3-base with LoRA adapters and a CORAL ordinal-regression head.

    Architecture:
      1. Backbone: DeBERTa-v3-base with LoRA adapters on attention projections.
         gradient_checkpointing_enable() trades compute for memory, allowing
         training on a single GPU with 1M samples.
      2. Pooling: Mean-pooling over the last hidden states, masked by the
         attention mask to exclude padding tokens.
      3. Classifier: Linear(hidden_size, 4) producing 4 binary logits for CORAL.
    """

    def __init__(self):
        super().__init__()
        from transformers import AutoModel
        from peft import LoraConfig, get_peft_model
        base = AutoModel.from_pretrained(MODEL_NAME)
        cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET, lora_dropout=LORA_DROPOUT, bias="none")
        self.backbone = get_peft_model(base, cfg)
        # Gradient checkpointing reduces peak GPU memory at the cost of ~20%
        # slower training, enabling the model to fit within single-GPU limits.
        self.backbone.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(0.1)
        # Single linear layer maps pooled representation to N_TASKS=4 logits.
        # These logits are used by CORAL for ordinal regression.
        self.classifier = nn.Linear(base.config.hidden_size, N_TASKS)

    def forward(self, ids, mask, ttids):
        """Forward pass: returns 4 binary logits for CORAL.

        Args:
            ids: input_ids (batch, seq_len)
            mask: attention_mask (batch, seq_len)
            ttids: token_type_ids (batch, seq_len)
        Returns:
            logits: (batch, 4) — one binary logit per CORAL threshold.
        """
        h = self.backbone(input_ids=ids, attention_mask=mask, token_type_ids=ttids).last_hidden_state
        # Mean pooling: average over sequence dimension, excluding padding
        # tokens via the attention mask. This is more robust than [CLS] pooling
        # for DeBERTa which does not use a standard [CLS] token.
        m = mask.unsqueeze(-1).float()
        p = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return self.classifier(self.dropout(p))


class CORAL(nn.Module):
    """CORAL (Consistent Rank Logits) loss for ordinal regression.

    Decomposes a 5-class ordinal label into 4 binary classification tasks:
      threshold k asks "is the rating strictly greater than k+1?"
    For example, rating=4 → thresholds [1,1,1,0] (greater than 1, 2, 3 but not 4).
    This preserves the ordinal structure of ratings (1 < 2 < 3 < 4 < 5)
    better than standard multi-class cross-entropy.
    """

    def forward(self, logits, labels):
        """Compute binary cross-entropy loss on the ordinal thresholds.

        Args:
            logits: (batch, 4) — predicted binary logits per threshold.
            labels: (batch,) — integer ratings 1–5.
        """
        # Build binary targets: t[:, k] = 1 if label > k+1, else 0.
        t = torch.zeros(logits.size(0), N_TASKS, device=logits.device, dtype=logits.dtype)
        for k in range(N_TASKS):
            t[:, k] = (labels.long() - 1 > k).float()
        return F.binary_cross_entropy_with_logits(logits, t, reduction="mean")


def predict(model, ds, bs=64):
    """Generate predictions for a dataset in eval mode.

    Args:
        model: trained DeBERTaLoRA model.
        ds: DS dataset wrapping (input_ids, attention_mask, token_type_ids).
        bs: batch size for inference.
    Returns:
        np.ndarray of predicted ratings in [1, 5].
    """
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
    print("DeBERTa-v3-base LoRA on 1M subsample", flush=True)
    print("=" * 60, flush=True)

    # --- Load data ---
    print("Loading data...", flush=True)
    td = np.load(os.path.join(MODEL_DIR, "train_tokens_1m.npz"), allow_pickle=True)
    input_ids = torch.from_numpy(td["input_ids"]).to(torch.int32)
    attn_mask = torch.from_numpy(td["attention_mask"]).to(torch.int32)
    ttids = torch.from_numpy(td["token_type_ids"]).to(torch.int32)
    y_train = np.load(os.path.join(FEAT_DIR, "y_train_1m.npy")).astype(np.float32)

    td2 = np.load(os.path.join(MODEL_DIR, "test_tokens.npz"), allow_pickle=True)
    test_ids = td2["ids"]
    t_ids = torch.from_numpy(td2["input_ids"]).to(torch.int32)
    t_mask = torch.from_numpy(td2["attention_mask"]).to(torch.int32)
    t_tt = torch.from_numpy(td2["token_type_ids"]).to(torch.int32)
    print(f"Train: {input_ids.shape}, Test: {t_ids.shape}", flush=True)

    # Load checkpoint for resume: latest.txt stores the filename of the most
    # recently saved checkpoint. If it exists, training resumes from that
    # fold/epoch, skipping already-completed folds entirely.
    resume = None
    latest = os.path.join(CKPT_DIR, "latest.txt")
    if os.path.exists(latest):
        name = open(latest).read().strip()
        path = os.path.join(CKPT_DIR, name)
        if os.path.exists(path):
            resume = torch.load(path, map_location="cpu", weights_only=False)
            print(f"Resuming from: {name} (fold={resume['fold']}, epoch={resume['epoch']})", flush=True)

    # --- Initialize predictions ---
    oof = np.zeros(len(y_train), dtype=np.float32)
    test_preds_list = []
    fold_rmses = []

    # 3 折 KFold (shuffle=True, seed=42): 打乱后均分，保证可复现
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    folds = list(kf.split(input_ids))

    # 断点续训: 从上次保存的 fold/epoch 继续，跳过已完成的 fold
    start_fold = resume["fold"] if resume else 1
    start_epoch = (resume["epoch"] + 1) if resume else 1

    # --- Cross-validation fold loop ---
    for fi, (tr_idx, va_idx) in enumerate(folds, 1):
        if fi < start_fold:
            print(f"Skipping fold {fi} (completed)", flush=True)
            continue

        print(f"\n{'='*60}\nFold {fi}/{N_FOLDS}\n{'='*60}", flush=True)

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

        best_rmse = float("inf")
        best_state = None
        patience = 0

        ep_start = start_epoch if fi == start_fold else 1
        if resume and fi == start_fold:
            model.load_state_dict(resume["model_state_dict"])
            opt.load_state_dict(resume["optimizer_state_dict"])
            sched.load_state_dict(resume["scheduler_state_dict"])
            scaler.load_state_dict(resume["scaler_state_dict"])
            best_rmse = resume.get("best_val_rmse", float("inf"))
            resume = None

        # --- Training loop ---
        for ep in range(ep_start, N_EPOCHS + 1):
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
                    # R-Drop: forward-pass the same batch twice (different
                    # dropout masks) and compute:
                    #   loss = (CORAL(l1) + CORAL(l2)) / 2 + α * MSE(l1, l2)
                    # The MSE term regularises the two outputs to be consistent,
                    # effectively constraining the model to be robust to dropout.
                    l1 = model(ids, mask, tt)
                    l2 = model(ids, mask, tt)
                    loss = (coral_fn(l1, lab) + coral_fn(l2, lab)) / 2 + R_DROP_ALPHA * F.mse_loss(l1, l2)

                # Scale loss by 1/GRAD_ACCUM for gradient accumulation:
                # gradients are accumulated over GRAD_ACCUM mini-batches before
                # the optimizer steps, simulating a larger effective batch size.
                scaler.scale(loss / GRAD_ACCUM).backward()
                del l1, l2, ids, mask, tt, lab

                if si % GRAD_ACCUM == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()

                n_samp += len(batch[0])
                if si % 1000 == 0:
                    eta = (len(tl) - si) / (si / (time.time() - t0))
                    print(f"  f{fi}e{ep} step {si}/{len(tl)}: loss={loss.item():.5f} ETA={eta:.0f}s", flush=True)
                if si % 5000 == 0:
                    gc.collect(); torch.cuda.empty_cache()

            if si % GRAD_ACCUM != 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()

            # --- Validation ---
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

            # --- Save checkpoint ---
            ckpt = {"fold": fi, "epoch": ep, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(), "scheduler_state_dict": sched.state_dict(),
                    "scaler_state_dict": scaler.state_dict(), "best_val_rmse": best_rmse, "patience_counter": patience,
                    "completed_folds": fi - 1}
            cp = os.path.join(CKPT_DIR, f"fold{fi}_epoch{ep}.pt")
            torch.save(ckpt, cp)
            with open(os.path.join(CKPT_DIR, "latest.txt"), "w") as f:
                f.write(os.path.basename(cp))

            if vrmse < best_rmse:
                best_rmse = vrmse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= PATIENCE:
                    print(f"  Early stopping", flush=True)
                    break

        # --- OOF predictions ---
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

        # --- Test predictions ---
        test_preds_list.append(np.clip(predict(model, test_ds), 1.0, 5.0))

        del model, best_state, train_ds, val_ds, tl, vl
        gc.collect(); torch.cuda.empty_cache()

    # --- Results ---
    avg_test = np.clip(np.mean(test_preds_list, axis=0), 1.0, 5.0)
    oof_rmse = np.sqrt(np.mean((oof - y_train)**2))

    print(f"\n{'='*60}", flush=True)
    print(f"DeBERTa-v3-base (1M, {N_FOLDS}f x {N_EPOCHS}e)", flush=True)
    print(f"OOF RMSE: {oof_rmse:.5f}", flush=True)
    print(f"Test: mean={avg_test.mean():.4f}, std={avg_test.std():.4f}", flush=True)

    # --- Save predictions ---
    np.save(os.path.join(MODEL_DIR, "deberta_v3base_1m_oof.npy"), oof)
    np.save(os.path.join(MODEL_DIR, "deberta_v3base_1m_test.npy"), avg_test)

    # --- Blend with stacking ---
    # 混合 DeBERTa 1M 预测与 stacking_v2 Ridge 预测
    # 混合公式: final = w * deberta + (1-w) * stacking_v2
    # 扫描 w 从 70% 到 100% (步长 5%)，DeBERTa 权重越高通常 Kaggle 成绩越好
    import pandas as pd
    ridge_test = np.load(os.path.join(MODEL_DIR, "stacking_v2_test.npy")).astype(np.float32)

    for w in range(70, 101, 5):
        blend = np.clip(w/100 * avg_test + (100-w)/100 * ridge_test, 1.0, 5.0)
        pd.DataFrame({"id": test_ids, "rating": blend}).to_csv(
            os.path.join(OUTPUT_DIR, f"submission-deb{w}-ridge{100-w}.csv"), index=False)

    pd.DataFrame({"id": test_ids, "rating": avg_test}).to_csv(
        os.path.join(OUTPUT_DIR, "submission-deberta-v3base-1m.csv"), index=False)

    print("Done!", flush=True)


if __name__ == "__main__":
    main()

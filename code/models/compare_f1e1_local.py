#!/usr/bin/env python
"""
Generate submission CSVs for 3M fold1 (epoch1 & epoch3) with VE 90% + Ridge 10%.
Also compares with old 1M fold1 predictions.

Outputs:
  output/submission-3m-f1e1-ve90-r10.csv   (fold1 epoch1)
  output/submission-3m-f1e3-ve90-r10.csv   (fold1 epoch3)

Usage:
  python code/models/compare_f1e1_local.py
"""

import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, "artifacts", "models")
FEAT_DIR = os.path.join(ROOT, "artifacts", "features")
OUTPUT_DIR = os.path.join(ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "microsoft/deberta-v3-base"
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET = ["query_proj", "value_proj"]
N_TASKS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DS(Dataset):
    def __init__(self, *tensors):
        self.t = tensors
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


def to_rating(logits):
    return 1.0 + torch.sigmoid(logits).sum(1)


def predict(model, ds, bs=64):
    model.eval()
    preds = []
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    with torch.no_grad():
        for b in dl:
            ids, mask, tt = [x.to(DEVICE).long() for x in b[:3]]
            with autocast("cuda", enabled=True):
                preds.append(to_rating(model(ids, mask, tt)).cpu().numpy())
    return np.concatenate(preds)


def variance_expand(preds, target_mean, target_std):
    p_mean, p_std = preds.mean(), preds.std()
    if p_std < 1e-9:
        return preds
    return np.clip((preds - p_mean) / p_std * target_std + target_mean, 1.0, 5.0)


def load_checkpoint_preds(ckpt_path, test_ds):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"  Checkpoint: fold={ckpt['fold']}, epoch={ckpt['epoch']}, best_val_rmse={ckpt.get('best_val_rmse', 'N/A')}")
    model = DeBERTaLoRA().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    preds = np.clip(predict(model, test_ds), 1.0, 5.0)
    del model; torch.cuda.empty_cache()
    return preds, ckpt


def main():
    print("=" * 70)
    print("3M fold1 Submissions: epoch1 & epoch3 (VE 90% + Ridge 10%)")
    print("=" * 70)

    # Load test data
    print("\n[1/5] Loading test data...")
    td2 = np.load(os.path.join(MODEL_DIR, "test_tokens.npz"), allow_pickle=True)
    test_ids = td2["ids"]
    t_ids = torch.from_numpy(td2["input_ids"]).to(torch.int32)
    t_mask = torch.from_numpy(td2["attention_mask"]).to(torch.int32)
    t_tt = torch.from_numpy(td2["token_type_ids"]).to(torch.int32)
    test_ds = DS(t_ids, t_mask, t_tt)
    print(f"  Test: {len(test_ids)} samples")

    # Load training labels for VE
    y_train = np.load(os.path.join(FEAT_DIR, "y_train.npy")).astype(np.float32)
    target_mean, target_std = y_train.mean(), y_train.std()
    print(f"  Labels: mean={target_mean:.4f}, std={target_std:.4f}")

    # Load old 1M fold1 predictions (reference)
    print("\n[2/5] Loading old 1M fold1 predictions (reference)...")
    old_path = os.path.join(MODEL_DIR, "deberta_lora_fold1_test.npy")
    old_preds = None
    if os.path.exists(old_path):
        old_preds = np.load(old_path)
        print(f"  Old 1M fold1: mean={old_preds.mean():.4f}, std={old_preds.std():.4f}")
    else:
        print(f"  WARNING: {old_path} not found!")

    # Load Ridge stacking predictions for blend
    ridge_path = os.path.join(MODEL_DIR, "stacking_v2_test.npy")
    ridge_preds = None
    if os.path.exists(ridge_path):
        ridge_preds = np.load(ridge_path).astype(np.float32)
        print(f"  Ridge stacking: mean={ridge_preds.mean():.4f}, std={ridge_preds.std():.4f}")
    else:
        print(f"  WARNING: {ridge_path} not found, will skip Ridge blend")

    # Generate predictions for epoch1 and epoch3
    ckpt_dir = os.path.join(MODEL_DIR, "checkpoints_base_full")
    epochs_to_check = {"e1": "fold1_epoch1.pt", "e3": "fold1_epoch3.pt"}
    results = {}

    for label, ckpt_name in epochs_to_check.items():
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"\n  SKIP: {ckpt_name} not found at {ckpt_path}")
            continue

        print(f"\n[3/5] Loading 3M fold1 {label} checkpoint...")
        preds, ckpt = load_checkpoint_preds(ckpt_path, test_ds)
        results[label] = preds
        print(f"  3M fold1 {label}: mean={preds.mean():.4f}, std={preds.std():.4f}")

    if not results:
        print("\nERROR: No checkpoints found!")
        return

    # Compare epoch1 vs epoch3
    if "e1" in results and "e3" in results:
        print("\n" + "=" * 70)
        print("EPOCH1 vs EPOCH3 COMPARISON")
        print("=" * 70)
        e1, e3 = results["e1"], results["e3"]
        corr_e1_e3 = np.corrcoef(e1, e3)[0, 1]
        diff_e1_e3 = e3 - e1
        print(f"  Correlation: {corr_e1_e3:.6f}")
        print(f"  Mean diff (e3 - e1): {diff_e1_e3.mean():.6f}")
        print(f"  Std diff: {diff_e1_e3.std():.6f}")
        print(f"  Samples with |diff| > 0.1: {(np.abs(diff_e1_e3) > 0.1).sum():,}")

    # Compare with old 1M
    if old_preds is not None:
        print("\n" + "=" * 70)
        print("OLD 1M vs NEW 3M COMPARISON")
        print("=" * 70)
        for label, preds in results.items():
            corr = np.corrcoef(old_preds, preds)[0, 1]
            diff = preds - old_preds
            print(f"\n  Old 1M vs 3M {label}:")
            print(f"    Correlation: {corr:.6f}")
            print(f"    Mean diff: {diff.mean():.6f}")
            print(f"    Std diff: {diff.std():.6f}")

    # Generate submission CSVs
    print("\n" + "=" * 70)
    print("GENERATING SUBMISSIONS (VE 90% + Ridge 10%)")
    print("=" * 70)

    for label, preds in results.items():
        epoch_name = "e1" if label == "e1" else "e3"

        # Apply VE
        ve_preds = variance_expand(preds, target_mean, target_std)
        print(f"\n  3M fold1 {label}:")
        print(f"    Raw: mean={preds.mean():.4f}, std={preds.std():.4f}")
        print(f"    VE:  mean={ve_preds.mean():.4f}, std={ve_preds.std():.4f}")

        # Blend with Ridge (VE 90% + Ridge 10%)
        if ridge_preds is not None:
            blend = np.clip(0.9 * ve_preds + 0.1 * ridge_preds, 1.0, 5.0)
            filename = f"submission-3m-f1{epoch_name}-ve90-r10.csv"
            pd.DataFrame({"id": test_ids, "rating": blend}).to_csv(
                os.path.join(OUTPUT_DIR, filename), index=False)
            print(f"    Blend (VE 90% + Ridge 10%): mean={blend.mean():.4f}, std={blend.std():.4f}")
            print(f"    Saved: {filename}")
        else:
            # VE only
            filename = f"submission-3m-f1{epoch_name}-ve-only.csv"
            pd.DataFrame({"id": test_ids, "rating": ve_preds}).to_csv(
                os.path.join(OUTPUT_DIR, filename), index=False)
            print(f"    VE only: mean={ve_preds.mean():.4f}, std={ve_preds.std():.4f}")
            print(f"    Saved: {filename}")

    # Summary
    print("\n" + "=" * 70)
    print("SUBMISSION SUMMARY")
    print("=" * 70)
    print(f"\n  Old 1M fold1 (reference): Kaggle = 0.61734")
    for label, preds in results.items():
        epoch_name = "e1" if label == "e1" else "e3"
        print(f"  3M fold1 {label}: output/submission-3m-f1{epoch_name}-ve90-r10.csv")

    print(f"\n  Key question: Does epoch1 > epoch3 on Kaggle?")
    print(f"  If yes -> overfitting confirmed, use epoch1")
    print(f"  If no  -> more training helps, use epoch3")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Compare 3M fold1_epoch1 predictions with old 1M fold1 predictions.
Does NOT submit to Kaggle - local comparison only.
"""

import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, "artifacts", "models")
FEAT_DIR = os.path.join(ROOT, "artifacts", "features")

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


def main():
    print("=" * 70)
    print("3M fold1_epoch1 vs Old 1M fold1 Comparison (LOCAL ONLY)")
    print("=" * 70)

    # Load test data
    print("\n[1/4] Loading test data...")
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

    # Load old 1M fold1 predictions
    print("\n[2/4] Loading old 1M fold1 predictions...")
    old_path = os.path.join(MODEL_DIR, "deberta_lora_fold1_test.npy")
    if os.path.exists(old_path):
        old_preds = np.load(old_path)
        print(f"  Old 1M fold1: mean={old_preds.mean():.4f}, std={old_preds.std():.4f}")
    else:
        print(f"  WARNING: {old_path} not found!")
        old_preds = None

    # Load 3M fold1_epoch1 checkpoint
    print("\n[3/4] Loading 3M fold1_epoch1 checkpoint...")
    ckpt_path = os.path.join(MODEL_DIR, "checkpoints_base_full", "fold1_epoch1.pt")
    if not os.path.exists(ckpt_path):
        print(f"  ERROR: {ckpt_path} not found!")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"  Checkpoint: fold={ckpt['fold']}, epoch={ckpt['epoch']}")
    print(f"  best_val_rmse: {ckpt.get('best_val_rmse', 'N/A')}")

    # Generate 3M fold1_epoch1 predictions
    print("\n[4/4] Generating 3M fold1_epoch1 predictions...")
    model = DeBERTaLoRA().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    new_preds = np.clip(predict(model, test_ds), 1.0, 5.0)
    del model; torch.cuda.empty_cache()
    print(f"  3M fold1_epoch1: mean={new_preds.mean():.4f}, std={new_preds.std():.4f}")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    if old_preds is not None:
        corr = np.corrcoef(old_preds, new_preds)[0, 1]
        diff = new_preds - old_preds
        abs_diff = np.abs(diff)

        print(f"\n{'Metric':<35} {'Old 1M fold1':<20} {'New 3M f1e1':<20}")
        print("-" * 75)
        print(f"{'Mean':<35} {old_preds.mean():<20.4f} {new_preds.mean():<20.4f}")
        print(f"{'Std':<35} {old_preds.std():<20.4f} {new_preds.std():<20.4f}")
        print(f"{'Min':<35} {old_preds.min():<20.4f} {new_preds.min():<20.4f}")
        print(f"{'Max':<35} {old_preds.max():<20.4f} {new_preds.max():<20.4f}")

        print(f"\nCorrelation (Pearson r): {corr:.6f}")
        print(f"Mean diff (new - old): {diff.mean():.6f}")
        print(f"Std diff: {diff.std():.6f}")
        print(f"Max |diff|: {abs_diff.max():.6f}")
        print(f"Samples with |diff| > 0.1: {(abs_diff > 0.1).sum():,} / {len(diff):,} ({(abs_diff > 0.1).mean()*100:.1f}%)")
        print(f"Samples with |diff| > 0.5: {(abs_diff > 0.5).sum():,} / {len(diff):,} ({(abs_diff > 0.5).mean()*100:.1f}%)")

        # VE comparison
        print(f"\nVariance Expansion (VE) comparison:")
        old_ve = variance_expand(old_preds, target_mean, target_std)
        new_ve = variance_expand(new_preds, target_mean, target_std)
        print(f"  Old VE: mean={old_ve.mean():.4f}, std={old_ve.std():.4f}")
        print(f"  New VE: mean={new_ve.mean():.4f}, std={new_ve.std():.4f}")
        ve_corr = np.corrcoef(old_ve, new_ve)[0, 1]
        print(f"  VE correlation: {ve_corr:.6f}")

        # Save predictions for later use
        np.save(os.path.join(MODEL_DIR, "deberta_3m_f1e1_test.npy"), new_preds)
        print(f"\nSaved: deberta_3m_f1e1_test.npy")

        # Generate blend candidates
        print(f"\nBlend candidates (local OOF proxy):")
        for w_old in [70, 80, 90, 95]:
            w_new = 100 - w_old
            blend = np.clip(w_old/100 * old_preds + w_new/100 * new_preds, 1.0, 5.0)
            blend_ve = variance_expand(blend, target_mean, target_std)
            print(f"  Old {w_old}% + New {w_new}%: mean={blend.mean():.4f}, std={blend.std():.4f}, VE std={blend_ve.std():.4f}")

    print("\n=== Done (LOCAL ONLY - no Kaggle submission) ===")


if __name__ == "__main__":
    main()

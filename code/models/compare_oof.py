#!/usr/bin/env python
"""Compare OOF performance: Old 1M fold1 vs New 3x3 full 3M model.

Usage: python compare_oof.py [--old-ckpt-dir checkpoints_lora] [--new-ckpt-dir checkpoints_base_full]

Expected outputs:
  - Per-fold and overall OOF RMSE for both models
  - Test prediction comparison (mean, std, correlation)
  - VE comparison with actual label distribution
  - Summary table printed to stdout
"""

import os, sys, time, argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast
from sklearn.model_selection import KFold

ROOT = "/hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating"
MODEL_DIR = os.path.join(ROOT, "artifacts", "models")
FEAT_DIR = os.path.join(ROOT, "artifacts", "features")

MODEL_NAME = "microsoft/deberta-v3-base"
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGET = ["query_proj", "value_proj"]
N_TASKS = 4
N_FOLDS = 3
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


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = DeBERTaLoRA().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def find_best_checkpoints(ckpt_dir):
    """Find highest epoch checkpoint per fold."""
    best = {}
    for f in os.listdir(ckpt_dir):
        if f.endswith(".pt") and f.startswith("fold"):
            parts = f.replace(".pt", "").split("_")
            fold = int(parts[0].replace("fold", ""))
            epoch = int(parts[1].replace("epoch", ""))
            if fold not in best or epoch > best[fold]:
                best[fold] = epoch
    return best


def variance_expand(preds, target_mean, target_std):
    """Apply variance expansion to predictions."""
    p_mean, p_std = preds.mean(), preds.std()
    if p_std < 1e-9:
        return preds
    return np.clip((preds - p_mean) / p_std * target_std + target_mean, 1.0, 5.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-ckpt-dir", default=None,
                        help="Old model checkpoint directory (relative to MODEL_DIR). "
                             "Auto-detects: checkpoints_v3base_1m > checkpoints_lora")
    parser.add_argument("--new-ckpt-dir", default="checkpoints_base_full",
                        help="New model checkpoint directory (relative to MODEL_DIR)")
    args = parser.parse_args()

    new_dir = os.path.join(MODEL_DIR, args.new_ckpt_dir)

    # Auto-detect old checkpoint directory
    if args.old_ckpt_dir:
        old_dir = os.path.join(MODEL_DIR, args.old_ckpt_dir)
    else:
        candidates = ["checkpoints_v3base_1m", "checkpoints_lora"]
        old_dir = None
        for c in candidates:
            p = os.path.join(MODEL_DIR, c)
            if os.path.exists(p) and any(f.endswith(".pt") for f in os.listdir(p)):
                old_dir = p
                print(f"  Auto-detected old model: {c}")
                break
        if old_dir is None:
            print(f"  WARNING: No old model checkpoints found in {candidates}")
            old_dir = os.path.join(MODEL_DIR, "checkpoints_v3base_1m")

    print("=" * 70)
    print("DeBERTa OOF Comparison: Old 1M fold1 vs New 3x3 Full 3M")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────
    print("\n[1/5] Loading data...")
    y_train = np.load(os.path.join(FEAT_DIR, "y_train.npy")).astype(np.float32)
    td = np.load(os.path.join(MODEL_DIR, "train_tokens.npz"), allow_pickle=True)
    tr_ids = torch.from_numpy(td["input_ids"]).to(torch.int32)
    tr_mask = torch.from_numpy(td["attention_mask"]).to(torch.int32)
    tr_tt = torch.from_numpy(td["token_type_ids"]).to(torch.int32)
    train_ds = DS(tr_ids, tr_mask, tr_tt)

    td2 = np.load(os.path.join(MODEL_DIR, "test_tokens.npz"), allow_pickle=True)
    test_ids = td2["ids"]
    t_ids = torch.from_numpy(td2["input_ids"]).to(torch.int32)
    t_mask = torch.from_numpy(td2["attention_mask"]).to(torch.int32)
    t_tt = torch.from_numpy(td2["token_type_ids"]).to(torch.int32)
    test_ds = DS(t_ids, t_mask, t_tt)

    print(f"  Train: {len(y_train)} samples, Test: {len(test_ids)} samples")
    print(f"  Label: mean={y_train.mean():.4f}, std={y_train.std():.4f}")

    # ── KFold splits (same as training) ───────────────────────────────
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    folds = list(kf.split(np.arange(len(y_train))))

    results = {}

    # ── Process OLD model ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"OLD MODEL: {args.old_ckpt_dir}")
    print(f"{'='*70}")

    if not os.path.exists(old_dir):
        print(f"  ERROR: Directory not found: {old_dir}")
        old_available = False
    else:
        old_ckpts = find_best_checkpoints(old_dir)
        old_available = len(old_ckpts) > 0
        if not old_available:
            print(f"  WARNING: No .pt checkpoints found in {old_dir}")

    if old_available:
        print(f"  Checkpoints found: {old_ckpts}")

        old_oof = np.zeros(len(y_train), dtype=np.float32)
        old_test_preds = []
        old_fold_rmses = []

        for fold, epoch in sorted(old_ckpts.items()):
            ckpt_path = os.path.join(old_dir, f"fold{fold}_epoch{epoch}.pt")
            print(f"\n  Fold {fold}, Epoch {epoch}:")
            model, ckpt = load_model(ckpt_path)
            val_rmse_ckpt = ckpt.get("best_val_rmse", "N/A")
            print(f"    Checkpoint best_val_rmse: {val_rmse_ckpt}")

            # OOF predictions on full 3M
            oof_preds = np.clip(predict(model, train_ds, bs=128), 1.0, 5.0)
            oof_rmse = np.sqrt(np.mean((oof_preds - y_train) ** 2))
            print(f"    Full 3M OOF RMSE: {oof_rmse:.5f}")
            print(f"    OOF: mean={oof_preds.mean():.4f}, std={oof_preds.std():.4f}")

            # Per-fold OOF (only on this fold's validation set)
            _, va_idx = folds[fold - 1]
            fold_oof_rmse = np.sqrt(np.mean((oof_preds[va_idx] - y_train[va_idx]) ** 2))
            print(f"    Fold {fold} val OOF RMSE: {fold_oof_rmse:.5f}")
            old_fold_rmses.append(fold_oof_rmse)

            old_oof += oof_preds / len(old_ckpts)

            # Test predictions
            test_preds = np.clip(predict(model, test_ds), 1.0, 5.0)
            old_test_preds.append(test_preds)
            print(f"    Test: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

            del model; torch.cuda.empty_cache()

        old_avg_test = np.clip(np.mean(old_test_preds, axis=0), 1.0, 5.0)
        old_avg_oof_rmse = np.sqrt(np.mean((old_oof - y_train) ** 2))

        results["old"] = {
            "oof_rmse": old_avg_oof_rmse,
            "fold_rmses": old_fold_rmses,
            "test_mean": old_avg_test.mean(),
            "test_std": old_avg_test.std(),
            "oof_mean": old_oof.mean(),
            "oof_std": old_oof.std(),
            "test_preds": old_avg_test,
        }

        print(f"\n  OLD Model Summary:")
        print(f"    Ensemble OOF RMSE (full 3M): {old_avg_oof_rmse:.5f}")
        print(f"    Per-fold val OOF RMSE: {[f'{r:.5f}' for r in old_fold_rmses]}")
        print(f"    Test: mean={old_avg_test.mean():.4f}, std={old_avg_test.std():.4f}")
    else:
        print("  No checkpoints found, skipping old model.")

    # ── Process NEW model ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"NEW MODEL: {args.new_ckpt_dir}")
    print(f"{'='*70}")

    if not os.path.exists(new_dir):
        print(f"  ERROR: Directory not found: {new_dir}")
        new_available = False
    else:
        new_ckpts = find_best_checkpoints(new_dir)
        new_available = len(new_ckpts) > 0

    if new_available:
        print(f"  Checkpoints found: {new_ckpts}")

        new_oof = np.zeros(len(y_train), dtype=np.float32)
        new_test_preds = []
        new_fold_rmses = []

        for fold, epoch in sorted(new_ckpts.items()):
            ckpt_path = os.path.join(new_dir, f"fold{fold}_epoch{epoch}.pt")
            print(f"\n  Fold {fold}, Epoch {epoch}:")
            model, ckpt = load_model(ckpt_path)
            val_rmse_ckpt = ckpt.get("best_val_rmse", "N/A")
            print(f"    Checkpoint best_val_rmse: {val_rmse_ckpt}")

            # OOF predictions on full 3M
            oof_preds = np.clip(predict(model, train_ds, bs=128), 1.0, 5.0)
            oof_rmse = np.sqrt(np.mean((oof_preds - y_train) ** 2))
            print(f"    Full 3M OOF RMSE: {oof_rmse:.5f}")
            print(f"    OOF: mean={oof_preds.mean():.4f}, std={oof_preds.std():.4f}")

            # Per-fold OOF
            _, va_idx = folds[fold - 1]
            fold_oof_rmse = np.sqrt(np.mean((oof_preds[va_idx] - y_train[va_idx]) ** 2))
            print(f"    Fold {fold} val OOF RMSE: {fold_oof_rmse:.5f}")
            new_fold_rmses.append(fold_oof_rmse)

            new_oof += oof_preds / len(new_ckpts)

            # Test predictions
            test_preds = np.clip(predict(model, test_ds), 1.0, 5.0)
            new_test_preds.append(test_preds)
            print(f"    Test: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

            del model; torch.cuda.empty_cache()

        new_avg_test = np.clip(np.mean(new_test_preds, axis=0), 1.0, 5.0)
        new_avg_oof_rmse = np.sqrt(np.mean((new_oof - y_train) ** 2))

        results["new"] = {
            "oof_rmse": new_avg_oof_rmse,
            "fold_rmses": new_fold_rmses,
            "test_mean": new_avg_test.mean(),
            "test_std": new_avg_test.std(),
            "oof_mean": new_oof.mean(),
            "oof_std": new_oof.std(),
            "test_preds": new_avg_test,
        }

        print(f"\n  NEW Model Summary:")
        print(f"    Ensemble OOF RMSE (full 3M): {new_avg_oof_rmse:.5f}")
        print(f"    Per-fold val OOF RMSE: {[f'{r:.5f}' for r in new_fold_rmses]}")
        print(f"    Test: mean={new_avg_test.mean():.4f}, std={new_avg_test.std():.4f}")
    else:
        print("  No checkpoints found, skipping new model.")

    # ── Comparison ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")

    if "old" in results and "new" in results:
        r_old = results["old"]
        r_new = results["new"]

        print(f"\n{'Metric':<35} {'OLD (1M fold1)':<20} {'NEW (3x3 full)':<20} {'Delta':<15}")
        print("-" * 90)
        print(f"{'OOF RMSE (full 3M)':<35} {r_old['oof_rmse']:<20.5f} {r_new['oof_rmse']:<20.5f} {r_new['oof_rmse']-r_old['oof_rmse']:+.5f}")
        print(f"{'OOF pred mean':<35} {r_old['oof_mean']:<20.4f} {r_new['oof_mean']:<20.4f}")
        print(f"{'OOF pred std':<35} {r_old['oof_std']:<20.4f} {r_new['oof_std']:<20.4f}")
        print(f"{'Test pred mean':<35} {r_old['test_mean']:<20.4f} {r_new['test_mean']:<20.4f}")
        print(f"{'Test pred std':<35} {r_old['test_std']:<20.4f} {r_new['test_std']:<20.4f}")
        print(f"{'Label mean':<35} {y_train.mean():<20.4f}")
        print(f"{'Label std':<35} {y_train.std():<20.4f}")

        # Per-fold comparison
        print(f"\nPer-fold val OOF RMSE:")
        for i in range(min(len(r_old["fold_rmses"]), len(r_new["fold_rmses"]))):
            print(f"  Fold {i+1}: OLD={r_old['fold_rmses'][i]:.5f}  NEW={r_new['fold_rmses'][i]:.5f}  delta={r_new['fold_rmses'][i]-r_old['fold_rmses'][i]:+.5f}")

        # Test prediction correlation
        corr = np.corrcoef(r_old["test_preds"], r_new["test_preds"])[0, 1]
        print(f"\nTest prediction correlation (Pearson r): {corr:.6f}")

        # VE comparison
        print(f"\nVariance Expansion (VE) comparison:")
        target_std = y_train.std()
        target_mean = y_train.mean()
        old_ve = variance_expand(r_old["test_preds"], target_mean, target_std)
        new_ve = variance_expand(r_new["test_preds"], target_mean, target_std)
        print(f"  OLD VE: mean={old_ve.mean():.4f}, std={old_ve.std():.4f}")
        print(f"  NEW VE: mean={new_ve.mean():.4f}, std={new_ve.std():.4f}")

        # Prediction difference analysis
        diff = r_new["test_preds"] - r_old["test_preds"]
        print(f"\nTest prediction difference (NEW - OLD):")
        print(f"  Mean diff: {diff.mean():.6f}")
        print(f"  Std diff:  {diff.std():.6f}")
        print(f"  Max |diff|: {np.abs(diff).max():.6f}")
        print(f"  Samples with |diff| > 0.5: {(np.abs(diff) > 0.5).sum()} / {len(diff)}")
        print(f"  Samples with |diff| > 1.0: {(np.abs(diff) > 1.0).sum()} / {len(diff)}")

        # Save comparison results
        out_dir = os.path.join(ROOT, "output")
        np.save(os.path.join(MODEL_DIR, "compare_old_test.npy"), r_old["test_preds"])
        np.save(os.path.join(MODEL_DIR, "compare_new_test.npy"), r_new["test_preds"])
        np.save(os.path.join(MODEL_DIR, "compare_old_oof.npy"), r_old.get("oof_mean", np.array([])))
        np.save(os.path.join(MODEL_DIR, "compare_new_oof.npy"), r_new.get("oof_mean", np.array([])))
        print(f"\nPredictions saved to {MODEL_DIR}/compare_*.npy")

    elif "old" in results:
        print("\nOnly OLD model results available.")
    elif "new" in results:
        print("\nOnly NEW model results available.")
    else:
        print("\nNo results to compare.")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()

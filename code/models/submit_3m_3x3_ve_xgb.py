#!/usr/bin/env python
"""
Generate submission: 3M 3×3 DeBERTa-base VE + XGBoost blend.

Steps:
  1. Regenerate DeBERTa test/OOF predictions from checkpoints_base_full/ (includes fold2_epoch3)
  2. Apply variance expansion to DeBERTa predictions
  3. Find optimal blend ratio via OOF grid search
  4. Save candidate submission CSV

Usage:
  python code/models/submit_3m_3x3_ve_xgb.py
  python code/models/submit_3m_3x3_ve_xgb.py --skip-deberta  # skip step 1 if .npy already up-to-date
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
OUTPUT_DIR = os.path.join(ROOT, "output")
CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints_base_full")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def find_best_checkpoints():
    best = {}
    for f in os.listdir(CKPT_DIR):
        if f.endswith(".pt") and f.startswith("fold"):
            parts = f.replace(".pt", "").split("_")
            fold = int(parts[0].replace("fold", ""))
            epoch = int(parts[1].replace("epoch", ""))
            if fold not in best or epoch > best[fold]:
                best[fold] = epoch
    return best


def step1_regenerate_deberta():
    """Regenerate DeBERTa test + OOF predictions from all fold checkpoints."""
    print("=" * 70)
    print("Step 1: Regenerate DeBERTa predictions (3M 3×3)")
    print("=" * 70)

    # Load data
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

    # KFold splits (same as training)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_splits = list(kf.split(np.arange(len(y_train))))

    # Find checkpoints
    best_ckpts = find_best_checkpoints()
    print(f"Checkpoints: {best_ckpts}")
    assert len(best_ckpts) == N_FOLDS, f"Expected {N_FOLDS} folds, found {len(best_ckpts)}"

    test_preds_list = []
    oof_preds_all = np.zeros(len(y_train), dtype=np.float32)
    oof_counts = np.zeros(len(y_train), dtype=np.int32)

    for fold, epoch in sorted(best_ckpts.items()):
        ckpt_path = os.path.join(CKPT_DIR, f"fold{fold}_epoch{epoch}.pt")
        print(f"\n  Fold {fold}, Epoch {epoch}:")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = DeBERTaLoRA().to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])

        # Test predictions
        test_preds = np.clip(predict(model, test_ds), 1.0, 5.0)
        test_preds_list.append(test_preds)
        print(f"    Test: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

        # OOF: only on this fold's validation set
        fold_idx = fold - 1
        _, va_idx = fold_splits[fold_idx]
        va_ds = DS(tr_ids[va_idx], tr_mask[va_idx], tr_tt[va_idx])
        va_oof = np.clip(predict(model, va_ds, bs=128), 1.0, 5.0)
        oof_preds_all[va_idx] = va_oof
        oof_counts[va_idx] += 1
        fold_rmse = np.sqrt(np.mean((va_oof - y_train[va_idx]) ** 2))
        print(f"    OOF RMSE (fold val): {fold_rmse:.5f}")

        del model; torch.cuda.empty_cache()

    # Verify coverage
    assert oof_counts.min() == 1 and oof_counts.max() == 1, "OOF coverage error"

    # Average test predictions
    avg_test = np.clip(np.mean(test_preds_list, axis=0), 1.0, 5.0)
    true_oof_rmse = np.sqrt(np.mean((oof_preds_all - y_train) ** 2))
    print(f"\n  True OOF RMSE: {true_oof_rmse:.5f}")
    print(f"  Test: mean={avg_test.mean():.4f}, std={avg_test.std():.4f}")

    # Save
    np.save(os.path.join(MODEL_DIR, "deberta_base_oof.npy"), oof_preds_all)
    np.save(os.path.join(MODEL_DIR, "deberta_base_ensemble_test.npy"), avg_test)
    np.save(os.path.join(MODEL_DIR, "deberta_3m_3x3_oof.npy"), oof_preds_all)
    np.save(os.path.join(MODEL_DIR, "deberta_3m_3x3_test.npy"), avg_test)
    print(f"  Saved: deberta_3m_3x3_oof.npy, deberta_3m_3x3_test.npy")

    return oof_preds_all, avg_test, test_ids, y_train


def step2_variance_expand(preds, y_train):
    """Apply variance expansion."""
    target_std = y_train.std()
    target_mean = y_train.mean()
    pred_std = preds.std()
    pred_mean = preds.mean()
    ve = np.clip((preds - pred_mean) / pred_std * target_std + target_mean, 1.0, 5.0)
    print(f"  VE: mean={ve.mean():.4f}, std={ve.std():.4f} (target: mean={target_mean:.4f}, std={target_std:.4f})")
    return ve


def step3_find_optimal_ratio(deberta_oof_ve, xgb_oof, y_train):
    """Grid search for optimal DeBERTa:XGBoost blend ratio on OOF."""
    print("\n" + "=" * 70)
    print("Step 3: Find optimal blend ratio (OOF grid search)")
    print("=" * 70)

    # DeBERTa-only and XGBoost-only baselines
    deb_rmse = np.sqrt(np.mean((deberta_oof_ve - y_train) ** 2))
    xgb_rmse = np.sqrt(np.mean((xgb_oof - y_train) ** 2))
    print(f"\n  DeBERTa VE OOF RMSE: {deb_rmse:.5f}")
    print(f"  XGBoost OOF RMSE:    {xgb_rmse:.5f}")

    # Grid search: deberta_weight from 0.50 to 1.00
    print(f"\n  {'deb_weight':>10} {'xgb_weight':>10} {'OOF RMSE':>12} {'vs deb_only':>12}")
    print(f"  {'-'*46}")

    best_rmse = float("inf")
    best_w = 1.0

    for w_deb_x100 in range(50, 101):
        w_deb = w_deb_x100 / 100.0
        w_xgb = 1.0 - w_deb
        blended = np.clip(w_deb * deberta_oof_ve + w_xgb * xgb_oof, 1.0, 5.0)
        rmse = np.sqrt(np.mean((blended - y_train) ** 2))
        delta = rmse - deb_rmse
        marker = " <-- best" if rmse < best_rmse else ""
        if w_deb_x100 % 5 == 0 or rmse < best_rmse:
            print(f"  {w_deb:>10.2f} {w_xgb:>10.2f} {rmse:>12.5f} {delta:>+12.5f}{marker}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = w_deb

    # Fine-grained search around best
    print(f"\n  Fine search around best ({best_w:.2f}):")
    for w_deb_x1000 in range(int((best_w - 0.05) * 1000), int((best_w + 0.05) * 1000) + 1, 5):
        w_deb = w_deb_x1000 / 1000.0
        if w_deb < 0 or w_deb > 1:
            continue
        w_xgb = 1.0 - w_deb
        blended = np.clip(w_deb * deberta_oof_ve + w_xgb * xgb_oof, 1.0, 5.0)
        rmse = np.sqrt(np.mean((blended - y_train) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = w_deb

    print(f"\n  Best ratio: DeBERTa {best_w:.3f} : XGBoost {1-best_w:.3f}")
    print(f"  Best OOF RMSE: {best_rmse:.5f}")
    print(f"  vs DeBERTa only: {(best_rmse - deb_rmse):+.5f} ({(best_rmse/deb_rmse-1)*100:+.2f}%)")
    print(f"  vs XGBoost only: {(best_rmse - xgb_rmse):+.5f} ({(best_rmse/xgb_rmse-1)*100:+.2f}%)")

    return best_w, best_rmse


def step4_generate_submission(deberta_test_ve, xgb_test, test_ids, w_deb, oof_rmse):
    """Generate final submission CSV."""
    print("\n" + "=" * 70)
    print("Step 4: Generate submission")
    print("=" * 70)

    w_xgb = 1.0 - w_deb
    blended = np.clip(w_deb * deberta_test_ve + w_xgb * xgb_test, 1.0, 5.0)

    deb_pct = int(round(w_deb * 100))
    xgb_pct = int(round(w_xgb * 100))
    oof_str = f"{oof_rmse:.4f}".replace(".", "p")
    filename = f"candidate-3m-3x3-ve{deb_pct}-xgb{xgb_pct}-oof{oof_str}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)

    import pandas as pd
    pd.DataFrame({"id": test_ids, "rating": blended}).to_csv(filepath, index=False)

    print(f"\n  Submission: {filename}")
    print(f"  Blend: DeBERTa VE {deb_pct}% + XGBoost {xgb_pct}%")
    print(f"  OOF RMSE: {oof_rmse:.5f}")
    print(f"  Test: mean={blended.mean():.4f}, std={blended.std():.4f}")
    print(f"  Saved: {filepath}")

    return filepath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-deberta", action="store_true",
                        help="Skip DeBERTa prediction regeneration (reuse existing .npy)")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 70)
    print("Submission: 3M 3×3 DeBERTa-base VE + XGBoost")
    print("=" * 70)

    y_train = np.load(os.path.join(FEAT_DIR, "y_train.npy")).astype(np.float32)

    # Step 1: DeBERTa predictions
    if args.skip_deberta:
        print("\n[Skip] Loading existing DeBERTa predictions...")
        oof_path = os.path.join(MODEL_DIR, "deberta_3m_3x3_oof.npy")
        test_path = os.path.join(MODEL_DIR, "deberta_3m_3x3_test.npy")
        if not os.path.exists(oof_path):
            oof_path = os.path.join(MODEL_DIR, "deberta_base_full_oof.npy")
            test_path = os.path.join(MODEL_DIR, "deberta_base_full_test.npy")
        deberta_oof = np.load(oof_path)
        deberta_test = np.load(test_path)
        td2 = np.load(os.path.join(MODEL_DIR, "test_tokens.npz"), allow_pickle=True)
        test_ids = td2["ids"]
    else:
        deberta_oof, deberta_test, test_ids, _ = step1_regenerate_deberta()

    # Step 2: VE
    print("\n" + "=" * 70)
    print("Step 2: Variance Expansion")
    print("=" * 70)
    deberta_oof_ve = step2_variance_expand(deberta_oof, y_train)
    deberta_test_ve = step2_variance_expand(deberta_test, y_train)

    # Step 3: Load XGBoost OOF and find optimal ratio
    xgb_oof_path = os.path.join(FEAT_DIR, "xgboost_full_oof.npy")
    xgb_test_path = os.path.join(FEAT_DIR, "xgboost_full_test.npy")
    if not os.path.exists(xgb_oof_path):
        print(f"\nERROR: {xgb_oof_path} not found. Run xgboost_full.py first.")
        sys.exit(1)
    xgb_oof = np.load(xgb_oof_path)
    xgb_test = np.load(xgb_test_path)
    print(f"\n  XGBoost OOF: mean={xgb_oof.mean():.4f}, std={xgb_oof.std():.4f}")
    print(f"  XGBoost Test: mean={xgb_test.mean():.4f}, std={xgb_test.std():.4f}")

    best_w, best_oof = step3_find_optimal_ratio(deberta_oof_ve, xgb_oof, y_train)

    # Step 4: Generate submission
    filepath = step4_generate_submission(deberta_test_ve, xgb_test, test_ids, best_w, best_oof)

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

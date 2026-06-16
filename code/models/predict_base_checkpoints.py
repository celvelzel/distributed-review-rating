#!/usr/bin/env python
"""Generate predictions from completed DeBERTa-base checkpoints."""
import os, sys, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast

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
        cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGET, lora_dropout=LORA_DROPOUT, bias="none")
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


def main():
    print("=" * 60, flush=True)
    print("Generate DeBERTa-base predictions from checkpoints", flush=True)
    print("=" * 60, flush=True)

    # Load test tokens
    td2 = np.load(os.path.join(MODEL_DIR, "test_tokens.npz"), allow_pickle=True)
    test_ids = td2["ids"]
    t_ids = torch.from_numpy(td2["input_ids"]).to(torch.int32)
    t_mask = torch.from_numpy(td2["attention_mask"]).to(torch.int32)
    t_tt = torch.from_numpy(td2["token_type_ids"]).to(torch.int32)
    test_ds = DS(t_ids, t_mask, t_tt)

    # Load y_train for OOF
    y_train = np.load(os.path.join(FEAT_DIR, "y_train.npy")).astype(np.float32)
    td = np.load(os.path.join(MODEL_DIR, "train_tokens.npz"), allow_pickle=True)
    tr_ids = torch.from_numpy(td["input_ids"]).to(torch.int32)
    tr_mask = torch.from_numpy(td["attention_mask"]).to(torch.int32)
    tr_tt = torch.from_numpy(td["token_type_ids"]).to(torch.int32)
    train_ds = DS(tr_ids, tr_mask, tr_tt)

    # Find best checkpoints (highest epoch for each fold)
    ckpts = sorted(os.listdir(CKPT_DIR))
    best_ckpts = {}
    for ck in ckpts:
        if ck.endswith(".pt"):
            parts = ck.replace(".pt", "").split("_")
            fold = int(parts[0].replace("fold", ""))
            epoch = int(parts[1].replace("epoch", ""))
            if fold not in best_ckpts or epoch > best_ckpts[fold]:
                best_ckpts[fold] = epoch

    print(f"\nBest checkpoints per fold: {best_ckpts}", flush=True)

    test_preds_list = []
    oof_rmse_list = []

    for fold, epoch in sorted(best_ckpts.items()):
        ckpt_path = os.path.join(CKPT_DIR, f"fold{fold}_epoch{epoch}.pt")
        print(f"\nFold {fold}, Epoch {epoch}:", flush=True)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = DeBERTaLoRA().to(DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])

        # Test predictions
        test_preds = np.clip(predict(model, test_ds), 1.0, 5.0)
        print(f"  Test: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}", flush=True)
        test_preds_list.append(test_preds)

        # OOF predictions
        oof_preds = np.clip(predict(model, train_ds, bs=128), 1.0, 5.0)
        oof_rmse = np.sqrt(np.mean((oof_preds - y_train)**2))
        print(f"  OOF RMSE: {oof_rmse:.5f}", flush=True)
        oof_rmse_list.append(oof_rmse)

        np.save(os.path.join(MODEL_DIR, f"deberta_base_fold{fold}_test.npy"), test_preds)
        np.save(os.path.join(MODEL_DIR, f"deberta_base_fold{fold}_oof.npy"), oof_preds)

        del model; torch.cuda.empty_cache()

    # Average all folds
    avg_test = np.clip(np.mean(test_preds_list, axis=0), 1.0, 5.0)
    avg_oof = np.mean(oof_rmse_list)
    print(f"\n{'='*60}", flush=True)
    print(f"Average OOF RMSE: {avg_oof:.5f}", flush=True)
    print(f"Average Test: mean={avg_test.mean():.4f}, std={avg_test.std():.4f}", flush=True)

    # Variance expansion
    target_std = y_train.std()
    pred_std = avg_test.std()
    scale = target_std / pred_std
    ve_test = np.clip((avg_test - avg_test.mean()) * scale + y_train.mean(), 1.0, 5.0)
    print(f"VE Test: mean={ve_test.mean():.4f}, std={ve_test.std():.4f}", flush=True)

    # Save
    np.save(os.path.join(MODEL_DIR, "deberta_base_ensemble_test.npy"), avg_test)
    np.save(os.path.join(MODEL_DIR, "deberta_base_ensemble_ve.npy"), ve_test)

    # Blends
    ridge_test = np.load(os.path.join(MODEL_DIR, "stacking_v2_test.npy")).astype(np.float32)
    for w in [85, 90, 95]:
        blend = np.clip(w/100 * ve_test + (100-w)/100 * ridge_test, 1.0, 5.0)
        pd.DataFrame({"id": test_ids, "rating": blend}).to_csv(
            os.path.join(OUTPUT_DIR, f"submission-base_ensemble_ve{w}_r{100-w}.csv"), index=False)

    pd.DataFrame({"id": test_ids, "rating": avg_test}).to_csv(
        os.path.join(OUTPUT_DIR, "submission-base_ensemble.csv"), index=False)

    import pandas as pd
    pd.DataFrame({"id": test_ids, "rating": ve_test}).to_csv(
        os.path.join(OUTPUT_DIR, "submission-base_ensemble_ve.csv"), index=False)

    print("Done!", flush=True)


if __name__ == "__main__":
    main()

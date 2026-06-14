#!/usr/bin/env python
"""Auto-generate best DeBERTa submission from all completed fold checkpoints.

Scans checkpoints_lora/ for completed folds, generates predictions,
applies post-processing, and creates submission CSV.
"""

import os
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "artifacts" / "models"
FEAT_DIR = ROOT / "artifacts" / "features"
OUTPUT_DIR = ROOT / "output"

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 128
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["query_proj", "value_proj"]
N_CLASSES = 5
N_TASKS = N_CLASSES - 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeBERTaLoRAModel(nn.Module):
    def __init__(self, model_name, num_tasks=N_TASKS):
        super().__init__()
        from transformers import AutoModel
        from peft import LoraConfig, get_peft_model
        base_model = AutoModel.from_pretrained(model_name)
        lora_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT, bias="none",
        )
        self.backbone = get_peft_model(base_model, lora_config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(base_model.config.hidden_size, num_tasks)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.classifier(self.dropout(pooled))


def logits_to_rating(logits):
    return 1.0 + torch.sigmoid(logits).sum(dim=1)


def predict_from_model(model, dataset, batch_size=64):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    preds = []
    with torch.no_grad():
        for batch in loader:
            ids, mask, ttids = [b.to(DEVICE) for b in batch[:3]]
            with torch.amp.autocast("cuda", enabled=True):
                preds.append(logits_to_rating(model(ids, mask, ttids)).cpu().numpy())
    return np.concatenate(preds)


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, *tensors):
        self.tensors = tensors
    def __len__(self):
        return len(self.tensors[0])
    def __getitem__(self, idx):
        return tuple(t[idx] for t in self.tensors)


def find_checkpoints():
    ckpt_dir = MODEL_DIR / "checkpoints_lora"
    ckpts = sorted(ckpt_dir.glob("fold*_epoch*.pt"))
    return ckpts


def main():
    t_start = time.perf_counter()
    print("=" * 60)
    print("Auto DeBERTa Submission Generator")
    print("=" * 60)

    ckpts = find_checkpoints()
    print(f"\nFound {len(ckpts)} checkpoints:")
    for c in ckpts:
        print(f"  {c.name}")

    if not ckpts:
        print("No checkpoints found!")
        return

    # Load test tokens
    print("\nLoading test tokens...")
    data = np.load(str(MODEL_DIR / "test_tokens.npz"), allow_pickle=True)
    test_ds = SimpleDataset(
        torch.from_numpy(data["input_ids"]).to(torch.int32),
        torch.from_numpy(data["attention_mask"]).to(torch.int32),
        torch.from_numpy(data["token_type_ids"]).to(torch.int32),
    )
    test_ids = data["ids"]
    print(f"  Test: {len(test_ids)} samples")

    # Load training labels for OOF
    y_train = np.load(str(FEAT_DIR / "y_train.npy")).astype(np.float32)
    train_data = np.load(str(MODEL_DIR / "train_tokens.npz"), allow_pickle=True)
    train_ds = SimpleDataset(
        torch.from_numpy(train_data["input_ids"]).to(torch.int32),
        torch.from_numpy(train_data["attention_mask"]).to(torch.int32),
        torch.from_numpy(train_data["token_type_ids"]).to(torch.int32),
    )

    # Generate predictions from each checkpoint
    test_preds_list = []
    oof_preds_list = []
    fold_names = []

    for ckpt_path in ckpts:
        print(f"\nProcessing {ckpt_path.name}...")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        model = DeBERTaLoRAModel(MODEL_NAME)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(DEVICE)

        # Test predictions
        test_preds = np.clip(predict_from_model(model, test_ds, batch_size=64), 1.0, 5.0)
        test_preds_list.append(test_preds)
        print(f"  Test: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

        # OOF predictions
        oof_preds = np.clip(predict_from_model(model, train_ds, batch_size=64), 1.0, 5.0)
        oof_preds_list.append(oof_preds)
        oof_rmse = np.sqrt(np.mean((oof_preds - y_train) ** 2))
        print(f"  OOF RMSE: {oof_rmse:.5f}")

        fold_name = ckpt_path.stem.split("_")[0]
        fold_names.append(fold_name)

        # Save individual fold
        np.save(str(MODEL_DIR / f"deberta_{fold_name}_test.npy"), test_preds)
        np.save(str(MODEL_DIR / f"deberta_{fold_name}_oof.npy"), oof_preds)

        model.cpu()
        del model
        torch.cuda.empty_cache()

    # Average all folds
    avg_test = np.clip(np.mean(test_preds_list, axis=0), 1.0, 5.0)
    avg_oof = np.clip(np.mean(oof_preds_list, axis=0), 1.0, 5.0)
    avg_oof_rmse = np.sqrt(np.mean((avg_oof - y_train) ** 2))

    print(f"\n{'='*60}")
    print(f"Ensemble of {len(ckpts)} folds:")
    print(f"  OOF RMSE: {avg_oof_rmse:.5f}")
    print(f"  Test: mean={avg_test.mean():.4f}, std={avg_test.std():.4f}")

    # Save ensemble
    np.save(str(MODEL_DIR / "deberta_lora_ensemble_test.npy"), avg_test)
    np.save(str(MODEL_DIR / "deberta_lora_ensemble_oof.npy"), avg_oof)

    # Try blending with Ridge stacking
    ridge_test = np.load(str(MODEL_DIR / "stacking_v2_test.npy")).astype(np.float32)

    best_rmse = float("inf")
    best_name = "deberta_only"
    best_test = avg_test

    combos = {"deberta_only": avg_test}

    for w_d in range(70, 101, 5):
        w_r = 100 - w_d
        blend = np.clip(w_d / 100 * avg_test + w_r / 100 * ridge_test, 1.0, 5.0)
        name = f"deberta{w_d}_ridge{w_r}"
        combos[name] = blend

    # Try variance matching
    target_std = y_train.std()
    pred_std = avg_test.std()
    if pred_std < target_std:
        scale = target_std / pred_std
        vm_test = np.clip((avg_test - avg_test.mean()) * scale + y_train.mean(), 1.0, 5.0)
        combos["deberta_vm"] = vm_test

        for w_d in range(70, 101, 5):
            w_r = 100 - w_d
            blend = np.clip(w_d / 100 * vm_test + w_r / 100 * ridge_test, 1.0, 5.0)
            name = f"deberta_vm{w_d}_ridge{w_r}"
            combos[name] = blend

    # We can't compute Kaggle RMSE, but save all variants
    for name, preds in combos.items():
        sub = pd.DataFrame({"id": test_ids, "rating": preds})
        sub.to_csv(OUTPUT_DIR / f"submission-{name}.csv", index=False)
        print(f"  {name}: mean={preds.mean():.4f}, std={preds.std():.4f}")

    # Save the default (deberta_only) as the main submission
    import pandas as pd
    sub = pd.DataFrame({"id": test_ids, "rating": avg_test})
    sub.to_csv(OUTPUT_DIR / "submission-deberta-ensemble.csv", index=False)
    print(f"\n  Main submission: submission-deberta-ensemble.csv")

    total_time = time.perf_counter() - t_start
    print(f"\n  Total time: {total_time:.1f}s")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate predictions from DeBERTa LoRA fold 1 checkpoint."""

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
        self.num_tasks = num_tasks

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        logits = self.classifier(self.dropout(pooled))
        return logits


def logits_to_rating(logits):
    return 1.0 + torch.sigmoid(logits).sum(dim=1)


def predict_from_dataset(model, dataset, batch_size=64):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids, attn_mask, token_type_ids = [b.to(DEVICE) for b in batch[:3]]
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(input_ids, attn_mask, token_type_ids)
                preds = logits_to_rating(logits)
            all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds)


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, token_type_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.token_type_ids[idx]


def main():
    t_start = time.perf_counter()
    print("=" * 60)
    print("DeBERTa LoRA: Generate predictions from fold 1 checkpoint")
    print("=" * 60)

    # Load checkpoint
    ckpt_path = MODEL_DIR / "checkpoints_lora" / "fold1_epoch1.pt"
    print(f"\n[1/4] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    print(f"  Fold: {ckpt['fold']}, Epoch: {ckpt['epoch']}")
    print(f"  Best val RMSE: {ckpt.get('best_val_rmse', 'N/A')}")

    # Load model
    print("\n[2/4] Loading model...")
    model = DeBERTaLoRAModel(MODEL_NAME)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    print(f"  Model loaded on {DEVICE}")

    # Load test tokens
    print("\n[3/4] Loading test tokens...")
    test_cache = MODEL_DIR / "test_tokens.npz"
    data = np.load(str(test_cache), allow_pickle=True)
    test_input_ids = torch.from_numpy(data["input_ids"]).to(torch.int32)
    test_attn_mask = torch.from_numpy(data["attention_mask"]).to(torch.int32)
    test_token_type_ids = torch.from_numpy(data["token_type_ids"]).to(torch.int32)
    test_ids = data["ids"]
    print(f"  Test: {test_input_ids.shape}")

    # Generate predictions
    print("\n[4/4] Generating predictions...")
    test_ds = SimpleDataset(test_input_ids, test_attn_mask, test_token_type_ids)
    test_preds = predict_from_dataset(model, test_ds, batch_size=64)
    test_preds = np.clip(test_preds, 1.0, 5.0)

    print(f"  Predictions: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

    # Save
    np.save(str(MODEL_DIR / "deberta_lora_fold1_test.npy"), test_preds)
    import pandas as pd
    sub = pd.DataFrame({"id": test_ids, "rating": test_preds})
    sub.to_csv(OUTPUT_DIR / "submission-deberta-lora-fold1.csv", index=False)
    print(f"\n  Submission saved: output/submission-deberta-lora-fold1.csv")

    # Also generate OOF predictions for fold 1 val set
    print("\n[BONUS] Generating fold 1 OOF predictions...")
    train_cache = MODEL_DIR / "train_tokens.npz"
    train_data = np.load(str(train_cache), allow_pickle=True)
    train_input_ids = torch.from_numpy(train_data["input_ids"]).to(torch.int32)
    train_attn_mask = torch.from_numpy(train_data["attention_mask"]).to(torch.int32)
    train_token_type_ids = torch.from_numpy(train_data["token_type_ids"]).to(torch.int32)
    y_train = np.load(str(FEAT_DIR / "y_train.npy")).astype(np.float32)

    # Generate OOF for all training data
    train_ds = SimpleDataset(train_input_ids, train_attn_mask, train_token_type_ids)
    oof_preds = predict_from_dataset(model, train_ds, batch_size=64)
    oof_preds = np.clip(oof_preds, 1.0, 5.0)

    oof_rmse = np.sqrt(np.mean((oof_preds - y_train) ** 2))
    print(f"  OOF RMSE (fold 1 only): {oof_rmse:.5f}")
    print(f"  OOF pred: mean={oof_preds.mean():.4f}, std={oof_preds.std():.4f}")

    np.save(str(MODEL_DIR / "deberta_lora_fold1_oof.npy"), oof_preds)

    total_time = time.perf_counter() - t_start
    print(f"\n  Total time: {total_time:.1f}s")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

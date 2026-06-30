#!/usr/bin/env python
"""Generate predictions from DeBERTa LoRA fold 1 checkpoint.

This script loads a single fold-1 checkpoint from the DeBERTa LoRA training
run and generates:
  1. Test-set predictions (deberta_lora_fold1_test.npy) — used for final
     Kaggle submission blending with stacking predictions.
  2. Train-set OOF predictions (deberta_lora_fold1_oof.npy) — used as a
     base-model column in the stacking ensemble.

The fold-1 checkpoint is specifically chosen because it produced the best
Kaggle leaderboard score (0.61734) when blended at 90% weight with stacking.

Workflow:
  [1/4] Load checkpoint from artifacts/models/checkpoints_lora/fold1_epoch1.pt
  [2/4] Reconstruct DeBERTa-v3-base + LoRA model and load trained weights
  [3/4] Load pre-tokenised test data from test_tokens.npz
  [4/4] Generate predictions with mixed-precision inference (batch_size=64)
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
    """DeBERTa-v3-base with LoRA adapters for ordinal regression.

    Mirrors the architecture in deberta_lora_1m.py: LoRA on query/value
    projections, mean pooling, and a linear classifier producing 4 binary
    logits for the CORAL ordinal regression framework.
    """

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
        """Forward pass returning 4 binary CORAL logits."""
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = outputs.last_hidden_state
        # Mean pooling with attention mask to exclude padding tokens
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        logits = self.classifier(self.dropout(pooled))
        return logits


def logits_to_rating(logits):
    """Convert CORAL binary logits to a continuous rating in [1, 5].

    Sigmoid each of the 4 logits to get P(rating > k+1), then sum:
    rating = 1 + Σ sigmoid(logit_k). Output range: (1.0, 5.0).
    """
    return 1.0 + torch.sigmoid(logits).sum(dim=1)


def predict_from_dataset(model, dataset, batch_size=64):
    """Run inference on a dataset and return predicted ratings as a numpy array.

    Uses mixed-precision (autocast) for faster inference on GPU.
    Predictions are converted from CORAL logits to continuous ratings via
    logits_to_rating(), which applies sigmoid + sum to get values in [1, 5].
    """
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

    # 加载 fold1 的检查点（该 fold 在 Kaggle 上取得最佳 0.61734 成绩）
    ckpt_path = MODEL_DIR / "checkpoints_lora" / "fold1_epoch1.pt"
    print(f"\n[1/4] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    print(f"  Fold: {ckpt['fold']}, Epoch: {ckpt['epoch']}")
    print(f"  Best val RMSE: {ckpt.get('best_val_rmse', 'N/A')}")

    # 重建模型结构并加载训练好的权重
    print("\n[2/4] Loading model...")
    model = DeBERTaLoRAModel(MODEL_NAME)
    model.load_state_dict(ckpt["model_state_dict"])  # 从检查点恢复 LoRA 权重
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

    # 批量推理: FP16 混合精度加速，batch_size=64; 预测裁剪到 [1, 5]
    print("\n[4/4] Generating predictions...")
    test_ds = SimpleDataset(test_input_ids, test_attn_mask, test_token_type_ids)
    test_preds = predict_from_dataset(model, test_ds, batch_size=64)
    test_preds = np.clip(test_preds, 1.0, 5.0)  # 裁剪到合法评分范围

    print(f"  Predictions: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

    # 保存测试预测: NPY 供后续混合, CSV 供 Kaggle 直接提交
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

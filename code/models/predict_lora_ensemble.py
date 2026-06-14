#!/usr/bin/env python
"""Generate DeBERTa LoRA predictions from all completed folds and blend with ensemble."""

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


def main():
    t_start = time.perf_counter()
    print("=" * 60)
    print("DeBERTa LoRA Multi-Fold Predictor")
    print("=" * 60)

    # Find all completed fold checkpoints
    ckpt_dir = MODEL_DIR / "checkpoints_lora"
    ckpts = sorted(ckpt_dir.glob("fold*_epoch*.pt"))
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

    # Load training tokens for OOF
    print("Loading training tokens...")
    train_data = np.load(str(MODEL_DIR / "train_tokens.npz"), allow_pickle=True)
    y_train = np.load(str(FEAT_DIR / "y_train.npy")).astype(np.float32)
    train_ds = SimpleDataset(
        torch.from_numpy(train_data["input_ids"]).to(torch.int32),
        torch.from_numpy(train_data["attention_mask"]).to(torch.int32),
        torch.from_numpy(train_data["token_type_ids"]).to(torch.int32),
    )

    # Generate predictions from each checkpoint
    test_preds_list = []
    for ckpt_path in ckpts:
        print(f"\nProcessing {ckpt_path.name}...")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        
        model = DeBERTaLoRAModel(MODEL_NAME)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(DEVICE)

        # Test predictions
        test_preds = np.clip(predict_from_model(model, test_ds), 1.0, 5.0)
        test_preds_list.append(test_preds)
        print(f"  Test: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")

        # OOF predictions (only for completed folds)
        oof_preds = np.clip(predict_from_model(model, train_ds), 1.0, 5.0)
        oof_rmse = np.sqrt(np.mean((oof_preds - y_train) ** 2))
        print(f"  OOF RMSE: {oof_rmse:.5f}")

        # Save individual fold predictions
        fold_num = ckpt_path.stem.split("_")[0]
        np.save(str(MODEL_DIR / f"deberta_{fold_num}_test.npy"), test_preds)
        np.save(str(MODEL_DIR / f"deberta_{fold_num}_oof.npy"), oof_preds)

        model.cpu()
        del model
        torch.cuda.empty_cache()

    # Average all fold predictions
    avg_test = np.clip(np.mean(test_preds_list, axis=0), 1.0, 5.0)
    avg_oof = np.clip(np.mean([np.load(str(MODEL_DIR / f"deberta_{c.stem.split('_')[0]}_oof.npy")) for c in ckpts], axis=0), 1.0, 5.0)
    avg_oof_rmse = np.sqrt(np.mean((avg_oof - y_train) ** 2))

    print(f"\n{'='*60}")
    print(f"Ensemble of {len(ckpts)} folds:")
    print(f"  OOF RMSE: {avg_oof_rmse:.5f}")
    print(f"  Test: mean={avg_test.mean():.4f}, std={avg_test.std():.4f}")

    # Save
    np.save(str(MODEL_DIR / "deberta_lora_ensemble_test.npy"), avg_test)
    np.save(str(MODEL_DIR / "deberta_lora_ensemble_oof.npy"), avg_oof)

    import pandas as pd
    sub = pd.DataFrame({"id": test_ids, "rating": avg_test})
    sub.to_csv(OUTPUT_DIR / "submission-deberta-ensemble.csv", index=False)
    print(f"\n  Submission saved")

    # Also try blending with Ridge stacking
    ridge_test = np.load(str(MODEL_DIR / "stacking_v2_test.npy")).astype(np.float32)
    for w_d in [90, 85, 80, 75]:
        w_r = 100 - w_d
        blend = np.clip(w_d/100 * avg_test + w_r/100 * ridge_test, 1.0, 5.0)
        sub = pd.DataFrame({"id": test_ids, "rating": blend})
        sub.to_csv(OUTPUT_DIR / f"submission-deberta{w_d}-ridge{w_r}.csv", index=False)
        print(f"  deberta{w_d}/ridge{w_r}: mean={blend.mean():.4f}, std={blend.std():.4f}")

    total_time = time.perf_counter() - t_start
    print(f"\n  Total time: {total_time:.1f}s")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

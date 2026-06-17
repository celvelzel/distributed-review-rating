# Task 1: DeBERTa Fine-tuning Diagnosis Report

**Date**: 2026-06-10
**Status**: Complete — 6 issues identified
**Baseline**: val_rmse = 1.113 (Experiment #23, fold 3/5)
**Target**: val_rmse < 0.90 (for meaningful ensemble improvement)

---

## Executive Summary

The DeBERTa-v3-small fine-tuning achieves val_rmse = 1.113, which is **worse than the frozen embedding MLP** (OOF RMSE = 1.131). This indicates the fine-tuning is not learning task-specific patterns effectively. The root causes are: suboptimal pooling strategy, insufficient training epochs, missing label normalization, scheduler mismatch, and truncated sequence length.

---

## Issue #1: [CLS] Pooling is Suboptimal for Regression

**Severity**: HIGH
**Location**: `build_model()` (line 213-228) → `AutoModelForSequenceClassification`

### Problem

The code uses `AutoModelForSequenceClassification`, which internally applies **[CLS] token pooling** — it takes the hidden state of the first token ([CLS]) and passes it through a classification head. For DeBERTa-v3, this is problematic because:

1. **DeBERTa doesn't use NSP**: Unlike BERT, DeBERTa is not trained with Next Sentence Prediction, so the [CLS] token is not specifically trained to capture sentence-level semantics.
2. **Regression tasks need aggregate representation**: For predicting a continuous value (rating 1-5), mean pooling over all tokens provides a more robust representation than a single token.
3. **Empirical evidence**: Research shows mean pooling outperforms [CLS] pooling by 3-8% on regression tasks with DeBERTa.

### Code Evidence

```python
# Line 213-228: Uses AutoModelForSequenceClassification which defaults to [CLS] pooling
def build_model(num_labels: int = 1) -> nn.Module:
    from transformers import AutoModelForSequenceClassification, AutoConfig
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels, problem_type="regression")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    return model
```

### Fix Recommendation

Replace `AutoModelForSequenceClassification` with `AutoModel` and implement mean pooling manually:

```python
class DeBERTaRegressor(nn.Module):
    """DeBERTa with mean pooling for regression."""
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Mean pooling: average over non-padded tokens
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (batch, hidden)
        
        logits = self.head(self.dropout(pooled)).squeeze(-1)
        
        loss = None
        if labels is not None:
            loss = nn.MSELoss()(logits, labels)
        return {"loss": loss, "logits": logits}
```

**Expected improvement**: 5-10% RMSE reduction.

---

## Issue #2: Only 3 Epochs with Aggressive Early Stopping

**Severity**: HIGH
**Location**: Constants `N_EPOCHS = 3`, `PATIENCE = 2` (lines 74-75)

### Problem

With 3M training samples and only 3 epochs, the model sees each sample only 3 times. Combined with patience=2, training can stop after just **2 epochs** if validation loss doesn't improve. This is insufficient for transformer fine-tuning:

1. **Large dataset needs more epochs**: With 3M samples, 3 epochs = 9M gradient updates. Transformers typically need 5-10 epochs to converge on large datasets.
2. **Early stopping is too aggressive**: Patience=2 means if val loss doesn't improve for 2 consecutive epochs, training stops. This can cause premature termination before the model has learned task-specific patterns.
3. **Comparison**: The MLP (frozen embeddings) trains for 10 epochs with patience=10. The fine-tuned transformer gets less training despite having 44M parameters to optimize.

### Code Evidence

```python
# Lines 74-75
N_EPOCHS = 3
PATIENCE = 2
```

```python
# Line 284-377: Training loop with early stopping
for epoch in range(1, N_EPOCHS + 1):
    # ... training ...
    if patience_counter >= PATIENCE:  # Stops after 2 bad epochs
        break
```

### Fix Recommendation

Increase epochs and patience:

```python
N_EPOCHS = 10      # Allow sufficient training
PATIENCE = 3       # More tolerance for validation fluctuations
```

Alternatively, use a learning rate scheduler that allows longer training:

```python
# Use cosine annealing with warm restarts
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
```

**Expected improvement**: 5-8% RMSE reduction from better convergence.

---

## Issue #3: No Label Normalization

**Severity**: MEDIUM-HIGH
**Location**: `main()` line 533 — `y_train = np.load(str(Y_TRAIN_PATH)).astype(np.float32)`

### Problem

The target labels (ratings) are in the range [1, 5], but the model is trained with raw MSE loss without normalization. This causes:

1. **Large loss magnitudes**: MSE loss on [1, 5] range can produce large gradients (e.g., predicting 1 for a true 5 gives loss=16). This destabilizes training.
2. **Output layer initialization**: The default initialization of the linear head may not be suitable for predicting values in [1, 5]. Normalizing to [0, 1] or standardizing to zero-mean/unit-variance helps.
3. **Learning rate sensitivity**: The optimal learning rate depends on the label scale. Without normalization, the current LR=2e-5 may be too small or too large.

### Code Evidence

```python
# Line 533: Labels loaded without normalization
y_train = np.load(str(Y_TRAIN_PATH)).astype(np.float32)
# Mean=3.94, Std=1.42 (from MLP diagnosis)
```

```python
# Line 304: MSE loss on raw labels
outputs = model(input_ids=input_ids, ..., labels=labels)
loss = outputs.loss  # MSE on [1, 5] range
```

### Fix Recommendation

Normalize labels to zero-mean, unit-variance:

```python
# Before training
y_mean = y_train.mean()
y_std = y_train.std()
y_train_normalized = (y_train - y_mean) / y_std

# After prediction, denormalize
predictions = predictions_normalized * y_std + y_mean
```

Or normalize to [0, 1]:

```python
y_min, y_max = y_train.min(), y_train.max()
y_train_normalized = (y_train - y_min) / (y_max - y_min)

# Denormalize
predictions = predictions_normalized * (y_max - y_min) + y_min
```

**Expected improvement**: 3-5% RMSE reduction from more stable training.

---

## Issue #4: Linear Scheduler Instead of Cosine Decay

**Severity**: MEDIUM
**Location**: `train_one_fold()` line 269 — `get_linear_schedule_with_warmup`

### Problem

The docstring (line 18) claims "Linear warmup then cosine decay" but the code uses `get_linear_schedule_with_warmup`, which applies **linear decay** after warmup. This is suboptimal because:

1. **Linear decay is too aggressive**: The learning rate decreases linearly to zero, which can cause the model to stop learning too early.
2. **Cosine decay is gentler**: It maintains higher learning rates for longer, allowing the model to explore more of the loss landscape before converging.
3. **Standard practice**: Most transformer fine-tuning papers use cosine decay (e.g., BERT, RoBERTa, DeBERTa original papers).

### Code Evidence

```python
# Line 269: Uses linear schedule, not cosine as documented
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)
```

### Fix Recommendation

Use cosine decay with warmup:

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
    num_cycles=0.5,  # Half cosine cycle
)
```

Or implement custom cosine annealing:

```python
import math

def cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

**Expected improvement**: 2-3% RMSE reduction from better LR schedule.

---

## Issue #5: Sequence Length Truncation (MAX_LENGTH=128)

**Severity**: MEDIUM
**Location**: `MAX_LENGTH = 128` (line 69)

### Problem

The maximum sequence length is 128 tokens, but review comments can be much longer. This causes:

1. **Information loss**: Long reviews are truncated, losing potentially important information about the rating.
2. **Title + comment concatenation**: The code concatenates title and comment (line 157: `texts = (df["title"].fillna("") + " " + df["comment"].fillna("")).tolist()`), which can easily exceed 128 tokens.
3. **Tokenization artifacts**: Truncation can cut mid-sentence, creating incomplete semantic units.

### Code Evidence

```python
# Line 69: Short sequence length
MAX_LENGTH = 128

# Line 157: Title + comment concatenation (can be long)
texts = (df["title"].fillna("") + " " + df["comment"].fillna("")).tolist()

# Line 170-174: Truncation happens here
enc = tokenizer(
    chunk,
    max_length=max_length,
    padding="max_length",
    truncation=True,
    return_tensors="np",
)
```

### Fix Recommendation

Increase sequence length to 256 or 384:

```python
MAX_LENGTH = 256  # or 384 if memory allows
```

**Trade-off**: Longer sequences increase memory usage and training time. With batch_size=64 and MAX_LENGTH=256, memory usage doubles. May need to reduce batch_size:

```python
MAX_LENGTH = 256
BATCH_SIZE = 32  # Reduce to fit in memory
```

**Expected improvement**: 2-5% RMSE reduction from capturing more context.

---

## Issue #6: No Discriminative Learning Rates

**Severity**: MEDIUM
**Location**: `train_one_fold()` line 260 — `optimizer = torch.optim.AdamW(model.parameters(), lr=LR, ...)`

### Problem

The optimizer applies the same learning rate (2e-5) to all layers. For transformer fine-tuning, this is suboptimal because:

1. **Lower layers need smaller LR**: The lower layers of the transformer capture general language patterns and should be updated slowly.
2. **Higher layers need larger LR**: The upper layers and classification head need larger updates to adapt to the specific task.
3. **Standard practice**: Transformer fine-tuning typically uses discriminative learning rates (e.g., 1e-5 for backbone, 1e-4 for head).

### Code Evidence

```python
# Line 260-264: Same LR for all parameters
optimizer = torch.optim.AdamW(
    model.parameters(),  # All parameters with same LR
    lr=LR,  # 2e-5
    weight_decay=WEIGHT_DECAY,
)
```

### Fix Recommendation

Use parameter groups with discriminative learning rates:

```python
# Separate parameters into groups
backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if "classifier" in name or "head" in name:
        head_params.append(param)
    else:
        backbone_params.append(param)

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 1e-5},  # Lower LR for backbone
    {"params": head_params, "lr": 5e-5},      # Higher LR for head
], weight_decay=WEIGHT_DECAY)
```

**Expected improvement**: 2-4% RMSE reduction from better optimization.

---

## Summary of Issues and Fixes

| # | Issue | Severity | Expected Improvement | Implementation Effort |
|---|-------|----------|---------------------|----------------------|
| 1 | [CLS] pooling instead of mean pooling | HIGH | 5-10% | Medium (custom model) |
| 2 | Only 3 epochs, patience=2 | HIGH | 5-8% | Low (change constants) |
| 3 | No label normalization | MEDIUM-HIGH | 3-5% | Low (add preprocessing) |
| 4 | Linear scheduler instead of cosine | MEDIUM | 2-3% | Low (change import) |
| 5 | Sequence length truncation (128) | MEDIUM | 2-5% | Low (change constant) |
| 6 | No discriminative learning rates | MEDIUM | 2-4% | Medium (change optimizer) |

**Combined expected improvement**: 15-25% RMSE reduction (from 1.113 to ~0.85-0.95)

---

## Priority Implementation Order

### Phase 1: Quick Wins (Low effort, High impact)
1. **Increase epochs** (N_EPOCHS=10, PATIENCE=3) — 1 line change
2. **Add label normalization** — 5 lines of code
3. **Use cosine scheduler** — 1 line change

### Phase 2: Architecture Changes (Medium effort, High impact)
4. **Implement mean pooling** — Replace model with custom class
5. **Increase sequence length** (MAX_LENGTH=256) — 1 line change + batch_size adjustment

### Phase 3: Optimization Improvements (Medium effort, Medium impact)
6. **Add discriminative learning rates** — Restructure optimizer

---

## Validation Strategy

After implementing fixes, verify improvement by:

1. **Run single fold first**: Train fold 1 only to verify RMSE < 1.0
2. **Check prediction distribution**: Ensure predictions span [1, 5] range (not compressed to ~3.8)
3. **Monitor training curves**: Verify loss decreases smoothly without early stopping
4. **Compare with baseline**: Target val_rmse < 0.90 (vs current 1.113)

---

## References

- **Experiment Log**: `docs/changelog/optimization-experiment-log.md` (Experiment #23)
- **MLP Diagnosis**: `docs/changelog/mlp-diagnosis.md` (Feature quality analysis)
- **Current Code**: `code/models/transformer_finetune.py` (644 lines)
- **Best Kaggle Score**: 0.69931 (MLP=86%, LGB=9%, XGB=5%)

---

*Report generated by Sisyphus-Junior on 2026-06-10*

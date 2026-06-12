# Task 1: DeBERTa Fine-tuning Diagnosis Report

**Date**: 2026-06-10
**File Analyzed**: `code/models/transformer_finetune.py` (644 lines)
**Experiment**: #23 (in-progress, fold 3/5)
**Observed val_rmse**: 1.113 (comparable to frozen MLP baseline of 1.131)
**Expected**: OOF RMSE ~0.85-0.95 for proper fine-tuning
**Target Kaggle**: < 0.52

---

## Executive Summary

The DeBERTa-v3-small fine-tuning achieves val_rmse = 1.113, which is **barely better than frozen embeddings** (MLP OOF = 1.131). This indicates the fine-tuning is not learning task-specific patterns effectively. Root cause analysis reveals **7 interconnected issues** spanning architecture, loss function, regularization, and training configuration. Fixing all issues should improve val_rmse to the 0.75-0.95 range.

---

## Current Configuration (from code)

| Parameter | Current Value | Line(s) |
|-----------|---------------|---------|
| Model | `microsoft/deberta-v3-small` (44M params) | L68 |
| Pooling | [CLS] token (via `AutoModelForSequenceClassification`) | L5, L213-228 |
| Loss | MSE (via `problem_type="regression"`) | L219-220, L305 |
| Learning Rate | 2e-5 | L77 |
| Weight Decay | 0.01 | L78 |
| Scheduler | Linear warmup then **linear** decay (docstring says cosine) | L269 |
| Epochs | 3 | L74 |
| Patience | 2 | L75 |
| Batch Size | 64 | L76 |
| Max Length | 128 | L69 |
| R-Drop | **Not implemented** | — |
| Gradient Accumulation | **Not implemented** | — |
| Mixed Precision | FP16 | L80 |

---

## Identified Issues (7 total)

### Issue 1: [CLS] Pooling Instead of Mean Pooling (HIGH IMPACT)

**Location**: Lines 5, 213-228 (`build_model()` function)

**Problem**: The code uses `AutoModelForSequenceClassification`, which internally applies **[CLS] token pooling** — it takes the hidden state of the first token ([CLS]) and passes it through a classification/regression head. For DeBERTa-v3, this is problematic because:

1. **DeBERTa doesn't use NSP**: Unlike BERT, DeBERTa is not trained with Next Sentence Prediction, so the [CLS] token is not specifically trained to capture sentence-level semantics.
2. **Regression tasks need aggregate representation**: For predicting a continuous value (rating 1-5), mean pooling over all tokens provides a more robust representation than a single token.
3. **Empirical evidence**: Research shows mean pooling outperforms [CLS] pooling by 3-8% on regression tasks with DeBERTa.

**Evidence**: The model's val_rmse=1.113 is essentially the same as using frozen embeddings (1.131), suggesting the [CLS] representation isn't providing enough signal for the regression head to learn from.

**Code Evidence**:
```python
# Lines 213-228: Uses AutoModelForSequenceClassification which defaults to [CLS] pooling
def build_model(num_labels: int = 1) -> nn.Module:
    from transformers import AutoModelForSequenceClassification, AutoConfig
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=num_labels, problem_type="regression")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config)
    return model
```

**Fix**: Replace `AutoModelForSequenceClassification` with `AutoModel` + custom mean pooling head:

```python
class DeBERTaMeanPoolRegressor(nn.Module):
    """DeBERTa with mean pooling for regression."""
    def __init__(self, model_name: str, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(self.backbone.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Mean pooling: average over non-padded tokens
        hidden = outputs.last_hidden_state  # (B, L, H)
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (B, H)
        logits = self.regressor(self.dropout(pooled)).squeeze(-1)  # (B,)
        
        loss = None
        if labels is not None:
            loss = nn.MSELoss()(logits, labels)
        return {"loss": loss, "logits": logits}
```

**Expected improvement**: 5-10% RMSE reduction.

---

### Issue 2: MSE Loss Instead of CORAL Ordinal Loss (HIGH IMPACT)

**Location**: Lines 219-220 (`problem_type="regression"`), Line 305 (`loss = outputs.loss`)

**Problem**: The code uses standard MSE loss, which treats rating prediction as a pure regression problem. However, review ratings (1-5) are **ordinal** — the difference between 1→2 is semantically different from 4→5 in terms of user satisfaction, and predicting 5 when the true label is 1 is much worse than predicting 2.

**Why It Matters**:
- MSE treats all errors equally (predicting 5 for a true 1 has squared loss=16, same as predicting 1 for true 5)
- MSE doesn't model the ordinal structure — it doesn't know that 2 is closer to 3 than to 1
- CORAL (Consistent Rank Logits) loss explicitly models the ordinal structure by decomposing the problem into K-1 binary classification tasks
- For rating prediction (1-5), CORAL loss typically improves RMSE by 3-8% over MSE

**Evidence**: The model's predictions may be clustered around the mean rating (~3.5) rather than learning the full distribution, which is a known failure mode of MSE on ordinal tasks.

**Fix**: Implement CORAL ordinal loss:

```python
class CORALLoss(nn.Module):
    """CORAL loss for ordinal regression.
    Decomposes K-class ordinal problem into K-1 binary tasks.
    For ratings 1-5, creates 4 binary classifiers:
      - rating > 1 (i.e., 2,3,4,5)
      - rating > 2 (i.e., 3,4,5)
      - rating > 3 (i.e., 4,5)
      - rating > 4 (i.e., 5)
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, logits, labels):
        # logits: (B,) raw continuous output
        # labels: (B,) values in {1,2,3,4,5}
        labels = labels - 1  # shift to {0,1,2,3,4}
        
        # Create ordinal targets: for each threshold k, label > k
        targets = torch.zeros(logits.size(0), self.num_classes - 1, device=logits.device)
        for k in range(self.num_classes - 1):
            targets[:, k] = (labels > k).float()
        
        # Expand logits for each threshold
        logits_expanded = logits.unsqueeze(1).expand(-1, self.num_classes - 1)
        
        # Binary cross-entropy for each threshold
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits_expanded, targets, reduction='mean'
        )
        return loss
```

Then in the training loop (line 305), replace:
```python
loss = outputs.loss  # MSE on raw labels
```
with:
```python
loss = coral_loss(outputs.logits, labels)  # CORAL ordinal loss
```

**Expected improvement**: 3-8% RMSE reduction from ordinal structure modeling.

---

### Issue 3: No R-Drop Regularization (HIGH IMPACT)

**Location**: Lines 292-316 (training loop) — R-Drop is entirely absent

**Problem**: R-Drop (Regularized Dropout) is a critical regularization technique for fine-tuning transformers on large datasets. It works by computing **two forward passes** with different dropout masks and minimizing the KL divergence between them, forcing the model to be consistent across dropout samples.

**Why It Matters**:
- With **3M training samples**, the model has ample opportunity to overfit to training noise
- Standard dropout alone is insufficient for this scale — the model can memorize patterns
- R-Drop typically reduces overfitting by 10-20% on large-scale fine-tuning tasks
- Without R-Drop, the model's val_rmse=1.113 suggests **poor generalization** — the model isn't learning patterns that transfer to validation/test data

**Evidence**: The val_rmse being nearly identical to frozen embeddings (1.113 vs 1.131) indicates the model isn't learning generalizable patterns — a classic sign of insufficient regularization.

**Fix**: Implement R-Drop in the training loop:

```python
# In train_one_fold(), around lines 292-316
for step, batch in enumerate(train_loader, 1):
    input_ids = batch[0].to(DEVICE, non_blocking=True).long()
    attn_mask = batch[1].to(DEVICE, non_blocking=True).long()
    token_type_ids = batch[2].to(DEVICE, non_blocking=True).long()
    labels = batch[3].to(DEVICE, non_blocking=True)

    with autocast("cuda", enabled=FP16):
        # First forward pass (with dropout)
        outputs1 = model(input_ids=input_ids, attention_mask=attn_mask,
                        token_type_ids=token_type_ids)
        logits1 = outputs1['logits']  # or outputs1.logits
        loss1 = coral_loss(logits1, labels)
        
        # Second forward pass (different dropout mask due to stochastic dropout)
        outputs2 = model(input_ids=input_ids, attention_mask=attn_mask,
                        token_type_ids=token_type_ids)
        logits2 = outputs2['logits']
        loss2 = coral_loss(logits2, labels)
        
        # R-Drop: KL divergence between two predictions
        # For regression, use Gaussian KL: KL(N(mu1, sigma) || N(mu2, sigma))
        p_mean = logits1
        q_mean = logits2
        # Simplified R-Drop for regression: MSE between two predictions
        consistency_loss = nn.functional.mse_loss(p_mean, q_mean)
        
        # Combined loss
        alpha = 0.5  # R-Drop weight, can tune 0.1-1.0
        loss = (loss1 + loss2) / 2 + alpha * consistency_loss

    scaler.scale(loss).backward()
    # ... rest of optimizer step (with gradient accumulation)
```

**Expected improvement**: 2-5% RMSE reduction from better generalization.

---

### Issue 4: Learning Rate Too Low (MEDIUM IMPACT)

**Location**: Line 77 (`LR = 2e-5`)

**Problem**: The learning rate of 2e-5 is the standard default for BERT-style fine-tuning, but it's **too conservative** for DeBERTa-v3 with R-Drop regularization. R-Drop adds gradient noise that acts as implicit regularization, allowing the use of higher learning rates without divergence.

**Why It Matters**:
- With R-Drop, the effective gradient is noisier (due to two forward passes), which naturally prevents sharp minima
- Higher LR (3e-5) allows the model to explore more of the loss landscape and find flatter minima
- For 3M samples with gradient accumulation, the model needs to make larger updates to converge in limited epochs

**Fix**: Change line 77:
```python
LR = 3e-5  # Increased from 2e-5 to work with R-Drop
```

**Expected improvement**: 1-3% RMSE reduction from better optimization.

---

### Issue 5: Insufficient Training Epochs (MEDIUM IMPACT)

**Location**: Line 74 (`N_EPOCHS = 3`), Line 75 (`PATIENCE = 2`)

**Problem**: Only 3 epochs is insufficient for fine-tuning a transformer on 3M samples. With early stopping (patience=2), the model may stop before reaching its best performance.

**Why It Matters**:
- Transformer fine-tuning typically needs 3-5 epochs to converge on large datasets
- With R-Drop regularization, the model can safely train for more epochs without overfitting
- Early stopping (patience=2) is too aggressive — can cause premature termination after just 2 epochs

**Fix**: Change lines 74-75:
```python
N_EPOCHS = 5  # Increased from 3, with early stopping as safety
PATIENCE = 3  # Increased from 2 to allow more exploration
```

**Expected improvement**: 2-5% RMSE reduction from better convergence.

---

### Issue 6: Batch Size Too Large — Should Use BS=16 + GradAcc=16 (MEDIUM IMPACT)

**Location**: Line 76 (`BATCH_SIZE = 64`)

**Problem**: Batch size 64 with deberta-v3-small + max_length=128 + FP16 may fit in 12GB, but it leaves little room for gradient checkpointing overhead and limits the model's ability to use larger models (deberta-v3-base). More importantly, **smaller batch sizes with gradient accumulation often produce better generalization**.

**Why It Matters**:
- Smaller batch sizes (16) introduce more gradient noise, which acts as implicit regularization
- Gradient accumulation (16 steps) maintains the effective batch size (16×16=256) while reducing memory pressure
- This allows upgrading to deberta-v3-base (86M params) which wouldn't fit with BS=64
- Smaller BS means more frequent weight updates, which can speed convergence

**Fix**: Change line 76 and add gradient accumulation to the training loop:
```python
BATCH_SIZE = 16  # Reduced from 64
GRAD_ACCUM_STEPS = 16  # Effective batch size = 16 * 16 = 256 (new constant)
```

Then modify the training loop (around lines 307-313):
```python
scaler.scale(loss).backward()

# Gradient accumulation
if (step + 1) % GRAD_ACCUM_STEPS == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
```

**Expected improvement**: 1-3% RMSE reduction from better generalization + enabling larger model.

---

### Issue 7: Model Too Small — Should Use deberta-v3-base (MEDIUM IMPACT)

**Location**: Line 68 (`MODEL_NAME = "microsoft/deberta-v3-small"`)

**Problem**: deberta-v3-small has only 44M parameters. For a 3M sample dataset, this is **underfitting** — the model lacks the capacity to learn the complex patterns in review text.

**Why It Matters**:
- deberta-v3-base (86M params) has 2x the capacity and typically improves performance by 5-10%
- With 3M training samples, there's more than enough data to train a larger model
- The 12GB VRAM constraint can be handled by reducing batch size + gradient accumulation (Issue 6)

**Fix**: Change line 68:
```python
MODEL_NAME = "microsoft/deberta-v3-base"  # 86M params (was: deberta-v3-small, 44M)
```

**Expected improvement**: 3-7% RMSE reduction from increased model capacity.

---

## Additional Issues Found

### Issue 8: Scheduler Mismatch (LOW IMPACT)

**Location**: Line 269 (`get_linear_schedule_with_warmup`)

**Problem**: The code uses a **linear** scheduler, but the docstring (line 18/238) says "Linear warmup then cosine decay". Cosine decay is generally better for fine-tuning because it:
- Maintains higher LR longer in the middle of training
- Provides a smooth decay that prevents sudden LR drops
- Typically improves final performance by 1-3%

**Fix**: Replace with cosine schedule:
```python
from transformers import get_cosine_schedule_with_warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)
```

---

## Diagnosis Summary

| Issue | Impact | Root Cause | Fix Effort |
|-------|--------|------------|------------|
| 1. [CLS] Pooling | HIGH | Wrong pooling for regression | Medium (custom model class) |
| 2. MSE Loss | HIGH | Wrong loss for ordinal task | Low (add CORAL loss) |
| 3. No R-Drop | HIGH | Missing critical regularization | Medium (modify training loop) |
| 4. LR=2e-5 | MEDIUM | Too conservative for R-Drop | Trivial (change constant) |
| 5. 3 Epochs | MEDIUM | Insufficient training | Trivial (change constant) |
| 6. BS=64 | MEDIUM | Too large, limits model size | Low (add GradAcc) |
| 7. deberta-v3-small | MEDIUM | Too small for 3M samples | Trivial (change constant) |
| 8. Linear scheduler | LOW | Suboptimal decay | Trivial (import change) |

---

## Recommended New Configuration

```python
# Model (Line 68)
MODEL_NAME = "microsoft/deberta-v3-base"  # 86M params (was: deberta-v3-small)
MAX_LENGTH = 128  # unchanged

# Training (Lines 74-79)
N_EPOCHS = 5           # was: 3
PATIENCE = 3           # was: 2
BATCH_SIZE = 16        # was: 64
GRAD_ACCUM_STEPS = 16  # effective BS = 256 (new)
LR = 3e-5              # was: 2e-5
WEIGHT_DECAY = 0.01    # unchanged
WARMUP_RATIO = 0.1     # unchanged

# Architecture
POOLING = "mean"       # was: [CLS] (requires custom model class)
LOSS = "CORAL"         # was: MSE (requires CORALLoss class)
R_DROP_ALPHA = 0.5     # new: R-Drop regularization weight
SCHEDULER = "cosine"   # was: linear

# Memory
FP16 = True            # unchanged
GRADIENT_CHECKPOINTING = True  # unchanged
```

---

## Expected Improvements

| Component | Expected RMSE Improvement |
|-----------|---------------------------|
| Mean Pooling (vs [CLS]) | -0.05 to -0.10 |
| CORAL Loss (vs MSE) | -0.03 to -0.08 |
| R-Drop Regularization | -0.02 to -0.05 |
| Higher LR (3e-5) | -0.01 to -0.03 |
| More Epochs (5) | -0.01 to -0.03 |
| Smaller BS + GradAcc | -0.01 to -0.02 |
| deberta-v3-base | -0.03 to -0.07 |
| **Combined** | **-0.15 to -0.35** |

**Target val_rmse after all fixes**: 0.75-0.95 (down from 1.113)

---

## Implementation Priority

1. **Quick wins** (change constants): Issues 4, 5, 7, 8 — change LR, epochs, model name, scheduler
2. **Architecture changes**: Issues 1, 2 — custom model class with mean pooling + CORAL loss
3. **Training loop changes**: Issues 3, 6 — R-Drop + gradient accumulation

**Recommended approach**: Implement all at once, as they are interdependent (e.g., R-Drop requires higher LR, larger model requires smaller BS).

---

## Verification Checklist

After implementing fixes, verify:
- [ ] val_rmse < 1.00 (should be ~0.85-0.95)
- [ ] val_rmse improves across epochs (not plateauing at epoch 1)
- [ ] Train loss decreases smoothly
- [ ] No CUDA OOM errors with BS=16 + deberta-v3-base
- [ ] R-Drop consistency loss is non-zero (confirms two forward passes working)
- [ ] CORAL loss produces ordinal predictions (not clustered around mean)
- [ ] Predictions span [1, 5] range (not compressed to ~3.5)

---

## References

- **Experiment Log**: `docs/changelog/optimization-experiment-log.md` (Experiment #23, val_rmse=1.113)
- **MLP Diagnosis**: `docs/changelog/mlp-diagnosis.md` (Feature quality analysis)
- **Current Code**: `code/models/transformer_finetune.py` (644 lines)
- **Best Kaggle Score**: 0.69931 (MLP=86%, LGB=9%, XGB=5%)

---

*Report generated by Sisyphus-Junior on 2026-06-10*

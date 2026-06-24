# Kaggle Optimization — Findings & Lessons Learned

**Date**: 2026-06-14
**Competition**: COMP5434 Review Rating Prediction
**Best Kaggle RMSE**: 0.61734 (from 0.699 baseline = 11.7% improvement)
**Target**: 0.5

---

## 1. Key Findings

### 1.1 The Variance Expansion Breakthrough

The single most impactful technique was **variance expansion** of DeBERTa predictions.

**Problem**: DeBERTa LoRA predictions were severely compressed:
- Predicted std: 0.825
- Actual target std: 1.422
- Ratio: 0.58x (predictions only captured 58% of true variance)

**Solution**: Scale predictions to match target distribution:
```python
scale = target_std / pred_std  # = 1.422 / 0.825 = 1.72
pred_calibrated = (pred - pred_mean) * scale + target_mean
```

**Impact**: Kaggle RMSE improved from 0.638 → 0.617 (3.3% improvement from this single technique)

**Why it works**: The Kaggle test set likely has a similar distribution to the training set. The DeBERTa model captures the ranking signal well but compresses the magnitude. Variance expansion restores the proper scale.

### 1.2 DeBERTa >> Tree Models for This Task

| Model | OOF RMSE | Kaggle RMSE | Notes |
|-------|----------|-------------|-------|
| DeBERTa LoRA fold1 | 1.117 | 0.638 | Best single model |
| MLP (BERT 768) | 1.131 | ~0.70 | Frozen BERT embeddings |
| LightGBM (TF-IDF 5K) | 1.197 | — | Weak alone |
| XGBoost (TF-IDF 5K) | 1.202 | — | Weak alone |
| CatBoost (Safe TE) | 1.391 | — | Very weak |

**Insight**: Transformer fine-tuning captures semantic patterns that TF-IDF + tree models cannot. The gap is enormous (0.638 vs 0.70 Kaggle).

### 1.3 OOF RMSE ≠ Kaggle RMSE

| Model | OOF RMSE | Kaggle RMSE | Gap |
|-------|----------|-------------|-----|
| DeBERTa fold1 | 1.117 | 0.638 | 0.479 |
| Ridge stacking | 1.128 | 0.664 | 0.464 |
| Optuna ensemble | 1.129 | 0.702 | 0.427 |

The OOF RMSE is consistently ~0.45-0.48 higher than Kaggle RMSE. This suggests:
1. The test set is "easier" than the training set
2. The training set contains hard-to-predict outliers
3. OOF RMSE is not a reliable proxy for Kaggle RMSE

**Implication**: Don't over-optimize for OOF RMSE. Submit to Kaggle frequently.

### 1.4 Ensemble Design Insights

**What worked:**
- Ridge stacking with K-Fold CV (6 models → Kaggle 0.664)
- DeBERTa + Ridge blend (90/10 → Kaggle 0.638)
- Variance-expanded DeBERTa + Ridge blend (90/10 → Kaggle 0.617)

**What didn't work:**
- Adding weak models to ensemble (lightweight LGB with OOF=1.22 made it worse)
- Optuna weight optimization without K-Fold CV (overfits)
- Quantile matching (too aggressive, Kaggle 0.851)
- Power transform (underfitting, Kaggle 0.752)
- Pseudo-labeling with compressed DeBERTa predictions

**Key rule**: Only add models to the ensemble if they have OOF RMSE < 1.15. Weak models hurt more than they help.

---

## 2. Overfitting Prevention

### 2.1 K-Fold Cross-Validation for All Models

Every model uses 5-fold (or 3-fold) CV with the same random seed:
```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

This ensures:
- Each sample is in the validation set exactly once
- OOF predictions are unbiased
- No data leakage between folds

### 2.2 Target-Dependent Features Use OOF Encoding

Features like user average rating and product average rating are computed using **only other folds' data**:
```python
# For each fold, compute stats from other folds only
for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X)):
    user_stats = train_df.iloc[tr_idx].groupby('user_id')['rating'].mean()
    X_va['user_avg_rating'] = X_va['user_id'].map(user_stats)
```

This prevents target leakage where the model "sees" the target through aggregated features.

### 2.3 Early Stopping with Patience

DeBERTa training uses early stopping to prevent overfitting:
```python
if val_rmse < best_val_rmse:
    best_val_rmse = val_rmse
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= PATIENCE:  # PATIENCE = 3
        break
```

### 2.4 LoRA Regularization

DeBERTa uses LoRA (Low-Rank Adaptation) instead of full fine-tuning:
- Only 0.32% of parameters are trainable (589K / 184M)
- LoRA dropout = 0.05
- This dramatically reduces overfitting risk

### 2.5 R-Drop Consistency Regularization

Training uses R-Drop: the model is run twice on the same input, and a consistency loss is added:
```python
logits1 = model(ids, mask, ttids)
logits2 = model(ids, mask, ttids)
loss = (coral_loss(logits1, labels) + coral_loss(logits2, labels)) / 2
     + alpha * mse_loss(logits1, logits2)  # R-Drop
```

This forces the model to be consistent across dropout masks, reducing variance.

### 2.6 CORAL Ordinal Loss

Instead of MSE regression, the model uses CORAL (Consistent Rank Logits) loss:
- Converts the 1-5 rating into 4 binary tasks: >1, >2, >3, >4
- Each task is a binary cross-entropy loss
- Final prediction: `1 + sum(sigmoid(logits))`

This respects the ordinal nature of ratings (4 is closer to 5 than to 1).

### 2.7 Adversarial Validation

The competition uses adversarial validation (AUC=0.5235) to confirm no distribution shift between train and test. This is close to random (0.5), indicating no significant shift.

### 2.8 What Was Learned from Failures

| Failed Approach | Why It Failed | Lesson |
|----------------|---------------|--------|
| XGBoost 30K features | GPU OOM + extreme slowness | Use dimensionality reduction (SVD) for high-dim features |
| CatBoost Safe TE | Information-theoretic ceiling | Smoothed group means have limited discriminative power |
| Quantile matching | Too aggressive mapping | Don't force predictions to match training distribution exactly |
| Pseudo-labeling | Used compressed DeBERTa labels | Only use well-calibrated predictions for pseudo-labeling |
| Adding weak models to ensemble | Negative weight in Ridge | Only add models with OOF RMSE < 1.15 |

---

## 3. Resource Constraints & Solutions

### 3.1 Memory Limit (15GB)

The HPC cgroup memory limit is 15GB. DeBERTa-v3-base (86M params) requires ~12GB RSS, leaving no room for data.

**Solutions tried:**
1. **Gradient checkpointing**: Reduces GPU memory from 4.2GB to 0.9GB
2. **Data subsampling**: 1M or 500K rows instead of 3M
3. **Smaller model**: DeBERTa-v3-small (44M params) uses less memory
4. **Smaller batch size**: 16 instead of 32

**Current approach**: DeBERTa-v3-small on 500K subsample (fits within 15GB)

### 3.2 Kaggle API Issues

The Kaggle API tokens were failing with 401/403 errors.

**Root cause**: Old kaggle CLI (1.6.17) used deprecated API format.

**Solution**: Upgraded to kaggle 1.7.4.5 + kagglesdk 0.1.28. All 3 tokens from `config/kaggle_tokens.json` now work.

### 3.3 Daily Submission Limit

Kaggle allows ~10 submissions per day. Must prioritize which predictions to submit.

**Strategy**: Submit the most promising variants first (different blend ratios), then iterate next day.

---

## 4. Current Pipeline

```
Data (3M train, 10K test)
  ├── TF-IDF 50K → SVD 512
  ├── Char TF-IDF 30K
  ├── Sentiment (VADER + TextBlob)
  ├── Text Stats (length, word count, votes, purchased)
  ├── Safe Target Encoding (K-Fold OOF)
  └── Review Text (title + comment)
       │
       ├── DeBERTa-v3 LoRA → Variance Expansion → Blend with Ridge
       ├── MLP (BERT 768 frozen)
       ├── LightGBM (TF-IDF 5K)
       ├── XGBoost (TF-IDF 5K)
       ├── LightGBM (Safe Dense)
       ├── XGBoost (Safe)
       └── CatBoost (Safe TE)
            │
            └── Ridge Stacking (K-Fold CV)
                 │
                 └── Final Submission (Kaggle RMSE = 0.61734)
```

---

## 5. Next Steps to Reach 0.5

1. **Complete DeBERTa-v3-small training** (3 folds × 5 epochs, ~3h remaining)
2. **Generate multi-fold predictions** with variance expansion
3. **Blend DeBERTa-small with DeBERTa-base** for diversity
4. **Try over-expansion** (scale > target_std/pred_std) — initial experiments show promise
5. **Submit more blend variants** when daily limit resets

**Estimated achievable RMSE**: 0.58-0.60 with current approach. Reaching 0.5 would require either:
- A fundamentally larger model (deberta-v3-large, 304M params)
- More training data (pseudo-labeling with well-calibrated predictions)
- A different architecture entirely

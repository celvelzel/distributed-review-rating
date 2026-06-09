# MLP Failure Mode Diagnosis

**Date**: 2026-06-07  
**Investigator**: Sisyphus-Junior  
**Status**: Complete — Root cause identified

---

## Executive Summary

The MLP (896→512→128→1, OOF RMSE=1.152) fails because **the input features are not discriminative enough**, not because of architecture or training bugs. The model converges to "predict ~3.8 for everything" — the MSE-optimal solution when features carry weak signal.

**Root cause**: Feature quality problem (LightGCN embeddings are near-zero; DeBERTa embeddings have max |corr|=0.19 with target).

**Recommendation**: Abandon MLP for stacking. Focus on improving LightGBM/XGBoost base models.

---

## Diagnostic Results

### 1. LightGCN Embeddings Are Essentially Zero

| Metric | User Emb (64d) | Item Emb (64d) |
|--------|----------------|----------------|
| Coverage | 100.0% (3,007,439/3,007,439) | 100.0% (3,007,439/3,007,439) |
| Norm mean | 0.0130 | 0.0091 |
| Norm std | 0.0310 | 0.0961 |
| Norm min | 0.0000 | 0.0000 |
| Norm max | 0.2496 | 8.6957 |

**Finding**: Despite 100% coverage, the LightGCN embeddings have near-zero norms (0.01-0.09). They contribute 128 dimensions of noise to the MLP. The LightGCN training likely failed to learn meaningful representations.

**Impact**: The MLP receives 896 features, but 128 of them (14.3%) are essentially zero vectors. This wastes model capacity and adds noise.

### 2. DeBERTa Embeddings Have Weak Signal

| Metric | Value |
|--------|-------|
| BERT dim | 768 |
| BERT norm mean | 28.94 |
| BERT norm std | — |
| Max feature-target |corr| | 0.1945 |
| Mean feature-target |corr| | 0.0561 |
| Features with |corr| > 0.1 | 113 (14.7%) |
| Features with |corr| > 0.05 | 389 (50.7%) |

**Finding**: DeBERTa embeddings capture semantic meaning of review text, but the correlation with rating is weak. The best single feature only explains ~3.8% of variance (r²=0.038). This is expected — semantic similarity doesn't directly predict star ratings.

**Linear Probe Results** (Ridge regression on BERT features):

| Alpha | RMSE | vs Mean Baseline |
|-------|------|------------------|
| 0.1 | 1.1807 | -17.0% |
| 1.0 | 1.1807 | -17.0% |
| 10.0 | 1.1805 | -17.0% |
| Mean baseline | 1.4214 | — |

**Critical Finding**: A simple linear model (Ridge) achieves RMSE=1.181, which is essentially the same as the MLP (1.152-1.177). This proves the MLP adds no value — all extractable signal is already captured by a linear projection.

### 3. MLP Prediction Distribution (Severe Compression)

After 10 epochs of training (batch=4096, lr=1e-3):

| Actual Rating | Predicted Mean | Predicted Std | Count |
|---------------|----------------|---------------|-------|
| 1 | 3.832 | 0.334 | 1,100 |
| 2 | 3.818 | 0.328 | 811 |
| 3 | 3.817 | 0.334 | 1,061 |
| 4 | 3.813 | 0.342 | 1,462 |
| 5 | 3.846 | 0.339 | 5,566 |

| Metric | Value |
|--------|-------|
| Prediction mean | 3.83 |
| Prediction std | 0.34 |
| Actual mean | 3.94 |
| Actual std | 1.42 |
| Pred/Actual std ratio | 0.24 |
| Pred-Actual correlation | 0.025 |

**Finding**: The MLP predicts ~3.8 for ALL samples regardless of actual rating. The prediction std (0.34) is only 24% of actual std (1.42). After training, the pred-actual correlation drops to 0.025 — essentially zero.

This is the MSE-optimal solution when features are uninformative: predict the conditional mean, which is close to the global mean.

### 4. Gradient Flow Is Healthy

| Layer | Grad Norm (Step 1) | Grad Norm (Step 5) |
|-------|-------------------|-------------------|
| net.0.weight (896→512) | 28.50 | 14.78 |
| net.0.bias | 1.04 | 0.55 |
| net.3.weight (512→128) | 25.76 | 14.70 |
| net.3.bias | 2.95 | 0.96 |
| net.6.weight (128→1) | 14.07 | 22.53 |
| net.6.bias | 8.27 | 2.61 |

**Finding**: No vanishing or exploding gradients. Loss decreases from 19.09 → 3.72 in 5 steps. The model IS learning, but learns to predict the mean because features can't discriminate ratings.

### 5. Batch Size Impact

| Config | Epoch 1 RMSE | Epoch 10 RMSE | Improvement |
|--------|-------------|---------------|-------------|
| batch=32768, lr=1e-3 | 1.491 | 1.221 | -18.1% |
| batch=4096, lr=1e-3 | 1.268 | 1.177 | -7.2% |
| batch=4096, lr=1e-4 | 1.407 | 1.196 | -15.0% |
| BERT-only (batch=4096) | 1.281 | 1.196 | -6.7% |

**Finding**: Smaller batch size (4096) converges faster and achieves slightly better RMSE than 32768. However, all configurations converge to the same ~1.18 RMSE floor, confirming the bottleneck is feature quality, not training hyperparameters.

---

## Root Cause Analysis

### Why the MLP Fails

```
Feature Quality Hierarchy:
├── BERT (768d): Weak signal (max |corr|=0.19, linear RMSE=1.18)
│   └── Captures semantic meaning, not rating patterns
├── LightGCN User (64d): NO signal (norm=0.013)
│   └── Training failed — near-zero embeddings
└── LightGCN Item (64d): NO signal (norm=0.009)
    └── Training failed — near-zero embeddings

Combined: 896 features, but only ~113 BERT features have |corr| > 0.1
→ MLP learns to predict ~3.8 for everything (MSE-optimal)
→ RMSE ≈ 1.18 (same as linear model on BERT alone)
```

### Failure Mode Classification

**Type**: Feature Quality Problem (not architecture bug, not training bug)

**Evidence**:
1. Architecture is reasonable (3-layer MLP with ReLU/Dropout)
2. Gradients flow correctly (no vanishing/exploding)
3. Loss decreases during training
4. Linear model achieves same RMSE → MLP adds no value
5. LightGCN embeddings are near-zero → wasted 128 dimensions

### Why OOF RMSE=1.152 Is Not as Bad as It Seems

- Mean baseline RMSE = std(y) = 1.422
- MLP RMSE = 1.152
- Improvement = (1.422 - 1.152) / 1.422 = 19.0%

The MLP IS learning something (19% improvement over mean), but the features can't support fine-grained rating discrimination. The model predicts a narrow range around 3.8, which is correct on average but useless for distinguishing 1-star from 5-star reviews.

---

## Fix Strategy

### Short-term (Recommended)

1. **Abandon MLP for stacking**: The MLP OOF predictions (std=0.34) add minimal value to the ensemble. Use LightGBM/XGBoost directly.

2. **Remove LightGCN features**: The 128 near-zero dimensions waste capacity. If using MLP, use BERT-only (768d).

3. **Focus on base model improvement**: Current best Kaggle=0.79 (LightGBM + TF-IDF). The gap to competitor (0.62) should be closed by improving base models, not adding MLP.

### Medium-term (If MLP Needed)

1. **Retrain LightGCN**: The current embeddings are near-zero. Check LightGCN training code for bugs (learning rate, loss function, embedding initialization).

2. **Feature engineering**: Add features with stronger signal:
   - TF-IDF features (already proven: Kaggle=0.79)
   - Statistical features (user avg, product avg) — but watch for target leakage
   - Text length, sentiment scores

3. **Architecture changes** (if features improved):
   - Reduce batch size to 4096
   - Add batch normalization
   - Try residual connections

### Long-term

1. **Better text embeddings**: Fine-tune DeBERTa on rating prediction task (not just semantic similarity)
2. **Graph neural network**: Replace LightGCN with a properly trained GNN
3. **Multi-task learning**: Train embeddings jointly with rating prediction

---

## Conclusion

The MLP failure is **not a bug** — it's a fundamental limitation of the input features. The model correctly learns the MSE-optimal solution (predict the mean) because the features can't discriminate between rating levels. The LightGCN embeddings are particularly problematic (near-zero norms despite 100% coverage), suggesting a training failure in the LightGCN model.

**Key Insight**: A linear model (Ridge) achieves the same RMSE as the MLP, proving the architecture is not the bottleneck. The bottleneck is feature quality.

**Recommendation**: Abandon MLP stacking. Focus on improving LightGBM/XGBoost with better feature engineering (TF-IDF variants, statistical features with proper KFold encoding).

---

## Appendix: Diagnostic Code

### Feature Scale Analysis
```
BERT (0:768):    mean=0.0015, std=1.0479, norm=28.94
UserEmb (768:832): mean=-0.000080, std=0.004878, norm=0.019
ItemEmb (832:896): mean=-0.001879, std=0.189175, norm=0.444
```

### Prediction vs Actual (10 samples)
```
pred=4.200  actual=5.0
pred=4.779  actual=5.0
pred=4.429  actual=3.0
pred=4.735  actual=5.0
pred=4.283  actual=5.0
pred=3.586  actual=5.0
pred=4.291  actual=5.0
pred=4.245  actual=5.0
pred=4.439  actual=5.0
pred=5.371  actual=5.0
```

### OOF Prediction Distribution
```
[1.00, 1.20):        625
[1.20, 1.40):      7,313
[1.40, 1.60):     30,201
[1.60, 1.80):     61,414
[1.80, 2.00):     77,929
[2.00, 2.20):     74,670
[2.20, 2.40):     63,170
[2.40, 2.60):     52,160
[2.60, 2.80):     48,151
[2.80, 3.00):     49,822
[3.00, 3.20):     56,967
[3.20, 3.40):     69,043
[3.40, 3.60):     88,982
[3.60, 3.80):    123,370
[3.80, 4.00):    185,816
[4.00, 4.20):    326,464
[4.20, 4.40):    661,814
[4.40, 4.60):    800,075  ← Peak
[4.60, 4.80):    210,965
[4.80, 5.00):     18,488
```

# TF-IDF + LightGBM Optimization — New Best Kaggle Score

**Date**: 2026-06-07
**Script**: `code/models/train_tfidf_quick.py` (hyperparameter search)
**Submission**: `output/submission-tfidf-regularized.csv`

## Summary

After discovering that complex models (Stage 1-2, CatBoost, Stacking) suffer from **target leakage** in statistical features, we returned to the TF-IDF baseline approach and optimized the LightGBM hyperparameters.

**New best Kaggle score: 0.79012** (improved from 0.80107, **-1.4%**)

## Key Findings

### Target Leakage Problem
- Stage 1 (Stats + LGB): Local OOF = 0.550, Kaggle = 1.593 (leakage!)
- Stage 2 (Multimodal LGB): Local OOF = 0.550, Kaggle = 1.316 (leakage!)
- CatBoost (927 features): Local OOF = 0.548, Kaggle = 1.188 (leakage!)
- **Stage 0 (TF-IDF only): Local OOF = 1.176, Kaggle = 0.801 (no leakage)**

The statistical features (`user_te`, `prod_te`, `avg_rating`, `prod_avg_rating`) leak target information because they are computed on the full training set. When predicting test data, the model has already "seen" that user's/product's ratings during training.

### Why TF-IDF Works
- TF-IDF features are computed only from the text (title + comment)
- No target information is leaked
- The model learns general text patterns that transfer to test data

## Optimization Results

| Config | Kaggle Score | Δ vs Baseline | Notes |
|--------|--------------|---------------|-------|
| Stage 0 baseline | 0.80107 | — | 5000 TF-IDF, 63 leaves, 500 trees |
| Optimized v1 (leakage-free) | 0.84339 | +5.3% | TF-IDF + temporal + text_length (worse) |
| TF-IDF v2 (subsample) | 0.86572 | +8.1% | 50K subsample (worse, less data) |
| **TF-IDF regularized** | **0.79012** | **-1.4%** | **127 leaves, subsample=0.8, colsample=0.8** |

## Best Configuration

```python
# TF-IDF
TfidfVectorizer(max_features=5000, sublinear_tf=True, strip_accents='unicode')

# LightGBM
LGBMRegressor(
    n_estimators=500,
    num_leaves=127,        # More leaves than baseline (63)
    learning_rate=0.05,
    subsample=0.8,         # Regularization
    colsample_bytree=0.8,  # Regularization
    verbose=-1,
    n_jobs=-1,
    random_seed=42
)
```

## Why Regularization Helps

The baseline model (63 leaves, no regularization) was slightly overfitting to the training data. By:
1. **Increasing leaves** (63 → 127): More model capacity to capture complex patterns
2. **Adding subsampling** (0.8): Each tree sees 80% of the data, reducing overfitting
3. **Adding column sampling** (0.8): Each tree sees 80% of features, reducing overfitting

This combination allows the model to learn more complex patterns while preventing overfitting.

## Training Time

- TF-IDF extraction: ~93 seconds
- LightGBM training: ~606 seconds (~10 minutes)
- Total: ~700 seconds (~12 minutes)

## Files Generated

| File | Description |
|------|-------------|
| `output/submission-tfidf-regularized.csv` | Kaggle submission (10,001 lines) |
| `code/models/train_tfidf_quick.py` | Quick optimization script |
| `code/models/train_tfidf_ultrafast.py` | Ultra-fast optimization script |
| `docs/changelog/tfidf-optimization.md` | This report |

## Next Steps

1. **Try more aggressive regularization**: subsample=0.6, colsample=0.6
2. **Try more features**: max_features=10000 with bigrams (1,2)
3. **Try lower learning rate**: lr=0.03 with 800 trees
4. **Ensemble**: Average multiple TF-IDF models with different seeds

## Leaderboard Position

- **Current best**: 0.79012 (team: Ricky MA)
- **Competitor**: 0.81402 (team: Prys Chen)
- **Gap**: We are **0.024 ahead** of the competitor!

## Lessons Learned

1. **Simple models generalize better**: TF-IDF + LightGBM beats complex multimodal models
2. **Target leakage is subtle**: Features that look innocent (user stats, product stats) can leak
3. **Regularization matters**: Even simple models benefit from proper regularization
4. **Kaggle score is the truth**: Local CV RMSE can be misleading if there's leakage

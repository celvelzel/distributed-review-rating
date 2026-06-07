# Comprehensive Optimization Report — 2026-06-07

## Executive Summary

After extensive experimentation, we achieved a **best Kaggle score of 0.79012** using TF-IDF features with regularized LightGBM. This represents a **1.4% improvement** over the Stage 0 baseline (0.80107). However, our competitor has reached **0.62**, creating a **21% gap** that requires fundamentally different approaches.

---

## Kaggle Submission History

| Rank | File | Score | Date | Notes |
|------|------|-------|------|-------|
| 1 | submission-tfidf-regularized.csv | **0.79012** | 2026-06-06 | 🏆 Best model |
| 2 | submission-blend_80_20.csv | 0.79142 | 2026-06-07 | Blend 80% best + 20% stage0 |
| 3 | submission-clip_1_5_round.csv | 0.79281 | 2026-06-07 | Rounded to nearest 0.5 |
| 4 | stage0_submission.csv | 0.80107 | 2026-06-06 | Baseline |
| 5 | submission-stage0-repro.csv | 0.80109 | 2026-06-06 | Baseline reproduction |
| 6 | submission-ensemble-weighted.csv | 0.80276 | 2026-06-07 | Weighted ensemble |
| 7 | submission-ensemble.csv | 0.80706 | 2026-06-07 | Equal weight ensemble |
| 8 | submission-optimized-v1.csv | 0.84339 | 2026-06-06 | TF-IDF + leakage-free features |
| 9 | submission-tfidf-v2.csv | 0.86572 | 2026-06-06 | 50K subsample |
| 10 | submission-lgb-kfold-final.csv | 1.18779 | 2026-06-06 | K-Fold features (leakage) |
| 11 | submission-260606-stage2.csv | 1.31628 | 2026-06-06 | Multimodal (leakage) |
| 12 | submission-260606-stage1.csv | 1.59341 | 2026-06-06 | Stats features (leakage) |

---

## Key Findings

### 1. Target Leakage is the Main Problem

The complex models (Stage 1-2, CatBoost, Stacking) had **worse** Kaggle scores (1.2-1.6) despite better local OOF RMSE (0.55) because statistical features leak target information:

- `user_te`, `prod_te` (target encoding)
- `avg_rating`, `prod_avg_rating` (aggregate statistics)
- `user_stats`, `product_stats` (behavioral statistics)

**Why it leaks**: These features are computed on the full training set. When predicting test data, the model has already "seen" that user's/product's ratings during training.

### 2. TF-IDF Features Generalize Best

TF-IDF features (5000 dimensions) are computed only from the text (title + comment) and don't leak target information. They transfer well to test data.

### 3. Regularization Helps

Adding regularization to LightGBM improved generalization:
- **More leaves** (63 → 127): More model capacity
- **Subsampling** (1.0 → 0.8): Each tree sees 80% of data
- **Column sampling** (1.0 → 0.8): Each tree sees 80% of features

### 4. Adding Features Hurts Performance

Surprisingly, adding leakage-free features (temporal, text_length, votes, purchased) **worsened** performance:
- TF-IDF only: 0.79012
- TF-IDF + extra features: 0.84339

This suggests the extra features add noise without useful signal.

### 5. Ensemble/Blending Doesn't Help

Ensembling multiple models or blending with Stage 0 didn't significantly improve over the single best model:
- Best single: 0.79012
- Ensemble: 0.80706
- Blend 80/20: 0.79142

---

## Experiments Tried

### TF-IDF Configurations
- max_features: 5000, 10000, 15000, 20000
- ngram_range: (1,1), (1,2), (1,3)
- sublinear_tf: True, False
- min_df: 1, 2, 5
- max_df: 0.9, 0.95, 1.0

### LightGBM Hyperparameters
- num_leaves: 31, 63, 127, 255
- learning_rate: 0.01, 0.03, 0.05, 0.1
- n_estimators: 200, 300, 500, 1000
- subsample: 0.6, 0.8, 1.0
- colsample_bytree: 0.6, 0.8, 1.0
- reg_alpha: 0, 0.1, 1.0
- reg_lambda: 0, 0.1, 1.0

### Other Approaches
- Adding leakage-free features (temporal, text_length, votes, purchased)
- K-Fold target encoding
- Ensemble of multiple models
- Blending best model with Stage 0
- Post-processing (rounding, different clipping)

---

## Best Model Configuration

```python
# TF-IDF
TfidfVectorizer(
    max_features=5000,
    sublinear_tf=True,
    strip_accents='unicode'
)

# LightGBM
LGBMRegressor(
    n_estimators=500,
    num_leaves=127,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1,
    n_jobs=-1,
    random_seed=42
)
```

**Training time**: ~700 seconds (93s TF-IDF + 606s LightGBM)

---

## Why Competitor is Better (0.62 vs 0.79)

The competitor's score of 0.62 is **21% better** than ours. Possible explanations:

1. **Neural networks**: They might be using transformer models (BERT, RoBERTa) fine-tuned on the text
2. **Better feature engineering**: Features we haven't thought of
3. **Better text preprocessing**: Different tokenization, cleaning, etc.
4. **Ensemble of many diverse models**: Averaging many different approaches
5. **Target encoding with proper K-Fold**: Avoiding leakage while using user/product stats

---

## Recommendations for Next Steps

To close the 21% gap, try:

1. **Neural networks**: Use the pre-computed BERT embeddings (768 dimensions) with a simple MLP
2. **Install XGBoost**: `pip install xgboost` and try it
3. **Character-level n-grams**: Try character-level features
4. **Better text preprocessing**: Different tokenization, cleaning, etc.
5. **Transformer fine-tuning**: Directly fine-tune a pre-trained language model

---

## Files Generated

| File | Description |
|------|-------------|
| `output/submission-tfidf-regularized.csv` | Best Kaggle submission (0.79012) |
| `code/models/comprehensive_optimization.py` | Comprehensive optimization script |
| `code/models/train_tfidf_quick.py` | Quick optimization script |
| `code/models/train_tfidf_ultrafast.py` | Ultra-fast optimization script |
| `code/models/create_ensemble.py` | Ensemble creation script |
| `code/models/postprocess_best.py` | Post-processing script |
| `output/submission-ensemble_*.csv` | Ensemble submissions (avg, weighted, median) |
| `output/submission-clip_*.csv` | Clipped submissions (various ranges) |
| `output/submission-rounded_*.csv` | Rounded submissions |
| `output/submission-blend_*.csv` | Blended submissions (various ratios) |
| `docs/changelog/metrics.json` | Updated with all experiment results |
| `tech_dashboard.html` | Updated with latest scores and findings |

---

## New Submissions Created (2026-06-07 12:30)

### Ensemble Methods
- **submission-ensemble_avg.csv** - Average of all existing submissions
- **submission-ensemble_weighted.csv** - Weighted average (best=50%, baseline=20%, others=30%)
- **submission-ensemble_median.csv** - Median ensemble

### Post-processing Methods
- **submission-clip_10_45.csv** - Clipped to [1.0, 4.5]
- **submission-clip_15_50.csv** - Clipped to [1.5, 5.0]
- **submission-clip_20_50.csv** - Clipped to [2.0, 5.0]
- **submission-rounded_05.csv** - Rounded to nearest 0.5
- **submission-rounded_1.csv** - Rounded to nearest integer

### Blending Methods
- **submission-blend_90_9.csv** - 90% best + 10% baseline
- **submission-blend_80_19.csv** - 80% best + 20% baseline
- **submission-blend_70_30.csv** - 70% best + 30% baseline
- **submission-blend_60_40.csv** - 60% best + 40% baseline

---

## Kaggle API Issue

The provided Kaggle API token (`KGAT_893a1232ecb6176168e09ae410e0c29d`) is returning 401 Unauthorized. This could be because:
1. The token is expired
2. The token format is incorrect (usually Kaggle tokens are 32-character hex strings without prefix)
3. The competition name is different

**Action Required**: Please regenerate your Kaggle API token from https://www.kaggle.com/settings and update the `~/.kaggle/kaggle.json` file.

---

## Conclusion

We achieved a best Kaggle score of **0.79012** using TF-IDF features with regularized LightGBM. This represents a 1.4% improvement over the baseline. However, the competitor has reached 0.62, creating a 21% gap that requires fundamentally different approaches (likely neural networks or advanced feature engineering).

The key insight is that **simple TF-IDF features generalize better than complex statistical features** due to target leakage issues. Regularization (subsample, colsample) helps improve generalization.

---

*Report updated: 2026-06-07 12:30:00*

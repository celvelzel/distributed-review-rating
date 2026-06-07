# Step 2: Kaggle Scores — Complete History

**Last Updated**: 2026-06-07 04:00:00

## Current Status

| Metric | Value |
|--------|-------|
| **Best Kaggle Score** | 0.79012 |
| **Competitor Score** | 0.62 |
| **Gap** | 21% |
| **Best Model** | TF-IDF + Regularized LightGBM |

---

## All Kaggle Submissions

| Rank | File | Score | Date | Description |
|------|------|-------|------|-------------|
| 1 | submission-tfidf-regularized.csv | **0.79012** | 2026-06-06 | TF-IDF 5K + LGB (127 leaves, subsample=0.8, colsample=0.8) |
| 2 | submission-blend_80_20.csv | 0.79142 | 2026-06-07 | Blend 80% best + 20% stage0 |
| 3 | submission-clip_1_5_round.csv | 0.79281 | 2026-06-07 | Best model rounded to nearest 0.5 |
| 4 | stage0_submission.csv | 0.80107 | 2026-06-06 | Stage 0 baseline (TF-IDF 5K + LGB 500 trees) |
| 5 | submission-stage0-repro.csv | 0.80109 | 2026-06-06 | Stage 0 reproduction |
| 6 | submission-ensemble-weighted.csv | 0.80276 | 2026-06-07 | Weighted ensemble (0.5, 0.3, 0.2) |
| 7 | submission-ensemble.csv | 0.80706 | 2026-06-07 | Equal weight ensemble of 3 models |
| 8 | submission-optimized-v1.csv | 0.84339 | 2026-06-06 | TF-IDF + leakage-free features (worse) |
| 9 | submission-tfidf-v2.csv | 0.86572 | 2026-06-06 | 50K subsample (worse, less data) |
| 10 | submission-lgb-kfold-final.csv | 1.18779 | 2026-06-06 | K-Fold features (leakage) |
| 11 | submission-260606-stage2.csv | 1.31628 | 2026-06-06 | Multimodal LGB (leakage) |
| 12 | submission-260606-stage1.csv | 1.59341 | 2026-06-06 | Stats features (leakage) |
| 13 | submission-260606.csv | 1.28850 | 2026-06-06 | Unknown (leakage) |

---

## Score Analysis

### Best Model Details
- **File**: submission-tfidf-regularized.csv
- **Score**: 0.79012
- **Model**: LightGBM with regularization
- **Features**: TF-IDF (5000 dimensions)
- **Parameters**:
  - n_estimators: 500
  - num_leaves: 127
  - learning_rate: 0.05
  - subsample: 0.8
  - colsample_bytree: 0.8
- **Training time**: ~700 seconds

### Why This Model Works
1. **No target leakage**: TF-IDF features are computed only from text
2. **Regularization**: subsample and colsample prevent overfitting
3. **More leaves**: 127 leaves capture more complex patterns

### Why Other Models Failed
1. **Stage 1-2, CatBoost, Stacking**: Target leakage in statistical features
2. **TF-IDF + extra features**: Extra features add noise
3. **Ensemble/Blending**: Models too similar, no diversity
4. **Subsample models**: Less training data hurts performance

---

## Competitor Analysis

**Competitor score**: 0.62
**Our best**: 0.79012
**Gap**: 21%

Possible reasons for competitor's better performance:
1. Neural networks (transformer models)
2. Better feature engineering
3. Better text preprocessing
4. Ensemble of many diverse models
5. Target encoding with proper K-Fold

---

## Recommendations

To close the 21% gap:
1. Try neural networks with BERT embeddings
2. Install and try XGBoost
3. Try character-level n-grams
4. Try better text preprocessing
5. Try transformer fine-tuning

---

## Files Location

All submission files are in: `output/submission-*.csv`

---

*Report generated: 2026-06-07 04:00:00*

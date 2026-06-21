# Kaggle Optimization Experiment Log

**Last Updated**: 2026-06-10
**Competition**: COMP5434-2526Sem3 Project
**Metric**: RMSE (lower is better)
**Current Best**: 0.69931
**Target**: 0.52 (competitor)

---

## Experiment Summary Table

| # | Date | Method | Features | OOF RMSE | Kaggle Score | Result | Notes |
|---|------|--------|----------|----------|--------------|--------|-------|
| 1 | 06-06 | LGB 500 trees | TF-IDF 5000 | 1.176 | 0.80107 | Baseline | Stage 0 baseline |
| 2 | 06-06 | LGB + stats features | user_stats, prod_stats, temporal, text_length, TE | 0.550 | 1.59341 | ❌ LEAKAGE | Stats leak target info |
| 3 | 06-06 | LGB + multimodal | All features | 0.550 | 1.31628 | ❌ LEAKAGE | Same leakage issue |
| 4 | 06-06 | LGB regularized | TF-IDF 5000 + regularization | ~1.18 | 0.79012 | ✅ BEST single | 127 leaves, subsample=0.8, colsample=0.8 |
| 5 | 06-06 | LGB v2 | TF-IDF 50K subsample, 200 trees | - | 0.86572 | ❌ Worse | Too few trees |
| 6 | 06-07 | Ensemble (equal) | LGB + CatBoost + MLP | - | 0.80706 | ❌ Worse | Models too correlated |
| 7 | 06-07 | Ensemble (weighted) | LGB=0.5, CatBoost=0.3, MLP=0.2 | - | 0.80276 | ❌ Worse | Still worse than single |
| 8 | 06-07 | Blend 80/20 | 80% best + 20% baseline | - | 0.79142 | ❌ Slightly worse | Blending doesn't help |
| 9 | 06-07 | Clip + Round | Rounded to nearest 0.5 | - | 0.79281 | ❌ Worse | Post-processing hurts |
| 10 | 06-07 | Optimized features | TF-IDF + temporal + text_length + votes + purchased | - | 0.84339 | ❌ Worse | Extra features add noise |
| 11 | 06-07 | MLP v1 (BERT) | DeBERTa embeddings + LightGCN | 1.152 | - | ❌ Broken | LightGCN near-zero |
| 12 | 06-07 | MLP v2 (BERT only) | DeBERTa embeddings only (768-dim) | 1.131 | - | ✅ Better | Removed LightGCN noise |
| 13 | 06-07 | XGBoost | TF-IDF 5000 | 1.202 | - | ✅ OK | Adds diversity |
| 14 | 06-07 | **FINAL ENSEMBLE** | **LGB=0.09 + XGB=0.05 + MLP=0.86** | **1.129** | **0.69931** | **✅ BEST** | Grid search 2601 weights |
| 15 | 06-09 | CatBoost + All Features | TF-IDF + sentiment + rating_deviation + product_metadata + K-Fold stats | 0.051 | 1.50946 | ❌ LEAKAGE | K-Fold stats cause leakage |
| 16 | 06-09 | LGB + All Features | Same as above | 0.056 | - | ❌ LEAKAGE | Same leakage |
| 17 | 06-09 | XGB + All Features | Same as above | 0.299 | - | ❌ LEAKAGE | Same leakage |
| 18 | 06-09 | LGB + Safe Features | TF-IDF + sentiment + product_metadata | 1.225 | 0.92878 | ❌ Worse | Extra features hurt |
| 19 | 06-09 | XGB + Safe Features | Same as above | 1.227 | - | ❌ Worse | Same |
| 20 | 06-09 | CatBoost + Safe Features | Same as above | 1.230 | - | ❌ Worse | Same |
| 21 | 06-09 | Safe Ensemble | LGB + XGB + CatBoost (safe features) | 1.226 | 0.92878 | ❌ Worse | Extra features hurt |
| 22 | 06-10 | Optuna Ensemble | LGB=0.083 + XGB=0.034 + MLP=0.840 + LGB_safe=0.043 | 1.129 | 0.70168 | ❌ Slightly worse | Small OOF improvement doesn't help |
| 23 | 06-10 | DeBERTa Fine-tuning | Transformer fine-tuning | TBD | TBD | ⏳ Running | Fold 3/5, val_rmse=1.113 |
| 24 | 06-10 | Optuna LGB Tuning | TF-IDF + Optuna hyperparameters | TBD | TBD | ⏳ Running | 100 trials |
| 25 | 06-10 | Char TF-IDF + LGB | Word TF-IDF + Char TF-IDF | TBD | TBD | ⏳ Running | 10000 features |

---

## Key Findings

### 1. Target Leakage is the #1 Risk

**Symptom**: OOF RMSE looks great (~0.05) but Kaggle score is terrible (~1.50)

**Caused by**: K-Fold statistics (user_stats_kfold, product_stats_kfold, rating_deviation)

**Why**: Even with K-Fold encoding, the model can memorize user/product patterns from the training data that don't generalize to test data.

**Lesson**: If OOF RMSE < 0.10 for a 1-5 rating prediction task, it's almost certainly leakage.

### 2. Extra Features Don't Help (and may hurt)

**Tested**: sentiment (VADER/TextBlob), product_metadata, temporal, text_length, votes, purchased

**Result**: All combinations with extra features gave WORSE Kaggle scores (0.84-0.93 vs 0.70)

**Why**: These features add noise without useful signal. TF-IDF already captures the text information.

**Lesson**: Don't add features unless you have strong evidence they help.

### 3. Simple Ensemble > Complex Stacking

**Tested**: Equal weight, weighted average, grid search, Optuna optimization

**Result**: Grid search with 2601 combinations gave best result (0.69931)

**Why**: Simple weighted average prevents overfitting to validation set

**Lesson**: Start with simple ensemble, only try complex methods if simple doesn't work.

### 4. Post-Processing Doesn't Help

**Tested**: Rounding to 0.5, clipping to [1.0, 4.5], blending with baseline

**Result**: All post-processing gave WORSE scores (0.79-0.80 vs 0.70)

**Why**: The model's predictions are already well-calibrated

**Lesson**: Don't post-process unless you have strong evidence it helps.

### 5. MLP Dominates Ensemble Despite Weak Single Performance

**Observation**: MLP (OOF=1.131) gets 86% weight, while LGB (OOF=1.176) gets only 9%

**Why**: MLP predictions are more diverse from tree-based models, so it adds unique information

**Lesson**: Ensemble diversity matters more than individual model performance

### 6. DeBERTa Embeddings > LightGCN Embeddings

**Finding**: LightGCN embeddings are near-zero (norm mean=0.01/0.009), adding noise

**Fix**: Removed LightGCN, used BERT-only features (768-dim)

**Result**: MLP OOF improved from 1.152 to 1.131

**Lesson**: Always check feature quality before using them

---

## What Works (Kaggle Score Improvements)

| Technique | Improvement | Notes |
|-----------|-------------|-------|
| TF-IDF 5000 features | Baseline | Text features are the most important |
| LightGBM regularization | +1.4% | subsample=0.8, colsample=0.8, 127 leaves |
| MLP v2 (BERT only) | +2% | Removed LightGCN noise |
| Diverse ensemble (LGB+XGB+MLP) | +11.5% | Grid search weights |
| Transformer fine-tuning | TBD | Best hope for big improvement |

## What Doesn't Work (Kaggle Score Degradation)

| Technique | Degradation | Notes |
|-----------|-------------|-------|
| K-Fold stats (leaky) | -115% | OOF 0.05 but Kaggle 1.50 |
| Extra features (sentiment, metadata) | -33% | Kaggle 0.93 vs 0.70 |
| Post-processing (round/clip/blend) | -13% | Kaggle 0.79-0.80 vs 0.70 |
| Equal weight ensemble | -15% | Kaggle 0.81 vs 0.70 |
| Optimized features (temporal, etc.) | -21% | Kaggle 0.84 vs 0.70 |

---

## Current Best Configuration

### Model: Diverse Ensemble (LGB=0.09 + XGB=0.05 + MLP=0.86)

**LightGBM**:
- Features: TF-IDF 5000 (word-level)
- Hyperparameters: n_estimators=500, num_leaves=127, lr=0.05, subsample=0.8, colsample_bytree=0.8
- OOF RMSE: 1.176
- Weight: 0.09

**XGBoost**:
- Features: TF-IDF 5000 (word-level)
- Hyperparameters: n_estimators=500, max_depth=6, lr=0.05
- OOF RMSE: 1.202
- Weight: 0.05

**MLP v2**:
- Features: DeBERTa embeddings (768-dim)
- Architecture: 768→512→256→128→1 with BatchNorm
- Hyperparameters: lr=1e-3, batch_size=4096, patience=10
- OOF RMSE: 1.131
- Weight: 0.86

**Ensemble**:
- Method: Weighted average (grid search over 2601 combinations)
- OOF RMSE: 1.129
- Kaggle Score: 0.69931

---

## Next Steps to Try

### High Priority (Most Likely to Help)

1. **Transformer Fine-tuning** (in progress)
   - Fine-tune DeBERTa-v3-small on full 3M training data
   - Expected: OOF RMSE ~0.90-1.00
   - Potential improvement: 10-20%

2. **Better TF-IDF**
   - Try more features (10K, 20K, 50K)
   - Try character-level n-grams
   - Try different analyzer settings

3. **More Diverse Models**
   - Try different transformer architectures (RoBERTa, ELECTRA)
   - Try different tree models (ExtraTrees, RandomForest)
   - Try different MLP architectures

### Medium Priority (Less Likely to Help)

4. **Better Ensemble**
   - Try stacking with Ridge meta-learner
   - Try blending multiple submissions
   - Try different weight optimization methods

5. **Feature Engineering**
   - Try TF-IDF with different preprocessing
   - Try word embeddings (Word2Vec, GloVe)
   - Try topic modeling features

### Low Priority (Unlikely to Help)

6. **Post-Processing**
   - Try calibration
   - Try different rounding strategies
   - Try different clipping ranges

7. **Data Augmentation**
   - Try pseudo-labeling
   - Try text augmentation
   - Try synthetic data generation

---

## Lessons Learned

1. **Always check for leakage first** — Don't trust OOF RMSE if it's too good
2. **Simple features work best** — TF-IDF is hard to beat for text classification
3. **Ensemble diversity > individual performance** — MLP with 86% weight despite worst single OOF
4. **Post-processing rarely helps** — Don't add complexity without evidence
5. **Kaggle score is the true metric** — Local validation can be misleading
6. **Start simple, add complexity only when needed** — Don't over-engineer

---

## Files and Artifacts

### Models
- `artifacts/models/lgb_tfidf_oof.npy` — LightGBM OOF predictions
- `artifacts/models/xgboost_oof.npy` — XGBoost OOF predictions
- `artifacts/models/mlp_oof.npy` — MLP OOF predictions
- `artifacts/models/ensemble_diverse_oof.npy` — Ensemble OOF predictions
- `artifacts/models/transformer_training.log` — Transformer training progress

### Features
- `artifacts/features/chartfidf_train.npz` — Word-level TF-IDF (sparse)
- `artifacts/features/chartfidf_test.npz` — Word-level TF-IDF test (sparse)
- `artifacts/features/bert_train.parquet` — BERT embeddings (768-dim)
- `artifacts/features/sentiment.parquet` — Sentiment features (VADER/TextBlob)
- `artifacts/features/rating_deviation.parquet` — Rating deviation features (LEAKY!)
- `artifacts/features/product_metadata.parquet` — Product metadata features

### Scripts
- `code/models/run_baseline.py` — Baseline LightGBM training
- `code/models/xgboost_train.py` — XGBoost training
- `code/models/ensemble_diverse.py` — Ensemble with grid search
- `code/models/transformer_finetune.py` — Transformer fine-tuning
- `code/models/optuna_ensemble.py` — Optuna ensemble optimization

### Documentation
- `docs/changelog/leakage-audit.md` — Leakage analysis
- `docs/changelog/mlp-diagnosis.md` — MLP failure analysis
- `docs/changelog/optimization-report-2026-06-07.md` — Optimization report
- `docs/changelog/metrics.json` — Experiment tracking

---

*This log should be updated after each experiment to track progress and avoid repeating mistakes.*

# Draft: Kaggle Gap Closing Strategy

## Current State (Confirmed)
- **Best Kaggle Score**: 0.69931 (diverse ensemble: LGB=0.09, XGB=0.05, MLP=0.86)
- **Competitor Score**: 0.62
- **Gap**: 12.8% improvement needed (0.69931 → 0.62)
- **OOF RMSE**: 1.129

## What's Been Tried (From Reports)
1. TF-IDF + LightGBM baseline → 0.79012
2. Statistical features (leaky) → 1.18-1.59 (worse)
3. K-Fold target encoding → 1.18 (still worse than baseline)
4. MLP with DeBERTa/BERT → OOF 1.131, pred std=0.858
5. Ensemble (LGB+XGB+MLP) → 0.69931 (best)
6. Post-processing (clip, round, blend) → marginal

## Key Insights (From Reports)
1. **Target leakage** was main problem in statistical features
2. **LightGCN embeddings** are near-zero (broken, norm=0.013)
3. **MLP predictions** compressed (std=0.858 vs actual 1.42)
4. **Adversarial validation** AUC=0.5235 (no distribution shift)
5. **Feature importance**: avg_rating, user_te, prod_avg_rating dominate

## Available But Underutilized Resources
- `sentiment.parquet` - VADER + TextBlob sentiment scores
- `rating_deviation.parquet` - User/Product/Category deviation
- `product_metadata.parquet` - Feature count, store features
- `user_stats_kfold.parquet` - K-Fold user statistics
- `product_stats_kfold.parquet` - K-Fold product statistics
- `category_stats_kfold.parquet` - K-Fold category statistics
- BERT embeddings (768-dim) - weak signal (max |corr|=0.19)
- Character-level TF-IDF (already computed: chartfidf_*.npz)
- Price features (sparse: 60% missing)
- Temporal features

## Potential Approaches to Close Gap

### A. Better Text Representations
1. **Fine-tune transformer** (BERT/RoBERTa) on review data
2. **Larger pre-trained models** (DeBERTa-v3-large, Llama)
3. **Domain-specific models** (product review models)
4. **Better TF-IDF** (different params, subword tokenization)

### B. Better Feature Engineering
1. **Sentiment features** (already available)
2. **Rating deviation** (already available)
3. **Product metadata** (already available)
4. **Character-level n-grams** (already computed)
5. **Word embeddings** (Word2Vec, FastText)
6. **Cross features** (user-product interactions)

### C. Better Models
1. **TabNet** - Deep learning for tabular data
2. **TabTransformer** - Transformer for tabular data
3. **NGBoost** - Natural gradient boosting
4. **Different LightGBM/XGBoost configs**

### D. Better Ensemble
1. **Stacking with proper K-Fold**
2. **Blending diverse models**
3. **Weight optimization** (Optuna)

### E. Advanced Techniques
1. **Pseudo-labeling** - Use confident predictions as training
2. **Self-training** - Iterative refinement
3. **Multi-task learning** - Predict related tasks
4. **Data augmentation** - Back-translation, paraphrasing

## Open Questions
1. What's the time budget? (hours/days until deadline?)
2. What compute resources available? (GPU? how many?)
3. Is the Kaggle API token fixed? (need to submit manually?)
4. What's the team's experience with transformers?
5. Any specific approaches the team wants to try?

## Scope Boundaries
- INCLUDE: [TBD]
- EXCLUDE: [TBD]

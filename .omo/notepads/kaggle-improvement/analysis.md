# Kaggle Improvement Analysis — 2026-06-07

## Current State
- **Best Kaggle Score**: 0.79012 (TF-IDF regularized LightGBM)
- **Competitor Score**: 0.62
- **Gap**: 21% (MASSIVE)
- **Best Local OOF**: 0.548 (CatBoost with 927 features)

## Critical Issues Identified

### 1. Local OOF vs Kaggle Gap (0.548 vs 0.79)
The K-Fold features achieve great local OOF but poor Kaggle performance. This suggests:
- Subtle leakage in feature generation that doesn't show in OOF
- Train/test distribution mismatch
- Features that don't generalize to unseen products/users

### 2. MLP is Broken (OOF RMSE = 1.152)
The MLP with DeBERTa + LightGCN embeddings performs terribly. Root causes:
- Architecture too simple (likely just 2-3 linear layers)
- No dropout/regularization
- Learning rate not tuned
- Batch size too small
- No learning rate scheduling

### 3. TF-IDF Only Works Because It's Simple
TF-IDF has no target leakage. All other features leak to some degree.

### 4. Ensemble Doesn't Help
All models use similar features → no diversity → ensemble doesn't improve.

## Improvement Strategies (Ranked by Expected Impact)

### Priority 1: Fix the MLP (Expected: -0.15 to -0.20 RMSE)
- Use DeBERTa embeddings (768d) + LightGCN (64d) = 832 features
- Architecture: 832 → 512 → 256 → 128 → 1
- Dropout: 0.3-0.5
- Batch size: 1024-4096
- Learning rate: 1e-3 with cosine annealing
- Weight decay: 1e-4
- Early stopping on validation

### Priority 2: Character-level TF-IDF (Expected: -0.02 to -0.05 RMSE)
- char_wb analyzer with n-gram range (3, 5)
- 10000-20000 features
- Often captures patterns word-level misses

### Priority 3: Better Text Preprocessing (Expected: -0.01 to -0.03 RMSE)
- Remove HTML tags more aggressively
- Handle contractions (don't → do not)
- Remove special characters
- Lemmatization

### Priority 4: XGBoost (Expected: -0.01 to -0.02 RMSE)
- Haven't tried it yet
- Different algorithm → more diversity for ensemble

### Priority 5: Better Ensemble (Expected: -0.02 to -0.05 RMSE)
- Use diverse models (LGB + CatBoost + XGBoost + MLP)
- Different feature subsets for each model
- Ridge meta-learner with proper CV

### Priority 6: Product Metadata Features (Expected: -0.01 to -0.02 RMSE)
- Price (log transform)
- Category embeddings
- Store embeddings
- Rating number (from prodInfo)

## Key Insight
The competitor's 0.62 score suggests they're using:
1. Fine-tuned transformer models (not just embeddings)
2. Better text preprocessing
3. Better ensemble of diverse models
4. Properly validated features

## Next Steps
1. Fix MLP training (biggest potential gain)
2. Add character-level TF-IDF
3. Try XGBoost
4. Build diverse ensemble

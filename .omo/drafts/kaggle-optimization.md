# Draft: Kaggle Performance Optimization Plan

## Current State (Confirmed)
- **Best Kaggle Score**: 0.79012 RMSE (TF-IDF 5K + Regularized LightGBM)
- **Local OOF Best**: 0.545 RMSE (Stacking ensemble - BUT HAS TARGET LEAKAGE)
- **Competitor Score**: 0.62 RMSE (21% gap)
- **Plan Progress**: Only 1/38 tasks complete from existing plan

## Key Technical Findings
1. **Target Leakage is #1 Problem**: Stages 1-2, CatBoost, and stacking use features that leak target info (user_te, prod_te, avg_rating). Local RMSE looks great (0.54x) but Kaggle scores are terrible (1.18-1.59).
2. **Simple TF-IDF wins on Kaggle**: Best Kaggle score comes from TF-IDF(5K) + regularized LGB -- no fancy features, no ensemble.
3. **Regularization is key**: subsample=0.8, colsample=0.8, num_leaves=127 outperforms baseline.
4. **Ensembling hurts when models lack diversity**: All tree-based models on similar features produce correlated predictions.
5. **Extra features add noise**: Temporal, text_length, votes, purchased features actually HURT Kaggle score (0.84339 vs 0.79012).

## Feature Status
### Implemented (Leak-Safe)
- TF-IDF 5000-dim (title+comment)
- DeBERTa-v3 768-dim embeddings (frozen)
- LightGCN user/item embeddings (64-dim each)
- Temporal features (year, month, weekday, hour, is_weekend)
- Text length features (title_len, comment_len, ratio, has_caps, has_exclamation)
- Price features (log_price, price_rank_in_category, price_bucket)
- K-Fold target encoding (user_te, prod_te) - SAFE but not used in best model

### Implemented (LEAKY - Do Not Use)
- user_stats (avg_rating, num_reviews, avg_votes, purchased_rate, rating_std)
- product_stats (prod_avg_rating, prod_num_reviews)
- category_stats (cat_avg_rating, cat_avg_price)

### NOT Implemented Yet
- Sentiment/polarity features
- Rating deviation features
- Product features list parsing
- Store/brand features
- Character-level TF-IDF
- XGBoost model

## Model Status
| Model | Status | OOF RMSE | Kaggle Score | Notes |
|-------|--------|----------|--------------|-------|
| LightGBM + TF-IDF | ✅ Done | 1.176 | 0.79012 | BEST KAGGLE |
| CatBoost + Stats | ⚠️ Leaky | 0.548 | 1.188 | Target leakage |
| MLP + DeBERTa | ⚠️ Broken | 1.152 | N/A | Architecture issue |
| Stacking (LGB+Cat+MLP) | ⚠️ Leaky | 0.545 | N/A | Uses leaky models |
| XGBoost | ❌ Not tried | - | - | Potential diversity |

## Kaggle Submission History (Top 5)
| Rank | File | Score | Notes |
|------|------|-------|-------|
| 1 | submission-tfidf-regularized.csv | 0.79012 | BEST |
| 2 | submission-blend_80_20.csv | 0.79142 | 80% best + 20% baseline |
| 3 | submission-clip_1_5_round.csv | 0.79281 | Rounded to 0.5 |
| 4 | stage0_submission.csv | 0.80107 | Baseline |
| 5 | submission-ensemble-weighted.csv | 0.80276 | Weighted ensemble |

## Requirements (Confirmed)
- **Priority**: Close the 21% gap (0.79→0.62) - Aggressive approach
- **Neural Models**: Fix MLP AND add XGBoost for maximum ensemble diversity
- **Time**: 1+ week - Full scope optimization
- **Test Strategy**: TDD (Test-Driven Development)
- **Must use leakage-safe features only**
- **Must use 5-fold OOF validation**
- **Must record experiment tracking**
- **Must generate reproducible results**

## Scope Boundaries
- INCLUDE: Feature engineering, model training, ensemble, submission, ablation experiments
- INCLUDE: Fix MLP architecture, add XGBoost, character-level TF-IDF
- INCLUDE: Comprehensive leakage audit and fix
- EXCLUDE: Full LLM fine-tuning (unless evidence shows improvement)

## Clearance Checklist
- [x] Core objective clearly defined - Close 21% gap (0.79→0.62)
- [x] Scope boundaries established - Aggressive, full scope, 1+ week
- [x] No critical ambiguities remaining - All questions answered
- [x] Technical approach decided - Fix leakage, fix MLP, add XGBoost, character TF-IDF, better ensemble
- [x] Test strategy confirmed - TDD
- [x] No blocking questions outstanding

→ ALL YES - Proceeding to plan generation

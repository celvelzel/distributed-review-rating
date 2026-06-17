# 2026-06-17: Graph Feature Optimization — XGBoost RMSE Breakthrough

## Summary

Optimized graph features model from Ridge (OOF 0.6679) to XGBoost (OOF 0.5304), achieving 34.3% improvement over former Ridge baseline (0.8076). Key breakthrough: tree-based models capture non-linear feature interactions that Ridge misses.

---

## Part 1: Bug Fix — User-Category Features

### Problem
`user_cat_avg_rating`, `user_cat_review_count`, `user_cat_deviation` were all zeros in training set.

### Root Cause
In `expand_graph_features.py`, the `reindex` call failed because:
- `train_df['id']` has 1M unique values but 3M rows (duplicates)
- `train_feats.reindex(train_df['id'].values)` mapped everything to NaN

### Fix
Removed the broken `reindex` call. The merge already preserves row order.

**Before:** OOF RMSE = 1.4195 (broken features)  
**After:** OOF RMSE = 0.6728 (correct features)

---

## Part 2: Feature Expansion

### Features Used (10 → 18)

**Expanded features (10d):**
- Store metadata: `store_product_count`, `store_avg_rating_number`, `store_total_rating_number`, `store_has_name`
- User deviation: `user_leniency`, `user_harshness`, `user_num_reviews_oof`
- User-category: `user_cat_avg_rating`, `user_cat_review_count`, `user_cat_deviation`

**Stats features (5d):**
- User: `user_avg_rating`, `user_num_reviews`, `user_rating_std`
- Product: `prod_avg_rating`, `prod_num_reviews`

**Interaction features (3d):**
- `leniency_x_reviews` = user_leniency × user_num_reviews_oof
- `cat_dev_x_reviews` = user_cat_deviation × user_cat_review_count
- `user_prod_diff` = user_avg_rating − prod_avg_rating

---

## Part 3: Model Optimization

### Ridge Regression (Baseline)
- Features: 18d (expanded + stats + interactions)
- OOF RMSE: 0.6679
- Alpha: 1.0 (all alphas tested: 0.01, 0.1, 1.0, 10.0, 100.0 — same result)

### XGBoost (Best)
- Features: 10d (top features by importance)
- Settings: lr=0.3, max_depth=4, subsample=0.8, colsample_bytree=0.8, rounds=150
- OOF RMSE: **0.5304** (5-fold on full 3M data)
- Runtime: ~6 min/fold, ~30 min total

**Feature importance (XGBoost):**
1. `user_cat_avg_rating` (dominant — 87% of importance)
2. `user_leniency`
3. `user_cat_deviation`
4. `user_harshness`
5. `leniency_x_reviews`
6. `user_prod_diff`
7. `prod_avg_rating`
8. `cat_dev_x_reviews`
9. `user_cat_review_count`
10. `user_num_reviews_oof`

### LightGBM (Timeout)
- Attempted on full data but timed out (>10 min/fold)
- On 50k sample: OOF RMSE = 0.5592

### GradientBoosting (sklearn)
- On 100k sample: OOF RMSE = 0.5516
- Too slow for full data

---

## Part 4: Results Comparison

| Model | Features | OOF RMSE | vs Former Ridge |
|-------|----------|----------|-----------------|
| Former Ridge (stats only) | 11d | 0.8076 | baseline |
| Ridge (expanded only) | 10d | 0.6728 | -16.7% |
| Ridge (expanded + stats) | 18d | 0.6679 | -17.3% |
| **XGBoost (full data)** | **10d** | **0.5304** | **-34.3%** |
| LightGBM (50k sample) | 18d | 0.5592 | -30.8% |
| GBM sklearn (100k sample) | 18d | 0.5516 | -31.7% |

---

## Part 5: Generated Submissions

| File | Description |
|------|-------------|
| `output/deberta_ve90_xgb10.csv` | DeBERTa VE 90% + XGBoost 10% |
| `output/deberta_ve95_xgb5.csv` | DeBERTa VE 95% + XGBoost 5% |
| `output/deberta_ve85_xgb15.csv` | DeBERTa VE 85% + XGBoost 15% |
| `output/deberta_ve80_xgb20.csv` | DeBERTa VE 80% + XGBoost 20% |

---

## Part 6: Technical Details

### Why XGBoost Beats Ridge
1. **Non-linear interactions**: XGBoost captures complex feature interactions (e.g., user_leniency × review_count)
2. **Feature importance**: Automatically selects most predictive features
3. **Robust to outliers**: Tree splits handle extreme values better than linear regression

### Why Top 10 Features Only
- Using all 18 features: OOF 0.5304 (same)
- Using top 10 features: Faster training, less overfitting risk
- `user_cat_avg_rating` dominates (87% importance) — other features add marginal value

### Overfitting Prevention
- 5-fold OOF validation on full 3M data
- Early stopping (20 rounds patience)
- Limited tree depth (max_depth=4)
- Subsampling (80% rows, 80% features per tree)

---

## Files Modified

| File | Change |
|------|--------|
| `code/features/expand_graph_features.py` | Fixed reindex bug, added parquet support |
| `code/features/test_expanded_features.py` | Updated to use kfold stats |
| `code/features/xgboost_full.py` | New: XGBoost training pipeline |
| `code/features/optimize_ridge_final.py` | New: Ridge optimization |
| `code/features/blend_expanded_ridge.py` | New: Blend submissions |
| `code/features/optimize_graph_features.py` | New: Multi-model comparison |

---

## Lessons Learned

1. **Tree models >> Linear models** for tabular data with interactions
2. **Feature engineering matters**: Interaction features improved Ridge by 0.7%, XGBoost by more
3. **Speed vs accuracy tradeoff**: XGBoost with fast settings (lr=0.3, depth=4) completed in 6 min/fold vs LightGBM timing out
4. **Feature selection**: Using top 10 features prevents overfitting without sacrificing accuracy
5. **Full data > Subsample**: XGBoost on full 3M data (0.5304) beat subsample (0.5438) by 2.5%

---

## Next Steps

1. Submit `deberta_ve90_xgb10.csv` to Kaggle
2. Compare with DeBERTa-large results when training completes
3. Consider pseudo-labeling with XGBoost predictions
4. Try ensemble of DeBERTa + XGBoost + Ridge for diversity

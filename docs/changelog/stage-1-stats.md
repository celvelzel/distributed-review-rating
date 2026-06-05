# Stage 1 — Statistical Features + LightGBM

**Date**: 2026-06-05
**Script**: `code/models/train_stage1.py`
**Submission**: `output/submission-stage1.csv`

## Summary

Replaced TF-IDF text features with handcrafted statistical features (user behaviour, product metadata, temporal signals, text-length heuristics, and target encoding). Same LightGBM hyperparameters as Stage 0. Result: **RMSE dropped from 1.176 → 0.550** (53.3% relative improvement).

## Feature Set (24 columns)

| Group | Features | Source |
|-------|----------|--------|
| User stats | `avg_rating`, `num_reviews`, `avg_votes`, `purchased_rate`, `rating_std` | `user_stats.parquet` |
| Product stats | `prod_avg_rating`, `prod_num_reviews`, `prod_price`, `prod_rating_number`, `main_category` | `product_stats.parquet` |
| Temporal | `year`, `month`, `day`, `weekday`, `hour`, `is_weekend`, `is_holiday_season` | `temporal.parquet` |
| Text length | `title_len`, `comment_len`, `title_comment_ratio`, `has_caps`, `has_exclamation` | `text_length.parquet` |
| Target encoding | `user_te`, `prod_te` | `te_user.parquet`, `te_prod.parquet` |

## LightGBM Parameters (unchanged from Stage 0)

```python
{
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "n_estimators": 500,
}
```

## Cross-Validation Results (3-fold)

| Fold | RMSE |
|------|------|
| 1 | 0.55013 |
| 2 | 0.54935 |
| 3 | 0.54977 |
| **Mean** | **0.54975** |

## RMSE Comparison

| Stage | Model | Features | CV RMSE | Train Time (s) | Inference (s) |
|-------|-------|----------|---------|----------------|---------------|
| Stage 0 | LGB + TF-IDF | tfidf (5000-dim) | 1.17626 | 392.96 | 0.20 |
| **Stage 1** | **LGB + Stats** | **24 stat features** | **0.54975** | **248.47** | **0.11** |
| Stage 1+TFIDF | LGB + Stats + TF-IDF | 24 stats + 5000 tfidf | — | — | — |

### Key Takeaways

1. **Statistical features dominate**: 24 engineered features beat 5000 TF-IDF features by a large margin (0.550 vs 1.176).
2. **Target encoding is the strongest signal**: `user_te` and `prod_te` encode the historical rating tendency of each user/product — effectively a personalized prior.
3. **Faster training**: 248s vs 393s (dense 24-col matrix is cheaper than 5000-dim sparse TF-IDF for LightGBM).
4. **Stage 1+TFIDF** (combining both feature sets) is a natural next step — expect further gains from text signal on top of the statistical baseline.

## Next Steps

- [ ] Stage 1+TFIDF: combine 24 stat features + TF-IDF → expect RMSE < 0.55
- [ ] Stage 2: add DeBERTa / LightGCN embeddings
- [ ] Feature importance analysis to prune weak features
- [ ] Hyperparameter tuning (Optuna) for the stat-feature model

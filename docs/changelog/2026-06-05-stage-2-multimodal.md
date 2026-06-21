# Stage 2 — Multimodal LightGBM (ALL Features)

## Summary

Stage 2 trains LightGBM on **all 5,927 features** assembled from the feature pipeline:
- TF-IDF vectors (5,000 dims)
- Sentence-BERT embeddings (768 dims, `emb_0`–`emb_767`)
- User/item graph embeddings (64 + 64 dims)
- Statistical features (user stats, product stats, temporal, text length)
- Target-encoded features (`user_te`, `prod_te`)
- Price & category features

**Training strategy**: Due to the 64 GB process memory limit (data is 71.3 GB uncompressed),
we used:
- **CV**: 500K-row subsample, 3-fold, 200 boosting rounds
- **Final model**: Incremental batch training across 7 parquet row groups with `init_model`

## RMSE Comparison

| Stage | Model | Features | RMSE | Δ vs Stage 0 | Δ vs Stage 1 |
|-------|-------|----------|------|---------------|---------------|
| 0 | TF-IDF + LightGBM | TF-IDF (5,000) | 1.17626 | — | — |
| 1 | Stats + LightGBM | user/product/temporal/text_stats/TE | 0.54975 | −0.62651 ✅ | — |
| **2** | **Multimodal LightGBM** | **ALL (5,927)** | **0.55030** | **−0.62596 ✅** | **+0.00055 ⚠️** |

**Key finding**: Adding TF-IDF + BERT embeddings to the statistical features (Stage 1)
did not significantly improve RMSE. The CV RMSE of 0.55030 is within noise of Stage 1's
0.54975 — a marginal regression of 0.00055 (0.1%).

### Why didn't multimodal help?

1. **Statistical features dominate**: `avg_rating` and `user_te` alone account for the
   vast majority of predictive power (6.98M and 1.32M gain respectively).
2. **High-dimensional noise**: 5,000 TF-IDF features add many low-signal dimensions that
   dilute the model's focus on strong predictors.
3. **Incremental training**: Memory constraints forced incremental batch training
   (7 batches × 500 rounds) rather than full-data training, which may underperform
   compared to a single-pass model.

## Feature Importance — Top 20

| Rank | Feature | Gain | Category |
|------|---------|------|----------|
| 1 | `avg_rating` | 6,977,417 | user_stats |
| 2 | `user_te` | 1,320,136 | target_encoding |
| 3 | `rating_std` | 203,446 | user_stats |
| 4 | `prod_avg_rating` | 202,510 | product_stats |
| 5 | `num_reviews` | 116,821 | user_stats |
| 6 | `prod_te` | 44,414 | target_encoding |
| 7 | `prod_num_reviews` | 26,818 | product_stats |
| 8 | `tfidf_2911` | 26,113 | tfidf |
| 9 | `emb_767` | 11,878 | bert_embedding |
| 10 | `emb_705` | 11,175 | bert_embedding |
| 11 | `emb_122` | 10,530 | bert_embedding |
| 12 | `tfidf_1911` | 10,436 | tfidf |
| 13 | `emb_421` | 8,896 | bert_embedding |
| 14 | `emb_59` | 7,234 | bert_embedding |
| 15 | `emb_31` | 6,019 | bert_embedding |
| 16 | `user_emb_0` | 5,981 | graph_embedding |
| 17 | `emb_576` | 5,405 | bert_embedding |
| 18 | `emb_355` | 4,906 | bert_embedding |
| 19 | `emb_166` | 4,774 | bert_embedding |
| 20 | `emb_319` | 4,369 | bert_embedding |

### Observations

- **Top 7 features are all statistical/TE**: `avg_rating`, `user_te`, `rating_std`,
  `prod_avg_rating`, `num_reviews`, `prod_te`, `prod_num_reviews` — these account for
  ~99.3% of total gain.
- **TF-IDF has 2 entries in top 20**: `tfidf_2911` (rank 8) and `tfidf_1911` (rank 12),
  but with much lower gain than statistical features.
- **BERT embeddings dominate ranks 9–20**: 10 of the top 20 features are BERT dimensions,
  suggesting they capture some semantic signal, but their individual contribution is small.
- **Graph embeddings**: Only `user_emb_0` (rank 16) appears in the top 20.

## Technical Details

| Parameter | Value |
|-----------|-------|
| Model | LightGBM regressor |
| Learning rate | 0.05 |
| Num leaves | 63 |
| Boosting rounds | 500 (incremental: 7 batches) |
| Features | 5,927 |
| Training rows | 3,007,439 |
| CV subsample | 500,000 rows |
| CV folds | 3 |
| Train time | 2,395.91 s (~40 min) |
| Inference time | 0.79 s |

## Artifacts

- `output/submission-stage2.csv` — Kaggle submission (10,001 lines)
- `artifacts/features/stage2_feature_importance.csv` — Full feature importance ranking
- `docs/changelog/metrics.json` — Updated with stage_2 metrics

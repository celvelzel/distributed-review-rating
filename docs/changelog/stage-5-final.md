# T23: Final Pipeline & Submission

## Summary

Generated final Kaggle submission using stacking ensemble test predictions.

**Submission**: `output/submission-final.csv` (10,001 lines: header + 10,000 predictions)

## Pipeline

```
stacking_test.npy (10,000) → clip [1, 5] → join with test IDs → submission-final.csv
```

- Input: `artifacts/models/stacking_test.npy` — Ridge meta-learner predictions
- IDs: `artifacts/etl/test.parquet` (id column)
- Output: `output/submission-final.csv` (id, rating)

## RMSE Progression

| Stage | Model | OOF RMSE | Δ vs Previous | Notes |
|-------|-------|----------|---------------|-------|
| 0 | TF-IDF + LGB | 1.1763 | — | Baseline (text-only) |
| 1 | Stats + LGB | 0.5498 | −0.6265 | User/product stats, temporal |
| 2 | All Features + LGB | 0.5503 | +0.0006 | TF-IDF + embeddings + stats |
| CatBoost | CatBoost (927 feat) | 0.5480 | −0.0023 | Best single model |
| **Stacking** | **Ridge ensemble** | **0.5453** | **−0.0027** | **Final model** |

**Total improvement from baseline**: 1.1763 → 0.5453 (**−53.6%** RMSE reduction)

## Stacking Ensemble Details

### Base Model OOF RMSE

| Model | OOF RMSE | Features | Notes |
|-------|----------|----------|-------|
| CatBoost | 0.5480 | 927 (non-TFIDF) | Best single model |
| LightGBM | 0.5524 | 927 (non-TFIDF) | 5-fold CV regenerated |
| MLP | 1.1520 | 896 (embeddings) | DeBERTa + LightGCN |

### Ridge Meta-Learner Coefficients

| Model | Coefficient | Relative Weight |
|-------|-------------|-----------------|
| CatBoost | 0.6837 | 68.4% |
| LightGBM | 0.3181 | 31.8% |
| MLP | 0.0438 | 4.4% |

- **Meta-learner**: Ridge Regression (α=1.0, fit_intercept=True)
- **Stacking CV**: 5-fold (same seed as base models)

### Per-Fold CatBoost RMSE

| Fold | RMSE |
|------|------|
| 1 | 0.54878 |
| 2 | 0.54761 |
| 3 | 0.54780 |
| 4 | 0.54828 |
| 5 | 0.54737 |
| **Mean** | **0.54797** |

## Training Time Summary

| Component | Time (s) | Notes |
|-----------|----------|-------|
| Stage 0 (TF-IDF + LGB) | 393.0 | Baseline |
| Stage 1 (Stats + LGB) | 248.5 | Feature engineering |
| Stage 2 (All Features + LGB) | 2395.9 | Full feature set |
| CatBoost (5-fold CV) | 10361.9 | Best single model |
| LGB OOF regeneration | 783.3 | 5-fold CV with 927 features |
| Stacking CV (Ridge) | 1.1 | Meta-learner fitting |
| **Total** | **~14,183** | **~3.9 hours** |

## Feature Count

| Feature Group | Count | Description |
|---------------|-------|-------------|
| User stats | ~15 | avg_rating, count, std, etc. |
| Product stats | ~20 | avg_rating, count, category stats |
| Temporal | ~8 | hour, weekday, month, days_since |
| Text length | ~5 | title_len, comment_len, word_count |
| Target encoding | ~10 | user/product TE (fold-safe) |
| TF-IDF (LGB) | ~5000 | Sparse text features (Stage 0/2 only) |
| BERT embeddings | 384 | Sentence-transformer embeddings |
| Price features | ~5 | price, log_price, price_ratio |
| Category stats | ~10 | category-level aggregations |
| LightGCN | 64 | Graph neural network embeddings |
| **Total (non-TFIDF)** | **927** | Used by CatBoost & best LGB |
| **Total (all)** | **~6000+** | Including TF-IDF sparse features |

## Submission Verification

| Check | Result |
|-------|--------|
| Line count | 10,001 (header + 10,000 data) ✅ |
| Header | `id,rating` ✅ |
| ID range | 0–9999 ✅ |
| Rating range | [1.0000, 5.0000] ✅ |
| All in [1, 5] | Yes ✅ |

## Files Generated

| File | Description |
|------|-------------|
| `code/models/predict.py` | Final prediction script |
| `output/submission-final.csv` | Kaggle submission (10,001 lines) |
| `docs/changelog/stage-5-final.md` | This report |
| `metrics.json` | Updated with final stacking RMSE |

# Step 2: Retrieve Historical Kaggle Scores

## Kaggle Authentication Issue

I encountered a 401 Unauthorized error when trying to retrieve Kaggle scores. Please regenerate your Kaggle API token and update the `kaggle.json` file.

## Local Scores from Documentation

| Stage | Model | OOF RMSE | Notes |
|-------|-------|----------|-------|
| 0 | TF-IDF + LGB | 1.1763 | Baseline (text-only) |
| 1 | Stats + LGB | 0.5498 | User/product stats, temporal |
| 2 | All Features + LGB | 0.5503 | TF-IDF + embeddings + stats |
| CatBoost | CatBoost (927 feat) | 0.5480 | Best single model |
| Stacking | Ridge ensemble | 0.5453 | Final model |

## Files Generated

- `docs/changelog/step1-history.md` - Historical overview
- `docs/changelog/step2-kaggle-scores.md` - This report

# Step 1: Read and Understand the History

## Project Summary

The project is a Kaggle competition for predicting review ratings (1–5 stars) using distributed data processing and machine learning. The project has progressed through several stages, improving from a baseline RMSE of 1.1763 to a final RMSE of 0.5453.

## Stages and Experiments

### Stage 0: Baseline (TF-IDF + LightGBM)
- **Model**: LightGBM
- **Features**: TF-IDF (5,000 dims)
- **RMSE**: 1.1763
- **Notes**: Initial baseline using text features only.

### Stage 1: Statistical Features + LightGBM
- **Model**: LightGBM
- **Features**: User stats, product stats, temporal, text length, target encoding
- **RMSE**: 0.54975
- **Notes**: Replaced TF-IDF with handcrafted statistical features, resulting in a 53.3% RMSE reduction.

### Stage 2: Multimodal LightGBM (ALL Features)
- **Model**: LightGBM
- **Features**: TF-IDF, sentence-BERT embeddings, user/item graph embeddings, statistical features, target-encoded features, price & category features
- **RMSE**: 0.55030
- **Notes**: Adding TF-IDF and BERT embeddings to Stage 1 features did not significantly improve RMSE (marginal regression of 0.00055).

### CatBoost
- **Model**: CatBoost
- **Features**: 927 non-TFIDF features
- **OOF RMSE**: 0.5480
- **Notes**: Best single model.

### Stacking Ensemble (Ridge Meta-Learner)
- **Model**: Ridge Regression (α=1.0)
- **Base Models**: CatBoost, LightGBM, MLP
- **OOF RMSE**: 0.5453
- **Notes**: Stacking improved RMSE by 0.0027 compared to the best single model.

## Key Findings

- **Statistical features dominate**: User/product stats and target encoding provide the majority of predictive power.
- **Text features add noise**: TF-IDF and BERT embeddings did not significantly improve RMSE.
- **Overfitting concern**: Stage 0 (simple TF-IDF) had a high RMSE, but subsequent complex models did not bring significant improvement.
- **Adversarial validation**: No significant distribution shift between train and test sets.

## Kaggle Authentication Issue

I encountered a 401 Unauthorized error when trying to retrieve Kaggle scores. Please regenerate your Kaggle API token and update the `kaggle.json` file.

## Local Scores

| Stage | Model | OOF RMSE | Notes |
|-------|-------|----------|-------|
| 0 | TF-IDF + LGB | 1.1763 | Baseline (text-only) |
| 1 | Stats + LGB | 0.5498 | User/product stats, temporal |
| 2 | All Features + LGB | 0.5503 | TF-IDF + embeddings + stats |
| CatBoost | CatBoost (927 feat) | 0.5480 | Best single model |
| Stacking | Ridge ensemble | 0.5453 | Final model |

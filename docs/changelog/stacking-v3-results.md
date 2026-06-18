# Stacking v3 Results

**Date**: 2026-06-18 19:31

**Base models**: 9 (lgb_tfidf, xgboost, mlp, lgb_safe_dense, xgboost_safe, catboost_safe, ensemble_diverse, xgb_graph_safe, lgb_graph_safe)

## Individual Model OOF RMSE

| Model | OOF RMSE | Mean | Std |
|-------|----------|------|-----|
| lgb_tfidf | 1.19651 | 3.9375 | 0.7693 |
| xgboost | 1.20156 | 3.9418 | 0.6969 |
| mlp | 1.13119 | 3.9467 | 0.8580 |
| lgb_safe_dense | 1.22464 | 3.9412 | 0.7164 |
| xgboost_safe | 1.22676 | 3.9413 | 0.7119 |
| catboost_safe | 1.23014 | 3.9412 | 0.7041 |
| ensemble_diverse | 1.12938 | 3.9457 | 0.8275 |
| xgb_graph_safe | 1.36246 | 3.9409 | 0.4081 |
| lgb_graph_safe | 1.36235 | 3.9408 | 0.4044 |

## Meta-Learner Comparison

| Meta-Learner | OOF RMSE |
|-------------|----------|
| ridge | 1.12046 |
| lgb | 1.11774 |
| catboost | 1.11799 |
| elasticnet | 1.12042 |
| ridge+lgb | 1.11774 ★ |

**Best**: ridge+lgb (OOF RMSE = 1.11774)

## Ridge Coefficients (avg across folds)

| Model | Coefficient |
|-------|-------------|
| ensemble_diverse | 0.7447 |
| lgb_graph_safe | 0.4040 |
| mlp | 0.1445 |
| lgb_safe_dense | 0.1279 |
| xgboost | 0.0591 |
| lgb_tfidf | -0.0064 |
| xgboost_safe | -0.0188 |
| catboost_safe | -0.0263 |
| xgb_graph_safe | -0.0640 |

## vs Stacking v2

- Mean absolute difference (test): 0.11116
- Max absolute difference (test): 1.25790
- Correlation (test): 0.981049

## DeBERTa 1M Blend Simulation

| Blend | Mean | Std |
|-------|------|-----|
| deb_ve95_sv3_5 | 4.0139 | 1.2075 |
| deb_ve90_sv3_10 | 4.0141 | 1.1836 |
| deb_ve85_sv3_15 | 4.0143 | 1.1598 |
| deb_ve80_sv3_20 | 4.0145 | 1.1362 |

## Recommended Kaggle Submission Order

1. `submission-stacking-v3.csv` — standalone stacking v3 (diagnostic)
2. `submission-deb1m-ve90-sv3-10.csv` — primary: mirrors 0.617 recipe
3. `submission-deb1m-ve85-sv3-15.csv` — if v3 significantly better than v2

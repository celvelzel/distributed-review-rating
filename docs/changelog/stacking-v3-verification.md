# Stacking v3 Verification Report

**Date**: 2026-06-18 19:31

## 1. Stacking v3 OOF Quality

| Metric | Value |
|--------|-------|
| OOF RMSE | 1.11774 |
| Mean | 3.9425 |
| Std | 0.8750 |
| Min | 1.20 |
| Max | 4.97 |

## 2. Stacking v3 vs v2

*stacking_v2_oof.npy not found — cannot compute direct OOF comparison*

Test prediction comparison (v3 vs v2):
- Mean |diff|: 0.11116
- Max |diff|: 1.25790
- Rows with |diff| > 0.05: 7040 / 10000
- Correlation: 0.981049

**Verdict**: CHANGED — test predictions differ significantly from v2

## 3. Meta-Learner Breakdown

| Meta-Learner | OOF RMSE | vs Best |
|-------------|----------|---------|
| ridge+lgb | 1.11774 ★ | +0.00000 |
| lgb | 1.11774 | +0.00000 |
| catboost | 1.11799 | +0.00025 |
| elasticnet | 1.12042 | +0.00268 |
| ridge | 1.12046 | +0.00272 |

## 4. DeBERTa 1M Blend Simulation

DeBERTa 1M fold1 test: mean=4.0219, std=0.8247
After VE: mean=4.0137, std=1.2315, scale=1.7242

| Blend Ratio | Test Mean | Test Std | Notes |
|-------------|-----------|----------|-------|
| 95% DeBERTa_VE + 5% v3 | 4.0139 | 1.2075 |  |
| 90% DeBERTa_VE + 10% v3 | 4.0141 | 1.1836 | ← mirrors 0.617 recipe |
| 85% DeBERTa_VE + 15% v3 | 4.0143 | 1.1598 |  |
| 80% DeBERTa_VE + 20% v3 | 4.0145 | 1.1362 |  |
| 75% DeBERTa_VE + 25% v3 | 4.0147 | 1.1128 |  |

**Baseline (v2):**
| Blend Ratio | Test Mean | Test Std | Notes |
|-------------|-----------|----------|-------|
| 95% DeBERTa_VE + 5% v2 | 4.0150 | 1.2081 |  |
| 90% DeBERTa_VE + 10% v2 | 4.0163 | 1.1848 | ← current best 0.61734 |
| 85% DeBERTa_VE + 15% v2 | 4.0176 | 1.1616 |  |
| 80% DeBERTa_VE + 20% v2 | 4.0189 | 1.1385 |  |

**90/10 blend per meta-learner variant:**
| Variant | Test Mean | Test Std |
|---------|-----------|----------|
| ridge | 4.0144 | 1.1840 |
| lgb | 4.0141 | 1.1836 |
| catboost | 4.0141 | 1.1837 |
| elasticnet | 4.0138 | 1.1839 |
| ridge+lgb | 4.0141 | 1.1836 |

## 5. Base Model Contribution

**Ridge coefficients (positive = helpful, negative = harmful):**

| Model | Coefficient | Signal Type |
|-------|-------------|-------------|
| ensemble_diverse | 0.7447 | Mixed ensemble |
| lgb_graph_safe | 0.4040 | Graph features (NEW) |
| mlp | 0.1445 | DeBERTa embedding |
| lgb_safe_dense | 0.1279 | Sentiment+Metadata |
| xgboost | 0.0591 | Text TF-IDF |
| lgb_tfidf | -0.0064 | Text TF-IDF |
| xgboost_safe | -0.0188 | Sentiment+Metadata |
| catboost_safe | -0.0263 | Sentiment+Metadata |
| xgb_graph_safe | -0.0640 | Graph features (NEW) |

**Graph models have POSITIVE Ridge weights**: {'lgb_graph_safe': 0.40400474816560744}
→ Graph features contribute useful signal to the meta-learner.

## 6. Recommendation

Stacking v3 test predictions differ from v2 but OOF comparison unavailable.
Submit to Kaggle to determine if the change is beneficial:
1. `submission-stacking-v3.csv` — standalone (diagnostic)
2. `submission-deb1m-ve90-sv3-10.csv` — primary

---
*Verification completed in 0.3s*
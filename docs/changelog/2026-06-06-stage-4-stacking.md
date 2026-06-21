# T20: Stacking Ensemble with Ridge Meta-Learner

**Date**: 2026-06-06

## Architecture
- **Meta-learner**: Ridge Regression (α=1.0, fit_intercept=True)
- **Base models**: LGB, CatBoost, MLP
- **Stacking CV**: 5-fold (same seed as base models)
- **Features**: Base model OOF predictions only (no raw features)

## Base Model OOF RMSE

| Model | OOF RMSE | Features | Notes |
|-------|----------|----------|-------|
| CatBoost | 0.54797 | 927 (non-TFIDF) | Best single model |
| LightGBM | 0.55239 | 927 (non-TFIDF) | Regenerated via 5-fold CV |
| MLP | 1.15201 | 896 (embeddings) | DeBERTa + LightGCN |
| **Stacking** | **0.54528** | — | Ridge meta-learner |

## LightGBM OOF (Regenerated)

| Fold | RMSE |
|------|------|
| 1 | 0.56020 |
| 2 | 0.55561 |
| 3 | 0.55464 |
| 4 | 0.55715 |
| 5 | 0.55206 |
| **Mean** | **0.55593** |

## Ridge Coefficients (Model Weights)

| Model | Coefficient | Interpretation |
|-------|-------------|----------------|
| LGB | 0.318077 | ~31.8% relative weight |
| CatBoost | 0.683746 | ~68.4% relative weight |
| MLP | 0.043784 | ~4.4% relative weight |

## Improvement

- Best single model RMSE: 0.54797
- Stacking OOF RMSE: 0.54528
- **Δ improvement: +0.00268** (✅ improved)

## Timing

- LGB OOF generation: 783.3s
- Stacking CV: 1.1s
- Total time: 784.6s

## Outputs

- Stacking OOF: `artifacts/models/stacking_oof.npy` (3,007,439,)
- Stacking test: `artifacts/models/stacking_test.npy` (10,000)

## Notes

- LGB OOF regenerated using 927 non-TFIDF features (same as CatBoost) via 5-fold CV
- LGB test predictions loaded from `output/submission-stage2.csv` (trained on 5927 features)
- Ridge Regression chosen over neural network meta-learner to prevent overfitting
- Predictions clipped to [1.0, 5.0]

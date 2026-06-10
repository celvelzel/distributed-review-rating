# Kaggle Optimization V2 — Push Towards 0.62

## TL;DR

> **Quick Summary**: Use unused features (sentiment, rating_deviation, product_metadata) with LightGBM/XGBoost, add CatBoost to ensemble, and try stacking meta-learner to push from 0.69931 towards 0.62.
> 
> **Target**: Kaggle score < 0.65
> **Current**: 0.69931
> **Gap**: ~7.5% improvement needed

---

## Current State

| Metric | Value |
|--------|-------|
| Kaggle Score | 0.69931 |
| Competitor | 0.62 |
| Gap | 12.8% |
| Ensemble OOF | 1.129 |

### Available Resources (UNUSED)

**Features:**
- `sentiment.parquet` - VADER + TextBlob sentiment scores
- `rating_deviation.parquet` - User/Product/Category deviation
- `product_metadata.parquet` - Feature count, store features
- `user_stats_kfold.parquet` - K-Fold user statistics
- `product_stats_kfold.parquet` - K-Fold product statistics

**Models:**
- `catboost_oof.npy` - CatBoost predictions
- `stacking_oof.npy` - Stacking meta-learner

---

## Optimization Tasks

### Wave 1: Feature-Rich Models (Parallel)

- [ ] 1. **LightGBM + All Features**
  - Load TF-IDF (5000) + sentiment + rating_deviation + product_metadata + user_stats_kfold + product_stats_kfold
  - Train with 5-fold OOF
  - Target: OOF RMSE < 1.10
  - Save: `artifacts/models/lgb_allfeatures_oof.npy`

- [ ] 2. **XGBoost + All Features**
  - Same features as Task 1
  - Train with 5-fold OOF
  - Target: OOF RMSE < 1.10
  - Save: `artifacts/models/xgboost_allfeatures_oof.npy`

- [ ] 3. **CatBoost + All Features**
  - Same features as Task 1
  - Train with 5-fold OOF
  - Target: OOF RMSE < 1.10
  - Save: `artifacts/models/catboost_allfeatures_oof.npy`

### Wave 2: Advanced Ensemble (After Wave 1)

- [ ] 4. **Stacking Meta-Learner**
  - Use OOF predictions from: MLP, LGB_TF-IDF, XGB_TF-IDF, LGB_All, XGB_All, CatBoost_All
  - Train Ridge/LightGBM as meta-learner
  - 5-fold OOF validation
  - Save: `artifacts/models/stacking_v2_oof.npy`

- [ ] 5. **Optuna Weight Optimization**
  - Search space: weights for 6+ models
  - Objective: minimize OOF RMSE
  - Trials: 1000
  - Compare with grid search

### Wave 3: Final Submission

- [ ] 6. **Generate Final Submission**
  - Best ensemble strategy from Wave 2
  - Submit to Kaggle
  - Record score

---

## Success Criteria

- [ ] Kaggle score < 0.68
- [ ] Kaggle score < 0.65
- [ ] Beat competitor (0.62)

---

## Execution Strategy

```
Wave 1 (Parallel):
├── Task 1: LGB + All Features [unspecified-high]
├── Task 2: XGB + All Features [unspecified-high]
└── Task 3: CatBoost + All Features [unspecified-high]

Wave 2 (Sequential):
├── Task 4: Stacking Meta-Learner [deep]
└── Task 5: Optuna Weight Optimization [unspecified-high]

Wave 3:
└── Task 6: Final Submission [quick]
```

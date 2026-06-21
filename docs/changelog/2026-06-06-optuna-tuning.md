# Optuna Hyperparameter Tuning Report (T21)

**Date**: 2026-06-06
**Script**: `code/models/optuna_tune.py`
**Output**: `artifacts/models/best_params.json`

## Summary

Optuna TPE-based hyperparameter optimization for LightGBM on the full 5927-feature
multimodal dataset. The search converged within 30 trials, identifying a configuration
that outperforms the default baseline by 19.9% on the tuning subsample.

## Configuration

| Parameter | Value |
|-----------|-------|
| Sampler | TPE (seed=42) |
| Trials | 30 |
| CV folds | 3 |
| Subsample rows | 2,000 |
| Boosting rounds | 20 |
| Features | 5,927 |

**Note**: Due to the extreme computational cost of LightGBM with 5927 features (~4.3s per
round per 1000 rows), a 2K-row subsample with 20 rounds was used to keep total wall time
under 75 minutes. Results are directionally correct but absolute RMSE values are higher
than full-data training.

## Search Space

| Hyperparameter | Values |
|----------------|--------|
| `num_leaves` | 31, 63, 127, 255 |
| `max_depth` | 6, 8, 10, 12 |
| `learning_rate` | 0.01, 0.05, 0.1 |
| `min_child_samples` | 10, 20, 50 |
| `feature_fraction` | 0.6, 0.8, 1.0 |
| `bagging_fraction` | 0.6, 0.8, 1.0 |

## Results

### Best Parameters (Trial #8)

```json
{
  "num_leaves": 127,
  "max_depth": 8,
  "learning_rate": 0.1,
  "min_child_samples": 20,
  "feature_fraction": 1.0,
  "bagging_fraction": 0.8,
  "bagging_freq": 1
}
```

### RMSE Comparison

| Configuration | RMSE (2K subsample) | Notes |
|---------------|---------------------|-------|
| Default (lr=0.05, leaves=63) | 0.89165 | Baseline |
| Best Optuna (trial #8) | 0.71368 | **-19.9% improvement** |
| Stage 2 full-data RMSE | 0.55030 | 3M rows, 500 rounds |
| Stage 1 stats-only RMSE | 0.54975 | 6 stats features |

### Top-5 Trials

| Trial | RMSE | num_leaves | max_depth | lr | min_child | feat_frac | bag_frac |
|-------|------|-----------|-----------|------|-----------|-----------|----------|
| 8 | 0.71368 | 127 | 8 | 0.1 | 20 | 1.0 | 0.8 |
| 25 | 0.71368 | 127 | 8 | 0.1 | 20 | 1.0 | 0.8 |
| 14 | 0.71890 | 255 | 6 | 0.1 | 20 | 1.0 | 0.8 |
| 2 | 0.74878 | 255 | 8 | 0.1 | 20 | 0.8 | 1.0 |
| 5 | 0.75113 | 63 | 6 | 0.1 | 20 | 0.8 | 0.8 |

### Worst-5 Trials

| Trial | RMSE | Key Issue |
|-------|------|-----------|
| 29 | 1.33065 | lr=0.01 (too low) |
| 4 | 1.32178 | lr=0.01 + low leaves + low frac |
| 15 | 1.27177 | lr=0.01 (too low) |
| 9 | 1.25572 | lr=0.01 + high depth |
| 6 | 1.05042 | lr=0.05 + high depth + low frac |

## Key Insights

1. **Learning rate dominates**: `lr=0.1` consistently outperforms `lr=0.05` and `lr=0.01`.
   With only 20 rounds, higher learning rates are essential for convergence.

2. **Moderate tree complexity wins**: `num_leaves=127` with `max_depth=8` outperforms both
   smaller (31/63) and larger (255) leaf counts. Overly deep trees (depth=12) hurt performance.

3. **Full feature fraction preferred**: `feature_fraction=1.0` consistently ranks among the
   best, suggesting all 5927 features carry signal. The model benefits from seeing all features.

4. **Moderate bagging helps**: `bagging_fraction=0.8` provides regularization without
   sacrificing too much signal.

5. **min_child_samples=20 is optimal**: Consistently selected in top trials. Lower values (10)
   show slight overfitting; higher values (50) underfit.

## Convergence Behavior

The TPE sampler converged by trial 14, repeatedly sampling the region around
`{num_leaves: 127-255, max_depth: 6-8, lr: 0.1}`. Trials 10-27 largely reproduced
the same two top configurations, indicating the search space was well-explored.

## Caveats

- **Subsample size**: Results are based on 2K rows (0.07% of training data). The relative
  ranking of hyperparameters should hold, but absolute RMSE will differ on full data.
- **Boosting rounds**: Only 20 rounds were used. With more rounds (500), the optimal
  learning rate may shift lower and the advantage of higher `num_leaves` may increase.
- **Feature interactions**: With 5927 features, the search space is large relative to the
  subsample. A larger sample or feature selection step could improve tuning quality.

## Recommended Next Steps

1. Apply best params to full-data training with 500 rounds
2. Consider `lr=0.05` with more rounds for potentially better generalization
3. Run a focused search around `{num_leaves: 100-200, max_depth: 6-10, lr: 0.05-0.1}`
   with a larger subsample (50K+ rows) if compute allows

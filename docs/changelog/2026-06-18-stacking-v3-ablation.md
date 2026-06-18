# 2026-06-18 Stacking V3 Ablation Study

## Overview

Ablation experiments to isolate the contribution of Stacking V3 improvements over V2 baseline.

## Ablation Design

| ID | File | Description | Meta-Learner | Base Models |
|----|------|-------------|--------------|-------------|
| A1 | ablation-v2-ridge.csv | Baseline: V2 config | Ridge α=1.0 | 7 (original) |
| A2 | ablation-v3-ridge.csv | + Graph features | Ridge α=1.0 | 9 (+xgb_graph, +lgb_graph) |
| A3 | ablation-v3-ridge-lgb.csv | + Ridge+LGB ensemble | Ridge+LGB (grid search) | 9 |

## What Changed in V3

1. **Added graph features**: `xgb_graph_safe` and `lgb_graph_safe` (graph-based user/product embeddings)
2. **Enhanced meta-learner**: Ridge+LGB ensemble with grid search for optimal blend weight (vs Ridge-only in V2)

## Results

| ID | Kaggle RMSE | vs Baseline | Δ |
|----|-------------|-------------|---|
| A1 (v2-ridge) | **0.61734** | — | — |
| A2 (v3-ridge) | 0.61746 | +0.00012 | Graph features slightly hurt |
| A3 (v3-ridge-lgb) | **0.61725** | -0.00009 | Ridge+LGB marginally better |

## Analysis

### Graph Features Impact (A1 vs A2)

- **Expected**: Graph features should capture user/product relationships not in TF-IDF
- **Actual**: +0.00012 RMSE degradation (within noise)
- **Interpretation**: Graph embeddings overlap with existing features (TF-IDF, target encoding) or are too noisy

### Meta-Learner Impact (A2 vs A3)

- **Expected**: Ridge+LGB should capture non-linear meta-learner patterns
- **Actual**: -0.00009 RMSE improvement (not significant)
- **Interpretation**: Ridge is already near-optimal for this stacking task; LGB adds complexity without payoff

### Overall Assessment

- All three ablations are within ±0.0002 RMSE of each other
- Stacking V3 improvements are **marginal** — not the path to significant gains
- Current bottleneck is **DeBERTa base model quality**, not meta-learner sophistication

## Recommendations

1. **Keep V3 config** (no regression, slight improvement)
2. **Consider removing graph features** to simplify pipeline (negligible impact)
3. **Focus on DeBERTa improvements**:
   - Better fine-tuning (learning rate, epochs, batch size)
   - Larger model (v3-large)
   - Better variance expansion strategy
   - Multi-fold ensemble

## Files

- `output/ablation-v2-ridge.csv` — A1 baseline
- `output/ablation-v3-ridge.csv` — A2 with graph features
- `output/ablation-v3-ridge-lgb.csv` — A3 with Ridge+LGB

---

*Created: 2026-06-18*

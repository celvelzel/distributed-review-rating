# 2026-06-16 Kaggle Optimization — Status Report

## Best Result
- **Kaggle RMSE: 0.61734** (DeBERTa-v3-base VE 90% + Ridge 10%) — UNCHANGED
- Baseline: 0.69931 → 11.7% improvement
- Target: 0.500 → 19% gap remains

## Today's Submissions (8 new)

| File | Kaggle RMSE | Description |
|------|-------------|-------------|
| base_ve_90_small_ve_10 | **0.63449** | Base VE 90% + Small VE 10% — NEW |
| base_ve_85_small_ve_15 | 0.63559 | Base VE 85% + Small VE 15% — NEW |
| base_ve_80_small_ve_20 | 0.63689 | Base VE 80% + Small VE 20% — NEW |
| base_ensemble_ve85_r15 | 0.67988 | 2-fold ensemble + Ridge 15% — NEW |
| base_ensemble_ve90_r10 | 0.69107 | 2-fold ensemble + Ridge 10% — NEW |
| base_ensemble_ve95_r5 | 0.70319 | 2-fold ensemble + Ridge 5% — NEW |
| base_ensemble | 0.70650 | 2-fold ensemble alone — NEW |
| base_ensemble_ve | 0.71620 | 2-fold ensemble VE alone — NEW |

## Key Findings

1. **Small model cannot beat base model** — Small VE best (0.634) vs base VE alone (0.633) vs base VE+Ridge (0.617). The small model adds diversity but not enough signal.

2. **Cross-fold averaging hurts** — Base ensemble (fold1+fold2 averaged) = 0.706, much worse than single fold1 (0.638). Different folds are on different data splits, averaging without OOF alignment is harmful.

3. **Base VE alone is competitive** — DeBERTa-v3-base VE alone = 0.633, close to the best 0.617. The 10% Ridge blend adds ~0.016 improvement.

4. **DeBERTa-v3-base FULL 3M training in progress** — Fold 3/3 epoch 2 running. GPU: 8.1GB/12.6GB (64%). ETA ~2.5h for completion.

## Small vs Base Model Analysis

| Metric | DeBERTa-v3-small | DeBERTa-v3-base |
|--------|------------------|-----------------|
| Params | 44M | 86M |
| OOF RMSE | 1.128 | 1.117 |
| Kaggle (alone) | 0.683 | 0.638 |
| Kaggle (VE) | 0.683 | 0.633 |
| Kaggle (VE+Ridge) | 0.645 | 0.617 |

**Conclusion:** Small model cannot beat base model. The 2x parameter difference translates to ~3.5% Kaggle improvement. To reach 0.5 target, we likely need:
- DeBERTa-v3-base with 5+ folds (currently 3)
- DeBERTa-v3-large (304M params) — needs ~20GB memory
- Multi-transformer ensemble

## Future Plan

### Immediate (today/tomorrow)
- Complete DeBERTa-v3-base 3-fold training (~2.5h remaining)
- Generate multi-fold predictions and submit
- Submit base+small VE blends (still have quota)

### Short-term (1-3 days)
- Increase folds to 5 for better ensemble diversity
- Train DeBERTa-v3-large (304M params) — request higher memory quota
- Try pseudo-labeling with best predictions

### Medium-term (1-2 weeks)
- Multi-transformer ensemble (base + large + small)
- Train complementary models (RoBERTa, ELECTRA)
- Better post-processing (quantile calibration)
- Target: Kaggle RMSE < 0.60

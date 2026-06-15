# 2026-06-15 Kaggle Optimization — Status Report

## Best Result
- **Kaggle RMSE: 0.61734** (DeBERTa-v3-base VE 90% + Ridge 10%)
- Baseline: 0.69931 → 11.7% improvement
- Target: 0.500 → 19% gap remains

## Today's Submissions (10)

| File | Kaggle RMSE | Description |
|------|-------------|-------------|
| submission-small_ve.csv | 0.68272 | DeBERTa-small VE alone |
| submission-small_ve_base5.csv | 0.67265 | Small VE + base 5% |
| submission-small_ve_ridge10.csv | 0.66161 | Small VE + Ridge 10% |
| submission-small_ve_base10.csv | 0.66327 | Small VE + base 10% |
| submission-small_ve_base15.csv | 0.65461 | Small VE + base 15% |
| submission-small_ve_ridge10_base5.csv | 0.65302 | Small VE + Ridge 10% + base 5% |
| submission-small_ve_base20.csv | 0.64672 | Small VE + base 20% |
| submission-small_ve_ridge10_base10.csv | 0.64520 | Small VE + Ridge 10% + base 10% |
| submission-small_ve_base25.csv | 0.63962 | Small VE + base 25% |

**Key finding:** DeBERTa-v3-small (44M params) is weaker than DeBERTa-v3-base (86M params). The small model's best blend (0.63962) doesn't beat the base model's fold1 result (0.61734).

## Learnings

1. **Variance expansion is the breakthrough technique** — DeBERTa predictions are compressed (std=0.82 vs target 1.42). Scaling to match target distribution improves Kaggle RMSE from 0.638 → 0.617.

2. **Model size matters** — DeBERTa-v3-base (86M) significantly outperforms DeBERTa-v3-small (44M) even with fewer folds.

3. **Memory is the bottleneck** — 15GB HPC limit prevents training v3-base on full 3M data. Subsampling to 500K-1M is required.

4. **Kaggle daily limit** — 10 submissions/day. Must prioritize the most promising variants.

5. **API token issues** — The `KGAT_95032a984dab4b2545f71383d9913c63` token was revoked. `KGAT_9895fa87525d5a9a3514ae8bd156320b` works for listing but hit daily write limit.

## Future Plan

### Immediate (when daily limit resets)
- Submit `base_ve_90_small_ve_10` (base VE 90% + small VE 10%)
- Submit `base_ve_85_small_ve_15` and `base_ve_80_small_ve_20`
- Try more aggressive variance expansion (scale > 1.72)

### Short-term (1-2 days)
- Resume DeBERTa-v3-base training with 1M subsample + gradient checkpointing
- Try DeBERTa-v3-base with 3 folds × 3 epochs (fits in 15GB with subsample)
- Generate multi-fold predictions for better ensemble

### Medium-term (3-7 days)
- Request higher memory quota to train v3-base on full 3M data
- Try DeBERTa-v3-large (304M params) — needs ~20GB memory
- Train complementary models (RoBERTa, ELECTRA) for ensemble diversity
- Pseudo-labeling: use best predictions to augment training data

### Long-term
- Multi-transformer ensemble (base + large + small)
- Better post-processing (quantile calibration, isotonic regression)
- Target: Kaggle RMSE < 0.60

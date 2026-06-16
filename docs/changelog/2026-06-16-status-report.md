# 2026-06-16 Kaggle Optimization — Status Report

## Best Result
- **Kaggle RMSE: 0.61734** (DeBERTa-v3-base VE 90% + Ridge 10%) — UNCHANGED
- Baseline: 0.69931 → 11.7% improvement
- Target: 0.500 → 19% gap remains

## Today's Submissions (10 total — ALL 10 DAILY SLOTS USED)

| File | Kaggle RMSE | Description |
|------|-------------|-------------|
| f1ve92_r8 | **0.69065** | ❌ NEW fold1 (full 3M) VE 92% + Ridge 8% |
| ve2favg90_r10 | 0.69107 | ❌ NEW 2-fold avg VE 90% + Ridge 10% |
| base_ve_90_small_ve_10 | 0.63449 | Base VE 90% + Small VE 10% |
| base_ve_85_small_ve_15 | 0.63559 | Base VE 85% + Small VE 15% |
| base_ve_80_small_ve_20 | 0.63689 | Base VE 80% + Small VE 20% |
| base_ensemble_ve85_r15 | 0.67988 | 2-fold ensemble + Ridge 15% |
| base_ensemble_ve90_r10 | 0.69107 | 2-fold ensemble + Ridge 10% |
| base_ensemble_ve95_r5 | 0.70319 | 2-fold ensemble + Ridge 5% |
| base_ensemble | 0.70650 | 2-fold ensemble alone |
| base_ensemble_ve | 0.71620 | 2-fold ensemble VE alone |

## Key Findings

1. **🔴 CRITICAL: Full 3M training predictions are WORSE than old 1M predictions** — New fold1 (full 3M, std=0.812) Kaggle=0.69 vs old fold1 (1M subsample, std=0.825) Kaggle=0.617. The predictions are correlated (r=0.966) but differ enough to cause 12% score degradation. Possible causes: different KFold seed, different training data composition, or model convergence issues.

2. **Old predictions still available** — `deberta_lora_fold1_test.npy` (June 13) exactly matches the 0.61734 submission. Use this for future submissions.

3. **Small model cannot beat base model** — Small VE best (0.634) vs base VE alone (0.633) vs base VE+Ridge (0.617). The small model adds diversity but not enough signal.

4. **Cross-fold averaging hurts** — Base ensemble (fold1+fold2 averaged) = 0.706, much worse than single fold1 (0.638). Different folds are on different data splits, averaging without OOF alignment is harmful.

5. **DeBERTa-v3-base FULL 3M training in progress** — Fold 3/3 epoch 2 running. GPU: 8.1GB/12.6GB (64%). ETA ~2.5h for completion. But predictions from this training are worse on Kaggle.

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

## Future Plan (Updated)

### Phase 1 — Investigate + OOF Ensemble (1-2 days)
- 🔴 Investigate why full 3M predictions are worse than old 1M predictions
- Complete fold 2 epoch 3 retrain
- Generate 3-fold OOF predictions using OLD predictions (deberta_lora_fold1)
- Try OOF-aligned ensemble with old predictions

### Phase 2 — DeBERTa-v3-large (3-5 days)
- 304M params, LoRA r=16, ~2.5GB VRAM fits RTX 3080 Ti
- 3f×3e training, ~3.4h/epoch, ~30h total
- Target: Kaggle 0.58-0.60

### Phase 3 — Pseudo-labeling + Multi-model Ensemble (5-10 days)
- Pseudo-labeling with best predictions
- Multi-transformer: base + large + RoBERTa-large
- OOF-aligned Ridge stacking

### Phase 4 — Final Tuning (10-15 days)
- Blend ratio optimization, variance expansion calibration
- Target: Kaggle RMSE < 0.58

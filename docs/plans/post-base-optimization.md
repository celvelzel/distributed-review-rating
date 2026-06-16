# Post-Base-Training Optimization Plan

## Current State
- **Best Kaggle RMSE: 0.61734** (DeBERTa-v3-base VE 90% + Ridge 10%)
- **Target: 0.500** — 19% gap
- **Deadline: July 1, 2026** — ~15 days remaining
- **DeBERTa-v3-base 3f×3e almost done** (fold 3/3, epoch 2 running)
- **Hardware: RTX 3080 Ti (12.6GB VRAM), 251GB RAM**

## Phase 1: Complete Base Training + OOF Ensemble (Day 1-2)

### 1a. Wait for fold 3 to complete (~2.5h)
- Fold 3 epoch 2 ETA ~52 min, then epoch 3 (~1.7h)
- Monitor: `tail -f artifacts/base_full_training.log`

### 1b. Retrain fold 2 epoch 3
- Set `latest.txt` to `fold2_epoch2.pt`, re-run `deberta_base_full.py`
- The fixed resume logic will pick up fold 2 from epoch 3
- ETA: ~1.7h

### 1c. Generate 3-fold OOF predictions
- Use `predict_base_checkpoints.py` to generate OOF for all 3 folds
- Build OOF-aligned ensemble (learn blend weights on OOF, apply to test)
- Expected improvement: +0.5-1% over single-fold predictions

### 1d. Submit OOF-aligned ensemble
- Variance-expanded 3-fold average + Ridge stacking
- Target: Kaggle RMSE < 0.61

## Phase 2: DeBERTa-v3-large (Day 3-5)

### Why large?
- 304M params vs 86M (3.5x bigger)
- DeBERTa-v3-base alone gets 0.638 → large should get ~0.60-0.62
- With VE + Ridge blend → target 0.58-0.60

### Feasibility
- **VRAM:** ~2.5GB with LoRA r=16, fits easily in 12.6GB
- **Speed:** ~2x slower than base (24 layers vs 12) → ~3.4h/epoch
- **3f×3e total:** ~30h → fits in 1.5 days
- **Memory:** 304M model weights ~600MB (FP16), system RAM fine

### Plan
1. Copy `deberta_base_full.py` → `deberta_large_full.py`
2. Change MODEL_NAME to `microsoft/deberta-v3-large`
3. Adjust BS if needed (start with BS=16, GradAcc=16 for same effective batch)
4. Run 3f×3e with checkpointing
5. Generate predictions, VE + Ridge blend, submit

## Phase 3: Pseudo-Labeling (Day 6-8)

### Approach
1. Use best model predictions on test set as pseudo-labels
2. Combine pseudo-labeled test data with training data
3. Retrain DeBERTa-base and/or large on augmented dataset
4. Use confidence weighting (high-confidence predictions get more weight)

### Risk
- Pseudo-labels can amplify errors if base model is wrong
- Need to clip/validate pseudo-labels to reasonable range (1-5)
- May not help if base model's errors are systematic

## Phase 4: Multi-Transformer Ensemble (Day 8-12)

### Models to try
- DeBERTa-v3-base (already done)
- DeBERTa-v3-large (Phase 2)
- RoBERTa-large (125M) — different pretraining, adds diversity
- ELECTRA-large (335M) — discriminator-based, different architecture

### Ensemble strategy
- OOF predictions from each model
- Ridge stacking with K-Fold CV
- Only add models that improve OOF RMSE

## Phase 5: Final Tuning (Day 12-15)

- Blend ratio optimization (grid search on OOF)
- Variance expansion calibration (try different scale factors)
- Submission selection (pick best from multiple approaches)
- Final Kaggle submissions

## Expected Score Trajectory

| Phase | Expected Kaggle RMSE | Improvement |
|-------|---------------------|-------------|
| Current best | 0.61734 | — |
| Phase 1 (3f OOF) | 0.610-0.615 | -0.5-1% |
| Phase 2 (large) | 0.58-0.60 | -2-4% |
| Phase 3 (pseudo) | 0.57-0.59 | -1-2% |
| Phase 4 (ensemble) | 0.55-0.58 | -2-4% |
| Phase 5 (tuning) | 0.53-0.56 | -1-3% |

**Realistic target: 0.55-0.58** (optimistic but achievable)
**Stretch target: 0.50** (requires everything to work perfectly)

## Decision Points

1. **After Phase 1:** If 3-fold OOF doesn't improve, skip to Phase 2 immediately
2. **After Phase 2:** If large model doesn't significantly beat base, focus on ensemble diversity (Phase 4)
3. **Phase 3 risk:** If pseudo-labeling hurts, revert and skip to Phase 4
4. **Time check at Day 10:** If < 0.58, focus on securing best possible score rather than chasing 0.50

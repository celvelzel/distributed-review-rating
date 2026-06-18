# 2026-06-17: 3M DeBERTa Stacking Experiment Analysis

## Summary

Tested 3M DeBERTa model with various stacking configurations on Kaggle. All 3M variants significantly underperformed the 1M baseline. Concluded that the 3M path should be abandoned.

## Submissions

| File | Kaggle RMSE | Description |
|------|-------------|-------------|
| deberta3m_ve90_stacking10.csv | 0.68126 | 3M 3f×3e avg VE 90% + Stacking 10% |
| submission-3m-f1e3-ve90-r10.csv | 0.68608 | 3M fold1 epoch3 VE 90% + Stacking 10% |
| submission-3m-f1e1-ve90-r10.csv | 0.71179 | 3M fold1 epoch1 VE 90% + Stacking 10% |
| deberta3m_ve90_ridge10.csv | 0.76285 | 3M 3f×3e VE 90% + Ridge 10% (XGBoost bug) |

## Analysis

### 1. Epoch Progression (fold1)
- epoch1: 0.71179 → epoch3: 0.68608
- Later epochs generalize better, but still far from 1M best (0.617)

### 2. Multi-fold Averaging Effect
- 3f×3e avg (0.681) < fold1 epoch3 (0.686)
- Averaging reduces variance but can't fix systematic overfitting

### 3. 3M vs 1M Gap
- Best 3M: 0.681 (3f×3e avg, VE 90% + Stacking 10%)
- Best 1M: 0.617 (fold1, VE 90% + Stacking 10%)
- **Gap: 0.064 (10.4%)**

### 4. Root Causes
1. **Overfitting**: 3M model capacity too large for the data
2. **VE + Stacking double-dipping**: 3M script applies VE then blends stacking; 1M uses stacking_v2_test.npy directly (which is already a calibrated meta-learner)
3. **XGBoost calibration bug**: `deberta3m_ve90_ridge10.csv` used XGBoost with mean=3.26 (should be 3.94), causing 0.763 Kaggle RMSE

## Stacking v2 Meta-Learner Clarification

The variable `ridge_test` in `deberta_lora_1m.py` (line 270) loads `stacking_v2_test.npy`, which is NOT necessarily Ridge. `stacking_v2.py` auto-selects the best of:
1. Ridge (α=1.0)
2. LightGBM
3. Ridge + LGB blend

The saved `.npy` is whichever had lowest OOF RMSE. The naming is misleading.

## Decision

**Abandon 3M path.** Focus on 1M DeBERTa for final submission. The 0.064 gap is too large to close with tuning.

## Files Affected
- `docs/progress/kaggle-optimization-progress.md` — Updated with new leaderboard data
- `output/deberta3m_ve*_stacking*.csv` — 8 new blend files (experiment only)
- `output/deberta_old_ve*_stacking*.csv` — 8 new blend files (1M model, not yet submitted)

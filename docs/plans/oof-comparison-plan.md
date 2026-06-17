# OOF Comparison Plan: Old 1M Fold1 vs New 3×3 Full 3M

## Goal
Compare the OOF prediction accuracy between:
- **Old model**: DeBERTa-v3-base LoRA, 1M subsample, fold1 (Kaggle best: 0.61734)
- **New model**: DeBERTa-v3-base LoRA, full 3M, 3 folds × 3 epochs (with patched fold2 epoch3)

## Script
`code/models/compare_oof.py`

## Steps on HPC

### 1. Ensure old model checkpoints exist
```bash
# Check which old checkpoint directory exists
ls -la /hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating/artifacts/models/checkpoints_v3base_1m/
ls -la /hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating/artifacts/models/checkpoints_lora/
```

The script auto-detects: tries `checkpoints_v3base_1m` first, then `checkpoints_lora`.

### 2. Ensure new model checkpoints include patched fold2 epoch3
```bash
ls -la /hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating/artifacts/models/checkpoints_base_full/
# Should show: fold1_epoch3.pt, fold2_epoch3.pt, fold3_epoch3.pt (and earlier epochs)
```

### 3. Run comparison
```bash
cd /hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating
conda activate SHPC-env

python code/models/compare_oof.py
```

Or with explicit checkpoint directories:
```bash
python code/models/compare_oof.py \
  --old-ckpt-dir checkpoints_v3base_1m \
  --new-ckpt-dir checkpoints_base_full
```

### 4. Expected output
- Per-fold OOF RMSE for both models
- Overall OOF RMSE comparison
- Test prediction correlation (Pearson r)
- Variance expansion comparison
- Prediction difference analysis
- Saved `.npy` files in `artifacts/models/compare_*.npy`

## Key Questions This Answers
1. Old model OOF RMSE vs new 3×3 OOF RMSE (on same full 3M data)
2. Per-fold validation OOF comparison
3. Test prediction correlation between old and new
4. Why does new model score worse on Kaggle despite similar/better OOF?

## Notes
- Both models use identical architecture (v3-base, LoRA r=16/alpha=32)
- OOF is computed on full 3M training data for fair comparison
- The old model was trained on 1M subset, so its OOF on the other 2M is out-of-distribution
- GPU required (~8GB VRAM per model load)

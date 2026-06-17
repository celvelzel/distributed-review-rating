# 2026-06-17 Training Status Report

## Base Model (DeBERTa-v3-base) — FULLY COMPLETE ✅

All 9 fold/epoch combinations done. Best val_rmse across folds:

| Fold | Epoch 1 | Epoch 2 | Epoch 3 |
|------|---------|---------|---------|
| 1 | 1.14668 | 1.13945 | 1.13857 |
| 2 | 1.14635 | 1.13915 | **1.13551** ✅ |
| 3 | 1.14563 | 1.13876 | 1.13693 |

- **Fold 2 epoch 3 completed at 12:23** via GPU swap (paused v3-large, ran fold2 e3, restarted v3-large)
- Fold 2 epoch 3 val_rmse=1.13551 is the best across all folds
- Total base training: ~12h (including 3h for fold2 e3 GPU swap)
- All checkpoints saved: fold1_e1-e3, fold2_e1-e3, fold3_e1-e3

## Large Model (DeBERTa-v3-large) — IN PROGRESS 🔄

- **PID:** 1380155 (nohup, background)
- **Resumed from:** fold1_epoch1 (val_rmse=1.17868)
- **Current:** Fold 1 Epoch 2 training
- **Config:** 435M params, LoRA r=16 α=32, BS=16, GradAcc=16, FP16
- **GPU:** 10.5GB / 12.6GB (83%), 86-97% utilization
- **Speed:** ~1.1 steps/s (BS=16), ~8.2h per epoch
- **Checkpoints:** fold1_epoch1.pt (887MB) saved
- **ETA:** ~2.9 days for 3f×3e (8 epochs remaining)

## Resource Usage

| Resource | Base Training | Large Training |
|----------|--------------|----------------|
| GPU VRAM | 7.7GB (61%) | 10.5GB (83%) |
| System RAM | 10.9GB | 10.4GB |
| Speed | ~100 steps/s (BS=32) | ~1.1 steps/s (BS=16) |
| Time/epoch | ~1.7h | ~8.2h |
| Total time | ~9.2h | ~74h |

## Key Decision: GPU Swap Strategy

v3-large uses 10.5GB VRAM, leaving only 2.1GB — not enough for base (7.7GB). Strategy:
1. Wait for v3-large first checkpoint (~10h)
2. Kill v3-large, run fold2 e3 on GPU (~3h)
3. Restart v3-large from checkpoint (zero progress lost)
4. Total delay to v3-large: ~13h wall-clock, zero training lost

## Next Steps

1. Monitor v3-large training — hourly status checks
2. After v3-large completes: generate predictions, VE + Ridge blend, submit
3. Compare base vs large predictions on Kaggle
4. If large is better: ensemble base + large with OOF alignment

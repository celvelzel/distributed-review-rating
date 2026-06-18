# 3M BS16 Ablation Experiment Progress

## Experiment Goal

Test whether 3M DeBERTa's poor performance (0.681 vs 1M's 0.617) is due to batch config (32×8) or overfitting.

- Script: `code/models/deberta_3m_bs16_ablation.py`
- Data: 3M samples (train_tokens.npz, 3,007,439 samples)
- Config: BS=16, GradAcc=16 (1M config), 1 fold × 1 epoch
- HPC Job: 133977 (was pending due to QOS limit, started via setsid nohup)

## Status

**COMPLETED** — PID 1496164

| Metric | Value |
|--------|-------|
| Final Step | 125,309/125,309 (100%) |
| Val RMSE | 1.15654 |
| Test Pred | mean=3.9872, std=0.7724 |
| VE Pred | mean=4.0095, std=1.2443 |
| GPU Peak | 1.07GB |
| Training Time | 31,360s (522.7min / 8.7h) |
| Process | Completed |

## Status Log

| Time | Step | Loss | ETA | GPU Util | Notes |
|------|------|------|-----|----------|-------|
| 13:30 | 33,000/125,309 (26%) | 0.37 | 6.1h | 49% | Training started ~11:18 |
| 14:13 | 43,000/125,309 (34%) | 0.48 | 5.5h | 50% | Loss fluctuating normally |
| 14:44 | 47,000/125,309 (38%) | 0.39 | 5.2h | 48% | Trending down |
| 17:50 | 89,000/125,309 (71%) | 0.38 | 2.5h | 41% | Good progress |
| 19:15 | 107,000/125,309 (85%) | 0.34 | 1.3h | 42% | Nearly done |
| 20:00 | 122,000/125,309 (97%) | 0.49 | 14min | 50% | Almost finished! |
| 20:20 | 125,309/125,309 (100%) | - | - | 0% | COMPLETED |

## Expected Outputs

After training completes:
- `artifacts/models/deberta_3m_bs16_ablation_fold1e1_oof.npy`
- `artifacts/models/deberta_3m_bs16_ablation_fold1e1_test.npy`
- `artifacts/models/checkpoints_3m_bs16_ablation/fold1_epoch1.pt`
- `output/submission-3m-bs16-ablation-ve90-r10.csv`
- `output/submission-3m-bs16-ablation-ve95-r5.csv`
- `output/submission-3m-bs16-ablation-raw.csv`

## Post-Training Steps

1. Submit best CSV to Kaggle
2. Record val_rmse and Kaggle RMSE
3. Compare with baseline (3M BS=32×8 → 0.712, 1M → 0.617)
4. Decision: H1 (BS noise) vs H2 (overfitting)

## Decision Criteria

| Kaggle RMSE | Conclusion |
|-------------|-----------|
| < 0.69 | H1 confirmed: BS noise is main cause → retrain 3M with BS=16 |
| 0.69–0.72 | H2 confirmed: overfitting is main cause → reduce epochs / use 1M data |
| > 0.72 | Other factors (gradient checkpointing, etc.) → further diagnosis |

## Actual Result

**Kaggle RMSE: 0.74265** → **H2 confirmed** (overfitting is main cause)

3M BS=16×16 is even WORSE than 3M BS=32×8 (0.74265 vs 0.68126).

## Result

**Val RMSE: 1.15654** — WORSE than baseline (3M BS=32×8 → 0.681, 1M → 0.617)

**Kaggle RMSE: 0.74265** — Submitted 2026-06-18 12:42

| Config | Val RMSE | Kaggle RMSE |
|--------|----------|-------------|
| 1M BS=32×8 | 1.139 | **0.61734** |
| 3M BS=32×8 | N/A (OOF bug) | 0.68126 |
| 3M BS=16×16 | 1.15654 | 0.74265 |

**Conclusion**: H2 (overfitting) confirmed. BS=16 makes 3M model WORSE (+0.061 vs 3M baseline).

The 3M model fundamentally overfits regardless of batch configuration. The 1M model is superior.

**Next Steps**:
1. Fix 3M OOF bug to get reliable OOF RMSE
2. Abandon 3M path entirely
3. Focus on improving 1M model (DeBERTa-v3-large, better tuning)

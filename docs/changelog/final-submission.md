# Final Submission — Diverse Ensemble (2026-06-08)

## Summary

Final Kaggle submission using a diverse ensemble of three models with optimized weights.

**Ensemble composition**: LightGBM + XGBoost + MLP (neural network)
**Optimal weights**: LGB=0.09, XGB=0.05, MLP=0.86
**OOF RMSE**: 1.12938

---

## Models & Weights

| Model | Weight | OOF RMSE | Notes |
|-------|--------|----------|-------|
| LightGBM | 0.09 | ~1.18 | TF-IDF features, regularized |
| XGBoost | 0.05 | ~1.18 | TF-IDF features |
| MLP | 0.86 | ~1.15 | 896→512→128→1, weak signal features |

**Why MLP dominates**: The MLP captures nonlinear patterns that gradient boosting misses, despite similar OOF RMSE. Weight optimization via `scipy.optimize.minimize` found MLP=0.86 produces the lowest ensemble RMSE.

---

## Submission Details

| Field | Value |
|-------|-------|
| File | `output/submission-final.csv` |
| Rows | 10,000 |
| Columns | `id`, `rating` |
| Prediction range | [1.6940, 4.5451] |
| Mean prediction | 3.9957 |
| Std prediction | 0.7587 |
| Clipped to [1,5] | Yes (0 samples needed clipping) |

---

## Kaggle Submission Status

**⚠️ API Token Expired**: The Kaggle API token returns 401 Unauthorized. Submission must be done manually.

### Manual submission steps:
1. Go to: https://www.kaggle.com/competitions/comp-5434-2526-sem-3-project/submit
2. Upload: `output/submission-final.csv`
3. Message: `Final ensemble (LGB=0.09, XGB=0.05, MLP=0.86)`

---

## Reference: Kaggle Score History

| Rank | File | Score | Date | Notes |
|------|------|-------|------|-------|
| 1 | submission-tfidf-regularized.csv | **0.79012** | 2026-06-06 | Previous best |
| 2 | submission-blend_80_20.csv | 0.79142 | 2026-06-07 | Blend 80% best + 20% stage0 |
| ... | submission-final.csv | **PENDING** | 2026-06-08 | Diverse ensemble (this submission) |

---

## Reproduction

```bash
# Generate submission CSV
module load Anaconda3/2023.03-1
python code/models/final_submission.py

# Submit to Kaggle (requires valid API token)
kaggle competitions submit \
  -c comp-5434-2526-sem-3-project \
  -f output/submission-final.csv \
  -m "Final ensemble (LGB=0.09, XGB=0.05, MLP=0.86)"
```

---

## Files

| File | Purpose |
|------|---------|
| `code/models/final_submission.py` | Submission generation script |
| `artifacts/models/ensemble_diverse_test.npy` | Ensemble test predictions |
| `output/submission-final.csv` | Final submission CSV |
| `output/submission-ensemble-diverse.csv` | Duplicate (same predictions) |

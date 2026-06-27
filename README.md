# COMP5434 Big Data Computing — Distributed Review Rating Prediction

Predict e-commerce review ratings (1–5 stars) using distributed data processing and machine learning.

## Final Result

| Metric | Value |
|--------|-------|
| **Best Kaggle RMSE** | **0.59770** |
| **Baseline RMSE** | 0.69931 |
| **Improvement** | 14.5% |
| **Leaderboard Rank** | 3rd (target: 2nd at 0.47361) |

## Final Technical Solution

### Model Combination

- **DeBERTa-v3-base** (1M subsample, 5f×5e, LoRA r=16) — Transformer-based language model
- **Stacking V3 ridge+lgb** (9 base models) — Meta-learner ensemble

### Best Blending Ratio

```
final_pred = 0.60 × VE(deberta_base) + 0.40 × Stacking_V3_ridge_lgb
```

### Variance Expansion (VE)

DeBERTa predictions are severely compressed (std=0.825 vs target std=1.422). VE restores correct prediction scale:

```python
scale = target_std / pred_std = 1.422 / 0.825 = 1.72
pred_calibrated = (pred - pred_mean) × scale + target_mean
```

### Key Findings

1. **VE ratio optimization**: 60% is optimal (not 90% as initially assumed)
2. **Stacking V3 is more important than VE**: Low VE ratio (50-65%) works better
3. **Training config matters more than data size**: 1M + 5f×5e >> 1M + 3f×3e
4. **Gradient checkpointing is incompatible with LoRA**: Causes LoRA B weights to be zero

## Repository Structure

```
├── .opencode/              # Project-local skills (kaggle-optimizer, kaggle-submission)
├── code/                  # All source code
│   ├── requirements.txt   # Python dependencies
│   ├── config.py          # Centralized configuration
│   ├── etl/               # Data extraction & cleaning
│   ├── features/          # Feature engineering
│   ├── models/            # Model training & prediction
│   ├── ensemble/          # Ensemble and blending scripts
│   ├── utils/             # Shared utilities
│   └── tests/             # Test suite
├── config/                # Kaggle tokens & env config (not in git)
├── data/                  # Dataset files (not in git)
├── docs/                  # Documentation
│   ├── changelog/         # Progress logs and experiment reports
│   ├── progress/          # Training progress tracking
│   ├── designs/           # Architecture and design documents
│   └── specification/     # Project requirements
├── output/                # Kaggle submissions
├── artifacts/             # Model checkpoints and predictions (not in git)
├── scripts/               # Utility scripts
└── tech_dashboard.html    # Real-time project status dashboard
```

## Quick Start

```bash
# Install dependencies
pip install -r code/requirements.txt

# Run individual stages
python -m code.etl.main          # ETL pipeline
python -m code.features.main     # Feature engineering
python -m code.models.train      # Model training
```

## Dataset

| File | Description |
|------|-------------|
| **train.csv** | ~3M training reviews with ratings (1–5 stars) |
| **test.csv** | 10K test reviews (rating to predict) |
| **prodInfo.csv** | 213K product metadata |
| **sample_submission.csv** | Submission format example |

Place data files in `data/` (excluded from git).

## Key Scripts

| Script | Description |
|--------|-------------|
| `code/models/deberta_lora_1m.py` | DeBERTa-v3-base LoRA training (1M, 5f×5e) |
| `code/models/deberta_large_full.py` | DeBERTa-v3-large LoRA training |
| `code/ensemble/stacking_v3.py` | Stacking V3 meta-learner |
| `code/ensemble/generate_large_submissions.py` | Generate Large model submissions |

## Documentation

| Document | Description |
|----------|-------------|
| [Code README](code/README.md) | Detailed usage guide |
| [Technical Dashboard](tech_dashboard.html) | Real-time project status |
| [VE Ratio Optimization](docs/progress/2026-06-20-ve-stacking-ratio-optimization.md) | VE optimization results |
| [Current Best Model](docs/progress/2026-06-21-current-best-model-analysis.md) | Best model analysis |
| [Training Status](docs/changelog/2026-06-27-training-status.md) | Latest training progress |

## Course Information

**Course**: COMP5434 Big Data Computing, PolyU 2026  
**Kaggle Competition**: [COMP5434 Project](https://www.kaggle.com/t/9e897d08dba249bb8a1312666e8ef8fd)

## License

Academic project — not for commercial use.

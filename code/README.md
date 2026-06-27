# COMP5434 Review Rating Prediction — Code Documentation

Distributed review rating prediction system using PySpark, gradient boosting, and transformer-based embeddings.

## Final Technical Solution

### Best Model: DeBERTa-v3-base + Stacking V3 Ensemble

| Component | Configuration | Kaggle RMSE |
|-----------|---------------|-------------|
| DeBERTa-v3-base | 1M subsample, 5f×5e, LoRA r=16 | 0.638 (VE only) |
| Stacking V3 ridge+lgb | 9 base models | — |
| **Final Blend** | **VE 60% + Stacking V3 rlg 40%** | **0.59770** |

### Model Architecture

```
DeBERTa-v3-base (86M params)
├── LoRA adaptation (r=16, alpha=32)
│   ├── query_proj
│   └── value_proj
├── Mean pooling
└── Classifier (1024 → 4 logits for CORAL loss)
```

### Training Configuration

```python
# DeBERTa-v3-base (1M)
MODEL_NAME = "microsoft/deberta-v3-base"
LORA_R, LORA_ALPHA = 16, 32
LORA_TARGET = ["query_proj", "value_proj"]
N_FOLDS, N_EPOCHS = 5, 5
BATCH_SIZE, GRAD_ACCUM = 32, 8
LR = 1e-5
```

### Variance Expansion (VE)

```python
def variance_expansion(pred, target_std=1.422, target_mean=3.941):
    """Expand compressed DeBERTa predictions to match target distribution"""
    ve = (pred - pred.mean()) / pred.std() * target_std + target_mean
    return np.clip(ve, 1.0, 5.0)
```

### Final Blending

```python
# Load predictions
deberta_pred = np.load("artifacts/models/deberta_lora_fold1_test.npy")
stacking_rlg = np.load("artifacts/models/stacking_v3_ridge+lgb_test.npy")

# Apply VE to DeBERTa predictions
ve_pred = variance_expansion(deberta_pred)

# Blend: VE 60% + Stacking V3 ridge+lgb 40%
final_pred = 0.60 * ve_pred + 0.40 * stacking_rlg
```

## Project Structure

```
code/
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── config.py           # Configuration constants
├── etl/                # Data extraction & cleaning
│   ├── main.py         # ETL entry point
│   └── ...
├── features/           # Feature engineering pipelines
│   ├── main.py         # Feature engineering entry point
│   └── ...
├── models/             # Model training & prediction
│   ├── deberta_lora_1m.py      # DeBERTa-v3-base training (1M, 5f×5e)
│   ├── deberta_large_full.py   # DeBERTa-v3-large training
│   ├── stacking_v3.py          # Stacking V3 meta-learner
│   └── ...
├── ensemble/           # Ensemble and blending scripts
│   ├── stacking_v3.py          # Stacking V3 implementation
│   └── generate_large_submissions.py
├── ablation/           # Ablation experiments
├── utils/              # Shared utilities
├── tests/              # Test suite
└── kaggle/             # Kaggle submission generation
```

## Key Scripts

### Model Training

| Script | Description | Usage |
|--------|-------------|-------|
| `deberta_lora_1m.py` | DeBERTa-v3-base LoRA (1M, 5f×5e) | `python code/models/deberta_lora_1m.py` |
| `deberta_large_full.py` | DeBERTa-v3-large LoRA (3M, 3f×3e) | `python code/models/deberta_large_full.py` |
| `stacking_v3.py` | Stacking V3 meta-learner | `python code/ensemble/stacking_v3.py` |

### Prediction Generation

| Script | Description | Usage |
|--------|-------------|-------|
| `generate_large_submissions.py` | Generate Large model submissions | `python code/ensemble/generate_large_submissions.py` |

## Dependencies

Install all dependencies:

```bash
pip install -r requirements.txt
```

Key packages:
- `pyspark==3.4.1` — Distributed data processing
- `torch>=2.0` — Deep learning framework
- `transformers>=4.30` — Pre-trained language models
- `peft>=0.5` — LoRA adaptation
- `lightgbm>=4.0` — Gradient boosting
- `xgboost>=1.7` — Gradient boosting
- `catboost>=1.2` — Gradient boosting
- `pandas>=2.0` — Data manipulation
- `scikit-learn>=1.3` — ML utilities

## Data Setup

1. Download the dataset from the Kaggle competition page.
2. Place files in `data/` directory:
   ```
   data/
   ├── train.csv          # ~3M training reviews with ratings
   ├── test.csv           # 10K test reviews (no ratings)
   ├── prodInfo.csv       # 213K product metadata
   └── sample_submission.csv  # Submission format template
   ```

**Important**: Do NOT commit data files to git. They are excluded via `.gitignore`.

## How to Run

### Full Pipeline

```bash
# 1. Install dependencies
pip install -r code/requirements.txt

# 2. Run the full pipeline
bash code/run.sh all

# 3. Or run individual stages:
bash code/run.sh etl          # Data extraction, transformation, loading
bash code/run.sh features     # Feature engineering
bash code/run.sh train        # Model training
bash code/run.sh predict      # Generate predictions
bash code/run.sh submit       # Create Kaggle submission CSV
```

### DeBERTa Training

```bash
# Train DeBERTa-v3-base (1M, 5f×5e)
python code/models/deberta_lora_1m.py

# Train DeBERTa-v3-large (3M, 3f×3e)
python code/models/deberta_large_full.py
```

### Stacking Ensemble

```bash
# Train Stacking V3 meta-learner
python code/ensemble/stacking_v3.py
```

## Known Issues

### Gradient Checkpointing vs LoRA

`gradient_checkpointing_enable()` is incompatible with LoRA when input tensors are integers (no `requires_grad`). This causes LoRA B weights to remain zero.

**Solution**: Do NOT use gradient checkpointing with LoRA.

### Batch Size Limitations

- DeBERTa-v3-large without gradient checkpointing: max batch_size=8 (12GB GPU)
- DeBERTa-v3-base: batch_size=32 works fine

## Experiments

### VE Ratio Optimization

| VE% | V3 rlg% | Kaggle RMSE |
|-----|---------|-------------|
| 90% | 10% | 0.61725 |
| 85% | 15% | 0.61115 |
| 50% | 50% | 0.60073 |
| 55% | 45% | 0.59862 |
| **60%** | **40%** | **0.59770** |
| 30% | 70% | 0.62090 |

### Ablation Studies

| Experiment | Kaggle RMSE | vs Baseline |
|------------|-------------|-------------|
| Baseline (V2 Ridge) | 0.61734 | — |
| + Graph features | 0.61746 | +0.00012 |
| + Ridge+LGB meta-learner | 0.61725 | -0.00009 |
| 3M BS=16×16 | 0.74265 | +0.12531 |
| 1M + 3f×3e | 1.53602 | +0.91868 |

## Team

- **Course**: COMP5434 Big Data Computing, PolyU 2026
- **Kaggle Competition**: [COMP5434 Project](https://www.kaggle.com/t/9e897d08dba249bb8a1312666e8ef8fd)

## License

Academic project — not for commercial use.

# COMP5434 Review Rating Prediction — Code Documentation

Distributed review rating prediction system using PySpark, gradient boosting, and transformer-based embeddings.

## Final Technical Solution

### Best Model: DeBERTa-v3-base + Stacking V3 Ensemble

| Component | Configuration | Kaggle RMSE |
|-----------|---------------|-------------|
| DeBERTa-v3-base | 1M subsample, 5f x 5e, LoRA r=16 | 0.638 (VE only) |
| Stacking V3 ridge+lgb | 9 base models | — |
| Final Blend | VE 60% + Stacking V3 rlg 40% | 0.59770 |

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
stacking_rlg = np.load("artifacts/models/stacking_v3_test.npy")

# Apply VE to DeBERTa predictions
ve_pred = variance_expansion(deberta_pred)

# Blend: VE 60% + Stacking V3 ridge+lgb 40%
final_pred = 0.60 * ve_pred + 0.40 * stacking_rlg
```

## Project Structure

```
code/
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── config.py               # Configuration constants
├── etl/                    # Data extraction & cleaning
│   ├── run_etl.py          # ETL entry point
│   ├── spark_etl.py        # PySpark ETL logic
│   └── eda.py              # Exploratory data analysis
├── features/               # Feature engineering pipelines
│   ├── run_stats.py        # Statistical features (user/product/category)
│   ├── sentiment.py        # VADER + TextBlob sentiment
│   ├── product_metadata.py # Product metadata features
│   ├── assemble_kfold.py   # Assemble leak-free K-fold features
│   └── ...
├── models/                 # Model training & prediction (15 scripts)
│   ├── deberta_lora_1m.py      # DeBERTa-v3-base LoRA training (1M, 5f x 5e)
│   ├── predict_lora_fold1.py   # DeBERTa fold-1 prediction (OOF + test)
│   ├── deberta_base_full.py    # DeBERTa-v3-base full training
│   ├── deberta_large_full.py   # DeBERTa-v3-large LoRA training
│   ├── xgboost_train.py        # XGBoost base model (TF-IDF)
│   ├── run_mlp.py              # MLP base model (BERT embeddings)
│   ├── train_safe_features.py  # LGB/XGB/CatBoost on safe features
│   ├── ensemble_diverse.py     # Diverse weighted ensemble
│   ├── train_graph_models.py   # Graph-feature base models
│   ├── stacking_v3.py          # Stacking V3 meta-learner
│   ├── submit_stacking_v3.py   # Final submission generator
│   ├── mlp.py                  # MLP architecture definition
│   ├── predict.py              # General prediction utility
│   ├── tfidf_baseline.py       # TF-IDF feature extraction
│   └── __init__.py             # Package initializer
├── ensemble/               # Ensemble and blending utilities
│   └── generate_large_submissions.py
├── utils/                  # Shared utilities
│   ├── spark_session.py    # Spark session factory
│   └── timer.py            # Stage timing utilities
└── tests/                  # Test suite
```

## Model List (15 scripts in code/models/)

| Script | Description | Type |
|--------|-------------|------|
| `deberta_lora_1m.py` | DeBERTa-v3-base LoRA training (1M subsample, 5-fold x 5-epoch) | Transformer training |
| `deberta_base_full.py` | DeBERTa-v3-base full-data training (HPC path) | Transformer training |
| `deberta_large_full.py` | DeBERTa-v3-large LoRA training (3M data) (HPC path) | Transformer training |
| `predict_lora_fold1.py` | DeBERTa fold-1 prediction (test + OOF) | Transformer prediction |
| `predict.py` | General-purpose prediction utility | Transformer prediction |
| `xgboost_train.py` | XGBoost base model on TF-IDF features | Base model |
| `run_mlp.py` | MLP base model on BERT embeddings (768-dim) | Base model |
| `mlp.py` | MLP architecture definition (768→512→256→128→1) | Architecture module |
| `train_safe_features.py` | LightGBM, XGBoost, CatBoost on safe (leak-free) features | Base model |
| `ensemble_diverse.py` | Diverse weighted ensemble (LGB + XGB + MLP) | Base model |
| `train_graph_models.py` | Graph-feature base models (XGBoost + LightGBM) | Base model |
| `tfidf_baseline.py` | TF-IDF feature extraction utility (shared by multiple models) | Feature utility |
| `stacking_v3.py` | Stacking V3 meta-learner (9 base models, 5 meta-learner candidates) | Meta-learner |
| `submit_stacking_v3.py` | Final submission generator (VE + stacking blend) | Submission |
| `__init__.py` | Package initializer | Package |

## Pipeline Execution Order

Run each command from the project root directory. Commands must be executed in order — each step depends on artifacts produced by the previous step.

### Step 1: Install Dependencies

```bash
pip install -r code/requirements.txt
```

### Step 2: Data Setup

Place `train.csv`, `test.csv`, `prodInfo.csv`, and `sample_submission.csv` in `data/`.

### Step 3: ETL Pipeline

```bash
python code/etl/run_etl.py
```

Outputs: `artifacts/etl/train.parquet`, `artifacts/etl/test.parquet`, `artifacts/etl/prodinfo.parquet`

### Step 4: Feature Engineering

```bash
# Statistical features (PySpark)
python code/features/run_stats.py

# Sentiment features
python code/features/sentiment.py

# Product metadata features
python code/features/product_metadata.py

# Assemble K-fold features and generate y_train.npy
python code/features/assemble_kfold.py
```

Outputs: `artifacts/features/user_stats.parquet`, `product_stats.parquet`, `category_stats.parquet`, `sentiment.parquet`, `product_metadata.parquet`, `X_train_kfold.parquet`, `X_test_kfold.parquet`, `y_train.npy`

### Step 5: DeBERTa Training

```bash
# DeBERTa-v3-base LoRA (1M subsample, 5-fold x 5-epoch) — best model
python code/models/deberta_lora_1m.py

# Optional: DeBERTa-v3-large LoRA (3M data, 3-fold x 3-epoch)
python code/models/deberta_large_full.py
```

Outputs: model checkpoints in `artifacts/models/checkpoints_v3base_1m/`

### Step 6: DeBERTa Prediction

```bash
python code/models/predict_lora_fold1.py
```

Outputs: `artifacts/models/deberta_lora_fold1_test.npy`, `artifacts/models/deberta_lora_fold1_oof.npy`

### Step 7: Base Model Training

```bash
# XGBoost on TF-IDF features
python code/models/xgboost_train.py

# MLP on BERT embeddings
python code/models/run_mlp.py

# LightGBM / XGBoost / CatBoost on safe features
python code/models/train_safe_features.py

# Diverse weighted ensemble (LGB + XGB + MLP)
python code/models/ensemble_diverse.py

# Graph-feature base models
python code/models/train_graph_models.py
```

Outputs (per model): `artifacts/models/{model_name}_oof.npy`, `artifacts/models/{model_name}_test.npy`

### Step 8: Stacking Ensemble

```bash
python code/models/stacking_v3.py
```

Outputs: `artifacts/models/stacking_v3_oof.npy`, `artifacts/models/stacking_v3_test.npy`, `artifacts/models/stacking_v3_results.json`

### Step 9: Final Submission

```bash
python code/models/submit_stacking_v3.py
```

Outputs: `output/submission-deb1m-ve60-sv3-40.csv` (best, Kaggle RMSE 0.59770) and additional diagnostic submissions at various blend ratios.

## HPC Path Notes

The following scripts contain hardcoded HPC paths that must be adjusted when running on a different machine:

| Script | Hardcoded Path | Variable |
|--------|---------------|----------|
| `deberta_base_full.py` | `/hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating` | `ROOT` (line 15) |
| `deberta_large_full.py` | `/hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating` | `ROOT` (line 15) |

All other scripts use relative paths derived from `__file__` (e.g., `Path(__file__).resolve().parents[2]`) and do not require modification.

To run `deberta_base_full.py` or `deberta_large_full.py` locally, update the `ROOT` variable at the top of each file to point to your local project root:

```python
# Before (HPC):
ROOT = "/hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating"

# After (local):
ROOT = "/path/to/your/distributed-review-rating"
```

Additionally, some feature scripts (e.g., `assemble_kfold.py`, `sentiment.py`) use relative paths like `"artifacts/features"` that assume the working directory is the project root. Always run commands from the project root directory.

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
- `mlflow>=2.5` — Experiment tracking
- `optuna>=3.3` — Hyperparameter optimization
- `gensim>=4.3` — Word embeddings
- `pyarrow>=12.0` — Parquet I/O
- `sentence-transformers>=2.2` — Sentence embeddings

## Data Setup

1. Download the dataset from the [Kaggle competition page](https://www.kaggle.com/t/9e897d08dba249bb8a1312666e8ef8fd).
2. Place files in `data/` directory:

```
data/
├── train.csv               # ~3M training reviews with ratings
├── test.csv                # 10K test reviews (no ratings)
├── prodInfo.csv            # 213K product metadata
└── sample_submission.csv   # Submission format template
```

Important: Do NOT commit data files to git. They are excluded via `.gitignore`.

## Known Issues

### Gradient Checkpointing vs LoRA

`gradient_checkpointing_enable()` is incompatible with LoRA when input tensors are integers (no `requires_grad`). This causes LoRA B weights to remain zero.

Solution: Do NOT use gradient checkpointing with LoRA.

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
| 60% | 40% | 0.59770 (best) |
| 30% | 70% | 0.62090 |

### Ablation Studies

| Experiment | Kaggle RMSE | vs Baseline |
|------------|-------------|-------------|
| Baseline (V2 Ridge) | 0.61734 | — |
| + Graph features | 0.61746 | +0.00012 |
| + Ridge+LGB meta-learner | 0.61725 | -0.00009 |
| 3M BS=16x16 | 0.74265 | +0.12531 |
| 1M + 3f x 3e | 1.53602 | +0.91868 |

## Team

- Course: COMP 5434 Big Data Computing, PolyU
- Team: Team 9
- Members: MA Jiyuan (25116696G), Yee Lok CHIU (25012923G), ZHANG Boxin (25054717G), YANG Ziting (25050315G)
- Kaggle Competition: [COMP5434 Project](https://www.kaggle.com/t/9e897d08dba249bb8a1312666e8ef8fd)

## License

Academic project — not for commercial use.

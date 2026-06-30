# COMP5434 Big Data Computing — Distributed Review Rating Prediction

Predict e-commerce review ratings (1–5 stars) from review text and product metadata using distributed data processing (Apache Spark) and machine learning (DeBERTa + gradient boosting + stacking ensemble). Evaluated by RMSE on a held-out test set.

## Team Information

| | |
|---|---|
| Team | Team 9 |
| Course | COMP 5434 Big Data Computing |
| Instructor | Prof. Jieming SHI |
| Members | MA Jiyuan (25116696G), Yee Lok CHIU (25012923G), ZHANG Boxin (25054717G), YANG Ziting (25050315G) |
| Kaggle Competition | [COMP5434 Project](https://www.kaggle.com/t/9e897d08dba249bb8a1312666e8ef8fd) |

## Best Kaggle Score

| Metric | Value |
|--------|-------|
| Best Kaggle RMSE | 0.59770 |
| (with keyword pair) | 0.58296 |
| Baseline RMSE | 0.69931 |
| Improvement | 14.5% |

## Final Solution Architecture

```
DeBERTa-v3-base (1M subsample, 5-fold x 5-epoch, LoRA r=16)
    |
    | Variance Expansion (VE)
    v
VE predictions (60% weight)
    +
Stacking V3 (9 base models, Ridge + LGB meta-learner) (40% weight)
    |
    v
final_pred = 0.60 * VE(deberta_base) + 0.40 * Stacking_V3_ridge_lgb
```

### Key Techniques

- LoRA (Low-Rank Adaptation) on DeBERTa-v3-base query/value projections (trainable params reduced from ~140M to ~1.2M)
- CORAL (Consistent Rank Logits) ordinal regression loss
- R-Drop regularisation (double forward pass + KL divergence penalty)
- Variance Expansion (VE): rescales DeBERTa's under-dispersed predictions to match the target distribution
- Stacking V3: 9 diverse base models (LGB, XGBoost, MLP, CatBoost, ensemble, graph models) with Ridge + LightGBM meta-learner

## Hardware Specifications and Efficiency Metrics

### Hardware

| Resource | Specification |
|----------|---------------|
| CPU | POLYU HPC 6-Cores |
| GPU | NVIDIA RTX 3080 Ti (12GB VRAM) |
| Memory | 32GB |
| Framework | Apache Spark 3.4.1 |
| Mode | Single-machine pseudo-distributed mode |

### Efficiency Metrics

| Task | Time |
|------|------|
| ETL preprocessing (tokenization for 3M samples) | ~16 minutes |
| DeBERTa-v3-base LoRA training (1M data, best model) | ~4.8–5.5 hours/epoch |
| DeBERTa-v3-large LoRA training (3M data) | ~6 hours/epoch |
| Total offline time (DeBERTa base) | ~120 hours |
| Total offline time (DeBERTa large) | ~54 hours |
| Total offline time (stacking ensemble) | ~9.2 hours |
| Inference time (DeBERTa large, OOF + test) | ~30 minutes |
| Inference time (tree-based models) | < 1 second |

## Repository Structure

```
distributed-review-rating/
├── code/                          # All source code
│   ├── README.md                  # Detailed code documentation
│   ├── requirements.txt           # Python dependencies
│   ├── config.py                  # Centralized configuration
│   ├── etl/                       # Data extraction & cleaning
│   │   ├── run_etl.py             # ETL entry point
│   │   ├── spark_etl.py           # PySpark ETL logic
│   │   └── eda.py                 # Exploratory data analysis
│   ├── features/                  # Feature engineering pipelines
│   │   ├── run_stats.py           # Statistical features (user/product/category)
│   │   ├── sentiment.py           # VADER + TextBlob sentiment features
│   │   ├── product_metadata.py    # Product metadata features
│   │   ├── assemble_kfold.py      # Assemble leak-free K-fold features
│   │   └── ...                    # Additional feature scripts
│   ├── models/                    # Model training & prediction (15 scripts)
│   │   ├── deberta_lora_1m.py     # DeBERTa-v3-base LoRA training (1M, 5f x 5e)
│   │   ├── predict_lora_fold1.py  # DeBERTa fold-1 prediction (OOF + test)
│   │   ├── deberta_base_full.py   # DeBERTa-v3-base full training
│   │   ├── deberta_large_full.py  # DeBERTa-v3-large LoRA training
│   │   ├── xgboost_train.py       # XGBoost base model (TF-IDF)
│   │   ├── run_mlp.py             # MLP base model (BERT embeddings)
│   │   ├── train_safe_features.py # LGB/XGB/CatBoost on safe features
│   │   ├── ensemble_diverse.py    # Diverse weighted ensemble
│   │   ├── train_graph_models.py  # Graph-feature base models
│   │   ├── stacking_v3.py         # Stacking V3 meta-learner
│   │   ├── submit_stacking_v3.py  # Final submission generator
│   │   ├── mlp.py                 # MLP architecture definition
│   │   ├── predict.py             # General prediction utility
│   │   ├── tfidf_baseline.py      # TF-IDF feature extraction
│   │   └── __init__.py            # Package initializer
│   ├── utils/                     # Shared utilities
│   │   ├── spark_session.py       # Spark session factory
│   │   └── timer.py               # Stage timing utilities
│   └── tests/                     # Test suite
├── config/                        # Kaggle tokens & env config (not in git)
├── data/                          # Dataset files (not in git)
├── docs/                          # Documentation
│   ├── changelog/                 # Progress logs and experiment reports
│   ├── progress/                  # Training progress tracking
│   ├── designs/                   # Architecture and design documents
│   ├── reports/                   # Project reports
│   ├── analysis/                  # Analysis documents
│   └── specification/             # Project requirements
├── output/                        # Kaggle submission CSVs
├── artifacts/                     # Model checkpoints and predictions (not in git)
├── scripts/                       # Utility scripts
└── tech_dashboard.html            # Real-time project status dashboard
```

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd distributed-review-rating

# Install dependencies
pip install -r code/requirements.txt
```

Key dependencies: `pyspark==3.4.1`, `torch>=2.0`, `transformers>=4.30`, `peft>=0.5`, `lightgbm>=4.0`, `xgboost>=1.7`, `catboost>=1.2`, `scikit-learn>=1.3`.

## Step-by-Step Execution Instructions

The following commands reproduce the full pipeline from raw data to final Kaggle submission. Run all commands from the project root directory.

### Step 1: Data Setup

Download the following files from the [Kaggle competition page](https://www.kaggle.com/t/9e897d08dba249bb8a1312666e8ef8fd) and place them in `data/`:

```
data/
├── train.csv               # ~3M training reviews with ratings
├── test.csv                # 10K test reviews (ratings to predict)
├── prodInfo.csv            # 213K product metadata records
└── sample_submission.csv   # Submission format template
```

### Step 2: ETL Pipeline

Clean text, impute missing values, join with product info, extract temporal features, and persist as Parquet.

```bash
python code/etl/run_etl.py
```

Expected output files:
- `artifacts/etl/train.parquet`
- `artifacts/etl/test.parquet`
- `artifacts/etl/prodinfo.parquet`
- `artifacts/etl/metrics.json`

### Step 3: Feature Engineering

Compute statistical, sentiment, product metadata, and K-fold features (all leak-free).

```bash
# Statistical features (user, product, category aggregations)
python code/features/run_stats.py

# Sentiment features (VADER + TextBlob)
python code/features/sentiment.py

# Product metadata features (feature list parsing, store aggregations)
python code/features/product_metadata.py

# Assemble K-fold features and generate y_train.npy
python code/features/assemble_kfold.py
```

Expected output files:
- `artifacts/features/user_stats.parquet`
- `artifacts/features/product_stats.parquet`
- `artifacts/features/category_stats.parquet`
- `artifacts/features/sentiment.parquet`
- `artifacts/features/product_metadata.parquet`
- `artifacts/features/X_train_kfold.parquet`
- `artifacts/features/X_test_kfold.parquet`
- `artifacts/features/y_train.npy`

### Step 4: DeBERTa Training

Train DeBERTa-v3-base with LoRA on a 1M-row subsample (5-fold cross-validation, 5 epochs per fold).

```bash
python code/models/deberta_lora_1m.py
```

Expected output files:
- `artifacts/models/checkpoints_v3base_1m/fold{1-5}_epoch{1-5}.pt` (model checkpoints)

### Step 5: DeBERTa Prediction

Generate test-set and OOF predictions from the best fold-1 checkpoint.

```bash
python code/models/predict_lora_fold1.py
```

Expected output files:
- `artifacts/models/deberta_lora_fold1_test.npy` (test predictions)
- `artifacts/models/deberta_lora_fold1_oof.npy` (OOF predictions for stacking)

### Step 6: Base Model Training

Train the 9 base models that feed into the stacking ensemble. Each script produces OOF and test prediction arrays.

```bash
# XGBoost on TF-IDF features
python code/models/xgboost_train.py

# MLP on BERT embeddings (requires BERT feature parquets in artifacts/features/)
python code/models/run_mlp.py

# LightGBM, XGBoost, CatBoost on safe (leak-free) features
python code/models/train_safe_features.py

# Diverse weighted ensemble (combines LGB + XGB + MLP)
python code/models/ensemble_diverse.py

# Graph-feature base models (requires expanded graph features)
python code/models/train_graph_models.py
```

Expected output files (per model, in `artifacts/models/`):
- `xgboost_oof.npy`, `xgboost_test.npy`
- `mlp_oof.npy`, `mlp_test.npy`
- `lgb_safe_dense_oof.npy`, `lgb_safe_dense_test.npy`
- `xgboost_safe_oof.npy`, `xgboost_safe_test.npy`
- `catboost_safe_oof.npy`, `catboost_safe_test.npy`
- `ensemble_diverse_oof.npy`, `ensemble_diverse_test.npy`
- `xgb_graph_safe_oof.npy`, `xgb_graph_safe_test.npy`
- `lgb_graph_safe_oof.npy`, `lgb_graph_safe_test.npy`

### Step 7: Stacking Ensemble

Run the Stacking V3 meta-learner over all 9 base model predictions. The script auto-selects the best meta-learner (Ridge, LightGBM, CatBoost, ElasticNet, or Ridge+LGB blend) by OOF RMSE.

```bash
python code/models/stacking_v3.py
```

Expected output files:
- `artifacts/models/stacking_v3_oof.npy`
- `artifacts/models/stacking_v3_test.npy`
- `artifacts/models/stacking_v3_results.json`
- `docs/changelog/stacking-v3-results.md`

### Step 8: Final Submission

Blend DeBERTa VE predictions with Stacking V3 predictions at the optimal 60/40 ratio and generate Kaggle submission CSVs.

```bash
python code/models/submit_stacking_v3.py
```

Expected output files (in `output/`):
- `submission-deb1m-ve60-sv3-40.csv` (best blend, Kaggle RMSE 0.59770)
- Additional diagnostic submissions at various blend ratios

## Data Description

### train.csv (~3M rows)

| Field | Description |
|-------|-------------|
| `id` | Unique review identifier |
| `rating` | Target variable (1–5 stars, integer) |
| `title` | Review title (text) |
| `comment` | Review body text |
| `user_id` | Reviewer identifier |
| `parent_prod_id` | Product identifier |
| `helpful_vote` | Number of helpful votes |
| `timestamp` | Review submission timestamp |

### test.csv (~10K rows)

Same schema as `train.csv` except `rating` is absent (to be predicted).

### prodInfo.csv (~213K rows)

| Field | Description |
|-------|-------------|
| `parent_prod_id` | Product identifier (join key) |
| `title` | Product title |
| `price` | Product price |
| `features` | Product feature list (string-encoded) |
| `store` | Store name |
| `rating_number` | Number of ratings on the product |
| `categories` | Product category hierarchy |

### sample_submission.csv

Two-column template (`id`, `rating`) demonstrating the expected submission format.

## Documentation

| Document | Description |
|----------|-------------|
| [Code README](code/README.md) | Detailed code documentation and model list |
| [Technical Dashboard](tech_dashboard.html) | Real-time project status |
| [VE Ratio Optimization](docs/progress/2026-06-20-ve-stacking-ratio-optimization.md) | VE optimization results |
| [Current Best Model](docs/progress/2026-06-21-current-best-model-analysis.md) | Best model analysis |
| [Training Status](docs/changelog/2026-06-27-training-status.md) | Latest training progress |
| [Stacking V3 Ablation](docs/changelog/2026-06-18-stacking-v3-ablation-study.md) | Stacking V3 ablation study |
| [Project Report v1](docs/reports/2026-06-23-project-report-v1.md) | Comprehensive project report |
| [Distributed Computing Analysis](docs/reports/distributed-computing-analysis.md) | Spark usage analysis |

## License

Academic project for COMP 5434 Big Data Computing, PolyU — not for commercial use.

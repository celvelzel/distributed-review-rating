# COMP5434 Review Rating Prediction

Distributed review rating prediction system using PySpark, gradient boosting, and transformer-based embeddings.

## Project Overview

This project predicts review ratings (1–5 stars) for e-commerce product reviews using a hybrid approach:

- **PySpark** for distributed data processing and feature engineering
- **Gradient boosting** (LightGBM, XGBoost, CatBoost) for tabular features
- **Transformer embeddings** (Sentence-BERT) for text features
- **Ensemble methods** to combine models for final predictions

The system processes ~3M training reviews, engineers user/product/text features, and produces Kaggle submissions.

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| GPU | None (CPU training) | 1x NVIDIA (CUDA 11.8+) |
| Disk | 10 GB free | 20+ GB free |
| Spark | Local mode | Local mode (pseudo-distributed) |

**Note**: PySpark runs in local mode with `spark.master=local[*]`. GPU accelerates transformer embedding extraction but is not required.

## Dependencies

Install all dependencies:

```bash
pip install -r requirements.txt
```

Key packages:
- `pyspark==3.4.1` — Distributed data processing
- `torch>=2.0` — Deep learning framework
- `transformers>=4.30` — Pre-trained language models
- `sentence-transformers>=2.2` — Sentence embeddings
- `lightgbm>=4.0` — Gradient boosting
- `xgboost>=1.7` — Gradient boosting
- `catboost>=1.2` — Gradient boosting
- `pandas>=2.0` — Data manipulation
- `scikit-learn>=1.3` — ML utilities
- `mlflow>=2.5` — Experiment tracking
- `optuna>=3.3` — Hyperparameter optimization

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

### Step-by-step reproduction:

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
bash code/run.sh ablation     # Run ablation experiments
```

### Quick help:

```bash
bash code/run.sh --help
```

## Stage-by-stage Guide

### Stage 1: ETL (`code/etl/`)
- Reads raw CSV files into Spark DataFrames
- Handles missing values (title, comment, price, votes)
- Type conversions and data cleaning
- Outputs cleaned Parquet files to `artifacts/etl/`

### Stage 2: Features (`code/features/`)
- **Text features**: TF-IDF, sentence-transformer embeddings
- **User features**: Average rating, review count, rating variance
- **Product features**: Average rating, review count, category stats
- **Time features**: Hour, weekday, month, days since first review
- **Cross features**: User-product interaction statistics
- Outputs assembled feature vectors to `artifacts/features/`

### Stage 3: Training (`code/models/`)
- Trains multiple models: LightGBM, XGBoost, CatBoost
- Hyperparameter tuning via Optuna
- Cross-validation with RMSE metric
- Model selection and ensemble weighting
- Saves models to `artifacts/models/`

### Stage 4: Prediction (`code/models/`)
- Loads trained models
- Generates predictions on test set
- Outputs raw predictions to `artifacts/predictions/`

### Stage 5: Submission (`code/kaggle/`)
- Formats predictions into `submission.csv`
- Validates submission format
- Ready for Kaggle upload

### Stage 6: Ablation (`code/ablation/`)
- Tests contribution of each feature group
- Measures impact of different models
- Produces ablation results table for report

## Ablation Experiments

Run ablation studies to evaluate feature/model contributions:

```bash
bash code/run.sh ablation
```

Experiments include:
- Text features only vs. full feature set
- Individual model performance comparison
- Ensemble vs. single model
- Hyperparameter sensitivity analysis

Results are saved to `artifacts/ablation/` and logged to MLflow.

## Website

Presentation website files are in `code/website/`. Build and view locally:

```bash
# If using a static site generator, build from code/website/
# Otherwise, open index.html directly in browser
```

## Tests

Run the test suite:

```bash
pytest code/tests/ -v
```

Tests cover:
- Data schema validation
- Feature engineering pipeline correctness
- Model prediction range validation
- Submission format compliance

## Project Structure

```
code/
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── run.sh              # Main entry point
├── config.py           # Configuration constants
├── etl/                # Data extraction & cleaning
├── features/           # Feature engineering pipelines
├── models/             # Model training & prediction
├── ablation/           # Ablation experiments
├── utils/              # Shared utilities
├── tests/              # Test suite
├── website/            # Presentation website
└── kaggle/             # Kaggle submission generation
```

## Team

- **Team Name**: [Team Name]
- **Members**: [Member 1], [Member 2], [Member 3]
- **Course**: COMP5434 Big Data Computing, PolyU 2026

## License

Academic project — not for commercial use.

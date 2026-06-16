# COMP5434 Big Data Computing — Distributed Review Rating Prediction

Predict e-commerce review ratings (1–5 stars) using distributed data processing and machine learning.

## Overview

This project builds a prediction system for review ratings using PySpark for distributed processing and gradient boosting models (LightGBM, XGBoost, CatBoost). The system processes ~3M training reviews and generates Kaggle submissions.

**Course**: COMP5434 Big Data Computing, PolyU 2026  
**Kaggle Competition**: [COMP5434 Project](https://www.kaggle.com/t/9e897d08dba249bb8a1312666e8ef8fd)

## Repository Structure

```
├── code/                  # All source code
│   ├── run.sh             # Main entry point
│   ├── requirements.txt   # Python dependencies
│   ├── config.py          # Centralized configuration
│   ├── etl/               # Data extraction & cleaning
│   ├── features/          # Feature engineering
│   ├── models/            # Model training & prediction
│   ├── ablation/          # Ablation experiments
│   ├── utils/             # Shared utilities
│   ├── tests/             # Test suite
│   └── kaggle/            # Kaggle submission
├── data/                  # Dataset files (not in git)
├── docs/                  # Documentation
│   ├── changelog/         # Progress logs and experiment reports
│   ├── designs/           # Architecture and design documents
│   └── specification/     # Project requirements
├── tech_dashboard.html    # Real-time project status dashboard
└── website/               # Presentation website
```

## Quick Start

```bash
# Install dependencies
pip install -r code/requirements.txt

# Run full pipeline
bash code/run.sh all

# See all options
bash code/run.sh --help
```

## Dataset

| File | Description |
|------|-------------|
| **train.csv** | ~3M training reviews with ratings (1–5 stars) |
| **test.csv** | 10K test reviews (rating to predict) |
| **prodInfo.csv** | 213K product metadata |
| **sample_submission.csv** | Submission format example |

Place data files in `data/` (excluded from git).

## Key Features

- **Distributed ETL**: PySpark-based data pipeline with Broadcast Join and Parquet persistence
- **Feature Engineering**: TF-IDF text features with configurable n-gram ranges
- **Model Training**: LightGBM with regularization and cross-validation
- **Kaggle Integration**: Automated submission file generation

## Documentation

| Document | Description |
|----------|-------------|
| [Code README](code/README.md) | Detailed usage guide |
| [Design Document](docs/designs/deepseek_design.md) | System architecture |
| [Specification](docs/specification/spec.txt) | Project requirements |
| [Status Report](docs/changelog/2026-06-12-status-report.md) | Latest progress and metrics |
| [Technical Dashboard](tech_dashboard.html) | Real-time project status |


# COMP5434 Big Data Computing — Distributed Review Rating Prediction

Predict e-commerce review ratings (1–5 stars) using distributed data processing and machine learning.

## Overview

This project builds a prediction system for review ratings using PySpark for distributed processing, gradient boosting models (LightGBM, XGBoost, CatBoost), and transformer-based text embeddings. The system processes ~3M training reviews and generates Kaggle submissions.

## Repository Structure

```
├── code/               # All source code
│   ├── run.sh          # Main entry point
│   ├── requirements.txt
│   ├── etl/            # Data extraction & cleaning
│   ├── features/       # Feature engineering
│   ├── models/         # Model training & prediction
│   ├── ablation/       # Ablation experiments
│   ├── utils/          # Shared utilities
│   ├── tests/          # Test suite
│   ├── website/        # Presentation website
│   └── kaggle/         # Kaggle submission
├── data/               # Dataset files (not in git)
├── docs/               # Design docs and specification
│   ├── designs/        # Architecture and design documents
│   └── specification/  # Project requirements
└── artifacts/          # Generated outputs (not in git)
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

- **train.csv** — 3M training reviews with ratings
- **test.csv** — 10K test reviews
- **prodInfo.csv** — 213K product metadata

Place data files in `data/` (excluded from git).

## Documentation

- [Code README](code/README.md) — Detailed usage guide
- [Design Document](docs/designs/deepseek_design.md) — System architecture
- [Specification](docs/specification/spec.txt) — Project requirements

## Team

**Team Name**: [Team Name]

| Role | Name |
|------|------|
| Member 1 | [Name] |
| Member 2 | [Name] |
| Member 3 | [Name] |

**Course**: COMP5434 Big Data Computing, PolyU 2026

## License

Academic project — not for commercial use.

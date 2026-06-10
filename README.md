# COMP5434 Big Data Computing — Distributed Review Rating Prediction

Predict e-commerce review ratings (1–5 stars) using distributed data processing and machine learning.

## Current Status

| Metric | Value | Notes |
|--------|-------|-------|
| **Best Kaggle Score** | 0.79012 | TF-IDF 5K + 正则化 LightGBM |
| **Competitor Score** | 0.62 | 领先我们 21% |
| **Target** | < 0.70 | 需要根本性方法改进 |

## Overview

This project builds a prediction system for review ratings using PySpark for distributed processing, gradient boosting models (LightGBM, XGBoost, CatBoost), and transformer-based text embeddings. The system processes ~3M training reviews and generates Kaggle submissions.

## Key Findings

1. **TF-IDF 特征泛化最好**: 纯文本特征无目标泄漏, 转移到测试集效果好
2. **统计特征存在目标泄漏**: user_te, prod_te, avg_rating 等特征导致 Kaggle 分数 1.2-1.6
3. **正则化有帮助**: subsample=0.8, colsample=0.8 提升泛化能力
4. **添加额外特征反而降低性能**: temporal, text_length 等特征增加噪声
5. **MLP 性能不佳**: DeBERTa 冻结嵌入 + LightGCN 的 OOF RMSE = 1.152

## Improvement Plan

### Phase 1: TF-IDF Enhancement (Target: 0.77-0.78)
- [ ] Character-level n-grams (char_wb, 2-5)
- [ ] Better text preprocessing (lowercase, remove special chars)
- [ ] Increase TF-IDF dimensions to 20K-50K
- [ ] Word + Character TF-IDF concatenation

### Phase 2: Model Diversity (Target: 0.75-0.76)
- [ ] XGBoost with optimized hyperparameters
- [ ] Ridge regression on TF-IDF
- [ ] Multi-TF-IDF configuration ensemble
- [ ] Stacking with Ridge meta-learner

### Phase 3: Deep Learning (Target: 0.70-0.72)
- [ ] SentenceTransformer embeddings (all-MiniLM-L6-v2)
- [ ] DeBERTa fine-tuning
- [ ] Pseudo-labeling for semi-supervised learning

### Phase 4: Advanced Optimization (Target: <0.70)
- [ ] Learning rate scheduling
- [ ] Data augmentation (synonym replacement)
- [ ] More complex ensemble strategies

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
- [Accuracy Improvement Analysis](docs/changelog/accuracy-improvement-analysis.md) — Detailed analysis and improvement plan
- [Technical Dashboard](tech_dashboard.html) — Real-time project status and metrics

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

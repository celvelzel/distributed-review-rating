# Learnings — Review Rating Iteration

## 2026-06-05 — Project Scaffold Created

### Structure
- Directory layout: `code/{etl,features,models,ablation,utils,tests,website,kaggle}/`
- Each Python package gets `__init__.py`
- Top-level `README.md` links to `code/`, `docs/`, `website/`
- `code/README.md` is comprehensive (145 lines) with all required sections

### Dependencies
- PySpark 3.4.1 pinned exactly, others use `>=` minimum versions
- Full list: pyspark, torch, transformers, lightgbm, catboost, xgboost, pandas, numpy, scikit-learn, mlflow, optuna, pytest, pyarrow, gensim, matplotlib, seaborn, sentence-transformers

### run.sh
- Uses `set -euo pipefail` for safety
- Color-coded output (GREEN, YELLOW, RED)
- Supports `--help`, `--verbose`, `--dry-run` flags
- Environment variables: `SPARK_MASTER`, `DATA_DIR`, `ARTIFACT_DIR`, `N_PARTITIONS`
- All stages stubbed with echo output and TODO comments
- Verified: `bash code/run.sh --help` prints usage and exits 0

### .gitignore
- Covers Python (`__pycache__/`, `*.pyc`), data (`data/*.parquet`), ML artifacts (`artifacts/`, `mlruns/`), Sisyphus internals, IDE files
- Existing `.gitignore` had minimal entries (`/data`, `.sisyphus`, `*.log`) — expanded significantly

### Windows Notes
- PowerShell's `bash` command uses WSL — use `"C:\Program Files\Git\bin\bash.exe"` for Git Bash
- `New-Item -Force` creates directories without error if they exist

---

## 2026-06-05 — Progress Save Point

### Completed
- **T1: Project scaffold** — FULLY COMPLETE
  - All directories created (code/{etl,features,models,ablation,utils,tests,website,kaggle}/)
  - code/README.md (145 lines), requirements.txt (17 packages), .gitignore (14 entries)
  - code/run.sh with --help, --verbose, --dry-run support
  - Top-level README.md links to code/, docs/, website/

### In Progress
- **T2: Data EDA** — PARTIALLY COMPLETE
  - code/etl/eda.py created (617 lines) — comprehensive script with PySpark + matplotlib
  - Script ran but was interrupted before generating outputs
  - Need to re-run: `python code/etl/eda.py`
  - Expected outputs: docs/changelog/eda-report.md, docs/changelog/metrics.json, docs/changelog/figures/eda-*.png

### Key Findings (from EDA run start)
- PySpark works without Hadoop (warning about winutils.exe is non-blocking)
- Spark log level can be set to WARN to reduce noise
- Data loads successfully: 3,007,439 train, 10,000 test, 213,593 prodInfo

### Next Steps (when resuming)
1. Re-run T2: `python code/etl/eda.py` (wait for completion, verify outputs)
2. Dispatch T3 (PySpark config), T4 (timer), T5 (website) in parallel
3. Then T6 (Spark ETL), then T7 (Stage 0 baseline)

### Environment Notes
- Python 3.x with pandas, numpy, matplotlib, seaborn installed
- PySpark 3.4.1 works in local[*] mode
- No Hadoop installed — PySpark runs without it (just warnings)


## EDA Learnings (2026-06-05)

### Dataset Structure
- train: 2,949,249 rows, 10 columns; test: 10,000 rows, 9 columns; prodInfo: 213,593 rows, 8 columns
- rating is integer 1-5, target variable
- time is Unix timestamp in milliseconds

### Class Imbalance
- Rating 5 is dominant class (~56%)
- Ratings 1-2 are minority — need stratified sampling or class weighting

### User/Product Overlap
- User overlap: 99.5% — very cold-start heavy
- Product overlap: 99.9% — product features more generalizable
- 0.5% cold-start users in test

### Text Features
- Median title length: ~18 chars
- Median comment length: ~119 chars
- TF-IDF and BERT embeddings will be key features

### Votes & Purchased
- Median votes: 0, highly skewed (most reviews have 0 votes)
- Purchased is binary, majority True — weak feature alone but could interact

### Price Data
- Only 84,878 / 84,916 products have price info — sparse
- Prices are right-skewed, consider log transform

### Implications for Modeling
1. Need robust handling of cold-start users (content-based fallback)
2. Product-level features (category, price, rating_number) are important
3. Text features (title + comment) are critical given short reviews
4. Temporal features could capture evolving review patterns
5. Class imbalance → consider focal loss or stratified K-fold

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

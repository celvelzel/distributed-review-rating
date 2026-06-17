# 2026-06-17 Project Structure Cleanup

## Summary

Major cleanup to fix project structure issues: removed dead code, archived old submissions, consolidated duplicate scripts, fixed security leaks, and improved portability.

## Changes Made

### Security Fixes
- **Kaggle token leak removed**: `submit_all.sh` and `wait_and_submit.sh` no longer contain hardcoded API tokens. Both now read from `config/kaggle_tokens.json` at runtime.

### Deleted
- `.omo/` — legacy Sisyphus/Omo agent state (superseded by `.mimocode/`)
- `.sisyphus/` — legacy Sisyphus agent state (superseded by `.mimocode/`)
- `code/ablation/` — empty directory with only `__init__.py`
- `code/run.sh` — all 6 stage runners were stubs (`# TODO` only), never implemented
- `code/features/run_lightgcn_csv.py` — duplicate of `run_lightgcn.py` with different output names

### Archived
- **222 CSV files** moved from `output/` to `output/archive/`
- **Top 20 submissions** (by Kaggle RMSE) kept in `output/`:
  - Best: `submission-dve90-r10.csv` (RMSE 0.61734)
  - 20th: `submission-stacking-v2.csv` (RMSE 0.66376)

### Consolidated
- **LightGCN scripts merged**: `run_lightgcn.py` now supports `--input-format parquet|csv` argument
  - Output filenames unified to `user_emb.npy`, `item_emb.npy`, `user2idx.json`, `item2idx.json`
  - `full_graph_pipeline.py` updated to use standard filenames
- **Agent directories**: only `.mimocode/` (current agent) and `.opencode/` (project skills) retained

### Updated
- `README.md` — removed references to `code/kaggle/`, `code/website/`, `code/ablation/`, `code/run.sh`; added `.opencode/` and `.mimocode/` to structure; updated Quick Start commands
- `submit_all.sh` — reads token from `config/kaggle_tokens.json`, removed hardcoded HPC path
- `wait_and_submit.sh` — reads token from `config/kaggle_tokens.json`, uses relative paths

## Known Issues (Not Fixed This Round)

| Issue | Status | Notes |
|-------|--------|-------|
| `user_stats.py` target leakage | Pending | `assemble.py` loads leaking version; needs `user_stats_kfold.py` replacement |
| Hardcoded HPC paths (`/hpc/puhome/...`) | Pending | 12 files, 28 occurrences in `code/models/` scripts |
| Hardcoded `/usr/bin/python3.8` | Pending | 7 PySpark feature scripts |
| 4 near-duplicate TF-IDF scripts | Deferred | User chose to keep all for now |

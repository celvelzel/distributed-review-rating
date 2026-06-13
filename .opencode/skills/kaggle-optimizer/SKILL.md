---
name: kaggle-optimizer
description: >
  End-to-end Kaggle competition optimizer for the COMP5434 review-rating project.
  Use when: (1) optimizing model performance and reducing RMSE, (2) running experiments
  and submitting to Kaggle, (3) iterating on feature engineering, model tuning, or ensemble
  strategies, (4) reviewing experiment logs and deciding next optimization step, or
  (5) any task related to improving the Kaggle leaderboard score.
  NOT for: initial data exploration, writing reports/presentations, or non-ML tasks.
---

# Kaggle Optimizer

Optimize the COMP5434 review-rating prediction system to minimize Kaggle RMSE.

## Competition

- Competition: `comp-5434-2526-sem-3-project`
- Metric: RMSE (lower is better)
- Submission format: CSV with `id` (int) and `rating` (float)
- API token: set via `KAGGLE_API_TOKEN` env var

## Core Loop

Every optimization iteration follows:

```
ASSESS -> PLAN -> IMPLEMENT -> VALIDATE -> SUBMIT -> RECORD
```

### Step 1: ASSESS -- Understand Current State

1. Read `docs/progress/kaggle-optimization-progress.md` for latest history
2. Read `.omo/plans/kaggle-optimization-v2.md` for planned work
3. Check artifacts in `artifacts/features/` and `artifacts/models/`
4. Review submissions: `kaggle competitions submissions -c comp-5434-2526-sem-3-project`
5. Identify: what features exist, what models are trained, current best RMSE

### Step 2: PLAN -- Design Experiment

Choose ONE optimization lever. Freely explore any approach that might reduce RMSE:

- Feature engineering (new features, better representations, embeddings)
- Model selection (any ML/DL approach -- tree models, neural nets, transformers, etc.)
- Hyperparameter tuning (Optuna, grid search, Bayesian optimization)
- Ensemble / stacking / blending strategies
- Data augmentation, sample weighting, loss function design
- Post-processing (clipping, rounding, bias correction)
- Any other technique with evidence of potential improvement

Document: hypothesis, what changes, expected impact.

### Step 3: IMPLEMENT -- Execute

- Write/modify code in `code/` following existing patterns
- Use `code/config.py` for paths (ROOT, DATA_DIR, ARTIFACTS_DIR, OUTPUT_DIR)
- Save model artifacts to `artifacts/models/`
- Save feature artifacts to `artifacts/features/`
- Generate submission CSV in `output/`

### Step 4: VALIDATE -- Check Before Submitting

1. Verify submission format: exactly 2 columns (`id` int, `rating` float)
2. Check for NaN/Inf values in predictions
3. Verify prediction range is reasonable (1-5 for ratings)
4. Run 5-fold OOF validation and record fold mean/std RMSE
5. High fold variance (>0.05 std) indicates poor generalization

### Step 5: SUBMIT -- Upload to Kaggle

```bash
export KAGGLE_API_TOKEN=KGAT_95032a984dab4b2545f71383d9913c63
kaggle competitions submit -c comp-5434-2526-sem-3-project -f <submission.csv> -m "<description>"
```

Retrieve score:

```bash
kaggle competitions submissions -c comp-5434-2526-sem-3-project --csv | head -3
```

### Step 6: RECORD -- Document Results

| Kaggle vs Previous | Action |
|---------------------|--------|
| RMSE < previous best | SUCCESS -- commit, update progress, continue |
| RMSE > previous by > 0.01 | REGRESSION -- revert, investigate |
| RMSE +/- 0.005 | NEUTRAL -- keep if adds diversity, otherwise revert |

After each iteration, complete ALL of these:

**6a. Write Changelog**

Create or append to `docs/changelog/YYYY-MM-DD-<topic>.md`:

```markdown
## [HH:MM] Experiment Name

- Hypothesis: ...
- Changes: ...
- Local CV RMSE: mean +/- std
- Kaggle RMSE: ...
- Conclusion: kept/reverted
```

**6b. Save Model Checkpoints**

Every training script MUST implement checkpoint saving:

```python
# Generic checkpoint pattern (adapt to your framework)
import json, os

checkpoint = {
    "epoch": current_epoch,
    "fold": current_fold,
    "model_state": model.state_dict(),      # PyTorch
    # "model": model.get_params(),           # sklearn
    # "model": model.save_model(),           # LightGBM/CatBoost/XGBoost
    "optimizer_state": optimizer.state_dict(),
    "best_metric": best_rmse,
    "config": experiment_config,
}
path = f"artifacts/checkpoints/{model_name}_f{current_fold}_e{current_epoch}.json"
os.makedirs(os.path.dirname(path), exist_ok=True)
torch.save(checkpoint, path)  # or json.dump / joblib.dump

# Resume: load latest checkpoint and continue
```

- Checkpoint dir: `artifacts/checkpoints/`
- HPC jobs are limited to ~30 hours. Training MUST support resume from checkpoint.
- On restart, load latest checkpoint and continue training from that point.
- Always save: model weights, optimizer state, current epoch/fold, best metric.

**6c. Update Technical Dashboard**

After any score change or major milestone, update `tech_dashboard.html`:

- Update the "Kaggle Score" metric cards (current best, gap to competitor)
- Update the submission history table with new entries
- Update the roadmap status (completed/active/future)
- Update "Key Highlights" and "Risks" sections if findings changed
- Update the "Last Updated" timestamp in footer
- Update "Next Steps" if priorities shifted

## Critical Rules

1. **Kaggle RMSE is the ONLY success metric.** Local RMSE is a proxy only.
2. **Leakage audit before every experiment.** Target-dependent features MUST use OOF.
3. **Never trust single split.** Always 5-fold OOF. Record fold mean AND std.
4. **Overfitting detection.** If local improves but Kaggle doesn't -> investigate.
5. **Simpler is better.** If simple approach beats complex, prefer simple.
6. **No technology bias.** Try anything. The only judge is Kaggle RMSE.

## Leakage Prevention

Before EVERY experiment:

- [ ] No feature uses target (rating) directly without OOF encoding
- [ ] User stats use only other users' data (K-Fold split)
- [ ] Product stats use only other products' data
- [ ] Parent product stats don't leak from child products in same fold
- [ ] No temporal leakage (no test-time info in training features)
- [ ] Validation split is stratified or temporal

**If unsure about leakage: DO NOT USE THE FEATURE.**

## File Locations

| Purpose | Path |
|---------|------|
| Training scripts | `code/models/*.py` |
| Feature engineering | `code/features/*.py` |
| Config | `code/config.py` |
| Output submissions | `output/*.csv` |
| Progress reports | `docs/progress/*.md` |
| Changelogs | `docs/changelog/*.md` |
| Optimization plans | `.omo/plans/*.md` |
| Experiment evidence | `.sisyphus/evidence/*.md` |
| Feature artifacts | `artifacts/features/*.npz` or `*.parquet` |
| Model checkpoints | `artifacts/checkpoints/` |
| Technical dashboard | `tech_dashboard.html` |

## Environment

- Hardware: RTX 3080 Ti (12GB VRAM) -- GPU OOM is a constraint for large models
- HPC job limit: ~30 hours per job -- code MUST support checkpoint/resume
- PySpark available for distributed processing

## Detailed ML Guidelines

For comprehensive ML engineering reference (experiment tracking, documentation
requirements), read `references/ml-engineering.md`.

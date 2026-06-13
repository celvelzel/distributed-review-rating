# ML Engineering Reference -- COMP5434 Review Rating

## Experiment Tracking Table

Maintain this table for every submission. Update after each Kaggle result.

| Exp ID | Date | Hypothesis | Features | Model | Local CV RMSE | Fold Std | Kaggle RMSE | Status |
|--------|------|------------|----------|-------|---------------|----------|-------------|--------|
| E001 | 2026-06-12 | TF-IDF baseline | word 5K | LGB | 1.176 | 0.012 | 0.801 | baseline |
| E002 | 2026-06-12 | +regularization | word 5K | LGB | 1.171 | 0.011 | 0.790 | best |

Status values: baseline / best / improved / reverted / failed / pending

## Overfitting Detection Patterns

**Pattern 1: Local-CV-vs-Kaggle Gap**

| Local CV RMSE | Kaggle RMSE | Diagnosis |
|---------------|-------------|-----------|
| 1.17 | 0.79 | Normal (public LB uses subset) |
| 0.55 | 1.30 | SEVERE LEAKAGE -- local is fake |
| 1.10 | 1.10 | Healthy, generalizes well |
| 1.00 | 1.05 | Slight overfit, acceptable |
| 0.80 | 1.20 | Major overfit, investigate features |

**Pattern 2: Fold Variance**

| Fold Std | Meaning | Action |
|----------|---------|--------|
| < 0.01 | Very stable | Good sign |
| 0.01-0.03 | Normal | Proceed |
| 0.03-0.05 | Borderline | Check data splits |
| > 0.05 | Unstable | Investigate before submitting |

**Pattern 3: Score Regression Timeline**

If consecutive submissions show no improvement:
- 2 submissions flat -> try different feature set
- 3 submissions flat -> fundamental approach change needed
- 5+ submissions flat -> revisit problem understanding

## Leakage Taxonomy

Detailed types with detection methods:

**Type A: Target Leakage (most common)**
- Symptom: Local CV excellent (RMSE < 0.8), Kaggle terrible (RMSE > 1.2)
- Cause: Feature encodes rating directly (e.g., product_avg_rating)
- Detection: Remove suspect feature, retrain, check if local CV drops to match Kaggle
- Fix: OOF encoding only for all target-dependent features

**Type B: Temporal Leakage**
- Symptom: Time-based CV much worse than random CV
- Cause: Training features use future information (e.g., stats computed after prediction time)
- Detection: Compare random-CV vs temporal-CV scores
- Fix: Only use information available before prediction timestamp

**Type C: Validation Contamination**
- Symptom: CV score good but LB score poor, fold std very low
- Cause: Validation set is not representative (e.g., same user in train and val)
- Detection: Check for user/product ID overlap between train/val
- Fix: GroupKFold by user_id or product_id

**Type D: Feature-Target Proxy**
- Symptom: Removing a "good" feature improves Kaggle score
- Cause: Feature is correlated with target in train but not test (distribution shift)
- Detection: Adversarial validation -- can a model distinguish train from test using this feature?
- Fix: Remove feature or use only if adversarial val shows no distribution shift

## Checkpoint Resume Protocol

For HPC jobs with ~30h limit:

```python
import glob, os

def find_latest_checkpoint(checkpoint_dir, model_name):
    pattern = f"{checkpoint_dir}/{model_name}_f*_e*.json"
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return torch.load(files[-1])  # or json.load / joblib.load

def train_with_resume(model_name, checkpoint_dir="artifacts/checkpoints"):
    ckpt = find_latest_checkpoint(checkpoint_dir, model_name)
    start_fold = ckpt["fold"] if ckpt else 0
    start_epoch = ckpt["epoch"] + 1 if ckpt else 0

    for fold in range(start_fold, n_folds):
        for epoch in range(start_epoch if fold == start_fold else 0, n_epochs):
            # ... train one epoch ...
            save_checkpoint(model_name, fold, epoch, metrics)
        start_epoch = 0  # reset for next fold
```

Key rules:
- Save after every epoch AND every fold
- Include fold number, epoch number, best metric, optimizer state
- On job restart, auto-detect latest checkpoint and resume
- Clean up old checkpoints periodically (keep best + last 2 per fold)

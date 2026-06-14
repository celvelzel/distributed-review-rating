---
name: kaggle-submission
description: >
  Submit prediction results to the COMP5434 Kaggle competition.
  Use when: (1) generating prediction files in output/, (2) model training is complete
  and needs to submit results, (3) uploading a submission CSV to Kaggle.
  NOT for: model training, feature engineering, or optimization.
---

# Kaggle Submission Skill

Submit prediction results to the COMP5434 Kaggle competition.

## Prerequisites

### 1. Install Kaggle CLI

```bash
pip install kaggle
```

### 2. Authenticate

Run the auth script — it tries 3 tokens from `config/kaggle_tokens.json` and saves the working one:

```bash
bash scripts/kaggle_auth.sh
```

If the script is unavailable, set the token manually:

```bash
export KAGGLE_API_TOKEN=<one of your tokens>
```

### 3. Verify Access

```bash
kaggle competitions submissions -c comp-5434-2526-sem-3-project --csv
```

If this returns a table of submissions, the setup is correct.

## When to Use

- After generating prediction files in `output/`
- When model training is complete and needs to submit results
- Any time you need to upload a submission CSV to Kaggle

## Competition Details

- **Competition Name**: `comp-5434-2526-sem-3-project`
- **Submission Format**: CSV with columns `id` and `rating`
- **Output Directory**: `output/`

## How to Submit

Run this command from the project root:

```bash
kaggle competitions submit \
  -c comp-5434-2526-sem-3-project \
  -f <path-to-submission.csv> \
  -m "<descriptive message>"
```

## Examples

```bash
# Submit stage0 baseline
kaggle competitions submit -c comp-5434-2526-sem-3-project -f output/stage0_submission.csv -m "Stage 0 baseline"

# Submit final model
kaggle competitions submit -c comp-5434-2526-sem-3-project -f output/submission-final.csv -m "Final XGBoost model"

# Submit stage1 results
kaggle competitions submit -c comp-5434-2526-sem-3-project -f output/submission-stage1.csv -m "Stage 1 predictions"
```

## Submission History

To view past submissions:

```bash
kaggle competitions submissions -c comp-5434-2526-sem-3-project
```

## Notes

- The API token is already configured via `KAGGLE_API_TOKEN` environment variable
- Submission file must have exactly 2 columns: `id` (integer) and `rating` (float)
- Common messages: "baseline", "stage1", "stage2", "final", "tuned params", etc.

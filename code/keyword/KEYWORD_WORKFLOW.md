# Keyword Workflow Guide (Final Robust Version)

This folder maps to the official production pipeline in `code/models/keyword_direct_pipeline.py`.

## Goal

Replace weak keyword heuristics with robust direct relationship pairs between title text and rating labels, while reducing overfitting risk.

The final approach is:

1. Learn keyword -> rating rules from train titles.
2. Validate each rule with out-of-fold (OOF) precision.
3. Apply only high-confidence direct rules on test.
4. Merge these direct overrides into advanced fine-tuned result CSVs.

## Key Anti-Overfitting Guardrails

- Uses 5-fold stratified OOF validation before approving a keyword rule.
- Requires minimum train support and OOF support.
- Requires minimum OOF precision, margin, and lift over class baseline.
- Rejects unstable keywords when fold-wise predicted label is inconsistent.
- Gives direct star phrases highest priority (`5 stars`, `one star`, etc.).
- Applies keyword overrides only when confidence is above merge threshold.

## Files and Outputs

Main script:

- `code/models/keyword_direct_pipeline.py`

Compatibility wrappers (this folder):

- `keyword_rule_finder.py`
- `title_pickup.py`
- `keyword_submission_merge.py`

Output artifacts:

- `data/direct_keyword_pairs.csv` - full rule mining report
- `data/direct_keyword_rate.csv` - direct predictions from title pairs
- `output/submissiona/submission-keyword-direct.csv` - merged final CSV when base submission is given

## Recommended End-to-End Command

Use the project venv Python to avoid environment mismatch:

```bash
./.venv/bin/python code/models/keyword_direct_pipeline.py \
  --keywords-file data/test_candidate_keywords.txt \
  --base-submission output/sub-deb1m-ve60-sv3rlg40.csv \
  --rules-out data/direct_keyword_pairs.csv \
  --direct-out data/direct_keyword_rate.csv \
  --merged-out output/submissiona/sub-deb1m-ve60-sv3rlg40-keyword.csv
```

## How to Integrate with Advanced Fine-Tune CSVs

Your advanced fine-tune output remains the base prediction file.
The direct keyword-rate CSV is a controlled override layer.

Integration order:

1. Generate advanced model CSV (DeBERTa/stacking/blend).
2. Run keyword direct pipeline using that CSV as `--base-submission`.
3. Submit merged output CSV in `output/submissiona/`.

This keeps your strong model signal and only replaces rows with high-confidence direct evidence.

## Compatibility Notes

Legacy scripts in the old temp folder can still be kept, but the official location is now `code/keyword/`.

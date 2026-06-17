#!/bin/bash
# Wait for DeBERTa training to complete, then generate predictions and submit
set -e

PROJECT="/hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating"
PYTHON="/hpc/puhome/25116696g/.conda/envs/SHPC-env/bin/python"
LOG="$PROJECT/artifacts/wait_and_submit.log"

echo "$(date): Waiting for DeBERTa training to complete..." >> "$LOG"

# Wait for training to finish
while ps aux | grep -q "[d]eberta_lora.py"; do
    sleep 60
done

echo "$(date): Training completed!" >> "$LOG"

# Generate predictions
cd "$PROJECT"
$PYTHON code/models/auto_submit.py >> "$LOG" 2>&1

# Try to submit
export KAGGLE_API_TOKEN=KGAT_95032a984dab4b2545f71383d9913c63
if kaggle competitions submit -c comp-5434-2526-sem-3-project \
    -f output/submission-deberta-ensemble.csv \
    -m "deberta-small-ensemble" 2>&1; then
    echo "$(date): Submission successful!" >> "$LOG"
    kaggle competitions submissions -c comp-5434-2526-sem-3-project --csv >> "$LOG" 2>&1
else
    echo "$(date): Submission failed (API key may need regeneration)" >> "$LOG"
fi

echo "$(date): Done" >> "$LOG"

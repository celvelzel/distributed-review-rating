#!/bin/bash
# Wait for DeBERTa training to complete, then generate predictions and submit
set -e

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKEN_FILE="$PROJECT/config/kaggle_tokens.json"
LOG="$PROJECT/artifacts/wait_and_submit.log"

if [ ! -f "$TOKEN_FILE" ]; then
    echo "ERROR: $TOKEN_FILE not found."
    exit 1
fi

export KAGGLE_API_TOKEN=$(python3 -c "import json; print(json.load(open('$TOKEN_FILE'))['tokens'][0])")

echo "$(date): Waiting for DeBERTa training to complete..." >> "$LOG"

# Wait for training to finish
while ps aux | grep -q "[d]eberta_lora.py"; do
    sleep 60
done

echo "$(date): Training completed!" >> "$LOG"

# Generate predictions
cd "$PROJECT"
python code/models/auto_submit.py >> "$LOG" 2>&1

# Try to submit
if kaggle competitions submit -c comp-5434-2526-sem-3-project \
    -f output/submission-deberta-ensemble.csv \
    -m "deberta-small-ensemble" 2>&1; then
    echo "$(date): Submission successful!" >> "$LOG"
    kaggle competitions submissions -c comp-5434-2526-sem-3-project --csv >> "$LOG" 2>&1
else
    echo "$(date): Submission failed (API key may need regeneration)" >> "$LOG"
fi

echo "$(date): Done" >> "$LOG"

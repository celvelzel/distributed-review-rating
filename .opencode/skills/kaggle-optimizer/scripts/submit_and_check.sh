#!/bin/bash
# Submit a prediction file to Kaggle and retrieve the score.
# Usage: bash submit_and_check.sh <submission.csv> "<message>"
#
# Example:
#   bash submit_and_check.sh output/submission-final.csv "LightGBM + TF-IDF 50K + SVD"

set -euo pipefail

SUBMISSION_FILE="${1:?Usage: submit_and_check.sh <file> <message>}"
MESSAGE="${2:?Usage: submit_and_check.sh <file> <message>}"
COMPETITION="comp-5434-2526-sem-3-project"

export KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-KGAT_95032a984dab4b2545f71383d9913c63}"

if [ ! -f "$SUBMISSION_FILE" ]; then
    echo "ERROR: File not found: $SUBMISSION_FILE"
    exit 1
fi

echo "=== Submitting to Kaggle ==="
echo "File: $SUBMISSION_FILE"
echo "Message: $MESSAGE"
echo ""

kaggle competitions submit -c "$COMPETITION" -f "$SUBMISSION_FILE" -m "$MESSAGE"

echo ""
echo "=== Retrieving latest score ==="
sleep 3
kaggle competitions submissions -c "$COMPETITION" --csv | head -3

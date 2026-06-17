#!/bin/bash
# Submit all pending predictions to Kaggle
# Usage: bash submit_all.sh
# Token is read from config/kaggle_tokens.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKEN_FILE="$SCRIPT_DIR/config/kaggle_tokens.json"

if [ ! -f "$TOKEN_FILE" ]; then
    echo "ERROR: $TOKEN_FILE not found. Create it with your Kaggle API tokens."
    exit 1
fi

export KAGGLE_API_TOKEN=$(python3 -c "import json; print(json.load(open('$TOKEN_FILE'))['tokens'][0])")
COMP="comp-5434-2526-sem-3-project"

echo "Testing Kaggle API..."
if ! kaggle competitions submissions -c $COMP --csv 2>/dev/null | head -1 > /dev/null; then
    echo "ERROR: Kaggle API not working. Check your token in $TOKEN_FILE"
    exit 1
fi

echo "Submitting best predictions..."
for f in output/submission-dve90-r10.csv \
         output/submission-dve95-r5.csv \
         output/submission-deberta-ve.csv \
         output/submission-base_ve_90_small_ve_10.csv; do
    if [ -f "$f" ]; then
        name=$(basename "$f" .csv)
        echo "  Submitting $name..."
        kaggle competitions submit -c $COMP -f "$f" -m "$name" 2>&1 | tail -1
        sleep 2
    fi
done

echo ""
echo "Checking scores..."
kaggle competitions submissions -c $COMP --csv 2>/dev/null | head -5

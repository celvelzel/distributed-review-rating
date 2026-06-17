#!/bin/bash
# Submit all pending predictions to Kaggle
# Usage: bash submit_all.sh
export KAGGLE_API_TOKEN=KGAT_95032a984dab4b2545f71383d9913c63
# If the above doesn't work, try:
# export KAGGLE_USERNAME=rickyma1028
# export KAGGLE_KEY=<new_key_here>

COMP="comp-5434-2526-sem-3-project"
cd /hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating

echo "Testing Kaggle API..."
if ! kaggle competitions submissions -c $COMP --csv 2>/dev/null | head -1 > /dev/null; then
    echo "ERROR: Kaggle API not working. Please update ~/.kaggle/kaggle.json with a new API token."
    echo "Go to: https://www.kaggle.com/settings -> API -> Create New Token"
    exit 1
fi

echo "Submitting best predictions..."
for f in output/submission-deberta_95_ridge_5.csv \
         output/submission-deberta_90_ridge_10.csv \
         output/submission-deberta-qm.csv \
         output/submission-deberta-ve.csv \
         output/submission-draw90-r10.csv \
         output/submission-deberta90-pseudo-ridge10.csv; do
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

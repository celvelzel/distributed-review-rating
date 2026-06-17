#!/bin/bash
# Auto-launch DeBERTa-v3-large training after base model finishes
# Run this as a nohup background process

LOG="/hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating/artifacts/training_monitor.log"
BASE_CKPT="/hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating/artifacts/models/checkpoints_base_full/fold3_epoch3.pt"
LARGE_SCRIPT="/hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating/code/models/deberta_large_full.py"
PYTHON="/hpc/puhome/25116696g/.conda/envs/SHPC-env/bin/python"

echo "$(date): Auto-launch script started, waiting for base training to complete..." >> "$LOG"

# Wait for base training to complete (fold3_epoch3.pt exists)
while [ ! -f "$BASE_CKPT" ]; do
    if ! pgrep -f "deberta_base_full.py" > /dev/null 2>&1; then
        if [ ! -f "$BASE_CKPT" ]; then
            echo "$(date): WARNING - base training process gone and no checkpoint!" >> "$LOG"
        fi
    fi
    sleep 60
done

echo "$(date): Base training complete! fold3_epoch3.pt found." >> "$LOG"
echo "$(date): Waiting 30 seconds for any cleanup..." >> "$LOG"
sleep 30

echo "$(date): Starting DeBERTa-v3-large training..." >> "$LOG"
$PYTHON -u "$LARGE_SCRIPT" 2>&1 | tee /hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating/artifacts/large_full_training.log

echo "$(date): DeBERTa-v3-large training complete!" >> "$LOG"

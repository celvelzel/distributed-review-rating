#!/bin/bash
# Monitor DeBERTa training and auto-generate predictions when new folds complete
set -e

PROJECT_DIR="/hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating"
PYTHON="/hpc/puhome/25116696g/.conda/envs/SHPC-env/bin/python"
CKPT_DIR="$PROJECT_DIR/artifacts/models/checkpoints_lora"
LOG_FILE="$PROJECT_DIR/artifacts/monitor_auto_submit.log"

echo "$(date): Monitoring started" >> "$LOG_FILE"

LAST_COUNT=0
while true; do
    # Count checkpoints
    CURRENT_COUNT=$(ls -1 "$CKPT_DIR"/fold*_epoch*.pt 2>/dev/null | wc -l)
    
    if [ "$CURRENT_COUNT" -gt "$LAST_COUNT" ]; then
        echo "$(date): New checkpoint detected ($CURRENT_COUNT total)" >> "$LOG_FILE"
        
        # Generate predictions
        cd "$PROJECT_DIR"
        $PYTHON code/models/auto_submit.py >> "$LOG_FILE" 2>&1
        
        echo "$(date): Predictions generated" >> "$LOG_FILE"
        LAST_COUNT=$CURRENT_COUNT
    fi
    
    # Check if training is still running
    if ! ps aux | grep -q "[d]eberta_lora.py"; then
        echo "$(date): Training completed or stopped" >> "$LOG_FILE"
        break
    fi
    
    sleep 300  # Check every 5 minutes
done

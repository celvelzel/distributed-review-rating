#!/bin/bash
# Monitor DeBERTa-v3-large training progress
# Runs every hour, logs to artifacts/monitoring.log

LOG_FILE="/hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating/artifacts/monitoring.log"
TRAIN_LOG="/hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating/artifacts/large_full_training.log"

echo "========================================" >> "$LOG_FILE"
echo "Monitoring Check: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Check if process is running
PID=$(ps -ef | grep deberta_large_full.py | grep -v grep | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "Status: ❌ Process NOT running" >> "$LOG_FILE"
else
    echo "Status: ✅ Process running (PID: $PID)" >> "$LOG_FILE"
fi

# Get latest training progress
echo "" >> "$LOG_FILE"
echo "Latest Progress:" >> "$LOG_FILE"
tail -5 "$TRAIN_LOG" >> "$LOG_FILE"

# Get val_rmse results
echo "" >> "$LOG_FILE"
echo "Val RMSE Results:" >> "$LOG_FILE"
grep -E "Fold.*Epoch.*val_rmse|OOF RMSE|Saved" "$TRAIN_LOG" >> "$LOG_FILE"

# GPU usage
echo "" >> "$LOG_FILE"
echo "GPU Usage:" >> "$LOG_FILE"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader >> "$LOG_FILE"

# Memory usage
echo "" >> "$LOG_FILE"
echo "Memory Usage:" >> "$LOG_FILE"
free -h | grep Mem >> "$LOG_FILE"

echo "" >> "$LOG_FILE"
echo "---" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

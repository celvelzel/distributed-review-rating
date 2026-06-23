#!/bin/bash
# Check DeBERTa-v3-large training progress

echo "========================================"
echo "Training Progress Check: $(date)"
echo "========================================"

# Check if process is running
PID=$(ps aux | grep "deberta_large_full.py" | grep -v grep | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "❌ Training process not found"
    exit 1
fi
echo "✅ Process running (PID: $PID)"

# Check GPU usage
echo ""
echo "GPU Usage:"
nvidia-smi | grep -A 3 "MiB" | tail -2

# Check latest checkpoint
echo ""
echo "Latest Checkpoints:"
ls -lht artifacts/models/checkpoints_large_full/*.pt | head -5

# Check if training is complete
LATEST_CKPT=$(cat artifacts/models/checkpoints_large_full/latest.txt 2>/dev/null)
echo ""
echo "Latest Checkpoint: $LATEST_CKPT"

if [ "$LATEST_CKPT" = "fold3_epoch3.pt" ]; then
    echo "🎉 Training Complete!"
else
    echo "🔄 Training in progress..."
fi

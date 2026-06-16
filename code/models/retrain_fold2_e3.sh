#!/bin/bash
# Retrain fold 2 epoch 3 only
# Run this AFTER fold 3 completes, to fill in the missing epoch

CKPT_DIR="/hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating/artifacts/models/checkpoints_base_full"

# Save current latest.txt (fold3_epoch3)
cp "$CKPT_DIR/latest.txt" "$CKPT_DIR/latest.txt.bak"

# Set latest to fold2_epoch2 so resume logic picks up fold 2 from epoch 3
echo "fold2_epoch2.pt" > "$CKPT_DIR/latest.txt"

echo "Starting fold 2 epoch 3 retrain..."
/hpc/puhome/25116696g/.conda/envs/SHPC-env/bin/python -u \
  /hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating/code/models/deberta_base_full.py \
  2>&1 | tee /hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating/artifacts/fold2_e3_retrain.log

# After completion, restore latest.txt to fold3_epoch3 (or whatever is newest)
# The script will update latest.txt itself, but we should restore the backup
# to keep the full training history
echo "Fold 2 epoch 3 retrain complete!"
echo "Latest checkpoint:"
cat "$CKPT_DIR/latest.txt"

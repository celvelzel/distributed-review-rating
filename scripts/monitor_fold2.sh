#!/bin/bash
# Monitor fold2 training and check prediction quality when done

LOG_FILE="artifacts/large_r16_retrain_v2.log"
PRED_FILE="artifacts/models/deberta_large_r16_fold2_test.npy"

echo "========================================"
echo "Monitoring Fold 2 Training"
echo "========================================"

while true; do
    # Check if fold2 prediction file exists
    if [ -f "$PRED_FILE" ]; then
        echo ""
        echo "$(date): Fold 2 prediction file found!"
        echo "========================================"
        
        # Check prediction statistics
        python3.8 -c "
import numpy as np

fold1 = np.load('artifacts/models/deberta_large_fold1_test.npy')
fold2 = np.load('$PRED_FILE')

print('=== Prediction Comparison ===')
print(f'Fold 1: mean={fold1.mean():.4f}, std={fold1.std():.4f}')
print(f'Fold 2: mean={fold2.mean():.4f}, std={fold2.std():.4f}')

ratio = fold2.std() / fold1.std()
print(f'\nStd Ratio (Fold2/Fold1): {ratio:.4f}')

if ratio < 0.5:
    print('\n❌ WARNING: Fold 2 std is much lower than Fold 1!')
    print('   This suggests LoRA B weights are still zero.')
    print('   Need to investigate further.')
elif ratio < 0.8:
    print('\n⚠️  CAUTION: Fold 2 std is somewhat lower than Fold 1.')
    print('   This might be acceptable but worth monitoring.')
else:
    print('\n✅ Fold 2 std is similar to Fold 1.')
    print('   Training appears to be working correctly!')
"
        break
    fi
    
    # Check if training is still running
    if ! ps aux | grep -q "[d]eberta_large_r16_retrain.py"; then
        echo "$(date): Training process not found!"
        break
    fi
    
    # Show progress
    echo "$(date): $(tail -1 $LOG_FILE)"
    sleep 300  # Check every 5 minutes
done

#!/bin/bash
# Wait for DeBERTa-v3-large training to complete, then generate submissions

echo "========================================"
echo "Waiting for DeBERTa-v3-large training"
echo "========================================"

while true; do
    # Check if process is running
    PID=$(ps aux | grep "deberta_large_full.py" | grep -v grep | awk '{print $2}')
    
    if [ -z "$PID" ]; then
        echo ""
        echo "$(date): Training process completed!"
        break
    fi
    
    # Check if test predictions exist
    if [ -f "artifacts/models/deberta_large_full_test.npy" ]; then
        echo ""
        echo "$(date): Test predictions found!"
        break
    fi
    
    echo "$(date): Still training... (PID: $PID)"
    sleep 60
done

echo ""
echo "========================================"
echo "Generating submissions..."
echo "========================================"

python3.8 code/ensemble/generate_large_submissions.py

echo ""
echo "========================================"
echo "Done! Submissions ready in output/"
echo "========================================"

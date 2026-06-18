#!/bin/bash
# Check ablation training status
PID=1496164
LOG_FILE="/hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating/logs/3m_bs16_ablation_direct.log"

# Process status
if ps -p $PID > /dev/null 2>&1; then
    echo "Process alive: yes"
    ps -p $PID -o pid,pcpu,etime,cputime --no-headers
else
    echo "Process alive: no"
    echo "Checking for output files..."
    ls -la /hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating/artifacts/models/deberta_3m_bs16_ablation_fold1e1_*.npy 2>/dev/null || echo "No output files found"
    exit 0
fi

# GPU status
echo "GPU status:"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo "NO GPU"

# Live output
echo "Latest output:"
cat /proc/$PID/fd/1 2>/dev/null | tail -5 || echo "Cannot read /proc/fd/1"

# Log file tail (may lag)
echo "Log file tail:"
tail -5 $LOG_FILE 2>/dev/null || echo "No log file"
#!/bin/bash
set -e
cd /hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating

echo "=== Step 2: stacking_v3.py ==="
python3.8 code/models/stacking_v3.py

echo "=== Step 3: verify_stacking_v3.py ==="
python3.8 code/models/verify_stacking_v3.py

echo "=== Step 4: submit_stacking_v3.py ==="
python3.8 code/models/submit_stacking_v3.py

echo "=== ALL DONE ==="

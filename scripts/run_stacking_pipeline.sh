#!/bin/bash
# Stacking V3 Pipeline - 最优流程留档
# 用途: 运行完整的 Stacking V3 流程 (当前最佳 Kaggle RMSE: 0.59770)
#
# 流程说明:
#   Step 1: train_graph_models.py    → 8个 .npy + graph_models_results.json
#   Step 2: stacking_v3.py           → 9 base models + 5 meta-learner, 输出 stacking_v3_*.npy
#   Step 3: verify_stacking_v3.py    → 对比 v2, 输出 stacking-v3-verification.md
#   Step 4: submit_stacking_v3.py    → 生成 Kaggle 提交 CSV
#
# 最终混合配方: VE 60% + Stacking V3 ridge+lgb 40%
# 对应提交文件: output/sub-deb1m-ve60-sv3rlg40.csv (Kaggle RMSE: 0.59770)

set -e
cd /hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating

echo "=== Step 1: train_graph_models.py ==="
python3.8 code/models/train_graph_models.py

echo "=== Step 2: stacking_v3.py ==="
python3.8 code/models/stacking_v3.py

echo "=== Step 3: verify_stacking_v3.py ==="
python3.8 code/models/verify_stacking_v3.py

echo "=== Step 4: submit_stacking_v3.py ==="
python3.8 code/models/submit_stacking_v3.py

echo "=== ALL DONE ==="

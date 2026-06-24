# 2026-06-23 DeBERTa-v3-large 提交记录

_更新日期: 2026-06-23 11:00_

## 目标

验证 DeBERTa-v3-large 模型是否能超越当前最佳的 Base 模型。

## 当前最佳

**Kaggle RMSE: 0.59770** (VE 60% + Stacking V3 ridge+lgb 40%)

## 提交结果

### Large 模型提交 (2026-06-23)

| # | 文件名 | Kaggle RMSE | vs 最佳 | 策略 | 状态 |
|---|--------|-------------|---------|------|------|
| 1 | sub-large-fold1-ve60-rlg40.csv | 0.67785 | +13.4% | Large fold1 VE 60% + RLG 40% | 🔄 重新训练中 |
| 2 | sub-large-fold1-ve-only.csv | 0.77886 | +30.3% | Large fold1 VE only (100%) | 🔄 重新训练中 |
| 3 | sub-large-ve60-rlg40.csv | 1.07035 | +79.1% | Large VE 60% + RLG 40% | 🔄 重新训练中 |
| 4 | sub-large-ve-only.csv | 1.47286 | +146.4% | Large VE only (100%) | 🔄 重新训练中 |

### 预测统计

| # | 文件名 | Mean | Std | Min | Max |
|---|--------|------|-----|-----|-----|
| 1 | sub-large-fold1-ve60-rlg40.csv | 4.013 | 1.041 | 1.111 | 4.862 |
| 2 | sub-large-fold1-ve-only.csv | 4.010 | 1.243 | 1.000 | 4.871 |
| 3 | sub-large-ve60-rlg40.csv | 4.022 | 0.739 | 1.228 | 4.884 |
| 4 | sub-large-ve-only.csv | 4.025 | 1.032 | 1.000 | 5.000 |

## 问题分析

### fold2/fold3 训练参数问题

DeBERTa-v3-large 模型使用以下配置训练:

| 参数 | 值 |
|------|-----|
| 模型 | `microsoft/deberta-v3-large` |
| 参数量 | 435M (308M backbone + 127M classifier) |
| 训练数据 | 3M 样本 (全量) |
| 训练配置 | 3 折 × 3 epoch, BS=64, GradAcc=2 |
| LoRA 配置 | r=32, alpha=64, target=[query_proj, value_proj, key_proj, output_proj, intermediate] |
| 学习率 | 3e-5 |

**问题原因**:

1. **参数量与训练配置不匹配**
   - Large 模型参数量 (435M) 是 Base (86M) 的 5 倍
   - 但训练配置 (3f×3e) 与 Base 模型相同
   - 导致 Large 模型严重欠拟合

2. **Val RMSE 对比**
   - Base 模型 (1M, 5f×5e): Val RMSE 1.117
   - Large 模型 (3M, 3f×3e): Val RMSE 1.420
   - Large 模型比 Base 差 27.1%

3. **fold2/fold3 具体表现**
   - Fold 2 Epoch 1: 1.41828
   - Fold 2 Epoch 2: 1.41794 (-0.02%)
   - Fold 3 Epoch 2: 1.42051
   - Fold 3 Epoch 3: 1.42045 (-0.004%)
   - 改善幅度极小，说明模型已达到训练极限

### Kaggle 分数对比

| 模型 | Val RMSE | Kaggle RMSE | 状态 |
|------|----------|-------------|------|
| DeBERTa-v3-base (1M) | 1.117 | 0.598 | 🏆 最佳 |
| DeBERTa-v3-large (3M) | 1.420 | 0.678 | 🔄 重新训练中 |

## 结论

1. **Large 模型首次训练配置无效**
   - 所有 4 次提交均不如当前最佳
   - 最佳 Large 提交 (0.678) 比 Base 最佳 (0.598) 差 13.4%

2. **原因明确**
   - fold2/fold3 训练参数配置 (3f×3e) 不足以训练 435M 参数的模型
   - 需要更长的训练配置 (如 5f×5e) 或更低的学习率

3. **下一步行动**
   - **状态: 🔄 重新训练中**
   - 调整训练配置后重新训练 Large 模型
   - 继续使用 Base 模型的 VE 60% + Stacking V3 ridge+lgb 40% 配方

## 相关文档

- 训练状态: `docs/progress/2026-06-23-training-status.md`
- 训练详情: `docs/changelog/2026-06-23-deberta-large-training.md`
- 技术看板: `tech_dashboard.html`

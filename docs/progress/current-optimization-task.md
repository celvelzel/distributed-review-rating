# 当前优化任务: DeBERTa-v3-large LoRA 优化

_更新日期: 2026-06-21_

## 任务背景

当前最佳 Kaggle RMSE 为 0.59770，来自 DeBERTa-v3-base (1M) + VE 60% + Stacking V3 ridge+lgb 40%。

为进一步提升性能，正在训练 DeBERTa-v3-large (435M params) 模型。

## 优化策略

### 问题诊断

DeBERTa-v3-large 旧配置 (LR=1e-5, LoRA r=16) 效果不佳:
- Fold 1 Epoch 2 Val RMSE: 1.160
- 比 Base 模型 (1.139) 差 1.9%

### 优化方向

| 方向 | 旧配置 | 新配置 | 预期效果 |
|------|--------|--------|----------|
| LoRA r | 16 | **32** | 增加模型容量 |
| LoRA alpha | 32 | **64** | 增加缩放因子 |
| LoRA dropout | 0.05 | **0.02** | 减少正则化 |
| Target modules | 2 个 | **5 个** | 更多参数可训练 |
| LR | 1e-5 | **3e-5** | 增加学习率 |
| Batch Size | 16 | **64** | 加速训练 |
| **可训练参数** | **1.57M** | **14.16M** | **9× 增加** |

### Target Modules 对比

| 旧配置 | 新配置 |
|--------|--------|
| query_proj | query_proj |
| value_proj | value_proj |
| — | key_proj |
| — | output_proj |
| — | dense |

## 训练状态

### 当前配置

| 参数 | 值 |
|------|-----|
| 脚本 | `code/models/deberta_large_full.py` |
| 模型 | `microsoft/deberta-v3-large` |
| 参数量 | 435M |
| 可训练参数 | 14.16M |
| BATCH_SIZE | 64 |
| GRAD_ACCUM | 4 |
| Effective BS | 256 |
| LR | 3e-5 |
| N_FOLDS | 3 |
| N_EPOCHS | 3 |
| Patience | 3 |

### Checkpoints

| 文件 | 状态 | Val RMSE |
|------|------|----------|
| fold1_epoch1.pt | ✅ | ~1.1787 |
| fold1_epoch2.pt | ✅ | 1.15961 |
| fold1_epoch3.pt | ✅ | 1.15910 |

### 已完成的实验

| 配置 | BS | LoRA r | LR | Val RMSE | Kaggle |
|------|-----|--------|-----|----------|--------|
| 旧配置 | 16 | 16 | 1e-5 | 1.160 | — |
| 旧配置+LR | 16 | 16 | 2e-5 | 1.159 | — |
| 新配置 | 64 | 32 | 3e-5 | ? | — |
| 新配置+BS128 | 128 | 32 | 3e-5 | ❌ OOM | — |

## 遇到的问题

### 1. CUDA OOM (BS=128)

**错误**: `torch.OutOfMemoryError: CUDA out of memory`

**原因**: BS=128 + 14.16M 可训练参数导致 GPU 内存不足

**解决**: 回退到 BS=64 (已验证可行)

### 2. GPU 利用率下降

**现象**: BS=128 时 GPU 利用率从 99% 降到 85%

**原因**: 数据加载开销和内存分配

**影响**: 训练时间实际更短 (每 epoch ~4.4h vs ~4.8h)

## 下一步

### 立即可做

1. **启动训练**: 使用 BS=64 配置启动 DeBERTa-v3-large 训练
2. **监控进度**: 每小时检查训练状态
3. **提交结果**: 训练完成后用 VE 60% + Stacking V3 ridge+lgb 40% 配方提交

### 预期结果

| 指标 | 预期值 |
|------|--------|
| Fold 1 Ep3 Val RMSE | < 1.159 (比旧配置好) |
| Kaggle RMSE | < 0.59770 (比当前最佳好) |
| 训练时间 | ~26h (BS=64) |

### 风险

1. **OOM**: BS=64 已验证可行，但 14.16M 可训练参数仍有风险
2. **过拟合**: 更多参数可能导致过拟合
3. **训练时间**: ~26h 可能不够完成所有 fold

## 文件位置

| 文件 | 路径 |
|------|------|
| 训练脚本 | `code/models/deberta_large_full.py` |
| Checkpoints | `artifacts/models/checkpoints_large_full/` |
| 训练日志 | `artifacts/large_full_training.log` |
| 监控日志 | `artifacts/monitoring.log` |
| 监控脚本 | `scripts/monitor_training.sh` |

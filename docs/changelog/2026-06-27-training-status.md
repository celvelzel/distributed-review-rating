# 2026-06-27 DeBERTa-v3-large LoRA r=16 重训练状态

_更新日期: 2026-06-27 18:00_

## 概述

DeBERTa-v3-large 模型使用 LoRA r=16 重新训练 Fold 2 和 Fold 3，以修复之前 LoRA B 权重为零的问题。

## 问题背景

### 之前的问题

1. **Fold 2/3 预测异常**: std=0.034 (正常应为 ~0.75)
2. **根本原因**: `gradient_checkpointing_enable()` 阻断了 LoRA 梯度流
3. **LoRA B 权重为零**: 模型无法学习

### 解决方案

1. **禁用 gradient checkpointing**: 移除 `gradient_checkpointing_enable()`
2. **使用原始配置**: 与 Fold 1 训练脚本完全一致
3. **调整 batch size**: 从 16 降到 8 以避免 OOM

## 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | `microsoft/deberta-v3-large` | 435M 参数 |
| LoRA Rank | r=16 | 与 Fold 1 一致 |
| LoRA Alpha | 32 | 与 Fold 1 一致 |
| LoRA Dropout | 0.05 | 与 Fold 1 一致 |
| LoRA Target | ["query_proj", "value_proj"] | 与 Fold 1 一致 |
| Batch Size | 8 | 避免 OOM |
| Gradient Accumulation | 32 | Effective batch = 256 |
| Learning Rate | 1e-5 | 与 Fold 1 一致 |
| Epochs | 3 | 每个 fold |
| Folds | 2 & 3 | Fold 1 已完成 |

## 训练进度

### Fold 2

| Epoch | 状态 | Val RMSE | 时间 |
|-------|------|----------|------|
| Epoch 1 | ✅ 完成 | — | ~17.6 小时 |
| Epoch 2 | 🔄 88000/250619 (35.1%) | — | 进行中 |
| Epoch 3 | ⏳ 待训练 | — | — |

### Fold 3

| Epoch | 状态 | Val RMSE | 时间 |
|-------|------|----------|------|
| Epoch 1 | ⏳ 待训练 | — | — |
| Epoch 2 | ⏳ 待训练 | — | — |
| Epoch 3 | ⏳ 待训练 | — | — |

### Checkpoints

| 文件 | 大小 | 时间 |
|------|------|------|
| fold2_epoch1.pt | 1.7G | 6/27 09:36 |

## 关键发现

### 1. Gradient Checkpointing 与 LoRA 不兼容

- `gradient_checkpointing_enable()` 会阻断 LoRA 梯度流
- 原因: 整数输入张量没有 `requires_grad=True`
- 解决: 禁用 gradient checkpointing

### 2. Batch Size 限制

- batch_size=16 训练时需要 11.42 GB (超出 GPU 容量)
- batch_size=8 训练时需要 7.85 GB (适合 12 GB GPU)
- Effective batch size 保持 256 (8 × 32)

### 3. 训练速度

- batch_size=8: ~17.6 小时/epoch
- 总训练时间: ~105.6 小时 (~4.4 天)

## 当前状态

- **训练进程**: 已停止 (PID: 2846087)
- **最新 Checkpoint**: fold2_epoch1.pt
- **Fold 2 Epoch 2**: 35.1% 完成
- **预计剩余时间**: ~51 小时 (~2.1 天)

## 下一步

1. **恢复训练**: 从 fold2_epoch1.pt 继续
2. **完成 Fold 2**: Epoch 2 和 Epoch 3
3. **完成 Fold 3**: 3 个 Epochs
4. **生成预测**: 使用训练好的模型生成测试集预测
5. **验证预测质量**: 确保 std ≈ 0.75 (与 Fold 1 一致)

## 相关文件

- 训练脚本: `code/models/deberta_large_r16_retrain.py`
- Checkpoints: `artifacts/models/checkpoints_large_full_r16/`
- 训练日志: `artifacts/large_r16_retrain_v2.log`
- 监控脚本: `scripts/monitor_fold2.sh`

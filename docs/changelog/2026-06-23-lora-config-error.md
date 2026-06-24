# 2026-06-23 DeBERTa-v3-large LoRA 配置错误分析

_更新日期: 2026-06-23 19:00_

## 问题概述

DeBERTa-v3-large 模型训练完成后，发现 fold2 和 fold3 的测试集预测严重异常 (std=0.034)，而 fold1 预测正常 (std=0.756)。

## 问题现象

### 预测分布对比

| Fold | Mean | Std | Unique Values | 状态 |
|------|------|-----|---------------|------|
| Fold 1 | 3.9682 | 0.7564 | 3523 | ✅ 正常 |
| Fold 2 | 3.9656 | 0.0352 | 439 | ❌ 异常 |
| Fold 3 | 3.9666 | 0.0338 | 409 | ❌ 异常 |

### OOF 预测问题

- OOF Mean: 1.3227 (目标: 3.9412)
- OOF < 1.0: 2,004,960 (66.7%)
- 50% 分位数: 0.0000

## 根本原因

### LoRA 配置不一致

| 参数 | Fold 1 | Fold 2/3 |
|------|--------|----------|
| LoRA Rank | r=16 | r=32 |
| LoRA Alpha | 32 | 64 |
| LoRA Target | ["query_proj", "value_proj"] | ["query_proj", "value_proj", "key_proj", "output_proj", "dense"] |
| LoRA Layers | 96 | 288 |
| Trainable Params | 1.57M (0.36%) | 14M (3.16%) |
| Checkpoint Size | 847M | 1.7G |

### 问题分析

1. **训练脚本在 fold1 完成后被修改**: LoRA 配置从 r=16 改为 r=32
2. **Fold 2/3 使用了错误的配置**: 导致模型训练失败
3. **模型预测崩溃**: LoRA r=32 配置导致模型输出几乎常数

### 证据

- Fold 1 checkpoint: LoRA r=16, 96 layers, std=0.7564 (正常)
- Fold 2 checkpoint: LoRA r=32, 288 layers, std=0.0352 (异常)
- Base model weights: 完全相同 (diff=0.000000)

## 解决方案

### 1. 删除旧的 fold2/3 checkpoints 和 predictions

```bash
rm -f artifacts/models/checkpoints_large_full/fold2_epoch*.pt
rm -f artifacts/models/checkpoints_large_full/fold3_epoch*.pt
rm -f artifacts/models/deberta_large_fold2_test.npy
rm -f artifacts/models/deberta_large_fold3_test.npy
rm -f artifacts/models/deberta_large_full_oof.npy
rm -f artifacts/models/deberta_large_full_test.npy
```

### 2. 重新训练 fold2/3 使用 LoRA r=16

创建新脚本: `code/models/deberta_large_r16_retrain.py`

**配置**:
- LoRA Rank: r=16 (与 fold1 一致)
- LoRA Alpha: 32
- LoRA Target: ["query_proj", "value_proj"]
- 其他参数: BS=64, LR=3e-5, 3f×3e

**状态**: 训练中 (PID: 2347999)

### 3. 使用 fold1-only 预测作为临时方案

生成了 fold1-only 提交:
- `sub-large-fold1-ve60-rlg40.csv` (Large fold1 VE 60% + Stacking V3 rlg 40%)
- `sub-large-fold1-ve-only.csv` (Large fold1 VE only)

## 教训

1. **训练配置必须一致**: 同一模型的所有 fold 必须使用相同的配置
2. **检查 checkpoint 元数据**: 训练完成后应验证 LoRA 配置
3. **监控预测分布**: 异常的 std 是训练失败的明显信号
4. **版本控制训练脚本**: 避免在训练过程中修改配置

## 相关文件

- 训练脚本 (旧): `code/models/deberta_large_full.py`
- 训练脚本 (新): `code/models/deberta_large_r16_retrain.py`
- Checkpoints: `artifacts/models/checkpoints_large_full/`
- 训练日志: `artifacts/large_r16_retrain.log`

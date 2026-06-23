# 2026-06-23 DeBERTa-v3-large 训练完成

_更新日期: 2026-06-23 18:05_

## 概述

DeBERTa-v3-large 模型训练已完成，正在生成测试集预测。

## 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | `microsoft/deberta-v3-large` |
| 参数量 | 435M (308M backbone + 127M classifier) |
| 训练数据 | 3M 样本 (全量) |
| 训练脚本 | `code/models/deberta_large_full.py` |
| 训练配置 | 3 折 × 3 epoch, BS=64, GradAcc=2 |
| LoRA 配置 | r=32, alpha=64, target=[query_proj, value_proj, key_proj, output_proj, intermediate] |
| 学习率 | 3e-5 |
| GPU | NVIDIA GeForce RTX 3080 Ti (12.6GB) |

## 训练进度

| Fold | Epoch 1 | Epoch 2 | Epoch 3 |
|------|---------|---------|---------|
| Fold 1 | ✅ Complete | ✅ Complete | ✅ Complete |
| Fold 2 | ✅ 1.41828 | ✅ 1.41794 | ✅ Complete |
| Fold 3 | ✅ Complete | ✅ 1.42051 | ✅ 1.42045 |

## Val RMSE 趋势

```
Fold 2 Epoch 1: 1.41828
Fold 2 Epoch 2: 1.41794 (-0.02%) ✅ 改善
Fold 3 Epoch 2: 1.42051
Fold 3 Epoch 3: 1.42045 (-0.004%) ✅ 改善
```

## 训练时间

- 每个 epoch: ~6 小时 (31327 steps)
- 总训练时间: ~54 小时
- 开始时间: 2026-06-21 11:38
- 完成时间: 2026-06-23 17:36

## Checkpoints

| 文件 | 大小 | 时间 |
|------|------|------|
| fold1_epoch1.pt | 847M | 6/17 08:59 |
| fold1_epoch2.pt | 847M | 6/17 22:44 |
| fold1_epoch3.pt | 1.7G | 6/20 16:41 |
| fold2_epoch1.pt | 1.7G | 6/22 03:23 |
| fold2_epoch2.pt | 1.7G | 6/22 09:21 |
| fold2_epoch3.pt | 1.7G | 6/22 15:19 |
| fold3_epoch1.pt | 1.7G | 6/22 22:14 |
| fold3_epoch2.pt | 1.7G | 6/23 04:12 |
| fold3_epoch3.pt | 1.7G | 6/23 17:36 |

## 后处理状态

训练完成后，脚本正在执行以下后处理:

1. ✅ Fold 3 Epoch 3 训练完成
2. 🔄 生成 Fold 3 OOF 预测 (CPU 密集)
3. 🔄 生成 Fold 3 测试集预测
4. ⏳ 平均所有 fold 测试集预测
5. ⏳ 保存 `deberta_large_full_test.npy`
6. ⏳ 生成 VE 预测和混合提交

## 预计完成时间

- 后处理: ~30 分钟
- 预计完成: 2026-06-23 18:30

## 下一步

1. 等待后处理完成
2. 生成 Large 模型 VE 预测
3. 生成 Large VE 60% + Stacking V3 ridge+lgb 40% 提交
4. 提交到 Kaggle 验证

## 关键观察

1. **Val RMSE 较高**: Large 模型 Val RMSE 1.420 vs Base 模型 1.117
2. **可能原因**: 
   - Large 模型参数更多 (435M vs 86M)，更容易过拟合
   - 3f×3e 训练配置可能不足以训练 Large 模型
   - 需要更多训练 epochs 或更低的学习率
3. **仍需验证**: 测试集预测可能比 Val RMSE 表现更好

## 相关文件

- 训练脚本: `code/models/deberta_large_full.py`
- Checkpoints: `artifacts/models/checkpoints_large_full/`
- 训练日志: `artifacts/large_full_training_resume.log`
- 监控日志: `artifacts/wait_for_large_training.log`

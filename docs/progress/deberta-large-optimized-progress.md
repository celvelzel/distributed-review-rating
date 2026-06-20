# DeBERTa-v3-large 优化训练进度

## 实验目的

优化 DeBERTa-v3-large 的 LoRA 配置，增加模型容量以改善 val_rmse。

## 配置对比

| 参数 | 旧配置 | 新配置 | 变化 |
|------|--------|--------|------|
| LoRA r | 16 | **32** | 2× |
| LoRA alpha | 32 | **64** | 2× |
| LoRA dropout | 0.05 | **0.02** | 减少正则化 |
| Target modules | 2 个 | **5 个** | 2.5× |
| LR | 2e-5 | **3e-5** | 1.5× |
| **可训练参数** | **1.57M** | **14.16M** | **9×** |

### Target Modules 对比

| 旧配置 | 新配置 |
|--------|--------|
| query_proj | query_proj |
| value_proj | value_proj |
| — | key_proj |
| — | output_proj |
| — | dense |

## 状态

**状态**: 🔄 进行中
**PID**: 1871744
**开始时间**: 2026-06-20 14:30
**恢复点**: fold1_epoch3.pt (继续 fold 2)

## 进度追踪

| 时间 | Fold | Epoch | Step | Loss | ETA | 备注 |
|------|------|-------|------|------|-----|------|
| 14:32 | 2 | 1 | 1000/125309 | 0.66555 | 23014s | 优化配置启动 |

## 预期结果

- 可训练参数: 14.16M (比旧配置 1.57M 多 9×)
- Fold 2 Epoch 1: ~6.4h
- Fold 2 (3 epochs): ~19.2h
- Fold 3 (3 epochs): ~19.2h
- **总计: ~38h (2026-06-22 ~06:00)**

## 关键对比

| 模型 | 可训练参数 | Fold 1 Ep3 Val RMSE |
|------|-----------|---------------------|
| Base (86M) | 0.59M | 1.13857 |
| Large 旧配置 (435M) | 1.57M | 1.15910 |
| **Large 新配置 (435M)** | **14.16M** | **?** |

## 文件位置

- 训练脚本: `code/models/deberta_large_full.py`
- Checkpoints: `artifacts/models/checkpoints_large_full/`
- 训练日志: `artifacts/large_full_training.log`

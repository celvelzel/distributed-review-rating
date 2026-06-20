# DeBERTa-v3-large Training Progress (LR=2e-5)

## 实验目的

用 LR=2e-5 重新训练 DeBERTa-v3-large，验证是否能改善 val_rmse。

## 配置

| 参数 | 值 |
|------|-----|
| 脚本 | `code/models/deberta_large_full.py` |
| 数据 | 3M (train_tokens.npz) |
| 模型 | DeBERTa-v3-large + LoRA (r=16, alpha=32) |
| Folds | 3 |
| Epochs | 3 |
| BS/GradAcc | 16/16 (eff BS=256) |
| **LR** | **2e-5** (从 1e-5 提高) |
| Patience | 3 |

## 状态

**状态**: 🔄 进行中
**PID**: 1821809
**开始时间**: 2026-06-20 12:00
**恢复点**: fold1_epoch2.pt (继续 epoch 3)

## 进度追踪

| 时间 | Fold | Epoch | Step | Loss | ETA | 备注 |
|------|------|-------|------|------|-----|------|
| 12:02 | 1 | 3 | 1000/125309 | 0.37535 | 18550s | 从 fold1_epoch2 恢复 |

## 预期结果

- Fold 1 Epoch 3: ~5.2h
- Fold 2 (3 epochs): ~15.5h
- Fold 3 (3 epochs): ~15.5h
- **总计: ~36h (约 1.5 天)**

## 关键对比

| 模型 | LR | Val RMSE | Kaggle |
|------|-----|----------|--------|
| Base (86M) | 3e-5 | 1.137 | 0.681 |
| Large (435M) LR=1e-5 | 1e-5 | 1.160 | — |
| **Large (435M) LR=2e-5** | **2e-5** | **?** | **?** |

## 文件位置

- 训练脚本: `code/models/deberta_large_full.py`
- Checkpoints: `artifacts/models/checkpoints_large_full/`
- 训练日志: `artifacts/large_full_training.log`

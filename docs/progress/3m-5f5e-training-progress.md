# 3M 5f×5e Training Progress

## 实验目的

验证 5f×5e 训练配置是否是 Kaggle 0.617 的关键因素（而非数据量）。

## 配置

| 参数 | 值 |
|------|-----|
| 脚本 | `code/models/deberta_lora_3m_5f5e.py` |
| 数据 | 3M (train_tokens.npz) |
| 模型 | DeBERTa-v3-base + LoRA (r=16, alpha=32) |
| Folds | 5 |
| Epochs | 5 |
| BS/GradAcc | 16/16 (eff BS=256) |
| LR | 3e-5 |
| Patience | 3 |

## 状态

**状态**: 🔄 进行中
**PID**: 1766184 (restart after crash)
**开始时间**: 2026-06-19 12:26
**重启时间**: 2026-06-20 00:30 (from fold1_epoch3.pt)

## 进度追踪

| 时间 | Fold | Epoch | Step | Loss | ETA | 备注 |
|------|------|-------|------|------|-----|------|
| 12:28 | 1 | 1 | 1000/150371 | 0.70370 | 11691s | 开始训练 |
| 12:30 | 1 | 1 | 2000/150371 | 0.74820 | 11498s | 正常进行 |
| 12:32 | 1 | 1 | 3000/150371 | 0.72712 | 11383s | loss 下降 |
| 12:34 | 1 | 1 | 4000/150371 | 0.66595 | 11280s | 稳定训练中 |
| 22:52 | 1 | 3 | 150371/150371 | — | — | Fold 1 Ep3 完成 val_rmse=1.39582 |
| ~00:15 | — | — | — | — | — | 进程被杀 (OOM/scheduler)，从 fold1_epoch3.pt 恢复 |
| 00:30 | 1 | 4 | 11000/150371 | 0.69434 | 10704s | Fold 1 Ep4 进行中 |

## 关键发现

**Explore agent 调查结果**: 原始 `deberta_lora.py` 使用 `deberta-v3-small` (44M params) 而不是 `deberta-v3-base` (86M params)！

| 模型 | 实际模型 | Val RMSE | Kaggle |
|------|---------|----------|--------|
| 1M Old | deberta-v3-**small** (44M) | 1.117 | 0.617 |
| 3M 3f×3e | deberta-v3-**base** (86M) | 1.137 | 0.681 |
| 3M 5f×5e | deberta-v3-**base** (86M) | 1.396 | — |

当前训练使用的是正确的 base 模型。val_rmse 1.396 仍然异常高，原因待查。

## 预计完成时间

- Fold 1 Epoch 5: ~03:00
- Fold 1 完成: ~06:30
- 全部完成 (5 folds × ~6h): **2026-06-21 ~20:00**

## 预期结果

- 速度: ~206 samples/s (12.875 steps/s)
- 每 epoch: ~150371 steps / 12.875 = ~3.25 小时
- 每 fold: ~5 epochs × 3.25h = ~16.25 小时 (有 early stopping)
- 总计: ~5 folds × 16h = ~80 小时 (约 3.3 天)

## 关键对比

| 模型 | 数据 | 配置 | OOF | Kaggle |
|------|------|------|-----|--------|
| 1M Old | 1M | 5f×5e | 1.117 | 0.617 |
| 1M Fair | 1M | 3f×3e | 1.298 | 1.536 |
| 3M | 3M | 3f×3e | 1.137 | 0.681 |
| **3M 5f×5e** | **3M** | **5f×5e** | **?** | **?** |

## 文件位置

- 训练脚本: `code/models/deberta_lora_3m_5f5e.py`
- Checkpoints: `artifacts/models/checkpoints_lora_3m_5f5e/`
- OOF: `artifacts/models/deberta_lora_3m_5f5e_oof.npy`
- Test: `artifacts/models/deberta_lora_3m_5f5e_test.npy`
- 训练日志: `artifacts/lora_3m_5f5e_training.log`

# 3M 5f×5e Training Progress

**Date**: 2026-06-19

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

**状态**: ⏸️ 已暂停 (Oracle 分析确认是过拟合)
**PID**: 1766184 (已停止)
**开始时间**: 2026-06-19 12:26
**停止时间**: 2026-06-20 12:00 (Oracle 分析后手动停止)

## 进度追踪

| 时间 | Fold | Epoch | Step | Loss | ETA | 备注 |
|------|------|-------|------|------|-----|------|
| 12:28 | 1 | 1 | 1000/150371 | 0.70370 | 11691s | 开始训练 |
| 22:52 | 1 | 3 | 150371/150371 | — | — | Fold 1 Ep3 完成 val_rmse=1.39582 |
| ~00:15 | — | — | — | — | — | 进程被杀，从 fold1_epoch3.pt 恢复 |
| 00:30 | 1 | 4 | 11000/150371 | 0.69434 | 10704s | Fold 1 Ep4 进行中 |
| 12:00 | 2 | 2 | 20000/150371 | 0.47220 | 9859s | Fold 1 完成 (1.39578), Fold 2 Ep2 进行中 |
| 12:00 | — | — | — | — | — | **手动停止** (Oracle 确认过拟合) |

## 关键发现

### 1. Explore Agent: deberta_lora.py 用的是 small 模型

原始 `deberta_lora.py` 使用 `deberta-v3-small` (44M params) 而不是 `deberta-v3-base` (86M params)。

### 2. Oracle: val_rmse 1.396 是 LoRA 过拟合

| 差异 | deberta_lora | deberta_base_full |
|------|-------------|-------------------|
| Folds | 5 | 3 |
| Epochs | 5 | 3 |
| Batch Size | 16 | 32 |
| **总 optimizer steps** | **~47K** | **~23K** |
| KFold 实现 | numpy | sklearn |

**核心结论**: LoRA 只有 ~3.5M 可训练参数，47K steps 导致过拟合。23K steps 是正确的训练量。

### 3. 不是架构问题，是训练配置问题

两个脚本的模型架构完全相同（mean pooling → dropout → linear），但训练配置差异导致 0.26 RMSE 差距。

## 结论

- **停止 3M 5f×5e 训练** — 已确认是过拟合，继续训练无意义
- **`deberta_base_full.py` (3f×3e) 是正确的配置** — val_rmse 1.137
- **如需改进，应从其他方向入手**：更大模型、不同架构、特征工程

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

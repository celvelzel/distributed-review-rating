# 1M vs 3M 公平对比实验进度

**Date**: 2026-06-18

## 实验目的

公平对比 1M 和 3M 数据对 DeBERTa 模型性能的影响，使用完全相同的训练脚本和配置。

## 实验设计

| 项目 | 配置 |
|------|------|
| 脚本 | `code/models/deberta_base_1m_fair.py` (1M) / `code/models/deberta_base_full.py` (3M) |
| 数据 | `train_tokens_1m.npz` (1M) / `train_tokens.npz` (3M) |
| BATCH_SIZE | 32 |
| GRAD_ACCUM | 8 |
| LR | 3e-5 |
| N_FOLDS | 3 |
| N_EPOCHS | 3 |
| LoRA | r=16, alpha=32, dropout=0.05 |

## 实验结果

**状态**: ✅ 完成
**开始时间**: 2026-06-18 23:14
**完成时间**: 2026-06-19 03:57
**总耗时**: 5.5 小时

### 关键发现

| 模型 | 数据 | 配置 | OOF RMSE | Kaggle |
|------|------|------|----------|--------|
| 1M Old | 1M | 5f×5e | 1.117 | **0.617** |
| **1M Fair** | **1M** | **3f×3e** | **1.298** | **1.536** |
| 3M | 3M | 3f×3e | 1.137 | 0.681 |

### 结论

**假设被推翻**: 3M 数据并非有害！

1. 1M + 3f×3e 的 OOF (1.298) 远差于 3M + 3f×3e (1.137)
2. 1M + 3f×3e 的 Kaggle (1.536) 远差于 3M + 3f×3e (0.681)
3. 1M Old 的成功来自 **5f×5e 训练配置**，不是数据量

### 下一步

1. 用旧脚本 (5f×5e) 训练 3M 数据
2. 验证是否能改善 Kaggle 分数
3. 探索为什么 5f×5e 配置更优

## 进度追踪

| 时间 | Fold | Epoch | Step | Val RMSE | ETA | 备注 |
|------|------|-------|------|----------|-----|------|
| 22:00 | - | - | - | - | - | 开始训练 |
| 23:14 | 1 | 1 | 1000/20833 | - | 1705s | loss=0.88449 |
| 23:16 | 1 | 1 | 2000/20833 | - | 1613s | loss=0.76829 |
| 23:18 | 1 | 1 | 3000/20833 | - | 1527s | loss=0.75319 |
| 23:20 | 1 | 1 | 4000/20833 | - | 1441s | loss=0.66104 |
| 23:22 | 1 | 1 | 5000/20833 | - | 1355s | loss=0.54133 |
| 23:24 | 1 | 1 | 6000/20833 | - | 1269s | loss=0.48581 |
| 23:26 | 1 | 1 | 7000/20833 | - | 1183s | loss=0.56020 |
| 23:28 | 1 | 1 | 8000/20833 | - | 1098s | loss=0.60815 |
| 23:30 | 1 | 1 | 9000/20833 | - | 1012s | loss=0.53321 |

**当前状态**: Fold 1 Epoch 1 进行中，约 43% 完成
**预计完成时间**: Fold 1 Epoch 1 约 23:45，全部训练约 2026-06-19 10:00

## 预期结果

- 如果 1M Kaggle < 0.681 → 证明 3M 数据有害
- 如果 1M Kaggle ≈ 0.681 → 证明数据量无影响，差距来自训练配置

## 对比组

| 实验 | 数据 | 脚本 | Val RMSE | Kaggle | 状态 |
|------|------|------|----------|--------|------|
| 1M Fair | 1M | `deberta_base_1m_fair.py` | - | - | 🔄 进行中 |
| 3M Baseline | 3M | `deberta_base_full.py` | 1.137 | 0.681 | ✅ 完成 |
| 3M BS16 | 3M | `deberta_3m_bs16_ablation.py` | 1.157 | 0.743 | ✅ 完成 |

## 文件位置

- 训练脚本: `code/models/deberta_base_1m_fair.py`
- Checkpoints: `artifacts/models/checkpoints_base_1m_fair/`
- OOF: `artifacts/models/deberta_base_1m_fair_oof.npy`
- Test: `artifacts/models/deberta_base_1m_fair_test.npy`
- 训练日志: `artifacts/base_1m_fair_training.log`

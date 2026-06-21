# 当前最佳模型来源分析

_更新日期: 2026-06-21_

## 当前最佳 Kaggle 分数

**Kaggle RMSE: 0.59770**

## 模型来源

### 1. 基础模型: DeBERTa-v3-base (3M — 原始 checkpoint)

| 参数 | 值 |
|------|-----|
| 模型 | `microsoft/deberta-v3-base` |
| 参数量 | 86M |
| 训练数据 | **3M 样本 (完整数据集)** |
| 训练脚本 | **`code/models/deberta_lora.py` (原始版本, 5f×5e)** |
| 训练配置 | 5 折 × 5 epoch, BS=16, GradAcc=16, LR=3e-5 |
| LoRA 配置 | r=16, alpha=32, target=[query, value] |
| Val RMSE | 1.117 |
| Checkpoint | `artifacts/models/checkpoints_lora/fold1_epoch1.pt` |
| 预测文件 | `artifacts/models/deberta_lora_fold1_test.npy` (由 `predict_lora_fold1.py` 生成) |

> **勘误**: 此前报告标注为 "1M 模型"，实际 `deberta_lora_fold1_test.npy` 来自 `deberta_lora.py` 在完整 3M 数据上训练的 checkpoint (`checkpoints_lora/fold1_epoch1.pt`)。`deberta_lora_1m.py` 是后续创建的脚本，输出到不同的 checkpoint 目录 (`checkpoints_v3base_1m/`) 和文件名 (`deberta_v3base_1m_test.npy`)。

**脚本演变**: 原始 `deberta_lora.py` 使用 deberta-v3-base + 5f×5e + 3M 数据。后来被修改为 deberta-v3-small + 3f×3e (当前版本)。`deberta_lora_1m.py` 是基于此脚本创建的 1M 子采样变体 (3f×3e, BS=16, GradAcc=16)，但从未产生过可匹敌原始 checkpoint 的结果。

### 2. Stacking V3 元学习器

| 参数 | 值 |
|------|-----|
| 脚本 | `code/models/stacking_v3.py` |
| 基础模型数量 | 9 个 |
| 元学习器 | Ridge + LightGBM |
| 输出文件 | `artifacts/models/stacking_v3_ridge+lgb_test.npy` |

**包含的基础模型 (9 个)**:
1. lgb_tfidf (Text TF-IDF)
2. xgboost (Text TF-IDF)
3. mlp (DeBERTa embedding features)
4. lgb_safe_dense (Sentiment + Metadata)
5. xgboost_safe (Sentiment + Metadata)
6. catboost_safe (Sentiment + Metadata)
7. ensemble_diverse (Diverse ensemble)
8. xgb_graph_safe (Graph features)
9. lgb_graph_safe (Graph features)

> **注意**: DeBERTa 预测 **不是** Stacking V3 的基础模型。DeBERTa 仅在 post-stacking blend 中使用 (VE 60% + Stacking V3 40%)。

### 3. 方差扩展 (VE)

| 参数 | 值 |
|------|-----|
| 目标标准差 | 1.422 |
| 目标均值 | 3.941 |
| 预测标准差 | ~0.825 |
| 缩放因子 | ~1.72 |

**公式**:
```python
ve_pred = (pred - pred.mean()) / pred.std() * 1.422 + 3.941
ve_pred = np.clip(ve_pred, 1.0, 5.0)
```

### 4. 最终混合

| 成分 | 比例 | 说明 |
|------|------|------|
| VE 预测 | 60% | DeBERTa-v3-base **(3M)** 预测经 VE 校准 |
| Stacking V3 ridge+lgb | 40% | 9 模型集成的元学习器预测 |

**公式**:
```python
blend = 0.60 * ve_pred + 0.40 * stacking_v3_rlg
blend = np.clip(blend, 1.0, 5.0)
```

## 关键发现

### 1. VE 比例优化

| VE% | V3 rlg% | Kaggle RMSE |
|-----|---------|-------------|
| 90% | 10% | 0.61725 |
| 88% | 12% | 0.61473 |
| 85% | 15% | 0.61115 |
| 50% | 50% | 0.60073 |
| 55% | 45% | 0.59862 |
| **60%** | **40%** | **0.59770** |
| 30% | 70% | 0.62090 |

**结论**: Stacking V3 ridge+lgb 比 VE 更重要！

### 2. 模型对比

| 模型 | 数据 | 配置 | Val RMSE | Kaggle |
|------|------|------|----------|--------|
| **DeBERTa-v3-base (原始)** | **3M** | **5f×5e** | **1.117** | **0.617** |
| DeBERTa-v3-base (fair) | 1M | 3f×3e | 1.298 | 1.536 |
| DeBERTa-v3-base (full) | 3M | 3f×3e | 1.137 | 0.681 |
| DeBERTa-v3-large (3M) | 3M | 3f×3e | 1.160 | — |

> **勘误**: 此前报告将第一行标注为 "1M"，实际最佳 checkpoint 来自 3M 数据的 5f×5e 训练。1M Fair 实验 (相同 3f×3e 配置) 证明 1M 数据在同等配置下远不如 3M (Kaggle 1.536 vs 0.681)。1M Old 的成功来自 5f×5e 训练配置 (更多折数 + epoch → 更充分的 LR 退火)，不是数据量。

**结论**: **5f×5e 训练配置**是关键因素，而非 1M vs 3M 数据量差异。

### 3. VE 为什么有效

DeBERTa 预测严重压缩:
- 预测 std: 0.825
- 目标 std: 1.422
- 仅捕获 58% 方差

VE 通过线性缩放恢复正确的预测尺度。

## 文件位置

| 文件 | 路径 |
|------|------|
| 最佳提交 | `output/sub-deb1m-ve60-sv3rlg40.csv` |
| DeBERTa Checkpoint | `artifacts/models/checkpoints_lora/fold1_epoch1.pt` |
| DeBERTa 预测 | `artifacts/models/deberta_lora_fold1_test.npy` (由 `predict_lora_fold1.py` 生成) |
| 预测生成脚本 | `code/models/predict_lora_fold1.py` |
| Stacking V3 预测 | `artifacts/models/stacking_v3_ridge+lgb_test.npy` |
| 训练标签 | `artifacts/features/y_train.npy` (3M 完整标签) |
| 测试 Token | `artifacts/models/test_tokens.npz` |

## 下一步

1. **5f×5e 配置复用**: 用 5f×5e 配置重新训练 DeBERTa-v3-large (原始 checkpoint 证明 5f×5e 优于 3f×3e)
2. **DeBERTa-v3-large 优化**: 增加 LoRA 容量 (r=32, alpha=64, 5 target modules)
3. **更多模型集成**: 将 Large 模型预测加入 Stacking
4. **伪标签**: 用高置信度预测扩充训练集

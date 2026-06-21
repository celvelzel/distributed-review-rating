# 当前最佳模型来源分析

_更新日期: 2026-06-21_

## 当前最佳 Kaggle 分数

**Kaggle RMSE: 0.59770**

## 模型来源

### 1. 基础模型: DeBERTa-v3-base (1M)

| 参数 | 值 |
|------|-----|
| 模型 | `microsoft/deberta-v3-base` |
| 参数量 | 86M |
| 训练数据 | 1M 样本 (从 3M 中采样) |
| 训练脚本 | `code/models/deberta_lora_1m.py` |
| 训练配置 | 5 折 × 5 epoch, BS=16, GradAcc=16, LR=3e-5 |
| LoRA 配置 | r=16, alpha=32, target=[query_proj, value_proj] |
| Val RMSE | 1.117 |
| Checkpoint | `artifacts/models/deberta_lora_fold1_test.npy` |

**关键发现**: 旧脚本使用 `deberta-v3-small` (44M params)，但实际训练的是 `deberta-v3-base` (86M params)。

### 2. Stacking V3 元学习器

| 参数 | 值 |
|------|-----|
| 脚本 | `code/models/stacking_v3.py` |
| 基础模型数量 | 9 个 |
| 元学习器 | Ridge + LightGBM |
| 输出文件 | `artifacts/models/stacking_v3_ridge+lgb_test.npy` |

**包含的基础模型**:
1. DeBERTa-v3-base (1M, 3M)
2. DeBERTa-v3-small (500K)
3. LightGBM (多种配置)
4. XGBoost (多种配置)
5. CatBoost
6. Graph features models

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
| VE 预测 | 60% | DeBERTa-v3-base 预测经 VE 校准 |
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

| 模型 | Val RMSE | Kaggle |
|------|----------|--------|
| DeBERTa-v3-base (1M, 5f×5e) | 1.117 | 0.617 |
| DeBERTa-v3-base (3M, 3f×3e) | 1.137 | 0.681 |
| DeBERTa-v3-large (3M, 3f×3e) | 1.160 | — |

**结论**: 1M 模型 + 5f×5e 配置是最佳组合。

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
| DeBERTa 预测 | `artifacts/models/deberta_lora_fold1_test.npy` |
| Stacking V3 预测 | `artifacts/models/stacking_v3_ridge+lgb_test.npy` |
| 训练标签 | `artifacts/features/y_train_1m.npy` |
| 测试 ID | `artifacts/models/test_tokens.npz` |

## 下一步

1. **DeBERTa-v3-large 优化**: 增加 LoRA 容量 (r=32, alpha=64, 5 target modules)
2. **更多模型集成**: 将 Large 模型预测加入 Stacking
3. **伪标签**: 用高置信度预测扩充训练集

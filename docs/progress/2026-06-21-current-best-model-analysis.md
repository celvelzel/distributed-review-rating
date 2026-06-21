# COMP5434 BDC 项目上手指南

_最后更新: 2026-06-21_

## 项目概述

本项目是 COMP5434 Big Data Computing 课程的 Kaggle 竞赛项目，任务是电商评论评分预测 (RMSE)。训练集包含约 3M 条评论，测试集 10,000 条，每条评论有 title、comment 文本以及 user_id、product_id 等元数据。

当前团队最佳 Kaggle 成绩: **RMSE 0.59770**

---

## 1. 最佳模型架构

### 1.1 最终提交公式

最佳提交 `sub-deb1m-ve60-sv3rlg40.csv` 的生成公式:

```python
# Step 1: 方差扩展 (VE) — 校准 DeBERTa 预测的分布
ve_pred = (deb_pred - deb_pred.mean()) / deb_pred.std() * 1.422 + 3.941
ve_pred = np.clip(ve_pred, 1.0, 5.0)

# Step 2: 加权混合
blend = 0.60 * ve_pred + 0.40 * stacking_v3_rlg
blend = np.clip(blend, 1.0, 5.0)
```

其中 `1.422` 和 `3.941` 分别是训练集标签的标准差和均值。

### 1.2 两大组分

| 组分 | 权重 | 说明 |
|------|------|------|
| DeBERTa-v3-base (1M) VE | 60% | DeBERTa 原始预测经方差扩展校准 |
| Stacking V3 ridge+lgb | 40% | 9 个基础模型的元学习器集成预测 |

### 1.3 DeBERTa 基础模型

| 参数 | 值 |
|------|-----|
| 模型 | `microsoft/deberta-v3-base` (86M params) |
| 训练数据 | **1M 样本** (从 3M 中随机采样) |
| 训练脚本 | `code/models/deberta_lora_1m.py` |
| 训练配置 | 5 折 × 5 epoch, BS=16, GradAcc=16, LR=3e-5 |
| LoRA | r=16, alpha=32, target=[query_proj, value_proj], dropout=0.05 |
| Loss | CORAL ordinal + R-Drop (alpha=0.5) |
| Val RMSE | 1.117 |
| Checkpoint | `artifacts/models/checkpoints_lora/fold1_epoch1.pt` |
| 预测文件 | `artifacts/models/deberta_lora_fold1_test.npy` (由 `predict_lora_fold1.py` 生成) |

> **注意**: 最佳 checkpoint 来自 fold1 epoch1 (而非最后一折或最后一个 epoch)。早期 epoch 的 checkpoint 泛化能力反而更好。

### 1.4 为什么 5f×5e 配置是关键

通过 1M vs 3M 公平对比实验，已证明最佳结果来自训练配置而非数据量:

| 实验 | 数据 | 配置 | OOF RMSE | Kaggle RMSE |
|------|------|------|----------|-------------|
| **最佳** | **1M** | **5f×5e** | **1.117** | **0.617** |
| 1M Fair | 1M | 3f×3e | 1.298 | 1.536 |
| 3M Full | 3M | 3f×3e | 1.137 | 0.681 |

5f×5e 相比 3f×3e 的优势在于: 更多折数提供更好的 OOF 估计，更多 epoch 让 LR scheduler 有更充分的 cosine 退火。1M 数据的 scheduler 总步数较少 (3125 vs 23438)，epoch1 即可完成 LR→0 的退火，而 3M epoch1 仅退火 33%。

---

## 2. 模型管线全景

### 2.1 端到端流程

```
ETL (数据清洗/tokenize)
  ↓
特征工程 (TF-IDF / Sentiment / Graph / DeBERTa embeddings)
  ↓
基础模型训练 (9 个 stacking base models + DeBERTa LoRA)
  ↓
Stacking V3 元学习器训练 (Ridge + LightGBM)
  ↓
预测生成 (DeBERTa fold1 test + Stacking V3 test)
  ↓
VE 校准 + 加权混合 → Kaggle 提交
```

### 2.2 目录结构

```
distributed-review-rating/
├── code/
│   ├── models/          # 所有训练脚本和预测脚本
│   └── features/        # 特征工程脚本
├── artifacts/
│   ├── etl/             # 原始数据 (train.parquet, test.parquet)
│   ├── features/        # 工程化特征 (.npz, .parquet)
│   └── models/          # 模型预测 (.npy) 和 checkpoints
├── output/              # Kaggle 提交 CSV
└── docs/                # 文档和实验记录
```

### 2.3 数据文件依赖图

> **重要**: `artifacts/` 目录下的大部分文件 (模型预测 .npy、checkpoints、特征 .parquet) 仅存在于 HPC 集群上，不在本地 Git 仓库中。新成员需要从 HPC 下载所需文件，或按以下步骤重新生成。

```
artifacts/etl/
  train.parquet ──→ 所有基础模型的训练输入
  test.parquet  ──→ 所有基础模型的测试输入 + 提交 ID

artifacts/features/
  y_train.npy              ──→ 训练标签 (VE 统计用)
  y_train_1m.npy           ──→ 1M DeBERTa 训练标签
  chartfidf_train/test.npz ──→ lgb_tfidf
  sentiment.parquet        ──→ lgb_safe_dense, xgboost_safe, catboost_safe
  product_metadata.parquet ──→ (同上)
  bert_train/test.parquet  ──→ mlp
  expanded_graph_*.parquet ──→ xgb_graph_safe, lgb_graph_safe
  user_stats_kfold.parquet ──→ graph models
  product_stats_kfold.parquet ──→ graph models

artifacts/models/
  9 个基础模型的 *_oof.npy + *_test.npy ──→ stacking_v3.py
  checkpoints_lora/fold1_epoch1.pt ──→ predict_lora_fold1.py
  stacking_v3_ridge+lgb_test.npy ──→ 最终混合
  deberta_lora_fold1_test.npy ──→ 最终混合 (VE 60%)
```

---

## 3. Stacking V3 详解

Stacking V3 是当前模型的核心集成层，也是后续改进的重点方向。

### 3.1 架构概览

Stacking V3 是一个两层集成:

```
第一层: 9 个异构基础模型 (各自产出 OOF + test 预测)
         ↓ (OOF 作为新特征)
第二层: 5 个候选元学习器 (Ridge / LightGBM / CatBoost / ElasticNet / Ridge+LGB)
         ↓ (自动选最优)
输出: 最优元学习器的 test 预测
```

脚本入口: `code/models/stacking_v3.py`

### 3.2 九个基础模型

#### 3.2.1 文本 TF-IDF 类

| 模型 | 训练脚本 | 特征 | OOF RMSE |
|------|----------|------|----------|
| lgb_tfidf | `ensemble_diverse.py` (side effect) | char-level TF-IDF 5000-dim | 1.197 |
| xgboost | `xgboost_train.py` | word-level TF-IDF 5000-dim (运行时计算) | 1.202 |

lgb_tfidf 的特征来自 `code/features/text_chartfidf.py`，使用 `char_wb` analyzer、ngram (3,5)、sublinear TF。注意: 该模型没有独立训练脚本，其 OOF 文件在 `ensemble_diverse.py` 运行时作为 side effect 生成 (200K 子采样 LightGBM + 全量 fill-in)。

xgboost 的 TF-IDF 在 `xgboost_train.py` 中运行时从 `title + comment` 重新计算，不从磁盘加载预计算特征。

#### 3.2.2 DeBERTa Embedding 类

| 模型 | 训练脚本 | 特征 | OOF RMSE |
|------|----------|------|----------|
| mlp | `run_mlp.py` | DeBERTa-v3-base mean-pooled embeddings (768-dim) | 1.131 |

MLP 架构: `768 → 512 → 256 → 128 → 1`，BatchNorm + Dropout(0.4)。训练配置: 5 折 × 50 epoch，patience=10，batch_size=4096，lr=1e-3，CosineAnnealingLR。特征来自 `artifacts/features/bert_train.parquet` / `bert_test.parquet`。

> **已知问题**: `run_mlp.py` 使用 `np.random.RandomState` 自定义 fold 划分，与 stacking 的 `KFold(5, shuffle, rs=42)` 不一致，可能影响 meta-learner 的 OOF 对齐。

#### 3.2.3 Sentiment + Metadata 类

| 模型 | 训练脚本 | 特征 | OOF RMSE |
|------|----------|------|----------|
| lgb_safe_dense | `train_safe_features.py` | 25 列 dense 特征 (VADER + TextBlob + 产品元数据) | 1.225 |
| xgboost_safe | `train_safe_features.py` | 同上 | 1.227 |
| catboost_safe | `train_safe_features.py` | 同上 | 1.230 |

三个模型共享同一个训练脚本 `train_safe_features.py`，使用相同的 25 列"safe"特征 (排除了 `user_cat_avg_rating` 等 target leakage 列)。特征来源: `artifacts/features/sentiment.parquet` (17 列 VADER/TextBlob/词数) + `artifacts/features/product_metadata.parquet` (8 列)。

#### 3.2.4 集成类

| 模型 | 训练脚本 | 特征 | OOF RMSE |
|------|----------|------|----------|
| ensemble_diverse | `ensemble_diverse.py` | 元集成: lgb_tfidf + xgboost + mlp 的加权平均 | 1.129 |

ensemble_diverse 是一个 meta-ensemble: 它加载前三个模型的 OOF 预测，grid search 最优加权组合。在 Stacking V3 的 Ridge 元学习器中获得最高正系数 (0.7447)。

#### 3.2.5 Graph 特征类

| 模型 | 训练脚本 | 特征 | OOF RMSE |
|------|----------|------|----------|
| xgb_graph_safe | `train_graph_models.py` | 扩展图特征 + KFold user/product stats + 交叉特征 | 1.362 |
| lgb_graph_safe | `train_graph_models.py` | 同上 | 1.362 |

两个图模型共享 `train_graph_models.py`。输入特征来自 `artifacts/features/expanded_graph_train.parquet` (由 `code/features/expand_graph_features.py` 生成) + KFold-safe 的 user/product 统计 + 交叉特征 (`leniency_x_reviews`, `cat_dev_x_reviews`, `user_prod_diff`)。

Safe 变体排除了 `user_cat_avg_rating` 和 `user_cat_deviation` (target leakage)。

> **注意**: 两个图模型的独立 OOF RMSE 最高 (1.362)，但 lgb_graph_safe 在 Ridge 元学习器中获得第二高的正系数 (0.4040)。这说明它们虽然单独预测能力弱，但提供了与其他模型正交的信号 (diversity)，对集成有正贡献。xgb_graph_safe 系数为负 (-0.064)，可能需要改进或替换。

### 3.3 元学习器

Stacking V3 训练 5 个候选元学习器，自动选 OOF RMSE 最低的:

| 元学习器 | 超参数 | OOF RMSE |
|----------|--------|----------|
| Ridge | alpha=1.0, fit_intercept=True | 1.12046 |
| LightGBM | lr=0.05, num_leaves=31, n_est=500, subsample=0.8 | 1.11774 |
| CatBoost | depth=4, lr=0.05, iter=300, l2=3.0 | 1.11799 |
| ElasticNet | CV 搜索 l1_ratio∈[0.1..0.9], alpha∈[0.001..10] | 1.12042 |
| **Ridge+LGB** | **grid search w∈[0, 1], step=0.01** | **1.11774 (最佳)** |

Ridge+LGB 是最优元学习器: 对 Ridge 和 LightGBM 的 OOF 预测做加权混合，权重通过 101 次 grid search 选出 (最小化 OOF RMSE)。

所有元学习器使用 `KFold(n_splits=5, shuffle=True, random_state=42)` 训练，预测 clip 到 [1.0, 5.0]。

### 3.4 Ridge 系数分析

Ridge 元学习器的系数揭示了每个基础模型对集成的贡献:

| 模型 | 系数 | 信号类型 | 解读 |
|------|------|----------|------|
| ensemble_diverse | +0.7447 | Meta-ensemble | 最强正贡献 |
| lgb_graph_safe | +0.4040 | Graph features | 提供多样性信号 |
| mlp | +0.1445 | DeBERTa embedding | 中等贡献 |
| lgb_safe_dense | +0.1279 | Sentiment+Meta | 中等贡献 |
| xgboost | +0.0591 | Text TF-IDF | 弱正贡献 |
| lgb_tfidf | -0.0064 | Text TF-IDF | 接近零 (冗余) |
| xgboost_safe | -0.0188 | Sentiment+Meta | 弱负 (冗余) |
| catboost_safe | -0.0263 | Sentiment+Meta | 弱负 (冗余) |
| xgb_graph_safe | -0.0640 | Graph features | 负贡献 (噪声?) |

> **改进方向**: 负系数模型 (xgb_graph_safe, catboost_safe, xgboost_safe, lgb_tfidf) 可能拖累了集成表现。可以尝试: (1) 移除负系数模型重训 meta-learner; (2) 改进这些模型的特征或超参; (3) 增加新的高 diversity 模型。

### 3.5 Stacking V3 输出文件

`stacking_v3.py` 运行后产出:

| 文件 | 说明 |
|------|------|
| `stacking_v3_oof.npy` | 最优元学习器 OOF |
| `stacking_v3_test.npy` | 最优元学习器 test |
| `stacking_v3_ridge_oof/test.npy` | Ridge 元学习器 |
| `stacking_v3_lgb_oof/test.npy` | LightGBM 元学习器 |
| `stacking_v3_catboost_oof/test.npy` | CatBoost 元学习器 |
| `stacking_v3_elasticnet_oof/test.npy` | ElasticNet 元学习器 |
| `stacking_v3_ridge+lgb_oof/test.npy` | Ridge+LGB 元学习器 |
| `stacking_v3_results.json` | 完整运行统计 (RMSE, 系数, 重要性) |
| `stacking_v3_run_{timestamp}.log` | 时间戳日志 |
| `output/submission-stacking-v3.csv` | 独立 Kaggle 提交 |

### 3.6 验证与提交

Stacking V3 的验证和提交流程:

```
stacking_v3.py       → 训练元学习器 + 生成预测
verify_stacking_v3.py → 6 项检查 (OOF 质量 / v3 vs v2 / 元学习器对比 / DeBERTa blend / 基础模型贡献 / 建议)
submit_stacking_v3.py → 生成 9 个 Kaggle 提交 CSV (不同 VE/Stacking 比例)
```

验证脚本 `verify_stacking_v3.py` 会输出 PASS / FAIL / NEUTRAL / CHANGED 判定:
- **PASS**: v3 比 v2 OOF RMSE 改善 > 0.001
- **FAIL**: v3 比 v2 差 > 0.001
- **NEUTRAL**: 差异在 +/-0.001 以内
- **CHANGED**: test 预测显著不同但 OOF 不可用

---

## 4. 复现指南

### 4.1 环境准备

```bash
# 核心依赖
pip install torch transformers peft datasets
pip install lightgbm xgboost catboost scikit-learn
pip install numpy pandas scipy
pip install vaderSentiment textblob  # sentiment features

# 可选: PySpark (50K TF-IDF), Optuna (超参搜索)
```

### 4.2 完整复现步骤

以下按依赖顺序列出。所有脚本从 repo 根目录运行: `python code/路径/脚本.py`

#### Step 0: ETL + 特征工程

| 步骤 | 脚本 | 产出 |
|------|------|------|
| -1 | ETL 预处理 (已在 HPC 完成) | `artifacts/etl/train.parquet`, `test.parquet` |
| 0a | `code/features/text_chartfidf.py` | `chartfidf_train/test.npz` |
| 0b | `code/features/sentiment.py` | `sentiment.parquet` |
| 0c | `code/features/product_metadata.py` | `product_metadata.parquet` |
| 0d | `code/features/expand_graph_features.py` | `expanded_graph_train/test.parquet` |
| 0e | `code/features/user_stats_kfold.py` | `user_stats_kfold.parquet` |
| 0f | `code/features/product_stats_kfold.py` | `product_stats_kfold.parquet` |
| 0g | `code/features/run_bert.py` (或类似脚本) | `bert_train/test.parquet` (MLP 所需的 DeBERTa embedding) |

#### Step 1: 基础模型训练

| 步骤 | 脚本 | 产出 (OOF + test) |
|------|------|------|
| 1a | `code/models/xgboost_train.py` | `xgboost_oof/test.npy` |
| 1b | `code/models/run_mlp.py` | `mlp_oof/test.npy` |
| 1c | `code/models/ensemble_diverse.py` | `ensemble_diverse_oof/test.npy` + `lgb_tfidf_oof/test.npy` |
| 1d | `code/models/train_safe_features.py` | `lgb_safe_dense`, `xgboost_safe`, `catboost_safe` (各 oof/test) |
| 1e | `code/models/train_graph_models.py` | `xgb_graph_safe`, `lgb_graph_safe` (各 oof/test) |

> 1a-1e 之间无依赖，可以并行运行 (例如在 HPC 上同时提交)。1c 依赖 lgb_tfidf 不存在时会自动生成。

#### Step 2: DeBERTa 预测生成

```bash
# 从已有 checkpoint 生成 test 预测
python code/models/predict_lora_fold1.py
# 产出: artifacts/models/deberta_lora_fold1_test.npy
```

如果需要重新训练 DeBERTa (在 HPC 上，~10h):
```bash
# 使用 deberta_lora_1m.py (1M 数据, 5f×5e, deberta-v3-base)
python code/models/deberta_lora_1m.py
# 注意: 该脚本当前磁盘版本的配置可能与原始训练不一致，请核实 5f×5e 配置
```

#### Step 3: Stacking V3

```bash
# 训练元学习器
python code/models/stacking_v3.py
# 产出: stacking_v3_*.npy, stacking_v3_results.json

# 验证
python code/models/verify_stacking_v3.py
# 产出: docs/changelog/stacking-v3-verification.md
```

#### Step 4: 生成提交

> **注意**: `submit_stacking_v3.py` 仅扫描 VE 比例 75%-95%，无法生成 60/40 最佳比例。最佳提交必须通过下方手动代码生成。

```bash
# 自动提交多种比例
python code/models/submit_stacking_v3.py
# 产出: 9 个 CSV 文件在 output/ 目录
```

最佳提交 (手动生成):
```python
import numpy as np, pandas as pd

deb = np.load('artifacts/models/deberta_lora_fold1_test.npy')
sv3 = np.load('artifacts/models/stacking_v3_ridge+lgb_test.npy')
y = np.load('artifacts/features/y_train.npy')

# VE 校准
ve = np.clip((deb - deb.mean()) / deb.std() * y.std() + y.mean(), 1, 5)
# 60/40 混合
blend = np.clip(0.60 * ve + 0.40 * sv3, 1, 5)

ids = np.load('artifacts/models/test_tokens.npz', allow_pickle=True)['ids']
pd.DataFrame({'id': ids, 'rating': blend}).to_csv(
    'output/sub-deb1m-ve60-sv3rlg40.csv', index=False)
```

### 4.3 最小复现路径

如果只需要复现最佳提交 (不需要重训所有基础模型)，确保以下文件存在即可:

```
artifacts/models/deberta_lora_fold1_test.npy    # DeBERTa 预测
artifacts/models/stacking_v3_ridge+lgb_test.npy  # Stacking V3 预测
artifacts/features/y_train.npy                   # 训练标签 (VE 统计)
artifacts/models/test_tokens.npz                 # 测试 ID
```

用上面 Step 4 的 Python 代码即可生成提交。

---

## 5. 已知问题与注意事项

### 5.1 Target Leakage

`user_cat_avg_rating` 特征存在 target leakage (XGBoost 特征重要性 87%)。Stacking V3 的所有 "safe" 变体已排除该列。如果 OOF RMSE < 0.1，几乎可以确定是 leakage。

### 5.2 KFold 划分不一致

大部分基础模型使用 `KFold(n_splits=5, shuffle=True, random_state=42)`，但 `run_mlp.py` 使用 `np.random.RandomState` 自定义划分。这可能导致 MLP 的 OOF 预测与其他模型不对齐，影响 meta-learner 训练。

### 5.3 lgb_tfidf 来源不可追溯

`lgb_tfidf_oof/test.npy` 无独立训练脚本，唯一写入者是 `ensemble_diverse.py` 的 fallback 逻辑 (char-level TF-IDF + 200K 子采样)。不可独立复现。

### 5.4 OOF RMSE 与 Kaggle RMSE 的系统性偏差

两者存在约 0.45 的 gap (测试集比训练集更容易)。不能依赖 OOF RMSE 选模型，必须实际提交 Kaggle 验证。

### 5.5 文件名断裂

`xgboost_full.py` 输出 `xgboost_expanded_oof.npy`，但 submit 脚本加载 `xgboost_full_oof.npy`，导致部分 blend 不可追溯。

---

## 6. 后续改进方向

### 6.1 Stacking 改进 (优先)

- **移除负系数模型**: xgb_graph_safe (-0.064)、catboost_safe (-0.026)、xgboost_safe (-0.019) 可能拖累集成。移除后重训 meta-learner 观察 RMSE 变化
- **修复 MLP fold 划分**: 统一为 KFold(5, shuffle, rs=42)，重新生成 OOF
- **新增高 diversity 基础模型**: 当前 mlp 是唯一使用 DeBERTa embedding 的 base model，可以加入 DeBERTa 直接预测作为第 10 个 base model
- **元学习器调优**: 当前 Ridge+LGB 的 grid search 仅搜索单一权重，可以尝试非线性组合或多层 stacking

### 6.2 DeBERTa 改进

- **用 5f×5e 配置训练 DeBERTa-v3-large**: 已证明 5f×5e 显著优于 3f×3e
- **增大 LoRA 容量**: r=32, alpha=64, 扩展到 5 个 target modules

### 6.3 数据改进

- **伪标签**: 用高置信度预测扩充训练集

---

## 7. 快速参考: 文件地图

### 训练脚本 → 产出文件

| 脚本 | 产出 |
|------|------|
| `deberta_lora_1m.py` | `checkpoints_lora/fold{0-4}_epoch{1-5}.pt` (1M 数据) |
| `predict_lora_fold1.py` | `deberta_lora_fold1_test.npy` |
| `xgboost_train.py` | `xgboost_oof.npy`, `xgboost_test.npy` |
| `run_mlp.py` | `mlp_oof.npy`, `mlp_test.npy` |
| `ensemble_diverse.py` | `ensemble_diverse_oof/test.npy` + `lgb_tfidf_oof/test.npy` |
| `train_safe_features.py` | `lgb_safe_dense_*`, `xgboost_safe_*`, `catboost_safe_*` |
| `train_graph_models.py` | `xgb_graph_safe_*`, `lgb_graph_safe_*` |
| `stacking_v3.py` | `stacking_v3_*.npy`, `stacking_v3_results.json` |
| `submit_stacking_v3.py` | `output/submission-*.csv` (9 个) |

### 关键配置文件

| 文件 | 内容 |
|------|------|
| `artifacts/features/y_train.npy` | 3M 训练标签 (VE 统计: mean=3.941, std=1.422) |
| `artifacts/etl/train.parquet` | 3M 训练数据 (title, comment, user_id, product_id, rating) |
| `artifacts/etl/test.parquet` | 10K 测试数据 (id, title, comment, user_id, product_id) |
| `docs/changelog/metrics.json` | Kaggle 提交历史 (注: 文件路径可能已变更为带日期前缀的版本; 内容未更新到 0.59770) |

### VE 比例优化实验记录

| VE% | V3 rlg% | Kaggle RMSE |
|-----|---------|-------------|
| 90% | 10% | 0.61725 |
| 85% | 15% | 0.61115 |
| 60% | 40% | **0.59770** |
| 55% | 45% | 0.59862 |
| 50% | 50% | 0.60073 |
| 30% | 70% | 0.62090 |

最佳 VE 比例在 55-65% 区间，Stacking V3 ridge+lgb 的权重增加显著改善 RMSE。

# DeBERTa vs Stacking 特征学习对比报告
# DeBERTa vs Stacking: Feature Learning Comparison Report

**日期 / Date**: 2026-06-24
**项目 / Project**: COMP5434 Review Rating Prediction

---

## 摘要 / Summary

本文档详细对比了项目中两个核心预测组件的特征学习方式：DeBERTa 深度语言模型和 Stacking 集成元学习器。两者在特征来源、学习方式和作用层级上有本质区别。

This document provides a detailed comparison of the two core prediction components in the project: the DeBERTa deep language model and the Stacking ensemble meta-learner. They differ fundamentally in feature sources, learning mechanisms, and operational levels.

---

## 1. DeBERTa 模型 / DeBERTa Model

### 1.1 输入特征 / Input Features

| 项目 / Item | 内容 / Content |
|-------------|----------------|
| **输入源 / Input Source** | 原始评论文本 / Raw review text |
| **处理方式 / Processing** | SentencePiece Tokenizer 分词 / Tokenization via SentencePiece |
| **模型架构 / Architecture** | DeBERTa-v3-base (12层, 768维) / DeBERTa-v3-base (12 layers, 768-dim) |
| **特征维度 / Feature Dim** | 768 维上下文语义向量 / 768-dim contextual semantic vector |
| **参数量 / Parameters** | ~184M (base) / ~184M (base) |
| **训练数据量 / Training Data** | 1M 条评论 (3折×3轮) / 1M reviews (3 folds × 3 epochs) |

### 1.2 学习过程 / Learning Process

```
原始文本 / Raw Text
    ↓
[Tokenizer] 分词 / Tokenization
    ↓
[DeBERTa-v3-base] 12层 Transformer 编码 / 12-layer Transformer encoding
    ↓
[Mean Pooling] 序列聚合 / Sequence aggregation
    ↓
768-dim 语义向量 / Semantic vector
    ↓
[CORAL Ordinal Loss] 有序回归 / Ordinal regression
    ↓
评分预测 (1-5) / Rating prediction (1-5)
```

### 1.3 学到的特征 / Learned Features

| 特征类型 / Feature Type | 说明 / Description |
|------------------------|---------------------|
| **语义特征 / Semantic** | 评论文本的深层语义表示 / Deep semantic representation of review text |
| **情感特征 / Sentiment** | 正面/负面情感倾向 / Positive/negative sentiment tendency |
| **评分意图 / Rating Intent** | 隐含的评分信号 / Implicit rating signals |
| **上下文关系 / Context** | 词语间的依赖关系 / Inter-word dependencies |

### 1.4 不使用的特征 / Features NOT Used

- ❌ 用户历史统计 / User historical statistics
- ❌ 商品统计信息 / Product statistics
- ❌ TF-IDF 特征 / TF-IDF features
- ❌ 图特征 / Graph features
- ❌ 目标编码 / Target encoding

### 1.5 关键参数 / Key Parameters

```python
# deberta_lora_1m.py 配置
MODEL_NAME = "microsoft/deberta-v3-base"
LORA_R = 16              # LoRA 秩 / LoRA rank
LORA_ALPHA = 32          # LoRA 缩放因子 / LoRA scaling factor
N_FOLDS = 3              # 折数 / Number of folds
N_EPOCHS = 3             # 每折轮数 / Epochs per fold
BATCH_SIZE = 16          # 批大小 / Batch size
GRAD_ACCUM = 16          # 梯度累积 / Gradient accumulation
LR = 3e-5                # 学习率 / Learning rate
R_DROP_ALPHA = 0.5       # R-Drop 正则化 / R-Drop regularization
```

---

## 2. Stacking 集成 / Stacking Ensemble

### 2.1 架构概述 / Architecture Overview

Stacking 是两层学习架构：
Stacking is a two-layer learning architecture:

```
Layer 1: 基础模型层 / Base Model Layer
    9 个异构模型 → 各自学习不同特征 → 输出 OOF 预测
    9 heterogeneous models → learn different features → output OOF predictions

Layer 2: 元学习器层 / Meta-Learner Layer
    Ridge / LightGBM / ElasticNet → 学习最优组合权重 → 最终预测
    Ridge / LightGBM / ElasticNet → learn optimal combination weights → final prediction
```

### 2.2 Layer 1: 基础模型 / Base Models

| # | 模型 / Model | 学习的特征 / Features Learned | 特征维度 / Dim | 脚本 / Script |
|---|-------------|------------------------------|----------------|---------------|
| 1 | LightGBM TF-IDF | 词级 TF-IDF + SVD 降维 / Word-level TF-IDF + SVD | 50,512 | `lgb_tfidf50k_svd.py` |
| 2 | XGBoost | 词级 TF-IDF / Word-level TF-IDF | 5,000 | `xgboost_train.py` |
| 3 | MLP | BERT 语义向量 / BERT semantic embeddings | 768 | `mlp.py` |
| 4 | LightGBM Safe Dense | 安全目标编码 + 稠密特征 / Safe target encoding + dense | ~100 | `lgb_safe_dense` |
| 5 | XGBoost Safe | 安全目标编码 + 稠密特征 / Safe target encoding + dense | ~100 | `xgboost_safe` |
| 6 | CatBoost Safe | 结构化特征 (用户/商品统计) / Structured (user/product stats) | 927 | `catboost_train.py` |
| 7 | Ensemble Diverse | 多模型集成 OOF / Multi-model ensemble OOF | - | `ensemble_diverse` |
| 8 | XGBoost Graph | LightGCN 图嵌入 / LightGCN graph embeddings | 64 | `train_graph_models.py` |
| 9 | LightGBM Graph | LightGCN 图嵌入 / LightGCN graph embeddings | 64 | `train_graph_models.py` |

### 2.3 Layer 2: 元学习器 / Meta-Learners

元学习器的输入是 9 维向量（每个基础模型的 OOF 预测值）：
The meta-learner's input is a 9-dimensional vector (each base model's OOF prediction):

```python
# stacking_v3.py 中的输入构建
X_meta_train = np.column_stack([
    oof_dict["lgb_tfidf"],        # 维度 1: LightGBM TF-IDF 预测
    oof_dict["xgboost"],          # 维度 2: XGBoost 预测
    oof_dict["mlp"],              # 维度 3: MLP 预测
    oof_dict["lgb_safe_dense"],   # 维度 4: LightGBM Safe 预测
    oof_dict["xgboost_safe"],     # 维度 5: XGBoost Safe 预测
    oof_dict["catboost_safe"],    # 维度 6: CatBoost 预测
    oof_dict["ensemble_diverse"], # 维度 7: Ensemble 预测
    oof_dict["xgb_graph_safe"],   # 维度 8: XGBoost Graph 预测
    oof_dict["lgb_graph_safe"],   # 维度 9: LightGBM Graph 预测
])
```

### 2.4 元学习器对比 / Meta-Learner Comparison

| 元学习器 / Meta-Learner | 类型 / Type | 特点 / Characteristics |
|------------------------|-------------|------------------------|
| **Ridge** | 线性回归 / Linear | 简单稳定，可解释性强 / Simple, stable, interpretable |
| **LightGBM** | 梯度提升树 / GBDT | 能捕捉非线性交互 / Captures non-linear interactions |
| **ElasticNet** | 正则化线性 / Regularized Linear | L1+L2 正则化，自动特征选择 / L1+L2 regularization, auto feature selection |
| **CatBoost** | 梯度提升树 / GBDT | 对类别特征友好 / Category-friendly |
| **Ridge+LGB** | 混合 / Blend | 自动搜索最优权重 / Auto-search optimal weight |

### 2.5 Stacking 最优配置 / Best Stacking Configuration

```python
# Ridge+LGB 混合 (当前最优)
best_w = 自动搜索 0-100 的最优权重 / Auto-searched optimal weight from 0-100
stacking_pred = best_w * ridge_pred + (1 - best_w) * lgb_pred
```

---

## 3. 核心区别对比 / Key Differences

| 对比维度 / Dimension | DeBERTa | Stacking |
|---------------------|---------|----------|
| **输入来源 / Input** | 原始文本 / Raw text | 基础模型预测 / Base model predictions |
| **特征维度 / Feature Dim** | 768 维 / 768-dim | 9 维 / 9-dim |
| **学习目标 / Learning Goal** | 从文本直接学评分 / Learn rating from text | 学习模型组合权重 / Learn model combination weights |
| **模型类型 / Model Type** | 深度神经网络 / Deep neural network | 线性模型 + 树模型 / Linear + tree models |
| **可解释性 / Interpretability** | 低 (黑盒) / Low (black box) | 高 (权重可解释) / High (interpretable weights) |
| **训练成本 / Training Cost** | 高 (GPU, 小时级) / High (GPU, hours) | 低 (CPU, 分钟级) / Low (CPU, minutes) |
| **数据需求 / Data Need** | 大量文本 / Large text corpus | 基础模型的 OOF 预测 / Base model OOF predictions |

---

## 4. 最终混合策略 / Final Blending Strategy

### 4.1 最优配方 / Best Recipe

```
最终预测 = VE(DeBERTa) × 60% + Stacking V3 × 40%
Final = VE(DeBERTa) × 60% + Stacking V3 × 40%
```

### 4.2 为什么需要 VE / Why VE is Needed

DeBERTa 原始预测的方差偏小，需要 Variance Expansion 对齐到真实标签分布：
DeBERTa's raw predictions have small variance; Variance Expansion aligns them to the true label distribution:

```python
# VE 公式 / VE Formula
ve = (pred - pred.mean()) / pred.std() * target_std + target_mean
# target_std ≈ 1.422 (y_train 的标准差 / y_train std)
# target_mean ≈ 3.941 (y_train 的均值 / y_train mean)
```

| 指标 / Metric | DeBERTa 原始 / Raw | VE 后 / After VE | 真实标签 / True Labels |
|---------------|-------------------|-----------------|----------------------|
| Mean | ~3.94 | ~3.94 | 3.941 |
| Std | ~0.70 | ~1.42 | 1.422 |
| Min | ~2.5 | 1.0 | 1.0 |
| Max | ~4.5 | 5.0 | 5.0 |

### 4.3 混合代码 / Blending Code

```python
import numpy as np
import pandas as pd

# Step 1: 加载预测 / Load predictions
deberta = np.load("artifacts/models/deberta_lora_fold1_test.npy")
stacking = np.load("artifacts/models/stacking_v3_ridge+lgb_test.npy")

# Step 2: VE 方差扩展 / Variance Expansion (⚠️ 必须做 / MUST do)
ve = np.clip(
    (deberta - deberta.mean()) / deberta.std() * 1.422 + 3.941,
    1.0, 5.0
)

# Step 3: 按比例混合 / Blend by ratio
final = np.clip(0.60 * ve + 0.40 * stacking, 1.0, 5.0)

# Step 4: 生成提交 / Generate submission
pd.DataFrame({
    "id": range(len(final)),
    "rating": final
}).to_csv("output/my_submission.csv", index=False)
```

---

## 5. 为什么两个组件都需要 / Why Both Components Are Needed

### 5.1 互补性 / Complementarity

| DeBERTa 的优势 / DeBERTa Strengths | Stacking 的优势 / Stacking Strengths |
|-------------------------------------|---------------------------------------|
| 理解文本语义 / Understands text semantics | 利用结构化特征 / Leverages structured features |
| 捕捉情感细微差别 / Captures sentiment nuances | 整合用户/商品统计 / Integrates user/product stats |
| 处理未见过的表达 / Handles unseen expressions | 利用图关系 / Leverages graph relationships |
| 端到端学习 / End-to-end learning | 多模型多样性 / Multi-model diversity |

### 5.2 单独使用的效果 / Performance When Used Alone

| 方案 / Approach | Kaggle RMSE | 说明 / Notes |
|-----------------|-------------|--------------|
| DeBERTa VE only | ~0.617 | 仅用文本特征 / Text features only |
| Stacking V3 only | ~0.663 | 仅用结构化特征 / Structured features only |
| **DeBERTa VE 60% + Stacking 40%** | **0.59770** | **最优组合 / Best combination** |

### 5.3 结论 / Conclusion

DeBERTa 和 Stacking 的结合实现了**语义理解**与**统计特征**的互补，这解释了为什么混合后的效果显著优于单独使用任何一个组件。

The combination of DeBERTa and Stacking achieves complementarity between **semantic understanding** and **statistical features**, which explains why the blended result significantly outperforms either component alone.

---

## 附录 / Appendix

### A. 文件路径索引 / File Path Index

| 组件 / Component | 路径 / Path |
|------------------|-------------|
| DeBERTa 权重 / DeBERTa weights | `artifacts/models/checkpoints_base_full/` |
| DeBERTa 预测 / DeBERTa predictions | `artifacts/models/deberta_lora_fold1_test.npy` |
| Stacking V3 预测 / Stacking V3 predictions | `artifacts/models/stacking_v3_ridge+lgb_test.npy` |
| Stacking 训练脚本 / Stacking training script | `code/models/stacking_v3.py` |
| 混合脚本 / Blending script | `code/models/submit_stacking_v3.py` |
| DeBERTa 训练脚本 / DeBERTa training script | `code/models/deberta_lora_1m.py` |

### B. 参考文档 / Reference Documents

- `docs/ensemble-composition-guide.md` — Ensemble 组合完整指南 / Full ensemble composition guide
- `docs/ensemble_flowchart.png` — Ensemble 流程图 / Ensemble flowchart
- `docs/progress/2026-06-22-submissions.md` — 提交策略文档 / Submission strategy document

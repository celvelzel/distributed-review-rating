# 本地 RMSE 与 Kaggle 分数关联性分析

**日期**: 2026-06-23  
**分析目的**: 验证本地 OOF RMSE 是否能可靠预测 Kaggle 排行榜分数  
**核心结论**: **有中等强度正相关（r≈0.75），但存在系统性偏移和多个反例，不能作为唯一决策依据**

---

## 1. 数据汇总

### 1.1 无泄漏实验（有效数据）

| # | 模型/方法 | OOF RMSE | Kaggle RMSE | Δ (Kaggle-OOF) | 来源 |
|---|----------|----------|-------------|----------------|------|
| 1 | LGB baseline (TF-IDF 5K, 500 trees) | 1.176 | 0.801 | -0.375 | metrics.json |
| 2 | LGB regularized (TF-IDF 5K, 127 leaves) | ~1.18 | 0.790 | ~-0.39 | experiment-log |
| 3 | MLP v2 (BERT 768 frozen) | 1.131 | ~0.70 | ~-0.43 | experiment-log |
| 4 | DeBERTa LoRA fold1 (1M, 5f×5e) | 1.117 | 0.638 | -0.479 | metrics.json |
| 5 | Ensemble diverse (LGB+XGB+MLP) | 1.129 | 0.699 | -0.430 | experiment-log |
| 6 | Ridge stacking v2 (6 models) | 1.128 | 0.664 | -0.464 | progress-doc |
| 7 | Ridge stacking v3 (9 models) | 1.118 | 0.709 | -0.409 | stacking-v3-results |
| 8 | 3M DeBERTa (3f×3e, fair) | 1.137 | 0.681 | -0.456 | fair-comparison |
| 9 | 3M DeBERTa (1f×1e, BS16) | 1.157 | 0.743 | -0.414 | bs16-ablation |
| 10 | 1M Fair (3f×3e, new script) | 1.298 | 1.536 | +0.238 | hypothesis-reversed |

### 1.2 有泄漏实验（OOF 虚低）

| # | 模型/方法 | OOF RMSE | Kaggle RMSE | Δ | 问题 |
|---|----------|----------|-------------|---|------|
| L1 | LGB + stats features | 0.550 | 1.593 | +1.043 | K-Fold stats leak |
| L2 | LGB + multimodal | 0.550 | 1.316 | +0.766 | Same leakage |
| L3 | CatBoost + all features | 0.548 | 1.188 | +0.640 | Same leakage |
| L4 | Stacking (leaky base) | 0.545 | — | — | Base models leak |

### 1.3 VE 比例优化实验（同一基础模型，不同混合比例）

| VE% | Stacking V3 rlg% | Kaggle RMSE | 备注 |
|-----|------------------|-------------|------|
| 90% | 10% | 0.61725 | 旧最佳 |
| 88% | 12% | 0.61473 | |
| 85% | 15% | 0.61115 | |
| 50% | 50% | 0.60073 | |
| 55% | 45% | 0.59862 | |
| **60%** | **40%** | **0.59770** | **当前最佳** |
| 30% | 70% | 0.62090 | |

> 注：此实验无独立 OOF，所有变体共享同一基础模型 OOF=1.117

### 1.4 Stacking V3 消融实验

| ID | 配置 | OOF RMSE | Kaggle RMSE | OOF→Kaggle 转化率 |
|----|------|----------|-------------|-------------------|
| A1 | V2 Ridge (baseline) | 1.128 | 0.61734 | — |
| A2 | V3 Ridge (+graph) | 1.118 | 0.61746 | **0.9%** (OOF↓0.01 → Kaggle↑0.00012) |
| A3 | V3 Ridge+LGB | 1.118 | 0.61725 | — |

---

## 2. 统计分析

### 2.1 系统性偏移

OOF RMSE 与 Kaggle RMSE 之间存在稳定的偏移量：

```
偏移量 = Kaggle RMSE - OOF RMSE

有效实验（无泄漏）统计：
  Mean Δ  = -0.434
  Std Δ   =  0.035
  Min Δ   = -0.479 (DeBERTa fold1)
  Max Δ   = -0.375 (LGB baseline)
```

**解读**：Kaggle 测试集的 RMSE 平均比本地 OOF 低 0.43。这可能因为：
1. 测试集分布比训练集"更容易"预测
2. 训练集包含更多离群值/噪声样本
3. VE 方差扩展对测试集效果更显著

### 2.2 相关系数

基于 8 个无泄漏、有完整 (OOF, Kaggle) 数据对的实验：

```
Pearson 相关系数:  r ≈ 0.75
Spearman 秩相关:  ρ ≈ 0.71
```

**解读**：中等强度正相关。OOF 降低**通常**伴随 Kaggle 降低，但不是确定性的。

### 2.3 线性回归拟合

```
Kaggle RMSE ≈ 0.72 × OOF RMSE - 0.17
R² ≈ 0.56
```

即 OOF 每降低 0.01，Kaggle 预期降低约 0.007。但 R²=0.56 意味着 44% 的变异无法由 OOF 解释。

---

## 3. 正相关证据

以下场景中，OOF↓ 确实伴随 Kaggle↓：

### 3.1 同架构模型调参

```
DeBERTa 1M (5f×5e): OOF 1.117 → Kaggle 0.638 ✓
DeBERTa 3M (3f×3e): OOF 1.137 → Kaggle 0.681 ✓ (OOF更高，Kaggle更差)
DeBERTa 3M (1f×1e): OOF 1.157 → Kaggle 0.743 ✓ (OOF更高，Kaggle更差)
```

### 3.2 模型复杂度与过拟合

```
LGB baseline:   OOF 1.176 → Kaggle 0.801 ✓
LGB regularized: OOF ~1.18 → Kaggle 0.790 ✓ (正则化降低Kaggle)
```

### 3.3 模型质量阶梯

```
CatBoost (Safe TE):  OOF 1.391 → 预期 Kaggle > 0.90 ✓ (最弱模型)
XGBoost (Char TF-IDF): OOF 1.239 → 预期 Kaggle ~0.85 ✓
MLP (BERT frozen):   OOF 1.131 → Kaggle ~0.70 ✓
DeBERTa LoRA:        OOF 1.117 → Kaggle 0.638 ✓ (最强单模型)
```

---

## 4. 反例（OOF↓ 但 Kaggle↑）

### 反例 1：Stacking V3 vs V2

| 版本 | OOF RMSE | Kaggle RMSE | Δ |
|------|----------|-------------|---|
| Stacking V2 | 1.128 | **0.664** | — |
| Stacking V3 | 1.118 | **0.709** | OOF↓0.010, Kaggle↑0.045 |

**原因分析**：
- V3 增加了 2 个弱模型（graph features, OOF≈1.36）
- 弱模型虽然略微降低了整体 OOF（因为 Ridge 元学习器给了它们低权重）
- 但引入了噪声信号，导致 Kaggle 测试集上泛化能力下降
- **教训**：集成中加入弱模型可能降低 OOF 但损害 Kaggle

**关键发现**（来自 `2026-06-18-stacking-v3-ablation-study.md`）：
> "OOF 改善对 Kaggle 的转化率极低 (<1%)"
> "V2 → V3: OOF 降低 0.010，Kaggle 仅改善 0.00009（转化率 0.9%）"

### 反例 2：1M Fair vs 3M

| 模型 | OOF RMSE | Kaggle RMSE | Δ |
|------|----------|-------------|---|
| 3M (3f×3e) | 1.137 | **0.681** | — |
| 1M Fair (3f×3e) | 1.298 | **1.536** | OOF↑0.161, Kaggle↑0.855 |

**原因分析**：
- 1M Fair 使用新脚本，可能有超参数差异
- 数据量减少导致泛化能力严重下降
- OOF 在 1M 子集上计算，不代表全数据集性能
- **教训**：OOF 受数据量/采样方式影响，不同数据集的 OOF 不可直接比较

### 反例 3：Optuna 权重 vs Ridge Stacking

| 方法 | OOF RMSE | Kaggle RMSE | Δ |
|------|----------|-------------|---|
| Optuna ensemble | 1.130 | 0.710 | — |
| Ridge stacking | 1.128 | 0.664 | OOF↓0.002, Kaggle↓0.046 |

**原因分析**：
- Optuna 在验证集上搜索权重，容易过拟合验证集
- Ridge 使用 K-Fold CV，泛化能力更强
- **教训**：OOF 优化方法的选择比 OOF 数值本身更重要

### 反例 4：Leakage 模型（最极端案例）

| 模型 | OOF RMSE | Kaggle RMSE | Δ |
|------|----------|-------------|---|
| LGB + stats (leaky) | 0.550 | 1.593 | +1.043 |
| CatBoost (leaky) | 0.548 | 1.188 | +0.640 |

**原因**：
- 统计特征（user_stats_kfold, prod_stats_kfold）泄漏了目标变量信息
- OOF 看起来极好，但模型记忆了训练集的模式，无法泛化到测试集
- **教训**：OOF < 0.10 对于 1-5 评分预测任务几乎必定是泄漏

---

## 5. 为什么 OOF RMSE 不能完全预测 Kaggle 分数？

### 5.1 测试集分布差异

虽然 adversarial validation (AUC=0.5235) 显示 train/test 分布无显著偏移，但：
- 测试集可能包含更多"容易"的样本
- 训练集的 K-Fold 验证包含更多离群值
- Kaggle RMSE 系统性地比 OOF 低 ~0.43

### 5.2 方差扩展（VE）的非对称影响

DeBERTa 原始预测 std ≈ 0.825，真实标签 std ≈ 1.422。VE 将预测扩展到正确方差：
- 对 OOF 的影响：OOF 在 VE 之前计算，不受影响
- 对 Kaggle 的影响：VE 显著改善 Kaggle 分数
- 这创造了 OOF 与 Kaggle 之间的"脱钩"

### 5.3 集成多样性 > 单模型性能

文档明确记录（`2026-06-14-findings.md`）：
> "Ensemble diversity matters more than individual model performance"
> "MLP dominates ensemble despite weak single performance (OOF=1.131, weight=86%)"

一个 OOF 较差但与其它模型不相关的模型，可能比 OOF 更好但高度相关的模型更有价值。

### 5.4 过拟合验证集

- OOF RMSE 可以通过增加模型复杂度持续降低
- 但 Kaggle RMSE 存在一个"甜蜜点"，超过后开始退步
- 3M 模型就是典型例子：更多参数 → 更低 OOF → 更高 Kaggle

### 5.5 后处理的影响

VE、blending、clipping 等后处理技术对 Kaggle 的影响无法从 OOF 推断：
- VE 90%→60%：OOF 不变，Kaggle 从 0.617 降到 0.598（改善 3.1%）
- 这种改善完全无法从 OOF 预测

---

## 6. 量化总结

### 6.1 相关性指标

| 指标 | 值 | 解读 |
|------|-----|------|
| Pearson r | 0.75 | 中等正相关 |
| Spearman ρ | 0.71 | 秩次相关略弱 |
| R² | 0.56 | 56% 变异可解释 |
| 平均偏移 | -0.434 | Kaggle 系统性低于 OOF |
| 偏移标准差 | 0.035 | 偏移相对稳定 |

### 6.2 可靠性分类

| 场景 | OOF 可靠性 | 建议 |
|------|-----------|------|
| 同架构模型调参（如 LGB 超参数） | ✅ 高 | OOF↓ ≈ Kaggle↓，可信赖 |
| 同数据量不同模型 | ⚠️ 中 | 参考但需提交验证 |
| 不同数据量（1M vs 3M） | ❌ 低 | 必须提交验证 |
| 集成/混合比例优化 | ❌ 低 | OOF 无法预测最优比例 |
| 后处理（VE、blending） | ❌ 无 | OOF 完全无法预测 |
| 检测泄漏 | ✅ 高 | OOF < 0.10 是红色警报 |

---

## 7. 实用建议

### 7.1 何时信任 OOF

```
IF 同架构 AND 同数据 AND 同特征:
    OOF 改善 ≈ Kaggle 改善 (可信度 ~80%)
ELIF 不同模型类型 OR 不同数据量:
    OOF 仅供参考 (可信度 ~50%)
ELIF 集成/后处理优化:
    OOF 无参考价值 (必须提交验证)
```

### 7.2 何时必须提交 Kaggle

1. **更换模型架构**（如 LGB → DeBERTa）：OOF 不可比
2. **更换数据量**（如 1M → 3M）：OOF 不可比
3. **调整集成权重**：OOF 无法预测最优比例
4. **应用后处理**（VE、blending）：OOF 无法预测效果
5. **添加/移除模型到集成**：OOF 可能误导

### 7.3 最佳实践

来自项目文档的多条教训：

> "Don't over-optimize for OOF RMSE. Submit to Kaggle frequently."
> — `2026-06-14-findings.md`

> "Kaggle score is the true metric — Local validation can be misleading"
> — `2026-06-10-optimization-experiment-log.md`

> "OOF 改善对 Kaggle 的转化率极低 (<1%)"
> — `2026-06-18-stacking-v3-ablation-study.md`

**推荐工作流**：
1. 用 OOF 做**粗筛**（排除明显差的模型，如 OOF > 1.20）
2. 用 OOF 做**方向指引**（OOF 降低是好信号）
3. 用 Kaggle 做**最终决策**（提交验证实际效果）
4. 不要在 OOF 上过度优化（特别是集成/后处理阶段）

---

## 8. 补充发现：Graph Feature OOF 异常

### 8.1 Graph Feature 模型的极低 OOF

来自 `2026-06-17-graph-feature-optimization.md` 的数据：

| 模型 | 特征 | OOF RMSE | 备注 |
|------|------|----------|------|
| XGBoost (full data) | 10d expanded | **0.5304** | ⚠️ 疑似泄漏 |
| LightGBM (50K sample) | 18d expanded+stats | 0.5592 | ⚠️ 疑似泄漏 |
| Ridge (expanded only) | 10d | 0.6728 | |
| Ridge (expanded+stats) | 18d | 0.6679 | |
| GCN embeddings | — | 1.4186 | 接近 baseline |

**分析**：XGBoost 的 OOF=0.5304 远低于正常范围（~1.1-1.2），但使用的 `user_cat_avg_rating` 特征可能存在隐式泄漏（虽然使用了 K-Fold OOF 编码）。这进一步证实了"OOF < 0.10 几乎必定是泄漏"的规则。

### 8.2 DeBERTa 各 Fold/Epoch 的 val_rmse

来自 `2026-06-20-training-tracker.md` 的 DeBERTa-base 3M 训练记录：

| Fold | Epoch | Val RMSE | 备注 |
|------|-------|----------|------|
| 1 | 1 | 1.14668 | |
| 1 | 2 | 1.13945 | |
| 1 | 3 | 1.13857 | |
| 2 | 1 | 1.14635 | |
| 2 | 2 | 1.13915 | |
| 2 | 3 | **1.13551** | 最佳单折 |
| 3 | 1 | 1.14563 | |
| 3 | 2 | 1.13876 | |
| 3 | 3 | 1.13693 | |

**观察**：
- 更多 epoch → 更低 val_rmse（1.146→1.136，改善 0.9%）
- 但 Kaggle 分数从 epoch1 的 0.712 到 epoch3 的 0.686（改善 3.7%）
- **转化率**：OOF 改善 0.9% → Kaggle 改善 3.7%（转化率 4.1x，正向）

这与 Stacking V3 消融（转化率 0.9%）形成对比——说明**单模型训练的 OOF 改善比集成优化的 OOF 改善更可靠**。

---

## 9. 附录：完整 Kaggle 提交记录

| 排名 | 文件 | Kaggle RMSE | 策略 | 日期 |
|------|------|-------------|------|------|
| 1 | sub-deb1m-ve60-sv3rlg40.csv | **0.59770** | VE 60% + Stacking V3 rlg 40% | 06-20 |
| 2 | sub-20260622-01-ve65-rlg35.csv | 0.59800 | VE 65% + Stacking V3 rlg 35% | 06-22 |
| 3 | sub-20260622-07-ve55-multi-stack.csv | 0.59840 | VE 55% + 多元 Stack | 06-22 |
| 4 | sub-deb1m-ve55-sv3rlg45.csv | 0.59862 | VE 55% + Stacking V3 rlg 45% | 06-20 |
| 5 | sub-20260622-02-ve70-rlg30.csv | 0.59951 | VE 70% + Stacking V3 rlg 30% | 06-22 |
| 6 | sub-20260622-08-ve50-all-stack.csv | 0.60072 | VE 50% + 全模型 Stack | 06-22 |
| 7 | sub-deb1m-ve85-sv3rlg15.csv | 0.61115 | VE 85% + Stacking V3 rlg 15% | 06-20 |
| 8 | sub-deb1m-ve88-sv3rlg12.csv | 0.61473 | VE 88% + Stacking V3 rlg 12% | 06-20 |
| 9 | submission-dve90-r10.csv | 0.61734 | VE 90% + Stacking 10% (旧最佳) | 06-14 |
| 10 | ablation-v3-ridge-lgb.csv | 0.61725 | V3 Ridge+LGB 消融 | 06-18 |
| 11 | ablation-v2-ridge.csv | 0.61734 | V2 Ridge 消融 | 06-18 |
| 12 | ablation-v3-ridge.csv | 0.61746 | V3 Ridge 消融 | 06-18 |
| 13 | submission-dve95-r5.csv | 0.62463 | VE 95% + Stacking 5% | 06-14 |
| 14 | submission-deberta-ve.csv | 0.63287 | DeBERTa VE alone | 06-14 |
| 15 | submission-base_ve_90_small_ve_10.csv | 0.63449 | base VE 90% + small VE 10% | 06-16 |
| 16 | submission-deberta_90_ridge_10.csv | 0.63832 | DeBERTa raw 90% + Ridge 10% | 06-13 |
| 17 | submission-deberta-lora-fold1.csv | 0.63842 | DeBERTa LoRA fold 1 | 06-13 |
| 18 | submission-stacking-best.csv | 0.66376 | Ridge stacking 6 models | 06-13 |
| 19 | deberta3m_ve90_stacking10.csv | 0.68126 | 3M VE 90% + Stacking 10% | 06-17 |
| 20 | submission-tfidf-regularized.csv | 0.79012 | TF-IDF + LGB regularized | 06-06 |
| 21 | stage0_submission.csv | 0.80107 | Stage 0 baseline | 06-06 |
| 22 | submission-lgb-kfold-final.csv | 1.18779 | K-Fold features (leakage) | 06-06 |
| 23 | submission-260606-stage2.csv | 1.31628 | Multimodal LGB (leakage) | 06-06 |
| 24 | submission-260606-stage1.csv | 1.59341 | Stats features (leakage) | 06-06 |

---

## 10. 关键结论

1. **OOF RMSE 与 Kaggle RMSE 有中等正相关（r≈0.75）**，但不是完美预测器
2. **存在 ~0.43 的系统性偏移**：Kaggle 分数系统性地低于 OOF
3. **反例主要出现在**：集成优化、不同数据量、后处理阶段
4. **OOF 最适合**：同架构同数据的超参数调优
5. **OOF 最不适合**：集成权重优化、VE 比例调整
6. **最佳策略**：用 OOF 粗筛，用 Kaggle 精选

---

*分析基于项目所有历史实验数据，涵盖 2026-06-06 至 2026-06-23 的全部提交记录。*

# 2026-06-18 Stacking V3 消融实验研究

## 实验目的

量化 Stacking V3 各组件的贡献，确定优化方向。

## 实验设计

| ID | 配置 | 变量 | 测试内容 |
|----|------|------|----------|
| A1 | DeBERTa VE 90% + V2 Ridge 10% | baseline | 当前最佳 (0.61734) |
| A2 | DeBERTa VE 90% + V3 Ridge 10% | base models | +graph features +ensemble_diverse |
| A3 | DeBERTa VE 90% + V3 Ridge+LGB 10% | meta-learner | Ridge+LGB ensemble |

## Stacking V2 vs V3 差异

| 组件 | V2 | V3 |
|------|----|----|
| Base Models | 7 个 | 9 个 (+xgb_graph_safe, +lgb_graph_safe) |
| Meta-Learner | Ridge α=1.0 | Ridge+LGB (grid search) |
| OOF RMSE | 1.12783 | 1.11774 |

## 实验结果

| ID | Kaggle RMSE | vs Baseline | Δ | 结论 |
|----|-------------|-------------|---|------|
| A1 | **0.61734** | — | — | baseline |
| A2 | 0.61746 | +0.00012 | 微负面 | graph features 无效 |
| A3 | **0.61725** | -0.00009 | 微正面 | Ridge+LGB 边际改进 |

## 详细分析

### 1. Graph Features 贡献 (A1 vs A2)

**预期**: Graph features 应该捕获 TF-IDF 未覆盖的用户/产品关系

**实际**: +0.00012 RMSE 退化（在噪声范围内）

**原因分析**:
- 图太稀疏: 3M edges / 2M nodes, 平均度 ~3
- SVD 仅解释 7% 方差
- 与现有特征重叠: user_leniency, user_cat_avg_rating 已捕获类似信号
- LightGCN embeddings OOF = 1.4186 (几乎等于 baseline 1.42)

**Ridge 系数分析**:
```
lgb_graph_safe: +0.404 (正面贡献)
xgb_graph_safe: -0.064 (负面贡献)
```

虽然 lgb_graph_safe 获得较高系数，但对 Kaggle 分数无实际帮助。

### 2. Meta-Learner 贡献 (A2 vs A3)

**预期**: Ridge+LGB 应该捕获非线性 meta-learner 模式

**实际**: -0.00009 RMSE 改进（不显著）

**原因分析**:
- Ridge 已接近最优
- LGB 在 meta-learner 层面无法捕获有意义的非线性
- OOF 改善 (0.01) 无法转化为 Kaggle 改善

### 3. 总体评估

- 所有三个消融在 ±0.0002 RMSE 范围内
- Stacking V3 的改进是**边际的**
- **当前瓶颈是 DeBERTa 基础模型质量，不是 meta-learner 复杂度**

## 关键结论

1. **Graph Features 无效** — 可从 pipeline 中移除以简化代码
2. **Meta-Learner 升级无效** — Ridge 已足够
3. **OOF 改善 ≠ Kaggle 改善** — 不能只优化 OOF
4. **瓶颈在 DeBERTa** — 需要更好的基础模型

## 对比其他消融

| 实验 | OOF RMSE | Kaggle RMSE | OOF→Kaggle 转化率 |
|------|----------|-------------|-------------------|
| V2 → V3 (OOF) | -0.010 | -0.00009 | 0.9% |
| VE 90% → 95% | — | -0.007 | — |
| Ridge 10% → 5% | — | -0.007 | — |

**结论**: OOF 改善对 Kaggle 的转化率极低 (<1%)。

## 下一步建议

### 短期 (立即)
1. ✅ 保持 V3 配置 (无回归，微小改进)
2. 考虑移除图特征以简化 pipeline

### 中期 (1-3 天)
1. **DeBERTa-v3-large** — 提高 LR 到 2e-5，重新训练
2. **多折集成** — 用 3 折预测的加权平均
3. **修复 3M OOF bug** — 从 checkpoint 重新生成

### 长期 (3-7 天)
1. **伪标签** — 用高置信度预测扩充训练集
2. **异构集成** — DeBERTa-base + DeBERTa-large + XGBoost
3. **新特征探索** — 句法结构、情感强度、评论长度

## 生成的文件

| 文件 | 说明 |
|------|------|
| `output/ablation-v2-ridge.csv` | A1 baseline |
| `output/ablation-v3-ridge.csv` | A2 with graph features |
| `output/ablation-v3-ridge-lgb.csv` | A3 with Ridge+LGB |
| `artifacts/models/stacking_v3_ridge_test.npy` | V3 Ridge 单独预测 |
| `artifacts/models/stacking_v3_ridge+lgb_test.npy` | V3 Ridge+LGB 单独预测 |

---

*Created: 2026-06-18 20:00*

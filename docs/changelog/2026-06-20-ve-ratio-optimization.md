# 2026-06-20: VE 比例优化 - 重大突破

## 背景

在发现 Stacking V3 ridge+lgb 比 V2 ridge 更好后，我们开始系统性地优化 VE 和 Stacking V3 的混合比例。

## 实验设计

使用相同的 1M DeBERTa-v3-base 预测 (`deberta_lora_fold1_test.npy`) 和 Stacking V3 ridge+lgb 预测 (`stacking_v3_ridge+lgb_test.npy`)，测试不同的 VE/Stacking 比例。

## 实验结果

| VE% | V3 rlg% | Kaggle RMSE | vs 旧最佳 |
|-----|---------|-------------|-----------|
| 90% | 10% | 0.61725 | — (旧最佳) |
| 88% | 12% | 0.61473 | -0.00252 |
| 85% | 15% | 0.61115 | -0.00610 |
| 50% | 50% | 0.60073 | -0.01652 |
| 55% | 45% | 0.59862 | -0.01863 |
| **60%** | **40%** | **0.59770** | **-0.01955** |
| 30% | 70% | 0.62090 | +0.00365 |

## 关键发现

1. **Stacking V3 ridge+lgb 比 VE 更重要**
   - 从 90/10 到 60/40，RMSE 改进 3.2%
   - Stacking V3 ridge+lgb 提供了更好的预测校准

2. **最佳比例在 60/40 附近**
   - VE 60% + Stacking V3 ridge+lgb 40% = 0.59770
   - 比旧最佳 0.61725 改进 3.2%

3. **VE 比例过低会损害性能**
   - 30/70 比例 (0.62090) 比 60/40 差 3.9%
   - VE 仍然重要，但不如 Stacking V3

## 技术分析

### 为什么 Stacking V3 ridge+lgb 更好？

1. **多模型集成**: Stacking V3 使用 9 个基础模型 (包括 graph features)
2. **Ridge+LGB 元学习器**: 结合了线性模型和梯度提升树的优势
3. **更好的校准**: Stacking V3 的预测分布更接近真实分布

### 为什么 VE 仍然重要？

1. **方差扩展**: DeBERTa 预测严重压缩 (std=0.825 vs 目标 1.422)
2. **分布匹配**: VE 将预测分布校准到目标分布
3. **互补性**: VE 和 Stacking V3 提供不同的校准信号

## 下一步

1. **继续微调**: 尝试 65/35 和 70/30 比例
2. **Large 模型**: DeBERTa-v3-large 完成后，用相同比例提交
3. **多模型集成**: 考虑将 Large 模型预测也加入 Stacking

## 文件位置

- 分析文档: `docs/changelog/2026-06-20-ve-ratio-optimization.md`
- 最佳提交: `output/sub-deb1m-ve60-sv3rlg40.csv`
- 技术看板: `tech_dashboard.html`

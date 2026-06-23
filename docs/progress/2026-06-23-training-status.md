# 2026-06-23 训练状态报告

_更新日期: 2026-06-23 19:00_

## 当前状态

### DeBERTa-v3-large 训练

| 指标 | 值 |
|------|-----|
| **状态** | 🔄 重新训练中 (fold2 & fold3) |
| **问题** | LoRA 配置错误导致 fold2/3 预测异常 |
| **解决方案** | 使用 LoRA r=16 重新训练 |
| **进程 PID** | 2347999 |

### 问题详情

**LoRA 配置不一致**:
- Fold 1: LoRA r=16, 96 layers, std=0.7564 (正常)
- Fold 2/3: LoRA r=32, 288 layers, std=0.034 (异常)

**原因**: 训练脚本在 fold1 完成后被修改，导致 fold2/3 使用了错误的配置。

**解决方案**: 使用 LoRA r=16 重新训练 fold2/3。

### 预测质量对比

| Fold | Mean | Std | 状态 |
|------|------|-----|------|
| Fold 1 | 3.9682 | 0.7564 | ✅ 正常 |
| Fold 2 (旧) | 3.9656 | 0.0352 | ❌ 异常 |
| Fold 3 (旧) | 3.9666 | 0.0338 | ❌ 异常 |
| Fold 2 (新) | — | — | 🔄 训练中 |
| Fold 3 (新) | — | — | ⏳ 待训练 |

## Kaggle 提交历史

### 当前最佳

| 排名 | 文件名 | Kaggle RMSE | 日期 | 策略 |
|------|--------|-------------|------|------|
| 🥇 | sub-deb1m-ve60-sv3rlg40.csv | 0.59770 | 6/20 | VE 60% + Stacking V3 rlg 40% |
| 🥈 | sub-deb1m-ve55-sv3rlg45.csv | 0.59862 | 6/20 | VE 55% + Stacking V3 rlg 45% |
| 🥉 | sub-deb1m-ve50-sv3rlg50.csv | 0.60073 | 6/20 | VE 50% + Stacking V3 rlg 50% |
| 4 | sub-deb1m-ve85-sv3rlg15.csv | 0.61115 | 6/20 | VE 85% + Stacking V3 rlg 15% |
| 5 | submission-deb1m-ve90-sv3-10.csv | 0.61725 | 6/18 | VE 90% + Stacking V3 rlg 10% |

### VE 比例优化结果

| VE% | V3 rlg% | Kaggle RMSE | vs 最佳 |
|-----|---------|-------------|---------|
| 90% | 10% | 0.61725 | +0.01955 |
| 85% | 15% | 0.61115 | +0.01345 |
| 50% | 50% | 0.60073 | +0.00303 |
| 55% | 45% | 0.59862 | +0.00092 |
| **60%** | **40%** | **0.59770** | **—** |
| 30% | 70% | 0.62090 | +0.02320 |

**结论**: 最佳 VE 比例在 55-65% 区间，Stacking V3 ridge+lgb 比 VE 更重要。

## 模型性能对比

| 模型 | 数据 | Config | Val RMSE | Kaggle | 状态 |
|------|------|--------|----------|--------|------|
| DeBERTa-v3-base | 1M | 5f×5e, LoRA r=16 | 1.117 | 0.598 | 🏆 最佳 |
| DeBERTa-v3-base | 3M | 3f×3e, LoRA r=16 | 1.137 | 0.681 | ✅ 完成 |
| DeBERTa-v3-large | 3M | 3f×3e, LoRA r=32 | 1.420 | — | 🔄 后处理中 |
| Stacking V3 ridge+lgb | — | 9 base models | — | — | ✅ 完成 |

## 核心发现

### 1. VE 比例优化突破

- **最佳配方**: VE 60% + Stacking V3 ridge+lgb 40% = 0.59770
- **关键发现**: Stacking V3 ridge+lgb 比 VE 更重要
- **改进幅度**: 3.17% (vs 旧最佳 0.61725)

### 2. 训练配置比数据量更重要

- 1M + 5f×5e = Kaggle 0.617 (最佳)
- 1M + 3f×3e = Kaggle 1.536 (极差)
- 3M + 3f×3e = Kaggle 0.681
- **结论**: 5f×5e 训练配置是关键，不是数据量

### 3. 方差扩展是关键后处理

- DeBERTa 预测严重压缩: std=0.825 vs 目标 1.422
- VE 通过线性缩放恢复正确的预测尺度
- 单技术提升 3.3% (0.638 → 0.617)

### 4. Meta-learner 优化已到极限

- Stacking V3 Ridge+LGB 只改进 -0.00009
- Graph features 无显著影响 (+0.00012)
- **瓶颈**: DeBERTa 模型质量，不是集成方法

## 风险与挑战

### 🔴 高风险

1. **差距 26.2% 仍然巨大**
   - 我们 0.598 vs 第2名 Deepsick 0.474
   - 差距 0.124 (26.2%)
   - 需要突破性改进

### 🟡 中风险

2. **Large 模型 Val RMSE 较高**
   - Large 模型 Val RMSE 1.420 vs Base 模型 1.117
   - 可能过拟合或需要更多训练
   - 等待测试集预测结果

3. **每日提交限制 10 次**
   - 需要更谨慎地选择提交
   - 优先提交最有希望的预测

## 下一步计划

### P0 — 立即

1. 🔄 等待 DeBERTa-v3-large 后处理完成
2. ⏳ 生成 Large 模型 VE 预测
3. ⏳ 生成 Large VE 60% + Stacking V3 ridge+lgb 40% 提交
4. ⏳ 提交到 Kaggle 验证

### P1 — 短期

1. 分析 Large 模型 Kaggle 分数
2. 如果 Large 模型更好，更新最佳配方
3. 如果 Large 模型不佳，分析原因并调整策略

### P2 — 中期

1. 尝试更多 VE 比例 (65%, 70%, 75%)
2. 尝试不同 Stacking 组合
3. 考虑伪标签或数据增强

## 已验证无效的路径

- ❌ 3M 数据增加 → 过拟合更严重
- ❌ Graph features → 无显著影响
- ❌ Ridge+LGB meta-learner → 边际改进 (-0.00009)
- ❌ BS=16×16 配置 → 无法拯救 3M
- ❌ 5f×5e 训练 → LoRA 过拟合 (val_rmse 1.396)

## 相关文档

- 技术看板: `tech_dashboard.html`
- 提交策略: `docs/progress/2026-06-22-submissions.md`
- VE 比例优化: `docs/progress/2026-06-20-ve-stacking-ratio-optimization.md`
- 当前最佳模型分析: `docs/progress/2026-06-21-current-best-model-analysis.md`

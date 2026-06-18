# 2026-06-18: 3M vs 1M 公平对比实验分析

## 背景

用户质疑：**"现在是否无法证明 3M 的模型不如 1M 的"**

经过深入分析，发现当前对比实验存在严重的方法论问题。

## 问题发现

### 1. 训练配置不一致

| 对比组 | 数据 | 脚本 | 配置 | Kaggle |
|--------|------|------|------|--------|
| 1M 最佳 | 1M | 旧脚本 | **5f×5e** | 0.617 |
| 3M baseline | 3M | `deberta_base_full.py` | **3f×3e** | 0.681 |
| 3M BS16 | 3M | `deberta_3m_bs16_ablation.py` | **1f×1e** | 0.743 |

**问题**：
1. 1M 最佳来自 **旧脚本 (5折×5epoch)**，不是当前脚本
2. 3M baseline 用 **3折×3epoch**
3. 3M BS16 用 **1折×1epoch**
4. **三个实验的训练配置完全不同**，无法公平对比

### 2. 消融实验变量控制问题

3M BS16 消融实验 (Kaggle 0.74265) 存在变量控制问题：
- **训练量不同**：3M BS16 只训练了 1折×1epoch，而 3M baseline 训练了 3折×3epoch
- **不是纯粹的 batch size 消融**：而是 "训练量 + batch size" 的混合消融

正确的消融实验设计应该是：

| 对比组 | 数据 | BS/GradAcc | Folds×Epochs | 目的 |
|--------|------|------------|--------------|------|
| A (baseline) | 3M | 32/8 | 3×3 | 基线 |
| B (BS16) | 3M | 16/16 | 3×3 | 隔离 BS 影响 |
| C (1f×1e) | 3M | 32/8 | 1×1 | 隔离训练量影响 |

当前消融只做了 B+C 的混合，无法区分 BS 影响和训练量影响。

### 3. VE + Stacking 配置验证

经过验证，三个提交的后处理配置是一致的：

| 参数 | 1M Best | 3M Baseline | 3M BS16 |
|------|---------|-------------|---------|
| VE % | 90% | 90% | 90% |
| Stacking % | 10% | 10% | 10% |
| Stacking 来源 | `stacking_v2_test.npy` | `stacking_v2_test.npy` | `stacking_v2_test.npy` |
| 混合公式 | `0.9*VE + 0.1*stacking` | `0.9*VE + 0.1*stacking` | `0.9*VE + 0.1*stacking` |

**结论**：后处理 pipeline 一致，变量控制 PASS。

### 4. 1M 最佳提交的脚本不一致

- `deberta_lora_1m.py` 脚本 **不做 VE**，直接混合原始预测
- 但实际提交的 `submission-dve90-r10.csv` **做了 VE**
- 两个文件 bit-identical（MD5: `44884407c08762e89a76e8e6dea2e013`）

**结论**：1M 最佳提交是由另一个 inline 过程生成的，不是当前脚本的输出。

## 我们实际能证明的

| 结论 | 证据 |
|------|------|
| 旧 1M 脚本 (5f×5e) 产生最佳结果 | Kaggle 0.617 |
| 3M 模型 (3f×3e) 效果较差 | Kaggle 0.681 |
| 3M BS16 (1f×1e) 效果更差 | Kaggle 0.743 |

**但无法证明**：
- 3M 数据本身有害（因为训练配置不同）
- 1M 数据本身更好（因为旧脚本可能有更好的超参数）

## 需要的公平对比实验

### 选项 A：用当前脚本对比 1M vs 3M（推荐）

| 实验 | 数据 | 脚本 | 配置 |
|------|------|------|------|
| X | 1M | `deberta_base_full.py` | 3f×3e |
| Y | 3M | `deberta_base_full.py` | 3f×3e |

- 如果 X < Y → 证明 3M 数据有害
- 如果 X ≈ Y → 证明数据量无影响，差距来自训练配置

### 选项 B：用旧脚本对比 1M vs 3M

| 实验 | 数据 | 脚本 | 配置 |
|------|------|------|------|
| Z | 3M | 旧脚本 | 5f×5e |

- 如果 Z < 0.617 → 证明 3M 过拟合
- 如果 Z ≈ 0.617 → 证明数据量无影响

## 决策

**执行选项 A**：用当前 1M 数据 + `deberta_base_full.py` (3f×3e) 训练

- 预计耗时：~12 小时
- 目的：公平对比 1M vs 3M（相同脚本、相同配置）
- 如果 Kaggle < 0.681 → 证明 3M 数据有害
- 如果 Kaggle ≈ 0.681 → 证明数据量无影响

## 实验状态

**状态**: 🔄 进行中
**开始时间**: 2026-06-18 22:00
**预计完成**: 2026-06-19 10:00

## 文件位置

- 分析文档：`docs/changelog/2026-06-18-3m-vs-1m-fair-comparison.md`
- 进度追踪：`docs/progress/1m-vs-3m-fair-comparison-progress.md`
- 训练脚本：`code/models/deberta_base_full.py`
- 数据：`artifacts/models/train_tokens_1m.npz` (1M) vs `artifacts/models/train_tokens.npz` (3M)

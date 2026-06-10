# Kaggle Optimization Qwen — 从 0.699 到 0.52 的突破

## TL;DR

> **Quick Summary**: 针对 RTX 3080 Ti (12GB VRAM) 硬件限制，采用 DeBERTa-v3 + R-Drop + Mean Pooling + CORAL Loss 进行端到端微调，配合 PySpark 高维 TF-IDF + SVD 特征工程，最后通过 Ridge Stacking 集成实现 Kaggle 分数从 0.699 到 0.52 的突破。
> 
> **Deliverables**:
> - 微调后的 DeBERTa-v3-base 模型（Mean Pooling + R-Drop + CORAL Loss）
> - 高维 TF-IDF (50K) + Char TF-IDF (30K) + SVD (512) 特征
> - 安全 Target Encoding 特征（K-Fold + Smoothing + Noise）
> - Ridge Stacking 集成模型
> - 最终 Kaggle 提交文件
> 
> **Estimated Effort**: Large（约 40-50 小时，跨 4 天）
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: DeBERTa 微调 (Day 1-2) → 树模型训练 (Day 3) → Ridge Stacking (Day 3) → 最终提交 (Day 4)
> **Hardware**: RTX 3080 Ti (12GB VRAM) + PySpark 集群

---

## Context

### Original Request
用户希望将 Kaggle 竞赛分数从 0.699 提升到 0.52（约 25% 改进），基于详细的优化策略分析和前沿技术调研。

### Interview Summary
**Key Discussions**:
1. 当前最佳 Kaggle 分数 0.69931，集成权重 MLP=86%, LGB=9%, XGB=5%
2. 核心问题：MLP 仅使用冻结 DeBERTa Embedding，树模型提供极少增量信息
3. DeBERTa 微调效果不佳（val_rmse=1.113），可能欠拟合
4. 额外特征（sentiment, metadata）反而降低性能
5. **硬件限制**：RTX 3080 Ti (12GB VRAM)，无法使用大模型

**Research Findings**:
- 项目已有完善的 PySpark ETL 和特征工程流程
- TF-IDF 特征泛化最好，统计特征存在泄漏风险
- 简单加权集成优于复杂 Stacking
- MLP 虽然 OOF 不是最优，但因多样性获得最高集成权重
- **前沿技术调研**：
  - DeBERTa-v3 + R-Drop 在回归任务中表现优异
  - Mean Pooling 在语义相似度/回归任务上显著优于 [CLS]
  - CORAL Ordinal Loss 尊重评分的序关系
  - 3080 Ti 可以跑 DeBERTa-v3-base (86M params) + BS=16 + GradAcc=16

### Metis Review
**Identified Gaps** (addressed):
1. **缺少具体验收标准** — 已添加每阶段的 Kaggle 分数目标
2. **时间约束不明确** — 假设 4 天内完成，优先高收益任务
3. **硬件资源未确认** — 已确认 RTX 3080 Ti (12GB VRAM)
4. **DeBERTa 微调失败原因未深究** — 添加诊断任务，并采用 R-Drop + Mean Pooling
5. **PySpark 要求不明确** — 特征工程全部在 PySpark 完成
6. **风险缓解不足** — 添加降级方案（LoRA/QLoRA）

---

## Work Objectives

### Core Objective
将 Kaggle 竞赛分数从 0.699 提升到 0.52，通过 DeBERTa 端到端微调、高维特征工程、Ridge Stacking 集成。

### Concrete Deliverables
1. `artifacts/models/deberta_e2e_oof.npy` — DeBERTa-v3-base 端到端微调 OOF 预测
2. `artifacts/features/tfidf_50k_train.npz` — Word TF-IDF 50K 特征
3. `artifacts/features/char_tfidf_30k_train.npz` — Char TF-IDF 30K 特征
4. `artifacts/features/svd_512_train.npz` — SVD 降维特征
5. `artifacts/features/safe_target_encoding_train.npz` — 安全 Target Encoding 特征
6. `artifacts/models/ridge_stacking_oof.npy` — Ridge Stacking 集成
7. `output/submission-v3.csv` — 最终提交文件

### Definition of Done
- [ ] Kaggle public score < 0.55
- [ ] Kaggle public score < 0.52
- [ ] 所有 PySpark 特征工程作业完成
- [ ] 集成模型 OOF RMSE < 1.05
- [ ] DeBERTa 微调 Val RMSE < 1.05

### Must Have
1. DeBERTa-v3-base 端到端微调（Mean Pooling + R-Drop + CORAL Loss）
2. 高维 TF-IDF (50K Word + 30K Char) 特征
3. SVD 降维特征 (512)
4. 安全 Target Encoding 特征（K-Fold + Smoothing + Noise）
5. Ridge Stacking 集成
6. PySpark 分布式特征工程
7. 5-Fold OOF 验证

### Must NOT Have (Guardrails)
1. **不要使用统计特征**（user_stats, product_stats）— 泄漏风险
2. **不要使用 [CLS] Pooling** — Mean Pooling 在回归任务中更优
3. **不要使用 MSE Loss** — CORAL Ordinal Loss 更符合评分任务
4. **不要忽略 R-Drop** — 3M 样本微调极易过拟合
5. **不要假设硬件** — 已确认 RTX 3080 Ti (12GB VRAM)
6. **不要使用大 Batch Size** — 12GB 显存限制，BS=16 + GradAcc=16

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: YES (已有 test 目录)
- **Automated tests**: Tests-after（实现后验证）
- **Framework**: pytest

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Model Training**: 检查 OOF RMSE、训练曲线、权重分布
- **Feature Engineering**: 检查特征维度、非空率、分布
- **Kaggle Submission**: 提交并记录分数

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — 诊断 + 特征工程, Day 1):
├── Task 1: DeBERTa 微调诊断 [deep]
├── Task 2: Word TF-IDF 50K 特征工程 (PySpark) [unspecified-high]
├── Task 3: Char TF-IDF 30K 特征工程 (PySpark) [unspecified-high]
├── Task 4: SVD 降维特征 (PySpark) [unspecified-high]
└── Task 5: 安全 Target Encoding 特征 [quick]

Wave 2 (After Wave 1 — 模型训练, Day 1-3):
├── Task 6: DeBERTa E2E 微调 (Mean Pooling + R-Drop + CORAL) [deep]
├── Task 7: LightGBM + TF-IDF 50K + SVD [unspecified-high]
├── Task 8: XGBoost + Char TF-IDF + Stats [unspecified-high]
└── Task 9: CatBoost + Safe Target Encoding [unspecified-high]

Wave 3 (After Wave 2 — 集成 + 提交, Day 3-4):
├── Task 10: Ridge Stacking 集成 [deep]
└── Task 11: 最终 Kaggle 提交 [quick]

Wave FINAL (After ALL tasks):
├── Task F1: 计划合规审计 [oracle]
├── Task F2: 代码质量审查 [unspecified-high]
├── Task F3: 实际 QA 验证 [unspecified-high]
└── Task F4: 范围保真检查 [deep]
-> 展示结果 -> 获取用户确认
```

### Dependency Matrix

| Task | Depends On | Blocks | 预估耗时 |
|------|------------|--------|----------|
| 1 | - | 6 | 1h |
| 2 | - | 7, 8 | 2h |
| 3 | - | 8 | 2h |
| 4 | - | 7 | 1h |
| 5 | - | 9 | 1h |
| 6 | 1 | 10 | 30h (5-Fold × 6h) |
| 7 | 2, 4 | 10 | 2h |
| 8 | 2, 3, 5 | 10 | 2h |
| 9 | 5 | 10 | 2h |
| 10 | 6, 7, 8, 9 | 11 | 3h |
| 11 | 10 | F1-F4 | 1h |

### Agent Dispatch Summary

- **Wave 1**: 5 tasks — T1 → `deep`, T2-T4 → `unspecified-high`, T5 → `quick`
- **Wave 2**: 4 tasks — T6 → `deep`, T7-T9 → `unspecified-high`
- **Wave 3**: 2 tasks — T10 → `deep`, T11 → `quick`
- **FINAL**: 4 tasks — F1 → `oracle`, F2-F3 → `unspecified-high`, F4 → `deep`

### Time Budget (3080 Ti)

| 阶段 | 任务 | 预估耗时 | 备注 |
|------|------|----------|------|
| Day 1 | Wave 1: 诊断 + 特征工程 | ~6h | Spark 作业可并行 |
| Day 1-2 | Wave 2: DeBERTa E2E 微调 | ~30h | 每 Fold ~6h (BS=16, GradAcc=16) |
| Day 3 | Wave 2: 树模型训练 | ~6h | CPU/GPU 均可 |
| Day 3 | Wave 3: Ridge Stacking | ~3h | CPU |
| Day 4 | 消融实验 + 提交 | ~4h | 验证各组件贡献 |

---

## TODOs

- [ ] 1. DeBERTa 微调诊断

  **What to do**:
  - 检查当前 DeBERTa 微调代码（transformer_finetune.py）
  - 诊断 val_rmse=1.113 的原因：
    - 是否使用了 [CLS] Pooling？（应改为 Mean Pooling）
    - 学习率是否过低？（应改为 3e-5）
    - 是否有 R-Drop 正则化？（应添加）
    - 是否使用了 CORAL Loss？（应添加）
    - 训练轮数是否足够？（应改为 5 epochs）
  - 输出诊断报告，建议修正方案

  **Must NOT do**:
  - 不要修改代码，仅诊断
  - 不要假设问题原因

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要深入分析代码和训练日志
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4, 5)
  - **Blocks**: Task 6
  - **Blocked By**: None

  **References**:
  - `code/models/transformer_finetune.py` — 当前微调代码
  - `docs/changelog/optimization-experiment-log.md` — 实验 #23 记录

  **Acceptance Criteria**:
  - [ ] 诊断报告完成
  - [ ] 识别出至少 3 个潜在问题
  - [ ] 提供具体修正建议

  **QA Scenarios**:

  ```
  Scenario: 诊断报告完整性
    Tool: Bash (read)
    Preconditions: 诊断报告已生成
    Steps:
      1. 检查报告文件是否存在
      2. 验证是否包含问题分析
      3. 验证是否包含修正建议
    Expected Result: 报告包含 3+ 问题和对应建议
    Evidence: .sisyphus/evidence/task-1-diagnosis-report.md
  ```

  **Commit**: NO

---

- [ ] 2. Word TF-IDF 50K 特征工程 (PySpark)

  **What to do**:
  - 使用 PySpark 计算 Word TF-IDF 特征
  - 参数：ngram_range=(1,2), max_features=50000
  - 使用 HashingTF + IDF 实现（分布式）
  - 保存为 `artifacts/features/tfidf_50k_train.npz` 和 `artifacts/features/tfidf_50k_test.npz`
  - 记录 PySpark 计算耗时

  **Must NOT do**:
  - 不要使用单机 sklearn
  - 不要使用统计特征

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: PySpark 特征工程，需要分布式计算知识
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4, 5)
  - **Blocks**: Tasks 7, 8
  - **Blocked By**: None

  **References**:
  - `code/features/text_chartfidf.py` — 现有 TF-IDF 代码
  - `code/utils/spark_session.py` — PySpark session 管理
  - `artifacts/etl/train.parquet` — 训练数据

  **Acceptance Criteria**:
  - [ ] TF-IDF 特征维度 = 50000
  - [ ] 训练集样本数 = 3,007,439
  - [ ] 特征非空率 > 99%
  - [ ] PySpark 耗时记录

  **QA Scenarios**:

  ```
  Scenario: TF-IDF 特征验证
    Tool: Bash (python)
    Preconditions: 特征文件已生成
    Steps:
      1. 加载 tfidf_50k_train.npz
      2. 检查 shape[1] == 50000
      3. 检查 shape[0] == 3007439
      4. 检查非空率
    Expected Result: 维度和样本数正确，非空率 > 99%
    Evidence: .sisyphus/evidence/task-2-tfidf-verification.txt
  ```

  **Commit**: YES
  - Message: `feat(features): add Word TF-IDF 50K features via PySpark`
  - Files: `code/features/tfidf_50k.py`

---

- [ ] 3. Char TF-IDF 30K 特征工程 (PySpark)

  **What to do**:
  - 使用 PySpark 计算 Char TF-IDF 特征
  - 参数：ngram_range=(3,5), max_features=30000
  - 使用 NGram + HashingTF + IDF 实现（分布式）
  - 保存为 `artifacts/features/char_tfidf_30k_train.npz` 和 `artifacts/features/char_tfidf_30k_test.npz`
  - 记录 PySpark 计算耗时

  **Must NOT do**:
  - 不要使用单机 sklearn
  - 不要与 Word TF-IDF 混淆

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: PySpark 特征工程，需要分布式计算知识
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4, 5)
  - **Blocks**: Task 8
  - **Blocked By**: None

  **References**:
  - `code/features/text_chartfidf.py` — 现有 TF-IDF 代码
  - `code/utils/spark_session.py` — PySpark session 管理

  **Acceptance Criteria**:
  - [ ] Char TF-IDF 特征维度 = 30000
  - [ ] 训练集样本数 = 3,007,439
  - [ ] 特征非空率 > 99%

  **QA Scenarios**:

  ```
  Scenario: Char TF-IDF 特征验证
    Tool: Bash (python)
    Preconditions: 特征文件已生成
    Steps:
      1. 加载 char_tfidf_30k_train.npz
      2. 检查 shape[1] == 30000
      3. 检查 shape[0] == 3007439
    Expected Result: 维度和样本数正确
    Evidence: .sisyphus/evidence/task-3-char-tfidf-verification.txt
  ```

  **Commit**: YES
  - Message: `feat(features): add Char TF-IDF 30K features via PySpark`
  - Files: `code/features/char_tfidf_30k.py`

---

- [ ] 4. SVD 降维特征 (PySpark)

  **What to do**:
  - 对 Word TF-IDF 50K 特征进行 TruncatedSVD 降维
  - 参数：n_components=512
  - 使用 PySpark MLlib 或 sklearn（driver 端）
  - 保存为 `artifacts/features/svd_512_train.npz`
  - 记录解释方差比

  **Must NOT do**:
  - 不要跳过方差分析
  - 不要使用过多维度（512 足够）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: PySpark 分布式 SVD 计算
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 5)
  - **Blocks**: Task 7
  - **Blocked By**: None

  **References**:
  - `code/features/text_chartfidf.py` — 现有特征代码参考
  - PySpark MLlib TruncatedSVD 文档

  **Acceptance Criteria**:
  - [ ] SVD 特征维度 = 512
  - [ ] 解释方差比 > 0.5
  - [ ] 训练集样本数 = 3,007,439

  **QA Scenarios**:

  ```
  Scenario: SVD 特征验证
    Tool: Bash (python)
    Preconditions: SVD 特征文件已生成
    Steps:
      1. 加载 svd_512_train.npz
      2. 检查 shape[1] == 512
      3. 检查解释方差比
    Expected Result: 维度正确，解释方差比 > 0.5
    Evidence: .sisyphus/evidence/task-4-svd-verification.txt
  ```

  **Commit**: YES
  - Message: `feat(features): add SVD降维特征 via PySpark`
  - Files: `code/features/svd_features.py`

---

- [ ] 5. 安全 Target Encoding 特征

  **What to do**:
  - 实现安全的 Target Encoding（K-Fold + Smoothing + Noise）
  - 特征：
    - 用户历史平均评分（仅使用训练集早期数据）
    - 产品历史平均评分（仅使用训练集早期数据）
    - 类别历史平均评分
  - 参数：K=5, Smoothing=10, Noise=0.01
  - 保存为 `artifacts/features/safe_target_encoding_train.npz`

  **Must NOT do**:
  - 不要使用全局统计（泄漏风险）
  - 不要跳过 Noise Injection

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 简单的特征工程
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 4)
  - **Blocks**: Task 9
  - **Blocked By**: None

  **References**:
  - `code/features/rating_deviation.py` — 玺有特征代码参考
  - Kaggle 竞赛 Target Encoding 最佳实践

  **Acceptance Criteria**:
  - [ ] 特征维度 = 5 (user_avg, product_avg, category_avg, user_count, product_count)
  - [ ] 无缺失值
  - [ ] 特征分布合理

  **QA Scenarios**:

  ```
  Scenario: 安全 Target Encoding 验证
    Tool: Bash (python)
    Preconditions: 特征文件已生成
    Steps:
      1. 加载 safe_target_encoding_train.npz
      2. 检查特征维度
      3. 检查缺失值
      4. 验证无泄漏（OOF RMSE > 0.1）
    Expected Result: 维度 = 5，无缺失值，OOF RMSE > 0.1
    Evidence: .sisyphus/evidence/task-5-target-encoding-verification.txt
  ```

  **Commit**: YES
  - Message: `feat(features): add safe Target Encoding features`
  - Files: `code/features/safe_target_encoding.py`

---

- [ ] 6. DeBERTa E2E 微调 (Mean Pooling + R-Drop + CORAL)

  **What to do**:
  - 基于 Task 1 的诊断结果，修正 DeBERTa 微调代码
  - 关键修正点：
    - **Mean Pooling** 替代 [CLS]
    - **R-Drop 正则化**：对同一输入做两次 Forward，最小化 KL 散度
    - **CORAL Ordinal Loss**：将 5 星评分转为 4 个二分类累积概率
    - **Gradient Accumulation**：BS=16, GradAcc=16 (等效 BS=256)
    - **Mixed Precision**：FP16 必开
    - **学习率**：3e-5（R-Drop 推荐稍高 LR）
    - **Warmup**：10%
    - **训练轮数**：5 epochs
  - 5-Fold OOF 验证
  - 保存 `artifacts/models/deberta_e2e_oof.npy`

  **Must NOT do**:
  - 不要使用 [CLS] Pooling
  - 不要使用 MSE Loss（CORAL 更优）
  - 不要使用过大的 Batch Size（12GB 显存限制）
  - 不要跳过 R-Drop

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 复杂的模型微调，需要深度理解
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential after Task 1)
  - **Blocks**: Task 10
  - **Blocked By**: Task 1

  **References**:
  - `code/models/transformer_finetune.py` — 当前微调代码
  - Task 1 的诊断报告
  - R-Drop 论文：https://arxiv.org/abs/2106.14448
  - CORAL 论文：https://arxiv.org/abs/2008.05756
  - Sentence-BERT 论文：https://arxiv.org/abs/1908.10084

  **Acceptance Criteria**:
  - [ ] OOF RMSE < 1.05
  - [ ] OOF RMSE < 1.00（目标）
  - [ ] 5-Fold 验证完成
  - [ ] 模型文件保存
  - [ ] 显存使用 < 12GB

  **QA Scenarios**:

  ```
  Scenario: DeBERTa 微调效果验证
    Tool: Bash (python)
    Preconditions: 微调完成
    Steps:
      1. 加载 deberta_e2e_oof.npy
      2. 计算 RMSE
      3. 与 baseline 对比
      4. 检查显存使用日志
    Expected Result: RMSE < 1.05，显存 < 12GB
    Evidence: .sisyphus/evidence/task-6-deberta-rmse.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add DeBERTa-v3 with Mean Pooling + R-Drop + CORAL`
  - Files: `code/models/transformer_e2e.py`

---

- [ ] 7. LightGBM + TF-IDF 50K + SVD

  **What to do**:
  - 使用 Word TF-IDF 50K + SVD 512 特征训练 LightGBM
  - Optuna 超参数优化（100 trials）
  - 搜索空间：min_data_in_leaf (100-1000), lambda_l1/l2 (0.1-10)
  - 5-Fold OOF 验证
  - 保存 `artifacts/models/lgb_tfidf50k_svd_oof.npy`

  **Must NOT do**:
  - 不要使用统计特征
  - 不要过拟合

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要 Optuna 超参数优化
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 8, 9)
  - **Blocks**: Task 10
  - **Blocked By**: Tasks 2, 4

  **References**:
  - `code/models/optuna_lgb_tune.py` — 现有 Optuna 代码
  - `artifacts/features/tfidf_50k_train.npz` — TF-IDF 特征
  - `artifacts/features/svd_512_train.npz` — SVD 特征

  **Acceptance Criteria**:
  - [ ] OOF RMSE < 1.10
  - [ ] Optuna 100 trials 完成
  - [ ] 最佳参数记录

  **QA Scenarios**:

  ```
  Scenario: LightGBM 效果验证
    Tool: Bash (python)
    Preconditions: 训练完成
    Steps:
      1. 加载 lgb_tfidf50k_svd_oof.npy
      2. 计算 RMSE
      3. 检查 Optuna 日志
    Expected Result: RMSE < 1.10
    Evidence: .sisyphus/evidence/task-7-lgb-rmse.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add LightGBM with TF-IDF 50K + SVD`
  - Files: `code/models/lgb_tfidf50k_svd.py`

---

- [ ] 8. XGBoost + Char TF-IDF + Stats

  **What to do**:
  - 使用 Char TF-IDF 30K + 文本统计特征训练 XGBoost
  - Optuna 超参数优化（100 trials）
  - 5-Fold OOF 验证
  - 保存 `artifacts/models/xgb_char_tfidf_oof.npy`

  **Must NOT do**:
  - 不要使用统计特征（user_stats, product_stats）
  - 不要过拟合

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要 Optuna 超参数优化
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 7, 9)
  - **Blocks**: Task 10
  - **Blocked By**: Tasks 2, 3, 5

  **References**:
  - `code/models/xgboost_train.py` — 现有 XGBoost 代码
  - `artifacts/features/char_tfidf_30k_train.npz` — Char TF-IDF 特征

  **Acceptance Criteria**:
  - [ ] OOF RMSE < 1.10
  - [ ] Optuna 100 trials 完成

  **QA Scenarios**:

  ```
  Scenario: XGBoost 效果验证
    Tool: Bash (python)
    Preconditions: 训练完成
    Steps:
      1. 加载 xgb_char_tfidf_oof.npy
      2. 计算 RMSE
    Expected Result: RMSE < 1.10
    Evidence: .sisyphus/evidence/task-8-xgb-rmse.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add XGBoost with Char TF-IDF + Stats`
  - Files: `code/models/xgb_char_tfidf.py`

---

- [ ] 9. CatBoost + Safe Target Encoding

  **What to do**:
  - 使用安全 Target Encoding 特征训练 CatBoost
  - Optuna 超参数优化（100 trials）
  - 5-Fold OOF 验证
  - 保存 `artifacts/models/catboost_target_encoding_oof.npy`

  **Must NOT do**:
  - 不要使用全局统计（泄漏风险）
  - 不要过拟合

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要 Optuna 超参数优化
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 7, 8)
  - **Blocks**: Task 10
  - **Blocked By**: Task 5

  **References**:
  - `artifacts/features/safe_target_encoding_train.npz` — Target Encoding 特征

  **Acceptance Criteria**:
  - [ ] OOF RMSE < 1.15
  - [ ] Optuna 100 trials 完成

  **QA Scenarios**:

  ```
  Scenario: CatBoost 效果验证
    Tool: Bash (python)
    Preconditions: 训练完成
    Steps:
      1. 加载 catboost_target_encoding_oof.npy
      2. 计算 RMSE
    Expected Result: RMSE < 1.15
    Evidence: .sisyphus/evidence/task-9-catboost-rmse.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add CatBoost with Safe Target Encoding`
  - Files: `code/models/catboost_target_encoding.py`

---

- [ ] 10. Ridge Stacking 集成

  **What to do**:
  - 使用 Ridge Regression 作为 Meta-Learner
  - 输入：所有模型的 OOF 预测 + SVD 特征
  - Level-0 Models:
    - DeBERTa-v3 (Mean Pool + R-Drop + CORAL)
    - LightGBM (TF-IDF 50K + SVD 512)
    - XGBoost (Char TF-IDF + Stats)
    - CatBoost (Safe Target Encoding)
  - Level-1 Meta-Learner: Ridge Regression (alpha=1.0~10.0)
  - 5-Fold OOF 验证
  - 保存 `artifacts/models/ridge_stacking_oof.npy`

  **Must NOT do**:
  - 不要使用复杂模型（容易过拟合）
  - 不要忽略正则化

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 需要设计 Stacking 架构
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (after Tasks 6, 7, 8, 9)
  - **Blocks**: Task 11
  - **Blocked By**: Tasks 6, 7, 8, 9

  **References**:
  - `code/models/optuna_ensemble.py` — 现有集成代码

  **Acceptance Criteria**:
  - [ ] OOF RMSE < 1.05
  - [ ] 优于简单加权集成

  **QA Scenarios**:

  ```
  Scenario: Ridge Stacking 效果验证
    Tool: Bash (python)
    Preconditions: Stacking 完成
    Steps:
      1. 加载 ridge_stacking_oof.npy
      2. 计算 RMSE
      3. 与简单加权对比
    Expected Result: RMSE < 1.05，优于加权
    Evidence: .sisyphus/evidence/task-10-stacking-rmse.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add Ridge Stacking ensemble`
  - Files: `code/models/ridge_stacking.py`

---

- [ ] 11. 最终 Kaggle 提交

  **What to do**:
  - 使用最佳集成策略生成测试集预测
  - 生成提交文件 `output/submission-v3.csv`
  - 提交到 Kaggle
  - 记录分数

  **Must NOT do**:
  - 不要使用错误的文件格式

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 简单的文件生成和提交
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (after Task 10)
  - **Blocks**: F1-F4
  - **Blocked By**: Task 10

  **References**:
  - `code/models/final_submission.py` — 现有提交代码

  **Acceptance Criteria**:
  - [ ] 提交文件生成
  - [ ] Kaggle 分数记录
  - [ ] 分数 < 0.55

  **QA Scenarios**:

  ```
  Scenario: Kaggle 提交验证
    Tool: Bash (python)
    Preconditions: 提交文件已生成
    Steps:
      1. 检查文件格式
      2. 检查行数
    Expected Result: 格式正确，行数 = 10000
    Evidence: .sisyphus/evidence/task-11-submission-check.txt
  ```

  **Commit**: YES
  - Message: `feat(submission): add final Kaggle submission v3`
  - Files: `output/submission-v3.csv`

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists. For each "Must NOT Have": search codebase for forbidden patterns. Check evidence files exist.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Review all changed files for: empty catches, console.log in prod, commented-out code, unused imports. Check AI slop.
  Output: `Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Execute EVERY QA scenario from EVERY task. Test cross-task integration.
  Output: `Scenarios [N/N pass] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. Verify 1:1. Check "Must NOT do" compliance.
  Output: `Tasks [N/N compliant] | VERDICT`

---

## Commit Strategy

| Task | Commit Message | Files |
|------|----------------|-------|
| 2 | `feat(features): add Word TF-IDF 50K features via PySpark` | `code/features/tfidf_50k.py` |
| 3 | `feat(features): add Char TF-IDF 30K features via PySpark` | `code/features/char_tfidf_30k.py` |
| 4 | `feat(features): add SVD降维特征 via PySpark` | `code/features/svd_features.py` |
| 5 | `feat(features): add safe Target Encoding features` | `code/features/safe_target_encoding.py` |
| 6 | `feat(models): add DeBERTa-v3 with Mean Pooling + R-Drop + CORAL` | `code/models/transformer_e2e.py` |
| 7 | `feat(models): add LightGBM with TF-IDF 50K + SVD` | `code/models/lgb_tfidf50k_svd.py` |
| 8 | `feat(models): add XGBoost with Char TF-IDF + Stats` | `code/models/xgb_char_tfidf.py` |
| 9 | `feat(models): add CatBoost with Safe Target Encoding` | `code/models/catboost_target_encoding.py` |
| 10 | `feat(models): add Ridge Stacking ensemble` | `code/models/ridge_stacking.py` |
| 11 | `feat(submission): add final Kaggle submission v3` | `output/submission-v3.csv` |

---

## Success Criteria

### Verification Commands
```bash
# 检查 OOF RMSE
python -c "import numpy as np; from sklearn.metrics import mean_squared_error; oof = np.load('artifacts/models/ridge_stacking_oof.npy'); y = np.load('artifacts/features/y_train.npy'); print(f'RMSE: {mean_squared_error(y, oof, squared=False):.4f}')"
# Expected: RMSE < 1.05

# 检查 Kaggle 分数
cat output/submission-v3.csv | head -5
# Expected: 格式正确
```

### Final Checklist
- [ ] Kaggle public score < 0.55
- [ ] Kaggle public score < 0.52
- [ ] 所有 PySpark 特征工程作业完成
- [ ] 集成模型 OOF RMSE < 1.05
- [ ] DeBERTa 微调 Val RMSE < 1.05
- [ ] 所有 "Must Have" present
- [ ] 所有 "Must NOT Have" absent

---

## Risk Mitigation

### 风险 1: DeBERTa 微调 OOM (12GB 显存限制)
**缓解**: 
- 使用 BS=16 + GradAcc=16 (等效 BS=256)
- 启用 FP16 混合精度
- 如果仍然 OOM，降 BS 到 12 或启用 gradient_checkpointing
- 如果仍然 OOM，使用 LoRA/QLoRA（备选方案）

### 风险 2: DeBERTa 微调仍然不佳
**缓解**: 如果 Task 6 的 OOF RMSE > 1.05，尝试：
- 增加 R-Drop 的 alpha 系数
- 调整学习率（2e-5 ~ 5e-5）
- 增加训练轮数到 7 epochs
- 如果仍然不佳，使用现有 MLP (OOF=1.131) 作为替代

### 风险 3: TF-IDF 50K 计算时间过长
**缓解**: 如果 PySpark 计算 > 2 小时，减少到 20K 特征。

### 风险 4: Ridge Stacking 过拟合
**缓解**: 使用强正则化（alpha=1.0-10.0），5-Fold 验证。

### 风险 5: Kaggle 分数未达目标
**缓解**: 如果 < 0.55 但 > 0.52，接受结果；如果 > 0.55，分析原因并调整策略。

---

## 大数据视角（COMP5434 课程要求）

### PySpark 使用点
1. Word TF-IDF 50K 特征计算（Task 2）
2. Char TF-IDF 30K 特征计算（Task 3）
3. SVD 降维（Task 4）

### 分布式优势说明
- 3M 样本的 TF-IDF 计算在单机上需要 > 10 分钟，PySpark 可以并行化
- SVD 降维可以使用 PySpark MLlib 的分布式实现
- 特征工程全部在 PySpark 完成，体现大数据处理能力

### 报告中强调
- 单机 vs PySpark 耗时对比
- 分布式特征工程的优势
- 3M 样本规模下的工程挑战
- RTX 3080 Ti 的显存限制与优化策略

---

## 前沿技术引用

1. **R-Drop**: Wu et al., 2021. "R-Drop: Regularized Dropout for Neural Networks"
   - https://arxiv.org/abs/2106.14448
2. **CORAL**: Cao et al., 2020. "Rankconsistent Ordinal Regression for Neural Networks"
   - https://arxiv.org/abs/2008.05756
3. **Sentence-BERT**: Reimers & Gurevych, 2019. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
   - https://arxiv.org/abs/1908.10084
4. **LoRA**: Hu et al., 2021. "LoRA: Low-Rank Adaptation of Large Language Models"
   - https://arxiv.org/abs/2106.09685

---

## 3080 Ti 显存估算

| 模型 | 参数量 | BS | SeqLen | 显存估算 |
|------|--------|----|--------|----------|
| DeBERTa-v3-base | 86M | 16 | 512 | 9-10GB |
| DeBERTa-v3-base | 86M | 12 | 512 | 7-8GB |
| DeBERTa-v3-base | 86M | 8 | 512 | 5-6GB |

**推荐配置**: BS=16, GradAcc=16, FP16, SeqLen=512
**如果 OOM**: BS=12, GradAcc=21, FP16, SeqLen=512
**如果仍然 OOM**: BS=8, GradAcc=32, FP16, SeqLen=512

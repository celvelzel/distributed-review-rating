# Kaggle Optimization V3 — 从 0.699 到 0.52 的突破

## TL;DR

> **Quick Summary**: 通过 DeBERTa 端到端微调、高维 TF-IDF + SVD 特征工程、以及 Ridge Stacking 集成，将 Kaggle 分数从 0.699 提升到 0.52 目标。
> 
> **Deliverables**:
> - 微调后的 DeBERTa-v3 模型（Regression Head）
> - 高维 TF-IDF (50K) + SVD (512-1024) 特征
> - Ridge Stacking 集成模型
> - 最终 Kaggle 提交文件
> 
> **Estimated Effort**: Large（约 15-20 小时）
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: DeBERTa 微调 → TF-IDF 扩展 → Ridge Stacking → 最终提交

---

## Context

### Original Request
用户希望将 Kaggle 竞赛分数从 0.699 提升到 0.52（约 25% 改进），基于详细的优化策略分析。

### Interview Summary
**Key Discussions**:
1. 当前最佳 Kaggle 分数 0.69931，集成权重 MLP=86%, LGB=9%, XGB=5%
2. 核心问题：MLP 仅使用冻结 DeBERTa Embedding，树模型提供极少增量信息
3. DeBERTa 微调效果不佳（val_rmse=1.113），可能欠拟合
4. 额外特征（sentiment, metadata）反而降低性能

**Research Findings**:
- 项目已有完善的 PySpark ETL 和特征工程流程
- TF-IDF 特征泛化最好，统计特征存在泄漏风险
- 简单加权集成优于复杂 Stacking
- MLP 虽然 OOF 不是最优，但因多样性获得最高集成权重

### Metis Review
**Identified Gaps** (addressed):
1. **缺少具体验收标准** — 已添加每阶段的 Kaggle 分数目标
2. **时间约束不明确** — 假设 1 周内完成，优先高收益任务
3. **硬件资源未确认** — 假设有 GPU（A100/V100），否则调整策略
4. **DeBERTa 微调失败原因未深究** — 添加诊断任务
5. **PySpark 要求不明确** — 特征工程全部在 PySpark 完成
6. **风险缓解不足** — 添加降级方案

---

## Work Objectives

### Core Objective
将 Kaggle 竞赛分数从 0.699 提升到 0.52，通过文本表示升级、特征工程优化、集成策略重构。

### Concrete Deliverables
1. `artifacts/models/deberta_e2e_oof.npy` — DeBERTa 端到端微调 OOF 预测
2. `artifacts/features/tfidf_50k_train.npz` — 高维 TF-IDF 特征
3. `artifacts/features/svd_1024_train.npz` — SVD 降维特征
4. `artifacts/models/ridge_stacking_oof.npy` — Ridge Stacking 集成
5. `output/submission-v3.csv` — 最终提交文件

### Definition of Done
- [ ] Kaggle public score < 0.55
- [ ] Kaggle public score < 0.52
- [ ] 所有 PySpark 特征工程作业完成
- [ ] 集成模型 OOF RMSE < 1.05

### Must Have
1. DeBERTa 端到端微调（非冻结 Embedding）
2. 高维 TF-IDF (50K) 特征
3. Ridge Stacking 集成
4. PySpark 分布式特征工程
5. 5-Fold OOF 验证

### Must NOT Have (Guardrails)
1. **不要使用统计特征**（user_stats, product_stats）— 泄漏风险
2. **不要使用 Target Encoding** — 除非有安全版本（K-Fold + Noise）
3. **不要过度调参** — 先建立强 baseline，再微调
4. **不要忽略 PySpark** — 特征工程必须体现分布式优势
5. **不要假设硬件** — 需确认 GPU 可用性

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
Wave 1 (Start Immediately — 诊断 + 特征工程):
├── Task 1: DeBERTa 微调诊断 [deep]
├── Task 2: TF-IDF 50K 特征工程 (PySpark) [unspecified-high]
├── Task 3: SVD 降维特征 (PySpark) [unspecified-high]
└── Task 4: 非文本安全特征 [quick]

Wave 2 (After Wave 1 — 模型训练):
├── Task 5: DeBERTa E2E 微调 (修正版) [deep]
├── Task 6: LightGBM + TF-IDF 50K [unspecified-high]
├── Task 7: XGBoost + TF-IDF 50K [unspecified-high]
└── Task 8: LightGBM + SVD 特征 [unspecified-high]

Wave 3 (After Wave 2 — 集成 + 提交):
├── Task 9: Ridge Stacking 集成 [deep]
├── Task 10: Optuna 权重优化 [unspecified-high]
└── Task 11: 最终 Kaggle 提交 [quick]

Wave FINAL (After ALL tasks):
├── Task F1: 计划合规审计 [oracle]
├── Task F2: 代码质量审查 [unspecified-high]
├── Task F3: 实际 QA 验证 [unspecified-high]
└── Task F4: 范围保真检查 [deep]
-> 展示结果 -> 获取用户确认
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|------------|--------|
| 1 | - | 5 |
| 2 | - | 6, 7, 8 |
| 3 | - | 8 |
| 4 | - | 6, 7 |
| 5 | 1 | 9 |
| 6 | 2, 4 | 9, 10 |
| 7 | 2, 4 | 9, 10 |
| 8 | 3 | 9, 10 |
| 9 | 5, 6, 7, 8 | 11 |
| 10 | 5, 6, 7, 8 | 11 |
| 11 | 9, 10 | F1-F4 |

### Agent Dispatch Summary

- **Wave 1**: 4 tasks — T1 → `deep`, T2-T3 → `unspecified-high`, T4 → `quick`
- **Wave 2**: 4 tasks — T5 → `deep`, T6-T8 → `unspecified-high`
- **Wave 3**: 3 tasks — T9 → `deep`, T10 → `unspecified-high`, T11 → `quick`
- **FINAL**: 4 tasks — F1 → `oracle`, F2-F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [ ] 1. DeBERTa 微调诊断

  **What to do**:
  - 检查当前 DeBERTa 微调代码（transformer_finetune.py）
  - 诊断 val_rmse=1.113 的原因：
    - 是否使用了 Mean Pooling？（当前仅 [CLS]）
    - 学习率是否过低？（当前 2e-5）
    - 是否有正则化？（Dropout, Weight Decay）
    - 训练轮数是否足够？（当前 3 epochs）
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
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4)
  - **Blocks**: Task 5
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

- [ ] 2. TF-IDF 50K 特征工程 (PySpark)

  **What to do**:
  - 使用 PySpark 计算 TF-IDF 特征
  - 参数：ngram_range=(1,3), max_features=50000
  - 保存为 `artifacts/features/tfidf_50k_train.npz` 和 `artifacts/features/tfidf_50k_test.npz`
  - 记录 PySpark 计算耗时

  **Must NOT do**:
  - 不要使用单机 Pandas
  - 不要使用统计特征

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: PySpark 特征工程，需要分布式计算知识
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4)
  - **Blocks**: Tasks 6, 7
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
  - Message: `feat(features): add TF-IDF 50K features via PySpark`
  - Files: `code/features/tfidf_50k.py`

---

- [ ] 3. SVD 降维特征 (PySpark)

  **What to do**:
  - 对 TF-IDF 50K 特征进行 TruncatedSVD 降维
  - 参数：n_components=512 或 1024（测试哪个更好）
  - 保存为 `artifacts/features/svd_512_train.npz` 和 `artifacts/features/svd_1024_train.npz`
  - 记录解释方差比

  **Must NOT do**:
  - 不要使用单机 sklearn
  - 不要跳过方差分析

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: PySpark 分布式 SVD 计算
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4)
  - **Blocks**: Task 8
  - **Blocked By**: None

  **References**:
  - `code/features/text_chartfidf.py` — 现有特征代码参考
  - PySpark MLlib TruncatedSVD 文档

  **Acceptance Criteria**:
  - [ ] SVD 特征维度 = 512 和 1024 两个版本
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
    Evidence: .sisyphus/evidence/task-3-svd-verification.txt
  ```

  **Commit**: YES
  - Message: `feat(features): add SVD降维特征 via PySpark`
  - Files: `code/features/svd_features.py`

---

- [ ] 4. 非文本安全特征

  **What to do**:
  - 提取无泄漏的文本统计特征：
    - 评论长度（字符数、词数）
    - 标点比例
    - 大写比例
    - 数字比例
  - 保存为 `artifacts/features/text_stats_train.npz`

  **Must NOT do**:
  - 不要使用用户/产品统计特征（泄漏风险）
  - 不要使用 Target Encoding

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 简单的文本统计特征
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3)
  - **Blocks**: Tasks 6, 7
  - **Blocked By**: None

  **References**:
  - `code/features/sentiment.py` — 现有特征代码参考

  **Acceptance Criteria**:
  - [ ] 特征维度 >= 4
  - [ ] 无缺失值
  - [ ] 特征分布合理

  **QA Scenarios**:

  ```
  Scenario: 文本统计特征验证
    Tool: Bash (python)
    Preconditions: 特征文件已生成
    Steps:
      1. 加载 text_stats_train.npz
      2. 检查特征维度
      3. 检查缺失值
    Expected Result: 维度 >= 4，无缺失值
    Evidence: .sisyphus/evidence/task-4-text-stats-verification.txt
  ```

  **Commit**: YES
  - Message: `feat(features): add safe text statistics features`
  - Files: `code/features/text_stats.py`

---

- [ ] 5. DeBERTa E2E 微调 (修正版)

  **What to do**:
  - 基于 Task 1 的诊断结果，修正 DeBERTa 微调代码
  - 关键修正点：
    - 使用 Mean Pooling 替代 [CLS]
    - 增加学习率到 5e-5
    - 增加训练轮数到 5 epochs
    - 添加 Multi-Sample Dropout
    - 使用 Huber Loss 替代 MSE
  - 5-Fold OOF 验证
  - 保存 `artifacts/models/deberta_e2e_oof.npy`

  **Must NOT do**:
  - 不要跳过诊断直接修改
  - 不要使用过大的学习率（> 1e-4）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 复杂的模型微调，需要深度理解
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential after Task 1)
  - **Blocks**: Tasks 9, 10
  - **Blocked By**: Task 1

  **References**:
  - `code/models/transformer_finetune.py` — 当前微调代码
  - Task 1 的诊断报告

  **Acceptance Criteria**:
  - [ ] OOF RMSE < 1.05
  - [ ] OOF RMSE < 1.00（目标）
  - [ ] 5-Fold 验证完成
  - [ ] 模型文件保存

  **QA Scenarios**:

  ```
  Scenario: DeBERTa 微调效果验证
    Tool: Bash (python)
    Preconditions: 微调完成
    Steps:
      1. 加载 deberta_e2e_oof.npy
      2. 计算 RMSE
      3. 与 baseline 对比
    Expected Result: RMSE < 1.05
    Evidence: .sisyphus/evidence/task-5-deberta-rmse.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add fine-tuned DeBERTa with Mean Pooling`
  - Files: `code/models/transformer_e2e.py`

---

- [ ] 6. LightGBM + TF-IDF 50K

  **What to do**:
  - 使用 TF-IDF 50K 特征训练 LightGBM
  - 添加文本统计特征（Task 4）
  - Optuna 超参数优化（100 trials）
  - 5-Fold OOF 验证
  - 保存 `artifacts/models/lgb_tfidf50k_oof.npy`

  **Must NOT do**:
  - 不要使用统计特征（user_stats, product_stats）
  - 不要过拟合

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要 Optuna 超参数优化
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 7, 8)
  - **Blocks**: Tasks 9, 10
  - **Blocked By**: Tasks 2, 4

  **References**:
  - `code/models/optuna_lgb_tune.py` — 现有 Optuna 代码
  - `artifacts/features/tfidf_50k_train.npz` — TF-IDF 特征

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
      1. 加载 lgb_tfidf50k_oof.npy
      2. 计算 RMSE
      3. 检查 Optuna 日志
    Expected Result: RMSE < 1.10
    Evidence: .sisyphus/evidence/task-6-lgb-rmse.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add LightGBM with TF-IDF 50K`
  - Files: `code/models/lgb_tfidf50k.py`

---

- [ ] 7. XGBoost + TF-IDF 50K

  **What to do**:
  - 使用 TF-IDF 50K 特征训练 XGBoost
  - 添加文本统计特征（Task 4）
  - Optuna 超参数优化（100 trials）
  - 5-Fold OOF 验证
  - 保存 `artifacts/models/xgb_tfidf50k_oof.npy`

  **Must NOT do**:
  - 不要使用统计特征
  - 不要过拟合

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要 Optuna 超参数优化
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6, 8)
  - **Blocks**: Tasks 9, 10
  - **Blocked By**: Tasks 2, 4

  **References**:
  - `code/models/xgboost_train.py` — 现有 XGBoost 代码
  - `artifacts/features/tfidf_50k_train.npz` — TF-IDF 特征

  **Acceptance Criteria**:
  - [ ] OOF RMSE < 1.10
  - [ ] Optuna 100 trials 完成

  **QA Scenarios**:

  ```
  Scenario: XGBoost 效果验证
    Tool: Bash (python)
    Preconditions: 训练完成
    Steps:
      1. 加载 xgb_tfidf50k_oof.npy
      2. 计算 RMSE
    Expected Result: RMSE < 1.10
    Evidence: .sisyphus/evidence/task-7-xgb-rmse.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add XGBoost with TF-IDF 50K`
  - Files: `code/models/xgb_tfidf50k.py`

---

- [ ] 8. LightGBM + SVD 特征

  **What to do**:
  - 使用 SVD 512/1024 特征训练 LightGBM
  - Optuna 超参数优化
  - 5-Fold OOF 验证
  - 保存 `artifacts/models/lgb_svd_oof.npy`

  **Must NOT do**:
  - 不要使用统计特征

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要 Optuna 超参数优化
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6, 7)
  - **Blocks**: Tasks 9, 10
  - **Blocked By**: Task 3

  **References**:
  - `artifacts/features/svd_512_train.npz` — SVD 特征

  **Acceptance Criteria**:
  - [ ] OOF RMSE < 1.15
  - [ ] Optuna 完成

  **QA Scenarios**:

  ```
  Scenario: LightGBM + SVD 效果验证
    Tool: Bash (python)
    Preconditions: 训练完成
    Steps:
      1. 加载 lgb_svd_oof.npy
      2. 计算 RMSE
    Expected Result: RMSE < 1.15
    Evidence: .sisyphus/evidence/task-8-lgb-svd-rmse.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add LightGBM with SVD features`
  - Files: `code/models/lgb_svd.py`

---

- [ ] 9. Ridge Stacking 集成

  **What to do**:
  - 使用 Ridge Regression 作为 Meta-Learner
  - 输入：所有模型的 OOF 预测 + SVD 特征
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
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 10, 11)
  - **Blocks**: Task 11
  - **Blocked By**: Tasks 5, 6, 7, 8

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
    Evidence: .sisyphus/evidence/task-9-stacking-rmse.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add Ridge Stacking ensemble`
  - Files: `code/models/ridge_stacking.py`

---

- [ ] 10. Optuna 权重优化

  **What to do**:
  - 使用 Optuna 优化集成权重
  - 搜索空间：所有模型的权重
  - 目标：最小化 OOF RMSE
  - 1000 trials
  - 保存最佳权重

  **Must NOT do**:
  - 不要使用太少 trials

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要 Optuna 优化
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 9, 11)
  - **Blocks**: Task 11
  - **Blocked By**: Tasks 5, 6, 7, 8

  **References**:
  - `code/models/optuna_ensemble.py` — 现有 Optuna 代码

  **Acceptance Criteria**:
  - [ ] 1000 trials 完成
  - [ ] 最佳权重记录

  **QA Scenarios**:

  ```
  Scenario: Optuna 优化验证
    Tool: Bash (python)
    Preconditions: 优化完成
    Steps:
      1. 检查 Optuna 日志
      2. 验证 trials 数量
    Expected Result: 1000 trials 完成
    Evidence: .sisyphus/evidence/task-10-optuna-log.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add Optuna weight optimization`
  - Files: `code/models/optuna_weight_opt.py`

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
  - **Parallel Group**: Wave 3 (after Tasks 9, 10)
  - **Blocks**: F1-F4
  - **Blocked By**: Tasks 9, 10

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
| 2 | `feat(features): add TF-IDF 50K features via PySpark` | `code/features/tfidf_50k.py` |
| 3 | `feat(features): add SVD降维特征 via PySpark` | `code/features/svd_features.py` |
| 4 | `feat(features): add safe text statistics features` | `code/features/text_stats.py` |
| 5 | `feat(models): add fine-tuned DeBERTa with Mean Pooling` | `code/models/transformer_e2e.py` |
| 6 | `feat(models): add LightGBM with TF-IDF 50K` | `code/models/lgb_tfidf50k.py` |
| 7 | `feat(models): add XGBoost with TF-IDF 50K` | `code/models/xgb_tfidf50k.py` |
| 8 | `feat(models): add LightGBM with SVD features` | `code/models/lgb_svd.py` |
| 9 | `feat(models): add Ridge Stacking ensemble` | `code/models/ridge_stacking.py` |
| 10 | `feat(models): add Optuna weight optimization` | `code/models/optuna_weight_opt.py` |
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
- [ ] 所有 "Must Have" present
- [ ] 所有 "Must NOT Have" absent

---

## Risk Mitigation

### 风险 1: DeBERTa 微调仍然不佳
**缓解**: 如果 Task 5 的 OOF RMSE > 1.05，跳过 DeBERTa，使用现有 MLP (OOF=1.131) 作为替代。

### 风险 2: TF-IDF 50K 计算时间过长
**缓解**: 如果 PySpark 计算 > 2 小时，减少到 20K 特征。

### 风险 3: Ridge Stacking 过拟合
**缓解**: 使用强正则化（alpha=1.0-10.0），5-Fold 验证。

### 风险 4: Kaggle 分数未达目标
**缓解**: 如果 < 0.55 但 > 0.52，接受结果；如果 > 0.55，分析原因并调整策略。

---

## 大数据视角（COMP5434 课程要求）

### PySpark 使用点
1. TF-IDF 50K 特征计算（Task 2）
2. SVD 降维（Task 3）
3. 文本统计特征（Task 4）

### 分布式优势说明
- 3M 样本的 TF-IDF 计算在单机上需要 > 10 分钟，PySpark 可以并行化
- SVD 降维可以使用 PySpark MLlib 的分布式实现
- 特征工程全部在 PySpark 完成，体现大数据处理能力

### 报告中强调
- 单机 vs PySpark 耗时对比
- 分布式特征工程的优势
- 3M 样本规模下的工程挑战

# Kaggle Optimization Gemini — 从 0.699 到 0.52 的突破

## TL;DR

> **Quick Summary**: 通过 LoRA 高效微调 DeBERTa、PySpark ALS/Doc2Vec 协同过滤特征、CORAL 序数回归损失、Cross-Attention 多模态融合四大前沿策略，打破当前 DeBERTa 冻结嵌入主导的特征瓶颈，将 Kaggle 分数从 0.699 推向 0.52。
> 
> **Deliverables**:
> - LoRA 微调的 DeBERTa 模型（显存降低 60%+）
> - ALS 用户/商品潜向量特征（32/64 维）
> - Doc2Vec 用户历史语言风格向量
> - CORAL 序数回归模型
> - Cross-Attention 多模态融合模型
> - SVD/PCA 降维后的稠密文本特征
> - 新的集成模型和 Kaggle 提交
> 
> **Estimated Effort**: XL
> **Parallel Execution**: YES - 5 waves
> **Critical Path**: Task 1 (LoRA DeBERTa) → Task 6 (CORAL 损失) → Task 9 (Cross-Attention) → Task 12 (最终集成)

---

## Context

### Original Request
用户希望将 Kaggle 评分从 0.69931 优化到 0.52（竞争对手分数），提出了四大优化策略：
1. PySpark ALS 用户-商品交互特征
2. SVD/PCA 文本特征降维
3. 回归转分类（期望值预测）
4. 安全目标编码（平滑 + 噪声）

**新增技术栈（基于 RTX 3080 Ti 12GB 显存优化）**：
5. LoRA/QLoRA 高效参数微调（显存降低 60%+）
6. FlashAttention-2 + PyTorch SDPA（注意力计算加速 2-3x）
7. BF16 混合精度训练（比 FP16 更稳定）
8. CORAL 序数回归损失（RMSE 提升 2-5%）
9. PySpark Word2Vec/Doc2Vec 用户历史语言风格向量
10. Feature Tokenizer + Cross-Attention 多模态融合

### Interview Summary
**Key Discussions**:
- 当前 MLP 占 86% 权重但仅依赖 DeBERTa 冻结嵌入，下游 Head 容量有限
- 树模型（LGB/XGB）在 5000 维稀疏 TF-IDF 上效率低下
- LightGCN 训练失败（嵌入范数趋近于 0）
- 目标编码遇到数据泄漏
- DeBERTa 微调欠拟合（val_rmse=1.113）
- **硬件约束**：RTX 3080 Ti 12GB 显存，全参数微调易 OOM
- **max_length 优化**：电商短评有效信息集中在前 100 多词，建议从 512 缩减到 128/256

**Research Findings**:
- 对抗验证 AUC=0.5235，无显著分布偏移
- PySpark 基础设施已就绪
- 3M 训练样本，10K 测试样本
- 课程要求：特征工程在 PySpark 上完成
- **LoRA 微调**：冻结原型权重，仅在 Attention 层引入低秩矩阵（r=8/16），显存降低 60%+
- **BF16 支持**：3080 Ti 原生支持 BF16，比 FP16 更稳定，不会梯度下陷
- **CORAL 损失**：将 5 分类重构为 4 个二分类（是否>1星、>2星、>3星、>4星），严格对齐 RMSE 距离惩罚
- **Doc2Vec 特征**：用户历史评论语言风格向量，天然免疫目标泄漏

### Metis Review
**Identified Gaps** (addressed):
- **ALS 冷启动问题**：需为未见过的用户/商品提供默认嵌入（全局均值或零向量）
- **分类任务标签分布不均**：需使用分层采样确保各折分布一致
- **SVD 组件数选择**：需通过方差解释比例确定最优维度
- **目标编码平滑参数**：需通过交叉验证选择最优 m 值
- **计算资源约束**：ALS 在 3M 样本上的训练时间需评估
- **课程要求**：需记录 PySpark vs 单机 Pandas 的耗时对比
- **显存约束**：RTX 3080 Ti 12GB，需使用 LoRA + BF16 + 梯度累积
- **max_length 选择**：电商短评建议 128 或 256，需实验验证
- **CORAL 单调性**：需确保 4 个二分类阈值的单调性约束

---

## Work Objectives

### Core Objective
通过 LoRA 高效微调 DeBERTa、PySpark ALS/Doc2Vec 协同过滤特征、CORAL 序数回归损失、Cross-Attention 多模态融合四大前沿策略，打破当前 DeBERTa 冻结嵌入主导的特征瓶颈，将 Kaggle 分数从 0.699 推向 0.52。

### Concrete Deliverables
- `artifacts/models/deberta_lora/` — LoRA 微调的 DeBERTa 模型
- `artifacts/features/als_user_factors.npy` — ALS 用户潜向量（32/64 维）
- `artifacts/features/als_item_factors.npy` — ALS 商品潜向量（32/64 维）
- `artifacts/features/user_doc2vec.npy` — 用户历史评论 Doc2Vec 向量
- `artifacts/features/item_doc2vec.npy` — 商品历史评价 Doc2Vec 向量
- `artifacts/features/tfidf_svd.npy` — TF-IDF SVD 降维特征（64/128 维）
- `artifacts/features/bert_pca.npy` — DeBERTa PCA 降维特征（64 维）
- `artifacts/features/target_encoded.npy` — 安全目标编码特征
- `artifacts/models/coral_oof.npy` — CORAL 序数回归模型 OOF 预测
- `artifacts/models/cross_attn_oof.npy` — Cross-Attention 融合模型 OOF 预测
- `artifacts/models/ensemble_v4_oof.npy` — 新集成模型 OOF 预测
- `output/submission-v4.csv` — 最终 Kaggle 提交

### Definition of Done
- [ ] Kaggle score < 0.65
- [ ] Kaggle score < 0.60
- [ ] Kaggle score < 0.55
- [ ] Kaggle score < 0.52 (beat competitor)

### Must Have
- PySpark ALS 实现（课程要求）
- PySpark Doc2Vec 用户历史语言风格特征（课程要求）
- LoRA 高效微调 DeBERTa（RTX 3080 Ti 12GB 显存约束）
- BF16 混合精度训练（3080 Ti 原生支持）
- CORAL 序数回归损失（比 MSE/CrossEntropy 更适合评分预测）
- 数据泄漏防护（K-Fold 验证）
- 冷启动处理（未见过的用户/商品）
- OOF RMSE 与 Kaggle 分数对齐验证
- PySpark vs 单机 Pandas 耗时对比记录（课程报告）

### Must NOT Have (Guardrails)
- **禁止**在 ALS 训练中使用测试集数据
- **禁止**在目标编码中使用当前折的标签
- **禁止**未经验证的特征直接加入集成
- **禁止**过度调参导致过拟合测试集
- **禁止**跳过 OOF 验证直接提交 Kaggle
- **禁止**使用 FP16（3080 Ti 对 BF16 支持更好，FP16 易梯度下陷）
- **禁止**全参数微调 DeBERTa（12GB 显存不够，必须用 LoRA）
- **禁止**max_length > 256（电商短评有效信息集中在前 100 多词）

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest in code/tests/)
- **Automated tests**: Tests-after
- **Framework**: pytest
- **Test Focus**: 数据泄漏检测、特征维度验证、OOF RMSE 阈值

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Feature Engineering**: Use Bash (python) — 运行脚本，验证输出维度和值范围
- **Model Training**: Use Bash (python) — 训练模型，验证 OOF RMSE
- **Ensemble**: Use Bash (python) — 集成预测，验证 Kaggle 提交格式

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — 特征工程 + LoRA 微调，最大并行):
├── Task 1: LoRA + BF16 + SDPA DeBERTa 微调 [deep]
├── Task 2: PySpark ALS 用户-商品潜向量 [unspecified-high]
├── Task 3: PySpark Doc2Vec 用户历史语言风格 [unspecified-high]
├── Task 4: TF-IDF SVD/LSA 降维 [quick]
├── Task 5: DeBERTa PCA 降维 [quick]
└── Task 6: 冷启动特征分析 [quick]

Wave 2 (After Wave 1 — CORAL 损失 + 稠密特征模型):
├── Task 7: CORAL 序数回归模型（DeBERTa LoRA）[deep]
├── Task 8: LightGBM + ALS + Doc2Vec 特征 [unspecified-high]
├── Task 9: LightGBM + SVD 特征 [unspecified-high]
└── Task 10: 安全目标编码（平滑 + 噪声）[unspecified-high]

Wave 3 (After Wave 2 — Cross-Attention 多模态融合):
├── Task 11: Feature Tokenizer + Cross-Attention 融合模型 [deep]
└── Task 12: 5 分类期望值模型（LightGBM）[unspecified-high]

Wave 4 (After Wave 3 — 集成优化):
├── Task 13: 新集成策略（加权平均 + Stacking）[unspecified-high]
├── Task 14: Optuna 权重优化 [unspecified-high]
└── Task 15: 最终 Kaggle 提交 [quick]

Wave FINAL (After ALL tasks — 验证):
├── Task F1: 计划合规审计 [oracle]
├── Task F2: 代码质量审查 [unspecified-high]
├── Task F3: 实际 QA 执行 [unspecified-high]
└── Task F4: 范围保真度检查 [deep]
-> 呈现结果 -> 获取用户确认

Critical Path: Task 1 → Task 7 → Task 11 → Task 13 → Task 15 → F1-F4 → 用户确认
Parallel Speedup: ~65% faster than sequential
Max Concurrent: 6 (Wave 1)
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|------------|--------|
| 1 | — | 7, 11 |
| 2 | — | 8, 11 |
| 3 | — | 8, 11 |
| 4 | — | 9, 11 |
| 5 | — | 11 |
| 6 | — | 8, 9, 12 |
| 7 | 1 | 13 |
| 8 | 2, 3, 6 | 13 |
| 9 | 4, 6 | 13 |
| 10 | — | 13 |
| 11 | 1, 2, 3, 4, 5 | 13 |
| 12 | 6 | 13 |
| 13 | 7, 8, 9, 10, 11, 12 | 14, 15 |
| 14 | 13 | 15 |
| 15 | 14 | F1-F4 |

### Agent Dispatch Summary

- **Wave 1**: 6 tasks — T1 → `deep`, T2-T3 → `unspecified-high`, T4-T6 → `quick`
- **Wave 2**: 4 tasks — T7 → `deep`, T8-T9 → `unspecified-high`, T10 → `unspecified-high`
- **Wave 3**: 2 tasks — T11 → `deep`, T12 → `unspecified-high`
- **Wave 4**: 3 tasks — T13-T14 → `unspecified-high`, T15 → `quick`
- **FINAL**: 4 tasks — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

### Wave 1: 特征工程 + LoRA 微调（最大并行）

- [ ] 1. LoRA + BF16 + SDPA DeBERTa 微调

  **What to do**:
  - 修改 transformer_finetune.py，引入 LoRA 高效微调
  - 冻结 DeBERTa 原型权重，仅在 Attention 层引入低秩矩阵（r=8 或 r=16）
  - 启用 BF16 混合精度训练（3080 Ti 原生支持）
  - 加载 DeBERTa 时开启 attn_implementation="sdpa"（PyTorch 2.0+ Scaled Dot-Product Attention）
  - 将 max_length 从 512 缩减到 128（电商短评有效信息集中在前 100 多词）
  - 使用梯度累积（gradient_accumulation_steps=4）扩大等效 Batch Size
  - 5-Fold OOF 验证
  - 记录显存占用和训练耗时
  - 保存 OOF 和测试预测

  **Must NOT do**:
  - 禁止全参数微调（12GB 显存不够）
  - 禁止使用 FP16（易梯度下陷）
  - 禁止 max_length > 256

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: LoRA 微调涉及 PEFT 库集成、模型架构修改、显存优化，需要深入的 NLP 和 PyTorch 知识
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4, 5, 6)
  - **Blocks**: Tasks 7, 11
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `code/models/transformer_finetune.py` — 现有 DeBERTa 微调代码，需要修改为 LoRA
  - `code/models/mlp.py` — MLP 模型架构参考

  **API/Type References**:
  - `data/train.csv` — 训练数据
  - `artifacts/etl/train.parquet` — ETL 处理后的训练数据

  **External References**:
  - PEFT LoRA 文档: https://huggingface.co/docs/peft/conceptual_guides/lora
  - PyTorch SDPA: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
  - DeBERTa-v3: https://huggingface.co/microsoft/deberta-v3-small

  **WHY Each Reference Matters**:
  - transformer_finetune.py: 现有微调代码，需要修改为 LoRA + BF16 + SDPA
  - PEFT LoRA: 高效参数微调的核心库
  - PyTorch SDPA: 注意力机制加速

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: LoRA 微调训练成功
    Tool: Bash (python)
    Preconditions: DeBERTa 模型已下载
    Steps:
      1. 运行 LoRA 微调脚本
      2. 检查显存占用是否 < 10GB（留 2GB 余量）
      3. 检查训练是否在 24 小时内完成（5-Fold）
      4. 检查 OOF RMSE 是否 < 1.05
      5. 验证 LoRA 权重是否正确保存
    Expected Result: 训练成功，显存 < 10GB，OOF RMSE < 1.05
    Failure Indicators: OOM、训练超时、OOF RMSE > 1.05
    Evidence: .sisyphus/evidence/task-1-lora-training.txt

  Scenario: BF16 精度验证
    Tool: Bash (python)
    Preconditions: LoRA 微调完成
    Steps:
      1. 检查模型参数是否为 BF16 格式
      2. 验证推理输出是否正常（无 NaN/Inf）
    Expected Result: BF16 格式正确，推理正常
    Failure Indicators: 参数格式错误、输出异常
    Evidence: .sisyphus/evidence/task-1-bf16-verify.txt

  Scenario: max_length 优化验证
    Tool: Bash (python)
    Preconditions: 训练数据已加载
    Steps:
      1. 统计训练集评论长度分布
      2. 验证 max_length=128 是否覆盖 95% 的评论
      3. 检查截断后的信息损失
    Expected Result: max_length=128 覆盖 > 95% 评论
    Failure Indicators: 覆盖率 < 90%
    Evidence: .sisyphus/evidence/task-1-maxlength.txt
  ```

  **Commit**: YES
  - Message: `feat(lora): add LoRA + BF16 + SDPA DeBERTa fine-tuning`
  - Files: `code/models/deberta_lora.py`
  - Pre-commit: `python code/models/deberta_lora.py`

- [ ] 2. PySpark ALS 用户-商品潜向量

  **What to do**:
  - 使用 PySpark ALS 训练用户-商品矩阵分解
  - 仅使用训练集的 (user_id, item_id, rating) 训练
  - 提取 User Factors 和 Item Factors（32 维和 64 维两版）
  - 为未见过的用户/商品提供默认嵌入（全局均值向量）
  - 保存为 artifacts/features/als_user_factors.npy 和 als_item_factors.npy
  - 记录 PySpark 训练耗时

  **Must NOT do**:
  - 禁止使用测试集数据训练 ALS
  - 禁止跳过冷启动处理
  - 禁止不记录耗时

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: PySpark ALS 实现涉及分布式计算和矩阵分解，需要较强的工程能力
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4, 5, 6)
  - **Blocks**: Tasks 8, 11
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `code/utils/spark_session.py` — PySpark 会话配置，使用 get_spark() 获取 SparkSession
  - `code/features/rating_deviation.py` — PySpark 特征工程模式，K-Fold 验证模式

  **API/Type References**:
  - `data/train.csv` — 训练数据，包含 id, user_id, parent_prod_id, rating 等列
  - `data/test.csv` — 测试数据，包含 id, user_id, parent_prod_id 等列

  **External References**:
  - PySpark ALS 文档: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.recommendation.ALS.html

  **WHY Each Reference Matters**:
  - spark_session.py: 展示如何正确配置 PySpark 会话
  - rating_deviation.py: 展示 K-Fold 验证模式，防止数据泄漏

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: ALS 特征生成成功
    Tool: Bash (python)
    Preconditions: artifacts/features/ 目录存在
    Steps:
      1. 运行 ALS 特征生成脚本
      2. 检查 als_user_factors.npy 和 als_item_factors.npy 是否生成
      3. 验证维度：User factors 应为 (3007439, 32) 或 (3007439, 64)
      4. 验证值范围：应在 [-1, 1] 之间
      5. 检查是否有 NaN 或 Inf
    Expected Result: 特征文件生成成功，维度正确，无异常值
    Failure Indicators: 文件不存在、维度错误、包含 NaN/Inf
    Evidence: .sisyphus/evidence/task-2-als-features.txt

  Scenario: 冷启动处理验证
    Tool: Bash (python)
    Preconditions: ALS 模型训练完成
    Steps:
      1. 检查测试集中未见过的用户/商品比例
      2. 验证这些用户的嵌入是否为全局均值
    Expected Result: 未见过的用户/商品使用全局均值填充
    Failure Indicators: 嵌入为零或随机值
    Evidence: .sisyphus/evidence/task-2-cold-start.txt

  Scenario: PySpark 耗时记录
    Tool: Bash (python)
    Preconditions: ALS 训练完成
    Steps:
      1. 检查日志中是否记录了 PySpark ALS 训练耗时
      2. 记录总耗时
    Expected Result: 耗时记录存在，可用于课程报告
    Failure Indicators: 无耗时记录
    Evidence: .sisyphus/evidence/task-2-timing.txt
  ```

  **Commit**: YES
  - Message: `feat(als): add PySpark ALS user-item latent factors`
  - Files: `code/features/als_features.py`
  - Pre-commit: `python code/features/als_features.py`

- [ ] 3. PySpark Doc2Vec 用户历史语言风格

  **What to do**:
  - 使用 PySpark Word2Vec/Doc2Vec 训练用户历史评论语言风格向量
  - 将同一个 user_id 过去写过的所有历史评论拼接在一起
  - 训练 Doc2Vec 模型，提取用户"历史语言风格向量"
  - 同样为商品提取"历史评价偏向向量"
  - 降维到 32 或 64 维
  - 保存为 artifacts/features/user_doc2vec.npy 和 item_doc2vec.npy
  - 记录 PySpark 训练耗时

  **Must NOT do**:
  - 禁止使用当前样本的标签（天然免疫目标泄漏）
  - 禁止不记录耗时

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: PySpark Word2Vec/Doc2Vec 实现涉及分布式 NLP，需要较强的工程能力
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4, 5, 6)
  - **Blocks**: Tasks 8, 11
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `code/utils/spark_session.py` — PySpark 会话配置
  - `code/features/rating_deviation.py` — PySpark 特征工程模式

  **API/Type References**:
  - `data/train.csv` — 训练数据，包含 user_id, parent_prod_id, title, comment
  - `data/test.csv` — 测试数据

  **External References**:
  - PySpark Word2Vec: https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Word2Vec.html

  **WHY Each Reference Matters**:
  - spark_session.py: PySpark 会话配置
  - train.csv: 确认数据格式

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Doc2Vec 特征生成成功
    Tool: Bash (python)
    Preconditions: 训练数据已加载
    Steps:
      1. 运行 Doc2Vec 特征生成脚本
      2. 检查 user_doc2vec.npy 和 item_doc2vec.npy 是否生成
      3. 验证维度：应为 (N, 32) 或 (N, 64)
      4. 检查是否有 NaN 或 Inf
    Expected Result: 特征文件生成成功，维度正确，无异常值
    Failure Indicators: 文件不存在、维度错误、包含 NaN/Inf
    Evidence: .sisyphus/evidence/task-3-doc2vec-features.txt

  Scenario: 天然免疫目标泄漏验证
    Tool: Bash (python)
    Preconditions: Doc2Vec 特征已生成
    Steps:
      1. 检查特征计算是否使用了当前样本的标签
      2. 验证特征仅基于历史评论文本
    Expected Result: 特征不依赖当前样本标签
    Failure Indicators: 特征计算中使用了 rating 列
    Evidence: .sisyphus/evidence/task-3-no-leakage.txt

  Scenario: PySpark 耗时记录
    Tool: Bash (python)
    Preconditions: Doc2Vec 训练完成
    Steps:
      1. 检查日志中是否记录了 PySpark Doc2Vec 训练耗时
      2. 记录总耗时
    Expected Result: 耗时记录存在，可用于课程报告
    Failure Indicators: 无耗时记录
    Evidence: .sisyphus/evidence/task-3-timing.txt
  ```

  **Commit**: YES
  - Message: `feat(doc2vec): add PySpark Doc2Vec user language style`
  - Files: `code/features/doc2vec_features.py`
  - Pre-commit: `python code/features/doc2vec_features.py`

- [ ] 4. TF-IDF SVD/LSA 降维

  **What to do**:
  - 使用 TruncatedSVD 对 TF-IDF 5000 维特征进行降维
  - 尝试 64 和 128 两种维度
  - 选择方差解释比例最高的维度
  - 将稀疏矩阵转换为稠密矩阵
  - 保存为 artifacts/features/tfidf_svd_64.npy 和 tfidf_svd_128.npy

  **Must NOT do**:
  - 禁止使用 PCA（TF-IDF 是稀疏矩阵，需用 TruncatedSVD）
  - 禁止不检查方差解释比例

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: SVD 降维是标准操作，sklearn 实现简单
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 5, 6)
  - **Blocks**: Tasks 9, 11
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `code/features/text_chartfidf.py` — TF-IDF 特征生成模式
  - `code/models/ensemble_diverse.py:60-61` — TF-IDF 特征加载路径

  **API/Type References**:
  - `artifacts/features/chartfidf_train.npz` — 已生成的 TF-IDF 稀疏矩阵
  - `artifacts/features/chartfidf_test.npz` — 测试集 TF-IDF 稀疏矩阵

  **External References**:
  - sklearn TruncatedSVD: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

  **WHY Each Reference Matters**:
  - text_chartfidf.py: 了解 TF-IDF 特征的生成方式和格式
  - ensemble_diverse.py: 确认 TF-IDF 特征的文件路径

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: SVD 降维成功
    Tool: Bash (python)
    Preconditions: artifacts/features/chartfidf_train.npz 存在
    Steps:
      1. 运行 SVD 降维脚本
      2. 检查 tfidf_svd_64.npy 和 tfidf_svd_128.npy 是否生成
      3. 验证维度：应为 (3007439, 64) 和 (3007439, 128)
      4. 检查方差解释比例是否记录
    Expected Result: 特征文件生成成功，方差解释比例 > 0.5
    Failure Indicators: 文件不存在、维度错误、方差解释比例过低
    Evidence: .sisyphus/evidence/task-4-svd-features.txt
  ```

  **Commit**: YES
  - Message: `feat(svd): add TF-IDF SVD dimensionality reduction`
  - Files: `code/features/svd_features.py`
  - Pre-commit: `python code/features/svd_features.py`

- [ ] 5. DeBERTa PCA 降维

  **What to do**:
  - 使用 PCA 对 DeBERTa 768 维嵌入进行降维
  - 降维到 64 维
  - 保存为 artifacts/features/bert_pca_64.npy
  - 记录方差解释比例

  **Must NOT do**:
  - 禁止不检查方差解释比例
  - 禁止降维到过低维度（如 < 32）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: PCA 降维是标准操作，sklearn 实现简单
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 4, 6)
  - **Blocks**: Task 11
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `code/models/run_mlp.py:48-49` — BERT 嵌入加载路径

  **API/Type References**:
  - `artifacts/features/bert_train.parquet` — DeBERTa 768 维嵌入（训练集）
  - `artifacts/features/bert_test.parquet` — DeBERTa 768 维嵌入（测试集）

  **External References**:
  - sklearn PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

  **WHY Each Reference Matters**:
  - run_mlp.py: 确认 BERT 嵌入的文件路径和格式

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: PCA 降维成功
    Tool: Bash (python)
    Preconditions: artifacts/features/bert_train.parquet 存在
    Steps:
      1. 运行 PCA 降维脚本
      2. 检查 bert_pca_64.npy 是否生成
      3. 验证维度：应为 (3007439, 64)
      4. 检查方差解释比例是否记录
    Expected Result: 特征文件生成成功，方差解释比例 > 0.7
    Failure Indicators: 文件不存在、维度错误、方差解释比例过低
    Evidence: .sisyphus/evidence/task-5-pca-features.txt
  ```

  **Commit**: YES
  - Message: `feat(pca): add DeBERTa PCA dimensionality reduction`
  - Files: `code/features/pca_features.py`
  - Pre-commit: `python code/features/pca_features.py`

- [ ] 6. 冷启动特征分析

  **What to do**:
  - 分析测试集中冷启动用户/商品的比例
  - 冷启动用户：训练集中从未出现过的 user_id
  - 冷启动商品：训练集中从未出现过的 parent_prod_id
  - 统计冷启动样本的评分分布（如果有）
  - 为冷启动策略提供数据支持

  **Must NOT do**:
  - 禁止不记录冷启动比例

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 数据分析任务，pandas 操作简单
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 4, 5)
  - **Blocks**: Tasks 8, 9, 12
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `data/train.csv` — 训练数据
  - `data/test.csv` — 测试数据

  **API/Type References**:
  - train.csv 列: id, user_id, parent_prod_id, rating, title, comment, votes, purchased, time
  - test.csv 列: id, user_id, parent_prod_id, title, comment, votes, purchased, time

  **WHY Each Reference Matters**:
  - train.csv/test.csv: 确认数据格式和列名

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 冷启动比例分析
    Tool: Bash (python)
    Preconditions: 训练和测试数据已加载
    Steps:
      1. 加载 train.csv 和 test.csv
      2. 提取训练集和测试集的 user_id 和 parent_prod_id
      3. 计算冷启动用户比例
      4. 计算冷启动商品比例
      5. 输出冷启动比例和样本数
    Expected Result: 冷启动比例记录，用于指导冷启动策略
    Failure Indicators: 无法加载数据或计算错误
    Evidence: .sisyphus/evidence/task-6-cold-start.txt
  ```

  **Commit**: YES
  - Message: `feat(analysis): add cold-start feature analysis`
  - Files: `code/features/cold_start_analysis.py`
  - Pre-commit: `python code/features/cold_start_analysis.py`

### Wave 2: CORAL 损失 + 稠密特征模型

- [ ] 7. CORAL 序数回归模型（DeBERTa LoRA）

  **What to do**:
  - 实现 CORAL (Consistent Rank Logits) 损失函数
  - 将 5 分类任务重构为 4 个二分类任务（是否 >1 星？是否 >2 星？是否 >3 星？是否 >4 星？）
  - 网络输出 4 个 Logits，共享最后一层权重以保证各档位阈值的单调性
  - 使用 LoRA 微调的 DeBERTa 作为编码器
  - 5-Fold OOF 验证
  - 记录 OOF RMSE
  - 保存 OOF 和测试预测

  **Must NOT do**:
  - 禁止不保证阈值单调性
  - 禁止使用纯 MSE 或 CrossEntropy（CORAL 更适合序数回归）

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: CORAL 损失实现涉及序数回归理论和自定义损失函数，需要深入的 ML 知识
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 8, 9, 10)
  - **Blocks**: Task 13
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `code/models/transformer_finetune.py` — DeBERTa 微调流程
  - `code/models/mlp.py` — MLP 模型架构

  **External References**:
  - CORAL 论文: "Rank-consistent ordinal regression for neural networks"
  - PyTorch 自定义损失: https://pytorch.org/docs/stable/notes/extending.html

  **WHY Each Reference Matters**:
  - transformer_finetune.py: 展示 DeBERTa 微调流程，需要修改损失函数
  - CORAL 论文: 序数回归的理论基础

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: CORAL 损失训练成功
    Tool: Bash (python)
    Preconditions: LoRA DeBERTa 模型已训练
    Steps:
      1. 运行 CORAL 模型训练脚本
      2. 检查 coral_oof.npy 和 coral_test.npy 是否生成
      3. 验证 OOF RMSE 是否 < 1.00
      4. 验证预测值范围是否在 [1, 5] 之间
      5. 检查 4 个 Logits 的单调性
    Expected Result: 模型训练成功，OOF RMSE < 1.00，预测值范围正确
    Failure Indicators: 训练失败、OOF RMSE > 1.00、预测值超出范围
    Evidence: .sisyphus/evidence/task-7-coral-training.txt

  Scenario: 单调性验证
    Tool: Bash (python)
    Preconditions: CORAL 模型训练完成
    Steps:
      1. 加载模型输出的 4 个 Logits
      2. 验证 P(>1) >= P(>2) >= P(>3) >= P(>4)
    Expected Result: 单调性约束满足
    Failure Indicators: 单调性违反
    Evidence: .sisyphus/evidence/task-7-monotonicity.txt
  ```

  **Commit**: YES
  - Message: `feat(coral): add CORAL ordinal regression model`
  - Files: `code/models/coral_model.py`
  - Pre-commit: `python code/models/coral_model.py`

- [ ] 8. LightGBM + ALS + Doc2Vec 特征

  **What to do**:
  - 使用 ALS 用户/商品潜向量 + Doc2Vec 用户历史语言风格作为特征训练 LightGBM
  - 特征组合：ALS User Factors + ALS Item Factors + User Doc2Vec + Item Doc2Vec + 基础特征
  - 使用 5-Fold OOF 验证
  - 尝试 32 维和 64 维 ALS 特征
  - 记录 OOF RMSE
  - 保存 OOF 和测试预测

  **Must NOT do**:
  - 禁止使用测试集数据训练
  - 禁止不使用 OOF 验证

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要组合多种特征并训练 LightGBM
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7, 9, 10)
  - **Blocks**: Task 13
  - **Blocked By**: Tasks 2, 3, 6

  **References**:

  **Pattern References**:
  - `code/models/ensemble_diverse.py` — LightGBM 训练模式，OOF 验证

  **API/Type References**:
  - `artifacts/features/als_user_factors.npy` — ALS 用户潜向量
  - `artifacts/features/als_item_factors.npy` — ALS 商品潜向量
  - `artifacts/features/user_doc2vec.npy` — 用户 Doc2Vec 向量
  - `artifacts/features/item_doc2vec.npy` — 商品 Doc2Vec 向量

  **WHY Each Reference Matters**:
  - ensemble_diverse.py: 展示 LightGBM 训练和 OOF 验证的标准流程

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: LightGBM + ALS + Doc2Vec 训练成功
    Tool: Bash (python)
    Preconditions: ALS 和 Doc2Vec 特征已生成
    Steps:
      1. 运行 LightGBM + ALS + Doc2Vec 训练脚本
      2. 检查 lgb_als_doc2vec_oof.npy 和 lgb_als_doc2vec_test.npy 是否生成
      3. 验证 OOF RMSE 是否记录
      4. 检查 OOF RMSE 是否 < 1.05
    Expected Result: 模型训练成功，OOF RMSE < 1.05
    Failure Indicators: 训练失败、OOF RMSE > 1.05
    Evidence: .sisyphus/evidence/task-8-lgb-als-doc2vec.txt

  Scenario: 特征重要性分析
    Tool: Bash (python)
    Preconditions: LightGBM 训练完成
    Steps:
      1. 提取 LightGBM 特征重要性
      2. 检查 ALS 和 Doc2Vec 特征的重要性排名
    Expected Result: ALS 和 Doc2Vec 特征在重要性排名中位于前列
    Failure Indicators: 特征重要性过低
    Evidence: .sisyphus/evidence/task-8-feature-importance.txt
  ```

  **Commit**: YES
  - Message: `feat(lgb-als): train LightGBM with ALS + Doc2Vec features`
  - Files: `code/models/train_lgb_als_doc2vec.py`
  - Pre-commit: `python code/models/train_lgb_als_doc2vec.py`

- [ ] 9. LightGBM + SVD 特征

  **What to do**:
  - 使用 TF-IDF SVD 降维特征训练 LightGBM
  - 特征组合：SVD 64 维 + SVD 128 维 + 基础特征
  - 使用 5-Fold OOF 验证
  - 记录 OOF RMSE
  - 保存 OOF 和测试预测

  **Must NOT do**:
  - 禁止使用测试集数据训练
  - 禁止不使用 OOF 验证

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要组合多种特征并训练 LightGBM
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7, 8, 10)
  - **Blocks**: Task 13
  - **Blocked By**: Tasks 4, 6

  **References**:

  **Pattern References**:
  - `code/models/ensemble_diverse.py` — LightGBM 训练模式

  **API/Type References**:
  - `artifacts/features/tfidf_svd_64.npy` — TF-IDF SVD 64 维特征
  - `artifacts/features/tfidf_svd_128.npy` — TF-IDF SVD 128 维特征

  **WHY Each Reference Matters**:
  - ensemble_diverse.py: 展示 LightGBM 训练流程

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: LightGBM + SVD 训练成功
    Tool: Bash (python)
    Preconditions: SVD 特征已生成
    Steps:
      1. 运行 LightGBM + SVD 训练脚本
      2. 检查 lgb_svd_oof.npy 和 lgb_svd_test.npy 是否生成
      3. 验证 OOF RMSE 是否记录
      4. 检查 OOF RMSE 是否 < 1.10
    Expected Result: 模型训练成功，OOF RMSE < 1.10
    Failure Indicators: 训练失败、OOF RMSE > 1.10
    Evidence: .sisyphus/evidence/task-9-lgb-svd.txt
  ```

  **Commit**: YES
  - Message: `feat(lgb-svd): train LightGBM with SVD features`
  - Files: `code/models/train_lgb_svd.py`
  - Pre-commit: `python code/models/train_lgb_svd.py`

- [ ] 10. 安全目标编码（平滑 + 噪声）

  **What to do**:
  - 实现安全的目标编码，防止数据泄漏
  - 使用平滑公式：μ̂ = (n·ȳ + m·μ_global) / (n + m)
  - n = 该用户/商品的样本数，ȳ = 均值，μ_global = 全局均值，m = 平滑权重
  - 尝试 m = 10, 20, 50 三种平滑权重
  - 添加随机噪声（高斯噪声，标准差为 0.01 * 全局标准差）
  - 使用 K-Fold 验证，每折的编码仅使用其他折的数据
  - 保存为 artifacts/features/target_encoded_user.npy 和 target_encoded_item.npy

  **Must NOT do**:
  - 禁止使用当前折的标签计算编码
  - 禁止不添加噪声
  - 禁止不使用平滑

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 目标编码需要仔细处理数据泄漏，实现较复杂
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7, 8, 9)
  - **Blocks**: Task 13
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `code/features/rating_deviation.py` — K-Fold 验证模式，防止数据泄漏
  - `code/features/assemble_kfold.py` — 特征组装模式

  **API/Type References**:
  - `data/train.csv` — 训练数据，包含 user_id, parent_prod_id, rating
  - `artifacts/features/y_train.npy` — 训练标签

  **External References**:
  - Target Encoding 最佳实践: https://medium.com/@pouryaayria/k-fold-target-encoding-for-high-cardinality-features

  **WHY Each Reference Matters**:
  - rating_deviation.py: 展示 K-Fold 验证模式，防止数据泄漏的关键参考

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 目标编码无泄漏验证
    Tool: Bash (python)
    Preconditions: 训练数据和标签已加载
    Steps:
      1. 运行目标编码脚本
      2. 检查 target_encoded_user.npy 和 target_encoded_item.npy 是否生成
      3. 验证维度：应为 (3007439,)
      4. 检查值范围：应在 [1, 5] 之间
      5. 验证 K-Fold 验证：每折的编码仅使用其他折的数据
    Expected Result: 编码文件生成成功，值范围正确，无泄漏
    Failure Indicators: 文件不存在、值超出范围、泄漏检测失败
    Evidence: .sisyphus/evidence/task-10-target-encoding.txt
  ```

  **Commit**: YES
  - Message: `feat(te): add safe target encoding with smoothing`
  - Files: `code/features/target_encoding_safe.py`
  - Pre-commit: `python code/features/target_encoding_safe.py`

### Wave 3: Cross-Attention 多模态融合

- [ ] 11. Feature Tokenizer + Cross-Attention 融合模型

  **What to do**:
  - 实现 Feature Tokenizer + Cross-Attention 多模态融合架构
  - 架构：
    ```
    [ 评论文本 (Tokens) ]  ───►  [ DeBERTa 编码器 ] ──┐
                                                     ├─► [ Cross-Attention 融合层 ] ─► 预测输出
    [ 用户/商品潜向量 (ALS) ] ──► [ Feature Tokenizer ] ─┘
    ```
  - 使用 LoRA 微调的 DeBERTa 作为文本编码器
  - 使用轻量级 2 层 Transformer 块实现 Cross-Attention
  - 让文本 Token 与用户/商品嵌入进行 Cross-Attention 交互
  - 使用 CORAL 损失函数
  - 5-Fold OOF 验证
  - 记录 OOF RMSE
  - 保存 OOF 和测试预测

  **Must NOT do**:
  - 禁止不使用 LoRA 微调的 DeBERTa
  - 禁止不使用 CORAL 损失

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Cross-Attention 融合涉及多模态架构设计和自定义 Transformer 块，需要深入的 NLP 和 PyTorch 知识
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 12)
  - **Blocks**: Task 13
  - **Blocked By**: Tasks 1, 2, 3, 4, 5

  **References**:

  **Pattern References**:
  - `code/models/transformer_finetune.py` — DeBERTa 微调流程
  - `code/models/mlp.py` — MLP 模型架构

  **External References**:
  - ACL 2024 TTT (Tabular-Text Transformer): 前沿多模态融合架构
  - FT-Transformer: 表格数据 Transformer 架构
  - PyTorch MultiheadAttention: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

  **WHY Each Reference Matters**:
  - transformer_finetune.py: 展示 DeBERTa 微调流程
  - ACL 2024 TTT: 前沿多模态融合架构参考

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Cross-Attention 融合模型训练成功
    Tool: Bash (python)
    Preconditions: LoRA DeBERTa 和 ALS 特征已生成
    Steps:
      1. 运行 Cross-Attention 融合模型训练脚本
      2. 检查 cross_attn_oof.npy 和 cross_attn_test.npy 是否生成
      3. 验证 OOF RMSE 是否 < 0.98
      4. 验证预测值范围是否在 [1, 5] 之间
    Expected Result: 模型训练成功，OOF RMSE < 0.98，预测值范围正确
    Failure Indicators: 训练失败、OOF RMSE > 0.98、预测值超出范围
    Evidence: .sisyphus/evidence/task-11-cross-attn.txt

  Scenario: Cross-Attention 权重分析
    Tool: Bash (python)
    Preconditions: Cross-Attention 模型训练完成
    Steps:
      1. 提取 Cross-Attention 权重
      2. 分析文本 Token 与用户/商品嵌入的交互模式
    Expected Result: Cross-Attention 权重显示有意义的交互
    Failure Indicators: 权重全为零或随机
    Evidence: .sisyphus/evidence/task-11-attn-weights.txt
  ```

  **Commit**: YES
  - Message: `feat(cross-attn): add Cross-Attention multimodal fusion`
  - Files: `code/models/cross_attn_model.py`
  - Pre-commit: `python code/models/cross_attn_model.py`

- [ ] 12. 5 分类期望值模型（LightGBM）

  **What to do**:
  - 使用 LightGBM 实现 5 分类任务
  - 特征组合：TF-IDF 5000 + ALS + Doc2Vec + SVD + 基础特征
  - 使用 multiclass objective
  - 最终预测：ŷ = Σ(i=1 to 5) i · P(y=i)
  - 使用 5-Fold OOF 验证
  - 记录 OOF RMSE
  - 保存 OOF 和测试预测

  **Must NOT do**:
  - 禁止不使用分层采样
  - 禁止不使用期望值转换

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要组合多种特征并训练多分类 LightGBM
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 11)
  - **Blocks**: Task 13
  - **Blocked By**: Task 6

  **References**:

  **Pattern References**:
  - `code/models/ensemble_diverse.py` — LightGBM 训练模式

  **API/Type References**:
  - `artifacts/features/chartfidf_train.npz` — TF-IDF 特征
  - `artifacts/features/als_user_factors.npy` — ALS 用户潜向量
  - `artifacts/features/als_item_factors.npy` — ALS 商品潜向量
  - `artifacts/features/user_doc2vec.npy` — 用户 Doc2Vec 向量
  - `artifacts/features/item_doc2vec.npy` — 商品 Doc2Vec 向量
  - `artifacts/features/tfidf_svd_64.npy` — SVD 特征

  **External References**:
  - LightGBM multiclass: https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective

  **WHY Each Reference Matters**:
  - ensemble_diverse.py: 展示 LightGBM 训练流程

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: LightGBM 多分类训练成功
    Tool: Bash (python)
    Preconditions: 所有特征已生成
    Steps:
      1. 运行 LightGBM 多分类训练脚本
      2. 检查 cls_lgb_oof.npy 和 cls_lgb_test.npy 是否生成
      3. 验证 OOF RMSE 是否记录
      4. 检查 OOF RMSE 是否 < 1.05
    Expected Result: 模型训练成功，OOF RMSE < 1.05
    Failure Indicators: 训练失败、OOF RMSE > 1.05
    Evidence: .sisyphus/evidence/task-12-cls-lgb.txt
  ```

  **Commit**: YES
  - Message: `feat(cls-lgb): add 5-class expected value model with LightGBM`
  - Files: `code/models/cls_lgb.py`
  - Pre-commit: `python code/models/cls_lgb.py`

### Wave 4: 集成优化

- [ ] 13. 新集成策略（加权平均 + Stacking）

  **What to do**:
  - 收集所有模型的 OOF 预测：
    - MLP (DeBERTa 768): 现有
    - LightGBM (TF-IDF 5000): 现有
    - XGBoost (TF-IDF 5000): 现有
    - LoRA DeBERTa: 新增
    - CORAL 序数回归: 新增
    - LightGBM + ALS + Doc2Vec: 新增
    - LightGBM + SVD: 新增
    - Cross-Attention 融合: 新增
    - 5 分类 LightGBM: 新增
  - 实现两种集成策略：
    1. 加权平均：使用 Optuna 优化权重
    2. Stacking：使用 Ridge 回归作为 meta-learner
  - 比较两种策略的 OOF RMSE
  - 选择最优策略
  - 保存集成 OOF 和测试预测

  **Must NOT do**:
  - 禁止不比较两种集成策略
  - 禁止不使用 OOF 验证

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要实现多种集成策略并比较效果
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Tasks 14, 15
  - **Blocked By**: Tasks 7, 8, 9, 10, 11, 12

  **References**:

  **Pattern References**:
  - `code/models/ensemble_diverse.py` — 集成模式，加权平均
  - `code/models/optuna_ensemble.py` — Optuna 权重优化

  **API/Type References**:
  - 所有模型的 OOF 预测文件

  **WHY Each Reference Matters**:
  - ensemble_diverse.py: 展示加权平均集成模式
  - optuna_ensemble.py: 展示 Optuna 权重优化方法

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 集成策略比较
    Tool: Bash (python)
    Preconditions: 所有模型 OOF 预测已生成
    Steps:
      1. 运行集成脚本
      2. 比较加权平均和 Stacking 的 OOF RMSE
      3. 选择最优策略
      4. 检查 ensemble_v4_oof.npy 和 ensemble_v4_test.npy 是否生成
      5. 验证 OOF RMSE 是否 < 0.95
    Expected Result: 集成成功，OOF RMSE < 0.95
    Failure Indicators: 集成失败、OOF RMSE > 0.95
    Evidence: .sisyphus/evidence/task-13-ensemble.txt

  Scenario: 集成权重分析
    Tool: Bash (python)
    Preconditions: 集成完成
    Steps:
      1. 提取集成权重（如果是加权平均）
      2. 分析各模型的贡献
    Expected Result: 权重分布合理，各模型有贡献
    Failure Indicators: 某个模型权重为 0 或 1
    Evidence: .sisyphus/evidence/task-13-weights.txt
  ```

  **Commit**: YES
  - Message: `feat(ensemble): add new ensemble strategy with CORAL/Cross-Attention`
  - Files: `code/models/ensemble_v4.py`
  - Pre-commit: `python code/models/ensemble_v4.py`

- [ ] 14. Optuna 权重优化

  **What to do**:
  - 使用 Optuna 优化集成权重
  - 搜索空间：每个模型的权重 ∈ [0, 1]
  - 目标：最小化 OOF RMSE
  - 尝试 1000 次试验
  - 记录最优权重和 OOF RMSE
  - 与网格搜索结果比较

  **Must NOT do**:
  - 禁止不记录最优权重
  - 禁止不比较 Optuna 和网格搜索

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要实现 Optuna 优化并比较不同方法
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 15
  - **Blocked By**: Task 13

  **References**:

  **Pattern References**:
  - `code/models/optuna_ensemble.py` — Optuna 集成优化

  **External References**:
  - Optuna 文档: https://optuna.org/

  **WHY Each Reference Matters**:
  - optuna_ensemble.py: 展示 Optuna 优化方法

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Optuna 优化成功
    Tool: Bash (python)
    Preconditions: 集成 OOF 预测已生成
    Steps:
      1. 运行 Optuna 优化脚本
      2. 检查最优权重和 OOF RMSE
      3. 与网格搜索结果比较
      4. 验证 Optuna 结果是否优于网格搜索
    Expected Result: Optuna 优化成功，结果优于或等于网格搜索
    Failure Indicators: 优化失败或结果差于网格搜索
    Evidence: .sisyphus/evidence/task-14-optuna.txt
  ```

  **Commit**: YES
  - Message: `feat(optuna): add Optuna weight optimization`
  - Files: `code/models/optuna_v4.py`
  - Pre-commit: `python code/models/optuna_v4.py`

- [ ] 15. 最终 Kaggle 提交

  **What to do**:
  - 使用最优集成策略生成测试集预测
  - Clip 到 [1, 5] 范围
  - 生成 submission-v4.csv
  - 提交到 Kaggle
  - 记录 Kaggle 分数

  **Must NOT do**:
  - 禁止不 Clip 到 [1, 5]
  - 禁止不记录 Kaggle 分数

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 生成提交文件是标准操作
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: F1-F4
  - **Blocked By**: Task 14

  **References**:

  **Pattern References**:
  - `code/models/final_submission.py` — 提交文件生成模式

  **API/Type References**:
  - `artifacts/models/ensemble_v4_test.npy` — 集成测试预测
  - `data/test.csv` — 测试数据 ID

  **WHY Each Reference Matters**:
  - final_submission.py: 展示提交文件生成流程

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 提交文件生成成功
    Tool: Bash (python)
    Preconditions: 集成测试预测已生成
    Steps:
      1. 运行提交文件生成脚本
      2. 检查 submission-v4.csv 是否生成
      3. 验证格式：列名应为 id, rating
      4. 验证行数：应为 10001（含表头）
      5. 验证 rating 范围：应在 [1, 5] 之间
    Expected Result: 提交文件生成成功，格式正确
    Failure Indicators: 文件不存在、格式错误、rating 超出范围
    Evidence: .sisyphus/evidence/task-15-submission.txt

  Scenario: Kaggle 分数记录
    Tool: Bash (python)
    Preconditions: 提交文件已生成
    Steps:
      1. 提交到 Kaggle（如果可能）
      2. 记录 Kaggle 分数
      3. 与目标分数（0.52）比较
    Expected Result: Kaggle 分数记录，用于评估优化效果
    Failure Indicators: 无法提交或分数未记录
    Evidence: .sisyphus/evidence/task-15-kaggle-score.txt
  ```

  **Commit**: YES
  - Message: `feat(submission): generate final Kaggle submission v4`
  - Files: `code/models/submission_v4.py`
  - Pre-commit: `python code/models/submission_v4.py`

---

## Final Verification Wave

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists. For each "Must NOT Have": search codebase for forbidden patterns. Check evidence files exist. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run linter + tests. Review all changed files for: `as any`/`@ts-ignore`, empty catches, console.log in prod, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names.
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Execute EVERY QA scenario from EVERY task. Test cross-task integration. Test edge cases: empty state, invalid input, rapid actions. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. Verify 1:1 — everything in spec was built, nothing beyond spec was built. Check "Must NOT do" compliance. Detect cross-task contamination. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Wave 1**: `feat(lora): add LoRA + BF16 + SDPA DeBERTa fine-tuning` — deberta_lora.py
- **Wave 1**: `feat(als): add PySpark ALS user-item latent factors` — als_features.py
- **Wave 1**: `feat(doc2vec): add PySpark Doc2Vec user language style` — doc2vec_features.py
- **Wave 1**: `feat(svd): add TF-IDF SVD dimensionality reduction` — svd_features.py
- **Wave 1**: `feat(pca): add DeBERTa PCA dimensionality reduction` — pca_features.py
- **Wave 2**: `feat(coral): add CORAL ordinal regression model` — coral_model.py
- **Wave 2**: `feat(lgb-als): train LightGBM with ALS + Doc2Vec features` — train_lgb_als.py
- **Wave 2**: `feat(lgb-svd): train LightGBM with SVD features` — train_lgb_svd.py
- **Wave 2**: `feat(te): add safe target encoding with smoothing` — target_encoding.py
- **Wave 3**: `feat(cross-attn): add Cross-Attention multimodal fusion` — cross_attn_model.py
- **Wave 3**: `feat(cls-lgb): add 5-class expected value model with LightGBM` — cls_lgb.py
- **Wave 4**: `feat(ensemble): add new ensemble strategy` — ensemble_v4.py
- **Wave 4**: `feat(optuna): add Optuna weight optimization` — optuna_v4.py
- **Wave 4**: `feat(submission): generate final Kaggle submission v4` — submission_v4.py

---

## Success Criteria

### Verification Commands
```bash
# ALS 特征验证
python -c "import numpy as np; u=np.load('artifacts/features/als_user_factors.npy'); print(f'User factors: {u.shape}, range=[{u.min():.3f}, {u.max():.3f}]')"
# Expected: User factors: (N, 32/64), range=[-1, 1]

# SVD 特征验证
python -c "import numpy as np; s=np.load('artifacts/features/tfidf_svd.npy'); print(f'TF-IDF SVD: {s.shape}, range=[{s.min():.3f}, {s.max():.3f}]')"
# Expected: TF-IDF SVD: (N, 64/128), range=[-1, 1]

# OOF RMSE 验证
python -c "import numpy as np; oof=np.load('artifacts/models/ensemble_v4_oof.npy'); y=np.load('artifacts/features/y_train.npy'); rmse=np.sqrt(np.mean((oof-y)**2)); print(f'Ensemble v4 OOF RMSE: {rmse:.4f}')"
# Expected: RMSE < 1.05

# Kaggle 提交验证
python -c "import pandas as pd; df=pd.read_csv('output/submission-v4.csv'); print(f'Shape: {df.shape}, columns: {list(df.columns)}'); print(f'Rating range: [{df.rating.min():.2f}, {df.rating.max():.2f}]')"
# Expected: Shape: (10001, 2), columns: ['id', 'rating'], Rating range: [1.00, 5.00]
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] Kaggle score < 0.52
- [ ] PySpark 耗时对比记录
- [ ] 课程报告所需数据准备完毕

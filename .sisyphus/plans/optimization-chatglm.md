# Kaggle竞赛优化计划 — optimization-chatglm

## TL;DR

> **Quick Summary**: 并行双线并进策略，同时推进DeBERTa-v3-base端到端微调（5分类+软标签，支持QLoRA）和树模型特征增强（TF-IDF 100K + SVD + 平滑Target Encoding + DeBERTa嵌入），最后使用Stacking集成，目标突破Kaggle 0.60。
> 
> **Deliverables**:
> - DeBERTa-v3-base 微调脚本（5分类+软标签，支持全参数和QLoRA）
> - 增强TF-IDF特征（50K词级 + 50K字符级 = 100K，SVD降维200维）
> - 平滑Target Encoding特征（GroupKFold + 平滑公式）
> - 计数/偏差特征（user_review_count, prod_review_count, user_rating_dev, prod_rating_dev）
> - DeBERTa嵌入特征（[CLS]向量768维，用于树模型）
> - 增强树模型（LightGBM, XGBoost, CatBoost）
> - Stacking集成（Ridge元学习器）
> - 最终提交文件
> 
> **Estimated Effort**: 6-10小时
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Task 1 (DeBERTa) → Task 5 (Stacking) → Task 6 (Submission)

---

## Context

### Original Request

用户提出了Kaggle竞赛优化方案，包括：
1. DeBERTa端到端微调（5分类+软标签）
2. 安全地重新引入用户/商品特征（计数/偏差特征，避免泄漏）
3. 增强树模型的文本表示（TF-IDF扩维 + SVD降维）
4. 集成策略升级（Stacking）
5. 其他技巧（后处理优化、伪标签）

### Interview Summary

**Key Discussions**:
- DeBERTa微调策略：使用`deberta-v3-base`（86M参数），支持全参数和QLoRA两种模式
- 用户/商品特征：使用平滑Target Encoding + GroupKFold + 计数/偏差特征
- 资源约束：3080ti 12GB显存
- 目标分数：尽可能低（激进目标，Kaggle < 0.60）
- 技术栈：HuggingFace Trainer + PEFT，放弃PySpark训练深度模型

**Research Findings**:
- Metis可行性分析：所有方案高度可行
- 当前MLP使用冻结DeBERTa嵌入，OOF RMSE 1.131
- 树模型只使用TF-IDF 5K特征，信息量不足
- 已有K-Fold版本的用户/商品统计特征，但因泄漏问题被排除
- DeBERTa-v3-base（86M参数）在12G显存上全参数微调完全可行

### Metis Review

**Identified Gaps** (addressed):
- DeBERTa微调应使用5分类+软标签，而非MSE回归
- 需要实现GroupKFold，而不是普通KFold
- 需要增加TF-IDF维度（50K词级 + 50K字符级）并使用SVD降维
- 需要实现平滑Target Encoding（使用GroupKFold + 平滑公式）
- 需要把DeBERTa嵌入（[CLS]向量768维）也喂给树模型
- 需要使用HuggingFace Trainer + PEFT，而不是手写训练循环

---

## Work Objectives

### Core Objective

通过并行双线并进策略，同时优化DeBERTa-v3-base端到端微调（5分类+软标签）和树模型特征增强（TF-IDF 100K + SVD + 平滑Target Encoding + DeBERTa嵌入），使用Stacking集成，突破Kaggle 0.60分数。

### Concrete Deliverables

1. `code/models/transformer_finetune_v2.py` - DeBERTa-v3-base 5分类+软标签微调（支持QLoRA）
2. `code/features/enhanced_tfidf.py` - TF-IDF 100K（50K词级 + 50K字符级）+ SVD 200维
3. `code/features/safe_target_encoding.py` - 平滑Target Encoding（GroupKFold + 平滑公式）
4. `code/features/count_deviation.py` - 计数/偏差特征
5. `code/features/deberta_embeddings.py` - DeBERTa嵌入特征（[CLS]向量768维）
6. `code/models/train_enhanced_trees.py` - 增强树模型训练
7. `code/models/stacking_v2.py` - Stacking集成
8. `code/models/final_submission_v2.py` - 最终提交生成

### Definition of Done

- [ ] DeBERTa-v3-base OOF RMSE < 1.0
- [ ] 增强树模型OOF RMSE < 1.05
- [ ] 集成OOF RMSE < 1.0
- [ ] Kaggle分数 < 0.60
- [ ] 所有代码有适当注释和文档
- [ ] 最终提交文件格式正确

### Must Have

- DeBERTa-v3-base使用5分类+软标签策略（86M参数）
- 支持全参数微调和QLoRA两种模式
- TF-IDF扩维到100K（50K词级 + 50K字符级）+ SVD降维到200维
- 平滑Target Encoding（GroupKFold + 平滑公式，C=10~50）
- 计数/偏差特征（user_review_count, prod_review_count, user_rating_dev, prod_rating_dev）
- DeBERTa嵌入特征（[CLS]向量768维，用于树模型）
- Stacking集成（Ridge元学习器）
- 后处理优化（clip到[1,5]）
- 使用HuggingFace Trainer + PEFT，而不是手写训练循环

### Must NOT Have (Guardrails)

- 不要使用deberta-v3-small（使用deberta-v3-base，86M参数）
- 不要修改原有`transformer_finetune.py`（保留作为对比）
- 不要使用LightGCN（已确认失败，嵌入近零）
- 不要使用伪标签（除非Wave 2效果好，Kaggle < 0.60）
- 不要使用PySpark训练深度模型（使用HuggingFace生态）
- 不要使用普通截断（使用双段截断法或max_length=256）

---

## Verification Strategy

### Test Decision

- **Infrastructure exists**: NO
- **Automated tests**: NO
- **Framework**: none
- **Rationale**: 这是Kaggle竞赛项目，没有测试基础设施，验证通过OOF RMSE和Kaggle分数

### QA Policy

每个任务完成后，执行以下验证：
1. 检查输出文件是否存在
2. 检查输出文件维度是否正确
3. 检查OOF RMSE是否达标
4. 如果是最终提交，提交Kaggle验证

---

## Execution Strategy

### Estimated Time

| 方案 | 时间 | 说明 |
|------|------|------|
| **最小（不含Wave 3）** | **3-4小时** | Wave 1 + Wave 2 |
| **完整（含Wave 3）** | **5-7小时** | Wave 1 + Wave 2 + Wave 3 |

#### Wave 1 详细时间（并行执行，2-3小时）

| 任务 | 资源 | 时间估算 | 备注 |
|------|------|----------|------|
| Task 1: DeBERTa-v3-base 微调 | 3080ti 12GB | 2-3小时 | 86M参数, 3M数据, 3 epochs, batch=16 |
| Task 2: 增强TF-IDF特征 | CPU/内存 | 30-45分钟 | 100K特征 + SVD 200维 |
| Task 3: 平滑TE + 计数/偏差 | CPU/内存 | 20-30分钟 | 3M数据, GroupKFold |
| Task 4: DeBERTa嵌入提取 | 3080ti 12GB | 1-1.5小时 | 3M数据, 768维 |
| Task 5: 增强树模型训练 | CPU/内存 | 1.5-2小时 | 3个模型, 5-fold CV |

**注意**: Task 1和Task 4都需要GPU，不能完全并行。建议先跑Task 1，再跑Task 4。

#### Wave 2 详细时间（串行，15-30分钟）

| 任务 | 资源 | 时间估算 |
|------|------|----------|
| Task 6: Stacking集成 | CPU | 5-10分钟 |
| Task 7: 最终提交生成 | CPU | 5-10分钟 |

#### Wave 3 详细时间（可选，2-3小时）

| 任务 | 资源 | 时间估算 |
|------|------|----------|
| Task 8: QLoRA长序列微调 | 3080ti 12GB | 2-3小时 |

### Parallel Execution Waves

```
Wave 1 (并行，2-3天):
├── Task 1: DeBERTa-v3-base 微调 [3080ti, 12GB显存]
├── Task 2: 增强TF-IDF特征 [CPU/内存]
├── Task 3: 平滑Target Encoding + 计数/偏差特征 [CPU/内存]
├── Task 4: DeBERTa嵌入特征 [3080ti, 12GB显存]
└── Task 5: 增强树模型训练 [CPU/内存]

Wave 2 (串行，0.5天):
├── Task 6: Stacking集成 [CPU]
└── Task 7: 最终提交生成 [CPU]

Wave 3 (可选，1-2天):
└── Task 8: QLoRA长序列微调（如果Wave 2效果好） [3080ti]
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|------------|--------|
| 1: DeBERTa-v3-base | None | 6 |
| 2: 增强TF-IDF | None | 5 |
| 3: 平滑TE + 计数/偏差 | None | 5 |
| 4: DeBERTa嵌入 | None | 5 |
| 5: 增强树模型 | 2, 3, 4 | 6 |
| 6: Stacking | 1, 5 | 7 |
| 7: 最终提交 | 6 | None |
| 8: QLoRA长序列 | 7 | None |

### Agent Dispatch Summary

- **Wave 1**: 5 tasks - T1, T4 → `unspecified-high`, T2, T3, T5 → `quick`
- **Wave 2**: 2 tasks - T6-T7 → `quick`
- **Wave 3**: 1 task - T8 → `unspecified-high` (optional)

---

## TODOs

- [ ] 1. DeBERTa-v3-base 端到端微调

  **What to do**:
  - 创建`code/models/transformer_finetune_v2.py`
  - 使用`microsoft/deberta-v3-base`（86M参数，而非deberta-v3-small的44M）
  - 实现5分类+软标签策略（num_labels=5, problem_type="single_label_classification"）
  - 使用HuggingFace Trainer + PEFT（支持全参数和QLoRA两种模式）
  - 训练配置（12G显存）：
    ```python
    training_args = TrainingArguments(
        output_dir="./deberta-v3-5cls",
        per_device_train_batch_size=16,      # 先试16，不行再降
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,       # 等效全局batch=32
        learning_rate=2e-5,
        num_train_epochs=3,
        max_length=256,                      # 电商评论256 token足够
        fp16=True,                           # 混合精度省显存
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
    )
    ```
  - 推理时使用软标签求期望：
    ```python
    probs = torch.softmax(logits, dim=-1)        # [batch, 5]
    star = torch.arange(1, 6, dtype=torch.float, device=probs.device)
    pred = (probs * star).sum(dim=-1, keepdim=True)  # 期望评分
    ```
  - 5-fold CV生成OOF预测和测试预测
  - 如果OOM：优先降per_device_train_batch_size到8/4，再配合更大的gradient_accumulation_steps

  **Must NOT do**:
  - 不要使用deberta-v3-small（使用deberta-v3-base，86M参数）
  - 不要修改原有`transformer_finetune.py`
  - 不要使用MSE回归（使用5分类+软标签）
  - 不要使用PySpark训练（使用HuggingFace Trainer）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 涉及深度学习模型微调，需要理解Transformer架构和HuggingFace生态
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4)
  - **Blocks**: Task 6 (Stacking)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `code/models/transformer_finetune.py` - 原有DeBERTa微调框架，参考其结构
  - `code/models/mlp.py` - MLP模型架构参考

  **API/Type References**:
  - `artifacts/etl/train.parquet` - 训练数据（包含title, comment, rating列）
  - `artifacts/etl/test.parquet` - 测试数据

  **External References**:
  - HuggingFace Transformers文档：https://huggingface.co/docs/transformers/
  - DeBERTa-v3-base模型：https://huggingface.co/microsoft/deberta-v3-base
  - PEFT文档：https://huggingface.co/docs/peft/
  - QLoRA论文：https://arxiv.org/abs/2305.14314

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: DeBERTa-v3-base 训练成功
    Tool: Bash
    Preconditions: 3080ti 12GB显存可用
    Steps:
      1. 运行 python code/models/transformer_finetune_v2.py
      2. 检查训练日志，确认5分类+软标签策略生效
      3. 检查OOF RMSE是否 < 1.0
    Expected Result: 训练完成，OOF RMSE < 1.0
    Failure Indicators: 训练崩溃、OOF RMSE > 1.1
    Evidence: .sisyphus/evidence/task-1-deberta-v2-training.log

  Scenario: 输出文件正确
    Tool: Bash
    Preconditions: 训练完成
    Steps:
      1. 检查 artifacts/models/transformer_v2_oof.npy 是否存在
      2. 检查 artifacts/models/transformer_v2_test.npy 是否存在
      3. 检查维度：OOF (3007439,), TEST (10000,)
    Expected Result: 文件存在，维度正确
    Failure Indicators: 文件不存在或维度错误
    Evidence: .sisyphus/evidence/task-1-output-files.txt

  Scenario: OOM处理
    Tool: Bash
    Preconditions: 训练过程中OOM
    Steps:
      1. 降低per_device_train_batch_size到8或4
      2. 增加gradient_accumulation_steps保持等效全局batch大小
      3. 如果仍然OOM，降低max_length到192或128
    Expected Result: 训练能够正常进行，不OOM
    Failure Indicators: 持续OOM无法解决
    Evidence: .sisyphus/evidence/task-1-oom-handling.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add DeBERTa-v3-base fine-tuning with 5-class + soft labels`
  - Files: `code/models/transformer_finetune_v2.py`
  - Pre-commit: None

---

- [ ] 2. 增强TF-IDF特征

  **What to do**:
  - 创建`code/features/enhanced_tfidf.py`
  - 实现TF-IDF 100K（50K词级 + 50K字符级）：
    ```python
    tfidf_word = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        analyzer="word",
        sublinear_tf=True,
    )
    tfidf_char = TfidfVectorizer(
        max_features=50000,
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
    )
    X_word = tfidf_word.fit_transform(train_texts)
    X_char = tfidf_char.fit_transform(train_texts)
    X_tfidf = hstack([X_word, X_char])  # [n_samples, 100k]
    ```
  - 实现SVD降维到200维：
    ```python
    svd = TruncatedSVD(n_components=200, random_state=42)
    X_svd = svd.fit_transform(X_tfidf)  # [n_samples, 200]
    ```
  - 保存向量化器和SVD模型（用于测试集转换）
  - 生成训练集和测试集特征

  **Must NOT do**:
  - 不要只使用词级TF-IDF（同时使用词级和字符级）
  - 不要超过200维（内存限制）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 简单的特征工程，使用sklearn标准API
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4)
  - **Blocks**: Task 5 (增强树模型)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `code/features/text_chartfidf.py` - 原有字符级TF-IDF实现，参考其结构

  **API/Type References**:
  - `artifacts/etl/train.parquet` - 训练数据（包含title和comment列）
  - `artifacts/etl/test.parquet` - 测试数据

  **External References**:
  - sklearn TfidfVectorizer文档：https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
  - sklearn TruncatedSVD文档：https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: TF-IDF + SVD 特征生成成功
    Tool: Bash
    Preconditions: 训练数据和测试数据存在
    Steps:
      1. 运行 python code/features/enhanced_tfidf.py
      2. 检查 artifacts/features/enhanced_tfidf_train.npz 是否存在
      3. 检查 artifacts/features/enhanced_tfidf_test.npz 是否存在
      4. 检查维度：TRAIN (3007439, 200), TEST (10000, 200)
    Expected Result: 文件存在，维度正确
    Failure Indicators: 文件不存在或维度错误
    Evidence: .sisyphus/evidence/task-2-tfidf-features.txt
  ```

  **Commit**: YES
  - Message: `feat(features): add enhanced TF-IDF with 100K (word+char) + SVD 200-dim`
  - Files: `code/features/enhanced_tfidf.py`
  - Pre-commit: None

---

- [ ] 3. 平滑Target Encoding + 计数/偏差特征

  **What to do**:
  - 创建`code/features/safe_target_encoding.py`
  - 实现平滑Target Encoding（GroupKFold + 平滑公式）：
    ```python
    # 平滑Target Encoding公式
    # smooth_mean = (n * mean + C * global_mean) / (n + C)
    # C取10~50，防止只有1条评论的user/prod严重泄漏
    
    def smooth_target_encoding(group_col, target_col, C=20):
        global_mean = target_col.mean()
        stats = train.groupby(group_col).agg(
            n=('rating', 'count'),
            mean=('rating', 'mean')
        )
        stats['smooth_mean'] = (stats['n'] * stats['mean'] + C * global_mean) / (stats['n'] + C)
        return stats['smooth_mean']
    ```
  - 使用GroupKFold（按user_id/product_id分组），保证同组样本不同时出现在训练和验证集
  - 实现用户平滑Target Encoding（user_smooth_te）
  - 实现商品平滑Target Encoding（prod_smooth_te）
  - 保存为parquet格式

  - 创建`code/features/count_deviation.py`
  - 实现用户评论数量（user_review_count）
  - 实现商品评论数量（prod_review_count）
  - 实现用户打分偏差（user_rating_dev = user_mean - global_mean）
  - 实现商品评分偏差（prod_rating_dev = prod_mean - global_mean）
  - 保存为parquet格式

  **Must NOT do**:
  - 不要使用普通Target Encoding（必须使用平滑版本）
  - 不要使用普通KFold（必须使用GroupKFold）
  - 不要跳过平滑处理（C=10~50）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 简单的特征工程，使用pandas标准API
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4)
  - **Blocks**: Task 5 (增强树模型)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `code/features/rating_deviation.py` - 原有偏差特征实现，参考其结构
  - `code/features/assemble_kfold.py` - 原有K-Fold特征组装，参考其结构

  **API/Type References**:
  - `artifacts/etl/train.parquet` - 训练数据（包含user_id, parent_prod_id, rating列）
  - `artifacts/etl/test.parquet` - 测试数据

  **External References**:
  - sklearn GroupKFold文档：https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
  - Target Encoding最佳实践：https://contrib.scikit-learn.org/category_encoders/targetencoder.html

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 平滑Target Encoding生成成功
    Tool: Bash
    Preconditions: 训练数据和测试数据存在
    Steps:
      1. 运行 python code/features/safe_target_encoding.py
      2. 检查 artifacts/features/safe_target_encoding.parquet 是否存在
      3. 检查列名：user_smooth_te, prod_smooth_te
      4. 检查无泄漏：使用GroupKFold + 平滑公式
    Expected Result: 文件存在，列名正确，无泄漏
    Failure Indicators: 文件不存在、列名错误、使用了普通Target Encoding
    Evidence: .sisyphus/evidence/task-3-safe-te.txt

  Scenario: 计数/偏差特征生成成功
    Tool: Bash
    Preconditions: 训练数据和测试数据存在
    Steps:
      1. 运行 python code/features/count_deviation.py
      2. 检查 artifacts/features/count_deviation.parquet 是否存在
      3. 检查列名：user_review_count, prod_review_count, user_rating_dev, prod_rating_dev
      4. 检查无泄漏：偏差特征使用全局均值计算
    Expected Result: 文件存在，列名正确，无泄漏
    Failure Indicators: 文件不存在、列名错误
    Evidence: .sisyphus/evidence/task-3-count-deviation.txt
  ```

  **Commit**: YES
  - Message: `feat(features): add smooth Target Encoding + count/deviation features`
  - Files: `code/features/safe_target_encoding.py`, `code/features/count_deviation.py`
  - Pre-commit: None

---

- [ ] 4. DeBERTa嵌入特征

  **What to do**:
  - 创建`code/features/deberta_embeddings.py`
  - 使用训练好的DeBERTa-v3-base模型提取[CLS]向量（768维）
  - 对训练集和测试集分别提取嵌入
  - 保存为numpy格式（用于树模型训练）

  **Must NOT do**:
  - 不要使用原有DeBERTa嵌入（使用新微调的模型）
  - 不要跳过测试集嵌入提取

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 需要加载和使用Transformer模型提取嵌入
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3)
  - **Blocks**: Task 5 (增强树模型)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `code/models/transformer_finetune_v2.py` - DeBERTa v2微调脚本，参考其模型加载

  **API/Type References**:
  - `artifacts/etl/train.parquet` - 训练数据（包含title和comment列）
  - `artifacts/etl/test.parquet` - 测试数据

  **External References**:
  - HuggingFace Transformers文档：https://huggingface.co/docs/transformers/

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: DeBERTa嵌入提取成功
    Tool: Bash
    Preconditions: DeBERTa v2模型训练完成
    Steps:
      1. 运行 python code/features/deberta_embeddings.py
      2. 检查 artifacts/features/deberta_train_emb.npy 是否存在
      3. 检查 artifacts/features/deberta_test_emb.npy 是否存在
      4. 检查维度：TRAIN (3007439, 768), TEST (10000, 768)
    Expected Result: 文件存在，维度正确
    Failure Indicators: 文件不存在或维度错误
    Evidence: .sisyphus/evidence/task-4-deberta-embeddings.txt
  ```

  **Commit**: YES
  - Message: `feat(features): add DeBERTa embeddings for tree models`
  - Files: `code/features/deberta_embeddings.py`
  - Pre-commit: None

---

- [ ] 5. 增强树模型训练

  **What to do**:
  - 创建`code/models/train_enhanced_trees.py`
  - 加载增强特征：
    - TF-IDF SVD 200维
    - 平滑Target Encoding（user_smooth_te, prod_smooth_te）
    - 计数/偏差特征（user_review_count, prod_review_count, user_rating_dev, prod_rating_dev）
    - DeBERTa嵌入（[CLS]向量768维）
  - 训练LightGBM、XGBoost、CatBoost
  - 5-fold CV生成OOF预测和测试预测
  - 保存OOF预测和测试预测

  **Must NOT do**:
  - 不要使用原有TF-IDF 5K特征（使用增强版）
  - 不要跳过5-fold CV（需要OOF预测用于Stacking）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 使用现有模型训练框架，只需修改特征加载
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (after Tasks 2, 3, 4)
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 4)
  - **Blocks**: Task 6 (Stacking)
  - **Blocked By**: Tasks 2, 3, 4

  **References**:

  **Pattern References**:
  - `code/models/ensemble_diverse.py` - 原有多样化集成实现，参考其训练流程
  - `code/models/train_safe_features.py` - 原有安全特征训练，参考其结构

  **API/Type References**:
  - `artifacts/features/enhanced_tfidf_train.npz` - 增强TF-IDF训练集
  - `artifacts/features/enhanced_tfidf_test.npz` - 增强TF-IDF测试集
  - `artifacts/features/safe_target_encoding.parquet` - 平滑Target Encoding
  - `artifacts/features/count_deviation.parquet` - 计数/偏差特征
  - `artifacts/features/deberta_train_emb.npy` - DeBERTa嵌入训练集
  - `artifacts/features/deberta_test_emb.npy` - DeBERTa嵌入测试集
  - `artifacts/features/y_train.npy` - 训练标签

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 增强树模型训练成功
    Tool: Bash
    Preconditions: 增强特征存在
    Steps:
      1. 运行 python code/models/train_enhanced_trees.py
      2. 检查 artifacts/models/lgb_enhanced_oof.npy 是否存在
      3. 检查 artifacts/models/xgb_enhanced_oof.npy 是否存在
      4. 检查 artifacts/models/catboost_enhanced_oof.npy 是否存在
      5. 检查OOF RMSE是否 < 1.05
    Expected Result: 训练完成，OOF RMSE < 1.05
    Failure Indicators: 训练崩溃、OOF RMSE > 1.10
    Evidence: .sisyphus/evidence/task-5-tree-models.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add enhanced tree models with TF-IDF SVD + smooth TE + DeBERTa emb`
  - Files: `code/models/train_enhanced_trees.py`
  - Pre-commit: None

---

- [ ] 6. Stacking集成

  **What to do**:
  - 创建`code/models/stacking_v2.py`
  - 收集所有OOF预测（DeBERTa v2, LGB, XGB, CatBoost）
  - 训练Ridge元学习器（alpha=1.0）
  - 5-fold CV生成Stacking OOF预测
  - 保存Stacking OOF预测和测试预测

  **Must NOT do**:
  - 不要使用加权平均（使用Stacking）
  - 不要跳过5-fold CV（需要OOF预测用于评估）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 使用sklearn标准API，只需修改输入
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (after Tasks 1, 5)
  - **Parallel Group**: Wave 2 (sequential)
  - **Blocks**: Task 7 (最终提交)
  - **Blocked By**: Tasks 1, 5

  **References**:

  **Pattern References**:
  - `code/models/stacking.py` - 原有Stacking实现，参考其结构

  **API/Type References**:
  - `artifacts/models/transformer_v2_oof.npy` - DeBERTa v2 OOF
  - `artifacts/models/lgb_enhanced_oof.npy` - LightGBM OOF
  - `artifacts/models/xgb_enhanced_oof.npy` - XGBoost OOF
  - `artifacts/models/catboost_enhanced_oof.npy` - CatBoost OOF
  - `artifacts/features/y_train.npy` - 训练标签

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Stacking集成成功
    Tool: Bash
    Preconditions: 所有OOF预测存在
    Steps:
      1. 运行 python code/models/stacking_v2.py
      2. 检查 artifacts/models/stacking_v2_oof.npy 是否存在
      3. 检查 artifacts/models/stacking_v2_test.npy 是否存在
      4. 检查OOF RMSE是否 < 1.0
    Expected Result: 集成完成，OOF RMSE < 1.0
    Failure Indicators: 集成崩溃、OOF RMSE > 1.05
    Evidence: .sisyphus/evidence/task-6-stacking.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add Stacking v2 with Ridge meta-learner`
  - Files: `code/models/stacking_v2.py`
  - Pre-commit: None

---

- [ ] 7. 最终提交生成

  **What to do**:
  - 创建`code/models/final_submission_v2.py`
  - 加载Stacking测试预测
  - 后处理：clip到[1,5]
  - 生成最终提交文件
  - 提交Kaggle验证

  **Must NOT do**:
  - 不要使用round（只使用clip）
  - 不要跳过Kaggle验证

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 简单的后处理和文件生成
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (after Task 6)
  - **Parallel Group**: Wave 2 (sequential)
  - **Blocks**: Task 8 (QLoRA长序列，可选)
  - **Blocked By**: Task 6

  **References**:

  **Pattern References**:
  - `code/models/final_submission.py` - 原有最终提交实现，参考其结构

  **API/Type References**:
  - `artifacts/models/stacking_v2_test.npy` - Stacking测试预测
  - `data/test.csv` - 测试数据（包含id列）

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 最终提交生成成功
    Tool: Bash
    Preconditions: Stacking测试预测存在
    Steps:
      1. 运行 python code/models/final_submission_v2.py
      2. 检查 output/submission-stacking-v2.csv 是否存在
      3. 检查格式：id,rating（10001行，包含header）
      4. 检查值范围：所有rating在[1,5]之间
    Expected Result: 文件存在，格式正确，值范围正确
    Failure Indicators: 文件不存在、格式错误、值超出范围
    Evidence: .sisyphus/evidence/task-7-submission.txt

  Scenario: Kaggle分数验证
    Tool: Bash
    Preconditions: 提交文件生成
    Steps:
      1. 提交到Kaggle
      2. 检查分数是否 < 0.60
    Expected Result: Kaggle分数 < 0.60
    Failure Indicators: Kaggle分数 > 0.65
    Evidence: .sisyphus/evidence/task-7-kaggle-score.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add final submission v2 with Stacking`
  - Files: `code/models/final_submission_v2.py`
  - Pre-commit: None

---

- [ ] 8. QLoRA长序列微调（可选）

  **What to do**:
  - 创建`code/models/transformer_finetune_v3.py`
  - 使用QLoRA微调DeBERTa-v3-base（max_length=512或1024）
  - 配置示例：
    ```python
    from transformers import BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # 或float16
    )
    
    lora_config = LoraConfig(
        r=16,                # 或32，看显存
        lora_alpha=32,       # 常见设成2*r
        target_modules=["query", "value"],  # DeBERTa的Q/V矩阵
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    
    # 加载4-bit模型
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/deberta-v3-base",
        num_labels=5,
        quantization_config=bnb_config,
        device_map={"": 0},   # 单卡
    )
    
    # 包一层LoRA
    model = get_peft_model(model, lora_config)
    ```
  - 12G卡上建议配置：
    - per_device_train_batch_size=4
    - gradient_accumulation_steps=4
    - gradient_checkpointing=True
  - 5-fold CV生成OOF预测和测试预测
  - 与全参数微调版本对比效果

  **Must NOT do**:
  - 不要跳过gradient_checkpointing（省显存）
  - 不要使用太大的batch_size（保持在4-8）

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 涉及QLoRA微调和PEFT库使用
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (after Task 7)
  - **Parallel Group**: Wave 3 (optional)
  - **Blocks**: None
  - **Blocked By**: Task 7

  **References**:

  **Pattern References**:
  - `code/models/transformer_finetune_v2.py` - DeBERTa v2全参数微调，参考其结构

  **API/Type References**:
  - `artifacts/etl/train.parquet` - 训练数据
  - `artifacts/etl/test.parquet` - 测试数据

  **External References**:
  - QLoRA论文：https://arxiv.org/abs/2305.14314
  - PEFT文档：https://huggingface.co/docs/peft/
  - BitsAndBytes文档：https://huggingface.co/docs/bitsandbytes/

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: QLoRA微调成功
    Tool: Bash
    Preconditions: 3080ti 12GB显存可用
    Steps:
      1. 运行 python code/models/transformer_finetune_v3.py
      2. 检查训练日志，确认QLoRA配置生效
      3. 检查OOF RMSE是否 < 0.95
    Expected Result: 训练完成，OOF RMSE < 0.95
    Failure Indicators: 训练崩溃、OOF RMSE > 1.0
    Evidence: .sisyphus/evidence/task-8-qlora-training.log

  Scenario: 长序列处理
    Tool: Bash
    Preconditions: QLoRA微调成功
    Steps:
      1. 检查max_length是否为512或1024
      2. 检查是否有OOM问题
      3. 如果OOM，降低max_length或batch_size
    Expected Result: 长序列处理正常，无OOM
    Failure Indicators: 持续OOM无法解决
    Evidence: .sisyphus/evidence/task-8-long-sequence.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add QLoRA fine-tuning with long sequence support`
  - Files: `code/models/transformer_finetune_v3.py`
  - Pre-commit: None

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit**
  检查所有"Must Have"是否实现，所有"Must NOT Have"是否避免。验证所有输出文件存在，维度正确。

- [ ] F2. **Kaggle Score Verification**
  提交最终文件到Kaggle，验证分数 < 0.65。如果分数不达标，分析原因并调整。

- [ ] F3. **Code Quality Review**
  检查代码注释、文档、错误处理。确保代码遵循项目现有风格。

---

## Commit Strategy

- **Task 1**: `feat(models): add DeBERTa-v3-base fine-tuning with 5-class + soft labels`
- **Task 2**: `feat(features): add enhanced TF-IDF with 100K (word+char) + SVD 200-dim`
- **Task 3**: `feat(features): add smooth Target Encoding + count/deviation features`
- **Task 4**: `feat(features): add DeBERTa embeddings for tree models`
- **Task 5**: `feat(models): add enhanced tree models with TF-IDF SVD + smooth TE + DeBERTa emb`
- **Task 6**: `feat(models): add Stacking v2 with Ridge meta-learner`
- **Task 7**: `feat(models): add final submission v2 with Stacking`
- **Task 8**: `feat(models): add QLoRA fine-tuning with long sequence support`

---

## Success Criteria

### Verification Commands

```bash
# 检查DeBERTa-v3-base OOF
python -c "import numpy as np; oof = np.load('artifacts/models/transformer_v2_oof.npy'); print(f'OOF RMSE: {np.sqrt(np.mean((oof - np.load(\"artifacts/features/y_train.npy\"))**2)):.4f}')"

# 检查增强树模型OOF
python -c "import numpy as np; oof = np.load('artifacts/models/lgb_enhanced_oof.npy'); print(f'LGB OOF RMSE: {np.sqrt(np.mean((oof - np.load(\"artifacts/features/y_train.npy\"))**2)):.4f}')"

# 检查Stacking OOF
python -c "import numpy as np; oof = np.load('artifacts/models/stacking_v2_oof.npy'); print(f'Stacking OOF RMSE: {np.sqrt(np.mean((oof - np.load(\"artifacts/features/y_train.npy\"))**2)):.4f}')"

# 检查最终提交
python -c "import csv; rows = list(csv.reader(open('output/submission-stacking-v2.csv'))); print(f'Rows: {len(rows)}, Header: {rows[0]}')"
```

### Final Checklist

- [ ] DeBERTa-v3-base OOF RMSE < 1.0
- [ ] 增强树模型OOF RMSE < 1.05
- [ ] 集成OOF RMSE < 1.0
- [ ] Kaggle分数 < 0.60
- [ ] 所有代码有适当注释和文档
- [ ] 最终提交文件格式正确
- [ ] 使用HuggingFace Trainer + PEFT（而非手写训练循环）
- [ ] 使用平滑Target Encoding + GroupKFold
- [ ] TF-IDF使用100K（50K词级 + 50K字符级）
- [ ] DeBERTa嵌入用于树模型训练

---

## Design Document

在执行过程中，Sisyphus将创建设计文档：
- 路径：`docs/superpowers/specs/2026-06-10-optimization-chatglm-design.md`
- 内容：包含项目背景、设计方案、技术细节、实现步骤、风险与缓解、成功标准

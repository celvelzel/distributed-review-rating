# COMP5434 Review Rating Prediction — 分级迭代与消融计划

## TL;DR

> **Quick Summary**: 综合 4 个设计方案 (ChatGPT / DeepSeek / GLM / Qwen) 选出一条"**Spark 全分布式 ETL + DeBERTa-v3 文本嵌入 + LightGCN 图嵌入 + Stacking 融合**"主线，在 PolyU COMP5434 项目上拿到代码 / 报告 / 演示三高 (A+)。
>
> **Deliverables**:
> - 端到端 ML pipeline (code/): Spark ETL → 多模态特征 → 多基模 → Stacking → 推理
> - 7 阶段迭代 (Stage 0–6), 每阶段独立提交 Kaggle 并写 changelog
> - 6+ 消融实验 (Stage 7) 逐一证明模块有效性
> - 静态技术展示网页 (website/) 给非技术组员
> - 报告 PDF + 5-10 张 slides + 完整 README + 可复现 run.sh
>
> **Estimated Effort**: Large (≈30+ 任务, 6 个并行 wave)
> **Parallel Execution**: YES — 5 个并行 wave + Final verification wave
> **Critical Path**: T1 (scaffold) → T6 (ETL) → T13 (DeBERTa) → T17 (LGB multimodal) → T20 (Stacking) → T23 (final pipeline) → T24-T29 (6 ablations) → T30 (webpage) → T34 (zip) → F1-F4

---

## Context

### Original Request
用户希望:
1. 对比 `docs/designs/` 下 4 个 LLM 生成的设计方案 (ChatGPT / DeepSeek / GLM / Qwen)
2. 设计一条分级迭代路线, 让系统性能最强 / 准确率最高
3. 每步迭代在 `docs/changelog/` 记录更新内容和性能提升
4. 迭代完成后做消融实验, 逐一移除 / 替换模块, 证明模块有效性
5. 维护一个静态网页, 帮助不熟悉技术的组员快速理解系统

### Interview Summary
**硬件 / 团队 / 优先级 (用户已确认)**:
- 运行环境: 本地单卡 GPU + PySpark 伪分布式
- 团队规模: 3 人小队
- 优先级: 自主评估 — 综合 A+ (Code 6% + Report 5% + Presentation 4% 三高)
- 数据状态: 已下载至 `data/` (train 3M 行 / test 10K 行 / prodInfo 213K 产品)

**Metis 关键反馈**:
- 优先级是综合分 (不是 Kaggle 排名), 每条线都要做, 但有取舍
- 需明确"得分点 → 任务"映射, 避免单纯追 RMSE
- 用户关注 changelog 和 ablation 完整性, 这两块要做得**可重跑、可对比**
- 网页是给非技术组员, 必须是"零技术门槛可看"

### Research Findings
**4 个设计核心差异** (见 `docs/designs/` 完整内容):
| 设计 | 文本能力 | 图能力 | 分布式 | 工程复杂度 | A+ 概率 |
|------|----------|--------|--------|------------|---------|
| ChatGPT | DeBERTa/RoBERTa | Node2Vec/GraphSAGE | Spark ETL | 高 | 高 |
| DeepSeek | TF-IDF | GraphFrames PageRank | 全 Spark | 中 | 中 |
| GLM | BERT base | GraphFrames | SynapseML | 中-高 | 中-高 |
| Qwen | DeBERTa-v3 + LoRA | LightGCN | Spark ETL + PyG | 很高 | 高 |

**综合判断**:
- Qwen 的 **DeBERTa-v3 + LightGCN + Stacking** 是 SOTA 路线, 准确率上限最高
- DeepSeek 的 **全分布式 Spark** 是工程能力展示最佳
- GLM 的 **5-step 流程 + PageRank** 是报告结构化最好
- ChatGPT 的 **三级 Stacking 思路** 最完整

**融合方案 (主线)**:
> DeBERTa-v3 (Qwen) + LightGCN (Qwen) + 全 Spark ETL (DeepSeek) + 5-step 流程 (GLM) + Stacking 框架 (ChatGPT)

---

## Work Objectives

### Core Objective
在 7 个递进阶段内构建并验证一个**多模态 + 分布式**评论评分预测系统, 在 Kaggle RMSE 上达到 ≤ 0.85 (Public) / ≤ 0.88 (Private) 的水平, 同时报告/演示/代码拿到 A+ 综合分。

### Concrete Deliverables
- `code/` — 完整 ML pipeline (ETL / 特征 / 模型 / 训练 / 推理 / 消融)
- `docs/changelog/stage-{0..6}-*.md` + `docs/changelog/ablation-*.md` + `docs/changelog/summary.md`
- `docs/changelog/metrics.json` — 每阶段结构化指标 (RMSE / 训练时间 / 推理时间 / 各模块贡献 %)
- `website/` — 静态网页 (index.html + styles.css + app.js + data/ 嵌入 JSON)
- `report/report.pdf` — 5 章 25-30 页, 包含消融表格与架构图
- `slides/slides.pdf` — 5-10 张
- `code/run.sh` — 一键跑完整 pipeline
- `code/README.md` — 复现指南
- `kaggle/submission-{stage}.csv` — 每阶段一次提交
- 顶层 `TeamName.zip`

### Definition of Done
- [ ] 端到端 pipeline 跑通, `bash code/run.sh` 一键复现
- [ ] 7 阶段 changelog 全部填写, 含数值证据
- [ ] 6 个消融实验跑完, 表格清晰
- [ ] 网页本地能打开, 展示路线 / 性能 / 亮点 / 缺点
- [ ] 报告 PDF 包含三大时间指标 + 消融 + 贡献表
- [ ] Slides 5-10 张
- [ ] README 含硬件/依赖/复现步骤
- [ ] Kaggle 最终一次提交, RMSE 在合理范围
- [ ] `TeamName.zip` 包含 code/ + slides.pdf + report.pdf

### Must Have (得分点 → 任务映射)
- [ ] **Spark 全分布式 ETL** (spec 硬性要求) → T6
- [ ] **多模态特征 (文本 + 统计 + 图)** (A+ 必备) → T8-T14
- [ ] **至少 1 个分布式模型** (effectiveness + efficiency) → T15 (LGB on Spark) 或 PyTorch 分布式
- [ ] **三大时间指标** (报告硬性要求) → T4 (计时器) + T28 (聚合)
- [ ] **消融实验** (报告 A+ 必备) → T22-T27
- [ ] **架构图 / 流程图** (演示硬性要求) → 报告 + slides + 网页
- [ ] **5-step 大数据分析流程** (GLM 推荐, 报告建议结构) → 报告
- [ ] **贡献表** (报告硬性要求) → 报告最后一页
- [ ] **Static webpage with 4 块内容** (用户硬性要求) → T5 + T30
- [ ] **Changelog 7 阶段 + 6 消融** (用户硬性要求) → 散落各任务, 在 T34 收尾时生成 `docs/changelog/summary.md` 总览

### Must NOT Have (Guardrails)
- **不做** ONNX Runtime 优化 (Qwen 提议, 复杂度高, 时间紧, 性价比低)
- **不做** W&B 实验追踪 (需账号, MLflow 本地替代足够)
- **不做** LoRA 微调 DeBERTa (Qwen 提议, 显存 / 时间风险大, off-the-shelf embedding 足够)
- **不做** 对抗验证 adversarial validation 作为强约束 (Qwen 提议, 仅作为 EDA 探索, 不阻塞流程)
- **不引入** 训练数据外的外部数据 / 词典 / 预训练外部语料
- **不重写** LightGBM / PyTorch / Spark 内部, 只调用公开 API
- **不** 提交超过 1 次 / 天的 Kaggle 提交 (避免 public leaderboard 过拟合, spec 未禁止但要谨慎)
- **不** 改动 `data/` 目录下的原始 CSV
- **不** 在 `docs/designs/` 上修改原始设计文档 (它们是参考材料)

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — 全部由 agent 自行验证。
> 报告/演示/网页/代码评审无需人介入 (Kaggle 提交除外, 用户手动操作)。

### Test Decision
- **Infrastructure exists**: NO (项目从零开始, 无现成 test 框架)
- **Automated tests**: YES (Tests-after) — 每个关键模块加单元测试 + 集成测试
- **Framework**: pytest (Python 生态最广, 与 PySpark / PyTorch 兼容)
- **Test scope**:
  - ETL: 加载 100 行样本, 验证 schema / 缺失处理 / 特征正确
  - Features: 单元测试每个 transform 函数的输入输出
  - Models: 模型保存/加载 round-trip + 推理 shape
  - End-to-end: 100 行子集跑完整 pipeline, 验证 submission shape
  - 时间指标: 验证计时器 decorator 正确累计
  - 网页: Playwright 打开 index.html 验证 4 大块内容存在

### QA Policy
每个任务必须有 **agent-executed QA scenarios** (至少 1 happy + 1 failure)。
证据保存到 `.sisyphus/evidence/task-{N}-{slug}.{ext}`:
- **ETL / Spark**: Bash + `spark-submit` 小样本 + 验证输出行数
- **特征工程**: Bash + 加载特征 parquet, pandas 验证 shape 和示例
- **模型训练**: Bash + 跑训练 + 验证模型文件存在 + 验证 metric 数值
- **Kaggle 提交**: 用户手动操作 (无法 agent 自动化), 但 agent 准备 submission.csv 并验证格式
- **网页**: Playwright 启动本地 server + 截图 + 验证 DOM
- **报告 / Slides**: 用户打开 PDF 验证 (agent 不能读 PDF, 仅生成)

---

## Execution Strategy

### Parallel Execution Waves

> **目标**: 5-8 任务 / wave, 最大化吞吐。Wave 之间有强依赖, Wave 内部全并行。

```
Wave 1 (Foundation — scaffold + infra, 7 tasks, 全并行):
├── T1: 项目脚手架 (folder, README, requirements.txt, .gitignore)
├── T2: 数据探索 EDA (notebook + 关键统计输出)
├── T3: PySpark 环境 + 配置 (spark config + local[N] 启动脚本)
├── T4: 计时器 + metrics 基建 (decorator + JSON log)
├── T5: 网页骨架 (index.html + CSS + 4 个 section 占位)
├── T6: Spark ETL 模块 (读 CSV → 清洗 → Join → 写 Parquet)
└── T7: Stage 0 Baseline (TF-IDF + LGB on Spark, 第一次 Kaggle 提交)

Wave 2 (Statistical features + EDA 增强, 5 tasks, 全并行, 依赖 T6):
├── T8: 用户 / 产品 / 类别 统计特征
├── T9: 时间 / 价格 / 长度 特征
├── T10: K-Fold Target Encoding (防泄漏)
├── T11: 对抗验证 (adversarial validation, EDA 增强)
└── T12: Stage 1 训练 + changelog + Kaggle 提交

Wave 3 (深度特征 — DeBERTa + LightGCN, 5 tasks, 全并行, 依赖 T6 + T8):
├── T13: DeBERTa-v3 嵌入提取 (PyTorch + Pandas UDF 分布式, 768d)
├── T14: LightGCN 图嵌入训练 (User-Item 二部图, 64d)
├── T15: 备选 Node2Vec 嵌入 (若 T14 时间紧, 备选)
├── T16: 多模态特征拼接 + 标准化
└── T17: Stage 2-3 训练 (LGB with all features) + changelog + 提交

Wave 4 (Stacking + 优化, 6 tasks, 全并行, 依赖 T13-T16):
├── T18: CatBoost 训练 (基模 2)
├── T19: MLP 训练 (基模 3, on embeddings)
├── T20: Stacking (OOF + Ridge meta)
├── T21: 超参调优 (Optuna, LGB 主模)
├── T22: 阈值舍入 + Clip [1, 5]
└── T23: Stage 4-5 最终 pipeline + 提交 + changelog

Wave 5 (消融实验 — 6 个, 全并行, 依赖 T23):
├── T24: Ablation A — 无文本特征 (仅 stats)
├── T25: Ablation B — 无图特征 (text + stats only)
├── T26: Ablation C — 无 Stacking (单 LGB)
├── T27: Ablation D — 无 K-Fold TE (用 naive mean)
├── T28: Ablation E — 无 DeBERTa (用 TF-IDF)
└── T29: Ablation F — 无 Clip (看 RMSE 分布)

Wave 6 (交付物 — 文档/网页/打包, 5 tasks, 全并行, 依赖 T24-T29):
├── T30: 网页内容填充 (从 metrics.json 生成可视化)
├── T31: 报告 PDF 生成 (LaTeX 或 markdown → PDF)
├── T32: Slides 生成 (5-10 张)
├── T33: README 完善 + 复现指南
└── T34: TeamName.zip 打包 + 验证

Wave FINAL (4 任务并行, 依赖 T30-T34):
├── F1: Plan compliance audit (oracle) — 对照本 plan 检查完成度
├── F2: Code quality review (unspecified-high) — 静态检查 + 测试
├── F3: Real manual QA (unspecified-high + playwright) — 端到端跑通
└── F4: Scope fidelity check (deep) — 是否有范围蔓延
```

### Dependency Matrix (任务依赖)

| Task | Depends on | Blocks |
|------|-----------|--------|
| T1 | None | T3-T5, T33 |
| T2 | None | T6, T11 |
| T3 | T1 | T6 |
| T4 | T1 | T28 |
| T5 | T1 | T30 |
| T6 | T1-T3, T2 | T8-T11, T13, T14 |
| T7 | T6 | T12 |
| T8 | T6 | T12, T13 |
| T9 | T6 | T12 |
| T10 | T6, T8 | T12 |
| T11 | T6 | T12 |
| T12 | T7-T11 | T17 |
| T13 | T6, T8 | T16, T17 |
| T14 | T6 | T16, T17 |
| T15 | T6 | T16, T17 (备选) |
| T16 | T13, T14 | T17 |
| T17 | T12, T16 | T18-T22 |
| T18 | T16 | T20 |
| T19 | T13, T14 | T20 |
| T20 | T17, T18, T19 | T23 |
| T21 | T17 | T23 |
| T22 | T23 | T28 |
| T23 | T20, T21, T22 | T24-T29 |
| T24-T29 | T23 | T30-T34 |
| T30 | T23-T29, T5 | T31, F1 |
| T31 | T23-T29 | F1 |
| T32 | T23-T29, T30 | F1 |
| T33 | T1, T23 | F1 |
| T34 | T30-T33 | F1 |
| F1-F4 | T34 | (交付) |

### Agent Dispatch Summary
- **Wave 1 (T1-T7)**: 7 tasks — T1, T4, T5 = `quick`; T2, T3 = `unspecified-high`; T6 = `unspecified-high`; T7 = `unspecified-high`
- **Wave 2 (T8-T12)**: 5 tasks — T8, T9 = `unspecified-high`; T10, T11 = `deep`; T12 = `unspecified-high`
- **Wave 3 (T13-T17)**: 5 tasks — T13, T14 = `ultrabrain`; T15 = `unspecified-high`; T16 = `unspecified-high`; T17 = `unspecified-high`
- **Wave 4 (T18-T23)**: 6 tasks — T18, T19 = `unspecified-high`; T20 = `ultrabrain`; T21 = `deep`; T22 = `quick`; T23 = `unspecified-high`
- **Wave 5 (T24-T29)**: 6 tasks — All `unspecified-high` (6 个独立 ablation, 跑一遍 6 个超参, 记录)
- **Wave 6 (T30-T34)**: 5 tasks — T30 = `visual-engineering`; T31, T32 = `writing`; T33 = `writing`; T34 = `quick`
- **FINAL**: 4 tasks — F1 = `oracle`; F2 = `unspecified-high`; F3 = `unspecified-high` (+ playwright); F4 = `deep`

---

## TODOs

> 任务格式: 一个任务 = 一个模块/关注点 = 1-3 个文件。
> 每个任务必须包含: 实施步骤 / 推荐 Agent / 并行信息 / 引用 / 验收 / QA 场景。
> **没有 QA 场景的任务视为不完整**。

---

- [x] 1. **项目脚手架 (Scaffold)**

  **What to do**:
  - 创建 `code/` 目录, 内含 `etl/`, `features/`, `models/`, `ablation/`, `utils/`, `tests/`, `website/`, `kaggle/`
  - 写 `code/README.md`: 项目说明 / 硬件要求 / 依赖列表 / 复现步骤
  - 写 `code/requirements.txt`: pyspark==3.4.1, torch, transformers, lightgbm, catboost, xgboost, pandas, numpy, scikit-learn, mlflow, optuna, pytest
  - 写 `.gitignore`: `__pycache__/`, `*.pyc`, `data/*.parquet`, `artifacts/`, `mlruns/`, `.sisyphus/evidence/`
  - 写 `code/run.sh`: 一键跑 `etl → features → train → predict → submit` 的 bash 脚本 (各阶段调用占位 stub, 后阶段会填充)
  - 顶层 README.md: 项目入口 (链接到 code/README.md, docs/changelog/summary.md, website/index.html)

  **Must NOT do**:
  - 不写具体实现代码 (留给后续任务)
  - 不创建 `data/` 下任何内容

  **Recommended Agent Profile**:
  - **Category**: `quick` — 单文件多, 复杂度低
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 1 first batch)
  - **Parallel Group**: Wave 1 (with T2-T7)
  - **Blocks**: T3, T4, T5, T6, T7, T33
  - **Blocked By**: None (start immediately)

  **References**:
  - 现有设计: `docs/designs/deepseek_design.md:183-198` (代码组织示例)
  - Spec: `docs/specification/spec.txt:7` (Submission 目录要求)

  **Acceptance Criteria**:
  - [ ] `code/README.md` 存在且 ≥ 50 行
  - [ ] `code/requirements.txt` 含全部依赖
  - [ ] `code/run.sh` 可执行 (`chmod +x`), 跑通空 stub 不报错
  - [ ] `.gitignore` 含 6+ 条目
  - [ ] `bash code/run.sh --help` 打印使用说明

  **QA Scenarios**:
  ```
  Scenario: 脚手架文件完整性
    Tool: Bash
    Preconditions: 项目已 git init
    Steps:
      1. ls code/ -la  (验证子目录)
      2. cat code/requirements.txt | wc -l (≥ 10 行)
      3. bash code/run.sh --help (验证脚本不崩)
    Expected Result: 所有子目录存在, 依赖列表完整, run.sh 退出码 0
    Evidence: .sisyphus/evidence/task-1-scaffold-files.txt

  Scenario: .gitignore 阻止敏感文件
    Tool: Bash
    Preconditions: code/ 已创建
    Steps:
      1. mkdir -p code/__pycache__ && touch code/__pycache__/test.pyc
      2. git check-ignore code/__pycache__/test.pyc  (期望: 退出码 0, 报告被 ignore)
    Expected Result: __pycache__ 被忽略
    Evidence: .sisyphus/evidence/task-1-gitignore.txt
  ```

  **Commit**: YES
  - Message: `chore: scaffold project structure`
  - Files: `code/`, `README.md`, `.gitignore`

---

- [ ] 2. **数据探索 EDA**

  **What to do**:
  - 创建 `code/etl/eda.py` 或 `code/notebooks/eda.ipynb` (建议 .py 模块化)
  - 加载 train.csv, test.csv, prodInfo.csv (Pandas 读前 100K 行 + PySpark 读全集统计)
  - 输出关键统计:
    - train 行数, test 行数, prodInfo 行数
    - 缺失率 (每列)
    - rating 分布 (1-5 各多少)
    - 评论长度分布 (title, comment)
    - user_id 唯一数, prod_id 唯一数, parent_prod_id 唯一数
    - 时间范围 (最早, 最晚)
    - train/test 用户/产品重叠率
    - 冷启动用户比例
  - 写报告到 `docs/changelog/eda-report.md` (含 5-8 张 matplotlib/seaborn 图)
  - 写 `docs/changelog/metrics.json` 初始结构 (空 metric 字段, 后续 stage 填)

  **Must NOT do**:
  - 不修改任何 CSV
  - 不训练任何模型

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — 多文件多步骤, 涉及可视化
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T6 (ETL 依赖 EDA 找出的字段特征), T11 (adversarial validation)
  - **Blocked By**: None

  **References**:
  - 数据样本: `data/train.csv` (第一行已确认 schema)
  - 4 个设计中关于字段的描述: `docs/designs/chatgpt_design.md:30-50`

  **Acceptance Criteria**:
  - [ ] `code/etl/eda.py` 存在且可跑 (`python code/etl/eda.py`)
  - [ ] `docs/changelog/eda-report.md` 含 5+ 统计点
  - [ ] `docs/changelog/figures/eda-*.png` 至少 5 张图
  - [ ] `docs/changelog/metrics.json` 含 schema 模板

  **QA Scenarios**:
  ```
  Scenario: EDA 脚本成功跑通
    Tool: Bash
    Preconditions: data/ 完整
    Steps:
      1. python code/etl/eda.py 2>&1 | tee .sisyphus/evidence/task-2-eda-log.txt
      2. 验证退出码 0
      3. ls docs/changelog/figures/eda-*.png | wc -l  (≥ 5)
    Expected Result: 退出码 0, 生成 ≥ 5 张图
    Evidence: .sisyphus/evidence/task-2-eda-log.txt

  Scenario: EDA 报告含关键指标
    Tool: Bash
    Preconditions: EDA 已跑
    Steps:
      1. grep -E "rows|missing|distribution" docs/changelog/eda-report.md | wc -l  (≥ 10)
    Expected Result: 报告含 ≥ 10 条统计描述
    Evidence: .sisyphus/evidence/task-2-report-coverage.txt
  ```

  **Commit**: YES
  - Message: `feat(eda): add exploratory data analysis`
  - Files: `code/etl/eda.py`, `docs/changelog/eda-report.md`, `docs/changelog/figures/`

---

- [ ] 3. **PySpark 环境与配置**

  **What to do**:
  - 创建 `code/utils/spark_session.py`: 单例 `get_spark()` 工厂函数, 配置 `local[N]`, `spark.sql.shuffle.partitions=200`, `spark.driver.memory=4g`, `spark.sql.broadcastTimeout=600`
  - 创建 `code/config.py`: 集中管理路径 (`DATA_DIR`, `ARTIFACTS_DIR`, `OUTPUT_DIR`), 阶段常量 (`STAGES = {0,1,...,6}`), 随机种子
  - 创建 `code/run_spark.sh`: `spark-submit` 模板, 包含 `--master local[*] --driver-memory 4g --conf spark.sql.broadcastTimeout=600`
  - 创建 `code/utils/__init__.py` 暴露公共接口

  **Must NOT do**:
  - 不安装 Spark (假设系统已有 PySpark)
  - 不写 ETL 逻辑 (留给 T6)

  **Recommended Agent Profile**:
  - **Category**: `quick` — 配置/模板, 单文件为主
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T6, T7
  - **Blocked By**: T1

  **References**:
  - DeepSeek 设计的 Spark 配置: `docs/designs/deepseek_design.md:150-160`
  - Qwen 设计的 broadcast: `docs/designs/qwen_design.md:60-85`

  **Acceptance Criteria**:
  - [ ] `code/utils/spark_session.py` 存在
  - [ ] `get_spark()` 跑通, 创建一个 SparkSession 实例
  - [ ] `code/config.py` 含 `STAGES = [0,1,2,3,4,5,6]`
  - [ ] `code/run_spark.sh` 含 `spark-submit` 模板
  - [ ] `python -c "from code.utils.spark_session import get_spark; s = get_spark(); print(s.version); s.stop()"` 退出码 0

  **QA Scenarios**:
  ```
  Scenario: SparkSession 可创建
    Tool: Bash
    Preconditions: T1 完成, PySpark 已装
    Steps:
      1. python -c "from code.utils.spark_session import get_spark; s = get_spark(); print('spark', s.version); s.stop()"
      2. 验证 stdout 含 "spark 3." 格式
    Expected Result: 退出码 0, 打印 Spark 版本
    Evidence: .sisyphus/evidence/task-3-spark-init.txt

  Scenario: Config 路径可解析
    Tool: Bash
    Preconditions: code/config.py 存在
    Steps:
      1. python -c "from code.config import DATA_DIR, ARTIFACTS_DIR, OUTPUT_DIR, STAGES; print(len(STAGES))"
    Expected Result: 输出 7 (STAGES 含 0-6 共 7 项)
    Evidence: .sisyphus/evidence/task-3-config.txt
  ```

  **Commit**: YES
  - Message: `chore(spark): add SparkSession factory and config`
  - Files: `code/utils/spark_session.py`, `code/config.py`, `code/run_spark.sh`

---

- [ ] 4. **计时器与 metrics 基建**

  **What to do**:
  - 创建 `code/utils/timer.py`:
    - `@timed(stage_name, metric_key)` decorator: 包裹函数, 自动记录 elapsed 秒数
    - `StageTimer` class: 维护当前 stage 的所有计时
    - `write_metrics(output_path)`: 写 JSON 到 `docs/changelog/metrics.json`
  - 定义 metrics.json schema:
    ```json
    {
      "stages": {
        "0": {"rmse": null, "train_time_sec": null, "inference_time_sec": null, "model": "tfidf_lgb", "features": ["tfidf"]},
        "1": {...},
        ...
      },
      "ablations": {
        "no_text": {"rmse": null, "delta_vs_full": null},
        ...
      }
    }
    ```
  - 创建 `code/tests/test_timer.py`: 单元测试 decorator 正确记录时间
  - 写一份使用示例到 `code/utils/README.md`

  **Must NOT do**:
  - 不在 ETL 内部硬编码时间打印 (用 decorator)

  **Recommended Agent Profile**:
  - **Category**: `quick` — 单工具, 单元测试明确
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T6, T7, T28
  - **Blocked By**: T1

  **References**:
  - GLM 设计的计时代码: `docs/designs/GLM_design.md:62-90`

  **Acceptance Criteria**:
  - [ ] `code/utils/timer.py` 存在
  - [ ] `docs/changelog/metrics.json` 初始 schema 写入
  - [ ] `pytest code/tests/test_timer.py -v` 全部 PASS (≥ 3 测试)
  - [ ] decorator 用法示例在 README

  **QA Scenarios**:
  ```
  Scenario: 计时器 decorator 准确
    Tool: Bash
    Preconditions: T4 完成
    Steps:
      1. pytest code/tests/test_timer.py -v
      2. 验证 ≥ 3 tests passed
    Expected Result: 全部 PASS
    Evidence: .sisyphus/evidence/task-4-timer-tests.txt

  Scenario: metrics.json 写入正确
    Tool: Bash
    Preconditions: T4 完成
    Steps:
      1. python -c "from code.utils.timer import write_metrics; write_metrics({'stages': {}}); import json; d=json.load(open('docs/changelog/metrics.json')); assert 'stages' in d"
    Expected Result: metrics.json 含 stages 键
    Evidence: .sisyphus/evidence/task-4-metrics-init.txt
  ```

  **Commit**: YES
  - Message: `feat(infra): add timing decorator and metrics infra`
  - Files: `code/utils/timer.py`, `code/tests/test_timer.py`, `docs/changelog/metrics.json`

---

- [ ] 5. **网页骨架 (Website Skeleton)**

  **What to do**:
  - 创建 `website/index.html`: 4 大块的占位结构
    - `<section id="roadmap">` 技术路线
    - `<section id="performance">` 系统性能
    - `<section id="highlights">` 亮点
    - `<section id="weaknesses">` 缺点
  - 创建 `website/styles.css`: 现代简洁风 (中文字体优先, 渐变色 header, 卡片式 sections)
  - 创建 `website/app.js`: 加载 `data/metrics.json`, 渲染表格 + 4 个 Chart.js 图 (RMSE 趋势 / 训练时间 / 推理时间 / 消融对比)
  - 创建 `website/data/metrics.json` 占位 (T4 已创建, 复制过来)
  - 创建 `website/README.md`: 部署说明 (本地 `python -m http.server`, GitHub Pages)

  **Must NOT do**:
  - 不引入 React/Vue (无 build step)
  - 不引入 Tailwind (单 CSS 即可)

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering` — 视觉/交互, 面向非技术用户
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T30
  - **Blocked By**: T1

  **References**:
  - ChatGPT 设计中关于"非技术组员理解" 的建议: `docs/designs/chatgpt_design.md:179-181`

  **Acceptance Criteria**:
  - [ ] `website/index.html` 存在, 含 4 个 section
  - [ ] `website/styles.css` 存在, 至少 100 行
  - [ ] `website/app.js` 存在
  - [ ] `python -m http.server 8000 --directory website/` 启动后, `curl localhost:8000` 返回 HTML

  **QA Scenarios**:
  ```
  Scenario: 网页 4 大块结构存在
    Tool: Bash
    Preconditions: T5 完成
    Steps:
      1. python -m http.server 8000 --directory website/ &
      2. sleep 1; curl -s localhost:8000/ | grep -c 'section id='  (期望 ≥ 4)
      3. kill %1
    Expected Result: 4 个 section 存在
    Evidence: .sisyphus/evidence/task-5-html-sections.txt

  Scenario: 页面用 Playwright 渲染
    Tool: Playwright
    Preconditions: T5 完成, server 启动
    Steps:
      1. mcp_playwright: navigate to http://localhost:8000/
      2. screenshot page
      3. assert 4 个 section 可见 (querySelectorAll('section').length === 4)
    Expected Result: 4 sections rendered
    Evidence: .sisyphus/evidence/task-5-page-screenshot.png
  ```

  **Commit**: YES
  - Message: `feat(website): add static website skeleton`
  - Files: `website/index.html`, `website/styles.css`, `website/app.js`, `website/README.md`

---

- [ ] 6. **Spark ETL 模块**

  **What to do**:
  - 创建 `code/etl/spark_etl.py`:
    - `load_train(spark)` → DataFrame
    - `load_test(spark)` → DataFrame
    - `load_prodinfo(spark)` → DataFrame
    - `clean_text(df, col_name)` → 清洗 HTML/URL/特殊字符, 转小写
    - `impute_missing(df)` → title/comment 填 "unknown", price 按 main_category 中位数, votes 填 0
    - `join_with_prodinfo(df_train, df_prodinfo)` → broadcast join on prod_id
    - `extract_time_features(df)` → year/month/weekday/hour/is_weekend
    - `persist_parquet(df, output_path)` → 写 Parquet
  - 创建 `code/etl/run_etl.py`: 跑完整 ETL, 输出 `artifacts/etl/train.parquet`, `test.parquet`, `prodinfo.parquet`
  - 创建 `code/tests/test_etl.py`: 单元测试 (100 行 sample 输入, 验证输出 schema/rows)

  **Must NOT do**:
  - 不做特征工程 (user_avg, prod_avg, etc., 留给 T8)
  - 不写模型训练 (留给 T7)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — 多函数, 涉及 Spark broadcast / cache / persist
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T7-T17
  - **Blocked By**: T1, T3, T2

  **References**:
  - DeepSeek 设计的 ETL: `docs/designs/deepseek_design.md:50-95`
  - Qwen 设计的 broadcast: `docs/designs/qwen_design.md:60-85`

  **Acceptance Criteria**:
  - [ ] `code/etl/spark_etl.py` 含 7+ 函数
  - [ ] `python code/etl/run_etl.py` 跑通, 生成 3 个 Parquet
  - [ ] `pytest code/tests/test_etl.py -v` 全部 PASS (≥ 5 测试)
  - [ ] Parquet 行数与 CSV 一致 (train 3007439, test 10000, prodInfo 213593)

  **QA Scenarios**:
  ```
  Scenario: ETL 端到端跑通
    Tool: Bash
    Preconditions: T1-T3, T2 完成, data/ 完整
    Steps:
      1. python code/etl/run_etl.py 2>&1 | tee .sisyphus/evidence/task-6-etl-log.txt
      2. 验证退出码 0
      3. ls artifacts/etl/*.parquet  (期望 3 个文件)
    Expected Result: 3 个 Parquet 写出
    Evidence: .sisyphus/evidence/task-6-etl-log.txt

  Scenario: ETL 单元测试
    Tool: Bash
    Preconditions: ETL 实现完成
    Steps:
      1. pytest code/tests/test_etl.py -v
      2. 验证 ≥ 5 tests passed
    Expected Result: 全部 PASS
    Evidence: .sisyphus/evidence/task-6-etl-tests.txt

  Scenario: 行数一致性
    Tool: Bash
    Preconditions: ETL 跑完
    Steps:
      1. python -c "import pyarrow.parquet as pq; t=pq.read_table('artifacts/etl/train.parquet'); print(t.num_rows)"
      2. 期望: 3007439
    Expected Result: 行数与 train.csv 一致
    Evidence: .sisyphus/evidence/task-6-row-count.txt
  ```

  **Commit**: YES
  - Message: `feat(etl): add Spark ETL pipeline with broadcast join`
  - Files: `code/etl/spark_etl.py`, `code/etl/run_etl.py`, `code/tests/test_etl.py`

---

- [ ] 7. **Stage 0 Baseline (TF-IDF + LightGBM)**

  **What to do**:
  - 创建 `code/features/text_tfidf.py`: `compute_tfidf(train_df, test_df, max_features=5000)` → sparse matrix
  - 创建 `code/models/lgb_baseline.py`: `train_lgb(X_train, y_train, X_val, y_val)` → model
  - 创建 `code/models/train_stage0.py`: 串起来, 5-fold CV, 训练 LGB on TF-IDF only
  - 创建 `code/models/predict.py`: `predict(model, X_test)` → 预测, clip [1, 5]
  - 第一次 Kaggle 提交: `kaggle/submission-stage0.csv` (id, rating)
  - 写 `docs/changelog/stage-0-baseline.md`:
    - 实施日期, 变更内容, 配置参数
    - 5-fold OOF RMSE
    - 训练时间, 推理时间
    - Kaggle Public/Private 分数 (用户提交后填)
  - 更新 `docs/changelog/metrics.json` 的 stage 0 字段

  **Must NOT do**:
  - 不加统计特征 (留给 Stage 1)
  - 不调超参 (默认 LightGBM 参数)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — 端到端 pipeline, 多文件协同
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (是 Wave 1 的收尾)
  - **Parallel Group**: Sequential (depends T6)
  - **Blocks**: T12
  - **Blocked By**: T1, T3, T4, T6

  **References**:
  - DeepSeek TF-IDF pipeline: `docs/designs/deepseek_design.md:65-80`
  - ChatGPT 路径: `docs/designs/chatgpt_design.md:228-280`

  **Acceptance Criteria**:
  - [ ] `python code/models/train_stage0.py` 跑通
  - [ ] 5-fold OOF RMSE < 1.3 (LGB on TF-IDF 应该 ≤ 1.3)
  - [ ] `kaggle/submission-stage0.csv` 生成, 10001 行 (含 header)
  - [ ] `docs/changelog/stage-0-baseline.md` 填写完整
  - [ ] metrics.json 的 stage 0 字段填值

  **QA Scenarios**:
  ```
  Scenario: Stage 0 训练完成
    Tool: Bash
    Preconditions: T6 完成
    Steps:
      1. python code/models/train_stage0.py 2>&1 | tee .sisyphus/evidence/task-7-stage0-train.txt
      2. 验证退出码 0
      3. grep "OOF RMSE" .sisyphus/evidence/task-7-stage0-train.txt  (期望: 数字 < 1.3)
    Expected Result: OOF RMSE 数字 < 1.3 打印
    Evidence: .sisyphus/evidence/task-7-stage0-train.txt

  Scenario: 提交 CSV 格式正确
    Tool: Bash
    Preconditions: Stage 0 训练完
    Steps:
      1. head -1 kaggle/submission-stage0.csv  (期望: id,rating)
      2. wc -l kaggle/submission-stage0.csv  (期望: 10001)
      3. awk -F',' 'NR>1 {if ($2<1 || $2>5) print "FAIL"}' kaggle/submission-stage0.csv | wc -l  (期望: 0)
    Expected Result: 10000 行数据, rating∈[1,5]
    Evidence: .sisyphus/evidence/task-7-submission-format.txt

  Scenario: Changelog 填写完整
    Tool: Bash
    Preconditions: Stage 0 完成
    Steps:
      1. grep -E "RMSE|Train|Inference" docs/changelog/stage-0-baseline.md | wc -l  (期望 ≥ 3)
    Expected Result: 3+ 指标记录
    Evidence: .sisyphus/evidence/task-7-changelog.txt
  ```

  **Commit**: YES
  - Message: `feat(stage-0): TF-IDF + LightGBM baseline, first submission`
  - Files: `code/features/text_tfidf.py`, `code/models/lgb_baseline.py`, `code/models/train_stage0.py`, `kaggle/submission-stage0.csv`, `docs/changelog/stage-0-baseline.md`, `docs/changelog/metrics.json`

---

- [ ] 8. **用户 / 产品 / 类别 统计特征**

  **What to do**:
  - 创建 `code/features/user_stats.py`:
    - `compute_user_avg_rating(df)` → 按 user_id groupBy, 计算 avg_rating, num_reviews, avg_votes, purchased_rate, rating_std
    - 输出: `artifacts/features/user_stats.parquet`
  - 创建 `code/features/product_stats.py`:
    - `compute_product_stats(df, prodinfo_df)` → prod_avg_rating (from train), prod_num_reviews, prod_price, prod_rating_number (from prodInfo), main_category
    - 输出: `artifacts/features/product_stats.parquet`
  - 创建 `code/features/category_stats.py`:
    - `compute_category_stats(df, prodinfo_df)` → 按 main_category 聚合, 类别平均评分, 类别平均价格, 类别评分方差
  - 创建 `code/features/run_stats.py`: 串起来跑

  **Must NOT do**:
  - 不做 K-Fold target encoding (留给 T10)
  - 不做时间特征 (留给 T9)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — 多个独立 groupBy, Spark 优化
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T10, T12
  - **Blocked By**: T6

  **References**:
  - DeepSeek 用户/产品聚合: `docs/designs/deepseek_design.md:78-91`
  - Qwen 的 K-Fold TE 思路: `docs/designs/qwen_design.md:104-106`

  **Acceptance Criteria**:
  - [ ] 3 个 Parquet 写出
  - [ ] user_avg_rating 数值范围 [1, 5]
  - [ ] product_price 缺失已被填充
  - [ ] `pytest code/tests/test_stats.py -v` 全部 PASS

  **QA Scenarios**:
  ```
  Scenario: 统计特征生成
    Tool: Bash
    Preconditions: T6 完成
    Steps:
      1. python code/features/run_stats.py 2>&1 | tee .sisyphus/evidence/task-8-stats.txt
      2. ls artifacts/features/*.parquet  (期望 3+)
    Expected Result: 3 个 parquet 写出
    Evidence: .sisyphus/evidence/task-8-stats.txt

  Scenario: 统计特征数值范围正确
    Tool: Bash
    Preconditions: T8 完成
    Steps:
      1. python -c "import pyarrow.parquet as pq; import pandas as pd; df=pq.read_table('artifacts/features/user_stats.parquet').to_pandas(); assert df['avg_rating'].between(1, 5).all(); print('OK')"
    Expected Result: 打印 OK
    Evidence: .sisyphus/evidence/task-8-range.txt
  ```

  **Commit**: YES
  - Message: `feat(features): add user/product/category statistical features`
  - Files: `code/features/user_stats.py`, `code/features/product_stats.py`, `code/features/category_stats.py`, `code/features/run_stats.py`, `code/tests/test_stats.py`

---

- [ ] 9. **时间 / 价格 / 长度 特征**

  **What to do**:
  - 创建 `code/features/temporal.py`:
    - `extract_temporal(df)` → year, month, day, weekday, hour, is_weekend, is_holiday_season (黑五/圣诞 flag)
  - 创建 `code/features/price_features.py`:
    - `extract_price(prodinfo_df)` → log_price, price_rank_in_category, price_bucket (low/mid/high)
  - 创建 `code/features/text_length.py`:
    - `extract_length(df)` → title_len, comment_len, title_comment_ratio, has_caps, has_exclamation
  - 串到 `code/features/run_stats.py` (与 T8 合并跑, 或单跑)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — 多独立特征, 易并行
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T12
  - **Blocked By**: T6

  **References**:
  - DeepSeek 时间特征: `docs/designs/deepseek_design.md:93-95`
  - ChatGPT 长度特征思路: `docs/designs/chatgpt_design.md:357-380`

  **Acceptance Criteria**:
  - [ ] 3 个特征文件生成
  - [ ] 数值特征无 NaN
  - [ ] `pytest code/tests/test_misc_features.py -v` PASS

  **QA Scenarios**:
  ```
  Scenario: 杂项特征无 NaN
    Tool: Bash
    Preconditions: T9 完成
    Steps:
      1. python -c "import pyarrow.parquet as pq; df=pq.read_table('artifacts/features/temporal.parquet').to_pandas(); assert df.isna().sum().sum() == 0; print('OK')"
    Expected Result: 打印 OK
    Evidence: .sisyphus/evidence/task-9-no-nan.txt
  ```

  **Commit**: YES
  - Message: `feat(features): add temporal/price/text-length features`
  - Files: `code/features/temporal.py`, `code/features/price_features.py`, `code/features/text_length.py`, `code/tests/test_misc_features.py`

---

- [ ] 10. **K-Fold Target Encoding (防泄漏)**

  **What to do**:
  - 创建 `code/features/target_encoding.py`:
    - `kf_target_encode(train_df, test_df, group_col, target_col, n_splits=5, smoothing=1.0)` → 输出 train 和 test 的 target encoding 列
    - 内部用 sklearn KFold, 在每一折用其他折的均值填充
    - 支持 group_col ∈ {user_id, prod_id, main_category, (user_id, prod_id)}
  - 创建 `code/tests/test_target_encoding.py`:
    - 验证 train 的 target encoding 不等于原始 target (无泄漏)
    - 验证 test 的 target encoding 用全量 train 计算
  - 写报告到 `docs/changelog/target-encoding-design.md`: 防泄漏流程图 (可用 mermaid)

  **Must NOT do**:
  - 不直接用 train 全集计算 target encoding (泄漏)

  **Recommended Agent Profile**:
  - **Category**: `deep` — 目标泄漏风险, 需要仔细测试
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T12, T17
  - **Blocked By**: T6, T8

  **References**:
  - Qwen 防泄漏设计: `docs/designs/qwen_design.md:104-106`
  - 通用 K-Fold TE 教程: (无内部 ref, 走 KFold 标准做法)

  **Acceptance Criteria**:
  - [ ] `code/features/target_encoding.py` 实现正确
  - [ ] `pytest code/tests/test_target_encoding.py -v` PASS (≥ 3 测试)
  - [ ] 文档含防泄漏流程图

  **QA Scenarios**:
  ```
  Scenario: K-Fold TE 不泄漏
    Tool: Bash
    Preconditions: T10 完成
    Steps:
      1. pytest code/tests/test_target_encoding.py -v
      2. 验证包含 "no_leakage" 或 "leakage" 的测试通过
    Expected Result: 全部 PASS
    Evidence: .sisyphus/evidence/task-10-te-tests.txt

  Scenario: test 用全量 train mean
    Tool: Bash
    Preconditions: T10 完成
    Steps:
      1. python -c "from code.features.target_encoding import kf_target_encode; from pyspark.sql import SparkSession; s=SparkSession.builder.master('local[1]').getOrCreate(); import pandas as pd; df=pd.DataFrame({'g':['a','a','b','b','c','c'],'y':[1,5,2,4,3,5]}); res=kf_target_encode(s.createDataFrame(df), None, 'g', 'y', n_splits=2); s.stop(); print(res.toPandas().to_dict())"
    Expected Result: 返回 DataFrame 含编码列
    Evidence: .sisyphus/evidence/task-10-te-output.txt
  ```

  **Commit**: YES
  - Message: `feat(features): add K-Fold target encoding with leak prevention`
  - Files: `code/features/target_encoding.py`, `code/tests/test_target_encoding.py`, `docs/changelog/target-encoding-design.md`

---

- [ ] 11. **对抗验证 (Adversarial Validation)**

  **What to do**:
  - 创建 `code/features/adversarial_validation.py`:
    - `adversarial_validate(train_df, test_df, feature_cols)`:
      - 给 train 标 0, test 标 1
      - 训练 LightGBM 二分类
      - 输出 AUC, Feature Importance
    - `identify_distribution_shift(train_df, test_df, threshold_auc=0.6)`:
      - AUC > threshold → 警告, 输出需要剔除或调整的特征列表
  - 跑一次, 输出到 `docs/changelog/adversarial-validation.md`:
    - AUC 数字
    - Top-10 分布差异特征
    - 建议的调整 (剔除 / 重新编码)
  - 应用建议 (如果有需要剔除的特征, 在 T16 特征拼接时去掉)

  **Recommended Agent Profile**:
  - **Category**: `deep` — 探索性分析, 不阻塞流程
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T16 (轻微, 仅当建议剔除特征时)
  - **Blocked By**: T6

  **References**:
  - Qwen 对抗验证: `docs/designs/qwen_design.md:108-114`

  **Acceptance Criteria**:
  - [ ] `code/features/adversarial_validation.py` 实现
  - [ ] `python code/features/adversarial_validation.py` 跑通
  - [ ] 报告含 AUC + Top-10 特征
  - [ ] 若 AUC > 0.6, 列建议

  **QA Scenarios**:
  ```
  Scenario: 对抗验证跑通
    Tool: Bash
    Preconditions: T6 完成
    Steps:
      1. python code/features/adversarial_validation.py 2>&1 | tee .sisyphus/evidence/task-11-adv.txt
      2. grep "AUC" .sisyphus/evidence/task-11-adv.txt  (期望: 含数字)
    Expected Result: AUC 数字打印
    Evidence: .sisyphus/evidence/task-11-adv.txt
  ```

  **Commit**: YES
  - Message: `feat(eda): add adversarial validation for distribution shift`
  - Files: `code/features/adversarial_validation.py`, `docs/changelog/adversarial-validation.md`

---

- [ ] 12. **Stage 1 训练 (LGB with stats features)**

  **What to do**:
  - 创建 `code/models/train_stage1.py`:
    - 加载 T8/T9 统计特征 + T10 K-Fold TE
    - 拼接成特征矩阵 (dense)
    - 5-fold CV 训练 LightGBM
    - 输出 OOF RMSE, submission-stage1.csv
  - 写 `docs/changelog/stage-1-stats.md`:
    - 对比 stage 0 (TF-IDF) vs stage 1 (stats only) vs stage 1+TFIDF (组合)
    - 3 个子实验 RMSE 对比表
  - 更新 metrics.json

  **Must NOT do**:
  - 不加 DeBERTa (留给 Stage 2)
  - 不加图特征 (留给 Stage 3)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — 端到端
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (Wave 2 收尾)
  - **Parallel Group**: Sequential (depends T8-T11)
  - **Blocks**: T17
  - **Blocked By**: T7, T8, T9, T10

  **References**:
  - DeepSeek 集成策略: `docs/designs/deepseek_design.md:138-148`
  - GLM 报告时间指标: `docs/designs/GLM_design.md:62-90`

  **Acceptance Criteria**:
  - [ ] OOF RMSE 比 stage 0 降低 (预期: stage 1 (stats only) ≤ stage 0)
  - [ ] 提交 CSV 10001 行
  - [ ] changelog 含对比表

  **QA Scenarios**:
  ```
  Scenario: Stage 1 OOF 优于 baseline
    Tool: Bash
    Preconditions: T7, T8-T10 完成
    Steps:
      1. python code/models/train_stage1.py 2>&1 | tee .sisyphus/evidence/task-12-stage1.txt
      2. grep "OOF RMSE" .sisyphus/evidence/task-12-stage1.txt
    Expected Result: OOF RMSE 数字 < Stage 0 数字
    Evidence: .sisyphus/evidence/task-12-stage1.txt
  ```

  **Commit**: YES
  - Message: `feat(stage-1): LGB with statistical features, second submission`
  - Files: `code/models/train_stage1.py`, `kaggle/submission-stage1.csv`, `docs/changelog/stage-1-stats.md`, `docs/changelog/metrics.json`

---

- [ ] 13. **DeBERTa-v3 文本嵌入提取**

  **What to do**:
  - 创建 `code/features/text_bert.py`:
    - `load_deberta(model_name="microsoft/deberta-v3-base")` → tokenizer + model
    - `extract_embeddings(texts: list[str], batch_size=64, max_len=128)` → numpy array (N, 768)
    - 输入格式: `text = title + " " + comment`
  - 创建 `code/features/run_bert_distributed.py`:
    - 用 PySpark `mapInPandas` 或 `pandas_udf` 分布式跑 embedding
    - 训练集: 全量 3M 评论, 批大小 64, max_len 128
    - 输出: `artifacts/features/bert_train.parquet` (id, emb_0..emb_767)
    - 测试集: 10K 评论
    - 输出: `artifacts/features/bert_test.parquet`
  - 计时并写入 metrics.json
  - 报告: `docs/changelog/stage-2-bert.md` (训练时间, GPU 利用率, 嵌入示例)

  **Must NOT do**:
  - 不做 LoRA 微调 (时间紧, 风险高)
  - 不训练 DeBERTa (off-the-shelf)

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain` — 深度学习 + 分布式推理, 显存/批大小优化
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: T16, T17, T19
  - **Blocked By**: T6, T8

  **References**:
  - Qwen DeBERTa 配置: `docs/designs/qwen_design.md:88-93`
  - ChatGPT 768 维选择: `docs/designs/chatgpt_design.md:265-280`
  - HuggingFace DeBERTa-v3-base 文档 (通过 context7)

  **Acceptance Criteria**:
  - [ ] DeBERTa-v3-base 加载成功
  - [ ] 训练集 3M 行的 embedding parquet 生成
  - [ ] 测试集 10K 行的 embedding parquet 生成
  - [ ] 总耗时记录 (预期 1-3 小时, 视 GPU)
  - [ ] GPU 利用率 > 70% (nvidia-smi 采样)

  **QA Scenarios**:
  ```
  Scenario: DeBERTa 嵌入维度正确
    Tool: Bash
    Preconditions: T13 完成
    Steps:
      1. python -c "import pyarrow.parquet as pq; df=pq.read_table('artifacts/features/bert_train.parquet').to_pandas(); emb_cols=[c for c in df.columns if c.startswith('emb_')]; print(len(emb_cols))"
      2. 期望: 768
    Expected Result: 768 维
    Evidence: .sisyphus/evidence/task-13-dim.txt

  Scenario: 训练集嵌入行数 = 3M
    Tool: Bash
    Preconditions: T13 完成
    Steps:
      1. python -c "import pyarrow.parquet as pq; t=pq.read_table('artifacts/features/bert_train.parquet'); print(t.num_rows)"
      2. 期望: 3007439
    Expected Result: 行数一致
    Evidence: .sisyphus/evidence/task-13-rows.txt
  ```

  **Commit**: YES
  - Message: `feat(features): extract DeBERTa-v3 embeddings (768d) for train+test`
  - Files: `code/features/text_bert.py`, `code/features/run_bert_distributed.py`, `artifacts/features/bert_train.parquet`, `artifacts/features/bert_test.parquet`, `docs/changelog/stage-2-bert.md`, `docs/changelog/metrics.json`

---

- [ ] 14. **LightGCN 图嵌入训练**

  **What to do**:
  - 创建 `code/features/build_graph.py`:
    - `build_bipartite_graph(train_df)` → NetworkX DiGraph 或 scipy sparse matrix
    - 节点: user_id, prod_id
    - 边: 评论 (含 rating 作为边权重, 可选)
  - 创建 `code/features/lightgcn.py`:
    - LightGCN 模型 (基于 PyTorch Geometric 或 numpy 实现)
    - 输入: 邻接矩阵 (稀疏, N_users + N_items, N_users + N_items)
    - 3 层 LightGCN, 64 维输出
    - 训练: BPR loss 或回归 (rating 预测)
    - 优化器: Adam, lr=1e-3, 50 epochs
  - 创建 `code/features/run_lightgcn.py`:
    - 加载 train.csv, 构建图
    - 训练 LightGCN
    - 输出 user_emb.npy, item_emb.npy
  - 写报告: `docs/changelog/stage-3-lightgcn.md` (训练 loss 曲线, 嵌入可视化 t-SNE 可选)

  **Must NOT do**:
  - 不使用 GraphSAGE / GAT (LightGCN 专注协同过滤, 推荐系统 SOTA)
  - 不做图节点分类 (我们的任务是回归)

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain` — GNN 实现, 邻接矩阵优化
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (与 T13, T15 平行)
  - **Parallel Group**: Wave 3
  - **Blocks**: T16, T17
  - **Blocked By**: T6

  **References**:
  - Qwen LightGCN 设计: `docs/designs/qwen_design.md:94-96`
  - LightGCN 论文: He et al. 2020 (可查 arxiv)

  **Acceptance Criteria**:
  - [ ] LightGCN 训练跑通
  - [ ] user_emb (n_users, 64), item_emb (n_items, 64) 保存
  - [ ] 训练 loss 下降 (前 10 epoch)
  - [ ] 报告含 loss 曲线图

  **QA Scenarios**:
  ```
  Scenario: LightGCN 输出维度
    Tool: Bash
    Preconditions: T14 完成
    Steps:
      1. python -c "import numpy as np; u=np.load('artifacts/features/user_emb.npy'); i=np.load('artifacts/features/item_emb.npy'); print(u.shape, i.shape)"
      2. 期望: (n_users, 64) (n_items, 64)
    Expected Result: 维度 64
    Evidence: .sisyphus/evidence/task-14-dim.txt

  Scenario: 训练 loss 下降
    Tool: Bash
    Preconditions: T14 完成
    Steps:
      1. grep "loss" docs/changelog/stage-3-lightgcn.md | wc -l  (期望 ≥ 10)
    Expected Result: 10+ loss 记录
    Evidence: .sisyphus/evidence/task-14-loss.txt
  ```

  **Commit**: YES
  - Message: `feat(features): add LightGCN bipartite graph embeddings (64d)`
  - Files: `code/features/build_graph.py`, `code/features/lightgcn.py`, `code/features/run_lightgcn.py`, `artifacts/features/user_emb.npy`, `artifacts/features/item_emb.npy`, `docs/changelog/stage-3-lightgcn.md`

---

- [ ] 15. **备选 Node2Vec 嵌入 (Fallback)**

  **What to do**:
  - 创建 `code/features/node2vec.py`:
    - 用 gensim 或 PyTorch 实现 Node2Vec
    - 64 维, walk_length=10, num_walks=20
    - 输出 user_emb, item_emb
  - 仅在 T14 LightGCN 训练失败 / 时间超限时启用

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — 备选, 简单实现
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (与 T13, T14 同时进行, 但仅 T14 失败时使用)
  - **Parallel Group**: Wave 3 (optional)
  - **Blocks**: T16 (仅当启用时)
  - **Blocked By**: T6

  **References**:
  - ChatGPT Node2Vec 建议: `docs/designs/chatgpt_design.md:425-449`

  **Acceptance Criteria**:
  - [ ] 脚本可跑 (但默认不调用, 由 T14 决定)
  - [ ] 输出 (n_users, 64), (n_items, 64)

  **QA Scenarios**:
  ```
  Scenario: Node2Vec 脚本可跑
    Tool: Bash
    Preconditions: T15 完成
    Steps:
      1. python -c "from code.features.node2vec import compute_node2vec; print('importable')"
    Expected Result: 打印 importable
    Evidence: .sisyphus/evidence/task-15-import.txt
  ```

  **Commit**: YES
  - Message: `feat(features): add Node2Vec fallback for graph embeddings`
  - Files: `code/features/node2vec.py`

---

- [ ] 16. **多模态特征拼接 + 标准化**

  **What to do**:
  - 创建 `code/features/assemble.py`:
    - `assemble_features(df_train, df_test)`:
      - 拼接: 统计特征 (T8, T9) + K-Fold TE (T10) + DeBERTa (T13) + LightGCN (T14)
      - 标准化: StandardScaler on dense, sparse 保持
      - 输出: `artifacts/features/X_train.parquet`, `X_test.parquet`
    - 处理 T11 提出的分布差异特征 (如建议剔除, 则排除)
  - 创建 `code/tests/test_assemble.py`: 验证拼接后 shape, 无 NaN

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — 多种特征类型拼接, 内存管理
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (依赖 T13+T14)
  - **Parallel Group**: Wave 3 (after T13/T14)
  - **Blocks**: T17
  - **Blocked By**: T13, T14, T11

  **References**:
  - ChatGPT 特征融合: `docs/designs/chatgpt_design.md:200-220`
  - Qwen 多模态特征: `docs/designs/qwen_design.md:40-44`

  **Acceptance Criteria**:
  - [ ] 拼接后 X_train (3007439, ~900+), X_test (10000, ~900+)
  - [ ] 无 NaN
  - [ ] `pytest code/tests/test_assemble.py -v` PASS

  **QA Scenarios**:
  ```
  Scenario: 拼接后 shape
    Tool: Bash
    Preconditions: T13, T14 完成
    Steps:
      1. python -c "import pyarrow.parquet as pq; t=pq.read_table('artifacts/features/X_train.parquet'); print(t.num_rows, t.num_columns)"
    Expected Result: 3007439 行, 900+ 列
    Evidence: .sisyphus/evidence/task-16-shape.txt
  ```

  **Commit**: YES
  - Message: `feat(features): assemble multi-modal feature matrix (text+stats+graph)`
  - Files: `code/features/assemble.py`, `code/tests/test_assemble.py`, `artifacts/features/X_train.parquet`, `artifacts/features/X_test.parquet`

---

- [ ] 17. **Stage 2-3 训练 (LGB with all features)**

  **What to do**:
  - 创建 `code/models/train_stage2.py`:
    - 加载 T16 拼接特征
    - 5-fold CV 训练 LightGBM
    - 输出 OOF RMSE, submission-stage2.csv
  - 写 `docs/changelog/stage-2-multimodal.md`:
    - 对比 stage 0 / 1 / 2
    - 记录训练时间, 推理时间
    - 输出 Feature Importance Top-20
  - 更新 metrics.json

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — 端到端
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (Wave 3 收尾)
  - **Parallel Group**: Sequential
  - **Blocks**: T18, T19, T20, T21
  - **Blocked By**: T12, T16

  **References**:
  - Qwen LightGBM 表格建模: `docs/designs/qwen_design.md:45-49`

  **Acceptance Criteria**:
  - [ ] OOF RMSE 显著低于 stage 1 (预期 ≤ 0.95)
  - [ ] Feature Importance 图生成
  - [ ] 提交 CSV 10001 行
  - [ ] changelog 完整

  **QA Scenarios**:
  ```
  Scenario: Stage 2 OOF 显著提升
    Tool: Bash
    Preconditions: T12, T16 完成
    Steps:
      1. python code/models/train_stage2.py 2>&1 | tee .sisyphus/evidence/task-17-stage2.txt
      2. grep "OOF RMSE" .sisyphus/evidence/task-17-stage2.txt
    Expected Result: OOF RMSE 数字 < Stage 1
    Evidence: .sisyphus/evidence/task-17-stage2.txt
  ```

  **Commit**: YES
  - Message: `feat(stage-2): LGB with all features (text+stats+graph)`
  - Files: `code/models/train_stage2.py`, `kaggle/submission-stage2.csv`, `docs/changelog/stage-2-multimodal.md`, `docs/changelog/metrics.json`

---

- [ ] 18. **CatBoost 基模训练**

  **What to do**:
  - 创建 `code/models/catboost_train.py`:
    - 5-fold CV 训练 CatBoostRegressor
    - 处理类别特征 (main_category, store, parent_prod_id)
    - 输出 OOF predictions, model artifacts
  - 创建 `code/models/run_catboost.py`: 串起来
  - 输出: `artifacts/models/catboost_oof.npy`, `artifacts/models/catboost_test.npy`

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — CatBoost 调参, 类别特征处理
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: T20
  - **Blocked By**: T16

  **References**:
  - Qwen CatBoost 选型: `docs/designs/qwen_design.md:13-15`

  **Acceptance Criteria**:
  - [ ] CatBoost OOF 预测生成 (n=3M)
  - [ ] CatBoost test 预测生成 (n=10K)
  - [ ] 5-fold OOF RMSE < stage 2 (LGB)

  **QA Scenarios**:
  ```
  Scenario: CatBoost 预测 shape
    Tool: Bash
    Preconditions: T18 完成
    Steps:
      1. python -c "import numpy as np; oof=np.load('artifacts/models/catboost_oof.npy'); print(oof.shape)"
      2. 期望: (3007439,)
    Expected Result: shape 正确
    Evidence: .sisyphus/evidence/task-18-shape.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add CatBoost base model for stacking`
  - Files: `code/models/catboost_train.py`, `code/models/run_catboost.py`, `artifacts/models/catboost_oof.npy`, `artifacts/models/catboost_test.npy`

---

- [ ] 19. **MLP 基模训练 (on embeddings)**

  **What to do**:
  - 创建 `code/models/mlp.py`:
    - MLP 架构: Input(832) → Linear(512) → ReLU → Dropout(0.3) → Linear(128) → ReLU → Dropout → Linear(1)
    - 输入: 拼接 DeBERTa (768) + LightGCN user/item (64+64) + 统计特征 (~64)
    - 损失: MSE
    - 优化器: Adam, lr=1e-3, weight_decay=1e-5
    - 训练: 5-fold, 30 epochs, early stopping
  - 创建 `code/models/run_mlp.py`:
    - 加载 embeddings, 拼接, 训练
    - 输出 MLP OOF, MLP test predictions
  - 写训练日志到 `docs/changelog/mlp-training.md` (含 loss 曲线)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — PyTorch MLP, embedding 拼接
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: T20
  - **Blocked By**: T13, T14

  **References**:
  - Qwen MLP 基模: `docs/designs/qwen_design.md:48-49`

  **Acceptance Criteria**:
  - [ ] MLP OOF 生成
  - [ ] MLP test 生成
  - [ ] 训练 loss 下降, 验证 loss 不爆
  - [ ] 5-fold OOF RMSE < stage 2

  **QA Scenarios**:
  ```
  Scenario: MLP 预测 shape
    Tool: Bash
    Preconditions: T19 完成
    Steps:
      1. python -c "import numpy as np; oof=np.load('artifacts/models/mlp_oof.npy'); print(oof.shape)"
      2. 期望: (3007439,)
    Expected Result: shape 正确
    Evidence: .sisyphus/evidence/task-19-shape.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add MLP base model on embeddings for stacking`
  - Files: `code/models/mlp.py`, `code/models/run_mlp.py`, `artifacts/models/mlp_oof.npy`, `artifacts/models/mlp_test.npy`, `docs/changelog/mlp-training.md`

---

- [ ] 20. **Stacking 融合 (Ridge Meta-learner)**

  **What to do**:
  - 创建 `code/models/stacking.py`:
    - 输入: LGB OOF (T17) + CatBoost OOF (T18) + MLP OOF (T19)
    - 训练 Ridge Regression 作为 meta-learner
    - 5-fold CV 验证
    - 输出 stacking OOF, stacking test predictions
  - 创建 `code/models/run_stacking.py`:
    - 跑完整 stacking
    - 报告各基模权重 (Ridge 系数)
  - 写 `docs/changelog/stage-4-stacking.md`:
    - 4 个模型 (LGB / CatBoost / MLP / Stacking) OOF RMSE 对比
    - Stacking 系数解释
    - 时间指标

  **Must NOT do**:
  - 不用复杂神经网络做 meta-learner (Qwen 建议, 防过拟合)
  - 不加额外特征 (只 3 个基模预测)

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain` — 集成学习, 权重平衡
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (依赖 T18, T19)
  - **Parallel Group**: Wave 4 (after T18-T19)
  - **Blocks**: T23
  - **Blocked By**: T17, T18, T19

  **References**:
  - Qwen Stacking 设计: `docs/designs/qwen_design.md:100-105`
  - ChatGPT Stacking 思路: `docs/designs/chatgpt_design.md:578-590`

  **Acceptance Criteria**:
  - [ ] Stacking OOF 生成
  - [ ] Stacking test 生成
  - [ ] OOF RMSE ≤ 最好基模 (LGB 或 CatBoost)
  - [ ] 系数可解释 (3 个值)

  **QA Scenarios**:
  ```
  Scenario: Stacking 不差于最佳基模
    Tool: Bash
    Preconditions: T17-T19 完成
    Steps:
      1. python code/models/run_stacking.py 2>&1 | tee .sisyphus/evidence/task-20-stacking.txt
      2. grep "OOF RMSE" .sisyphus/evidence/task-20-stacking.txt
    Expected Result: OOF RMSE 数字 < min(LGB, CatBoost, MLP) OOF
    Evidence: .sisyphus/evidence/task-20-stacking.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add Stacking with Ridge meta-learner`
  - Files: `code/models/stacking.py`, `code/models/run_stacking.py`, `artifacts/models/stacking_oof.npy`, `artifacts/models/stacking_test.npy`, `docs/changelog/stage-4-stacking.md`

---

- [ ] 21. **超参调优 (Optuna)**

  **What to do**:
  - 创建 `code/models/optuna_tune.py`:
    - 优化目标: LightGBM OOF RMSE (5-fold)
    - 搜索空间:
      - num_leaves: [31, 63, 127, 255]
      - max_depth: [6, 8, 10, 12]
      - learning_rate: [0.01, 0.05, 0.1]
      - min_child_samples: [10, 20, 50]
      - feature_fraction: [0.6, 0.8, 1.0]
      - bagging_fraction: [0.6, 0.8, 1.0]
    - 50 trials, 5-fold CV
  - 输出: 最佳超参 `artifacts/models/best_params.json`
  - 写 MLflow 记录 (本地) 到 `mlruns/`
  - 报告: `docs/changelog/optuna-tuning.md`

  **Recommended Agent Profile**:
  - **Category**: `deep` — 贝叶斯优化, 多 trial 调度
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: T23
  - **Blocked By**: T17

  **References**:
  - Qwen MLOps 思路: `docs/designs/qwen_design.md:18-19`
  - Optuna 文档 (内部已知)

  **Acceptance Criteria**:
  - [ ] 50 trials 跑完
  - [ ] 最佳超参文件生成
  - [ ] 最佳 OOF RMSE < 默认超参 OOF RMSE
  - [ ] MLflow UI 可启动 (`mlflow ui`)

  **QA Scenarios**:
  ```
  Scenario: Optuna 找到更优超参
    Tool: Bash
    Preconditions: T21 完成
    Steps:
      1. cat artifacts/models/best_params.json
      2. python -c "import json; d=json.load(open('artifacts/models/best_params.json')); print(d)"
    Expected Result: JSON 含 num_leaves, max_depth 等
    Evidence: .sisyphus/evidence/task-21-params.txt
  ```

  **Commit**: YES
  - Message: `feat(models): Optuna hyperparameter tuning for LightGBM`
  - Files: `code/models/optuna_tune.py`, `artifacts/models/best_params.json`, `docs/changelog/optuna-tuning.md`

---

- [ ] 22. **阈值舍入 + Clip [1, 5]**

  **What to do**:
  - 创建 `code/models/postprocess.py`:
    - `optimal_round(predictions, y_true, granularity=0.5)`:
      - 网格搜索最优舍入阈值 (0.5, 1.5, 2.5, 3.5, 4.5)
      - 比较 round vs floor vs ceil
      - 选择 OOF RMSE 最低的方案
    - `clip_15(predictions)` → clip to [1, 5]
  - 创建 `code/models/run_postprocess.py`:
    - 应用到 stacking 预测
    - 输出 `artifacts/models/final_predictions.npy`
  - 写 `docs/changelog/postprocessing.md`:
    - 不同舍入策略 RMSE 对比
    - Clip 必要性分析

  **Recommended Agent Profile**:
  - **Category**: `quick` — 简单后处理
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: T23
  - **Blocked By**: T20 (need predictions)

  **References**:
  - GLM 设计的 clip: `docs/designs/GLM_design.md:34-35`
  - Qwen 阈值舍入: `docs/designs/qwen_design.md:56-57`

  **Acceptance Criteria**:
  - [ ] optimal_round 找到最优策略
  - [ ] clip 后预测在 [1, 5]
  - [ ] 应用后 OOF RMSE 进一步降低

  **QA Scenarios**:
  ```
  Scenario: Clip 后范围正确
    Tool: Bash
    Preconditions: T22 完成
    Steps:
      1. python -c "import numpy as np; p=np.load('artifacts/models/final_predictions.npy'); assert p.min() >= 1 and p.max() <= 5; print('OK', p.min(), p.max())"
    Expected Result: 打印 OK + 范围
    Evidence: .sisyphus/evidence/task-22-clip.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add post-processing (clip, optimal rounding)`
  - Files: `code/models/postprocess.py`, `code/models/run_postprocess.py`, `artifacts/models/final_predictions.npy`, `docs/changelog/postprocessing.md`

---

- [ ] 23. **Stage 4-5 最终 Pipeline + 提交**

  **What to do**:
  - 创建 `code/run.sh` 完整版 (填充 T1 stub):
    ```bash
    #!/bin/bash
    set -e
    python code/etl/run_etl.py
    python code/features/run_stats.py
    python code/features/run_bert_distributed.py
    python code/features/run_lightgcn.py
    python code/features/assemble.py
    python code/models/run_catboost.py
    python code/models/run_mlp.py
    python code/models/train_stage2.py  # 用 best params
    python code/models/run_stacking.py
    python code/models/run_postprocess.py
    python code/models/predict.py  # 生成最终 submission
    ```
  - 跑完整 pipeline
  - 生成 `kaggle/submission-final.csv`
  - 写 `docs/changelog/stage-5-final.md`:
    - 5 阶段对比 (stage 0, 1, 2, 3, 4)
    - 完整 metrics 表格
    - 训练时间, 推理时间, 总离线时间
  - 更新 metrics.json

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high` — 端到端集成
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (Wave 4 收尾)
  - **Parallel Group**: Sequential
  - **Blocks**: T24-T29, T30-T34
  - **Blocked By**: T20, T21, T22

  **References**:
  - DeepSeek run.sh: `docs/designs/deepseek_design.md:200-211`

  **Acceptance Criteria**:
  - [ ] `bash code/run.sh` 退出码 0
  - [ ] `kaggle/submission-final.csv` 生成 10001 行
  - [ ] 最终 OOF RMSE 记录
  - [ ] 三段时间指标 (per-epoch, offline, inference) 记录
  - [ ] metrics.json 完整更新

  **QA Scenarios**:
  ```
  Scenario: 端到端 pipeline 跑通
    Tool: Bash
    Preconditions: T20-T22 完成
    Steps:
      1. bash code/run.sh 2>&1 | tee .sisyphus/evidence/task-23-pipeline.txt
      2. 验证退出码 0
      3. head -1 kaggle/submission-final.csv  (期望 id,rating)
      4. wc -l kaggle/submission-final.csv  (期望 10001)
    Expected Result: 完整跑通, submission 正确
    Evidence: .sisyphus/evidence/task-23-pipeline.txt

  Scenario: 最终 OOF RMSE 合理
    Tool: Bash
    Preconditions: T23 完成
    Steps:
      1. python -c "import json; d=json.load(open('docs/changelog/metrics.json')); print(d['stages'])"
    Expected Result: stages 含 5 项, 每项有 rmse
    Evidence: .sisyphus/evidence/task-23-metrics.txt
  ```

  **Commit**: YES
  - Message: `feat(stage-5): end-to-end pipeline + final submission`
  - Files: `code/run.sh` (filled), `kaggle/submission-final.csv`, `docs/changelog/stage-5-final.md`, `docs/changelog/metrics.json`

---

- [ ] 24. **Ablation A — 无文本特征 (仅 stats)**

  **What to do**:
  - 创建 `code/ablation/run_ablation_a.py`:
    - 复用 T16 拼接, 但剔除 DeBERTa 768 维
    - 跑完整 stacking pipeline
    - 记录 OOF RMSE
  - 输出: `docs/changelog/ablation-a-no-text.md`
  - 更新 metrics.json 的 ablations 字段

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5 (with T25-T29)
  - **Blocks**: T30
  - **Blocked By**: T23

  **Acceptance Criteria**:
  - [ ] OOF RMSE > 完整版 (说明文本有贡献)
  - [ ] 报告含 delta_rmse

  **QA Scenarios**:
  ```
  Scenario: 无文本 RMSE 显著变差
    Tool: Bash
    Preconditions: T23 完成
    Steps:
      1. python code/ablation/run_ablation_a.py 2>&1 | tee .sisyphus/evidence/task-24-abl-a.txt
      2. grep "OOF RMSE" .sisyphus/evidence/task-24-abl-a.txt
    Expected Result: RMSE > 完整版
    Evidence: .sisyphus/evidence/task-24-abl-a.txt
  ```

  **Commit**: YES
  - Message: `experiment(ablation): remove text features (DeBERTa)`
  - Files: `code/ablation/run_ablation_a.py`, `docs/changelog/ablation-a-no-text.md`, `docs/changelog/metrics.json`

---

- [ ] 25. **Ablation B — 无图特征 (text + stats only)**

  **What to do**:
  - 复用 T16 拼接, 剔除 LightGCN user_emb + item_emb (128 维)
  - 跑完整 stacking
  - 报告 `docs/changelog/ablation-b-no-graph.md`

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5
  - **Blocks**: T30
  - **Blocked By**: T23

  **Acceptance Criteria**:
  - [ ] OOF RMSE 上升
  - [ ] 报告含 delta_rmse

  **QA Scenarios**:
  ```
  Scenario: 无图 RMSE 变差
    Tool: Bash
    Preconditions: T23 完成
    Steps:
      1. python code/ablation/run_ablation_b.py 2>&1 | tee .sisyphus/evidence/task-25-abl-b.txt
      2. grep "OOF RMSE" .sisyphus/evidence/task-25-abl-b.txt
    Expected Result: RMSE > 完整版
    Evidence: .sisyphus/evidence/task-25-abl-b.txt
  ```

  **Commit**: YES
  - Message: `experiment(ablation): remove graph features (LightGCN)`
  - Files: `code/ablation/run_ablation_b.py`, `docs/changelog/ablation-b-no-graph.md`, `docs/changelog/metrics.json`

---

- [ ] 26. **Ablation C — 无 Stacking (单 LGB)**

  **What to do**:
  - 跑单 LightGBM (T17) 即可, 不做 Stacking
  - 报告 `docs/changelog/ablation-c-no-stacking.md`

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5
  - **Blocks**: T30
  - **Blocked By**: T23

  **Acceptance Criteria**:
  - [ ] OOF RMSE > Stacking 版本
  - [ ] 报告含 delta_rmse

  **QA Scenarios**:
  ```
  Scenario: 无 Stacking 变差
    Tool: Bash
    Preconditions: T23 完成
    Steps:
      1. python code/ablation/run_ablation_c.py 2>&1 | tee .sisyphus/evidence/task-26-abl-c.txt
      2. grep "OOF RMSE" .sisyphus/evidence/task-26-abl-c.txt
    Expected Result: RMSE > Stacking 版本
    Evidence: .sisyphus/evidence/task-26-abl-c.txt
  ```

  **Commit**: YES
  - Message: `experiment(ablation): remove Stacking (use single LGB)`
  - Files: `code/ablation/run_ablation_c.py`, `docs/changelog/ablation-c-no-stacking.md`, `docs/changelog/metrics.json`

---

- [ ] 27. **Ablation D — 无 K-Fold TE (用 naive mean)**

  **What to do**:
  - 跑 naive target encoding (直接用 train 全集均值, 不分折) → 已知会泄漏
  - 但 OOF RMSE 反而"虚低" (因泄漏)
  - 报告 `docs/changelog/ablation-d-no-kfold-te.md`:
    - 展示 naive TE 的 OOF RMSE (可能更"好", 但不可信)
    - 强调为什么必须 K-Fold (无泄漏)

  **Recommended Agent Profile**:
  - **Category**: `deep` — 防泄漏论证
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5
  - **Blocks**: T30
  - **Blocked By**: T23

  **Acceptance Criteria**:
  - [ ] 报告含 naive vs K-Fold 对比
  - [ ] 强调"虚低"问题

  **QA Scenarios**:
  ```
  Scenario: naive TE 演示
    Tool: Bash
    Preconditions: T23 完成
    Steps:
      1. python code/ablation/run_ablation_d.py 2>&1 | tee .sisyphus/evidence/task-27-abl-d.txt
      2. grep -E "naive|kfold" .sisyphus/evidence/task-27-abl-d.txt | wc -l  (期望 ≥ 2)
    Expected Result: 含两种结果
    Evidence: .sisyphus/evidence/task-27-abl-d.txt
  ```

  **Commit**: YES
  - Message: `experiment(ablation): naive target encoding (shows leakage risk)`
  - Files: `code/ablation/run_ablation_d.py`, `docs/changelog/ablation-d-no-kfold-te.md`, `docs/changelog/metrics.json`

---

- [ ] 28. **Ablation E — 无 DeBERTa (用 TF-IDF 替代)**

  **What to do**:
  - 用 TF-IDF (T7 用的 5000 维) 替代 DeBERTa 768 维
  - 跑完整 stacking
  - 报告 `docs/changelog/ablation-e-tfidf-vs-bert.md`:
    - 对比 DeBERTa vs TF-IDF 的贡献
    - 强调语义能力差异

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5
  - **Blocks**: T30
  - **Blocked By**: T23

  **Acceptance Criteria**:
  - [ ] OOF RMSE 上升 (但比无文本好)
  - [ ] 报告含 DeBERTa vs TF-IDF 数值对比

  **QA Scenarios**:
  ```
  Scenario: TF-IDF 替代 DeBERTa
    Tool: Bash
    Preconditions: T23 完成
    Steps:
      1. python code/ablation/run_ablation_e.py 2>&1 | tee .sisyphus/evidence/task-28-abl-e.txt
      2. grep "OOF RMSE" .sisyphus/evidence/task-28-abl-e.txt
    Expected Result: RMSE 在 (无文本, 完整) 之间
    Evidence: .sisyphus/evidence/task-28-abl-e.txt
  ```

  **Commit**: YES
  - Message: `experiment(ablation): replace DeBERTa with TF-IDF`
  - Files: `code/ablation/run_ablation_e.py`, `docs/changelog/ablation-e-tfidf-vs-bert.md`, `docs/changelog/metrics.json`

---

- [ ] 29. **Ablation F — 无 Clip (看 RMSE 分布)**

  **What to do**:
  - 跑 stacking 预测, 不 clip
  - 检查预测分布 (min, max, mean, std)
  - 报告 `docs/changelog/ablation-f-no-clip.md`:
    - 预测范围 (如 [0.3, 5.7] 表明有溢出)
    - 与 Clip 后 RMSE 对比
    - 强调 RMSE 对极端值敏感

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5
  - **Blocks**: T30
  - **Blocked By**: T23

  **Acceptance Criteria**:
  - [ ] 预测分布记录
  - [ ] 报告含 clip vs no-clip RMSE

  **QA Scenarios**:
  ```
  Scenario: 无 Clip 演示
    Tool: Bash
    Preconditions: T23 完成
    Steps:
      1. python code/ablation/run_ablation_f.py 2>&1 | tee .sisyphus/evidence/task-29-abl-f.txt
      2. grep -E "min|max|clip" .sisyphus/evidence/task-29-abl-f.txt | wc -l  (期望 ≥ 3)
    Expected Result: 含 min/max/clip 描述
    Evidence: .sisyphus/evidence/task-29-abl-f.txt
  ```

  **Commit**: YES
  - Message: `experiment(ablation): remove Clip [1,5] post-processing`
  - Files: `code/ablation/run_ablation_f.py`, `docs/changelog/ablation-f-no-clip.md`, `docs/changelog/metrics.json`

---

- [ ] 30. **网页内容填充**

  **What to do**:
  - 更新 `website/data/metrics.json`: 合并所有 stage + ablation 数据
  - 完善 `website/index.html` 4 个 section:
    - **技术路线**: 7 阶段时间线, mermaid 流程图
    - **系统性能**: 表格 + Chart.js 4 图 (RMSE 趋势 / 训练时间 / 推理时间 / 消融对比)
    - **亮点**: 5-7 条 bullet (DeBERTa, LightGCN, K-Fold TE, Stacking, Adversarial, 分布式)
    - **缺点**: 3-5 条 bullet (时间紧, 单卡限制, 没用 ONNX, LightGCN 简化等)
  - 完善 `website/app.js`: 从 metrics.json 加载并渲染
  - 加 `website/figures/`: 4-6 张关键 PNG (从 changelog 复制)
  - 更新 `website/README.md`: 含部署步骤

  **Must NOT do**:
  - 不引入 JS 框架
  - 不用 Tailwind / 任何 build step

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 6
  - **Blocks**: T31, T32
  - **Blocked By**: T5, T23-T29

  **Acceptance Criteria**:
  - [ ] 4 个 section 都有真实内容
  - [ ] Chart.js 4 图正确渲染
  - [ ] Playwright 截图验证视觉

  **QA Scenarios**:
  ```
  Scenario: 网页内容完整
    Tool: Playwright
    Preconditions: T30 完成, server 启动
    Steps:
      1. mcp_playwright: navigate to http://localhost:8000/
      2. assert 每 section 有 ≥ 1 个内容元素
      3. assert 至少 4 个 canvas (Chart.js) 渲染
      4. screenshot 整页
    Expected Result: 4 sections + 4 charts 渲染
    Evidence: .sisyphus/evidence/task-30-web-screenshot.png

  Scenario: 移动端响应式
    Tool: Playwright
    Preconditions: T30 完成
    Steps:
      1. setViewportSize 375x667  (iPhone SE)
      2. screenshot
    Expected Result: 不破版
    Evidence: .sisyphus/evidence/task-30-mobile.png
  ```

  **Commit**: YES
  - Message: `feat(website): populate content with metrics and visualizations`
  - Files: `website/index.html`, `website/app.js`, `website/data/metrics.json`, `website/figures/`, `website/README.md`

---

- [ ] 31. **报告 PDF 生成**

  **What to do**:
  - 选择工具: `pandoc` (markdown → PDF) 或 `weasyprint` (HTML → PDF) — 推荐 pandoc
  - 创建 `report/report.md`:
    - **Chapter 1 Problem Definition** (1-2 页)
    - **Chapter 2 Data Analysis** (3-4 页, 含 EDA 图)
    - **Chapter 3 Solution & Implementation** (8-10 页, 5-step 流程 + 架构图 + 各模块)
    - **Chapter 4 Performance Evaluation** (6-8 页, 含三段时间指标 + 5 阶段对比表 + 6 消融实验表)
    - **Chapter 5 Summary & Future Work** (2-3 页)
    - **References** (1 页)
    - **Contribution Table** (1 页)
  - 转换为 `report/report.pdf`
  - 准备 `report/figures/`: 至少 10 张 (架构图 / 流程图 / RMSE 趋势 / 消融表 / 时间分布 等)

  **Must NOT do**:
  - 报告不写无关内容 (如自我评价)
  - 不堆字数 (A+ 重视"concrete facts/evidence")

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 6
  - **Blocks**: T34
  - **Blocked By**: T23-T29, T30

  **Acceptance Criteria**:
  - [ ] `report/report.pdf` 存在, 20-30 页
  - [ ] 含 5 阶段对比表
  - [ ] 含 6 消融实验表
  - [ ] 含三段时间指标 (per-epoch / offline / inference)
  - [ ] 含贡献表

  **QA Scenarios**:
  ```
  Scenario: 报告含关键章节
    Tool: Bash
    Preconditions: T31 完成
    Steps:
      1. ls report/report.pdf  (期望存在)
      2. pdftotext report/report.pdf - 2>/dev/null | grep -E "Ablation|Training Time|Inference|Offline" | wc -l  (期望 ≥ 4)
    Expected Result: PDF 含 4+ 关键术语
    Evidence: .sisyphus/evidence/task-31-pdf-check.txt
  ```

  **Commit**: YES
  - Message: `docs(report): generate final report PDF`
  - Files: `report/report.md`, `report/report.pdf`, `report/figures/`

---

- [ ] 32. **Slides 生成 (5-10 张)**

  **What to do**:
  - 创建 `slides/slides.md` 或 `slides/slides.html` (revealjs 风格)
  - 5-10 张 slides:
    1. Title (团队名, 题目)
    2. Problem & Data
    3. Architecture (大图)
    4. Spark Distributed Design
    5. Multi-modal Features
    6. Stacking Ensemble
    7. Results (RMSE 对比)
    8. Ablation Study
    9. Conclusion + Future Work
    10. Q&A
  - 转换 `slides/slides.pdf`

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 6
  - **Blocks**: T34
  - **Blocked By**: T23-T29, T30

  **Acceptance Criteria**:
  - [ ] `slides/slides.pdf` 存在, 5-10 页
  - [ ] 含架构图, 结果图, 消融表

  **QA Scenarios**:
  ```
  Scenario: Slides 页数
    Tool: Bash
    Preconditions: T32 完成
    Steps:
      1. pdfinfo slides/slides.pdf 2>/dev/null | grep "Pages"  (期望 5-10)
    Expected Result: 5-10 页
    Evidence: .sisyphus/evidence/task-32-slides.txt
  ```

  **Commit**: YES
  - Message: `docs(slides): generate presentation slides (5-10 pages)`
  - Files: `slides/slides.md`, `slides/slides.pdf`, `slides/figures/`

---

- [ ] 33. **README 完善 + 复现指南**

  **What to do**:
  - 完善 `code/README.md`:
    - 项目简介 (3-5 句)
    - 硬件要求 (单卡 GPU, 16GB+ RAM)
    - 依赖安装 (`pip install -r requirements.txt`)
    - 数据准备 (放在 data/ 下)
    - 复现步骤 (按 stage 0-5 顺序)
    - 跑消融 (`python code/ablation/run_ablation_*.py`)
    - 查看网页 (`cd website && python -m http.server`)
    - 跑测试 (`pytest code/tests/`)
  - 完善顶层 `README.md`:
    - 团队名, 成员, 学号
    - 项目入口
    - 链接: code/, website/, report.pdf, slides.pdf
  - 创建 `code/tests/test_pipeline.py`: 端到端 smoke test (100 行子集)

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 6
  - **Blocks**: T34
  - **Blocked By**: T1, T23

  **Acceptance Criteria**:
  - [ ] code/README.md ≥ 80 行
  - [ ] 顶层 README.md ≥ 30 行
  - [ ] 复现步骤实际可跑 (smoke test 通过)

  **QA Scenarios**:
  ```
  Scenario: README 复现性
    Tool: Bash
    Preconditions: T33 完成
    Steps:
      1. wc -l code/README.md  (期望 ≥ 80)
      2. wc -l README.md  (期望 ≥ 30)
    Expected Result: 字数达标
    Evidence: .sisyphus/evidence/task-33-readme.txt
  ```

  **Commit**: YES
  - Message: `docs: finalize README with reproducibility guide`
  - Files: `code/README.md`, `README.md`, `code/tests/test_pipeline.py`

---

- [ ] 34. **TeamName.zip 打包 + 验证**

  **What to do**:
  - 创建打包脚本 `code/build_zip.sh`:
    ```bash
    #!/bin/bash
    TEAM="TeamName"  # 用户改
    rm -rf $TEAM && mkdir $TEAM
    cp -r code $TEAM/code
    cp report/report.pdf $TEAM/report.pdf
    cp slides/slides.pdf $TEAM/slides.pdf
    zip -r $TEAM.zip $TEAM
    ```
  - 跑打包, 生成 `TeamName.zip`
  - 验证 zip 内:
    - code/ 完整
    - report.pdf 存在
    - slides.pdf 存在

  **Must NOT do**:
  - 不打包 data/ (太大)
  - 不打包 artifacts/, mlruns/, __pycache__/

  **Recommended Agent Profile**:
  - **Category**: `quick` — 简单打包
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO (Wave 6 收尾)
  - **Parallel Group**: Sequential
  - **Blocks**: F1
  - **Blocked By**: T30-T33

  **Acceptance Criteria**:
  - [ ] `TeamName.zip` 生成
  - [ ] zip 内含 code/, report.pdf, slides.pdf
  - [ ] zip 大小合理 (< 200MB, 不含 data)

  **QA Scenarios**:
  ```
  Scenario: Zip 包结构
    Tool: Bash
    Preconditions: T34 完成
    Steps:
      1. unzip -l TeamName.zip | grep -E "code|report|slides" | wc -l  (期望 ≥ 3)
    Expected Result: 3+ 关键条目
    Evidence: .sisyphus/evidence/task-34-zip.txt
  ```

  **Commit**: YES
  - Message: `build: package final TeamName.zip`
  - Files: `code/build_zip.sh`, `TeamName.zip`

---

## Final Verification Wave (MANDATORY)

> 4 个评审 agent 并行运行, 必须全部 APPROVE 才算完成。
> 评审通过后, **必须等用户明确 OK 才能标记完成**。

- [ ] F1. **Plan Compliance Audit** — `oracle`
  端到端读 plan, 验证:
  - 每个 "Must Have" 在代码中能找到对应实现 (读文件 / curl / run command)
  - 每个 "Must NOT Have" 不在代码中 (搜索禁忌 pattern)
  - `.sisyphus/evidence/` 中每个任务的证据文件存在
  - 6 个 ablation 表格齐全
  - 7 阶段 changelog 齐全
  - 网页 4 大块 (技术路线 / 性能 / 亮点 / 缺点) 都有
  - 报告 / slides / README / zip 都齐
  **输出**: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  运行 `pytest code/tests/` (如果 T33 加了测试), `python -m py_compile code/**/*.py` 检查语法, 静态扫描:
  - `as any` / `@ts-ignore` 风格问题 (Python: `# noqa` 不当使用)
  - 死代码 / 未使用 import / print() 残留
  - AI-slop: 过度注释, 抽象层过深, 通用名 (data/result/temp)
  - README 复现步骤实际可跑
  **输出**: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high` (+ `playwright` skill)
  从干净环境跑:
  - `bash code/run.sh` 完整跑一次
  - 验证 submission.csv 格式 (id, rating, rating∈[1,5])
  - 打开 website/index.html 截图, 验证 4 大块内容
  - 读 1 篇报告 PDF, 验证含消融表格
  - 跑 1 次 ablation 实验验证脚本可用
  **输出**: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  对每个 T1-T34 读 "What to do" 和 git log diff:
  - 1:1 验证 (spec 内容都实现, 无遗漏)
  - 无范围蔓延 (没实现 spec 之外的功能)
  - 无 cross-task 污染 (T N 没动 T M 的文件)
  - 无未说明的临时文件
  **输出**: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- 每个 T1-T34 任务完成后, 一次 atomic commit
- 格式: `type(scope): short description`
  - T1: `chore: scaffold project structure`
  - T6: `feat(etl): add spark ETL pipeline`
  - T13: `feat(features): add DeBERTa-v3 embedding extraction`
  - T24-T29: `experiment(ablation): remove text features`
- 提交频率: 每任务一次, 不批量
- 关键 commit 必须包含 metrics 更新 (docs/changelog/metrics.json)

---

## Success Criteria

### 验证命令

```bash
# 1. 项目完整性
ls code/run.sh code/README.md website/index.html
# 期望: 全部存在

# 2. 端到端可跑
bash code/run.sh
# 期望: 退出码 0, 生成 kaggle/submission-final.csv

# 3. 提交格式
head -1 kaggle/submission-final.csv
# 期望: id,rating
wc -l kaggle/submission-final.csv
# 期望: 10001 (含 header)

# 4. 7 阶段 changelog
ls docs/changelog/stage-*.md | wc -l
# 期望: ≥ 7

# 5. 6 消融实验
ls docs/changelog/ablation-*.md | wc -l
# 期望: ≥ 6

# 6. 网页可访问 (本地)
python -m http.server 8000 --directory website/ &
curl -s http://localhost:8000/ | head -5
# 期望: HTML 内容返回

# 7. 单元测试
pytest code/tests/ -v
# 期望: 全部 PASS

# 8. 报告 + slides
test -f report.pdf -a -f slides.pdf
# 期望: 全部存在
```

### Final Checklist
- [ ] 所有 "Must Have" 在代码中可定位
- [ ] 所有 "Must NOT Have" 缺席
- [ ] Kaggle 至少 1 次最终提交, RMSE 记录在 metrics.json
- [ ] 7 阶段 changelog 全部填写, 含数值证据
- [ ] 6 个消融实验报告 + 对比表
- [ ] 网页 4 大块 (技术路线 / 性能 / 亮点 / 缺点) 完整
- [ ] 报告 PDF 含 3 大时间指标 + 消融 + 贡献表
- [ ] Slides 5-10 张
- [ ] README 复现步骤实际可跑
- [ ] TeamName.zip 包含 code/ + slides.pdf + report.pdf
- [ ] F1-F4 全部 APPROVE
- [ ] 用户最终明确 OK

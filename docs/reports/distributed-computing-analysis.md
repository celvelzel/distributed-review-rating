# 项目分布式计算元素分析

> 本文档梳理 COMP5434 Review Rating Prediction 项目中所有涉及分布式计算的技术组件，供项目报告撰写参考。

---

## 1. 整体架构

项目采用**混合并行架构**，针对不同计算密集型任务选用最适合的并行策略：

| 层次 | 技术 | 应用场景 |
|------|------|----------|
| 数据处理层 | PySpark (local[*]) | ETL、特征工程、TF-IDF、SVD |
| 进程并行层 | Python multiprocessing | 情感分析、文本统计 |
| GPU 并行层 | PyTorch + CUDA | Transformer 嵌入提取、深度模型微调 |
| HPC 后台执行层 | HPC (nohup 后台执行) | 大规模模型训练、Stacking 流程 |
| 存储层 | Apache Parquet | 列式分区存储，Spark/Pandas 双兼容 |

---

## 2. PySpark — 核心分布式计算框架

### 2.1 环境配置

- **版本**: PySpark 3.4.1
- **运行模式**: `local[*]`（伪分布式，利用本机全部 CPU 核心）
- **Driver 内存**: 4g–20g（按任务调整）
- **Shuffle 分区数**: 200
- **Broadcast 超时**: 600s

统一的 SparkSession 管理位于 `code/utils/spark_session.py`：

```python
_spark = (
    SparkSession.builder.appName(app_name)
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.broadcastTimeout", "600")
    .config("spark.pyspark.python", sys.executable)
    .config("spark.pyspark.driver.python", sys.executable)
    .getOrCreate()
)
```

**启动方式**：`code/run_spark.sh` 是统一的 spark-submit 启动器，用于执行 ETL 和特征工程脚本：

```bash
bash code/run_spark.sh python code/etl/run_etl.py
# 内部执行: spark-submit --master local[*] --driver-memory 4g code/etl/run_etl.py
```

### 2.2 ETL 数据管道

**入口**: `code/etl/run_etl.py`
**核心逻辑**: `code/etl/spark_etl.py`

ETL 管道使用 PySpark DataFrame API 处理约 3M 条训练数据，完整流程如下：

| 步骤 | 操作 | Spark 算子 |
|------|------|-----------|
| 1. 数据加载 | 读取 CSV，类型转换 | `spark.read.csv()`, `withColumn().cast()` |
| 2. 数据过滤 | 去除无效评分和空值 | `filter()`, `between()`, `isNotNull()` |
| 3. 文本清洗 | 去除特殊字符、规范化 | `regexp_replace()`, `lower()` |
| 4. 缺失值填补 | 填充空值 | `fillna()`, `when().otherwise()` |
| 5. 数据关联 | Join 产品信息 | `join(how="left")`（含 Broadcast Join） |
| 6. 时间特征 | 提取年月日等 | `from_unixtime()`, `year()`, `month()` 等 |
| 7. 持久化 | 输出 Parquet | `write.mode("overwrite").parquet()` |

**关键优化**：
- 全部列以 string 类型读入后再 cast，避免 schema 推断导致的多行评论溢出问题
- 产品信息表（213K 行）通过 Broadcast Join 广播到各 worker，避免 Shuffle

### 2.3 统计特征工程

统计特征通过 PySpark 的 `groupBy().agg()` 实现分布式聚合，处理 3M 行训练数据：

| 模块 | 文件 | 聚合维度 | 输出特征 |
|------|------|---------|---------|
| 用户统计 | `code/features/user_stats.py` | `user_id` | avg_rating, num_reviews, avg_votes, purchased_rate, rating_std |
| 产品统计 | `code/features/product_stats.py` | `parent_prod_id` | prod_avg_rating, prod_num_reviews, prod_price, prod_rating_number |
| 类别统计 | `code/features/category_stats.py` | `main_category` | cat_avg_rating, cat_avg_price, cat_rating_std |

示例代码（用户统计）：

```python
stats = (
    df.groupBy("user_id")
    .agg(
        F.avg("rating").alias("avg_rating"),
        F.count("*").alias("num_reviews"),
        F.avg("votes").alias("avg_votes"),
        F.avg(F.when(F.col("purchased") == "True", 1.0).otherwise(0.0)).alias("purchased_rate"),
        F.stddev_pop("rating").alias("rating_std"),
    )
)
```

### 2.4 行级特征提取

以下模块对每行数据独立计算，利用 PySpark 的分布式 DataFrame 实现并行：

| 模块 | 文件 | 提取内容 |
|------|------|---------|
| 时间特征 | `code/features/temporal.py` | year, month, day, weekday, hour, is_weekend, is_holiday_season |
| 文本长度 | `code/features/text_length.py` | title_len, comment_len, title_comment_ratio, has_caps, has_exclamation |
| 价格特征 | `code/features/price_features.py` | log_price, price_rank_in_category（Window 函数）, price_bucket |

### 2.5 TF-IDF — PySpark ML Pipeline

**文件**: `code/features/tfidf_50k.py`

使用 PySpark ML Pipeline 构建分布式 TF-IDF 特征提取：

```
RegexTokenizer → NGram(n=2,3) → 拼接 → HashingTF(50K) → IDF
```

**配置**:
- 最大特征维度: 50,000
- N-gram 范围: (1, 3)
- Driver 内存: 20g（高维稀疏矩阵需要更多内存）

### 2.6 SVD 降维 — PySpark MLlib

**文件**: `code/features/svd_features.py`

使用 PySpark MLlib 的 `TruncatedSVD` 对 TF-IDF 50K 特征进行分布式降维：

- 目标维度: 512 / 1024
- 输入: TF-IDF 稀疏矩阵
- 输出: SVD 降维后的密集特征

---

## 3. Python multiprocessing — 进程级并行

部分 CPU 密集型特征工程使用 Python `multiprocessing.Pool` 实现多进程并行：

| 模块 | 文件 | 并行数 | 处理内容 |
|------|------|--------|---------|
| 情感分析 | `code/features/sentiment.py` | min(cpu_count(), 8) | VADER compound/pos/neu/neg + TextBlob polarity/subjectivity + 正负词计数 |
| 文本统计 | `code/features/text_stats.py` | min(cpu_count(), 8) | 字符数、词数、标点比例、大写比例、数字比例 |

工作原理：将 3M 条数据分为 N 个 chunk，每个进程独立处理一个 chunk，最后合并结果。

---

## 4. GPU 并行 — 深度学习训练

### 4.1 嵌入提取

**文件**: `code/features/text_bert.py`

使用 DeBERTa-v3-base 预训练模型在 GPU 上提取文本嵌入：

- 模型: microsoft/deberta-v3-base (86M 参数)
- 输出维度: 768
- 推理方式: batch 推理 (batch_size=64), FP32
- 池化策略: Mean Pooling（对所有 token 嵌入取平均）

### 4.2 模型微调

| 脚本 | 模型 | 数据量 | GPU 优化技术 |
|------|------|--------|-------------|
| `transformer_e2e.py` | DeBERTa-v3-base | 3M | FP16 混合精度, 梯度累积 (GradAcc=21) |
| `deberta_lora.py` | DeBERTa-v3-base + LoRA | 3M | FP16, Gradient Checkpointing, 仅 ~0.5-3M 可训练参数 |
| `deberta_lora_3m_5f5e.py` | DeBERTa-v3-base + LoRA | 3M | 同上，5折×5轮配置 |
| `deberta_base_full.py` | DeBERTa-v3-base + LoRA | 3M | HPC GPU, 251GB RAM, R-Drop 正则化 |
| `deberta_large_full.py` | DeBERTa-v3-large + LoRA | 3M | HPC GPU, 大模型 (304M 参数) |
| `deberta_small_500k.py` | DeBERTa-v3-small + LoRA | 500K | 低显存优化 (15GB 限制) |
| `mlp.py` / `run_mlp.py` | MLP (768→512→256→128→1) | BERT 嵌入 | GPU 训练, BatchNorm |

**关键技术**：
- **LoRA (Low-Rank Adaptation)**: 冻结预训练权重，仅训练低秩适配器，大幅减少显存占用和训练时间
- **CORAL Ordinal Loss**: 将 5 级评分预测转化为 4 个二元累积分类任务
- **FP16 混合精度**: 使用 `torch.amp.GradScaler` 加速训练并减少显存
- **Gradient Checkpointing**: 用计算换内存，支持更大 batch size

> **注意**：项目所有 GPU 训练均为单卡单设备执行，未使用 DDP、DataParallel、DeepSpeed 或 accelerate 等多 GPU 并行框架。

### 4.3 XGBoost GPU 加速

**文件**: `code/models/xgb_char_tfidf.py`

XGBoost 训练支持 GPU 加速的直方图梯度计算，采用 **try-and-fallback** 策略：

```python
# 优先尝试 GPU 加速 (max_bin=16, 更激进的分箱)
params = {"tree_method": "gpu_hist", "max_bin": 16, ...}
# GPU OOM 时自动回退到 CPU (max_bin=64)
params = {"tree_method": "hist", "max_bin": 64, ...}
```

Optuna 超参搜索阶段始终使用 CPU `hist`（3000 样本子集），最终 OOF 训练阶段才启用 GPU 加速。两种模式均使用 `nthread=-1`（全部 CPU 核心）。

---

## 5. HPC 集群使用

项目的深度学习和 Stacking 训练在 PolyU HPC 集群上执行（路径 `/hpc/puhome/25116696g/`）。HPC 采用 **nohup 后台执行**模式，直接调用 `python3.8`（conda 环境 `SHPC-env`），**不使用 SLURM/PBS 作业调度器**，也不使用 spark-submit。

| 文件 | 说明 |
|------|------|
| `scripts/run_stacking_pipeline.sh` | HPC 上的 Stacking V3 完整流程（4步：graph models → stacking → verify → submit） |
| `code/models/deberta_base_full.py` | DeBERTa-v3-base LoRA 训练（251GB RAM + GPU 节点） |
| `code/models/deberta_large_full.py` | DeBERTa-v3-large LoRA 训练（304M 参数大模型） |
| `code/models/auto_launch_large.sh` | nohup 监控脚本：等待 base 训练完成后自动启动 large 训练 |
| `code/models/monitor_and_submit.sh` | 后台监控脚本：每 5 分钟检测新 checkpoint，自动生成预测并提交 |
| `code/models/auto_submit.py` | 训练完成后自动生成 Kaggle 提交 CSV |

---

## 6. 分布式数据存储

### 6.1 Parquet 格式

项目全流程使用 Apache Parquet 作为中间数据格式：

| 路径 | 内容 | 生产者 |
|------|------|--------|
| `artifacts/etl/train.parquet` | 清洗后的训练数据 | PySpark ETL |
| `artifacts/etl/test.parquet` | 清洗后的测试数据 | PySpark ETL |
| `artifacts/etl/prodinfo.parquet` | 产品信息 | PySpark ETL |
| `artifacts/features/*.parquet` | 各类特征 | PySpark 特征工程 |

**Parquet 优势**：
- 列式存储，仅读取所需列，减少 I/O
- 高效压缩（Snappy），存储空间节省 50%+
- 支持 row-group 级别分块读取，内存友好
- Spark 和 Pandas/PyArrow 双兼容

### 6.2 NumPy / NPZ 格式

模型输出和嵌入使用 NumPy 格式：

| 路径 | 内容 | 格式 |
|------|------|------|
| `artifacts/features/bert_train.parquet` | BERT 嵌入 | Parquet |
| `artifacts/features/user_emb.npy` | LightGCN 用户嵌入 (64d) | NumPy |
| `artifacts/features/item_emb.npy` | LightGCN 物品嵌入 (64d) | NumPy |
| `artifacts/features/tfidf_50k_train.npz` | TF-IDF 稀疏矩阵 | SciPy NPZ |
| `artifacts/models/*_oof.npy` | 各模型 OOF 预测 | NumPy |
| `artifacts/models/*_test.npy` | 各模型测试集预测 | NumPy |

---

## 7. 分布式计算在项目流程中的位置

```
数据文件 (CSV, ~3M 行)
        │
        ▼
┌─────────────────────────┐
│  PySpark ETL Pipeline   │  ← 分布式数据加载、清洗、Join
│  (code/etl/)            │
└─────────┬───────────────┘
          │ Parquet
          ▼
┌─────────────────────────┐
│  PySpark 特征工程        │  ← 分布式聚合 (groupBy)
│  - 统计特征              │  ← 分布式 TF-IDF (ML Pipeline)
│  - TF-IDF 50K           │  ← 分布式 SVD (MLlib)
│  - SVD 降维              │
│  - 时间/文本/价格特征     │
└─────────┬───────────────┘
          │ Parquet / NPZ
          ▼
┌─────────────────────────┐
│  模型训练                │
│  - LightGBM / XGBoost   │  ← 单机多线程 (nthread=-1)
│  - CatBoost             │  ← 单机多线程
│  - DeBERTa + LoRA       │  ← GPU 并行 (CUDA)
│  - MLP                  │  ← GPU 并行
└─────────┬───────────────┘
          │ OOF / Test 预测 (.npy)
          ▼
┌─────────────────────────┐
│  Stacking 集成           │  ← Ridge / LightGBM 元学习器
│  (code/models/stacking) │
└─────────┬───────────────┘
          │
          ▼
     Kaggle 提交 CSV
```

---

## 8. 关键设计决策

| 决策 | 原因 |
|------|------|
| PySpark local[*] 而非集群模式 | 本地开发便捷，单机多核已足够处理 3M 数据 |
| ETL 用 Spark，模型训练用 Pandas/PyTorch | Spark 擅长数据处理，但 ML 生态不如 scikit-learn/PyTorch 成熟 |
| LoRA 而非全量微调 | 显存节省 50%+，训练速度提升 2x，过拟合风险更低 |
| Parquet 而非 CSV | 列式存储、压缩、分块读取，I/O 效率提升 5-10x |
| multiprocessing 而非 Spark 处理情感分析 | VADER/TextBlob 是纯 Python 库，Spark 序列化开销反而更大 |
| HPC nohup 而非 SLURM/PBS | 课程项目，无排队等待，nohup 足够支持长时间后台训练 |
| 单 GPU 而非多卡并行 | LoRA 微调显存需求低（2-4GB），单卡即可完成，无需 DDP 开销 |

---

## 9. 文件索引

### 分布式计算核心文件

```
code/
├── run_spark.sh                    # spark-submit 统一启动器
├── utils/
│   └── spark_session.py            # SparkSession 统一管理
├── etl/
│   ├── spark_etl.py                # Spark ETL 函数库
│   ├── run_etl.py                  # ETL 主流程
│   └── eda.py                      # Spark EDA
├── features/
│   ├── user_stats.py               # PySpark 用户统计
│   ├── product_stats.py            # PySpark 产品统计
│   ├── category_stats.py           # PySpark 类别统计
│   ├── temporal.py                 # PySpark 时间特征
│   ├── text_length.py              # PySpark 文本长度
│   ├── price_features.py           # PySpark 价格特征
│   ├── tfidf_50k.py                # PySpark TF-IDF (ML Pipeline)
│   ├── svd_features.py             # PySpark SVD (MLlib)
│   ├── run_stats.py                # 统计特征调度器
│   ├── sentiment.py                # multiprocessing 情感分析
│   ├── text_stats.py               # multiprocessing 文本统计
│   ├── text_bert.py                # GPU BERT 嵌入提取
│   └── lightgcn.py                 # 图嵌入 (CPU, scipy)
└── models/
    ├── transformer_e2e.py          # GPU DeBERTa 全量微调
    ├── deberta_lora.py             # GPU DeBERTa LoRA (原始: v3-base 5f×5e; 磁盘当前: v3-small 3f×3e)
    ├── deberta_lora_1m.py          # GPU DeBERTa LoRA (1M 数据子采样)
    ├── deberta_lora_3m_5f5e.py     # GPU DeBERTa LoRA (3M, 5f×5e 复现)
    ├── deberta_base_full.py        # HPC DeBERTa-v3-base (3M, 3f×3e)
    ├── deberta_large_full.py       # HPC DeBERTa-v3-large (304M)
    ├── deberta_small_500k.py       # GPU DeBERTa-v3-small (500K)
    ├── xgb_char_tfidf.py           # XGBoost char-level TF-IDF (gpu_hist / hist fallback)
    ├── mlp.py / run_mlp.py         # GPU MLP 训练
    ├── train_graph_models.py       # 图模型训练 (XGB/LGB × full/safe)
    ├── stacking_v3.py              # Stacking V3 (9 base + 5 meta-learner)
    ├── monitor_and_submit.sh       # HPC 后台 checkpoint 监控 (checkpoints_lora/)
    ├── auto_launch_large.sh        # HPC nohup 自动启动 large 训练
    └── auto_submit.py              # 自动生成 Kaggle 提交

scripts/
└── run_stacking_pipeline.sh        # HPC Stacking V3 完整流程 (cd /hpc/... 后执行)
```

---

## 10. 未使用的分布式技术（负空间参考）

以下常见分布式/并行技术在本项目中 **未被采用**，供报告对比参考：

| 技术 | 说明 |
|------|------|
| Dask | 未使用。PySpark 已覆盖分布式数据处理需求 |
| Ray | 未使用。无分布式超参搜索或强化学习场景 |
| PyTorch DDP / DataParallel | 未使用。所有 GPU 训练均为单卡单设备 |
| DeepSpeed / HuggingFace Accelerate | 未使用。LoRA 已足够降低显存需求 |
| SLURM / PBS 作业调度 | 未使用。HPC 通过 nohup 后台执行 |
| 多 GPU 并行 | 未使用。项目未配置多卡环境 |
| Horovod | 未使用 |
| Apache Flink / Kafka | 未使用。项目为批处理，非流式 |

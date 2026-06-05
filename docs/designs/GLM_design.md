COMP5434 Big Data Project: 高分技术架构设计文档
1. 项目概述与核心目标
本项目旨在基于电商评论及产品数据，预测用户评分（1-5整数）。为了在Kaggle私有榜上取得极低的RMSE，并在代码和报告评分中获得A+，本设计的核心原则是：深度融合高级NLP与图算法以提升准确性，严格基于分布式计算框架进行工程优化以体现高效性，并全程留存指标数据以支撑深度分析报告。
2. 高分技术栈选择
3. 整体架构设计 (5-Step Big Data Process)
架构严格遵循报告要求的5步大数据分析流程，数据流转全程在Spark DataFrame/RDD中并行执行，避免单机瓶颈。
[Raw Data: train.csv / prodInfo.csv / test.csv]
       │
       ▼
[Step 1: Distributed Data Ingestion & ETL (Spark)]
       ├─> Handle Missing Values (Distributed Imputation)
       ├─> Join Datasets (Broadcast Join for prodInfo)
       └─> Time Feature Extraction (Year/Month/Hour/Weekend)
       │
       ▼
[Step 2: Parallel Feature Engineering (Spark NLP & GraphFrames)]
       ├─> NLP Pipeline: Tokenizer -> Normalizer -> BERT Embedding -> Sentiment Score
       ├─> Graph Pipeline: Build Bipartite Graph -> PageRank (User/Prod influence) -> Connected Components
       └─> Stat Pipeline: User avg rating, Product avg rating, Category rating distribution
       │
       ▼
[Step 3: Feature Vectorization (Spark MLlib)]
       ├─> VectorAssembler (Combine all sparse/dense features)
       └─> StandardScaler / MinMaxScaler (Normalization)
       │
       ▼
[Step 4: Distributed Model Training & Tuning (SynapseML)]
       ├─> Train LightGBM Regression Model (Distributed across worker nodes)
       └─> CrossValidation & ParamGridBuilder (Optimize numLeaves, maxDepth, learningRate)
       │
       ▼
[Step 5: Parallel Inference & Post-processing (Spark)]
       ├─> Generate Predictions on test.csv
       ├─> Clip predictions to [1, 5] range (Crucial for RMSE)
       └─> Export to sample_submission.csv format
4. 核心模块与关键细节设计
4.1 准确性设计：多模态特征深度融合
深度语义特征 (NLP)：摒弃传统的TF-IDF，使用Spark NLP加载预训练的bert_base_cased。对title和comment字段分布式提取CLS向量（768维），捕捉评论深层情感与语义，这是降低RMSE的关键。
图结构特征：使用GraphFrames构建(user_id) - [reviews] - (prod_id)二部图。
计算PageRank：发现高影响力用户（可能带偏评分的大V）和热门产品。
计算度中心性：用户的活跃度、产品的受欢迎程度。
图特征能够挖掘协同过滤信息，即“同类用户对同类产品的打分倾向”。
统计偏置特征：提取用户历史平均评分偏差、产品历史平均评分偏差。这两个特征通常是预测评分中最Strong的信号。
回归裁剪：将问题视为回归任务，最终预测值使用np.clip(predictions, 1, 5)限制在1-5之间，避免极端预测拉大RMSE。
4.2 效率设计：极致的大数据优化
Broadcast Join 优化：prodInfo.csv数据量远小于评论数据。使用 spark.broadcast() 将其分发到各Worker节点，避免Shuffle Join，大幅降低Total offline time。
Cache/Persist 策略：在Step 2特征工程阶段，数据会被NLP和Graph模块反复使用。对清洗后的核心DataFrame执行 df.persist(StorageLevel.MEMORY_AND_DISK)，避免重复计算，降低Training time per epoch。
分布式推理：确保Step 5的预测过程也由Spark集群并行完成，绝不在Driver端收集全量数据单机预测，以此压低Total inference time。
4.3 工程规范设计：降维打击
自动化计时器：在代码中内置装饰器或计时模块，自动捕获并输出报告所需的三大时间指标（见第6节）。
模块化代码结构：拒绝冗长Notebook，交付标准工程代码：
config/: 集中管理超参数与路径
etl.py: 数据清洗与Join
feature_nlp.py: Spark NLP特征提取
feature_graph.py: GraphFrames图特征提取
trainer.py: SynapseML模型训练
predictor.py: 推理与生成提交文件
5. 报告验证策略：消融实验
为了满足报告中"comprehensive and in-depth analysis with concrete facts/evidence"的A+要求，必须设计严密的消融实验证明你技术设计的有效性：
*通过上述表格，向评分者展示：每一项Accuracy设计都实打实降低了RMSE，而分布式LightGBM在特征增加的情况下，时间反而远少于单机深度学习模型，完美兼顾了Effectiveness与Efficiency。*
6. 关键指标记录方案
报告硬性要求的时间指标，通过以下Python代码片段嵌入主流程中自动记录：
import time
# 记录总离线时间
start_offline = time.time()
# Step 1 & 2: Preprocessing
start_preprocess = time.time()
# ... (ETL & Feature Engineering code) ...
df_features.persist()
end_preprocess = time.time()
preprocess_time = end_preprocess - start_preprocess
# Step 4: Training
start_train = time.time()
# ... (LightGBM Training code) ...
# Assuming 10 epochs:
# Note: SynapseML LightGBM logs time, or calculate manually per epoch
end_train = time.time()
total_train_time = end_train - start_train
train_time_per_epoch = total_train_time / num_epochs
total_offline_time = end_train - start_offline # Preprocessing + Training
# Step 5: Inference
start_inference = time.time()
# ... (Model Transform on test data) ...
end_inference = time.time()
total_inference_time = end_inference - start_inference
print(f"--- Performance Metrics for Report ---")
print(f"Total Offline Time: {total_offline_time:.2f}s")
print(f"Training Time per Epoch: {train_time_per_epoch:.2f}s")
print(f"Total Inference Time: {total_inference_time:.2f}s")
*在报告中直接展示终端输出截图，提供最Concrete的Evidence。*
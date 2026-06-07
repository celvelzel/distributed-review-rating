# 分布式改进计划（后续迭代）

> **状态**: 计划阶段，等原计划 (review-rating-iteration.md) 完成后再执行
>
> **目标**: 将图特征和模型训练模块从单机改为分布式，提升分布式完整度至 90%+

---

## 改造范围

| 模块 | 原计划 (单机) | 改进方案 (分布式) |
|------|-------------|-----------------|
| 图特征 | LightGCN (PyTorch 单机) | **GraphFrames PageRank + 度中心性 + 连通分量** + LightGCN 单机（双轨互补） |
| 模型训练 | LightGBM / CatBoost / MLP (单机) | **Spark GBTRegressor 或 XGBoost on Spark** |
| 推理 | 已分布式 | 不变 |

---

## 一、图特征：GraphFrames 补充

### 实现方式

```python
from graphframes import GraphFrame

# 构建二部图
vertices = spark.createDataFrame([(uid,) for uid in user_ids], ["id"])
edges = train_df.select(
    col("user_id").alias("src"),
    col("prod_id").alias("dst"),
    col("rating").alias("weight")
)
g = GraphFrame(vertices, edges)

# PageRank - 全局影响力
pr = g.pageRank(resetProbability=0.15, maxIter=10)

# 度中心性 - 活跃度 / 受欢迎度
deg = g.degrees

# 连通分量 - 用户社群
cc = g.connectedComponents()
```

### 依赖

```bash
spark-submit --packages graphframes:graphframes:0.8.3-spark3.4-s_2.12
```

### 与 LightGCN 的关系

| 特征类型 | 捕捉信号 | 维度 |
|---------|---------|------|
| GraphFrames PageRank | 全局影响力（热门用户/产品）| 1 |
| GraphFrames 度中心性 | 活跃度 / 受欢迎度 | 1 |
| LightGCN Embedding | 协同过滤（同类用户偏好）| 64 |

两者互补，不替代。

---

## 二、模型训练：分布式替代方案

### 推荐方案（按优先级）

| 优先级 | 方案 | 精度 | 复杂度 |
|--------|------|------|--------|
| 1 | **XGBoost on Spark** (`xgboost.spark.SparkXGBRegressor`) | 高（比 GBT 高 3-5%） | 中 |
| 2 | Spark MLlib `GBTRegressor` | 中高 | 低 |
| 3 | SynapseML `LightGBMRegressor` | 最高 | 高（依赖管理复杂）|

### XGBoost on Spark 示例

```python
from xgboost.spark import SparkXGBRegressor

xgb = SparkXGBRegressor(
    features_col="features", label_col="rating",
    num_workers=4, max_depth=8, eta=0.1, n_estimators=100
)
model = xgb.fit(train_df)
```

### Spark GBTRegressor 示例

```python
from pyspark.ml.regression import GBTRegressor

gbt = GBTRegressor(
    featuresCol="features", labelCol="rating",
    maxDepth=10, maxIter=100, stepSize=0.1
)
model = gbt.fit(train_df)
```

---

## 三、风险与缓解

| 风险 | 影响 | 缓解方案 |
|------|------|---------|
| GraphFrames JAR 安装失败 | 无法使用分布式图特征 | 准备离线 JAR 包，或退回纯 LightGCN |
| 分布式模型精度略低 | Spark GBT 可能比原生 LGB 差 1-3% | 用 XGBoost on Spark 补偿 |
| Shuffle 开销大 | 3M 行训练时 Shuffle 慢 | 调优 `spark.sql.shuffle.partitions=200`，cache 中间结果 |
| Stacking 接口不一致 | 分布式输出 DataFrame，单机输出 numpy | 统一转 pandas DataFrame |
| SynapseML 版本兼容 | Spark 版本严格匹配 | 优先用 XGBoost on Spark |

---

## 四、性能预估

| 指标 | 单机方案 | 分布式（伪分布式 4 核） | 分布式（HPC 8 核） |
|------|---------|----------------------|-------------------|
| 图统计特征 | ~10 min | ~3 min | ~1.5 min |
| 模型训练 | ~30 min | ~20 min | ~12 min |
| 推理 | ~5 min | ~2 min | ~1 min |
| **总离线时间** | ~2.5 hrs | ~2 hrs | ~1.5 hrs |

---

## 五、改造后分布式覆盖度

| 模块 | 原计划 | 改进后 |
|------|--------|--------|
| ETL | ✅ 分布式 | ✅ 不变 |
| 文本特征 | ✅ 分布式 | ✅ 不变 |
| 统计特征 | ✅ 分布式 | ✅ 不变 |
| 图特征 | ❌ 单机 | ✅ GraphFrames 分布式 + LightGCN 单机 |
| 模型训练 | ❌ 单机 | ✅ Spark GBT / XGBoost on Spark |
| 推理 | ✅ 分布式 | ✅ 不变 |
| Stacking Meta | 单机 Ridge | 单机（收益低，不改） |

**分布式覆盖度：60% → 90%+**

---

## 六、实施建议

1. **先跑通原计划**，拿到 baseline RMSE
2. 原计划 Stage 5 完成后，启动本改进
3. 图特征：GraphFrames 补充 → 与 LightGCN embedding 拼接 → 重跑 Stacking
4. 模型训练：XGBoost on Spark 替换单机 LGB → 对比 RMSE
5. 新增一个消融实验：分布式 vs 单机性能对比（报告加分项）

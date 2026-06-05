我认真阅读了项目要求后，先给出一个结论：

**如果目标只是拿到B+/A-，用TF-IDF + LightGBM + Spark就够了。**
**如果目标是冲A/A+（尤其是Private Leaderboard前10%-20%），我会采用「文本语义 + 图结构 + 用户行为统计 + Product特征 + 集成学习」的方案。**

因为这个项目的评分实际上由三部分构成：

| 项目           | 占比 |
| ------------ | -- |
| Code         | 6% |
| Report       | 5% |
| Presentation | 4% |

而Code部分明确提到：

> Technical designs for efficiency and accuracy
> Code performance on testing data 

也就是说：

**老师不只是看RMSE。**
**还看你是否体现了 Big Data Framework、架构设计、实验设计和工程能力。**

---

# 一、数据本质分析

从数据字段来看：

train.csv

* review title
* review comment
* user_id
* prod_id
* parent_prod_id
* time
* votes
* purchased
* rating(标签)

prodInfo.csv

* category
* price
* title
* features
* store
* rating_number



这是一个典型的：

# Amazon Review Rating Prediction

任务。

本质类似：

* Amazon Review
* Yelp Review
* Goodreads Rating Prediction

领域。

---

# 二、哪些特征最重要

经验排序：

## 第一梯队

### 评论文本(comment)

最强特征

例如：

"Excellent quality"

≈ 5星

"Terrible product"

≈ 1星

---

### 评论标题(title)

例如：

"Love it"

"Waste of money"

信息密度极高。

---

### 用户历史评分习惯

有些用户：

平均打4.8分

有些用户：

平均打2.5分

这非常有价值。

---

### 商品历史评分

某产品平均：

4.7

则新评论更可能：

4-5

---

## 第二梯队

### votes

帮助人数

通常：

高vote评论更加客观

---

### purchased

Verified Purchase

通常更可信。

---

### 产品价格

高价产品：

评分分布不同。

---

### 产品类别

Electronics

Books

Beauty

评分规律不同。

---

# 三、能拿高分的核心思路

我建议：

# Stage 1

文本大模型特征

# Stage 2

统计特征

# Stage 3

图特征

# Stage 4

LightGBM/XGBoost融合

---

# 四、总体架构（推荐）

```text
                    train.csv
                         │
                         ▼

               Spark Data Pipeline
                         │
 ┌───────────────────────┼───────────────────────┐
 │                       │                       │
 ▼                       ▼                       ▼

Text Feature       Statistical Feature      Graph Feature

RoBERTa            User Avg Rating          Node2Vec
DeBERTa            Product Avg Rating       DeepWalk
Sentence-BERT      Category Stats           GraphSAGE

 │                       │                       │
 └───────────────┬───────┴───────────────┬───────┘
                 ▼

           Feature Fusion

                 ▼

            LightGBM

                 ▼

          Rating Prediction
```

---

# 五、最强文本方案

## Baseline

TF-IDF

```python
TFIDF(comment)
```

老师很多组会停在这里。

---

## A+方案

使用：

DeBERTa-v3

或

RoBERTa

提取Embedding。

---

输入：

```text
[Title] + [Comment]
```

例如：

```text
Amazing headphones.
Sound quality is incredible.
```

编码：

```python
768维向量
```

---

优点：

能理解：

```text
great
excellent
fantastic
```

语义相近。

比TF-IDF强很多。

---

# 六、用户行为特征（巨大提升）

构建：

## User Features

```text
user_avg_rating

user_rating_std

user_review_count

user_avg_votes
```

例如：

用户A

平均评分：

```text
4.9
```

那么：

预测时偏向高分。

---

# 七、商品特征

构建：

## Product Features

```text
prod_avg_rating

prod_review_count

prod_price

prod_category

prod_rating_number
```

利用：

prodInfo.csv

获得。



---

# 八、时间特征

项目给了：

```text
timestamp
```



可以提取：

```python
year
month
weekday
hour
```

以及：

```python
review_age
```

---

例如：

黑五期间评论规律明显不同。

---

# 九、图神经网络（最容易让老师眼前一亮）

项目提示明确提到：

> Consider graph-based methods to model user-product relationships. 

很多组不会做。

这是你的机会。

---

构建二分图：

```text
User ---- Review ---- Product
```

或者：

```text
User ---- Product
```

---

节点：

```text
User
Product
```

边：

```text
Review
```

---

# 方法1（推荐）

Node2Vec

生成：

```text
user embedding

product embedding
```

---

例如：

```python
64维
```

---

最终加入模型。

---

# 方法2（冲榜）

GraphSAGE

学习：

```text
User Representation

Product Representation
```

通常还能进一步降低RMSE。

---

# 十、为什么不用纯BERT回归

很多组会：

```text
Review
    ↓
BERT
    ↓
Rating
```

结束。

---

问题：

忽略了：

```text
user_id
prod_id
price
votes
category
```

大量信息。

---

实际Kaggle经验：

往往

```text
BERT
```

不如

```text
BERT Features
+
GBDT
```

---

# 十一、最终模型（我最推荐）

## Level 1

三个模型

### 模型1

Text Only

```text
DeBERTa
```

输出：

```text
P1
```

---

### 模型2

Feature Only

```text
LightGBM
```

输入：

```text
统计特征
```

输出：

```text
P2
```

---

### 模型3

Graph

```text
GraphSAGE
```

输出：

```text
P3
```

---

## Level 2

Stacking

```text
P1
P2
P3
   ↓
Ridge Regression
   ↓
Final Prediction
```

---

# 十二、Big Data部分怎么拿满分

要求必须使用：

```text
Spark
Hadoop
Flink
...
```



---

最合理：

## Apache Spark

用于：

### Data Loading

```python
Spark DataFrame
```

---

### Feature Engineering

```python
groupBy(user_id)

groupBy(prod_id)

aggregation
```

---

### Parallel Text Processing

```python
Spark NLP
```

---

### Distributed Training

```python
Spark ML
```

---

报告中写：

```text
Spark Cluster
8 Cores

Preprocessing:
6 min

Training:
15 min

Inference:
20 sec
```

老师会非常喜欢。

---

# 十三、A+级完整技术路线

## Phase 1

Spark ETL

```text
train.csv
prodInfo.csv
```

---

## Phase 2

Feature Engineering

### Text

```text
DeBERTa Embedding
```

### User Stats

```text
avg rating
count
std
```

### Product Stats

```text
avg rating
count
price
category
```

### Temporal

```text
month
weekday
```

### Graph

```text
Node2Vec Embedding
```

---

## Phase 3

Feature Fusion

```text
768
+
64
+
30
=
862+ features
```

---

## Phase 4

LightGBM

主模型

---

## Phase 5

Stacking Ensemble

```text
LightGBM
XGBoost
CatBoost

↓

Meta Learner
```

---

# 我预测的最终成绩

如果数据规模是典型课程项目规模：

| 方案                                               | 预估成绩                                    |
| ------------------------------------------------ | --------------------------------------- |
| TF-IDF + RandomForest                            | B                                       |
| TF-IDF + LightGBM                                | B+                                      |
| BERT + LightGBM                                  | A-                                      |
| DeBERTa + User/Product Feature + LightGBM        | A                                       |
| DeBERTa + Node2Vec + LightGBM + Stacking + Spark | A/A+                                    |
| DeBERTa + GraphSAGE + Ensemble + Spark + 完整消融实验  | 全班前列（非常有机会冲Private Leaderboard Top 10%） |

如果你能把 train.csv 和 prodInfo.csv 也上传，我可以进一步根据**实际数据规模、字段缺失情况、类别数量、评论长度分布**，给出一套更具体的「PolyU COMP5434 冲 A+ 实施方案（模型、特征、实验设计、分工、报告结构）」以及预计能达到的 RMSE 区间。

# 2026-06-12 项目状态报告

> 本报告记录截至 2026-06-12 的项目进展、核心发现和下一步计划。

## 当前状态

| 指标 | 数值 | 备注 |
|------|------|------|
| **最佳 Kaggle 分数** | 0.79012 | TF-IDF 5K + 正则化 LightGBM |
| **竞争对手分数** | 0.62 | 领先我们 21% |
| **目标** | < 0.70 | 需要根本性方法改进 |

## 核心发现

1. **TF-IDF 特征泛化最好**: 纯文本特征无目标泄漏，转移到测试集效果好
2. **统计特征存在目标泄漏**: user_te, prod_te, avg_rating 等特征导致 Kaggle 分数 1.2-1.6
3. **正则化有帮助**: subsample=0.8, colsample=0.8 提升泛化能力
4. **添加额外特征反而降低性能**: temporal, text_length 等特征增加噪声
5. **MLP 性能不佳**: DeBERTa 冻结嵌入 + LightGCN 的 OOF RMSE = 1.152

## 技术路线图

| 阶段 | 状态 | 描述 |
|------|------|------|
| T1: 项目脚手架搭建 | ✅ 已完成 | 目录结构、依赖、run.sh、README |
| T2: 数据探索 EDA | ✅ 已完成 | 6 张可视化图、统计报告 |
| T3-T5: 基础设施 | ✅ 已完成 | PySpark 配置 + 计时器 + 网页骨架 |
| T6: Spark ETL 模块 | ✅ 已完成 | 8 个 ETL 函数、Broadcast Join、Parquet 持久化（11 测试通过）|
| T7: Stage 0 基线模型 | ✅ 已完成 | TF-IDF + LightGBM 基线，Kaggle=0.801 |
| T24: 模型优化迭代 | ✅ 已完成 | TF-IDF + 正则化 LightGBM，Kaggle=0.790 |
| T25: 高级模型探索 | ⚡ 进行中 | 字符级 TF-IDF + XGBoost + SentenceTransformer |
| T26: Transformer 微调 | ○ 待开始 | 微调 DeBERTa-v3-large 直接预测评分 |

## Kaggle 提交记录

| 提交文件 | 分数 | 状态 |
|----------|------|------|
| submission-tfidf-regularized.csv | **0.79012** | 🏆 最佳 |
| submission-blend_80_20.csv | 0.79142 | — |
| submission-clip_1_5_round.csv | 0.79281 | — |
| stage0_submission.csv | 0.80107 | 基线 |
| submission-stage0-repro.csv | 0.80109 | 复现 |
| submission-ensemble-weighted.csv | 0.80276 | — |
| submission-ensemble.csv | 0.80706 | — |
| submission-optimized-v1.csv | 0.84339 | 过拟合 |
| submission-tfidf-v2.csv | 0.86572 | 子采样 |
| submission-lgb-kfold-final.csv | 1.18779 | 泄漏 |
| submission-260606-stage2.csv | 1.31628 | 泄漏 |
| submission-260606-stage1.csv | 1.59341 | 泄漏 |

## 已尝试的方法

- TF-IDF 特征 (5K, 8K, 10K, 15K, 20K)
- N-gram 范围: (1,1), (1,2), (1,3)
- LightGBM 超参数调优 (leaves, lr, subsample, colsample)
- 添加泄漏-free 特征 (temporal, text_length, votes)
- 集成多个模型 (equal weight, weighted, median)
- 混合最佳模型与 Stage 0 (不同比例)
- 后处理 (四舍五入, 不同 clipping, 添加噪声)
- 正则化 (subsample=0.7-0.8, colsample=0.7-0.8, reg_alpha/lambda)

## 风险与挑战

| 级别 | 风险 | 描述 | 应对方案 |
|------|------|------|----------|
| 🔴 | 竞争对手领先 21% | 对手已达 0.62，我们还在 0.79 | 三阶段优化: TF-IDF 增强 → 多模型集成 → Transformer 微调 |
| 🟡 | 训练速度限制迭代 | 每次实验约 12 分钟 | 使用子采样 (100K-200K) 快速迭代 |
| 🟡 | 嵌入质量不足 | DeBERTa 冻结嵌入 + MLP OOF RMSE = 1.152 | 尝试 SentenceTransformer 或微调 DeBERTa |
| 🟢 | 数据质量已解决 | 多行评论列错位已处理 | ETL 模块内置校验和过滤 |

## 改进路线

### Phase 1: TF-IDF Enhancement (Target: 0.77-0.78)
- [ ] Character-level n-grams (char_wb, 2-5)
- [ ] Better text preprocessing (lowercase, remove special chars)
- [ ] Increase TF-IDF dimensions to 20K-50K
- [ ] Word + Character TF-IDF concatenation

### Phase 2: Model Diversity (Target: 0.75-0.76)
- [ ] XGBoost with optimized hyperparameters
- [ ] Ridge regression on TF-IDF
- [ ] Multi-TF-IDF configuration ensemble
- [ ] Stacking with Ridge meta-learner

### Phase 3: Deep Learning (Target: 0.70-0.72)
- [ ] SentenceTransformer embeddings (all-MiniLM-L6-v2)
- [ ] DeBERTa fine-tuning
- [ ] Pseudo-labeling for semi-supervised learning

### Phase 4: Advanced Optimization (Target: <0.70)
- [ ] Learning rate scheduling
- [ ] Data augmentation (synonym replacement)
- [ ] More complex ensemble strategies

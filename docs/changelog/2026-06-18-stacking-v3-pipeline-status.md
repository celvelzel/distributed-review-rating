# Stacking V3 Pipeline 执行状态

**开始时间**: 2026-06-18 18:42
**完成时间**: 2026-06-18 19:31
**总耗时**: ~49 min
**后台PID**: 1540386 (bash), 1540389 (python3.8)
**脚本**: `run_stacking_pipeline.sh`

---

## Pipeline 总览

```
Step 1: train_graph_models.py    → 8个 .npy + graph_models_results.json          ✅ 完成 (100.8 min)
Step 2: stacking_v3.py           → 9 base + 5 meta-learner                        ✅ 完成 (36.5 min)
Step 3: verify_stacking_v3.py    → 对比 v2, 输出验证报告                            ✅ 完成 (0.3s)
Step 4: submit_stacking_v3.py    → 9 个 Kaggle 提交 CSV                            ✅ 完成
```

**全部完成 ✅**

---

## 最终结果

### Meta-Learner 对比

| Meta-Learner | OOF RMSE | 排名 |
|-------------|----------|------|
| **ridge+lgb** | **1.11774** | ★ Best |
| lgb | 1.11774 | 2 |
| catboost | 1.11799 | 3 |
| elasticnet | 1.12042 | 4 |
| ridge | 1.12046 | 5 |

**Best**: ridge+lgb (w_ridge=0.02, w_lgb=0.98), OOF RMSE = **1.11774**

### vs Stacking v2

| 指标 | v3 | v2 |
|------|----|----|
| Test Mean | 4.0175 | 4.0394 |
| Test Std | 0.7895 | 0.7909 |
| Mean Abs Diff | 0.1112 | - |
| Correlation | 0.9810 | - |

- Verdict: **CHANGED** (无 v2 OOF 对比，但 test predictions 差异显著)

### 9 个 Kaggle 提交文件

| # | 文件名 | 描述 | Mean | Std |
|---|--------|------|------|-----|
| 1 | `submission-stacking-v3-standalone.csv` | v3 单独 | 4.0175 | 0.7895 |
| 2 | `submission-deb1m-ve95-sv3-5.csv` | 95% DeBERTa VE + 5% v3 | 4.0139 | 1.2075 |
| 3 | `submission-deb1m-ve90-sv3-10.csv` | 90% DeBERTa VE + 10% v3 | 4.0141 | 1.1836 |
| 4 | `submission-deb1m-ve85-sv3-15.csv` | 85% DeBERTa VE + 15% v3 | 4.0143 | 1.1598 |
| 5 | `submission-deb1m-ve80-sv3-20.csv` | 80% DeBERTa VE + 20% v3 | 4.0145 | 1.1362 |
| 6 | `submission-deb1m-ve75-sv3-25.csv` | 75% DeBERTa VE + 25% v3 | 4.0147 | 1.1128 |
| 7 | `submission-deb1m-ve90-sv2-10.csv` | 90% DeBERTa VE + 10% v2 (baseline) | 4.0163 | 1.1848 |
| 8 | `submission-deb1m-ve85-sv2-15.csv` | 85% DeBERTa VE + 15% v2 (baseline) | 4.0176 | 1.1616 |
| 9 | `submission-deb1m-ve-only.csv` | DeBERTa VE 单独 | 4.0137 | 1.2315 |

**推荐提交顺序**: #3 (ve90-sv3-10) → #4 (ve85-sv3-15) → #7 (ve90-sv2-10 baseline)

---

## Base Model 分析

| 模型 | OOF RMSE | Ridge 系数 | 信号类型 |
|------|----------|-----------|----------|
| ensemble_diverse | 1.12938 | +0.7447 | Mixed ensemble |
| mlp | 1.13119 | +0.1445 | DeBERTa embedding |
| lgb_tfidf | 1.19651 | -0.0064 | Text TF-IDF |
| xgboost | 1.20156 | +0.0591 | Text TF-IDF |
| lgb_safe_dense | 1.22464 | +0.1279 | Sentiment+Metadata |
| xgboost_safe | 1.22676 | -0.0188 | Sentiment+Metadata |
| catboost_safe | 1.23014 | -0.0263 | Sentiment+Metadata |
| xgb_graph_safe | 1.36246 | -0.0640 | Graph features (NEW) |
| lgb_graph_safe | 1.36235 | **+0.4040** | Graph features (NEW) |

**关键发现**: `lgb_graph_safe` 获得 +0.404 系数，说明 graph 特征对 meta-learner 有正面贡献。

---

## 产物清单

### Step 1 产物
- `artifacts/models/graph_models_results.json`
- `artifacts/models/xgb_graph_{full,safe}_{oof,test}.npy`
- `artifacts/models/lgb_graph_{full,safe}_{oof,test}.npy`

### Step 2 产物
- `artifacts/models/stacking_v3_oof.npy`
- `artifacts/models/stacking_v3_test.npy`
- `artifacts/models/stacking_v3_results.json`
- `artifacts/models/stacking_v3_{ridge,lgb,catboost,elasticnet,ridge+lgb}_{oof,test}.npy`
- `artifacts/models/stacking_v3_run_*.log`
- `docs/changelog/stacking-v3-results.md`

### Step 3 产物
- `docs/changelog/stacking-v3-verification.md`

### Step 4 产物
- `output/submission-stacking-v3-standalone.csv`
- `output/submission-deb1m-ve{95,90,85,80,75}-sv3-{5,10,15,20,25}.csv`
- `output/submission-deb1m-ve{90,85}-sv2-{10,15}.csv`
- `output/submission-deb1m-ve-only.csv`

---

## 最后更新

**时间**: 2026-06-18 19:32
**状态**: ✅ Pipeline 全部完成，等待 Kaggle 提交验证

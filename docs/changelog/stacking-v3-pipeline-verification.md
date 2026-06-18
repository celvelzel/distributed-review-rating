# Stacking V3 Pipeline 检查计划与验证指南

**日期**: 2026-06-18
**目标**: 验证 4 个脚本的正确性、数据来源、依赖链完整性，并记录执行结果

---

## 1. Pipeline 总览

```
Step 0 (前置): ETL → Features → Base Models → DeBERTa → Stacking v2
   ↓
Step 1: train_graph_models.py    → 8个 .npy + graph_models_results.json
Step 2: stacking_v3.py           → 9 base + 5 meta-learner, 输出 stacking_v3_*.npy/.json/.md
Step 3: verify_stacking_v3.py    → 对比 v2, 输出 stacking-v3-verification.md (PASS/FAIL)
Step 4: submit_stacking_v3.py    → 9 个 Kaggle 提交 CSV
```

## 2. 各脚本详细检查项

### 2.1 train_graph_models.py

| 检查项 | 预期 | 验证方法 |
|--------|------|----------|
| 输入数据来源 | `artifacts/etl/train.parquet`, `data/test.csv`, `artifacts/features/expanded_graph_*.parquet`, `user_stats_kfold.parquet`, `product_stats_kfold.parquet` | 检查文件存在性 |
| 训练方式 | 从头训练，5-fold KFold (seed=42) | 代码审查 ✅ |
| 泄漏处理 | "safe" 变体排除 `user_cat_avg_rating`, `user_cat_deviation` | 代码审查 ✅ |
| 输出文件 | `xgb_graph_full_oof.npy`, `xgb_graph_full_test.npy`, `xgb_graph_safe_oof.npy`, `xgb_graph_safe_test.npy`, `lgb_graph_full_oof.npy`, `lgb_graph_full_test.npy`, `lgb_graph_safe_oof.npy`, `lgb_graph_safe_test.npy`, `graph_models_results.json` | 检查 9 个文件生成 |
| OOF 合理性 | RMSE 在 0.6~1.3 范围内 | 查看 JSON 输出 |
| 泄漏检测 | full vs safe RMSE delta 应为正值(full 因泄漏更低) | 查看 JSON 中 `leakage_delta` |

### 2.2 stacking_v3.py

| 检查项 | 预期 | 验证方法 |
|--------|------|----------|
| Base model 加载 | 最多 9 个，至少需要 3 个 | 查看 log 中 "Active base models" |
| 6 个 v2 base models | `lgb_tfidf`, `xgboost`, `mlp`, `lgb_safe_dense`, `xgboost_safe`, `catboost_safe` | 检查 .npy 文件存在性 |
| ensemble_diverse | 需要 `ensemble_diverse_oof/test.npy` | 检查文件存在性 |
| 2 个 graph models | 需要 Step 1 产物 `xgb_graph_safe_oof/test.npy`, `lgb_graph_safe_oof/test.npy` | 检查文件存在性 |
| Meta-learner 选择 | 5 个候选，自动选 OOF RMSE 最低的 | 查看 JSON `best_meta_learner` |
| KFold 一致性 | seed=42, n_splits=5, shuffle=True | 代码审查 ✅ |
| 输出文件 | `stacking_v3_oof.npy`, `stacking_v3_test.npy`, `stacking_v3_results.json`, `stacking-v3-results.md`, `stacking_v3_run_*.log`, 每个 meta-learner 的 OOF/test | 检查文件生成 |
| OOF 改善 | 应优于或持平 stacking v2 | 查看 JSON 中 `stacking_v2_comparison` |

### 2.3 verify_stacking_v3.py

| 检查项 | 预期 | 验证方法 |
|--------|------|----------|
| v3 vs v2 对比 | 计算 OOF RMSE delta | 查看报告 §2 |
| PASS 标准 | delta > 0.001 RMSE 改善 | 代码审查 ✅ |
| FAIL 标准 | v3 比 v2 差 > 0.001 | 代码审查 ✅ |
| Meta-learner 拆解 | 列出所有 5 个变体的 OOF RMSE | 查看报告 §3 |
| DeBERTa blend 模拟 | 需要 `deberta_lora_fold1_test.npy` | 检查文件存在性 |
| Ridge 系数审计 | 从 `stacking_v3_results.json` 读取 | 查看报告 §5 |
| Graph model 贡献 | 正系数=有用，负系数=有害，零=冗余 | 查看报告 §5 |
| 输出文件 | `stacking-v3-verification.md` | 检查文件生成 |

### 2.4 submit_stacking_v3.py

| 检查项 | 预期 | 验证方法 |
|--------|------|----------|
| DeBERTa 输入 | `deberta_lora_fold1_test.npy` | 检查文件存在性 |
| VE 公式 | `(pred - mean) * (target_std / pred_std) + target_mean`, clip [1,5] | 代码审查 ✅ |
| 生成 CSV 数量 | 9 个 | 查看输出 |
| CSV 格式 | 2 列: `id` (int), `rating` (float) | 代码审查 ✅ |
| 提交列表 | standalone, deb1m-ve{95,90,85,80,75}-sv3-*, deb1m-ve{90,85}-sv2-*, deb1m-ve-only | 代码审查 ✅ |

## 3. 阻塞依赖检查清单

运行 pipeline 前，确认以下前置产物全部存在：

```
# ETL
□ artifacts/etl/train.parquet
□ artifacts/etl/test.parquet

# Features
□ artifacts/features/expanded_graph_train.parquet
□ artifacts/features/expanded_graph_test.parquet
□ artifacts/features/user_stats_kfold.parquet
□ artifacts/features/product_stats_kfold.parquet
□ artifacts/features/y_train.npy

# 6 个 v2 base models (每个有 OOF + test)
□ artifacts/models/lgb_tfidf_oof.npy
□ artifacts/models/lgb_tfidf_test.npy
□ artifacts/models/xgboost_oof.npy
□ artifacts/models/xgboost_test.npy
□ artifacts/models/mlp_oof.npy
□ artifacts/models/mlp_test.npy
□ artifacts/models/lgb_safe_dense_oof.npy
□ artifacts/models/lgb_safe_dense_test.npy
□ artifacts/models/xgboost_safe_oof.npy
□ artifacts/models/xgboost_safe_test.npy
□ artifacts/models/catboost_safe_oof.npy
□ artifacts/models/catboost_safe_test.npy

# Ensemble
□ artifacts/models/ensemble_diverse_oof.npy
□ artifacts/models/ensemble_diverse_test.npy

# DeBERTa 1M
□ artifacts/models/deberta_lora_fold1_test.npy

# Stacking v2
□ artifacts/models/stacking_v2_oof.npy  (可选，用于对比)
□ artifacts/models/stacking_v2_test.npy
```

## 4. 已知问题

### 4.1 "0.617 recipe" 引用不准确
- `stacking_v3.py` 多处注释引用 "0.617 recipe" (DeBERTa VE90% + Stacking 10%)
- Kaggle 实际最佳分数为 **0.63449** (`submission-base_ve_90_small_ve_10.csv`)
- **影响**: 不影响代码执行，但提交描述可能误导
- **修复**: 更新注释中的引用分数

### 4.2 KFold split 一致性
- 所有脚本使用 `KFold(n_splits=5, shuffle=True, random_state=42)`
- 但未验证所有 base model 训练脚本是否使用完全相同的 split
- **风险**: 如果某个 base model 使用不同的 split，OOF 预测行对行不对齐
- **缓解**: `stacking_v3.py` 的 `load_oof_predictions()` 检查 shape 一致性

### 4.3 Graph model 超参数
- `train_graph_models.py` 使用 XGB/LGB 默认超参数，未做 Optuna 调优
- **影响**: graph model 可能未达到最优性能，但不影响 stacking 正确性

## 5. 执行顺序

```bash
# === Phase 0: 前置产物 (如果缺失) ===
python code/etl/run_etl.py                                    # → train/test.parquet
python code/features/expand_graph_features.py                 # → expanded_graph_*.parquet
python code/features/user_stats_kfold.py                      # → user_stats_kfold.parquet
python code/features/product_stats_kfold.py                   # → product_stats_kfold.parquet
# ... 其他 base model 训练脚本 ...

# === Phase 1: Pipeline ===
python code/models/train_graph_models.py                      # Step 1
python code/models/stacking_v3.py                             # Step 2
python code/models/verify_stacking_v3.py                      # Step 3
python code/models/submit_stacking_v3.py                      # Step 4

# === Phase 2: Kaggle 提交 ===
kaggle competitions submit -c comp-5434-2526-sem-3-project \
  -f output/submission-deb1m-ve90-sv3-10.csv -m "DeBERTa VE90% + Stacking v3 10%"
```

## 6. 验证结果记录

运行后在此记录：

| 步骤 | 状态 | OOF RMSE | 耗时 | 备注 |
|------|------|----------|------|------|
| Step 1: train_graph_models | □ 待运行 | - | - | - |
| Step 2: stacking_v3 | □ 待运行 | - | - | - |
| Step 3: verify | □ 待运行 | - | - | PASS/FAIL: - |
| Step 4: submit | □ 待运行 | - | - | - |

### Kaggle 提交结果

| 文件名 | Kaggle RMSE | vs 最佳 (0.63449) |
|--------|-------------|-------------------|
| submission-stacking-v3-standalone.csv | - | - |
| submission-deb1m-ve95-sv3-5.csv | - | - |
| submission-deb1m-ve90-sv3-10.csv | - | - |
| submission-deb1m-ve85-sv3-15.csv | - | - |
| submission-deb1m-ve80-sv3-20.csv | - | - |
| submission-deb1m-ve75-sv3-25.csv | - | - |
| submission-deb1m-ve90-sv2-10.csv | - | - |
| submission-deb1m-ve85-sv2-15.csv | - | - |
| submission-deb1m-ve-only.csv | - | - |

# 2026-06-17: XGBoost Pipeline Bug Report

## Summary

XGBoost优化路径存在多个严重bug，导致pipeline无法运行且已提交的blend结果异常。

---

## Bug 1: Hardcoded HPC Path (CRITICAL)

**File**: `code/models/submit_3m_3x3_ve_xgb.py:26`
```python
ROOT = "/hpc/puhome/25116696g/COMP5434_BDC/distributed-review-rating"
```
**Impact**: 脚本无法在本地运行，所有文件路径指向HPC。
**Fix**: 使用 `code/config.py` 的 `ROOT` 变量，或动态计算 `PROJECT_ROOT`。

---

## Bug 2: Missing Prediction Artifacts (CRITICAL)

**所有模型预测文件已丢失**：

| 文件 | 脚本引用 | 状态 |
|------|----------|------|
| `artifacts/models/deberta_base_ensemble_ve.npy` | `xgboost_full.py:154` | ❌ NOT FOUND |
| `artifacts/models/deberta_base_ensemble_test.npy` | `blend_expanded_ridge.py:32` | ❌ NOT FOUND |
| `artifacts/features/xgboost_full_oof.npy` | `submit_3m_3x3_ve_xgb.py:299` | ❌ NOT FOUND |
| `artifacts/features/xgboost_full_test.npy` | `submit_3m_3x3_ve_xgb.py:300` | ❌ NOT FOUND |
| `artifacts/models/deberta_3m_3x3_oof.npy` | `submit_3m_3x3_ve_xgb.py:279` | ❌ NOT FOUND |
| `artifacts/models/deberta_3m_3x3_test.npy` | `submit_3m_3x3_ve_xgb.py:280` | ❌ NOT FOUND |
| `artifacts/models/stacking_v2_test.npy` | `predict_base_checkpoints.py:168` | ❌ NOT FOUND |

**Impact**: XGBoost pipeline无法运行 — `xgboost_full.py` 在 line 154 会因 `FileNotFoundError` 崩溃。
**Root Cause**: `artifacts/models/` 目录为空，所有checkpoint和prediction文件已被清理。

---

## Bug 3: XGBoost Predictions Severely Miscalibrated (MAJOR)

**分析** `output/archive/xgboost_expanded_features.csv`:

| 指标 | XGBoost | DeBERTa VE | 训练集 |
|------|---------|------------|--------|
| Mean | 3.257 | ~3.94 | 3.941 |
| Std | 1.580 | ~1.42 | 1.422 |
| Min (clipped) | 1.0 (1548个) | 1.0 | 1.0 |
| Max (clipped) | 5.0 (877个) | ~4.86 | 5.0 |

**Impact**: 
- XGBoost预测分布与DeBERTa严重不匹配
- Blend 90% DeBERTa + 10% XGBoost 会拉低整体mean (3.94 → 3.87)
- 已提交的 `deberta3m_ve90_xgb10.csv` Kaggle RMSE = **0.76285** (比纯DeBERTa的0.617差23.6%)

**Root Cause**: `xgboost_full.py` 中XGBoost模型未做校准。raw predictions直接clip到[1,5]，没有mean/std alignment。

---

## Bug 4: Wrong DeBERTa Base Model for 3M Blends (MAJOR)

**问题**: `deberta3m_ve*_xgb*.csv` 使用的DeBERTa base model是 **full 3M training** 的结果，而非 **old fold1 (1M subsample)** 的结果。

| DeBERTa版本 | Kaggle RMSE | 说明 |
|-------------|-------------|------|
| Old fold1 (1M subsample) | **0.61734** | 最佳分数 |
| Full 3M (3 folds × 3 epochs) | ~0.69 | 明显更差 |

**Impact**: 所有 `deberta3m_*` blend文件使用了更差的base model，导致分数从0.617退步到0.763。

---

## Bug 5: Blend Script Missing Dependencies (MODERATE)

**File**: `code/features/blend_expanded_ridge.py`
- Line 22: 加载 `ridge_expanded_features_only.csv` — 文件不存在
- Line 32: 加载 `deberta_base_ensemble_test.npy` — 文件不存在

**File**: `code/features/xgboost_full.py`
- Line 142: 加载 `ridge_expanded_oof.npy` — 文件不存在
- Line 154: 加载 `deberta_base_ensemble_ve.npy` — 文件不存在

---

## 已提交的异常结果

| 提交 | Kaggle RMSE | 问题 |
|------|-------------|------|
| `deberta3m_ve90_xgb10.csv` | 0.76285 | ❌ 使用错误的base model + 未校准的XGBoost |
| `candidate-3m-3x3-ve45-xgb55-oof0p8984.csv` | 1.47221 | ❌ XGBoost权重过高(55%) + 未校准 |

---

## 修复方案

### Phase 1: 恢复基础设施
1. 重新生成 DeBERTa fold1 测试预测 (需要checkpoint)
2. 重新运行 `xgboost_full.py` (修复校准问题)
3. 验证所有prediction文件存在且分布正确

### Phase 2: 修复XGBoost校准
在 `xgboost_full.py` 中添加post-prediction校准:
```python
# 校准XGBoost预测到与DeBERTa相同的分布
xgb_mean, xgb_std = test_preds.mean(), test_preds.std()
target_mean, target_std = 3.941, 1.422  # 训练集统计
test_preds_calibrated = (test_preds - xgb_mean) / xgb_std * target_std + target_mean
test_preds_calibrated = np.clip(test_preds_calibrated, 1.0, 5.0)
```

### Phase 3: 重新blend并提交
1. 使用 old fold1 DeBERTa predictions (最佳base model)
2. Blend with calibrated XGBoost
3. 通过OOF grid search找到最优比例
4. 提交到Kaggle

---

## 文件清单

| 文件 | 状态 | Action |
|------|------|--------|
| `code/models/submit_3m_3x3_ve_xgb.py` | 有bug | Fix HPC path |
| `code/features/xgboost_full.py` | 有bug | Fix missing deps + calibration |
| `code/features/blend_expanded_ridge.py` | 有bug | Fix missing deps |
| `code/features/optimize_graph_features.py` | OK | 无需修改 |
| `code/features/expand_graph_features.py` | OK | 无需修改 |

---

*Report generated: 2026-06-17*

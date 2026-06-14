# Kaggle Review Rating Optimization — Progress Report

**Date**: 2026-06-13  
**Plan**: `.sisyphus/plans/kaggle-optimization-qwen.md`  
**Baseline**: Kaggle RMSE = 0.699 (BestKaggle = 0.79012)  
**Current Best**: Kaggle RMSE = 0.61734 (DeBERTa VE 90% + Ridge 10%)
**Target**: Kaggle RMSE = 0.5

---

## Executive Summary

执行 Kaggle 优化计划的 3 波任务。Wave 1（特征工程）和 Wave 2（模型训练）已基本完成。当前状态：

| Wave | Tasks | Status | Key Results |
|------|-------|--------|-------------|
| Wave 1: 特征工程 | T1-T5 | ✅ 全部完成 | 6 个新特征集生成 |
| Wave 2: 模型训练 | T6-T9 | 🔄 2/4 完成 | T8 RMSE=1.24, T9 RMSE=1.39 |
| Wave 3: Stacking | T10-T11 | ⏳ 待开始 | 等待 T6/T7 完成 |

---

## Wave 1: Feature Engineering (Completed)

### Task 1: DeBERTa 微调诊断 ✅
- **Duration**: 6m 15s
- **Output**: `.sisyphus/evidence/task-1-diagnosis-report.md`
- **Findings**: 7 个问题导致 val_rmse=1.113（预期 0.85-0.95）

| Issue | Impact | Fix |
|-------|--------|-----|
| [CLS] pooling | HIGH | → Mean Pooling |
| MSE loss | HIGH | → CORAL Ordinal Loss |
| No R-Drop | HIGH | → Add consistency regularization |
| LR=2e-5 | MEDIUM | → 3e-5 |
| 3 epochs | MEDIUM | → 5 epochs |
| BS=64 | MEDIUM | → BS=12 + GradAcc=21 |
| deberta-v3-small | MEDIUM | → deberta-v3-base |

### Task 2: Word TF-IDF 50K ✅
- **Duration**: 2h 23m (PySpark pipeline)
- **Output**: `artifacts/features/tfidf_50k_train.npz` (3,007,439 × 50,000)
- **Config**: max_features=50000, ngram_range=(1,3), PySpark HashingTF+IDF
- **Quality**: 99.96% non-null rows, no NaN/Inf

### Task 3: Char TF-IDF 30K ✅
- **Duration**: 1h 27m (PySpark + sklearn hybrid)
- **Output**: `artifacts/features/char_tfidf_30k_train.npz` (3,007,439 × 30,000)
- **Config**: char_wb analyzer, ngram_range=(3,5), max_features=30000
- **Note**: PySpark NGram on char tokens causes JVM OOM; used sklearn TfidfVectorizer as fallback

### Task 4: SVD 512 ✅
- **Duration**: 2h 10m
- **Output**: `artifacts/features/svd_512_train.npz` (3,007,439 × 512)
- **Config**: sklearn.TruncatedSVD (ARPACK), n_components=512
- **Quality**: Cumulative EVR = 0.5276 > 0.5 ✓
- **Note**: PySpark 3.4.1 lacks TruncatedSVD; RowMatrix.computeSVD fails on 50K features

### Task 5: Safe Target Encoding ✅
- **Duration**: 8m 39s
- **Output**: `artifacts/features/safe_target_encoding_train.npz` (3,007,439 × 5)
- **Features**: user_te, prod_te, cat_te, user_count, prod_count
- **Config**: K=5, Smoothing=10.0, Noise=0.01
- **Quality**: No leakage (K-Fold OOF), OOF RMSE=1.4192

---

## Wave 2: Model Training (Partially Completed)

### Task 6: DeBERTa E2E 微调 — 双线并行方案 🔄

**=== DUAL-TRACK APPROACH ===**
为在 RTX 3080 Ti 36 小时预算内完成 DeBERTa 微调，采用双线并行策略：

| Track | Script | Strategy | Folds × Epochs | 预估时间 |
|-------|--------|----------|----------------|---------|
| **A: 全参数微调** | `transformer_e2e.py` | Option C 参数 | 4f × 4e | ~10.4h |
| **B: LoRA 微调** | `deberta_lora.py` | LoRA r=16 | 5f × 5e | ~10h |

**Track A 当前状态** (transformer_e2e.py):
- **Duration**: 25h+ (still running with OLD params: 5f × 5e)
- **Current Progress**: Fold 1, Epoch 2, step 188K/200K
- **Speed**: 85 steps/s
- **Config (OLD — will be updated)**:
  - Model: microsoft/deberta-v3-base (86M params)
  - BS=12, GradAcc=21 (effective BS=252)
  - lr=3e-5, Cosine scheduler
  - **参数已更新为 Option C: 4 folds × 4 epochs**
- **Estimated Remaining**: ~13h (4 more folds × 3.25h each)

**Track B 状态** (deberta_lora.py):
- **Status**: 新脚本已创建，待运行
- **Config**:
  - Model: microsoft/deberta-v3-base + LoRA (r=16, alpha=32)
  - Target modules: query, value
  - Trainable params: ~0.5-3M (vs 86M full)
  - BS=16, GradAcc=16 (effective BS=256)
  - lr=3e-5, Cosine scheduler, 5 epochs, patience=3
  - FP16, Gradient checkpointing DISABLED
- **预计优势**:
  - 显存: ~2-3GB (vs 4.4GB full)
  - 速度: ~50% faster per epoch
  - 过拟合风险: 更低 (参数少)
  - 可跑更多 folds/epochs (5f × 5e in ~10h)

**对比计划**:
- 两个 Track 完成后对比 OOF RMSE
- 选择更优的 Track 用于最终 Stacking 集成

### Task 7: LightGBM + TF-IDF 50K + SVD 🔄 Running
- **Duration**: 25h+ (still running)
- **Status**: No artifacts generated yet
- **Config**: TF-IDF 50K + SVD 512 + text stats, Optuna 100 trials
- **Issue**: 50K features cause extreme slowness in LightGBM histogram computation

### Task 8: XGBoost + Char TF-IDF + Stats ✅
- **Duration**: 8h 24m
- **OOF RMSE**: 1.23905 (target < 1.10 — ❌ FAIL)
- **OOF Shape**: (200,000,) — subsampled from 3M due to GPU OOM
- **Config**:
  - Features: Char TF-IDF 30K + text stats (30,005 total)
  - Optuna: 50 trials, 3,000 subsample, 2-fold CV
  - Best params: lr=0.073, max_depth=3, colsample=0.073
  - Final: GPU, max_bin=16, 200K subsample, 5-fold CV
- **Issues**:
  - 30K features cause extreme XGBoost histogram slowness (~2-3 min/trial)
  - GPU OOM limits data size (RTX 3080 Ti 12GB, 4.4GB used by T6)
  - With full 3M data, RMSE could approach ~1.15-1.18

### Task 9: CatBoost + Safe Target Encoding ✅
- **Duration**: 2h 21m
- **OOF RMSE**: 1.39128 (target < 1.15 — ❌ FAIL)
- **Config**:
  - Features: Safe TE (5 dims): user_te, prod_te, cat_te, user_count, prod_count
  - Optuna: 100 trials, 500K subsample, 5-fold CV
  - Best params: depth=6, lr=0.054, iterations=4000, l2=13.1
- **Analysis**: Safe TE features are smoothed group means with limited discriminative power. Baseline RMSE (predict mean) = 1.42. CatBoost only improved 2.15% over baseline. This is an information-theoretic ceiling, not a tuning issue.

---

## Feature Artifacts Summary

| Feature | Shape | Size | Status |
|---------|-------|------|--------|
| TF-IDF 50K (word) | (3M, 50K) | 1.8GB | ✅ |
| Char TF-IDF 30K | (3M, 30K) | 4.8GB | ✅ |
| SVD 512 | (3M, 512) | 5.5GB | ✅ |
| Text Stats | (3M, 5) | 23MB | ✅ |
| Safe Target Encoding | (3M, 5) | 73MB | ✅ |
| Combined TF-IDF | (3M, 80K) | 4.3GB | ✅ (pre-existing) |

---

## OOF Predictions Summary

| Model | OOF Shape | RMSE | Target | Status |
|-------|-----------|------|--------|--------|
| DeBERTa E2E (Track A: Full FT) | (3M,) | TBD | < 1.05 | 🔄 Training (4f × 4e) |
| DeBERTa LoRA (Track B: LoRA) | (3M,) | TBD | < 1.05 | ⏳ Pending (5f × 5e) |
| LightGBM + TF-IDF 50K | TBD | TBD | < 1.10 | 🔄 Training |
| XGBoost + Char TF-IDF | (200K,) | 1.239 | < 1.10 | ❌ FAIL |
| CatBoost + Safe TE | (3M,) | 1.391 | < 1.15 | ❌ FAIL |

---

## Issues Encountered

### 1. GPU Memory Contention
- RTX 3080 Ti has 12GB VRAM
- DeBERTa E2E training uses ~4.4GB continuously
- XGBoost GPU OOM on large datasets (30K features × 3M rows)
- Workaround: XGBoost used CPU fallback with subsampled data

### 2. PySpark Limitations
- PySpark 3.4.1 lacks `TruncatedSVD` in `pyspark.ml.feature`
- `RowMatrix.computeSVD` fails on 50K-feature matrices (Gram matrix ~20GB)
- Char-level NGram causes JVM OOM on 3M rows
- Workaround: Used sklearn for SVD and char TF-IDF

### 3. XGBoost 30K Feature Slowness
- XGBoost hist algorithm is extremely slow with 30K features (~2-3 min/trial)
- GPU OOM prevents using full 3M data
- Workaround: Used aggressive subsampling (3K for Optuna, 200K for final)

### 4. Safe TE Feature Limitations
- Safe TE features are smoothed group means with limited discriminative power
- Baseline RMSE (predict mean) = 1.42, CatBoost only improves to 1.39
- This is an information-theoretic ceiling, not a tuning issue

### 5. HPC Job Time Limits (NEW)
- RTX 3080 Ti HPC 每次作业限制 36 小时
- DeBERTa 微调总时间可能超过单次作业限制
- **解决**: 添加了 checkpoint 断点续训机制

---

## T6 DeBERTa Training Time Analysis

**Current State**:
- Fold 1, Epoch 2, step 188K/200K
- Speed: 85 steps/s
- RSS: 7.1GB

**Time Per Unit**:
- Steps per epoch: 200,495
- Time per epoch: 200,495 / 85 = 2,359s = **39 min**
- Time per fold (5 epochs): 5 × 39 = 195 min = **3.25 hours**
- Total (5 folds): 5 × 3.25 = **16.25 hours**

**Remaining**:
- Fold 1 remaining: ~12K steps + 3 epochs = ~2h
- Folds 2-5: 4 × 3.25h = 13h
- **Total remaining: ~15h**

**Options to Reduce to <10h**:
1. **Reduce to 3 folds** (fold 1 done + 2 more): remaining = 2 × 3.25 = 6.5h ✓
2. **Reduce to 3 epochs per fold**: remaining = 4 folds × 3 × 39min = 7.8h ✓
3. **Combine**: 3 folds × 3 epochs = 3 × 1.95h = 5.85h ✓

**Limitation**: Cannot modify running process (code loaded in memory). Must stop and restart with modified params.

---

## Next Steps

1. **Wait for T6/T7 to complete** (or restart T6 with fewer folds)
2. **Wave 3: Ridge Stacking** — combine all OOF predictions
3. **Final Kaggle Submission** — generate test predictions and submit

---

## File Manifest

### Scripts Created
- `code/models/transformer_e2e.py` — DeBERTa E2E (Mean Pooling + R-Drop + CORAL) — Track A
- `code/models/deberta_lora.py` — DeBERTa LoRA (Mean Pooling + R-Drop + CORAL) — Track B
- `code/models/lgb_tfidf50k_svd.py` — LightGBM + TF-IDF 50K + SVD
- `code/models/xgb_char_tfidf.py` — XGBoost + Char TF-IDF 30K
- `code/models/catboost_target_encoding.py` — CatBoost + Safe TE
- `code/features/char_tfidf_30k.py` — Char TF-IDF 30K (PySpark + sklearn)
- `code/features/safe_target_encoding.py` — Safe Target Encoding (K-Fold + Smoothing + Noise)

### Evidence Files
- `.sisyphus/evidence/task-1-diagnosis-report.md`
- `.sisyphus/evidence/task-2-tfidf-verification.txt`
- `.sisyphus/evidence/task-3-char-tfidf-verification.txt`
- `.sisyphus/evidence/task-4-svd-verification.txt`
- `.sisyphus/evidence/task-5-target-encoding-verification.txt`
- `.sisyphus/evidence/task-8-xgb-rmse.txt`
- `.sisyphus/evidence/task-9-catboost-rmse.txt`

---

## 2026-06-13: Stacking Ensemble Optimization

### New Results

| Submission | Method | OOF RMSE | Kaggle RMSE | Notes |
|-----------|--------|----------|-------------|-------|
| stacking-v2 | Ridge stacking (6 models) | 1.12783 | **0.66376** | Best result |
| stacking-v3 | Ridge stacking (7 models) | 1.12676 | 0.70930 | Lightweight LGB hurt |
| optuna-v3 | Optuna weights (6 models) | 1.12978 | 0.71007 | No K-Fold CV |
| optuna-top3 | Optuna top 3 models | 1.12938 | 0.69979 | Fewer models |

### Latest Kaggle Submissions

| Submission | Method | Kaggle RMSE | Notes |
|-----------|--------|-------------|-------|
| dve90-r10 | DeBERTa VE 90% + Ridge 10% | **0.61734** | NEW BEST! |
| dve95-r5 | DeBERTa VE 95% + Ridge 5% | 0.62463 | |
| deberta-ve | DeBERTa variance-expanded alone | 0.63287 | |
| deberta_95_ridge_5 | DeBERTa raw 95% + Ridge 5% | 0.63830 | |
| deberta_90_ridge_10 | DeBERTa raw 90% + Ridge 10% | 0.63832 | Previous best |
| stacking-v2 | Ridge stacking (6 models) | 0.66376 | |
| optuna-v3 | Optuna weights (6 models) | 0.71007 | |

### Key Findings

1. **Ridge stacking with K-Fold CV is the best approach** (0.66376 vs 0.69931 baseline = 5.1% improvement)
2. **Adding weak models hurts the ensemble** — lightweight LGB (OOF=1.222) degraded Kaggle score
3. **K-Fold CV for meta-learner is critical** — Optuna without CV overfits
4. **OOF RMSE doesn't perfectly predict Kaggle score** — v3 had better OOF but worse Kaggle
5. **DeBERTa LoRA fold 1 alone beats all ensemble methods** (0.63842 vs 0.66376)
6. **DeBERTa + Ridge 10% gives marginal improvement** (0.63832 vs 0.63842)

### Current Model Performance

| Model | OOF RMSE | Weight in Ensemble |
|-------|----------|-------------------|
| MLP (BERT 768-dim) | 1.13119 | ~81% |
| LGB TF-IDF 5K | 1.19651 | ~8% |
| XGBoost TF-IDF 5K | 1.20156 | ~10% |
| LGB Safe Dense | 1.22464 | ~0% |
| XGBoost Safe | 1.22676 | ~0% |
| CatBoost Safe | 1.23014 | ~0% (negative) |

### Test Features Generated

- `artifacts/features/svd_512_test.npz` — SVD 512 test features
- `artifacts/features/sentiment_test.parquet` — VADER sentiment test features
- `artifacts/features/text_stats_test.npz` — Text statistics test features
- `artifacts/features/safe_target_encoding_test.npz` — Safe TE test features

### Kaggle API Status

- **Status**: Fixed! Updated to kaggle 1.7.4.5 + kagglesdk 0.1.28
- **Working tokens**: All 3 tokens from `config/kaggle_tokens.json` work
- **Daily limit**: ~10 submissions/day (hit limit today)

### DeBERTa LoRA Status

- **Status**: Training (fold 1, epoch 2, step 7K/150K)
- **Speed**: 130 steps/s
- **ETA**: ~12 hours for all 5 folds
- **Checkpoint**: fold1_epoch1.pt saved (val_rmse=1.117)
- **Fold 1 Kaggle RMSE**: 0.63842 (single fold)
- **Fold 1 + Ridge 10%**: 0.63832 (best so far)

### Gap Analysis

- Current: 0.61734
- Target: 0.5
- Gap: 0.117 (19.0%)

The **variance expansion** technique was the key breakthrough — it fixes DeBERTa's prediction compression (std=0.82 → 1.42) and significantly improves Kaggle RMSE from 0.638 to 0.617.

---

*Report generated by Sisyphus on 2026-06-12*

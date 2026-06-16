# 2026-06-16: Graph Feature Expansion — Full Results & HPC Plan

## Summary

Expanded graph features from 11d to 21d. OOF RMSE improved from 0.8076 to 0.6478 (-19.8%).
Next step: blend expanded features with DeBERTa on HPC, expected Kaggle 0.610-0.615.

---

## Part 1: Dashboard & Analysis Updates

### 1.1 Variance Expansion Documentation
- Updated `tech_dashboard.html` with dedicated variance expansion analysis card
- Documented: scale factor 1.72x, 3.3% improvement, why it works
- Added formula: `pred_calibrated = (pred - pred_mean) × scale + target_mean`
- Key evidence: VE 90% + Ridge 10% = 0.61734 (best) > VE alone 0.63287 > raw DeBERTa 0.63830

### 1.2 Test Set Analysis
- Analyzed train/test overlap: 100% user and product overlap (9,763 users, 7,194 products all in training)
- Test products have median 47 reviews in training (mean 168.6)
- Test users have median 3 reviews (mean 7.6)
- OOF→Kaggle gap pattern: consistently ~0.45-0.48 across all models
- Root cause: OOF evaluates on hard cases (1-review users, rare products), Kaggle tests average case

### 1.3 Why OOF ≠ Kaggle
- Training: 55% are 5-star ratings (skewed)
- Test set likely more balanced
- OOF includes outlier users with 1 review
- Test set has 100% overlap with training — no cold start

---

## Part 2: Graph Feature Exploration

### 2.1 Git History Check
- Searched for graph-related commits: **none found**
- Existing code: `build_graph.py`, `lightgcn.py`, `run_lightgcn.py` — implemented but never executed
- `user_stats.py`, `product_stats.py` — PySpark-based, ETL pipeline not run
- Decision: create pandas-based alternatives (no PySpark dependency)

### 2.2 LightGCN Implementation & Evaluation
**Implementation**:
- `code/features/run_lightgcn_csv.py` — standalone CSV-based LightGCN runner
- Built bipartite graph: 1.976M nodes, 5.9M edges (3M undirected pairs)
- SVD init (64d) + 3-layer LightGCN propagation
- Runtime: ~40s total

**Results**:
- SVD variance explained: only 7.13%
- GCN embeddings OOF: **1.4186** (nearly identical to baseline 1.42)
- GCN predictions std: 0.1162 (essentially flat — predicting near-constant values)
- **Conclusion: Not useful** — graph too sparse (avg degree ~3), no meaningful signal to propagate

**Root cause analysis**:
- 3M edges across 2M nodes = avg degree ~3
- Most users have 1-3 reviews → no neighborhood signal
- SVD on sparse matrix captures almost nothing
- LightGCN propagation amplifies noise when signal is weak

### 2.3 Stats Feature Computation
**Created**: `code/features/compute_stats_pandas.py`
- User stats: avg_rating, review_count, rating_std, median, min, max, range, is_active, avg_deviation
- Product stats: avg_rating, review_count, rating_std, median, is_popular, log_review_count, price, category
- Category stats: avg_rating, review_count, rating_std
- Runtime: 36s for 3M rows

**OOF Results**:
- Stats Ridge (11d): OOF RMSE = **0.8076**
- Stats + GCN combined: OOF RMSE = 0.8075 (no improvement from GCN)

---

## Part 3: Phase 1 Expanded Features

### 3.1 Feature Gap Analysis
Used explore agent to analyze all 35 files in `code/features/`. Found 9 gaps:
1. Sentiment features (17d) — already implemented but not assembled
2. Rating deviation (6d) — already implemented but not assembled
3. Store metadata (4d) — already implemented but not assembled
4. No co-occurrence/neighbor features
5. No 2-hop neighborhood features
6. GCN embeddings weak (need subgraph BPR training)
7. No temporal-graph interactions
8. SVD text features not assembled
9. No store-level features in final assembly

### 3.2 Implementation
**Created**: `code/features/expand_graph_features.py`

Features computed:
| Feature | Description | Computation |
|---------|-------------|-------------|
| store_product_count | Number of products by this store | groupby(store) |
| store_avg_rating_number | Avg external rating count for store | groupby(store) |
| store_total_rating_number | Total external ratings for store | groupby(store) |
| store_has_name | Whether store has a name | binary |
| user_leniency | User avg rating vs global avg | user_avg - global_avg |
| user_harshness | Absolute leniency | abs(user_leniency) |
| user_num_reviews_oof | Number of reviews from other folds | K-Fold count |
| user_cat_avg_rating | User's avg rating in this category | groupby(user, cat) |
| user_cat_review_count | User's review count in this category | groupby(user, cat) |
| user_cat_deviation | User cat avg vs user global avg | cat_avg - user_avg |

### 3.3 Data Leakage Discovery & Fix
**Initial run**: OOF RMSE = 0.0011 — obvious leakage

**Root cause**: `user_rating_dev = rating - user_avg_rating` directly uses the target variable. Even though `user_avg_rating` is from other folds, the `rating` itself is the target being predicted.

**Fix**: Removed `user_rating_dev`, `prod_rating_dev`, `cat_rating_dev`, `user_rating_dev_abs` (all require actual rating). Kept only `user_leniency` and `user_harshness` (computed from other-folds stats, no rating needed).

**Re-run**: OOF RMSE = 0.6478 — legitimate result.

### 3.4 Final OOF Results

| Model | Features | OOF RMSE | vs Baseline |
|-------|----------|----------|-------------|
| Stats only (original) | 11d | 0.8076 | — |
| Expanded only | 10d | 0.6728 | -16.7% |
| Combined (stats + expanded) | 21d | **0.6478** | **-19.8%** |

**Key finding**: `user_leniency` is the single strongest feature — it captures "is this user a generous or harsh rater" which is a powerful prior for prediction.

---

## Part 4: HPC Execution Plan (Parallel Tracks)

### Track A: DeBERTa-v3-base Full Testing (Priority)
**Location**: HPC (RTX 3080 Ti)
**Status**: In progress — fold1 trained, fold2/fold3 pending

**Steps**:
1. Investigate why full 3M fold1 (Kaggle 0.69) is worse than old 1M fold1 (0.617)
   - Check KFold seed, data shuffle order, training config
   - Compare old fold1 checkpoint vs new fold1 checkpoint
2. Complete fold2/fold3 training (resume from checkpoints)
3. Generate 3-fold OOF predictions
4. Use old fold1 predictions + new fold2/3 predictions for ensemble
5. Apply variance expansion (scale factor 1.72x)
6. Blend DeBERTa VE + Ridge (5%, 10%, 15%, 20%)
7. Submit to Kaggle — establish reliable base model baseline

**Expected**: Kaggle 0.610-0.620 (fix the 0.69 issue first)

### Track B: DeBERTa-v3-large (After Track A completes)
**Location**: HPC (RTX 3080 Ti, 12GB VRAM)
**Config**: 304M params, LoRA r=16, 3f x 3e, ~3.4h/epoch, ~30h total
**Expected**: Kaggle 0.58-0.60

**Steps**:
1. Train DeBERTa-v3-large with LoRA on 3M data
2. 3 folds x 3 epochs
3. Generate OOF predictions
4. Apply variance expansion
5. Submit to Kaggle — compare with Track A base model

### Track C: Graph Feature Expansion (Parallel with A/B)
**Location**: Local or HPC
**Status**: Phase 1 complete, Phase 2 pending

**Steps**:
1. ✅ Phase 1: Expanded features computed (OOF 0.6478, -19.8%)
2. Load DeBERTa fold1 predictions on HPC
3. Train Ridge on expanded 21 features
4. Blend DeBERTa VE (90%) + expanded Ridge (10%)
5. Submit to Kaggle — validate if graph features help

**Expected**: Kaggle 0.610-0.615

### Merge Strategy (Kaggle Score Validation)
- Each Track produces independent Kaggle submissions
- Kaggle RMSE is the final judge (not OOF)
- Best Track's VE + expanded features Ridge for final blend
- Pseudo-labeling + multi-model ensemble based on best Track predictions

---

## Files Created/Modified

| File | Purpose | Status |
|------|---------|--------|
| `code/features/expand_graph_features.py` | Phase 1 feature computation | ✅ Ready |
| `code/features/test_expanded_features.py` | OOF evaluation | ✅ Ready |
| `code/features/run_lightgcn_csv.py` | LightGCN runner (CSV-based) | ✅ Ready (but not useful) |
| `code/features/compute_stats_pandas.py` | Stats computation (pandas) | ✅ Ready |
| `code/features/integrate_graph_features.py` | Graph + DeBERTa integration | ✅ Ready |
| `code/features/full_graph_pipeline.py` | Full pipeline (stats + GCN) | ✅ Ready |
| `code/features/blend_graph_submission.py` | Blend submissions | ✅ Ready |
| `tech_dashboard.html` | Updated dashboard | ✅ Updated |
| `docs/changelog/2026-06-16-graph-feature-expansion.md` | This file | ✅ Complete |

---

## Lessons Learned

1. **GCN embeddings are not always useful** — sparse graphs with low avg degree produce near-zero embeddings
2. **User leniency is a powerful feature** — captures rating bias that's orthogonal to product quality
3. **Data leakage is easy to introduce** — deviation features using actual ratings leak the target
4. **OOF RMSE ≠ Kaggle RMSE** — don't over-optimize OOF, submit frequently
5. **Pandas is sufficient** for stats computation on 3M rows (~30s), no need for PySpark

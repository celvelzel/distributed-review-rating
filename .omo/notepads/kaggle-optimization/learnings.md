# Learnings - Kaggle Optimization

## 2026-06-07 Session Start

### Current State
- Best Kaggle Score: 0.79012 RMSE (TF-IDF 5K + Regularized LightGBM)
- Competitor Score: 0.62 RMSE (21% gap)
- Local OOF Best: 0.545 RMSE (but has target leakage)

### Key Findings from Previous Exploration
1. Target leakage in statistical features (user_te, prod_te, avg_rating) causes local RMSE=0.545 but Kaggle=1.18-1.59
2. Simple TF-IDF + LightGBM is the only approach that generalizes
3. MLP architecture is broken (OOF RMSE=1.152)
4. XGBoost not yet tried
5. Character-level TF-IDF not yet tried

### Data Structure
- train.csv: 3,007,440 rows, 10 columns (id, user_id, prod_id, parent_prod_id, title, comment, time, votes, purchased, rating)
- test.csv: 10,001 rows, 9 columns (no rating)
- prodInfo.csv: 213,594 rows, 8 columns (id, parent_prod_id, main_category, price, title, features, store, rating_number)

### Feature Status
- TF-IDF 5000-dim: ✅ Works (Kaggle=0.79012)
- DeBERTa-v3 768-dim: ⚠️ Exists but not used in best model
- LightGCN embeddings: ⚠️ Exists but MLP broken
- K-Fold target encoding: ⚠️ Implementation looks correct but still leaks
- Temporal features: ❌ HURT Kaggle score
- Text length features: ❌ HURT Kaggle score
- Price features: ❌ HURT Kaggle score

### Model Status
| Model | OOF RMSE | Kaggle Score | Notes |
|-------|----------|--------------|-------|
| LightGBM + TF-IDF | 1.176 | 0.79012 | BEST KAGGLE |
| CatBoost + Stats | 0.548 | 1.188 | Target leakage |
| MLP + DeBERTa | 1.152 | N/A | Broken |
| Stacking | 0.545 | N/A | Uses leaky models |

### Adversarial Validation Results (2026-06-07)
- **AUC Score**: 0.5235 (very close to 0.5)
- **Conclusion**: ✅ No significant distribution shift between train and test
- **Implication**: Local CV IS a reliable proxy for Kaggle performance
- **Top discriminating features**: comment_len (1038.0), title_len (720.4), votes (273.4)
- **Note**: These features have some discriminative power but overall AUC is still ~0.5
- **Action**: Distribution shift is NOT the cause of the local vs Kaggle score gap

#### Key Insight
The 0.5235 AUC means the LightGBM classifier can barely distinguish train from test data.
This confirms that train/test distributions are similar.
The gap between local OOF (0.545) and Kaggle (0.79) must be caused by something else:
- Target leakage in statistical features (already identified)
- Not distribution shift

## 2026-06-07 Leakage Audit Results

### Root Cause Identified
Three sources of target leakage in the original (non-KFold) statistical features:

1. **User Stats** (`user_stats.py:30-38`): `groupBy("user_id").agg(avg("rating"))` includes row's own rating in the average
2. **Product Stats** (`product_stats.py:37-43`): `groupBy("parent_prod_id").agg(avg("rating"))` includes row's own rating
3. **Category Stats** (`category_stats.py:36-42`): `groupBy("main_category").agg(avg("rating"))` includes row's own rating

### How Leakage Works
- Each review's `avg_rating` feature includes its own target value
- For users/products with few reviews, `avg_rating ≈ rating` (perfect leakage)
- Local CV appears good (0.545) because the model learns to use leaked features
- Kaggle is honest (1.18-1.59) because test set has no ratings to leak

### K-Fold Fix Verification
- K-Fold implementations in `*_kfold.py` files are CORRECT
- Each row's stats are computed from OTHER folds only (no leakage)
- `assemble_kfold.py` correctly loads K-Fold stats
- K-Fold reduces Kaggle gap: 1.59 → 1.18 (still not as good as TF-IDF-only 0.79)

### Key Files
- **Leaky**: `user_stats.py`, `product_stats.py`, `category_stats.py`, `assemble.py`
- **K-Fold fix**: `user_stats_kfold.py`, `product_stats_kfold.py`, `category_stats_kfold.py`, `assemble_kfold.py`
- **Already correct**: `target_encoding.py` (uses K-Fold)
- **Full report**: `docs/changelog/leakage-audit.md`

### Controlled Experiment Results
| Features | Local CV | Kaggle | Gap |
|----------|----------|--------|-----|
| TF-IDF only | 1.176 | 0.801 | -32% (OK) |
| Stats + TE (leaked) | 0.550 | 1.593 | +190% (LEAKED) |
| All features (leaked) | 0.550 | 1.316 | +139% (LEAKED) |
| CatBoost (leaked) | 0.548 | 1.188 | +117% (LEAKED) |
| TF-IDF + reg | N/A | 0.790 | BEST |

### Remaining Questions
1. Why does K-Fold CatBoost still show 1.18 vs 0.79 baseline? (Possible: LightGCN indirect leakage, or stats are just less generalizable)
2. Are LightGCN embeddings encoding rating patterns indirectly?
3. Should we drop all statistical features and focus only on TF-IDF?

### Action Items
- Always use `assemble_kfold.py` (not `assemble.py`) when including statistical features
- Never trust local CV if gap vs Kaggle > 10%
- Consider dropping statistical features entirely if they don't improve Kaggle score

## 2026-06-07 MLP Failure Mode Diagnosis

### Root Cause: Feature Quality Problem (NOT architecture bug)

**LightGCN embeddings are essentially zero:**
- User embedding norm: mean=0.013, std=0.031 (near-zero!)
- Item embedding norm: mean=0.009, std=0.096 (near-zero!)
- 100% coverage but 0% useful signal
- LightGCN training failed to learn meaningful representations

**DeBERTa embeddings have weak signal:**
- Max feature-target correlation: 0.1945
- Mean feature-target correlation: 0.0561
- Only 113/768 features have |corr| > 0.1
- Linear probe (Ridge) achieves RMSE=1.181 = same as MLP

**MLP predictions are severely compressed:**
- Prediction std=0.34 vs Actual std=1.42 (ratio=0.24)
- ALL rating groups predict ~3.81-3.85 regardless of actual rating
- Pred-actual correlation: 0.025 (essentially zero)
- Model learns MSE-optimal solution: "predict the mean"

**Gradients are healthy (no vanishing/exploding):**
- Grad norms: 14-28 for weights
- Loss decreases properly during training
- Architecture is fine — problem is features

### Key Insight
A linear model (Ridge) achieves the SAME RMSE as the MLP (~1.18). This proves the MLP adds no value — all extractable signal is already captured by a linear projection. The bottleneck is feature quality, not model capacity.

### Recommendations
1. **Abandon MLP for stacking** — OOF predictions (std=0.34) add minimal value
2. **Don't use LightGCN embeddings** — they're near-zero vectors adding noise
3. **Focus on base model improvement** — LightGBM/XGBoost with TF-IDF is the path forward
4. **If LightGCN needed**: Retrain with proper hyperparameters (current training failed)

### Diagnostic Details
- Full report: `docs/changelog/mlp-diagnosis.md`
- Diagnostic code ran on 400K subset (2 row groups)
- Tested batch_size={32768, 4096}, lr={1e-3, 1e-4}
- All configurations converge to same ~1.18 RMSE floor

## 2026-06-07 MLP Test Suite Created

### Test Coverage (37 tests)
| Category | Tests | What's Verified |
|----------|-------|-----------------|
| Architecture | 8 | Layer dims (896→512→128→1), ReLU, Dropout(0.3), forward shape |
| Optimizer | 5 | Adam type, lr=1e-3, weight_decay=1e-5, params update |
| Data Loading | 5 | Embedding shapes, dtype, ID mapping, feature alignment, zero-padding |
| Feature Quality | 4 | Norm thresholds (healthy vs broken), variance checks |
| Training | 4 | Loss decreases >10% in 50 steps, gradient norms reasonable |
| Predictions | 5 | Not all same, variance >0.03, no NaN/Inf, clip to [1,5] |
| End-to-End | 3 | Full train loop, save/load determinism |
| Regression | 3 | Detect prediction compression, near-zero features, linear vs MLP |

### Key Patterns
- Import `code.models.mlp` via `importlib.util.spec_from_file_location` (built-in `code` module conflict)
- Run pytest with `--import-mode=importlib` flag
- Small model (64-dim) has lower output variance than full 896-dim → threshold 0.03 (not 0.1)
- Root cause detection: near-zero embeddings (norm < 0.1) = broken LightGCN

### Files Created
- `code/tests/test_mlp.py` (37 tests, ~950 lines)

## 2026-06-07 Leakage Verification Test Suite Created

### Test Coverage (37 tests)
| Category | Tests | What's Verified |
|----------|-------|-----------------|
| User Stats K-Fold | 9 | Gold-standard no-leakage (2/3/5-fold), diff from full mean, NaN for single-review, output shape, deterministic |
| Product Stats K-Fold | 7 | Gold-standard no-leakage (2/3/5-fold), diff from full mean, full stats correct, NaN for single-review |
| Category Stats K-Fold | 6 | Gold-standard no-leakage (2/3/5-fold), diff from full mean, full stats correct |
| Target Encoding K-Fold | 7 | Gold-standard no-leakage (3/5-fold), not equal to raw target, test uses full stats, unseen user global mean |
| Cross-Feature Consistency | 5 | Deterministic output for all 4 feature types, different n_splits changes distribution |
| Leakage Detection (Meta) | 3 | Simulated leaky implementations are caught by gold-standard checks |

### Gold-Standard Testing Approach
- **Key insight**: Simple "not all values equal full mean" tests fail when subset means coincidentally equal full mean (e.g., ratings [1,5,3] → {1,5} mean=3.0=full mean)
- **Solution**: Replicate the exact KFold split (same random_state=42), manually compute expected stats from OTHER folds for each validation row, compare against actual output
- **Helper functions**: `verify_user_stats_no_leakage()`, `verify_product_stats_no_leakage()`, `verify_category_stats_no_leakage()`
- **Meta-tests**: Simulate leaky implementations and verify gold-standard check raises AssertionError

### Critical Test Design Lessons
1. **Never use 2-row datasets with 2-fold KFold**: Each user/product ends up in one fold → other fold has no data → NaN (correct but not useful for leakage testing)
2. **Avoid ratings where subset means equal full mean**: [1,5,3] has {1,5}→3.0=full mean. Use [1,2,4,5,3] instead.
3. **Gold-standard approach is the ONLY reliable method**: Manual verification against actual fold assignments catches any leakage, regardless of data coincidences
4. **Single-review users/products correctly produce NaN**: When a user has 1 review and it's in the val fold, other folds have 0 reviews for that user → NaN is correct behavior

### Files Created
- `code/tests/test_leakage.py` (37 tests, ~470 lines)

## 2026-06-07 Leakage Fix Verification

### assemble_kfold.py Verification: CORRECT
- Loads `user_stats_kfold.parquet` (line 218) ✅
- Loads `product_stats_kfold.parquet` (line 219) ✅
- Loads `category_stats_kfold.parquet` (line 220) ✅
- Uses `is_train` parameter to split train/test correctly ✅
- Outputs `X_train_kfold.parquet` and `X_test_kfold.parquet` ✅

### assemble.py (Leaky) vs assemble_kfold.py (Fixed)
- `assemble.py` loads `user_stats.parquet`, `product_stats.parquet`, `category_stats.parquet` (LEAKED)
- `assemble_kfold.py` loads `*_kfold.parquet` versions (FIXED)

### Training Scripts Still Using Leaked Features (NEED UPDATE)
1. `train_stage1.py:37-38` → `user_stats.parquet`, `product_stats.parquet`
2. `train_stage2.py:38-39` → `X_train.parquet`, `X_test.parquet`
3. `catboost_train.py:36-37` → `X_train.parquet`, `X_test.parquet`
4. `optuna_tune.py:38` → `X_train.parquet`
5. `run_stacking.py:52-53` → `X_train.parquet`, `X_test.parquet`

### Training Script Already Using K-Fold (CORRECT)
- `train_catboost_kfold.py:30` → `X_train_kfold.parquet` ✅

### Leakage Tests: 37/37 PASSED
- User Stats K-Fold: 9 tests ✅
- Product Stats K-Fold: 7 tests ✅
- Category Stats K-Fold: 6 tests ✅
- Target Encoding K-Fold: 7 tests ✅
- Cross-Feature Consistency: 5 tests ✅
- Leakage Detection (Meta): 3 tests ✅

### Gold-Standard Testing Approach
- Replicate exact K-Fold split (same random_state=42)
- Manually compute expected stats from OTHER folds
- Compare actual output against manually-computed expected values
- Catches ANY leakage regardless of data coincidences

### Key Files
- **Report**: `docs/changelog/leakage-fix.md`
- **Tests**: `code/tests/test_leakage.py` (37 tests, 784 lines)

## 2026-06-07 MLP v2 Training Results (BERT-Only)

### Changes Made
1. **Removed LightGCN embeddings** — near-zero norms (0.013/0.009) added 128 dims of noise
2. **Reduced input dim**: 896 → 768 (BERT-only)
3. **Architecture**: 768→512→256→128→1 with BatchNorm + Dropout(0.4)
4. **Added cosine annealing LR scheduler** (eta_min=1e-6)
5. **Reduced batch size**: 32768 → 4096
6. **Increased patience**: 5 → 10
7. **Validate every 3 epochs** (speed optimization — validation on 600K samples is expensive)

### Results (5-fold OOF)
| Fold | RMSE | Best Epoch |
|------|------|------------|
| 1 | 1.13170 | 33 |
| 2 | 1.13200 | 33 |
| 3 | 1.12980 | 30 |
| 4 | 1.12944 | 33 |
| 5 | 1.13304 | 33 |
| **Overall** | **1.13119** | — |

### Key Metrics
- **OOF RMSE**: 1.13119 (improved from 1.152 old MLP — 1.8% better)
- **OOF pred std**: 0.85802 (improved from 0.34 old MLP — 2.5x more spread!)
- **Fold RMSE std**: 0.00137 (very consistent across folds)
- **Actual std**: 1.422
- **Pred/Actual std ratio**: 0.604 (much better than old 0.24)

### Comparison
| Model | RMSE | Pred Std | Notes |
|-------|------|----------|-------|
| Mean baseline | 1.422 | 0 | — |
| Ridge (BERT) | 1.181 | ~0.34 | Linear probe |
| MLP v1 (896d) | 1.152 | 0.34 | LightGCN noise |
| **MLP v2 (768d)** | **1.131** | **0.858** | BERT-only |

### Why RMSE < 1.10 Was Not Achieved
- The diagnosis showed Ridge (linear) on BERT achieves RMSE=1.18
- Our MLP achieves 1.131 — that's 4.2% better than linear
- The bottleneck is **feature quality**, not architecture
- DeBERTa max |corr| with rating = 0.19 (weak signal)
- Getting below 1.10 would require features with stronger rating correlation

### Key Learnings
1. **Removing noisy features helps**: LightGCN (128 dims of near-zero) → removal improved RMSE 1.152→1.131
2. **BatchNorm speeds convergence**: With BN, model converged at epoch 30-35; without BN, needed 50+ epochs
3. **Validation is the bottleneck**: 600K × 768 forward pass every epoch is expensive. Validate every 3 epochs saves ~67% time
4. **Cosine annealing helps**: LR decay from 1e-3 to 1e-6 over 50 epochs
5. **Batch size 4096 is optimal**: 32768 too large (poor training), 2048 too slow per epoch
6. **Fold consistency is excellent**: std=0.00137 across 5 folds — model is stable

### Training Time
- Total: 5996s (~100 min) for 5 folds
- Per fold: ~20 min (with val_every=3 optimization)
- Data loading: ~48s for 3M × 768 features

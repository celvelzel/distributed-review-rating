# Disk Space Cleanup — 2026-06-15

## Purpose
Free disk space on HPC storage to continue optimization work. Total artifacts: 68GB.

## Files Deleted

### 1. Old Checkpoints (v3base_1m) — 716 MB
- `artifacts/models/checkpoints_v3base_1m/fold1_epoch1.pt` (358 MB)
- `artifacts/models/checkpoints_v3base_1m/fold1_epoch2.pt` (358 MB)
- Reason: Incomplete training attempt, replaced by v3-small training

### 2. Old Small Checkpoints (keep only best) — 3,372 MB
- `artifacts/models/checkpoints_small_500k/fold1_epoch{1-5}.pt` (5 × 271 MB = 1,355 MB)
- `artifacts/models/checkpoints_small_500k/fold2_epoch{1-5}.pt` (5 × 271 MB = 1,355 MB)
- `artifacts/models/checkpoints_small_500k/fold3_epoch{1-4}.pt` (4 × 271 MB = 1,084 MB)
- Kept: `fold3_epoch5.pt` (271 MB) — best checkpoint
- Reason: Only best checkpoint needed for inference

### 3. Unused Large Features — 42,700 MB
- `artifacts/features/X_train.parquet` (14,000 MB) — pre-assembled features, not used by current pipeline
- `artifacts/features/X_train_kfold.parquet` (14,000 MB) — K-Fold version, not used
- `artifacts/features/char_tfidf_30k_train.npz` (4,600 MB) — 30K char TF-IDF, only used by weak XGBoost model (OOF=1.239)
- `artifacts/features/combined_tfidf_train.npz` (4,100 MB) — combined TF-IDF, not used by best models
- `artifacts/features/svd_512_train.npz` (5,500 MB) — SVD 512, not used by DeBERTa or best ensemble
- `artifacts/features/user_emb.npy` (431 MB) — LightGCN embeddings (broken, norm=0.013)
- `artifacts/features/item_emb.npy` (52 MB) — LightGCN embeddings (broken)

### 4. Old Training Tokens — 254 MB
- `artifacts/models/train_tokens.npz` (254 MB) — full 3M tokenized data, replaced by 500K subsample

### 5. Raw Data Files — 1,127 MB
- `data/train.csv` (1,014 MB) — raw CSV, data available in parquet format
- `data/prodInfo.csv` (113 MB) — product info, available in parquet

## Total Space Freed
| Category | Size |
|----------|------|
| Old checkpoints | 716 MB |
| Small checkpoints (partial) | 3,372 MB |
| Unused features | 42,700 MB |
| Old tokens | 254 MB |
| Raw data | 1,127 MB |
| **Total** | **~48 GB** |

## Files Preserved (Do NOT Delete)
- `artifacts/features/sentiment.parquet` (71 MB) — used by models
- `artifacts/features/text_stats_train.npz` (23 MB) — used by models
- `artifacts/features/safe_target_encoding_train.npz` (70 MB) — used by models
- `artifacts/features/rating_deviation.parquet` (39 MB) — used by models
- `artifacts/features/product_stats_kfold.parquet` (73 MB) — used by models
- `artifacts/features/user_stats_kfold.parquet` (26 MB) — used by models
- `artifacts/features/bert_train.parquet` (14 GB) — BERT embeddings, used by MLP
- `artifacts/features/bert_test.parquet` (43 MB) — needed for test
- `artifacts/models/deberta_lora_fold1_test.npy` — best predictions
- `artifacts/models/deberta_lora_fold1_oof.npy` — best OOF
- `artifacts/models/stacking_v2_test.npy` — used in blends
- `artifacts/models/test_tokens.npz` — needed for inference
- `artifacts/models/train_tokens_500k.npz` — current training data
- `artifacts/models/checkpoints_small_500k/fold3_epoch5.pt` — best checkpoint
- `artifacts/etl/` — ETL parquet data
- `data/test.csv` — test data

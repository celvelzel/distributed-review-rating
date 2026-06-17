# Kaggle Gap Closing Strategy — From 0.69931 to Beat 0.62

## TL;DR

> **Quick Summary**: Close 12.8% gap with competitor by trying 5 parallel strategies: transformer fine-tuning, better feature engineering, advanced ensembles, deep tabular models, and pseudo-labeling. Each strategy targets different aspects of the prediction problem.
>
> **Deliverables**:
> - Fine-tuned transformer model for review text
> - Enhanced feature set using underutilized features (sentiment, rating_deviation, product_metadata, char TF-IDF)
> - Advanced ensemble with optimized weights
> - Deep tabular model (TabNet/TabTransformer)
> - Pseudo-labeled augmented training data
> - New Kaggle submission(s) targeting < 0.62
>
> **Estimated Effort**: XL (multiple parallel work streams)
> **Parallel Execution**: YES - 5 waves
> **Critical Path**: Feature engineering → Model training → Ensemble → Submission

---

## Context

### Original Request
User wants to close the 12.8% gap with Kaggle competitor (current: 0.69931, target: 0.62) by trying different technical approaches.

### Interview Summary
**Key Discussions**:
- Current best: 0.69931 using diverse ensemble (LGB=0.09, XGB=0.05, MLP=0.86)
- Competitor: 0.62
- Dataset: ~3M training reviews, 10K test, predict 1-5 star ratings
- Available resources: 1 modest GPU, flexible timeline, Kaggle API works

**Research Findings**:
- Target leakage was main historical problem (now fixed with K-Fold)
- LightGCN embeddings are broken (near-zero norm=0.013)
- MLP predictions are compressed (std=0.858 vs actual 1.42)
- Adversarial validation AUC=0.5235 (no distribution shift)
- Feature importance: avg_rating, user_te, prod_avg_rating dominate
- Available unused features: sentiment.parquet, rating_deviation.parquet, product_metadata.parquet, char TF-IDF

---

## Work Objectives

### Core Objective
Close the 12.8% gap between current best Kaggle score (0.69931) and competitor (0.62) through systematic experimentation with multiple technical approaches.

### Concrete Deliverables
1. Fine-tuned transformer model (BERT/RoBERTa) for review text
2. Enhanced feature set using underutilized features
3. Advanced ensemble with optimized weights
4. Deep tabular model (TabNet/TabTransformer)
5. Pseudo-labeled augmented training data
6. New Kaggle submission(s) targeting < 0.62

### Definition of Done
- [ ] At least one approach achieves Kaggle score < 0.65
- [ ] Best approach achieves Kaggle score < 0.62 (beat competitor)
- [ ] All approaches documented with reproducible code
- [ ] Ensemble combines best approaches for final submission

### Must Have
- Use K-Fold target encoding (avoid leakage)
- Validate with adversarial validation (AUC ~0.5)
- Track all experiments with metrics
- Submit to Kaggle via API
- Document all approaches and results

### Must NOT Have (Guardrails)
- NO target leakage (no full-dataset statistics)
- NO broken features (LightGCN embeddings)
- NO overfitting to local CV (validate with Kaggle)
- NO undocumented experiments
- NO submissions without local validation

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest in code/tests/)
- **Automated tests**: Tests-after (implement first, validate with Kaggle)
- **Framework**: pytest + Kaggle API validation

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.omo/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Model Training**: Verify OOF RMSE < baseline (1.129)
- **Feature Engineering**: Verify features load correctly, no NaN/Inf
- **Ensemble**: Verify weights sum to 1.0, predictions in [1,5]
- **Submission**: Verify Kaggle API submission succeeds, score recorded

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation - Feature Engineering):
├── Task 1: Enhanced Feature Assembly [unspecified-high]
├── Task 2: Character-level TF-IDF Optimization [unspecified-high]
├── Task 3: Sentiment Feature Integration [quick]
├── Task 4: Rating Deviation Feature Integration [quick]
└── Task 5: Product Metadata Feature Integration [quick]

Wave 2 (Model Training - Parallel):
├── Task 6: Transformer Fine-tuning (BERT) [deep]
├── Task 7: TabNet Training [deep]
├── Task 8: TabTransformer Training [deep]
├── Task 9: LightGBM with Enhanced Features [unspecified-high]
├── Task 10: XGBoost with Enhanced Features [unspecified-high]
└── Task 11: CatBoost with Enhanced Features [unspecified-high]

Wave 3 (Ensemble & Optimization):
├── Task 12: Pseudo-labeling [deep]
├── Task 13: Advanced Ensemble with Weight Optimization [unspecified-high]
├── Task 14: Stacking Meta-Learner [unspecified-high]
└── Task 15: Blend Diverse Models [unspecified-high]

Wave 4 (Final Submission):
├── Task 16: Generate Final Submission [quick]
├── Task 17: Kaggle API Submission [quick]
└── Task 18: Results Documentation [writing]

Wave FINAL (Verification):
├── Task F1: Plan Compliance Audit [oracle]
├── Task F2: Code Quality Review [unspecified-high]
├── Task F3: Real Manual QA [unspecified-high]
└── Task F4: Scope Fidelity Check [deep]
-> Present results -> Get explicit user okay
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|------------|--------|
| 1-5 | None | 6-11 |
| 6-11 | 1-5 | 12-15 |
| 12 | 6-11 | 13-15 |
| 13-15 | 6-11 | 16-18 |
| 16-18 | 13-15 | F1-F4 |
| F1-F4 | 16-18 | User OK |

### Agent Dispatch Summary

- **Wave 1**: 5 tasks - T1 → `unspecified-high`, T2 → `unspecified-high`, T3-T5 → `quick`
- **Wave 2**: 6 tasks - T6-T8 → `deep`, T9-T11 → `unspecified-high`
- **Wave 3**: 4 tasks - T12 → `deep`, T13-T15 → `unspecified-high`
- **Wave 4**: 3 tasks - T16-T17 → `quick`, T18 → `writing`
- **FINAL**: 4 tasks - F1 → `oracle`, F2-F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [ ] 1. Enhanced Feature Assembly

  **What to do**:
  - Create `code/features/assemble_enhanced.py` that loads and combines:
    - TF-IDF features (5000-dim) from `artifacts/features/chartfidf_train.npz` and `chartfidf_test.npz`
    - Sentiment features from `artifacts/features/sentiment.parquet`
    - Rating deviation features from `artifacts/features/rating_deviation.parquet`
    - Product metadata from `artifacts/features/product_metadata.parquet`
    - K-Fold user stats from `artifacts/features/user_stats_kfold.parquet`
    - K-Fold product stats from `artifacts/features/product_stats_kfold.parquet`
    - K-Fold category stats from `artifacts/features/category_stats_kfold.parquet`
    - BERT embeddings (768-dim) from `artifacts/features/bert_train.parquet` and `bert_test.parquet`
  - Handle missing values (fill with 0 or median)
  - Save assembled features to `artifacts/features/X_train_enhanced.parquet` and `X_test_enhanced.parquet`
  - Verify feature dimensions and no NaN/Inf

  **Must NOT do**:
  - NO target leakage (no full-dataset statistics)
  - NO LightGCN embeddings (broken)
  - NO features that cause leakage

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2-5)
  - **Blocks**: Tasks 6-11
  - **Blocked By**: None

  **References**:
  - `code/features/assemble_kfold.py` - Reference for K-Fold assembly pattern
  - `code/features/sentiment.py` - Sentiment feature generation
  - `code/features/rating_deviation.py` - Rating deviation feature generation
  - `code/features/product_metadata.py` - Product metadata feature generation
  - `artifacts/features/` - All feature files

  **Acceptance Criteria**:
  - [ ] `artifacts/features/X_train_enhanced.parquet` exists with correct dimensions
  - [ ] `artifacts/features/X_test_enhanced.parquet` exists with correct dimensions
  - [ ] No NaN/Inf values in assembled features
  - [ ] Feature count matches expected (TF-IDF + sentiment + deviation + metadata + stats + BERT)

  **QA Scenarios**:
  ```
  Scenario: Feature assembly produces correct output
    Tool: Bash (Python)
    Preconditions: All feature files exist in artifacts/features/
    Steps:
      1. Run: python code/features/assemble_enhanced.py
      2. Load X_train_enhanced.parquet and check shape
      3. Load X_test_enhanced.parquet and check shape
      4. Verify no NaN/Inf values
    Expected Result: Train shape (3007439, N), Test shape (10000, N), no NaN/Inf
    Evidence: .omo/evidence/task-1-feature-assembly.txt
  ```

  **Commit**: YES
  - Message: `feat(features): add enhanced feature assembly`
  - Files: `code/features/assemble_enhanced.py`

- [ ] 2. Character-level TF-IDF Optimization

  **What to do**:
  - Create `code/features/optimize_chartfidf.py` that:
    - Loads existing char TF-IDF from `artifacts/features/chartfidf_train.npz` and `chartfidf_test.npz`
    - Tests different configurations:
      - ngram_range: (2,4), (2,5), (3,5), (3,6)
      - max_features: 10000, 20000, 50000
      - min_df: 2, 5, 10
      - max_df: 0.9, 0.95, 1.0
    - Evaluates each config with LightGBM (5-fold OOF)
    - Selects best configuration based on OOF RMSE
    - Saves optimized char TF-IDF to `artifacts/features/chartfidf_optimized_train.npz` and `chartfidf_optimized_test.npz`

  **Must NOT do**:
  - NO target leakage
  - NO overfitting to local CV

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3-5)
  - **Blocks**: Tasks 6-11
  - **Blocked By**: None

  **References**:
  - `code/features/text_chartfidf.py` - Existing char TF-IDF implementation
  - `artifacts/features/chartfidf_meta.json` - Current char TF-IDF config
  - `code/models/train_tfidf_optimized.py` - Reference for TF-IDF optimization

  **Acceptance Criteria**:
  - [ ] Best char TF-IDF config identified and documented
  - [ ] Optimized char TF-IDF files saved
  - [ ] OOF RMSE improvement documented

  **QA Scenarios**:
  ```
  Scenario: Char TF-IDF optimization finds better config
    Tool: Bash (Python)
    Preconditions: chartfidf_train.npz and chartfidf_test.npz exist
    Steps:
      1. Run: python code/features/optimize_chartfidf.py
      2. Check output for best config
      3. Verify optimized files exist
    Expected Result: Best config documented, optimized files saved
    Evidence: .omo/evidence/task-2-chartfidf-optimization.txt
  ```

  **Commit**: YES
  - Message: `feat(features): optimize character-level TF-IDF`
  - Files: `code/features/optimize_chartfidf.py`

- [ ] 3. Sentiment Feature Integration

  **What to do**:
  - Create `code/features/integrate_sentiment.py` that:
    - Loads sentiment features from `artifacts/features/sentiment.parquet`
    - Checks for missing values and handles them
    - Creates additional sentiment features:
      - sentiment_diff (VADER - TextBlob)
      - sentiment_abs (absolute sentiment)
      - sentiment_category (positive/negative/neutral)
    - Saves enhanced sentiment features to `artifacts/features/sentiment_enhanced.parquet`

  **Must NOT do**:
  - NO target leakage
  - NO features that leak rating information

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-2, 4-5)
  - **Blocks**: Tasks 6-11
  - **Blocked By**: None

  **References**:
  - `code/features/sentiment.py` - Sentiment feature generation
  - `artifacts/features/sentiment.parquet` - Existing sentiment features

  **Acceptance Criteria**:
  - [ ] Enhanced sentiment features saved
  - [ ] No NaN/Inf values
  - [ ] Feature dimensions documented

  **QA Scenarios**:
  ```
  Scenario: Sentiment features integrated correctly
    Tool: Bash (Python)
    Preconditions: sentiment.parquet exists
    Steps:
      1. Run: python code/features/integrate_sentiment.py
      2. Load sentiment_enhanced.parquet and check shape
      3. Verify no NaN/Inf values
    Expected Result: Enhanced sentiment features saved, no NaN/Inf
    Evidence: .omo/evidence/task-3-sentiment-integration.txt
  ```

  **Commit**: YES
  - Message: `feat(features): integrate enhanced sentiment features`
  - Files: `code/features/integrate_sentiment.py`

- [ ] 4. Rating Deviation Feature Integration

  **What to do**:
  - Create `code/features/integrate_deviation.py` that:
    - Loads rating deviation features from `artifacts/features/rating_deviation.parquet`
    - Checks for missing values and handles them
    - Creates additional deviation features:
      - deviation_squared (deviation^2)
      - deviation_abs (absolute deviation)
      - deviation_category (high/medium/low)
    - Saves enhanced deviation features to `artifacts/features/rating_deviation_enhanced.parquet`

  **Must NOT do**:
  - NO target leakage
  - NO features that leak rating information

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-3, 5)
  - **Blocks**: Tasks 6-11
  - **Blocked By**: None

  **References**:
  - `code/features/rating_deviation.py` - Rating deviation feature generation
  - `artifacts/features/rating_deviation.parquet` - Existing rating deviation features

  **Acceptance Criteria**:
  - [ ] Enhanced rating deviation features saved
  - [ ] No NaN/Inf values
  - [ ] Feature dimensions documented

  **QA Scenarios**:
  ```
  Scenario: Rating deviation features integrated correctly
    Tool: Bash (Python)
    Preconditions: rating_deviation.parquet exists
    Steps:
      1. Run: python code/features/integrate_deviation.py
      2. Load rating_deviation_enhanced.parquet and check shape
      3. Verify no NaN/Inf values
    Expected Result: Enhanced deviation features saved, no NaN/Inf
    Evidence: .omo/evidence/task-4-deviation-integration.txt
  ```

  **Commit**: YES
  - Message: `feat(features): integrate enhanced rating deviation features`
  - Files: `code/features/integrate_deviation.py`

- [ ] 5. Product Metadata Feature Integration

  **What to do**:
  - Create `code/features/integrate_metadata.py` that:
    - Loads product metadata from `artifacts/features/product_metadata.parquet`
    - Checks for missing values and handles them
    - Creates additional metadata features:
      - feature_count (number of product features)
      - has_features (binary: has features or not)
      - store_encoded (encoded store name)
      - category_encoded (encoded main_category)
    - Saves enhanced metadata features to `artifacts/features/product_metadata_enhanced.parquet`

  **Must NOT do**:
  - NO target leakage
  - NO features that leak rating information

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-4)
  - **Blocks**: Tasks 6-11
  - **Blocked By**: None

  **References**:
  - `code/features/product_metadata.py` - Product metadata feature generation
  - `artifacts/features/product_metadata.parquet` - Existing product metadata features

  **Acceptance Criteria**:
  - [ ] Enhanced product metadata features saved
  - [ ] No NaN/Inf values
  - [ ] Feature dimensions documented

  **QA Scenarios**:
  ```
  Scenario: Product metadata features integrated correctly
    Tool: Bash (Python)
    Preconditions: product_metadata.parquet exists
    Steps:
      1. Run: python code/features/integrate_metadata.py
      2. Load product_metadata_enhanced.parquet and check shape
      3. Verify no NaN/Inf values
    Expected Result: Enhanced metadata features saved, no NaN/Inf
    Evidence: .omo/evidence/task-5-metadata-integration.txt
  ```

  **Commit**: YES
  - Message: `feat(features): integrate enhanced product metadata features`
  - Files: `code/features/integrate_metadata.py`

- [ ] 6. Transformer Fine-tuning (BERT)

  **What to do**:
  - Create `code/models/finetune_bert.py` that:
    - Loads review text from `data/train.csv` (title + comment)
    - Uses pre-trained `bert-base-uncased` or `roberta-base`
    - Fine-tunes with:
      - Learning rate: 2e-5, 3e-5, 5e-5
      - Batch size: 16, 32
      - Epochs: 3, 4, 5
      - Warmup: 10% of steps
      - Weight decay: 0.01
    - Uses 5-fold cross-validation
    - Saves OOF predictions to `artifacts/models/bert_finetuned_oof.npy`
    - Saves test predictions to `artifacts/models/bert_finetuned_test.npy`
    - Records best hyperparameters and OOF RMSE

  **Must NOT do**:
  - NO target leakage
  - NO overfitting (use proper validation)
  - NO using LightGCN embeddings

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7-11)
  - **Blocks**: Tasks 12-15
  - **Blocked By**: Tasks 1-5

  **References**:
  - `code/models/transformer_finetune.py` - Existing transformer fine-tuning code
  - `code/models/mlp.py` - MLP architecture reference
  - `artifacts/features/bert_train.parquet` - Existing BERT embeddings
  - `data/train.csv` - Training data with text

  **Acceptance Criteria**:
  - [ ] Fine-tuned BERT model saved
  - [ ] OOF RMSE < 1.10 (improvement over baseline 1.129)
  - [ ] Test predictions saved
  - [ ] Best hyperparameters documented

  **QA Scenarios**:
  ```
  Scenario: BERT fine-tuning achieves improvement
    Tool: Bash (Python)
    Preconditions: data/train.csv exists, GPU available
    Steps:
      1. Run: python code/models/finetune_bert.py
      2. Check OOF RMSE in output
      3. Verify bert_finetuned_oof.npy and bert_finetuned_test.npy exist
    Expected Result: OOF RMSE < 1.10, prediction files saved
    Evidence: .omo/evidence/task-6-bert-finetuning.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add BERT fine-tuning for review text`
  - Files: `code/models/finetune_bert.py`

- [ ] 7. TabNet Training

  **What to do**:
  - Create `code/models/train_tabnet.py` that:
    - Loads enhanced features from `artifacts/features/X_train_enhanced.parquet`
    - Implements TabNet architecture:
      - Input dim: feature count from enhanced features
      - Output dim: 1 (regression)
      - n_d: 8, 16, 32
      - n_a: 8, 16, 32
      - n_steps: 3, 5, 7
      - gamma: 1.0, 1.5, 2.0
      - lambda_sparse: 1e-3, 1e-4, 1e-5
    - Uses 5-fold cross-validation
    - Saves OOF predictions to `artifacts/models/tabnet_oof.npy`
    - Saves test predictions to `artifacts/models/tabnet_test.npy`
    - Records best hyperparameters and OOF RMSE

  **Must NOT do**:
  - NO target leakage
  - NO overfitting

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 8-11)
  - **Blocks**: Tasks 12-15
  - **Blocked By**: Tasks 1-5

  **References**:
  - `pytorch-tabnet` library documentation
  - `artifacts/features/X_train_enhanced.parquet` - Enhanced features
  - `code/models/mlp.py` - Neural network training reference

  **Acceptance Criteria**:
  - [ ] TabNet model trained and saved
  - [ ] OOF RMSE < 1.10
  - [ ] Test predictions saved
  - [ ] Best hyperparameters documented

  **QA Scenarios**:
  ```
  Scenario: TabNet achieves improvement
    Tool: Bash (Python)
    Preconditions: X_train_enhanced.parquet exists
    Steps:
      1. Run: python code/models/train_tabnet.py
      2. Check OOF RMSE in output
      3. Verify tabnet_oof.npy and tabnet_test.npy exist
    Expected Result: OOF RMSE < 1.10, prediction files saved
    Evidence: .omo/evidence/task-7-tabnet-training.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add TabNet training for tabular data`
  - Files: `code/models/train_tabnet.py`

- [ ] 8. TabTransformer Training

  **What to do**:
  - Create `code/models/train_tabtransformer.py` that:
    - Loads enhanced features from `artifacts/features/X_train_enhanced.parquet`
    - Implements TabTransformer architecture:
      - Input dim: feature count from enhanced features
      - Output dim: 1 (regression)
      - n_heads: 4, 8
      - n_layers: 2, 3, 4
      - dim: 32, 64, 128
      - dropout: 0.1, 0.2, 0.3
    - Uses 5-fold cross-validation
    - Saves OOF predictions to `artifacts/models/tabtransformer_oof.npy`
    - Saves test predictions to `artifacts/models/tabtransformer_test.npy`
    - Records best hyperparameters and OOF RMSE

  **Must NOT do**:
  - NO target leakage
  - NO overfitting

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6-7, 9-11)
  - **Blocks**: Tasks 12-15
  - **Blocked By**: Tasks 1-5

  **References**:
  - `pytorch-tabular` or `tab-transformer-pytorch` library
  - `artifacts/features/X_train_enhanced.parquet` - Enhanced features
  - `code/models/mlp.py` - Neural network training reference

  **Acceptance Criteria**:
  - [ ] TabTransformer model trained and saved
  - [ ] OOF RMSE < 1.10
  - [ ] Test predictions saved
  - [ ] Best hyperparameters documented

  **QA Scenarios**:
  ```
  Scenario: TabTransformer achieves improvement
    Tool: Bash (Python)
    Preconditions: X_train_enhanced.parquet exists
    Steps:
      1. Run: python code/models/train_tabtransformer.py
      2. Check OOF RMSE in output
      3. Verify tabtransformer_oof.npy and tabtransformer_test.npy exist
    Expected Result: OOF RMSE < 1.10, prediction files saved
    Evidence: .omo/evidence/task-8-tabtransformer-training.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add TabTransformer training for tabular data`
  - Files: `code/models/train_tabtransformer.py`

- [ ] 9. LightGBM with Enhanced Features

  **What to do**:
  - Create `code/models/train_lgb_enhanced.py` that:
    - Loads enhanced features from `artifacts/features/X_train_enhanced.parquet`
    - Uses Optuna for hyperparameter optimization:
      - num_leaves: 31-255
      - learning_rate: 0.01-0.1
      - n_estimators: 200-1000
      - subsample: 0.6-1.0
      - colsample_bytree: 0.6-1.0
      - reg_alpha: 0-1.0
      - reg_lambda: 0-1.0
    - Uses 5-fold cross-validation
    - Saves OOF predictions to `artifacts/models/lgb_enhanced_oof.npy`
    - Saves test predictions to `artifacts/models/lgb_enhanced_test.npy`
    - Records best hyperparameters and OOF RMSE

  **Must NOT do**:
  - NO target leakage
  - NO overfitting

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6-8, 10-11)
  - **Blocks**: Tasks 12-15
  - **Blocked By**: Tasks 1-5

  **References**:
  - `code/models/train_tfidf_optimized.py` - LightGBM optimization reference
  - `code/models/optuna_tune.py` - Optuna tuning reference
  - `artifacts/features/X_train_enhanced.parquet` - Enhanced features

  **Acceptance Criteria**:
  - [ ] LightGBM model trained with optimized hyperparameters
  - [ ] OOF RMSE < 1.10
  - [ ] Test predictions saved
  - [ ] Best hyperparameters documented

  **QA Scenarios**:
  ```
  Scenario: LightGBM with enhanced features achieves improvement
    Tool: Bash (Python)
    Preconditions: X_train_enhanced.parquet exists
    Steps:
      1. Run: python code/models/train_lgb_enhanced.py
      2. Check OOF RMSE in output
      3. Verify lgb_enhanced_oof.npy and lgb_enhanced_test.npy exist
    Expected Result: OOF RMSE < 1.10, prediction files saved
    Evidence: .omo/evidence/task-9-lgb-enhanced.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add LightGBM with enhanced features`
  - Files: `code/models/train_lgb_enhanced.py`

- [ ] 10. XGBoost with Enhanced Features

  **What to do**:
  - Create `code/models/train_xgb_enhanced.py` that:
    - Loads enhanced features from `artifacts/features/X_train_enhanced.parquet`
    - Uses Optuna for hyperparameter optimization:
      - max_depth: 3-10
      - learning_rate: 0.01-0.1
      - n_estimators: 200-1000
      - subsample: 0.6-1.0
      - colsample_bytree: 0.6-1.0
      - reg_alpha: 0-1.0
      - reg_lambda: 0-1.0
    - Uses 5-fold cross-validation
    - Saves OOF predictions to `artifacts/models/xgb_enhanced_oof.npy`
    - Saves test predictions to `artifacts/models/xgb_enhanced_test.npy`
    - Records best hyperparameters and OOF RMSE

  **Must NOT do**:
  - NO target leakage
  - NO overfitting

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6-9, 11)
  - **Blocks**: Tasks 12-15
  - **Blocked By**: Tasks 1-5

  **References**:
  - `code/models/xgboost_train.py` - XGBoost training reference
  - `code/models/optuna_tune.py` - Optuna tuning reference
  - `artifacts/features/X_train_enhanced.parquet` - Enhanced features

  **Acceptance Criteria**:
  - [ ] XGBoost model trained with optimized hyperparameters
  - [ ] OOF RMSE < 1.10
  - [ ] Test predictions saved
  - [ ] Best hyperparameters documented

  **QA Scenarios**:
  ```
  Scenario: XGBoost with enhanced features achieves improvement
    Tool: Bash (Python)
    Preconditions: X_train_enhanced.parquet exists
    Steps:
      1. Run: python code/models/train_xgb_enhanced.py
      2. Check OOF RMSE in output
      3. Verify xgb_enhanced_oof.npy and xgb_enhanced_test.npy exist
    Expected Result: OOF RMSE < 1.10, prediction files saved
    Evidence: .omo/evidence/task-10-xgb-enhanced.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add XGBoost with enhanced features`
  - Files: `code/models/train_xgb_enhanced.py`

- [ ] 11. CatBoost with Enhanced Features

  **What to do**:
  - Create `code/models/train_catboost_enhanced.py` that:
    - Loads enhanced features from `artifacts/features/X_train_enhanced.parquet`
    - Uses Optuna for hyperparameter optimization:
      - depth: 4-10
      - learning_rate: 0.01-0.1
      - iterations: 200-1000
      - l2_leaf_reg: 1-10
      - border_count: 32-255
    - Uses 5-fold cross-validation
    - Saves OOF predictions to `artifacts/models/catboost_enhanced_oof.npy`
    - Saves test predictions to `artifacts/models/catboost_enhanced_test.npy`
    - Records best hyperparameters and OOF RMSE

  **Must NOT do**:
  - NO target leakage
  - NO overfitting

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6-10)
  - **Blocks**: Tasks 12-15
  - **Blocked By**: Tasks 1-5

  **References**:
  - `code/models/catboost_train.py` - CatBoost training reference
  - `code/models/train_catboost_kfold.py` - CatBoost K-Fold reference
  - `artifacts/features/X_train_enhanced.parquet` - Enhanced features

  **Acceptance Criteria**:
  - [ ] CatBoost model trained with optimized hyperparameters
  - [ ] OOF RMSE < 1.10
  - [ ] Test predictions saved
  - [ ] Best hyperparameters documented

  **QA Scenarios**:
  ```
  Scenario: CatBoost with enhanced features achieves improvement
    Tool: Bash (Python)
    Preconditions: X_train_enhanced.parquet exists
    Steps:
      1. Run: python code/models/train_catboost_enhanced.py
      2. Check OOF RMSE in output
      3. Verify catboost_enhanced_oof.npy and catboost_enhanced_test.npy exist
    Expected Result: OOF RMSE < 1.10, prediction files saved
    Evidence: .omo/evidence/task-11-catboost-enhanced.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add CatBoost with enhanced features`
  - Files: `code/models/train_catboost_enhanced.py`

- [ ] 12. Pseudo-labeling

  **What to do**:
  - Create `code/models/pseudo_labeling.py` that:
    - Loads OOF predictions from all trained models (Tasks 6-11)
    - Identifies confident predictions (e.g., prediction std < threshold)
    - Creates pseudo-labeled training data:
      - Use confident predictions as additional training samples
      - Weight pseudo-labels by confidence
    - Trains models on augmented dataset (original + pseudo-labeled)
    - Saves augmented OOF predictions to `artifacts/models/pseudo_labeled_oof.npy`
    - Saves augmented test predictions to `artifacts/models/pseudo_labeled_test.npy`

  **Must NOT do**:
  - NO target leakage
  - NO using pseudo-labels from test set (only from train OOF)

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 13-15)
  - **Blocks**: Tasks 16-18
  - **Blocked By**: Tasks 6-11

  **References**:
  - Pseudo-labeling literature and best practices
  - `artifacts/models/*_oof.npy` - OOF predictions from trained models

  **Acceptance Criteria**:
  - [ ] Pseudo-labeled dataset created
  - [ ] Models retrained on augmented data
  - [ ] OOF RMSE improvement documented
  - [ ] Test predictions saved

  **QA Scenarios**:
  ```
  Scenario: Pseudo-labeling improves performance
    Tool: Bash (Python)
    Preconditions: OOF predictions from Tasks 6-11 exist
    Steps:
      1. Run: python code/models/pseudo_labeling.py
      2. Check OOF RMSE improvement
      3. Verify pseudo_labeled_oof.npy and pseudo_labeled_test.npy exist
    Expected Result: OOF RMSE improvement documented, prediction files saved
    Evidence: .omo/evidence/task-12-pseudo-labeling.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add pseudo-labeling for training augmentation`
  - Files: `code/models/pseudo_labeling.py`

- [ ] 13. Advanced Ensemble with Weight Optimization

  **What to do**:
  - Create `code/models/ensemble_advanced.py` that:
    - Loads OOF predictions from all trained models (Tasks 6-12)
    - Implements multiple ensemble strategies:
      - Simple average
      - Weighted average (optimized weights)
      - Median ensemble
      - Trimmed mean ensemble
    - Uses Optuna for weight optimization:
      - Objective: minimize OOF RMSE
      - Constraints: weights sum to 1.0, each weight >= 0
    - Saves best ensemble OOF to `artifacts/models/ensemble_advanced_oof.npy`
    - Saves best ensemble test to `artifacts/models/ensemble_advanced_test.npy`
    - Records best ensemble strategy and weights

  **Must NOT do**:
  - NO target leakage
  - NO overfitting to local CV

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 12, 14-15)
  - **Blocks**: Tasks 16-18
  - **Blocked By**: Tasks 6-11

  **References**:
  - `code/models/create_ensemble.py` - Existing ensemble creation
  - `code/models/ensemble_diverse.py` - Diverse ensemble reference
  - `artifacts/models/*_oof.npy` - OOF predictions from trained models

  **Acceptance Criteria**:
  - [ ] Best ensemble strategy identified
  - [ ] Optimized weights recorded
  - [ ] OOF RMSE improvement over individual models
  - [ ] Test predictions saved

  **QA Scenarios**:
  ```
  Scenario: Advanced ensemble improves over individual models
    Tool: Bash (Python)
    Preconditions: OOF predictions from Tasks 6-12 exist
    Steps:
      1. Run: python code/models/ensemble_advanced.py
      2. Check best ensemble strategy and weights
      3. Verify ensemble_advanced_oof.npy and ensemble_advanced_test.npy exist
    Expected Result: Best ensemble identified, weights optimized, prediction files saved
    Evidence: .omo/evidence/task-13-ensemble-advanced.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add advanced ensemble with weight optimization`
  - Files: `code/models/ensemble_advanced.py`

- [ ] 14. Stacking Meta-Learner

  **What to do**:
  - Create `code/models/stacking_v2.py` that:
    - Loads OOF predictions from all trained models (Tasks 6-12)
    - Implements stacking with multiple meta-learners:
      - Ridge Regression
      - LightGBM
      - XGBoost
      - Neural Network (small MLP)
    - Uses 5-fold cross-validation for meta-learner training
    - Saves best stacking OOF to `artifacts/models/stacking_v2_oof.npy`
    - Saves best stacking test to `artifacts/models/stacking_v2_test.npy`
    - Records best meta-learner and coefficients

  **Must NOT do**:
  - NO target leakage
  - NO overfitting

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 12-13, 15)
  - **Blocks**: Tasks 16-18
  - **Blocked By**: Tasks 6-11

  **References**:
  - `code/models/stacking.py` - Existing stacking implementation
  - `code/models/run_stacking.py` - Stacking runner
  - `artifacts/models/*_oof.npy` - OOF predictions from trained models

  **Acceptance Criteria**:
  - [ ] Best meta-learner identified
  - [ ] Stacking coefficients recorded
  - [ ] OOF RMSE improvement over individual models
  - [ ] Test predictions saved

  **QA Scenarios**:
  ```
  Scenario: Stacking meta-learner improves over individual models
    Tool: Bash (Python)
    Preconditions: OOF predictions from Tasks 6-12 exist
    Steps:
      1. Run: python code/models/stacking_v2.py
      2. Check best meta-learner and coefficients
      3. Verify stacking_v2_oof.npy and stacking_v2_test.npy exist
    Expected Result: Best meta-learner identified, coefficients recorded, prediction files saved
    Evidence: .omo/evidence/task-14-stacking-v2.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add stacking meta-learner v2`
  - Files: `code/models/stacking_v2.py`

- [ ] 15. Blend Diverse Models

  **What to do**:
  - Create `code/models/blend_diverse.py` that:
    - Loads OOF predictions from all trained models (Tasks 6-12)
    - Implements diverse blending strategies:
      - Blend by model type (tree vs neural)
      - Blend by feature type (TF-IDF vs enhanced)
      - Blend by training strategy (standard vs pseudo-labeled)
    - Uses Optuna for blend weight optimization
    - Saves best blend OOF to `artifacts/models/blend_diverse_oof.npy`
    - Saves best blend test to `artifacts/models/blend_diverse_test.npy`
    - Records best blend strategy and weights

  **Must NOT do**:
  - NO target leakage
  - NO overfitting

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 12-14)
  - **Blocks**: Tasks 16-18
  - **Blocked By**: Tasks 6-11

  **References**:
  - `code/models/ensemble_diverse.py` - Diverse ensemble reference
  - `code/models/create_ensemble.py` - Ensemble creation reference
  - `artifacts/models/*_oof.npy` - OOF predictions from trained models

  **Acceptance Criteria**:
  - [ ] Best blend strategy identified
  - [ ] Optimized blend weights recorded
  - [ ] OOF RMSE improvement over individual models
  - [ ] Test predictions saved

  **QA Scenarios**:
  ```
  Scenario: Diverse blending improves over individual models
    Tool: Bash (Python)
    Preconditions: OOF predictions from Tasks 6-12 exist
    Steps:
      1. Run: python code/models/blend_diverse.py
      2. Check best blend strategy and weights
      3. Verify blend_diverse_oof.npy and blend_diverse_test.npy exist
    Expected Result: Best blend strategy identified, weights optimized, prediction files saved
    Evidence: .omo/evidence/task-15-blend-diverse.txt
  ```

  **Commit**: YES
  - Message: `feat(models): add diverse model blending`
  - Files: `code/models/blend_diverse.py`

- [ ] 16. Generate Final Submission

  **What to do**:
  - Create `code/models/final_submission_v2.py` that:
    - Loads best ensemble/blend predictions from Tasks 13-15
    - Clips predictions to [1, 5]
    - Generates submission CSV with columns: id, rating
    - Saves to `output/submission-final-v2.csv`
    - Records prediction statistics (mean, std, min, max)

  **Must NOT do**:
  - NO predictions outside [1, 5]
  - NO missing IDs

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 17-18)
  - **Blocks**: F1-F4
  - **Blocked By**: Tasks 12-15

  **References**:
  - `code/models/final_submission.py` - Existing final submission script
  - `code/models/predict.py` - Prediction script
  - `artifacts/models/ensemble_advanced_test.npy` - Best ensemble predictions

  **Acceptance Criteria**:
  - [ ] Submission CSV generated with correct format
  - [ ] All predictions in [1, 5]
  - [ ] All IDs present
  - [ ] Prediction statistics recorded

  **QA Scenarios**:
  ```
  Scenario: Final submission CSV is valid
    Tool: Bash (Python)
    Preconditions: Best ensemble predictions exist
    Steps:
      1. Run: python code/models/final_submission_v2.py
      2. Load submission-final-v2.csv and check format
      3. Verify all predictions in [1, 5]
      4. Verify all IDs present
    Expected Result: Valid submission CSV with correct format
    Evidence: .omo/evidence/task-16-final-submission.txt
  ```

  **Commit**: YES
  - Message: `feat(submission): generate final Kaggle submission v2`
  - Files: `code/models/final_submission_v2.py`

- [ ] 17. Kaggle API Submission

  **What to do**:
  - Create `code/kaggle/submit_v2.py` that:
    - Loads submission CSV from `output/submission-final-v2.csv`
    - Submits to Kaggle via API
    - Records submission ID and score
    - Saves submission history to `docs/changelog/kaggle-submissions-v2.md`

  **Must NOT do**:
  - NO invalid submissions
  - NO missing API credentials

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 16, 18)
  - **Blocks**: F1-F4
  - **Blocked By**: Task 16

  **References**:
  - `code/kaggle/` - Existing Kaggle submission scripts
  - Kaggle API documentation

  **Acceptance Criteria**:
  - [ ] Submission succeeds via API
  - [ ] Submission ID recorded
  - [ ] Score recorded when available

  **QA Scenarios**:
  ```
  Scenario: Kaggle API submission succeeds
    Tool: Bash (Python)
    Preconditions: submission-final-v2.csv exists, Kaggle API token valid
    Steps:
      1. Run: python code/kaggle/submit_v2.py
      2. Check output for submission ID
      3. Verify submission recorded in history
    Expected Result: Submission succeeds, ID recorded
    Evidence: .omo/evidence/task-17-kaggle-submission.txt
  ```

  **Commit**: YES
  - Message: `feat(kaggle): submit to Kaggle via API`
  - Files: `code/kaggle/submit_v2.py`

- [ ] 18. Results Documentation

  **What to do**:
  - Create `docs/changelog/kaggle-gap-closing-results.md` that:
    - Documents all approaches tried
    - Records OOF RMSE and Kaggle scores for each approach
    - Compares with baseline (0.69931) and target (0.62)
    - Identifies best approach and why it worked
    - Provides recommendations for future improvements

  **Must NOT do**:
  - NO undocumented experiments
  - NO missing results

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 16-17)
  - **Blocks**: F1-F4
  - **Blocked By**: Tasks 12-15

  **References**:
  - `docs/changelog/optimization-report-2026-06-07.md` - Previous optimization report
  - `docs/changelog/step2-kaggle-scores.md` - Kaggle score history
  - All experiment results from Tasks 6-15

  **Acceptance Criteria**:
  - [ ] All approaches documented
  - [ ] All results recorded
  - [ ] Comparison with baseline and target
  - [ ] Best approach identified
  - [ ] Recommendations provided

  **QA Scenarios**:
  ```
  Scenario: Results documentation is complete
    Tool: Bash (Python)
    Preconditions: All experiment results available
    Steps:
      1. Create docs/changelog/kaggle-gap-closing-results.md
      2. Verify all approaches documented
      3. Verify all results recorded
    Expected Result: Complete documentation of all approaches and results
    Evidence: .omo/evidence/task-18-results-documentation.md
  ```

  **Commit**: YES
  - Message: `docs: add Kaggle gap closing results documentation`
  - Files: `docs/changelog/kaggle-gap-closing-results.md`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, curl endpoint, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .omo/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run linter + tests. Review all changed files for: `as any`/`@ts-ignore`, empty catches, console.log in prod, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names.
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration. Test edge cases: empty state, invalid input, rapid actions. Save to `.omo/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **Wave 1**: `feat(features): add enhanced feature assembly` - code/features/*.py
- **Wave 2**: `feat(models): add transformer/tabnet/tabtransformer training` - code/models/*.py
- **Wave 3**: `feat(ensemble): add advanced ensemble methods` - code/models/*.py
- **Wave 4**: `feat(submission): generate final Kaggle submission` - output/*.csv

---

## Success Criteria

### Verification Commands
```bash
# Run tests
pytest code/tests/ -v

# Generate submission
python code/models/final_submission.py

# Submit to Kaggle
kaggle competitions submit -c comp-5434-2526-sem-3-project -f output/submission-final.csv -m "New approach"

# Check score
kaggle competitions submissions -c comp-5434-2526-sem-3-project
```

### Final Checklist
- [ ] At least one approach achieves Kaggle score < 0.65
- [ ] Best approach achieves Kaggle score < 0.62 (beat competitor)
- [ ] All approaches documented with reproducible code
- [ ] Ensemble combines best approaches for final submission
- [ ] All tests pass
- [ ] No target leakage
- [ ] No broken features used

# Kaggle Performance Optimization Plan

## TL;DR

> **Quick Summary**: Close the 21% gap in Kaggle RMSE (0.79→0.62) by investigating and fixing target leakage, repairing the broken MLP architecture, adding XGBoost for ensemble diversity, and implementing character-level TF-IDF features.
> 
> **Deliverables**:
> - Fixed target leakage in statistical features
> - Repaired MLP architecture with proper training
> - XGBoost model for ensemble diversity
> - Character-level TF-IDF features
> - Diverse ensemble (LGB + XGBoost + MLP)
> - 5-fold OOF validation for all models
> - TDD test coverage for all new code
> - Comprehensive experiment tracking
> 
> **Estimated Effort**: 1 week (full scope)
> **Parallel Execution**: YES - 5 waves
> **Critical Path**: Task 1 (leakage investigation) → Task 4 (fix leakage) → Task 8 (safe features) → Task 12 (ensemble) → Task 15 (final submission)

---

## Context

### Original Request
Maximize Kaggle leaderboard performance for review rating prediction competition. Current best score: 0.79012 RMSE. Competitor score: 0.62 RMSE (21% gap).

### Interview Summary
**Key Discussions**:
- Priority: Close the 21% gap (aggressive approach)
- Neural Models: Fix MLP AND add XGBoost for maximum ensemble diversity
- Time: 1+ week (full scope)
- Test Strategy: TDD (Test-Driven Development)

**Research Findings**:
- Target leakage in statistical features (user_te, prod_te, avg_rating) causes local RMSE=0.545 but Kaggle=1.18-1.59
- Simple TF-IDF + LightGBM is the only approach that generalizes (Kaggle=0.79012)
- MLP architecture is broken (OOF RMSE=1.152)
- XGBoost not yet tried
- Character-level TF-IDF not yet tried
- K-Fold target encoding implementation looks correct but still leaks

### Metis Review
**Identified Gaps** (addressed):
- WHY things are broken not investigated → Added investigation tasks (Tasks 1-3)
- Guardrails not set → Added leakage verification, Kaggle submission budget, CV-Kaggle alignment
- Acceptance criteria missing → Added incremental targets (0.75, 0.70, 0.65)
- Edge cases not addressed → Added cold start, rating distribution, text quality checks
- Assumptions not validated → Added controlled experiments to validate each assumption

---

## Work Objectives

### Core Objective
Close the 21% gap in Kaggle RMSE (0.79→0.62) by fixing target leakage, adding model diversity, and implementing proper validation.

### Concrete Deliverables
- Leakage-free statistical features (user_stats, product_stats, category_stats)
- Fixed MLP architecture (896→512→256→128→1)
- XGBoost model with TF-IDF features
- Character-level TF-IDF features
- Diverse ensemble (LGB + XGBoost + MLP)
- 5-fold OOF validation for all models
- TDD test coverage for all new code
- Comprehensive experiment tracking in metrics.json

### Definition of Done
- [x] Kaggle score < 0.75 (first milestone)
- [x] Kaggle score < 0.70 (second milestone)
- [ ] Kaggle score < 0.65 (final target)
- [x] All models use leakage-free features
- [x] All models validated with 5-fold OOF
- [x] All new code has TDD tests
- [x] All experiments documented in metrics.json

### Must Have
- Leakage-free features only
- 5-fold OOF validation for all models
- TDD test coverage for all new code
- Experiment tracking in metrics.json
- Adversarial validation for distribution shift

### Must NOT Have (Guardrails)
- Target leakage in any feature
- Ensemble of leaky models
- Features without leakage verification
- Kaggle submissions without local CV verification
- More than 20 total Kaggle submissions
- More than 5 Kaggle submissions per day
- Transformer fine-tuning (expensive, may not help)
- Complex stacking with 5+ models
- Post-processing (rounding, clipping, blending)

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** - ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest, 3 test files)
- **Automated tests**: TDD (Test-Driven Development)
- **Framework**: pytest
- **TDD**: Each task follows RED (failing test) → GREEN (minimal impl) → REFACTOR

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.omo/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Feature Code**: Use Bash (pytest) - Run tests, verify no leakage, check shapes
- **Model Code**: Use Bash (pytest) - Run tests, verify OOF RMSE, check predictions
- **Pipeline Code**: Use Bash (python) - Run pipeline, verify outputs, check metrics

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately - investigation + foundation):
├── Task 1: Investigate leakage mechanism [deep]
├── Task 2: Investigate MLP failure mode [deep]
├── Task 3: Run adversarial validation [quick]
├── Task 4: Write leakage verification tests [unspecified-high]
└── Task 5: Write MLP validation tests [unspecified-high]

Wave 2 (After Wave 1 - fix core issues):
├── Task 6: Fix MLP architecture (depends: 2, 5) [deep]
├── Task 7: Fix leakage in feature assembly (depends: 1, 4) [deep]
├── Task 8: Add XGBoost model (depends: 7) [unspecified-high]
└── Task 9: Add character-level TF-IDF (depends: 7) [unspecified-high]

Wave 3 (After Wave 2 - add safe features):
├── Task 10: Add sentiment features (depends: 7) [unspecified-high]
├── Task 11: Add rating deviation features (depends: 7) [unspecified-high]
└── Task 12: Add product metadata features (depends: 7) [unspecified-high]

Wave 4 (After Wave 3 - ensemble):
├── Task 13: Train diverse ensemble (depends: 6, 8, 9) [deep]
├── Task 14: Optimize ensemble weights (depends: 13) [unspecified-high]
└── Task 15: Generate final submission (depends: 14) [quick]

Wave FINAL (After ALL tasks — 4 parallel reviews, then user okay):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: Task 1 → Task 7 → Task 8 → Task 13 → Task 15 → F1-F4 → user okay
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 5 (Wave 1)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|------------|--------|------|
| 1 | - | 4, 7 | 1 |
| 2 | - | 5, 6 | 1 |
| 3 | - | - | 1 |
| 4 | 1 | 7 | 1 |
| 5 | 2 | 6 | 1 |
| 6 | 2, 5 | 13 | 2 |
| 7 | 1, 4 | 8, 9, 10, 11, 12 | 2 |
| 8 | 7 | 13 | 2 |
| 9 | 7 | 13 | 2 |
| 10 | 7 | 13 | 3 |
| 11 | 7 | 13 | 3 |
| 12 | 7 | 13 | 3 |
| 13 | 6, 8, 9 | 14 | 4 |
| 14 | 13 | 15 | 4 |
| 15 | 14 | F1-F4 | 4 |

### Agent Dispatch Summary

- **Wave 1**: 5 tasks - T1-T2 → `deep`, T3 → `quick`, T4-T5 → `unspecified-high`
- **Wave 2**: 4 tasks - T6-T7 → `deep`, T8-T9 → `unspecified-high`
- **Wave 3**: 3 tasks - T10-T12 → `unspecified-high`
- **Wave 4**: 3 tasks - T13 → `deep`, T14 → `unspecified-high`, T15 → `quick`
- **FINAL**: 4 tasks - F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. Investigate Leakage Mechanism

  **What to do**:
  - Read `code/features/assemble.py` and `code/features/assemble_kfold.py` to understand how features are joined
  - Read `code/features/target_encoding.py` to understand K-Fold implementation
  - Read `code/features/user_stats_kfold.py`, `product_stats_kfold.py`, `category_stats_kfold.py`
  - Run a controlled experiment: add ONE feature at a time and measure Kaggle score
  - Document the EXACT leakage mechanism (where is it coming from?)
  - Write a leakage audit report in `docs/changelog/leakage-audit.md`

  **Must NOT do**:
  - Do NOT add new features before understanding leakage
  - Do NOT trust local CV if it doesn't match Kaggle (±5%)
  - Do NOT skip the controlled experiment

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires deep investigation of complex code interactions
  - **Skills**: []
    - No specialized skills needed - pure code analysis

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4, 5)
  - **Blocks**: Tasks 4, 7
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References** (existing code to follow):
  - `code/features/assemble.py` - Feature assembly pipeline (LEAKY version)
  - `code/features/assemble_kfold.py` - Feature assembly with K-Fold stats (leak-safe)
  - `code/features/target_encoding.py` - K-Fold target encoding with Bayesian smoothing

  **API/Type References** (contracts to implement against):
  - `docs/changelog/metrics.json` - Experiment tracking format

  **Test References** (testing patterns to follow):
  - `code/tests/test_target_encoding.py` - Existing leakage test patterns

  **External References** (libraries and frameworks):
  - N/A

  **WHY Each Reference Matters**:
  - `assemble.py` vs `assemble_kfold.py`: Compare to find where leakage is introduced
  - `target_encoding.py`: Verify K-Fold implementation is correct
  - `metrics.json`: Understand what experiments have been tried and their results

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Leakage mechanism identified
    Tool: Bash (python)
    Preconditions: All feature files read
    Steps:
      1. Run controlled experiment: add ONE feature at a time
      2. Measure local CV RMSE for each feature
      3. Compare with Kaggle score for each feature
      4. Identify which features cause CV-Kaggle gap > 20%
    Expected Result: Exact leakage mechanism documented with file:line references
    Failure Indicators: CV-Kaggle gap > 20% for any feature
    Evidence: .omo/evidence/task-1-leakage-mechanism.md

  Scenario: Leakage audit report generated
    Tool: Bash (python)
    Preconditions: Leakage mechanism identified
    Steps:
      1. Write leakage audit report to docs/changelog/leakage-audit.md
      2. Include: which features leak, why they leak, how to fix
      3. Include: controlled experiment results table
    Expected Result: docs/changelog/leakage-audit.md exists with complete analysis
    Failure Indicators: Report missing or incomplete
    Evidence: .omo/evidence/task-1-leakage-audit.md
  ```

  **Commit**: YES
  - Message: `feat(investigate): add leakage investigation tests`
  - Files: `docs/changelog/leakage-audit.md`
  - Pre-commit: N/A

- [x] 2. Investigate MLP Failure Mode

  **What to do**:
  - Read `code/models/mlp.py` and `code/models/run_mlp.py` to understand architecture
  - Check data loading: are features aligned correctly?
  - Check feature quality: what percentage of users/items have valid LightGCN embeddings?
  - Check loss computation: is MSE loss correct for rating prediction?
  - Run a diagnostic: train MLP for 1 epoch and print predictions vs actuals
  - Document findings in `docs/changelog/mlp-diagnosis.md`

  **Must NOT do**:
  - Do NOT fix MLP before understanding WHY it fails
  - Do NOT assume architecture is the problem without evidence
  - Do NOT skip the diagnostic training

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires deep investigation of neural network training dynamics
  - **Skills**: []
    - No specialized skills needed - pure code analysis

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4, 5)
  - **Blocks**: Tasks 5, 6
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References** (existing code to follow):
  - `code/models/mlp.py` - MLP architecture definition
  - `code/models/run_mlp.py` - MLP training script

  **API/Type References** (contracts to implement against):
  - `docs/changelog/metrics.json` - Experiment tracking format

  **Test References** (testing patterns to follow):
  - `code/tests/test_target_encoding.py` - Existing test patterns

  **External References** (libraries and frameworks):
  - PyTorch documentation for MLP architecture

  **WHY Each Reference Matters**:
  - `mlp.py`: Understand current architecture (896→512→128→1)
  - `run_mlp.py`: Understand training setup (batch_size=32768, lr=1e-3)
  - `metrics.json`: Check if MLP has been tried before and what results were

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: MLP failure mode identified
    Tool: Bash (python)
    Preconditions: MLP code read
    Steps:
      1. Run diagnostic: train MLP for 1 epoch
      2. Print predictions vs actuals for first batch
      3. Check if predictions are all same value (not learning)
      4. Check if features are zero-padded (LightGCN embeddings)
    Expected Result: Exact failure mode documented (data bug, architecture bug, or training bug)
    Failure Indicators: Predictions all same, features zero-padded, loss not decreasing
    Evidence: .omo/evidence/task-2-mlp-diagnosis.md

  Scenario: MLP diagnosis report generated
    Tool: Bash (python)
    Preconditions: MLP failure mode identified
    Steps:
      1. Write MLP diagnosis report to docs/changelog/mlp-diagnosis.md
      2. Include: failure mode, root cause, fix strategy
      3. Include: diagnostic training results
    Expected Result: docs/changelog/mlp-diagnosis.md exists with complete analysis
    Failure Indicators: Report missing or incomplete
    Evidence: .omo/evidence/task-2-mlp-diagnosis-report.md
  ```

  **Commit**: YES
  - Message: `feat(investigate): add MLP validation tests`
  - Files: `docs/changelog/mlp-diagnosis.md`
  - Pre-commit: N/A

- [x] 3. Run Adversarial Validation

  **What to do**:
  - Read `code/features/adversarial_validation.py` to understand implementation
  - Run adversarial validation on train vs test sets
  - Calculate AUC score for distribution shift detection
  - If AUC > 0.6, investigate which features cause the shift
  - Document findings in `docs/changelog/adversarial-validation-results.md`

  **Must NOT do**:
  - Do NOT skip adversarial validation
  - Do NOT assume train/test distributions are similar without evidence
  - Do NOT ignore AUC > 0.6

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple script execution and analysis
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4, 5)
  - **Blocks**: None
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References** (existing code to follow):
  - `code/features/adversarial_validation.py` - Existing adversarial validation implementation

  **API/Type References** (contracts to implement against):
  - `docs/changelog/metrics.json` - Experiment tracking format

  **Test References** (testing patterns to follow):
  - N/A

  **External References** (libraries and frameworks):
  - scikit-learn for AUC calculation

  **WHY Each Reference Matters**:
  - `adversarial_validation.py`: Understand how distribution shift is detected
  - `metrics.json`: Record adversarial validation results

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Adversarial validation completed
    Tool: Bash (python)
    Preconditions: adversarial_validation.py exists
    Steps:
      1. Run adversarial validation script
      2. Calculate AUC score
      3. If AUC > 0.6, identify which features cause shift
    Expected Result: AUC score documented, shift features identified if AUC > 0.6
    Failure Indicators: Script fails, AUC > 0.6 without investigation
    Evidence: .omo/evidence/task-3-adversarial-validation.md

  Scenario: Adversarial validation report generated
    Tool: Bash (python)
    Preconditions: Adversarial validation completed
    Steps:
      1. Write results to docs/changelog/adversarial-validation-results.md
      2. Include: AUC score, shift features (if any), recommendations
    Expected Result: docs/changelog/adversarial-validation-results.md exists
    Failure Indicators: Report missing or incomplete
    Evidence: .omo/evidence/task-3-adversarial-validation-report.md
  ```

  **Commit**: YES
  - Message: `feat(validation): run adversarial validation`
  - Files: `docs/changelog/adversarial-validation-results.md`
  - Pre-commit: N/A

- [x] 4. Write Leakage Verification Tests

  **What to do**:
  - Read `code/tests/test_target_encoding.py` to understand existing test patterns
  - Write comprehensive leakage verification tests for all feature types:
    - User stats (user_stats_kfold.py)
    - Product stats (product_stats_kfold.py)
    - Category stats (category_stats_kfold.py)
    - Target encoding (target_encoding.py)
  - Tests should verify:
    - Train features use K-Fold (not full stats)
    - Test features use full train stats (correct behavior)
    - No feature uses its own target value
  - Save tests to `code/tests/test_leakage.py`

  **Must NOT do**:
  - Do NOT write tests without understanding leakage mechanism (Task 1)
  - Do NOT skip any feature type
  - Do NOT write tests that pass even with leakage

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires understanding of leakage patterns and test design
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 5)
  - **Blocks**: Task 7
  - **Blocked By**: Task 1 (needs leakage mechanism understanding)

  **References**:

  **Pattern References** (existing code to follow):
  - `code/tests/test_target_encoding.py` - Existing leakage test patterns
  - `code/features/user_stats_kfold.py` - K-Fold user stats implementation
  - `code/features/product_stats_kfold.py` - K-Fold product stats implementation
  - `code/features/category_stats_kfold.py` - K-Fold category stats implementation

  **API/Type References** (contracts to implement against):
  - `docs/changelog/metrics.json` - Experiment tracking format

  **Test References** (testing patterns to follow):
  - `code/tests/test_target_encoding.py` - Existing test patterns

  **External References** (libraries and frameworks):
  - pytest documentation

  **WHY Each Reference Matters**:
  - `test_target_encoding.py`: Learn existing test patterns for leakage detection
  - `*_kfold.py`: Understand what each feature computes and how K-Fold is applied

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Leakage tests written and passing
    Tool: Bash (pytest)
    Preconditions: Task 1 completed (leakage mechanism understood)
    Steps:
      1. Run pytest code/tests/test_leakage.py
      2. Verify all tests pass
      3. Verify tests cover all feature types
    Expected Result: All leakage tests pass, all feature types covered
    Failure Indicators: Tests fail, missing coverage
    Evidence: .omo/evidence/task-4-leakage-tests.md

  Scenario: Leakage tests catch actual leakage
    Tool: Bash (pytest)
    Preconditions: Leakage tests written
    Steps:
      1. Temporarily introduce leakage in one feature
      2. Run tests
      3. Verify test fails
    Expected Result: Test catches introduced leakage
    Failure Indicators: Test passes despite introduced leakage
    Evidence: .omo/evidence/task-4-leakage-tests-catch.md
  ```

  **Commit**: YES
  - Message: `feat(tests): add leakage verification tests`
  - Files: `code/tests/test_leakage.py`
  - Pre-commit: `pytest code/tests/test_leakage.py`

- [x] 5. Write MLP Validation Tests

  **What to do**:
  - Read `code/tests/test_target_encoding.py` to understand existing test patterns
  - Write comprehensive MLP validation tests:
    - Data loading: features aligned correctly?
    - Feature quality: percentage of valid LightGCN embeddings
    - Architecture: layers connected correctly?
    - Training: loss decreasing?
    - Predictions: not all same value?
  - Save tests to `code/tests/test_mlp.py`

  **Must NOT do**:
  - Do NOT write tests without understanding MLP failure mode (Task 2)
  - Do NOT skip data loading validation
  - Do NOT write tests that pass even with broken MLP

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires understanding of neural network training dynamics
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 4)
  - **Blocks**: Task 6
  - **Blocked By**: Task 2 (needs MLP failure mode understanding)

  **References**:

  **Pattern References** (existing code to follow):
  - `code/tests/test_target_encoding.py` - Existing test patterns
  - `code/models/mlp.py` - MLP architecture definition
  - `code/models/run_mlp.py` - MLP training script

  **API/Type References** (contracts to implement against):
  - `docs/changelog/metrics.json` - Experiment tracking format

  **Test References** (testing patterns to follow):
  - `code/tests/test_target_encoding.py` - Existing test patterns

  **External References** (libraries and frameworks):
  - PyTorch documentation for MLP testing

  **WHY Each Reference Matters**:
  - `test_target_encoding.py`: Learn existing test patterns
  - `mlp.py`: Understand current architecture to test
  - `run_mlp.py`: Understand training setup to test

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: MLP validation tests written and passing
    Tool: Bash (pytest)
    Preconditions: Task 2 completed (MLP failure mode understood)
    Steps:
      1. Run pytest code/tests/test_mlp.py
      2. Verify all tests pass
      3. Verify tests cover data loading, architecture, training
    Expected Result: All MLP validation tests pass
    Failure Indicators: Tests fail, missing coverage
    Evidence: .omo/evidence/task-5-mlp-tests.md

  Scenario: MLP tests catch actual issues
    Tool: Bash (pytest)
    Preconditions: MLP tests written
    Steps:
      1. Temporarily break MLP architecture
      2. Run tests
      3. Verify test fails
    Expected Result: Test catches introduced breakage
    Failure Indicators: Test passes despite broken architecture
    Evidence: .omo/evidence/task-5-mlp-tests-catch.md
  ```

  **Commit**: YES
  - Message: `feat(tests): add MLP validation tests`
  - Files: `code/tests/test_mlp.py`
  - Pre-commit: `pytest code/tests/test_mlp.py`

- [x] 6. Fix MLP Architecture

  **What to do**:
  - Based on Task 2 findings, fix MLP architecture:
    - If data bug: fix data loading and feature alignment
    - If architecture bug: redesign to 896→512→256→128→1 with dropout 0.3-0.5
    - If training bug: fix learning rate, batch size, optimizer
  - Implement proper training:
    - Adam optimizer (lr=1e-3, weight_decay=1e-5)
    - MSE loss
    - Batch size 1024-4096 (not 32768)
    - Cosine annealing LR scheduler
    - Early stopping patience=10
  - Run 5-fold OOF validation
  - Document OOF RMSE and fold variance
  - Update `code/models/mlp.py` and `code/models/run_mlp.py`

  **Must NOT do**:
  - Do NOT use batch_size=32768 (too large, causes poor training)
  - Do NOT skip 5-fold OOF validation
  - Do NOT trust single-fold results

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires deep understanding of neural network training dynamics
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential after Tasks 2, 5)
  - **Blocks**: Task 13
  - **Blocked By**: Tasks 2, 5

  **References**:

  **Pattern References** (existing code to follow):
  - `code/models/mlp.py` - Current MLP architecture (broken)
  - `code/models/run_mlp.py` - Current training script (broken)
  - `code/models/run_baseline.py` - Working training pattern (TF-IDF + LGB)

  **API/Type References** (contracts to implement against):
  - `docs/changelog/metrics.json` - Experiment tracking format

  **Test References** (testing patterns to follow):
  - `code/tests/test_mlp.py` - MLP validation tests (from Task 5)

  **External References** (libraries and frameworks):
  - PyTorch documentation for MLP architecture
  - PyTorch documentation for learning rate schedulers

  **WHY Each Reference Matters**:
  - `mlp.py`: Understand current architecture to fix
  - `run_mlp.py`: Understand current training setup to fix
  - `run_baseline.py`: Learn working training pattern
  - `test_mlp.py`: Verify fixes pass validation tests

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: MLP trains successfully
    Tool: Bash (python)
    Preconditions: MLP architecture fixed
    Steps:
      1. Run 5-fold OOF validation
      2. Record OOF RMSE for each fold
      3. Check that OOF RMSE < 1.0 (better than predicting mean)
      4. Check that fold variance < 0.05
    Expected Result: OOF RMSE < 1.0, fold variance < 0.05
    Failure Indicators: OOF RMSE > 1.0, fold variance > 0.05
    Evidence: .omo/evidence/task-6-mlp-training.md

  Scenario: MLP predictions are diverse
    Tool: Bash (python)
    Preconditions: MLP trained
    Steps:
      1. Generate predictions on test set
      2. Check that predictions are not all same value
      3. Check that prediction distribution matches training distribution
    Expected Result: Predictions diverse, distribution matches training
    Failure Indicators: All predictions same, distribution mismatch
    Evidence: .omo/evidence/task-6-mlp-predictions.md
  ```

  **Commit**: YES
  - Message: `feat(model): fix MLP architecture`
  - Files: `code/models/mlp.py`, `code/models/run_mlp.py`
  - Pre-commit: `pytest code/tests/test_mlp.py`

- [x] 7. Fix Leakage in Feature Assembly

  **What to do**:
  - Based on Task 1 findings, fix leakage in feature assembly:
    - If leakage is in joining: fix join logic in assemble_kfold.py
    - If leakage is in computation: fix K-Fold implementation
    - If leakage is in test features: fix test feature computation
  - Verify fix with leakage tests (Task 4)
  - Run 5-fold OOF validation with fixed features
  - Document OOF RMSE and Kaggle score alignment
  - Update `code/features/assemble_kfold.py`

  **Must NOT do**:
  - Do NOT add new features before fixing leakage
  - Do NOT trust local CV if it doesn't match Kaggle (±5%)
  - Do NOT skip leakage verification tests

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires deep understanding of feature engineering and leakage patterns
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential after Tasks 1, 4)
  - **Blocks**: Tasks 8, 9, 10, 11, 12
  - **Blocked By**: Tasks 1, 4

  **References**:

  **Pattern References** (existing code to follow):
  - `code/features/assemble_kfold.py` - Current feature assembly (may have leakage)
  - `code/features/target_encoding.py` - K-Fold target encoding implementation
  - `code/features/user_stats_kfold.py` - K-Fold user stats implementation
  - `code/features/product_stats_kfold.py` - K-Fold product stats implementation
  - `code/features/category_stats_kfold.py` - K-Fold category stats implementation

  **API/Type References** (contracts to implement against):
  - `docs/changelog/metrics.json` - Experiment tracking format

  **Test References** (testing patterns to follow):
  - `code/tests/test_leakage.py` - Leakage verification tests (from Task 4)

  **External References** (libraries and frameworks):
  - N/A

  **WHY Each Reference Matters**:
  - `assemble_kfold.py`: Current assembly pipeline to fix
  - `*_kfold.py`: Understand K-Fold implementation to verify
  - `test_leakage.py`: Verify fixes pass leakage tests

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Leakage fixed and verified
    Tool: Bash (pytest)
    Preconditions: Task 1 completed (leakage mechanism understood)
    Steps:
      1. Run pytest code/tests/test_leakage.py
      2. Verify all tests pass
      3. Run 5-fold OOF validation with fixed features
      4. Compare OOF RMSE with Kaggle score
      5. Verify CV-Kaggle gap < 5%
    Expected Result: All leakage tests pass, CV-Kaggle gap < 5%
    Failure Indicators: Tests fail, CV-Kaggle gap > 5%
    Evidence: .omo/evidence/task-7-leakage-fixed.md

  Scenario: Fixed features improve Kaggle score
    Tool: Bash (python)
    Preconditions: Leakage fixed
    Steps:
      1. Train LightGBM with fixed features
      2. Generate submission
      3. Submit to Kaggle
      4. Compare with baseline (0.79012)
    Expected Result: Kaggle score < 0.79 (improvement)
    Failure Indicators: Kaggle score > 0.79 (no improvement)
    Evidence: .omo/evidence/task-7-kaggle-improvement.md
  ```

  **Commit**: YES
  - Message: `feat(features): fix leakage in feature assembly`
  - Files: `code/features/assemble_kfold.py`
  - Pre-commit: `pytest code/tests/test_leakage.py`

- [x] 8. Add XGBoost Model

  **What to do**:
  - Create `code/models/xgboost_train.py` following LightGBM pattern
  - Use same TF-IDF features as best model (5000-dim)
  - Use same 5-fold OOF validation
  - Tune hyperparameters with Optuna (30 trials)
  - Record OOF RMSE and fold variance
  - Compare with LightGBM OOF RMSE
  - Document findings in `docs/changelog/xgboost-training.md`

  **Must NOT do**:
  - Do NOT use leaky features
  - Do NOT skip 5-fold OOF validation
  - Do NOT trust single-fold results

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Standard model training with hyperparameter tuning
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 9)
  - **Blocks**: Task 13
  - **Blocked By**: Task 7 (needs fixed features)

  **References**:

  **Pattern References** (existing code to follow):
  - `code/models/run_baseline.py` - Working LightGBM training pattern
  - `code/models/train_tfidf_optimized.py` - TF-IDF optimization pattern
  - `code/models/optuna_tune.py` - Optuna hyperparameter tuning pattern

  **API/Type References** (contracts to implement against):
  - `docs/changelog/metrics.json` - Experiment tracking format

  **Test References** (testing patterns to follow):
  - `code/tests/test_target_encoding.py` - Existing test patterns

  **External References** (libraries and frameworks):
  - XGBoost documentation
  - Optuna documentation

  **WHY Each Reference Matters**:
  - `run_baseline.py`: Learn working training pattern
  - `train_tfidf_optimized.py`: Learn TF-IDF optimization
  - `optuna_tune.py`: Learn Optuna tuning pattern

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: XGBoost trains successfully
    Tool: Bash (python)
    Preconditions: XGBoost model created
    Steps:
      1. Run 5-fold OOF validation
      2. Record OOF RMSE for each fold
      3. Check that OOF RMSE < 1.0
      4. Check that fold variance < 0.05
    Expected Result: OOF RMSE < 1.0, fold variance < 0.05
    Failure Indicators: OOF RMSE > 1.0, fold variance > 0.05
    Evidence: .omo/evidence/task-8-xgboost-training.md

  Scenario: XGBoost predictions are diverse from LightGBM
    Tool: Bash (python)
    Preconditions: XGBoost trained
    Steps:
      1. Generate XGBoost predictions on test set
      2. Generate LightGBM predictions on test set
      3. Calculate correlation between predictions
      4. Check that correlation < 0.95 (diverse)
    Expected Result: Correlation < 0.95 (diverse predictions)
    Failure Indicators: Correlation > 0.95 (similar predictions)
    Evidence: .omo/evidence/task-8-xgboost-diversity.md
  ```

  **Commit**: YES
  - Message: `feat(model): add XGBoost model`
  - Files: `code/models/xgboost_train.py`, `docs/changelog/xgboost-training.md`
  - Pre-commit: `pytest code/tests/`

- [x] 9. Add Character-level TF-IDF

  **What to do**:
  - Create `code/features/text_chartfidf.py` following text_bert.py pattern
  - Implement character-level TF-IDF with:
    - analyzer='char_wb'
    - ngram_range=(3, 5)
    - max_features=5000
    - sublinear_tf=True
  - Integrate into feature assembly pipeline
  - Run 5-fold OOF validation
  - Compare with word-level TF-IDF OOF RMSE
  - Document findings in `docs/changelog/chartfidf-results.md`

  **Must NOT do**:
  - Do NOT use leaky features
  - Do NOT skip 5-fold OOF validation
  - Do NOT assume character TF-IDF helps without evidence

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Standard feature engineering with validation
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 6, 8)
  - **Blocks**: Task 13
  - **Blocked By**: Task 7 (needs fixed features)

  **References**:

  **Pattern References** (existing code to follow):
  - `code/features/text_bert.py` - Text embedding pattern
  - `code/features/assemble_kfold.py` - Feature assembly pattern

  **API/Type References** (contracts to implement against):
  - `docs/changelog/metrics.json` - Experiment tracking format

  **Test References** (testing patterns to follow):
  - `code/tests/test_target_encoding.py` - Existing test patterns

  **External References** (libraries and frameworks):
  - scikit-learn TfidfVectorizer documentation

  **WHY Each Reference Matters**:
  - `text_bert.py`: Learn text embedding pattern
  - `assemble_kfold.py`: Learn how to integrate new features

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Character TF-IDF improves performance
    Tool: Bash (python)
    Preconditions: Character TF-IDF implemented
    Steps:
      1. Run 5-fold OOF validation with character TF-IDF
      2. Compare with word-level TF-IDF OOF RMSE
      3. Check that OOF RMSE improves by > 0.01
    Expected Result: OOF RMSE improves by > 0.01
    Failure Indicators: No improvement or degradation
    Evidence: .omo/evidence/task-9-chartfidf-improvement.md

  Scenario: Character TF-IDF features are valid
    Tool: Bash (pytest)
    Preconditions: Character TF-IDF implemented
    Steps:
      1. Run leakage tests
      2. Verify features are not leaky
      3. Check feature shapes and types
    Expected Result: All tests pass, features valid
    Failure Indicators: Tests fail, invalid features
    Evidence: .omo/evidence/task-9-chartfidf-validation.md
  ```

  **Commit**: YES
  - Message: `feat(features): add character-level TF-IDF`
  - Files: `code/features/text_chartfidf.py`, `docs/changelog/chartfidf-results.md`
  - Pre-commit: `pytest code/tests/`

- [x] 10. Add Sentiment Features

  **What to do**:
  - Create `code/features/sentiment.py` following text_length.py pattern
  - Implement sentiment features:
    - VADER sentiment scores (positive, negative, neutral, compound)
    - TextBlob polarity and subjectivity
    - Positive/negative word counts
    - Title-comment sentiment agreement
  - Integrate into feature assembly pipeline
  - Run 5-fold OOF validation
  - Compare with baseline OOF RMSE
  - Document findings in `docs/changelog/sentiment-results.md`

  **Must NOT do**:
  - Do NOT use leaky features
  - Do NOT skip 5-fold OOF validation
  - Do NOT assume sentiment helps without evidence

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Standard feature engineering with validation
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 11, 12)
  - **Blocks**: Task 13
  - **Blocked By**: Task 7 (needs fixed features)

  **References**:

  **Pattern References** (existing code to follow):
  - `code/features/text_length.py` - Text feature pattern
  - `code/features/assemble_kfold.py` - Feature assembly pattern

  **API/Type References** (contracts to implement against):
  - `docs/changelog/metrics.json` - Experiment tracking format

  **Test References** (testing patterns to follow):
  - `code/tests/test_target_encoding.py` - Existing test patterns

  **External References** (libraries and frameworks):
  - VADER documentation
  - TextBlob documentation

  **WHY Each Reference Matters**:
  - `text_length.py`: Learn text feature pattern
  - `assemble_kfold.py`: Learn how to integrate new features

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Sentiment features improve performance
    Tool: Bash (python)
    Preconditions: Sentiment features implemented
    Steps:
      1. Run 5-fold OOF validation with sentiment features
      2. Compare with baseline OOF RMSE
      3. Check that OOF RMSE improves by > 0.005
    Expected Result: OOF RMSE improves by > 0.005
    Failure Indicators: No improvement or degradation
    Evidence: .omo/evidence/task-10-sentiment-improvement.md

  Scenario: Sentiment features are valid
    Tool: Bash (pytest)
    Preconditions: Sentiment features implemented
    Steps:
      1. Run leakage tests
      2. Verify features are not leaky
      3. Check feature shapes and types
    Expected Result: All tests pass, features valid
    Failure Indicators: Tests fail, invalid features
    Evidence: .omo/evidence/task-10-sentiment-validation.md
  ```

  **Commit**: YES
  - Message: `feat(features): add sentiment features`
  - Files: `code/features/sentiment.py`, `docs/changelog/sentiment-results.md`
  - Pre-commit: `pytest code/tests/`

- [x] 11. Add Rating Deviation Features

  **What to do**:
  - Create `code/features/rating_deviation.py` following user_stats_kfold.py pattern
  - Implement rating deviation features:
    - User rating deviation from user mean
    - Product rating deviation from product mean
    - Category rating deviation from category mean
    - User leniency/harshness score
  - Use K-Fold target encoding for safety
  - Integrate into feature assembly pipeline
  - Run 5-fold OOF validation
  - Compare with baseline OOF RMSE
  - Document findings in `docs/changelog/rating-deviation-results.md`

  **Must NOT do**:
  - Do NOT use leaky features
  - Do NOT skip 5-fold OOF validation
  - Do NOT assume rating deviation helps without evidence

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Standard feature engineering with validation
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 10, 12)
  - **Blocks**: Task 13
  - **Blocked By**: Task 7 (needs fixed features)

  **References**:

  **Pattern References** (existing code to follow):
  - `code/features/user_stats_kfold.py` - K-Fold user stats pattern
  - `code/features/product_stats_kfold.py` - K-Fold product stats pattern
  - `code/features/assemble_kfold.py` - Feature assembly pattern

  **API/Type References** (contracts to implement against):
  - `docs/changelog/metrics.json` - Experiment tracking format

  **Test References** (testing patterns to follow):
  - `code/tests/test_leakage.py` - Leakage verification tests

  **External References** (libraries and frameworks):
  - N/A

  **WHY Each Reference Matters**:
  - `*_kfold.py`: Learn K-Fold pattern for safe target encoding
  - `assemble_kfold.py`: Learn how to integrate new features
  - `test_leakage.py`: Verify features are not leaky

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Rating deviation features improve performance
    Tool: Bash (python)
    Preconditions: Rating deviation features implemented
    Steps:
      1. Run 5-fold OOF validation with rating deviation features
      2. Compare with baseline OOF RMSE
      3. Check that OOF RMSE improves by > 0.005
    Expected Result: OOF RMSE improves by > 0.005
    Failure Indicators: No improvement or degradation
    Evidence: .omo/evidence/task-11-rating-deviation-improvement.md

  Scenario: Rating deviation features are leak-free
    Tool: Bash (pytest)
    Preconditions: Rating deviation features implemented
    Steps:
      1. Run leakage tests
      2. Verify features are not leaky
      3. Check that K-Fold encoding is correct
    Expected Result: All tests pass, features leak-free
    Failure Indicators: Tests fail, features leaky
    Evidence: .omo/evidence/task-11-rating-deviation-validation.md
  ```

  **Commit**: YES
  - Message: `feat(features): add rating deviation features`
  - Files: `code/features/rating_deviation.py`, `docs/changelog/rating-deviation-results.md`
  - Pre-commit: `pytest code/tests/`

- [x] 12. Add Product Metadata Features

  **What to do**:
  - Create `code/features/product_metadata.py` following price_features.py pattern
  - Implement product metadata features:
    - Product features list parsing (feature count, feature length)
    - Store/brand features (brand-level average ratings, review counts)
    - Product title embedding (using DeBERTa)
    - Product feature embedding (using DeBERTa)
  - Integrate into feature assembly pipeline
  - Run 5-fold OOF validation
  - Compare with baseline OOF RMSE
  - Document findings in `docs/changelog/product-metadata-results.md`

  **Must NOT do**:
  - Do NOT use leaky features
  - Do NOT skip 5-fold OOF validation
  - Do NOT assume product metadata helps without evidence

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Standard feature engineering with validation
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 10, 11)
  - **Blocks**: Task 13
  - **Blocked By**: Task 7 (needs fixed features)

  **References**:

  **Pattern References** (existing code to follow):
  - `code/features/price_features.py` - Price feature pattern
  - `code/features/assemble_kfold.py` - Feature assembly pattern

  **API/Type References** (contracts to implement against):
  - `docs/changelog/metrics.json` - Experiment tracking format

  **Test References** (testing patterns to follow):
  - `code/tests/test_target_encoding.py` - Existing test patterns

  **External References** (libraries and frameworks):
  - prodInfo.csv schema

  **WHY Each Reference Matters**:
  - `price_features.py`: Learn product metadata feature pattern
  - `assemble_kfold.py`: Learn how to integrate new features
  - `prodInfo.csv`: Understand available product metadata

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Product metadata features improve performance
    Tool: Bash (python)
    Preconditions: Product metadata features implemented
    Steps:
      1. Run 5-fold OOF validation with product metadata features
      2. Compare with baseline OOF RMSE
      3. Check that OOF RMSE improves by > 0.005
    Expected Result: OOF RMSE improves by > 0.005
    Failure Indicators: No improvement or degradation
    Evidence: .omo/evidence/task-12-product-metadata-improvement.md

  Scenario: Product metadata features are valid
    Tool: Bash (pytest)
    Preconditions: Product metadata features implemented
    Steps:
      1. Run leakage tests
      2. Verify features are not leaky
      3. Check feature shapes and types
    Expected Result: All tests pass, features valid
    Failure Indicators: Tests fail, invalid features
    Evidence: .omo/evidence/task-12-product-metadata-validation.md
  ```

  **Commit**: YES
  - Message: `feat(features): add product metadata features`
  - Files: `code/features/product_metadata.py`, `docs/changelog/product-metadata-results.md`
  - Pre-commit: `pytest code/tests/`

- [x] 13. Train Diverse Ensemble

  **What to do**:
  - Create `code/models/ensemble_diverse.py` following create_ensemble.py pattern
  - Implement diverse ensemble:
    - LightGBM (TF-IDF 5K)
    - XGBoost (TF-IDF 5K)
    - MLP (DeBERTa + LightGCN)
    - LightGBM (character TF-IDF)
  - Use simple weighted average (not stacking)
  - Run 5-fold OOF validation
  - Optimize weights with Optuna
  - Record OOF RMSE and fold variance
  - Document findings in `docs/changelog/ensemble-diverse-results.md`

  **Must NOT do**:
  - Do NOT use leaky models
  - Do NOT use complex stacking (simple weighted average only)
  - Do NOT skip 5-fold OOF validation

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires understanding of ensemble diversity and weight optimization
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (sequential after Tasks 6, 8, 9)
  - **Blocks**: Task 14
  - **Blocked By**: Tasks 6, 8, 9

  **References**:

  **Pattern References** (existing code to follow):
  - `code/models/create_ensemble.py` - Ensemble pattern
  - `code/models/run_baseline.py` - Training pattern

  **API/Type References** (contracts to implement against):
  - `docs/changelog/metrics.json` - Experiment tracking format

  **Test References** (testing patterns to follow):
  - `code/tests/test_target_encoding.py` - Existing test patterns

  **External References** (libraries and frameworks):
  - Optuna documentation

  **WHY Each Reference Matters**:
  - `create_ensemble.py`: Learn ensemble pattern
  - `run_baseline.py`: Learn training pattern

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Diverse ensemble improves performance
    Tool: Bash (python)
    Preconditions: All base models trained
    Steps:
      1. Run 5-fold OOF validation with diverse ensemble
      2. Compare with best single model OOF RMSE
      3. Check that OOF RMSE improves by > 0.01
    Expected Result: OOF RMSE improves by > 0.01
    Failure Indicators: No improvement or degradation
    Evidence: .omo/evidence/task-13-ensemble-improvement.md

  Scenario: Ensemble predictions are diverse
    Tool: Bash (python)
    Preconditions: Ensemble trained
    Steps:
      1. Generate ensemble predictions on test set
      2. Calculate correlation between base model predictions
      3. Check that average correlation < 0.9
    Expected Result: Average correlation < 0.9 (diverse predictions)
    Failure Indicators: Average correlation > 0.9 (similar predictions)
    Evidence: .omo/evidence/task-13-ensemble-diversity.md
  ```

  **Commit**: YES
  - Message: `feat(ensemble): train diverse ensemble`
  - Files: `code/models/ensemble_diverse.py`, `docs/changelog/ensemble-diverse-results.md`
  - Pre-commit: `pytest code/tests/`

- [x] 14. Optimize Ensemble Weights

  **What to do**:
  - Create `code/models/ensemble_weights.py` following optuna_tune.py pattern
  - Optimize ensemble weights with Optuna:
    - Search space: weight for each base model (0.0 to 1.0)
    - Objective: minimize OOF RMSE
    - Trials: 100
  - Record optimal weights
  - Run 5-fold OOF validation with optimized weights
  - Compare with equal-weight ensemble
  - Document findings in `docs/changelog/ensemble-weights-results.md`

  **Must NOT do**:
  - Do NOT use leaky models
  - Do NOT skip 5-fold OOF validation
  - Do NOT trust single-fold results

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Standard hyperparameter optimization
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (sequential after Task 13)
  - **Blocks**: Task 15
  - **Blocked By**: Task 13

  **References**:

  **Pattern References** (existing code to follow):
  - `code/models/optuna_tune.py` - Optuna optimization pattern
  - `code/models/ensemble_diverse.py` - Ensemble pattern

  **API/Type References** (contracts to implement against):
  - `docs/changelog/metrics.json` - Experiment tracking format

  **Test References** (testing patterns to follow):
  - `code/tests/test_target_encoding.py` - Existing test patterns

  **External References** (libraries and frameworks):
  - Optuna documentation

  **WHY Each Reference Matters**:
  - `optuna_tune.py`: Learn Optuna optimization pattern
  - `ensemble_diverse.py`: Learn ensemble implementation

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Optimized weights improve performance
    Tool: Bash (python)
    Preconditions: Ensemble trained
    Steps:
      1. Run 5-fold OOF validation with optimized weights
      2. Compare with equal-weight ensemble
      3. Check that OOF RMSE improves by > 0.005
    Expected Result: OOF RMSE improves by > 0.005
    Failure Indicators: No improvement or degradation
    Evidence: .omo/evidence/task-14-weights-improvement.md

  Scenario: Optimized weights are reasonable
    Tool: Bash (python)
    Preconditions: Weights optimized
    Steps:
      1. Print optimized weights
      2. Check that no weight is > 0.8 (not dominated by one model)
      3. Check that no weight is < 0.05 (all models contribute)
    Expected Result: Weights balanced, all models contribute
    Failure Indicators: One model dominates, some models contribute nothing
    Evidence: .omo/evidence/task-14-weights-validation.md
  ```

  **Commit**: YES
  - Message: `feat(ensemble): optimize ensemble weights`
  - Files: `code/models/ensemble_weights.py`, `docs/changelog/ensemble-weights-results.md`
  - Pre-commit: `pytest code/tests/`

- [x] 15. Generate Final Submission

  **What to do**:
  - Create `code/models/final_submission.py` following predict.py pattern
  - Generate final submission with optimized ensemble:
    - Load all base model predictions
    - Apply optimized weights
    - Clip predictions to [1, 5]
    - Round to nearest 0.5 (optional)
  - Validate submission format
  - Submit to Kaggle
  - Record Kaggle score
  - Document findings in `docs/changelog/final-submission.md`

  **Must NOT do**:
  - Do NOT use leaky models
  - Do NOT skip submission validation
  - Do NOT skip Kaggle score recording

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple script execution
  - **Skills**: []
    - No specialized skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (sequential after Task 14)
  - **Blocks**: F1-F4
  - **Blocked By**: Task 14

  **References**:

  **Pattern References** (existing code to follow):
  - `code/models/predict.py` - Submission generation pattern
  - `code/models/ensemble_weights.py` - Ensemble weights

  **API/Type References** (contracts to implement against):
  - `docs/changelog/metrics.json` - Experiment tracking format
  - `data/sampleSubmission.csv` - Submission format

  **Test References** (testing patterns to follow):
  - `code/tests/test_target_encoding.py` - Existing test patterns

  **External References** (libraries and frameworks):
  - Kaggle CLI documentation

  **WHY Each Reference Matters**:
  - `predict.py`: Learn submission generation pattern
  - `ensemble_weights.py`: Load optimized weights
  - `sampleSubmission.csv`: Verify submission format

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Final submission generated
    Tool: Bash (python)
    Preconditions: Ensemble weights optimized
    Steps:
      1. Run final submission script
      2. Verify submission.csv exists
      3. Verify submission format matches sampleSubmission.csv
      4. Verify predictions are in [1, 5]
    Expected Result: submission.csv valid, predictions in [1, 5]
    Failure Indicators: File missing, format wrong, predictions out of range
    Evidence: .omo/evidence/task-15-final-submission.md

  Scenario: Final submission achieves target
    Tool: Bash (kaggle)
    Preconditions: submission.csv generated
    Steps:
      1. Submit to Kaggle
      2. Record Kaggle score
      3. Compare with target (0.75, 0.70, 0.65)
    Expected Result: Kaggle score < 0.75 (first milestone)
    Failure Indicators: Kaggle score > 0.75
    Evidence: .omo/evidence/task-15-kaggle-score.md
  ```

  **Commit**: YES
  - Message: `feat(submission): generate final submission`
  - Files: `code/models/final_submission.py`, `output/submission-final.csv`, `docs/changelog/final-submission.md`
  - Pre-commit: `pytest code/tests/`

---

## Final Verification Wave

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, curl endpoint, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .omo/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `pytest` + linter. Review all changed files for: `as any`/`@ts-ignore`, empty catches, console.log in prod, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names (data/result/item/temp).
  Output: `Tests [PASS/FAIL] | Lint [PASS/FAIL] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration (features working together, not isolation). Test edge cases: empty state, invalid input, rapid actions. Save to `.omo/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination: Task N touching Task M's files. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **1**: `feat(investigate): add leakage investigation tests` - tests/test_leakage.py
- **2**: `feat(investigate): add MLP validation tests` - tests/test_mlp.py
- **3**: `feat(validation): run adversarial validation` - docs/changelog/adversarial-validation.md
- **4**: `feat(features): fix leakage in feature assembly` - code/features/assemble_kfold.py
- **5**: `feat(model): fix MLP architecture` - code/models/mlp.py
- **6**: `feat(model): add XGBoost model` - code/models/xgboost_train.py
- **7**: `feat(features): add character-level TF-IDF` - code/features/text_chartfidf.py
- **8**: `feat(features): add sentiment features` - code/features/sentiment.py
- **9**: `feat(features): add rating deviation features` - code/features/rating_deviation.py
- **10**: `feat(features): add product metadata features` - code/features/product_metadata.py
- **11**: `feat(ensemble): train diverse ensemble` - code/models/ensemble_diverse.py
- **12**: `feat(ensemble): optimize ensemble weights` - code/models/ensemble_weights.py
- **13**: `feat(submission): generate final submission` - code/models/final_submission.py
- **14**: `docs(tracking): update metrics.json` - docs/changelog/metrics.json

---

## Success Criteria

### Verification Commands
```bash
pytest code/tests/  # Expected: all tests pass
python code/models/final_submission.py  # Expected: submission.csv generated
kaggle competitions submit -c comp-5434-2526-sem-3-project -f output/submission-final.csv -m "Final ensemble"  # Expected: score < 0.75
```

### Final Checklist
- [ ] Kaggle score < 0.75 (first milestone)
- [ ] Kaggle score < 0.70 (second milestone)
- [ ] Kaggle score < 0.65 (final target)
- [ ] All models use leakage-free features
- [ ] All models validated with 5-fold OOF
- [ ] All new code has TDD tests
- [ ] All experiments documented in metrics.json
- [ ] No "Must NOT Have" violations

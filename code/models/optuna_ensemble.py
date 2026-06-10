#!/usr/bin/env python
"""Optuna-based ensemble weight optimization.

Combines 6 diverse models using Optuna (1000 trials) to find optimal weights.
Models (all SAFE — no target leakage):
  1. LGB TF-IDF       — OOF ≈ 1.176
  2. XGBoost TF-IDF   — OOF ≈ 1.202
  3. MLP BERT          — OOF ≈ 1.131
  4. LGB Safe Dense    — OOF ≈ 1.224
  5. XGBoost Safe      — OOF ≈ 1.226
  6. CatBoost Safe     — OOF ≈ 1.230

Strategy: Two-phase optimization
  Phase 1: TPE sampler (500 trials) — global exploration
  Phase 2: CMA-ES sampler (500 trials) — local refinement

Search space: weight for each model in [0, 1], normalized to sum=1.
Objective: minimize OOF RMSE.
"""

from __future__ import annotations

import sys
import time
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

# Suppress Optuna logs for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── path setup ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────
MODEL_DIR = ROOT / "artifacts" / "models"
OUTPUT_DIR = ROOT / "output"
FEAT_DIR = ROOT / "artifacts" / "features"
ETL_DIR = ROOT / "artifacts" / "etl"

RANDOM_SEED = 42
N_TRIALS = 1000

# Model OOF/TEST paths — all SAFE (no leakage)
MODELS = {
    "lgb_tfidf": {
        "oof": MODEL_DIR / "lgb_tfidf_oof.npy",
        "test": MODEL_DIR / "lgb_tfidf_test.npy",
    },
    "xgboost": {
        "oof": MODEL_DIR / "xgboost_oof.npy",
        "test": MODEL_DIR / "xgboost_test.npy",
    },
    "mlp": {
        "oof": MODEL_DIR / "mlp_oof.npy",
        "test": MODEL_DIR / "mlp_test.npy",
    },
    "lgb_safe_dense": {
        "oof": MODEL_DIR / "lgb_safe_dense_oof.npy",
        "test": MODEL_DIR / "lgb_safe_dense_test.npy",
    },
    "xgboost_safe": {
        "oof": MODEL_DIR / "xgboost_safe_oof.npy",
        "test": MODEL_DIR / "xgboost_safe_test.npy",
    },
    "catboost_safe": {
        "oof": MODEL_DIR / "catboost_safe_oof.npy",
        "test": MODEL_DIR / "catboost_safe_test.npy",
    },
}

# Output paths
OPTUNA_OOF_PATH = MODEL_DIR / "optuna_ensemble_oof.npy"
OPTUNA_TEST_PATH = MODEL_DIR / "optuna_ensemble_test.npy"
OPTUNA_WEIGHTS_PATH = MODEL_DIR / "optuna_ensemble_weights.json"
SUBMISSION_PATH = OUTPUT_DIR / "submission-optuna-ensemble.csv"


# ══════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════

def load_oof_predictions() -> tuple:
    """Load all OOF predictions and test predictions."""
    oof_dict = {}
    test_dict = {}

    for name, paths in MODELS.items():
        if not paths["oof"].exists():
            log.warning(f"  Skipping {name}: OOF file not found ({paths['oof']})")
            continue
        if not paths["test"].exists():
            log.warning(f"  Skipping {name}: test file not found ({paths['test']})")
            continue

        oof_dict[name] = np.load(str(paths["oof"])).astype(np.float32)
        test_dict[name] = np.load(str(paths["test"])).astype(np.float32)
        log.info(f"  Loaded {name}: OOF={oof_dict[name].shape}, test={test_dict[name].shape}")

    if len(oof_dict) < 2:
        raise ValueError(f"Need at least 2 models, found {len(oof_dict)}")

    return oof_dict, test_dict


def load_y_train() -> np.ndarray:
    """Load training labels."""
    y_train_path = FEAT_DIR / "y_train.npy"
    if y_train_path.exists():
        y_train = np.load(str(y_train_path)).astype(np.float32)
        log.info(f"  y_train: {y_train.shape}  mean={y_train.mean():.3f}")
    else:
        train_df = pd.read_parquet(ETL_DIR / "train.parquet", columns=["rating"])
        y_train = train_df["rating"].values.astype(np.float32)
        del train_df
    return y_train


def load_test_ids() -> np.ndarray:
    """Load test IDs."""
    return pd.read_parquet(ETL_DIR / "test.parquet", columns=["id"])["id"].values


# ══════════════════════════════════════════════════════════════════════
# Optuna Optimization
# ══════════════════════════════════════════════════════════════════════

def create_objective(
    oof_dict: dict,
    y_train: np.ndarray,
    model_names: list,
):
    """Create Optuna objective function."""

    def objective(trial: optuna.Trial) -> float:
        # Suggest weights for each model
        weights = []
        for name in model_names:
            w = trial.suggest_float(f"w_{name}", 0.0, 1.0)
            weights.append(w)

        # Normalize weights to sum to 1.0
        weights = np.array(weights, dtype=np.float64)
        weight_sum = weights.sum()
        if weight_sum < 1e-10:
            return float("inf")  # Avoid division by zero
        weights = weights / weight_sum

        # Compute ensemble OOF predictions
        ensemble_oof = np.zeros_like(y_train, dtype=np.float32)
        for name, w in zip(model_names, weights):
            ensemble_oof += w * oof_dict[name]

        # Clip predictions to valid range
        ensemble_oof = np.clip(ensemble_oof, 1.0, 5.0)

        # Compute RMSE
        rmse = float(np.sqrt(np.mean((ensemble_oof - y_train) ** 2)))

        return rmse

    return objective


def run_optuna_optimization(
    oof_dict: dict,
    y_train: np.ndarray,
    model_names: list,
    n_trials: int = N_TRIALS,
) -> tuple:
    """Run Optuna optimization to find optimal ensemble weights.
    
    Two-phase strategy:
      Phase 1 (TPE): Global exploration with tree-structured Parzen estimator
      Phase 2 (CMA-ES): Local refinement around best solution found
    """
    log.info(f"\n{'='*60}")
    log.info(f"Running Optuna optimization ({n_trials} trials)")
    log.info(f"{'='*60}")

    # Create objective function
    objective = create_objective(oof_dict, y_train, model_names)

    n_phase1 = n_trials // 2  # 500 trials for TPE
    n_phase2 = n_trials - n_phase1  # 500 trials for CMA-ES

    # ── Phase 1: TPE (global exploration) ────────────────────────────
    log.info(f"\n  Phase 1: TPE sampler ({n_phase1} trials)")
    study_tpe = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED, n_startup_trials=50),
        pruner=optuna.pruners.NopPruner(),
    )

    t0 = time.perf_counter()
    study_tpe.optimize(objective, n_trials=n_phase1, show_progress_bar=True)
    t_phase1 = time.perf_counter() - t0
    log.info(f"  Phase 1 done: RMSE={study_tpe.best_value:.5f}  ({t_phase1:.1f}s)")

    # ── Phase 2: TPE with narrow range (local refinement) ──────────────
    log.info(f"\n  Phase 2: TPE narrow search ({n_phase2} trials)")
    # Use best params from Phase 1 and narrow search range
    best_params_tpe = study_tpe.best_params

    # Build search space centered on best params with ±0.15 range
    narrow_ranges = {}
    for name in model_names:
        key = f"w_{name}"
        center = best_params_tpe[key]
        lo = max(0.0, center - 0.15)
        hi = min(1.0, center + 0.15)
        narrow_ranges[key] = (lo, hi)

    def objective_narrow(trial: optuna.Trial) -> float:
        weights = []
        for name in model_names:
            key = f"w_{name}"
            lo, hi = narrow_ranges[key]
            w = trial.suggest_float(key, lo, hi)
            weights.append(w)

        # Normalize weights to sum to 1.0
        weights = np.array(weights, dtype=np.float64)
        weight_sum = weights.sum()
        if weight_sum < 1e-10:
            return float("inf")
        weights = weights / weight_sum

        # Compute ensemble OOF predictions
        ensemble_oof = np.zeros_like(y_train, dtype=np.float32)
        for name, w in zip(model_names, weights):
            ensemble_oof += w * oof_dict[name]
        ensemble_oof = np.clip(ensemble_oof, 1.0, 5.0)

        # Compute RMSE
        rmse = float(np.sqrt(np.mean((ensemble_oof - y_train) ** 2)))
        return rmse

    study_narrow = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED + 1, n_startup_trials=20),
        pruner=optuna.pruners.NopPruner(),
    )

    t0 = time.perf_counter()
    study_narrow.optimize(objective_narrow, n_trials=n_phase2, show_progress_bar=True)
    t_phase2 = time.perf_counter() - t0
    log.info(f"  Phase 2 done: RMSE={study_narrow.best_value:.5f}  ({t_phase2:.1f}s)")

    # ── Compare and select best ──────────────────────────────────────
    if study_tpe.best_value <= study_narrow.best_value:
        best_study = study_tpe
        log.info(f"\n  Winner: Phase 1 (TPE broad)")
    else:
        best_study = study_narrow
        log.info(f"\n  Winner: Phase 2 (TPE narrow)")

    best_trial = best_study.best_trial
    best_rmse = best_trial.value

    # ── Fine grid search refinement ──────────────────────────────────
    log.info(f"\n  Phase 3: Fine grid search refinement")
    best_params_narrow = best_study.best_params
    best_rmse_overall = best_study.best_value
    best_weights_overall = None

    # Grid search around best weights with fine granularity
    center_weights = np.array([best_params_narrow[f"w_{name}"] for name in model_names])
    center_weights = center_weights / center_weights.sum()

    # Precompute model arrays for efficiency
    model_arrays = np.stack([oof_dict[name] for name in model_names])  # (6, N)

    # Fine search: ±0.02 around MLP weight with 0.005 step
    n_grid = 0
    for delta_mlp in np.arange(-0.02, 0.025, 0.005):
        for delta_lgb in np.arange(-0.015, 0.02, 0.005):
            w_mlp = center_weights[2] + delta_mlp
            w_lgb = center_weights[0] + delta_lgb
            w_xgb = center_weights[1]
            w_lgb_safe = center_weights[3] - delta_mlp - delta_lgb

            # Skip invalid weights
            if w_mlp < 0 or w_lgb < 0 or w_xgb < 0 or w_lgb_safe < 0:
                continue
            if w_mlp > 1 or w_lgb > 1 or w_xgb > 1 or w_lgb_safe > 1:
                continue

            weights = np.array([w_lgb, w_xgb, w_mlp, w_lgb_safe,
                                center_weights[4], center_weights[5]])
            weights = weights / weights.sum()

            # Compute RMSE efficiently using vectorized operations
            ensemble_oof = np.dot(weights, model_arrays)
            ensemble_oof = np.clip(ensemble_oof, 1.0, 5.0)
            rmse = float(np.sqrt(np.mean((ensemble_oof - y_train) ** 2)))
            n_grid += 1

            if rmse < best_rmse_overall:
                best_rmse_overall = rmse
                best_weights_overall = weights.copy()

    log.info(f"  Grid search: {n_grid} combinations tested")
    log.info(f"  Best grid RMSE: {best_rmse_overall:.5f}")

    if best_weights_overall is not None:
        best_weights = best_weights_overall
        log.info(f"  Grid search improved weights!")
    else:
        # Use best from optimization
        raw_weights = np.array([best_params_narrow[f"w_{name}"] for name in model_names])
        best_weights = raw_weights / raw_weights.sum()

    best_rmse = best_rmse_overall

    elapsed = t_phase1 + t_phase2
    log.info(f"\n  Optimization completed in {elapsed:.1f}s")
    log.info(f"  Best OOF RMSE: {best_rmse:.5f}")
    log.info(f"  Best weights (normalized):")
    for name, w in zip(model_names, best_weights):
        log.info(f"    {name:20s}: {w:.4f}")

    return best_weights, best_rmse, best_study


# ══════════════════════════════════════════════════════════════════════
# Ensemble Generation
# ══════════════════════════════════════════════════════════════════════

def generate_ensemble_predictions(
    oof_dict: dict,
    test_dict: dict,
    weights: np.ndarray,
    model_names: list,
) -> tuple:
    """Generate ensemble OOF and test predictions."""
    log.info(f"\n{'='*60}")
    log.info("Generating ensemble predictions")
    log.info(f"{'='*60}")

    # Get array size from first model
    n_train = len(next(iter(oof_dict.values())))
    n_test = len(next(iter(test_dict.values())))

    # Compute ensemble OOF
    ensemble_oof = np.zeros(n_train, dtype=np.float32)
    for name, w in zip(model_names, weights):
        ensemble_oof += w * oof_dict[name]
    ensemble_oof = np.clip(ensemble_oof, 1.0, 5.0)

    # Compute ensemble test
    ensemble_test = np.zeros(n_test, dtype=np.float32)
    for name, w in zip(model_names, weights):
        ensemble_test += w * test_dict[name]
    ensemble_test = np.clip(ensemble_test, 1.0, 5.0)

    log.info(f"  Ensemble OOF: {ensemble_oof.shape}  mean={ensemble_oof.mean():.3f}")
    log.info(f"  Ensemble test: {ensemble_test.shape}  mean={ensemble_test.mean():.3f}")

    return ensemble_oof, ensemble_test


# ══════════════════════════════════════════════════════════════════════
# Save Results
# ══════════════════════════════════════════════════════════════════════

def save_results(
    ensemble_oof: np.ndarray,
    ensemble_test: np.ndarray,
    weights: np.ndarray,
    model_names: list,
    best_rmse: float,
    test_ids: np.ndarray,
    study: optuna.Study,
) -> None:
    """Save ensemble predictions, weights, and submission."""
    log.info(f"\n{'='*60}")
    log.info("Saving results")
    log.info(f"{'='*60}")

    # Save OOF predictions
    np.save(str(OPTUNA_OOF_PATH), ensemble_oof)
    log.info(f"  OOF predictions → {OPTUNA_OOF_PATH}")

    # Save test predictions
    np.save(str(OPTUNA_TEST_PATH), ensemble_test)
    log.info(f"  Test predictions → {OPTUNA_TEST_PATH}")

    # Save weights as JSON
    weights_dict = {
        "best_rmse": best_rmse,
        "weights": {name: float(w) for name, w in zip(model_names, weights)},
        "n_trials": N_TRIALS,
        "random_seed": RANDOM_SEED,
    }
    with open(OPTUNA_WEIGHTS_PATH, "w") as f:
        json.dump(weights_dict, f, indent=2)
    log.info(f"  Weights → {OPTUNA_WEIGHTS_PATH}")

    # Save submission CSV
    submission = pd.DataFrame({"id": test_ids, "rating": ensemble_test})
    submission.to_csv(SUBMISSION_PATH, index=False)
    log.info(f"  Submission → {SUBMISSION_PATH}")

    # Print study statistics
    log.info(f"\n  Study statistics:")
    log.info(f"    Total trials: {len(study.trials)}")
    log.info(f"    Best trial: {study.best_trial.number}")
    log.info(f"    Best RMSE: {study.best_value:.5f}")

    # Top 5 trials
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float("inf"))
    log.info(f"\n  Top 5 trials:")
    for i, trial in enumerate(sorted_trials[:5], 1):
        raw_weights = [trial.params[f"w_{name}"] for name in model_names]
        norm_weights = np.array(raw_weights) / sum(raw_weights)
        weights_str = ", ".join(f"{name}={w:.3f}" for name, w in zip(model_names, norm_weights))
        log.info(f"    {i}. Trial {trial.number}: RMSE={trial.value:.5f}  [{weights_str}]")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    t_start = time.perf_counter()
    print("=" * 60)
    print("Optuna Ensemble Weight Optimization")
    print(f"  Trials: {N_TRIALS}")
    print(f"  Models: {len(MODELS)}")
    print("=" * 60)

    # Ensure output directories exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────────
    log.info("\n[1/4] Loading OOF predictions and labels …")
    oof_dict, test_dict = load_oof_predictions()
    y_train = load_y_train()
    test_ids = load_test_ids()

    model_names = list(oof_dict.keys())
    log.info(f"\n  Models: {model_names}")

    # Compute individual model RMSEs
    log.info("\n  Individual model OOF RMSEs:")
    for name in model_names:
        rmse = float(np.sqrt(np.mean((oof_dict[name] - y_train) ** 2)))
        log.info(f"    {name:20s}: {rmse:.5f}")

    # ── 2. Run Optuna optimization ───────────────────────────────────
    best_weights, best_rmse, study = run_optuna_optimization(
        oof_dict, y_train, model_names, n_trials=N_TRIALS,
    )

    # ── 3. Generate ensemble predictions ─────────────────────────────
    ensemble_oof, ensemble_test = generate_ensemble_predictions(
        oof_dict, test_dict, best_weights, model_names,
    )

    # Verify OOF RMSE
    actual_rmse = float(np.sqrt(np.mean((ensemble_oof - y_train) ** 2)))
    log.info(f"\n  Verified OOF RMSE: {actual_rmse:.5f}")

    # ── 4. Save results ──────────────────────────────────────────────
    save_results(
        ensemble_oof, ensemble_test, best_weights, model_names,
        actual_rmse, test_ids, study,
    )

    # ── Summary ──────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_start

    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"  Best OOF RMSE: {actual_rmse:.5f}")
    log.info(f"  Best weights:")
    for name, w in zip(model_names, best_weights):
        log.info(f"    {name:20s}: {w:.4f}")

    # Compare with baseline
    baseline_rmse = 1.129  # Current ensemble (LGB=0.09, XGB=0.05, MLP=0.86)
    delta = baseline_rmse - actual_rmse
    log.info(f"\n  Baseline RMSE: {baseline_rmse:.5f}")
    log.info(f"  Improvement:   {delta:+.5f}  {'✅ improved' if delta > 0 else '⚠️  worse'}")

    if actual_rmse < 1.129:
        log.info(f"\n  ✅ Target OOF RMSE < 1.129 achieved!")
    else:
        log.info(f"\n  ⚠️  OOF RMSE >= 1.129 — consider more trials or models")

    log.info(f"\n  Total time: {total_time:.1f}s")
    log.info(f"  Submission: {SUBMISSION_PATH}")
    log.info("\n=== Done ===")


if __name__ == "__main__":
    main()

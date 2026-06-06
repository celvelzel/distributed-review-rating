#!/usr/bin/env python
"""Optuna hyperparameter tuning for LightGBM (T21).

Optimizes OOF RMSE via 3-fold CV on a 2K-row subsample for speed.
5927 features make each LightGBM iteration expensive; small sample
is necessary to keep total wall time under 45 minutes.

Search space: num_leaves, max_depth, learning_rate, min_child_samples,
              feature_fraction, bagging_fraction.

Outputs: artifacts/models/best_params.json
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pyarrow.parquet as pq
from sklearn.model_selection import KFold

# ── path setup ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.utils.timer import timed, StageTimer  # noqa: F401

# ── constants ──────────────────────────────────────────────────────────
FEAT_DIR = ROOT / "artifacts" / "features"
X_TRAIN_PATH = FEAT_DIR / "X_train.parquet"
Y_TRAIN_PATH = FEAT_DIR / "y_train.npy"

MODELS_DIR = ROOT / "artifacts" / "models"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"

N_TRIALS = 30
N_SPLITS = 3
N_SAMPLE_ROWS = 2_000
RANDOM_SEED = 42
N_ESTIMATORS = 20


# ── data loading ───────────────────────────────────────────────────────
def load_subsample(n_sample: int = N_SAMPLE_ROWS) -> Tuple[np.ndarray, np.ndarray]:
    """Load a random subsample from the first parquet row group."""
    print(f"Loading subsample ({n_sample:,} rows) from parquet …")
    y_all = np.load(str(Y_TRAIN_PATH)).astype(np.float32)

    pf = pq.ParquetFile(str(X_TRAIN_PATH))
    n_total = pf.metadata.num_rows

    # For small samples, load from first row group only
    table = pf.read_row_group(0)
    df = table.to_pandas()
    n_rg0 = len(df)
    del table

    rng = np.random.RandomState(RANDOM_SEED)
    idx = np.sort(rng.choice(n_rg0, size=min(n_sample, n_rg0), replace=False))
    X = df.values[idx].astype(np.float32)
    y = y_all[idx].copy()
    del df
    gc.collect()
    print(f"Subsample ready: X={X.shape}, y={y.shape}")
    return X, y


# ── single-fold train/eval ─────────────────────────────────────────────
def _train_eval(
    params: dict,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
) -> float:
    """Train LightGBM and return clipped RMSE on validation set."""
    ds_tr = lgb.Dataset(X_tr, y_tr, free_raw_data=True)
    ds_va = lgb.Dataset(X_va, y_va, free_raw_data=True)
    model = lgb.train(
        params,
        ds_tr,
        num_boost_round=N_ESTIMATORS,
        valid_sets=[ds_va],
        callbacks=[lgb.log_evaluation(period=0)],
    )
    preds = np.clip(model.predict(X_va), 1.0, 5.0)
    rmse = float(np.sqrt(np.mean((preds - y_va) ** 2)))
    del ds_tr, ds_va, model, preds
    gc.collect()
    return rmse


# ── k-fold CV ──────────────────────────────────────────────────────────
def cv_rmse(params: dict, X: np.ndarray, y: np.ndarray, n_splits: int = N_SPLITS) -> float:
    """Run k-fold CV and return mean RMSE."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    rmses: List[float] = []
    for tr_idx, va_idx in kf.split(X):
        rmse = _train_eval(params, X[tr_idx], y[tr_idx], X[va_idx], y[va_idx])
        rmses.append(rmse)
    return float(np.mean(rmses))


# ── objective function ─────────────────────────────────────────────────
def make_objective(X: np.ndarray, y: np.ndarray):
    """Return an Optuna objective closure over the fixed data."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "seed": RANDOM_SEED,
            "num_leaves": trial.suggest_categorical("num_leaves", [31, 63, 127, 255]),
            "max_depth": trial.suggest_categorical("max_depth", [6, 8, 10, 12]),
            "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.05, 0.1]),
            "min_child_samples": trial.suggest_categorical("min_child_samples", [10, 20, 50]),
            "feature_fraction": trial.suggest_categorical("feature_fraction", [0.6, 0.8, 1.0]),
            "bagging_fraction": trial.suggest_categorical("bagging_fraction", [0.6, 0.8, 1.0]),
            "bagging_freq": 1,
        }
        mean_rmse = cv_rmse(params, X, y)
        print(f"  trial {trial.number:>2d}: RMSE={mean_rmse:.5f}  params={trial.params}")
        return mean_rmse

    return objective


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("Optuna Hyperparameter Tuning for LightGBM (T21)")
    print("=" * 60)
    t_start = time.perf_counter()

    # 1. Load data subsample
    print("\n[1/4] Loading data subsample …")
    X, y = load_subsample(N_SAMPLE_ROWS)

    # 2. Evaluate default baseline
    print("\n[2/4] Evaluating default baseline …")
    default_params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "seed": RANDOM_SEED,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
    }
    default_rmse = cv_rmse(default_params, X, y)
    print(f"  Default RMSE = {default_rmse:.5f}")

    # 3. Run Optuna study
    print(f"\n[3/4] Running Optuna study ({N_TRIALS} trials, {N_SPLITS}-fold CV) …")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        study_name="lgb_rmse_tuning",
    )
    study.optimize(make_objective(X, y), n_trials=N_TRIALS, show_progress_bar=False)

    best_trial = study.best_trial
    best_rmse = best_trial.value
    best_params = best_trial.params

    best_params_full = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "seed": RANDOM_SEED,
        "n_estimators": N_ESTIMATORS,
        **best_params,
    }

    print(f"\nBest trial #{best_trial.number}: RMSE = {best_rmse:.5f}")
    print(f"  Params: {best_params}")

    # 4. Save results
    print("\n[4/4] Saving results …")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    trial_history = []
    for t in study.trials:
        trial_history.append({
            "trial": t.number,
            "rmse": round(t.value, 6) if t.value is not None else None,
            "params": t.params,
            "state": t.state.name,
        })

    output = {
        "best_params": best_params_full,
        "best_rmse": round(best_rmse, 6),
        "default_rmse": round(default_rmse, 6),
        "improvement": round(default_rmse - best_rmse, 6),
        "n_trials": N_TRIALS,
        "n_folds": N_SPLITS,
        "n_sample_rows": N_SAMPLE_ROWS,
        "random_seed": RANDOM_SEED,
        "trial_history": trial_history,
    }

    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(f"  Best params → {BEST_PARAMS_PATH}")

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"  Default RMSE:  {default_rmse:.5f}")
    print(f"  Best RMSE:     {best_rmse:.5f}")
    print(f"  Improvement:   {default_rmse - best_rmse:+.5f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

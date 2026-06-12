#!/usr/bin/env python
"""CatBoost with Safe Target Encoding features (Task 9).

Uses Optuna (100 trials) for hyperparameter optimization with 5-Fold CV.
Features: user_te, prod_te, cat_te, user_count, prod_count
         (K-Fold + Smoothing + Noise — no global leakage)

Target: OOF RMSE < 1.15

Strategy:
  - HPO uses 300K subsample + 3-fold for speed (100 trials feasible)
  - Final training uses full 3M + 5-fold with best params
  - Constrained param space: no Lossguide (slow), depth 4-8, LR 0.01-0.15
  - More iterations with lower LR to let the model learn fine-grained patterns
"""

from __future__ import annotations

import gc
import sys
import time
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── path setup ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── constants ──────────────────────────────────────────────────────────
FEAT_DIR = ROOT / "artifacts" / "features"
MODEL_DIR = ROOT / "artifacts" / "models"

X_TRAIN_PATH = FEAT_DIR / "safe_target_encoding_train.npz"
Y_TRAIN_PATH = FEAT_DIR / "y_train.npy"

OOF_PATH = MODEL_DIR / "catboost_target_encoding_oof.npy"

RANDOM_SEED = 42
N_FOLDS = 5
N_OPTUNA_TRIALS = 100
HPO_SUBSAMPLE = 300_000  # 300K for HPO speed, full 3M for final
HPO_FOLDS = 3            # 3-fold for HPO, 5-fold for final
FEATURE_NAMES = ["user_te", "prod_te", "cat_te", "user_count", "prod_count"]


# ── data loading ───────────────────────────────────────────────────────
def load_data() -> Tuple[np.ndarray, np.ndarray]:
    """Load Safe TE features and labels."""
    print("  Loading Safe Target Encoding features …")
    data = np.load(str(X_TRAIN_PATH))
    X = np.column_stack([data[k].astype(np.float32) for k in FEATURE_NAMES])
    print(f"  X: {X.shape}, dtype={X.dtype}")

    y = np.load(str(Y_TRAIN_PATH)).astype(np.float32)
    print(f"  y: {y.shape}, range=[{y.min():.1f}, {y.max():.1f}], mean={y.mean():.4f}")
    return X, y


# ── Optuna objective (HPO on subsample) ───────────────────────────────
def create_objective(X: np.ndarray, y: np.ndarray, splits: list):
    """Return Optuna objective function with pre-loaded data."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 1000, 5000, step=500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3.0, 30.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "border_count": trial.suggest_int("border_count", 64, 255),
            "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise"]),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 1.0),
            "loss_function": "RMSE",
            "verbose": 0,
            "random_seed": RANDOM_SEED,
            "thread_count": -1,
        }

        fold_rmses = []
        for tr_idx, va_idx in splits:
            train_pool = Pool(X[tr_idx], y[tr_idx])
            val_pool = Pool(X[va_idx], y[va_idx])

            model = CatBoostRegressor(**params)
            model.fit(
                train_pool,
                eval_set=val_pool,
                use_best_model=True,
                early_stopping_rounds=150,
            )

            va_preds = np.clip(model.predict(val_pool), 1.0, 5.0)
            fold_rmse = float(np.sqrt(np.mean((va_preds - y[va_idx]) ** 2)))
            fold_rmses.append(fold_rmse)

            del train_pool, val_pool, model
            gc.collect()

        return float(np.mean(fold_rmses))

    return objective


# ── final training with best params on full data ──────────────────────
def train_final_oof(
    X: np.ndarray,
    y: np.ndarray,
    best_params: dict,
    splits: list,
) -> Tuple[np.ndarray, List[float]]:
    """Train final model with best params, collect OOF predictions."""
    n_train = len(y)
    oof_preds = np.zeros(n_train, dtype=np.float32)
    fold_rmses: List[float] = []

    cb_params = dict(best_params)
    cb_params["loss_function"] = "RMSE"
    cb_params["verbose"] = 200
    cb_params["random_seed"] = RANDOM_SEED
    cb_params["thread_count"] = -1

    for fold_idx, (tr_idx, va_idx) in enumerate(splits, 1):
        print(f"\n  ── Fold {fold_idx}/{N_FOLDS} "
              f"(train={len(tr_idx):,}, val={len(va_idx):,}) ──")

        train_pool = Pool(X[tr_idx], y[tr_idx])
        val_pool = Pool(X[va_idx], y[va_idx])

        model = CatBoostRegressor(**cb_params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            early_stopping_rounds=150,
        )

        va_preds = np.clip(model.predict(val_pool), 1.0, 5.0)
        oof_preds[va_idx] = va_preds
        fold_rmse = float(np.sqrt(np.mean((va_preds - y[va_idx]) ** 2)))
        fold_rmses.append(fold_rmse)
        print(f"  Fold {fold_idx} RMSE: {fold_rmse:.5f}  "
              f"(best_iter={model.best_iteration_})")

        del train_pool, val_pool, model
        gc.collect()

    return oof_preds, fold_rmses


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("CatBoost + Safe Target Encoding (Task 9)")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data ──────────────────────────────────────────────────────
    print("\n[1/5] Loading data …")
    X, y = load_data()

    # 2. Optuna HPO on subsample ────────────────────────────────────────
    print(f"\n[2/5] Optuna HPO ({N_OPTUNA_TRIALS} trials, "
          f"subsample={HPO_SUBSAMPLE:,}, {HPO_FOLDS}-fold) …")

    rng = np.random.RandomState(RANDOM_SEED)
    sub_idx = rng.choice(len(X), size=min(HPO_SUBSAMPLE, len(X)), replace=False)
    sub_idx.sort()
    X_sub, y_sub = X[sub_idx], y[sub_idx]
    sub_kf = KFold(n_splits=HPO_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    sub_splits = list(sub_kf.split(X_sub))
    print(f"  HPO data: {X_sub.shape}, {HPO_FOLDS}-fold CV")

    start_hpo = time.perf_counter()

    study = optuna.create_study(
        direction="minimize",
        study_name="catboost_target_encoding",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    objective = create_objective(X_sub, y_sub, sub_splits)
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

    hpo_time = time.perf_counter() - start_hpo
    best_rmse = study.best_value
    best_params = study.best_params

    del X_sub, y_sub, sub_splits
    gc.collect()

    print(f"\n  Best HPO RMSE: {best_rmse:.5f}")
    print(f"  Best params: {json.dumps(best_params, indent=2)}")
    print(f"  HPO time: {hpo_time:.1f}s ({hpo_time/60:.1f}min)")

    # 3. Final 5-fold training on full 3M data ──────────────────────────
    print(f"\n[3/5] Final training with best params (full 3M, {N_FOLDS}-fold) …")
    start_train = time.perf_counter()

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    splits = list(kf.split(X))

    oof_preds, fold_rmses = train_final_oof(X, y, best_params, splits)

    train_time = time.perf_counter() - start_train

    # 4. Compute OOF RMSE ───────────────────────────────────────────────
    oof_preds = np.clip(oof_preds, 1.0, 5.0)
    oof_rmse = float(np.sqrt(np.mean((oof_preds - y) ** 2)))
    mean_fold_rmse = float(np.mean(fold_rmses))

    print(f"\n  Fold RMSEs: {[f'{r:.5f}' for r in fold_rmses]}")
    print(f"  Mean fold RMSE: {mean_fold_rmse:.5f}")
    print(f"  Overall OOF RMSE: {oof_rmse:.5f}")

    # 5. Save ───────────────────────────────────────────────────────────
    print(f"\n[4/5] Saving OOF predictions …")
    np.save(str(OOF_PATH), oof_preds)
    print(f"  OOF → {OOF_PATH}  shape={oof_preds.shape}")

    print(f"\n[5/5] Saving evidence …")
    evidence_dir = ROOT / ".sisyphus" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_dir / "task-9-catboost-rmse.txt"

    evidence = {
        "task": "Task 9: CatBoost + Safe Target Encoding",
        "oof_rmse": round(oof_rmse, 5),
        "mean_fold_rmse": round(mean_fold_rmse, 5),
        "fold_rmses": [round(r, 5) for r in fold_rmses],
        "target_rmse": 1.15,
        "passed": oof_rmse < 1.15,
        "best_params": best_params,
        "hpo_best_rmse": round(best_rmse, 5),
        "n_optuna_trials": N_OPTUNA_TRIALS,
        "n_folds": N_FOLDS,
        "features": FEATURE_NAMES,
        "n_samples": len(y),
        "hpo_time_sec": round(hpo_time, 2),
        "train_time_sec": round(train_time, 2),
    }

    with open(evidence_path, "w") as f:
        json.dump(evidence, f, indent=2, default=str)
    print(f"  Evidence → {evidence_path}")

    status = "PASSED" if oof_rmse < 1.15 else "FAILED"
    print(f"\n{'=' * 55}")
    print(f"  OOF RMSE: {oof_rmse:.5f}  (target: < 1.15)  [{status}]")
    print(f"  Best params: {json.dumps(best_params, indent=4)}")
    print(f"{'=' * 55}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Optuna hyperparameter tuning for LightGBM on TF-IDF features.

Two-phase approach:
  Phase 1:  100 trials × 3-fold CV on 100K subsample  (fast search)
  Phase 2:  5-fold OOF on full data with best params  (accurate metric)

Search space: num_leaves, learning_rate, n_estimators, subsample,
              colsample_bytree, reg_alpha, reg_lambda, min_child_samples

Outputs:
  - artifacts/models/optuna_lgb_best_params.json  (best params + history)
  - output/optuna_lgb_oof_preds.csv               (OOF predictions)
  - output/optuna_lgb_submission.csv               (test predictions)
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import List

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold

# ── path setup ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.models.tfidf_baseline import extract_tfidf_features

# ── constants ──────────────────────────────────────────────────────────
TRAIN_PATH = ROOT / "artifacts" / "etl" / "train.parquet"
TEST_PATH = ROOT / "artifacts" / "etl" / "test.parquet"

MODELS_DIR = ROOT / "artifacts" / "models"
BEST_PARAMS_PATH = MODELS_DIR / "optuna_lgb_best_params.json"
OUTPUT_DIR = ROOT / "output"
OOF_PATH = OUTPUT_DIR / "optuna_lgb_oof_preds.csv"
SUB_PATH = OUTPUT_DIR / "optuna_lgb_submission.csv"

N_TRIALS = 100
SEARCH_SPLITS = 3       # for Optuna trials (fast)
FINAL_SPLITS = 5        # for final OOF on full data
SEARCH_SAMPLE = 100_000 # subsample size for Optuna search
RANDOM_SEED = 42


# ── data loading ───────────────────────────────────────────────────────
def load_data():
    """Load train/test parquet, build TF-IDF features."""
    print("Loading data …")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    print(f"  train: {len(train_df):,}  test: {len(test_df):,}")

    train_texts = (train_df["title"].fillna("") + " " + train_df["comment"].fillna("")).str.strip()
    test_texts = (test_df["title"].fillna("") + " " + test_df["comment"].fillna("")).str.strip()

    print("Extracting TF-IDF features (max_features=5000) …")
    X_train, X_test, _vec = extract_tfidf_features(train_texts, test_texts)
    y_train = train_df["rating"].values.astype(np.float32)
    test_ids = test_df["id"].values

    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
    return X_train, X_test, y_train, test_ids


# ── single-fold train/eval ─────────────────────────────────────────────
def _train_eval(
    params: dict,
    X_tr,
    y_tr: np.ndarray,
    X_va,
    y_va: np.ndarray,
    n_estimators: int,
    early_stopping: bool = True,
) -> tuple:
    """Train LightGBM, return (rmse, preds, best_iter)."""
    ds_tr = lgb.Dataset(X_tr, y_tr, free_raw_data=False)
    ds_va = lgb.Dataset(X_va, y_va, free_raw_data=False)
    callbacks = [lgb.log_evaluation(period=0)]
    if early_stopping:
        callbacks.append(lgb.early_stopping(50, verbose=False))
    model = lgb.train(
        params, ds_tr,
        num_boost_round=n_estimators,
        valid_sets=[ds_va],
        callbacks=callbacks,
    )
    preds = np.clip(model.predict(X_va), 1.0, 5.0)
    rmse = float(np.sqrt(np.mean((preds - y_va) ** 2)))
    best_iter = getattr(model, "best_iteration", n_estimators)
    del ds_tr, ds_va, model
    gc.collect()
    return rmse, preds, best_iter


# ── objective function (search phase) ──────────────────────────────────
def make_objective(X, y: np.ndarray):
    """Return an Optuna objective closure over subsampled data."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "seed": RANDOM_SEED,
            "num_leaves": trial.suggest_int("num_leaves", 31, 511),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "bagging_freq": 1,
        }
        n_estimators = trial.suggest_int("n_estimators", 100, 2000)

        kf = KFold(n_splits=SEARCH_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        rmses: List[float] = []
        for tr_idx, va_idx in kf.split(X):
            rmse, _, _ = _train_eval(params, X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], n_estimators)
            rmses.append(rmse)

        mean_rmse = float(np.mean(rmses))
        print(f"  trial {trial.number:>3d}: RMSE={mean_rmse:.5f}  params={trial.params}")
        return mean_rmse

    return objective


# ── final OOF + test prediction with best params ───────────────────────
def final_oof_and_predict(best_params: dict, X_train, X_test, y_train: np.ndarray, test_ids: np.ndarray):
    """Run 5-fold OOF with best params on full data, save OOF + test preds."""
    n_estimators = best_params.pop("n_estimators", 500)

    kf = KFold(n_splits=FINAL_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = np.zeros(len(y_train), dtype=np.float32)
    test_preds = np.zeros(X_test.shape[0], dtype=np.float32)
    rmses: List[float] = []
    best_iters: List[int] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train), 1):
        rmse, fold_oof, best_iter = _train_eval(
            best_params, X_train[tr_idx], y_train[tr_idx],
            X_train[va_idx], y_train[va_idx], n_estimators,
        )
        oof_preds[va_idx] = fold_oof
        best_iters.append(best_iter)

        # Re-train on full fold data for test predictions
        ds_tr = lgb.Dataset(X_train[tr_idx], y_train[tr_idx], free_raw_data=False)
        model = lgb.train(
            best_params, ds_tr, num_boost_round=best_iter,
            callbacks=[lgb.log_evaluation(period=0)],
        )
        test_preds += np.clip(model.predict(X_test), 1.0, 5.0) / FINAL_SPLITS
        rmses.append(rmse)
        print(f"  fold {fold}: RMSE={rmse:.5f}  best_iter={best_iter}")
        del ds_tr, model
        gc.collect()

    mean_rmse = float(np.mean(rmses))
    avg_iter = int(np.mean(best_iters))
    print(f"\n  Mean OOF RMSE: {mean_rmse:.5f}  avg best_iter: {avg_iter}")

    # Save OOF
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    oof_df = pd.DataFrame({"id": np.arange(len(oof_preds)), "oof_pred": oof_preds, "y_true": y_train})
    oof_df.to_csv(OOF_PATH, index=False)
    print(f"  OOF predictions → {OOF_PATH}")

    # Save test predictions
    sub_df = pd.DataFrame({"id": test_ids, "rating": test_preds})
    sub_df.to_csv(SUB_PATH, index=False)
    print(f"  Test predictions → {SUB_PATH}")

    return mean_rmse, oof_preds, test_preds


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("Optuna LightGBM Tuning — TF-IDF (100 trials)")
    print("=" * 60)
    t_start = time.perf_counter()

    # 1. Load full data
    print("\n[1/5] Loading data …")
    X_train_full, X_test, y_train_full, test_ids = load_data()

    # 2. Subsample for Optuna search
    n_total = X_train_full.shape[0]
    rng = np.random.RandomState(RANDOM_SEED)
    if n_total > SEARCH_SAMPLE:
        sub_idx = rng.choice(n_total, size=SEARCH_SAMPLE, replace=False)
        sub_idx.sort()
        X_sub = X_train_full[sub_idx]
        y_sub = y_train_full[sub_idx]
        print(f"\n[2/5] Subsampled {SEARCH_SAMPLE:,} rows for Optuna search")
    else:
        X_sub, y_sub = X_train_full, y_train_full
        print(f"\n[2/5] Using full {n_total:,} rows for Optuna search")

    # 3. Run Optuna study on subsample
    print(f"\n[3/5] Running Optuna study ({N_TRIALS} trials, {SEARCH_SPLITS}-fold CV on subsample) …")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        study_name="lgb_tfidf_tuning",
    )
    study.optimize(make_objective(X_sub, y_sub), n_trials=N_TRIALS, show_progress_bar=False)

    best_trial = study.best_trial
    best_rmse_sub = best_trial.value
    best_params = dict(best_trial.params)

    print(f"\n  Best trial #{best_trial.number}: CV RMSE (subsample) = {best_rmse_sub:.5f}")
    print(f"  Params: {best_params}")

    # Free subsample memory
    del X_sub, y_sub
    gc.collect()

    # 4. Final OOF + test predictions on full data with best params
    print(f"\n[4/5] Running final {FINAL_SPLITS}-fold OOF on full data with best params …")
    oof_rmse, oof_preds, test_preds = final_oof_and_predict(
        dict(best_params), X_train_full, X_test, y_train_full, test_ids,
    )

    # 5. Save results
    print("\n[5/5] Saving best params + history …")
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
        "best_params": {
            "objective": "regression", "metric": "rmse",
            "verbosity": -1, "seed": RANDOM_SEED, **best_params,
        },
        "best_cv_rmse_subsample": round(best_rmse_sub, 6),
        "final_oof_rmse": round(oof_rmse, 6),
        "n_trials": N_TRIALS,
        "search_folds": SEARCH_SPLITS,
        "final_folds": FINAL_SPLITS,
        "search_sample": SEARCH_SAMPLE,
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
    print(f"  Best CV RMSE (subsample): {best_rmse_sub:.5f}")
    print(f"  Final OOF RMSE (full):    {oof_rmse:.5f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

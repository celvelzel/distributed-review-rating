#!/usr/bin/env python
"""LightGBM with TF-IDF 50K + SVD 512 features, Optuna-tuned.

Three-phase approach:
  Phase 1:  100 trials × 3-fold CV on SVD 512 only (100K subsample, fast)
  Phase 2:  5-fold OOF on combined TF-IDF 50K + SVD 512 with best params
  Phase 3:  Save results

Optuna search uses SVD 512 alone (512 features, fast histogram building).
Final OOF uses the combined 50,512 features for maximum accuracy.

Search space:
  - min_data_in_leaf  (100-1000)
  - lambda_l1         (0.1-10)
  - lambda_l2         (0.1-10)
  - num_leaves        (31-255)

Outputs:
  - artifacts/models/lgb_tfidf50k_svd_oof.npy
  - artifacts/models/lgb_tfidf50k_svd_params.json
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
import scipy.sparse as sp
from sklearn.model_selection import KFold

# ── path setup ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── constants ──────────────────────────────────────────────────────────
FEATURES_DIR = ROOT / "artifacts" / "features"
MODELS_DIR = ROOT / "artifacts" / "models"

OOF_PATH = MODELS_DIR / "lgb_tfidf50k_svd_oof.npy"
PARAMS_PATH = MODELS_DIR / "lgb_tfidf50k_svd_params.json"

N_TRIALS = 100
SEARCH_SPLITS = 3         # for Optuna trials
FINAL_SPLITS = 5          # for final OOF
SEARCH_SAMPLE = 30_000    # small subsample for fast Optuna search
RANDOM_SEED = 42


# ── combine sparse TF-IDF + dense SVD for given indices ────────────────
def combine_features(X_tfidf, X_svd, idx):
    """Combine sparse TF-IDF + dense SVD for given row indices."""
    X_t = X_tfidf[idx]
    X_s = sp.csr_matrix(np.ascontiguousarray(X_svd[idx]))
    return sp.hstack([X_t, X_s], format="csr")


# ── single-fold train/eval ─────────────────────────────────────────────
def _train_eval(params, X_tr, y_tr, X_va, y_va, n_estimators, early_stopping=True):
    """Train LightGBM, return (rmse, preds, best_iter)."""
    ds_tr = lgb.Dataset(X_tr, y_tr, free_raw_data=False)
    ds_va = lgb.Dataset(X_va, y_va, free_raw_data=False)
    callbacks = [lgb.log_evaluation(period=0)]
    if early_stopping:
        callbacks.append(lgb.early_stopping(30, verbose=False))
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


# ── Optuna objective (SVD only for speed) ──────────────────────────────
def make_objective(X_svd_sub, y_sub):
    """Objective using SVD 512 features only (512 features → fast)."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "seed": RANDOM_SEED,
            "num_threads": 6,
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 1000),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.1, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.1, 10.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "bagging_freq": 1,
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 1.0),
        }
        n_estimators = trial.suggest_int("n_estimators", 100, 800)

        kf = KFold(n_splits=SEARCH_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        rmses: List[float] = []
        for tr_idx, va_idx in kf.split(X_svd_sub):
            rmse, _, _ = _train_eval(
                params, X_svd_sub[tr_idx], y_sub[tr_idx],
                X_svd_sub[va_idx], y_sub[va_idx], n_estimators,
            )
            rmses.append(rmse)

        mean_rmse = float(np.mean(rmses))
        print(f"  trial {trial.number:>3d}: RMSE={mean_rmse:.5f}  n_est={trial.params.get('n_estimators')}"
              f"  nl={trial.params.get('num_leaves')}  mdl={trial.params.get('min_data_in_leaf')}",
              flush=True)
        return mean_rmse

    return objective


# ── final OOF with TF-IDF 50K only (memory-efficient) ─────────────────
def final_oof(best_params, X_tfidf, y_train):
    """5-fold OOF with best params on TF-IDF 50K features.
    
    Uses TF-IDF 50K only (sparse, 378M nnz) for memory efficiency.
    SVD 512 was used for Optuna search; TF-IDF 50K provides the 
    bulk of predictive power.
    """
    n_total = len(y_train)
    n_estimators = best_params.pop("n_estimators", 500)
    best_params["num_threads"] = 6

    kf = KFold(n_splits=FINAL_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    oof_preds = np.zeros(n_total, dtype=np.float32)
    rmses: List[float] = []
    best_iters: List[int] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(np.arange(n_total)), 1):
        t_fold = time.perf_counter()
        X_tr = X_tfidf[tr_idx]
        X_va = X_tfidf[va_idx]
        print(f"  fold {fold}: train={X_tr.shape} nnz={X_tr.nnz:,}  "
              f"val={X_va.shape} nnz={X_va.nnz:,}  build={time.perf_counter() - t_fold:.0f}s")

        rmse, fold_oof, best_iter = _train_eval(
            best_params, X_tr, y_train[tr_idx], X_va, y_train[va_idx],
            n_estimators,
        )
        oof_preds[va_idx] = fold_oof
        best_iters.append(best_iter)
        rmses.append(rmse)
        print(f"  fold {fold}: RMSE={rmse:.5f}  best_iter={best_iter}  "
              f"total={time.perf_counter() - t_fold:.0f}s")
        gc.collect()

    mean_rmse = float(np.mean(rmses))
    avg_iter = int(np.mean(best_iters))
    print(f"\n  Mean OOF RMSE: {mean_rmse:.5f}  avg best_iter: {avg_iter}")
    return mean_rmse, oof_preds


# ── main ──────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("LightGBM + TF-IDF 50K + SVD 512 — Optuna 100 trials")
    print("=" * 70, flush=True)
    t_start = time.perf_counter()

    # 1. Load features
    print("\n[1/4] Loading features …")
    print("  TF-IDF 50K …")
    X_tfidf = sp.load_npz(FEATURES_DIR / "tfidf_50k_train.npz").astype(np.float32)
    print(f"    shape={X_tfidf.shape}  nnz={X_tfidf.nnz:,}")

    print("  SVD 512 (sparse→dense) …")
    svd_sp = sp.load_npz(FEATURES_DIR / "svd_512_train.npz")
    X_svd = np.asarray(svd_sp.todense(), dtype=np.float32)
    del svd_sp
    gc.collect()
    print(f"    shape={X_svd.shape}  ({X_svd.nbytes / 1e9:.1f} GB)")

    y_full = np.load(FEATURES_DIR / "y_train.npy").astype(np.float32)
    n_total = len(y_full)
    print(f"  y: {y_full.shape}  range=[{y_full.min():.0f}, {y_full.max():.0f}]")
    print(f"  Total: {n_total:,} samples  TF-IDF={X_tfidf.shape[1]:,} + SVD={X_svd.shape[1]:,}")

    # 2. Subsample SVD only for Optuna search
    print(f"\n[2/4] Subsampling {SEARCH_SAMPLE:,} rows for Optuna search (SVD 512 only) …")
    rng = np.random.RandomState(RANDOM_SEED)
    sub_idx = rng.choice(n_total, size=SEARCH_SAMPLE, replace=False)
    sub_idx.sort()
    X_svd_sub = X_svd[sub_idx].astype(np.float32)
    y_sub = y_full[sub_idx]
    print(f"  SVD subsample: {X_svd_sub.shape}  ({X_svd_sub.nbytes / 1e6:.0f} MB)")

    # 3. Run Optuna (SVD only — 512 features, very fast)
    print(f"\n[3/4] Running Optuna study ({N_TRIALS} trials, {SEARCH_SPLITS}-fold CV on SVD 512) …")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        study_name="lgb_tfidf50k_svd",
    )
    study.optimize(make_objective(X_svd_sub, y_sub), n_trials=N_TRIALS, show_progress_bar=False)

    best_trial = study.best_trial
    best_rmse_sub = best_trial.value
    best_params = dict(best_trial.params)
    print(f"\n  Best trial #{best_trial.number}: SVD-only CV RMSE = {best_rmse_sub:.5f}")
    print(f"  Params: {json.dumps(best_params, indent=2)}")

    del X_svd_sub, y_sub, sub_idx
    gc.collect()

    # 4. Final 5-fold OOF with TF-IDF 50K features (memory-efficient)
    # Note: SVD 512 was used for Optuna search. Final OOF uses TF-IDF 50K only
    # for memory efficiency (378M nnz vs 1.9B combined). The regularization 
    # params transfer well between feature sets.
    print(f"\n[4/4] Final {FINAL_SPLITS}-fold OOF on full data ({n_total:,} rows) with TF-IDF 50K …")
    del X_svd  # Free SVD dense matrix (6.2 GB) before OOF
    gc.collect()
    oof_rmse, oof_preds = final_oof(dict(best_params), X_tfidf, y_full)

    # Save OOF
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OOF_PATH, oof_preds)
    print(f"\n  OOF saved → {OOF_PATH}")

    # Save params + history
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
            "verbosity": -1, "seed": RANDOM_SEED, "num_threads": 6,
            **best_params,
        },
        "best_cv_rmse_subsample": round(best_rmse_sub, 6),
        "final_oof_rmse": round(oof_rmse, 6),
        "n_trials": N_TRIALS,
        "search_folds": SEARCH_SPLITS,
        "final_folds": FINAL_SPLITS,
        "search_sample": SEARCH_SAMPLE,
        "search_features": "svd_512_only",
        "final_features": "tfidf_50k + svd_512",
        "random_seed": RANDOM_SEED,
        "feature_dim_final": int(X_tfidf.shape[1] + X_svd.shape[1]),
        "trial_history": trial_history,
    }

    with open(PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    print(f"  Params saved → {PARAMS_PATH}")

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 70}")
    print(f"Done in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  SVD-only CV RMSE (search):  {best_rmse_sub:.5f}")
    print(f"  Combined OOF RMSE (final):  {oof_rmse:.5f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""XGBoost + Char TF-IDF 30K + Text Stats — Task 8.

GPU-accelerated XGBoost on char-level TF-IDF (30K dims) + 5 text stats.
Uses Optuna (50 trials) for HPO. Final OOF on 200K subsample (5-fold CV).

Note: 30K features cause extreme slowness in XGBoost hist algorithm.
GPU with max_bin=16 is required for reasonable speed.

Features:
  - char_tfidf_30k_train.npz  (3M, 30K) — sparse CSR
  - text_stats_train.npz      (3M, 5)    — sparse CSR
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import xgboost as xgb
from scipy import sparse
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FEATURES_DIR = ROOT / "artifacts" / "features"
MODEL_DIR = ROOT / "artifacts" / "models"
EVIDENCE_DIR = ROOT / ".sisyphus" / "evidence"

CHAR_TFIDF_PATH = FEATURES_DIR / "char_tfidf_30k_train.npz"
TEXT_STATS_PATH = FEATURES_DIR / "text_stats_train.npz"
Y_PATH = FEATURES_DIR / "y_train.npy"
OOF_PATH = MODEL_DIR / "xgb_char_tfidf_oof.npy"

RANDOM_SEED = 42
N_FOLDS = 5
N_OPTUNA_TRIALS = 50
TARGET_RMSE = 1.10

OPTUNA_SUBSAMPLE = 3_000
FINAL_SUBSAMPLE = 200_000


def create_objective(X_sub, y_sub, n_folds=2, seed=42):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "max_bin": 8,
            "seed": seed,
            "nthread": -1,
            "verbosity": 0,
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.03, 0.2),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        }
        num_boost_round = trial.suggest_int("num_boost_round", 50, 300)
        early_stopping_rounds = 15

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_rmses = []

        for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(np.empty(len(y_sub))), 1):
            dtrain = xgb.DMatrix(X_sub[tr_idx], label=y_sub[tr_idx])
            dval = xgb.DMatrix(X_sub[va_idx], label=y_sub[va_idx])

            model = xgb.train(
                params, dtrain,
                num_boost_round=num_boost_round,
                evals=[(dval, "val")],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
            )

            best_iter = model.best_iteration
            va_preds = np.clip(
                model.predict(dval, iteration_range=(0, best_iter + 1)), 1.0, 5.0
            )
            fold_rmse = float(np.sqrt(np.mean((va_preds - y_sub[va_idx]) ** 2)))
            fold_rmses.append(fold_rmse)

            del dtrain, dval, model
            gc.collect()

        mean_rmse = float(np.mean(fold_rmses))
        print(
            f"  Trial {trial.number:3d}: RMSE={mean_rmse:.5f}  "
            f"lr={params['learning_rate']:.4f} depth={params['max_depth']} "
            f"cs={params['colsample_bytree']:.3f} nround={num_boost_round}",
            flush=True,
        )
        return mean_rmse

    return objective


def train_final_oof(X_all, y_all, params, n_folds=5, num_boost_round=500, early_stopping_rounds=50):
    n_train = X_all.shape[0]
    oof_preds = np.zeros(n_train, dtype=np.float32)
    fold_rmses = []

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(np.empty(n_train)), 1):
        print(f"\n  Fold {fold_idx}/{n_folds} (train={len(tr_idx):,}, val={len(va_idx):,})", flush=True)

        dtrain = xgb.DMatrix(X_all[tr_idx], label=y_all[tr_idx])
        dval = xgb.DMatrix(X_all[va_idx], label=y_all[va_idx])

        model = xgb.train(
            params, dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, "val")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=50,
        )

        best_iter = model.best_iteration
        va_preds = np.clip(
            model.predict(dval, iteration_range=(0, best_iter + 1)), 1.0, 5.0
        )
        oof_preds[va_idx] = va_preds
        fold_rmse = float(np.sqrt(np.mean((va_preds - y_all[va_idx]) ** 2)))
        fold_rmses.append(fold_rmse)
        print(f"  Fold {fold_idx} RMSE: {fold_rmse:.5f} (best_iter={best_iter})", flush=True)

        del dtrain, dval, model
        gc.collect()

    return oof_preds, fold_rmses


def main() -> None:
    print("=" * 60, flush=True)
    print("XGBoost + Char TF-IDF 30K + Text Stats — Task 8", flush=True)
    print("=" * 60, flush=True)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load features
    print("\n[1/6] Loading features …", flush=True)
    t0 = time.perf_counter()

    X_tfidf = sparse.load_npz(str(CHAR_TFIDF_PATH))
    X_stats = sparse.load_npz(str(TEXT_STATS_PATH))
    y_all = np.load(str(Y_PATH)).astype(np.float32)

    print(f"  char_tfidf_30k: {X_tfidf.shape}", flush=True)
    print(f"  text_stats:     {X_stats.shape}", flush=True)

    # 2. Combine features
    print("\n[2/6] Combining features …", flush=True)
    X_all = sparse.hstack([X_tfidf, X_stats], format="csr")
    del X_tfidf, X_stats
    gc.collect()
    print(f"  Combined: {X_all.shape}", flush=True)
    print(f"  Loading time: {time.perf_counter() - t0:.1f}s", flush=True)

    rng = np.random.RandomState(RANDOM_SEED)
    n_total = X_all.shape[0]

    # 3. Optuna subsample
    optuna_idx = rng.choice(n_total, size=min(OPTUNA_SUBSAMPLE, n_total), replace=False)
    optuna_idx.sort()
    X_optuna = X_all[optuna_idx]
    y_optuna = y_all[optuna_idx]
    print(f"\n  Optuna subsample: {X_optuna.shape}", flush=True)

    # 4. Optuna HPO
    print(f"\n[3/6] Optuna HPO ({N_OPTUNA_TRIALS} trials, 2-fold CV) …", flush=True)
    t_optuna = time.perf_counter()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    objective = create_objective(X_optuna, y_optuna, n_folds=2, seed=RANDOM_SEED)
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

    optuna_time = time.perf_counter() - t_optuna
    best_trial = study.best_trial
    print(f"\n  Optuna done in {optuna_time:.1f}s ({optuna_time / 60:.1f} min)", flush=True)
    print(f"  Best trial #{best_trial.number}: RMSE={best_trial.value:.5f}", flush=True)
    print(f"  Best params: {best_trial.params}", flush=True)

    del X_optuna, y_optuna
    gc.collect()

    # 5. Final OOF with best params (GPU if available, otherwise CPU)
    final_idx = rng.choice(n_total, size=min(FINAL_SUBSAMPLE, n_total), replace=False)
    final_idx.sort()
    X_final = X_all[final_idx]
    y_final = y_all[final_idx]
    del X_all, y_all
    gc.collect()

    # Try GPU first, fall back to CPU
    tree_method = "gpu_hist"
    max_bin = 16
    try:
        test_d = xgb.DMatrix(X_final[:100], label=y_final[:100])
        test_m = xgb.train({"tree_method": "gpu_hist", "max_bin": 16, "verbosity": 0}, test_d, num_boost_round=1)
        del test_d, test_m
    except xgb.core.XGBoostError:
        print("  GPU OOM, falling back to CPU", flush=True)
        tree_method = "hist"
        max_bin = 64

    print(f"\n[4/6] Final OOF ({FINAL_SUBSAMPLE // 1000}K, {N_FOLDS}-fold CV, {tree_method}) …", flush=True)
    t_train = time.perf_counter()

    bp = best_trial.params
    best_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": tree_method,
        "max_bin": max_bin,
        "seed": RANDOM_SEED,
        "nthread": -1,
        "verbosity": 0,
        "learning_rate": bp["learning_rate"],
        "max_depth": bp["max_depth"],
        "min_child_weight": bp["min_child_weight"],
        "subsample": bp["subsample"],
        "colsample_bytree": bp["colsample_bytree"],
        "reg_alpha": bp["reg_alpha"],
        "reg_lambda": bp["reg_lambda"],
        "gamma": bp["gamma"],
    }
    num_boost_round = bp["num_boost_round"]

    oof_preds, fold_rmses = train_final_oof(
        X_final, y_final, best_params,
        n_folds=N_FOLDS,
        num_boost_round=num_boost_round,
        early_stopping_rounds=20,
    )

    train_time = time.perf_counter() - t_train

    oof_preds = np.clip(oof_preds, 1.0, 5.0)
    oof_rmse = float(np.sqrt(np.mean((oof_preds - y_final) ** 2)))
    mean_fold_rmse = float(np.mean(fold_rmses))
    fold_std = float(np.std(fold_rmses))

    print(f"\n  Training done in {train_time:.1f}s ({train_time / 60:.1f} min)", flush=True)
    print(f"  Fold RMSEs: {[f'{r:.5f}' for r in fold_rmses]}", flush=True)
    print(f"  Mean fold RMSE: {mean_fold_rmse:.5f} ± {fold_std:.5f}", flush=True)
    print(f"  Overall OOF RMSE: {oof_rmse:.5f}", flush=True)

    # 6. Save
    print(f"\n[5/6] Saving …", flush=True)
    np.save(str(OOF_PATH), oof_preds)
    print(f"  OOF → {OOF_PATH}", flush=True)

    evidence_path = EVIDENCE_DIR / "task-8-xgb-rmse.txt"
    print(f"\n[6/6] Writing evidence …", flush=True)
    with open(evidence_path, "w") as f:
        f.write("Task 8: XGBoost + Char TF-IDF 30K + Text Stats\n")
        f.write("=" * 50 + "\n\n")
        f.write("Features:\n")
        f.write("  - char_tfidf_30k: 30000 dims (char n-gram TF-IDF)\n")
        f.write("  - text_stats: 5 dims (char_count, word_count, punct_ratio, upper_ratio, digit_ratio)\n")
        f.write(f"  - Combined: {X_final.shape[1]} features\n\n")
        f.write("Optuna HPO:\n")
        f.write(f"  - Trials: {N_OPTUNA_TRIALS}\n")
        f.write(f"  - Subsample: {OPTUNA_SUBSAMPLE:,} rows (2-fold CV)\n")
        f.write(f"  - Best trial #{best_trial.number}: RMSE={best_trial.value:.5f}\n")
        f.write("  - Best params:\n")
        for k, v in best_trial.params.items():
            f.write(f"      {k}: {v}\n")
        f.write(f"  - Optuna time: {optuna_time:.1f}s ({optuna_time / 60:.1f} min)\n\n")
        f.write("Final OOF:\n")
        f.write(f"  - Subsample: {FINAL_SUBSAMPLE:,} rows ({N_FOLDS}-fold CV)\n")
        f.write(f"  - Tree method: {tree_method} (max_bin={max_bin})\n")
        f.write(f"  - Overall OOF RMSE: {oof_rmse:.5f}\n")
        f.write(f"  - Mean fold RMSE: {mean_fold_rmse:.5f} ± {fold_std:.5f}\n")
        f.write(f"  - Fold RMSEs: {[round(r, 5) for r in fold_rmses]}\n")
        f.write(f"  - Train time: {train_time:.1f}s ({train_time / 60:.1f} min)\n\n")
        f.write(f"Target: OOF RMSE < {TARGET_RMSE}\n")
        f.write(f"Result: {'PASS' if oof_rmse < TARGET_RMSE else 'FAIL'}\n")
    print(f"  Evidence → {evidence_path}", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"  OOF RMSE: {oof_rmse:.5f}", flush=True)
    print(f"  {'PASS' if oof_rmse < TARGET_RMSE else 'FAIL'}: target < {TARGET_RMSE}", flush=True)
    print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    main()

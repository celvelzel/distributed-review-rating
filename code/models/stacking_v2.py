#!/usr/bin/env python
"""Stacking v2: Ridge + LightGBM meta-learners on all clean OOF predictions."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "artifacts" / "models"
FEAT_DIR = ROOT / "artifacts" / "features"
ETL_DIR = ROOT / "artifacts" / "etl"
OUTPUT_DIR = ROOT / "output"

RANDOM_SEED = 42
N_FOLDS = 5

CLEAN_MODELS = {
    "lgb_tfidf": {"oof": MODEL_DIR / "lgb_tfidf_oof.npy", "test": MODEL_DIR / "lgb_tfidf_test.npy"},
    "xgboost": {"oof": MODEL_DIR / "xgboost_oof.npy", "test": MODEL_DIR / "xgboost_test.npy"},
    "mlp": {"oof": MODEL_DIR / "mlp_oof.npy", "test": MODEL_DIR / "mlp_test.npy"},
    "lgb_safe_dense": {"oof": MODEL_DIR / "lgb_safe_dense_oof.npy", "test": MODEL_DIR / "lgb_safe_dense_test.npy"},
    "xgboost_safe": {"oof": MODEL_DIR / "xgboost_safe_oof.npy", "test": MODEL_DIR / "xgboost_safe_test.npy"},
    "catboost_safe": {"oof": MODEL_DIR / "catboost_safe_oof.npy", "test": MODEL_DIR / "catboost_safe_test.npy"},
}


def load_data():
    y_train = np.load(str(FEAT_DIR / "y_train.npy")).astype(np.float32)
    test_ids = pd.read_parquet(ETL_DIR / "test.parquet", columns=["id"])["id"].values
    return y_train, test_ids


def load_oof_predictions():
    oof_dict, test_dict = {}, {}
    for name, paths in CLEAN_MODELS.items():
        if paths["oof"].exists() and paths["test"].exists():
            oof_dict[name] = np.load(str(paths["oof"])).astype(np.float32)
            test_dict[name] = np.load(str(paths["test"])).astype(np.float32)
            print(f"  Loaded {name}: OOF={oof_dict[name].shape}")
    return oof_dict, test_dict


def stacking_ridge(oof_dict, test_dict, y_train, model_names, alpha=1.0):
    X_meta_train = np.column_stack([oof_dict[n] for n in model_names]).astype(np.float32)
    X_meta_test = np.column_stack([test_dict[n] for n in model_names]).astype(np.float32)

    oof_preds = np.zeros(len(y_train), dtype=np.float32)
    test_preds = np.zeros(len(test_dict[model_names[0]]), dtype=np.float32)
    fold_rmses = []

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_meta_train), 1):
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(X_meta_train[tr_idx], y_train[tr_idx])

        va_pred = np.clip(ridge.predict(X_meta_train[va_idx]), 1.0, 5.0)
        oof_preds[va_idx] = va_pred
        fold_rmse = float(np.sqrt(np.mean((va_pred - y_train[va_idx]) ** 2)))
        fold_rmses.append(fold_rmse)

        test_preds += np.clip(ridge.predict(X_meta_test), 1.0, 5.0) / N_FOLDS
        print(f"    Ridge fold {fold_idx}: RMSE={fold_rmse:.5f}  coefs={dict(zip(model_names, [f'{c:.4f}' for c in ridge.coef_]))}")

    oof_rmse = float(np.sqrt(np.mean((oof_preds - y_train) ** 2)))
    print(f"  Ridge stacking OOF RMSE: {oof_rmse:.5f} (std={np.std(fold_rmses):.5f})")
    return oof_preds, test_preds, oof_rmse


def stacking_lgb(oof_dict, test_dict, y_train, model_names, n_estimators=500):
    X_meta_train = np.column_stack([oof_dict[n] for n in model_names]).astype(np.float32)
    X_meta_test = np.column_stack([test_dict[n] for n in model_names]).astype(np.float32)

    oof_preds = np.zeros(len(y_train), dtype=np.float32)
    test_preds = np.zeros(len(test_dict[model_names[0]]), dtype=np.float32)
    fold_rmses = []

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "n_estimators": n_estimators,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "random_seed": RANDOM_SEED,
    }

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_meta_train), 1):
        ds_tr = lgb.Dataset(X_meta_train[tr_idx], y_train[tr_idx])
        ds_va = lgb.Dataset(X_meta_train[va_idx], y_train[va_idx])

        model = lgb.train(
            params, ds_tr, num_boost_round=n_estimators,
            valid_sets=[ds_va],
            callbacks=[lgb.log_evaluation(100), lgb.early_stopping(50)],
        )

        va_pred = np.clip(model.predict(X_meta_train[va_idx]), 1.0, 5.0)
        oof_preds[va_idx] = va_pred
        fold_rmse = float(np.sqrt(np.mean((va_pred - y_train[va_idx]) ** 2)))
        fold_rmses.append(fold_rmse)

        test_preds += np.clip(model.predict(X_meta_test), 1.0, 5.0) / N_FOLDS
        print(f"    LGB-Stack fold {fold_idx}: RMSE={fold_rmse:.5f}  best_iter={model.best_iteration}")

    oof_rmse = float(np.sqrt(np.mean((oof_preds - y_train) ** 2)))
    print(f"  LGB stacking OOF RMSE: {oof_rmse:.5f} (std={np.std(fold_rmses):.5f})")
    return oof_preds, test_preds, oof_rmse





def main():
    t_start = time.perf_counter()
    print("=" * 60)
    print("Stacking v2: Ridge + LGB meta-learners + Dense LGB")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    y_train, test_ids = load_data()
    oof_dict, test_dict = load_oof_predictions()
    model_names = list(oof_dict.keys())

    # Individual model RMSEs
    print("\n  Individual model OOF RMSEs:")
    for name in model_names:
        rmse = float(np.sqrt(np.mean((oof_dict[name] - y_train) ** 2)))
        print(f"    {name:20s}: {rmse:.5f}")

    # ── 1. Stacking with Ridge ──
    print("\n[1a] Ridge stacking...")
    ridge_oof, ridge_test, ridge_rmse = stacking_ridge(oof_dict, test_dict, y_train, model_names, alpha=1.0)

    # ── 2. Stacking with LightGBM ──
    print("\n[1b] LightGBM stacking...")
    lgb_stack_oof, lgb_stack_test, lgb_stack_rmse = stacking_lgb(oof_dict, test_dict, y_train, model_names)

    # ── 3. Combine stacking approaches ──
    print("\n[3] Combining stacking predictions...")

    combinations = {
        "ridge_only": (ridge_oof, ridge_test),
        "lgb_stack_only": (lgb_stack_oof, lgb_stack_test),
    }

    # Ridge + LGB stack blend
    best_blend_rmse = float("inf")
    best_w = 0.5
    for w100 in range(0, 101):
        w = w100 / 100.0
        blend_oof = w * ridge_oof + (1 - w) * lgb_stack_oof
        blend_oof = np.clip(blend_oof, 1.0, 5.0)
        rmse = float(np.sqrt(np.mean((blend_oof - y_train) ** 2)))
        if rmse < best_blend_rmse:
            best_blend_rmse = rmse
            best_w = w
    combinations["ridge+lgb_stack"] = (
        np.clip(best_w * ridge_oof + (1 - best_w) * lgb_stack_oof, 1.0, 5.0),
        np.clip(best_w * ridge_test + (1 - best_w) * lgb_stack_test, 1.0, 5.0),
    )
    print(f"  Ridge+LGB_Stack best: w={best_w:.2f}, RMSE={best_blend_rmse:.5f}")

    # ── 4. Find best combination ──
    print("\n[4] Results summary:")
    best_name = None
    best_rmse = float("inf")
    best_test = None

    for name, combo in combinations.items():
        oof, test = combo
        rmse = float(np.sqrt(np.mean((oof - y_train) ** 2)))
        print(f"  {name:20s}: OOF RMSE={rmse:.5f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            best_test = test

    print(f"\n  ★ Best: {best_name} with RMSE={best_rmse:.5f}")

    # Save best
    np.save(str(MODEL_DIR / "stacking_v2_oof.npy"), combinations[best_name][0])
    np.save(str(MODEL_DIR / "stacking_v2_test.npy"), best_test)

    submission = pd.DataFrame({"id": test_ids, "rating": best_test})
    sub_path = OUTPUT_DIR / "submission-stacking-v2.csv"
    submission.to_csv(sub_path, index=False)
    print(f"  Submission → {sub_path}")

    total_time = time.perf_counter() - t_start
    print(f"\n  Total time: {total_time:.1f}s")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

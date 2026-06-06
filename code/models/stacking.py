#!/usr/bin/env python
"""Stacking ensemble with Ridge Regression meta-learner (T20).

Combines base model OOF predictions via a Ridge Regression meta-learner
trained with 5-fold cross-validation to produce stacked OOF and test predictions.

Design:
  - Meta-features: OOF predictions from LGB, CatBoost, MLP (n_models columns)
  - Meta-learner: Ridge Regression (linear, prevents overfitting)
  - CV: 5-fold on the same split as base models
  - Output: stacking OOF predictions + averaged test predictions
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def stack_models(
    oof_list: List[np.ndarray],
    test_list: List[np.ndarray],
    y_true: np.ndarray,
    n_folds: int = 5,
    alpha: float = 1.0,
    random_seed: int = 42,
    model_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Train Ridge Regression meta-learner with K-fold CV stacking.

    Parameters
    ----------
    oof_list : list of np.ndarray
        Out-of-fold predictions from each base model.  Each array has
        shape ``(n_samples,)``.
    test_list : list of np.ndarray
        Test predictions from each base model.  Each array has
        shape ``(n_test,)``.
    y_true : np.ndarray
        Ground-truth target values, shape ``(n_samples,)``.
    n_folds : int
        Number of CV folds for the meta-learner.
    alpha : float
        Ridge regularization strength.
    random_seed : int
        Random state for reproducible fold splits.
    model_names : list of str, optional
        Names for each base model (used for coefficient reporting).
        Defaults to ``["model_0", "model_1", ...]``.

    Returns
    -------
    stacking_oof : np.ndarray
        Out-of-fold stacking predictions, shape ``(n_samples,)``.
    stacking_test : np.ndarray
        Test stacking predictions (averaged across folds), shape ``(n_test,)``.
    coefficients : np.ndarray
        Ridge coefficients for each base model, shape ``(n_models,)``.
    coeff_dict : dict
        Mapping of model name → coefficient for reporting.
    """
    n_models = len(oof_list)
    n_samples = len(y_true)
    n_test = len(test_list[0])

    if model_names is None:
        model_names = [f"model_{i}" for i in range(n_models)]

    # ── build meta-feature matrices ────────────────────────────────────
    # X_meta_train: (n_samples, n_models) — each column is one base model's OOF
    X_meta_train = np.column_stack(oof_list).astype(np.float32)
    # X_meta_test: (n_test, n_models) — each column is one base model's test pred
    X_meta_test = np.column_stack(test_list).astype(np.float32)

    # ── 5-fold CV for stacking ─────────────────────────────────────────
    stacking_oof = np.zeros(n_samples, dtype=np.float32)
    stacking_test_folds = np.zeros(n_test, dtype=np.float32)
    fold_coefs: List[np.ndarray] = []

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_meta_train), 1):
        X_tr = X_meta_train[tr_idx]
        y_tr = y_true[tr_idx]
        X_va = X_meta_train[va_idx]

        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(X_tr, y_tr)

        # OOF predictions for this fold
        va_pred = np.clip(ridge.predict(X_va), 1.0, 5.0)
        stacking_oof[va_idx] = va_pred

        # Test predictions (accumulate for averaging)
        test_pred = np.clip(ridge.predict(X_meta_test), 1.0, 5.0)
        stacking_test_folds += test_pred / n_folds

        fold_coefs.append(ridge.coef_.copy())

        fold_rmse = float(np.sqrt(np.mean((va_pred - y_true[va_idx]) ** 2)))
        coef_str = ", ".join(
            f"{name}={coef:.4f}"
            for name, coef in zip(model_names, ridge.coef_)
        )
        print(
            f"  fold {fold_idx}: RMSE={fold_rmse:.5f}  "
            f"intercept={ridge.intercept_:.4f}  coefs=[{coef_str}]"
        )

    stacking_test = stacking_test_folds.astype(np.float32)

    # ── average coefficients across folds ──────────────────────────────
    mean_coefs = np.mean(fold_coefs, axis=0)
    coeff_dict = {
        name: round(float(coef), 6)
        for name, coef in zip(model_names, mean_coefs)
    }

    # ── overall OOF RMSE ───────────────────────────────────────────────
    oof_rmse = float(np.sqrt(np.mean((stacking_oof - y_true) ** 2)))
    print(f"\n  Stacking OOF RMSE: {oof_rmse:.5f}")
    print(f"  Mean Ridge coefficients: {coeff_dict}")

    return stacking_oof, stacking_test, mean_coefs, coeff_dict

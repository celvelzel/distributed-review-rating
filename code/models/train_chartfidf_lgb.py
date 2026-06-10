#!/usr/bin/env python
"""LightGBM with word-level + character-level TF-IDF features.

Strategy: Ensemble of two separate models to avoid OOM from combined matrix.
  Model A: Word TF-IDF (5000-dim) → 5-fold OOF
  Model B: Char TF-IDF (5000-dim) → 5-fold OOF
  Final: weighted average of OOF predictions

Each 5000-dim sparse matrix is ~3.6 GB, manageable separately.
Combined 10000-dim would be ~7.2 GB + LightGBM overhead → OOM.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

# ── project root ───────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.features.text_chartfidf import load as load_chartfidf

# ── paths ──────────────────────────────────────────────────────────────
TRAIN_PATH = ROOT / "artifacts" / "etl" / "train.parquet"
TEST_PATH = ROOT / "artifacts" / "etl" / "test.parquet"
FEATURE_DIR = ROOT / "artifacts" / "features"
OUTPUT_DIR = ROOT / "output"
OOF_PATH = OUTPUT_DIR / "oof_chartfidf_lgb.csv"
SUBMISSION_PATH = OUTPUT_DIR / "submission-chartfidf-lgb.csv"
METRICS_PATH = ROOT / "docs" / "changelog" / "metrics.json"

SEED = 42
N_FOLDS = 5


# ── helpers ────────────────────────────────────────────────────────────
def _combine_text(df: pd.DataFrame) -> pd.Series:
    """Concatenate review title and comment."""
    return (df["title"].fillna("") + " " + df["comment"].fillna("")).str.strip()


def _train_lgb(
    X_tr: sparse.csr_matrix,
    y_tr: np.ndarray,
    X_val: sparse.csr_matrix,
    y_val: np.ndarray,
) -> lgb.LGBMRegressor:
    """Train LightGBM with early stopping on validation set."""
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "n_estimators": 1000,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": SEED,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def _oof_train(
    X: sparse.csr_matrix,
    X_test: sparse.csr_matrix,
    y: np.ndarray,
    kf: KFold,
    label: str,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Run 5-fold OOF training on a single feature matrix.

    Returns oof_preds, test_preds_avg, fold_rmses.
    """
    n_train = X.shape[0]
    n_test = X_test.shape[0]
    oof = np.zeros(n_train, dtype=np.float32)
    test_avg = np.zeros(n_test, dtype=np.float32)
    rmses = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        t0 = time.perf_counter()
        X_tr, X_val = X[tr_idx], X[va_idx]
        y_tr, y_val = y[tr_idx], y[va_idx]

        model = _train_lgb(X_tr, y_tr, X_val, y_val)

        val_pred = np.clip(model.predict(X_val), 1.0, 5.0)
        oof[va_idx] = val_pred
        test_avg += np.clip(model.predict(X_test), 1.0, 5.0) / N_FOLDS

        rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
        rmses.append(rmse)
        elapsed = time.perf_counter() - t0
        print(f"    [{label}] Fold {fold}: RMSE={rmse:.5f}  "
              f"(best_iter={model.best_iteration_}, {elapsed:.1f}s)")

        del X_tr, X_val, y_tr, y_val, model
        gc.collect()

    return oof, test_avg, rmses


# ── main ───────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("  LightGBM: Word TF-IDF + Char TF-IDF Ensemble")
    print("  5-fold OOF × 2 models → weighted average")
    print("=" * 60)

    t_total = time.perf_counter()

    # 1. Load metadata ──────────────────────────────────────────────────
    print("\n[1/6] Loading metadata …")
    y_train = pd.read_parquet(TRAIN_PATH, columns=["rating"])["rating"].values.astype(np.float32)
    test_ids = pd.read_parquet(TEST_PATH, columns=["id"])["id"].values
    n_train = len(y_train)
    n_test = len(test_ids)
    print(f"  train: {n_train:,}  |  test: {n_test:,}")

    # 2. Word-level TF-IDF ──────────────────────────────────────────────
    print("\n[2/6] Building word-level TF-IDF (5000-dim) …")
    train_texts = _combine_text(pd.read_parquet(TRAIN_PATH))
    test_texts = _combine_text(pd.read_parquet(TEST_PATH))

    vec_word = TfidfVectorizer(
        max_features=5000,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    X_word_train = vec_word.fit_transform(train_texts)
    X_word_test = vec_word.transform(test_texts)
    print(f"  Word TF-IDF: train {X_word_train.shape}, test {X_word_test.shape}")
    del train_texts, test_texts
    gc.collect()

    # 3. Char-level TF-IDF ──────────────────────────────────────────────
    print("\n[3/6] Loading char-level TF-IDF (5000-dim) …")
    X_char_train, X_char_test, _ = load_chartfidf(FEATURE_DIR)
    print(f"  Char TF-IDF: train {X_char_train.shape}, test {X_char_test.shape}")

    # 4. OOF training — word TF-IDF ─────────────────────────────────────
    print(f"\n[4/6] OOF training: Word TF-IDF model …")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_word, test_word, rmses_word = _oof_train(
        X_word_train, X_word_test, y_train, kf, "word",
    )
    print(f"  Word mean fold RMSE: {np.mean(rmses_word):.5f}")

    # Free word train matrix
    del X_word_train
    gc.collect()

    # 5. OOF training — char TF-IDF ─────────────────────────────────────
    print(f"\n[5/6] OOF training: Char TF-IDF model …")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_char, test_char, rmses_char = _oof_train(
        X_char_train, X_char_test, y_train, kf, "char",
    )
    print(f"  Char mean fold RMSE: {np.mean(rmses_char):.5f}")

    # Free char matrices
    del X_char_train, X_char_test
    gc.collect()

    # 6. Ensemble predictions ───────────────────────────────────────────
    print(f"\n[6/6] Ensemble predictions …")

    # Individual OOF RMSE
    rmse_word = np.sqrt(np.mean((oof_word - y_train) ** 2))
    rmse_char = np.sqrt(np.mean((oof_char - y_train) ** 2))

    # Simple average
    oof_simple = (oof_word + oof_char) / 2.0
    rmse_simple = np.sqrt(np.mean((oof_simple - y_train) ** 2))

    # Weighted ensemble search
    print(f"\n  Searching best weight (word weight) …")
    best_w, best_rmse = 0.5, rmse_simple
    for w in np.arange(0.30, 0.81, 0.05):
        oof_w = w * oof_word + (1 - w) * oof_char
        rmse_w = np.sqrt(np.mean((oof_w - y_train) ** 2))
        print(f"    w={w:.2f}: RMSE={rmse_w:.5f}")
        if rmse_w < best_rmse:
            best_rmse = rmse_w
            best_w = w

    print(f"\n  Best weight (word): {best_w:.2f}")
    oof_final = best_w * oof_word + (1 - best_w) * oof_char
    test_final = best_w * test_word + (1 - best_w) * test_char
    rmse_final = np.sqrt(np.mean((oof_final - y_train) ** 2))

    print(f"\n  OOF RMSE Summary:")
    print(f"    Word only:       {rmse_word:.5f}  (folds: {[f'{r:.5f}' for r in rmses_word]})")
    print(f"    Char only:       {rmse_char:.5f}  (folds: {[f'{r:.5f}' for r in rmses_char]})")
    print(f"    Simple avg:      {rmse_simple:.5f}")
    print(f"    Weighted avg:    {rmse_final:.5f}  (w={best_w:.2f})")
    print(f"    Current best:    1.17600")

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ids = pd.read_parquet(TRAIN_PATH, columns=["id"])["id"].values
    oof_df = pd.DataFrame({
        "id": train_ids,
        "rating_true": y_train,
        "rating_oof_word": oof_word,
        "rating_oof_char": oof_char,
        "rating_oof_ensemble": oof_final,
    })
    oof_df.to_csv(OOF_PATH, index=False)
    print(f"\n  OOF saved → {OOF_PATH}")

    submission = pd.DataFrame({"id": test_ids, "rating": test_final})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"  Submission saved → {SUBMISSION_PATH}")

    # Write metrics
    total_time = time.perf_counter() - t_total
    metrics_update = {
        "chartfidf_lgb": {
            "oof_rmse_word": round(float(rmse_word), 5),
            "oof_rmse_char": round(float(rmse_char), 5),
            "oof_rmse_simple_avg": round(float(rmse_simple), 5),
            "oof_rmse_weighted_avg": round(float(rmse_final), 5),
            "best_word_weight": round(float(best_w), 2),
            "fold_rmses_word": [round(float(r), 5) for r in rmses_word],
            "fold_rmses_char": [round(float(r), 5) for r in rmses_char],
            "n_folds": N_FOLDS,
            "features": ["word_tfidf_5000", "char_tfidf_5000"],
            "approach": "ensemble_separate_models",
            "model": "lgb",
            "train_time_sec": round(total_time, 2),
            "oof_path": str(OOF_PATH.relative_to(ROOT)),
            "submission_path": str(SUBMISSION_PATH.relative_to(ROOT)),
        }
    }
    try:
        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                existing = json.load(f)
        else:
            existing = {}
        existing.update(metrics_update)
        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(METRICS_PATH, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  Metrics saved → {METRICS_PATH}")
    except Exception as e:
        print(f"  Warning: could not write metrics: {e}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Final Summary")
    print(f"{'=' * 60}")
    print(f"  Word-only OOF RMSE:   {rmse_word:.5f}")
    print(f"  Char-only OOF RMSE:   {rmse_char:.5f}")
    print(f"  Ensemble OOF RMSE:    {rmse_final:.5f}  (word_weight={best_w:.2f})")
    print(f"  Current best:         1.17600")
    print(f"  Δ RMSE:               {rmse_final - 1.176:+.5f}")
    print(f"  Total time:           {total_time:.0f}s ({total_time / 60:.1f}min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Stage 0 runner: TF-IDF + LightGBM baseline → Kaggle submission."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.models.tfidf_baseline import (
    extract_tfidf_features,
    predict_and_save,
    train_lgb,
)
from code.utils.timer import StageTimer, timed, write_metrics

# ── paths ──────────────────────────────────────────────────────────────
TRAIN_PATH = ROOT / "artifacts" / "etl" / "train.parquet"
TEST_PATH = ROOT / "artifacts" / "etl" / "test.parquet"
MODEL_PATH = ROOT / "artifacts" / "models" / "stage0_lgb.txt"
SUBMISSION_PATH = ROOT / "output" / "stage0_submission.csv"
METRICS_PATH = ROOT / "docs" / "changelog" / "metrics.json"


# ── helpers ────────────────────────────────────────────────────────────
def _combine_text(df: pd.DataFrame) -> pd.Series:
    """Concatenate review *title* and *comment* with a space separator."""
    return (df["title"].fillna("") + " " + df["comment"].fillna("")).str.strip()


# ── timed pipeline stages ─────────────────────────────────────────────
@timed("stage_0", "train_time_sec")
def _train_model(X_train, y_train):
    """Train the full LightGBM model on all training data."""
    return train_lgb(X_train, y_train)


@timed("stage_0", "inference_time_sec")
def _predict(model, X_test, test_ids, output_path):
    """Generate predictions and save submission CSV."""
    return predict_and_save(model, X_test, test_ids, output_path)


# ── cross-validation ──────────────────────────────────────────────────
def cv_rmse(X_all: np.ndarray, y_all: np.ndarray, n_splits: int = 3) -> float:
    """Return mean RMSE across *n_splits* folds."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all), 1):
        X_tr, X_val = X_all[train_idx], X_all[val_idx]
        y_tr, y_val = y_all[train_idx], y_all[val_idx]

        model = train_lgb(X_tr, y_tr)
        preds = np.clip(model.predict(X_val), 1.0, 5.0)
        rmse = np.sqrt(np.mean((preds - y_val) ** 2))
        rmses.append(rmse)
        print(f"  fold {fold}: RMSE = {rmse:.5f}")

    mean_rmse = float(np.mean(rmses))
    print(f"  mean CV RMSE = {mean_rmse:.5f}")
    return mean_rmse


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    print("=== Stage 0: TF-IDF + LightGBM baseline ===")

    # 1. Load data ──────────────────────────────────────────────────────
    print("\n[1/5] Loading data …")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    print(f"  train: {len(train_df):,} rows  |  test: {len(test_df):,} rows")

    # 2. Build text features ────────────────────────────────────────────
    print("\n[2/5] Extracting TF-IDF features (max_features=5000) …")
    train_texts = _combine_text(train_df)
    test_texts = _combine_text(test_df)
    X_train, X_test, _vectorizer = extract_tfidf_features(train_texts, test_texts)
    y_train = train_df["rating"].values.astype(np.float32)
    test_ids = test_df["id"].values

    print(f"  X_train shape: {X_train.shape}  |  X_test shape: {X_test.shape}")

    # 3. Cross-validation ───────────────────────────────────────────────
    print("\n[3/5] 3-fold cross-validation …")
    mean_rmse = cv_rmse(X_train, y_train, n_splits=3)

    # 4. Train full model + predict ─────────────────────────────────────
    print("\n[4/5] Training full model on all training data …")
    timer = StageTimer()
    model = _train_model(X_train, y_train, stage_timer=timer)

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODEL_PATH))
    print(f"  Model saved → {MODEL_PATH}")

    print("\n[5/5] Predicting on test set …")
    submission = _predict(model, X_test, test_ids, str(SUBMISSION_PATH), stage_timer=timer)
    print(f"  Submission saved → {SUBMISSION_PATH}  ({len(submission):,} rows)")

    # 5. Write metrics ──────────────────────────────────────────────────
    timings = timer.to_dict().get("stage_0", {})
    metrics_update = {
        "stages": {
            "0": {
                "rmse": round(mean_rmse, 5),
                "train_time_sec": round(timings.get("train_time_sec", 0.0), 2),
                "inference_time_sec": round(timings.get("inference_time_sec", 0.0), 2),
            }
        }
    }
    write_metrics(str(METRICS_PATH), metrics_update)
    print(f"\n  Metrics updated → {METRICS_PATH}")
    print(f"  stage_0 RMSE: {mean_rmse:.5f}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

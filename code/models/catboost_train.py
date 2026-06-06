#!/usr/bin/env python
"""CatBoost base model for stacking (T18).

5-fold CV CatBoostRegressor on feature matrix.
Produces OOF predictions and test predictions for the stacking layer.

Memory budget: 64 GB (cgroup).  Full 5927-col matrix = 71 GB → doesn't fit.
We use non-TFIDF columns (927) = ~11 GB.  To stay under 64 GB, we reload
the parquet data for each fold so that the large X_train array is freed
before CatBoost builds its internal histograms.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold

# ── path setup ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.utils.timer import write_metrics

# ── constants ──────────────────────────────────────────────────────────
FEAT_DIR = ROOT / "artifacts" / "features"
MODEL_DIR = ROOT / "artifacts" / "models"
X_TRAIN_PATH = FEAT_DIR / "X_train.parquet"
X_TEST_PATH = FEAT_DIR / "X_test.parquet"
Y_TRAIN_PATH = FEAT_DIR / "y_train.npy"
METRICS_PATH = ROOT / "docs" / "changelog" / "metrics.json"

OOF_PATH = MODEL_DIR / "catboost_oof.npy"
TEST_PATH = MODEL_DIR / "catboost_test.npy"

CB_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "loss_function": "RMSE",
    "verbose": 100,
    "random_seed": 42,
    "thread_count": -1,
    "early_stopping_rounds": 100,
    "eval_metric": "RMSE",
}

RANDOM_SEED = 42
N_FOLDS = 5


# ── data loading ───────────────────────────────────────────────────────
def _select_columns(all_cols: List[str]) -> List[str]:
    """Return non-TFIDF column names."""
    return [c for c in all_cols if not c.startswith("tfidf_")]


def _read_parquet_cols(path: Path, columns: List[str]) -> np.ndarray:
    """Read selected columns from parquet → float32 numpy (row-group by row-group)."""
    pf = pq.ParquetFile(str(path))
    n_groups = pf.metadata.num_row_groups
    chunks: List[np.ndarray] = []
    for rg in range(n_groups):
        table = pf.read_row_group(rg, columns=columns)
        df = table.to_pandas()
        arr = df.values.astype(np.float32)
        del df, table
        chunks.append(arr)
        gc.collect()
    X = np.vstack(chunks)
    del chunks
    gc.collect()
    return X


def load_test(columns: List[str]) -> np.ndarray:
    """Load X_test (small, 10K rows) — stays in memory."""
    return _read_parquet_cols(X_TEST_PATH, columns)


def load_y() -> np.ndarray:
    """Load y_train."""
    return np.load(str(Y_TRAIN_PATH)).astype(np.float32)


# ── cross-validation + OOF ────────────────────────────────────────────
def train_catboost_oof(
    columns: List[str],
    y_all: np.ndarray,
    X_test: np.ndarray,
    params: dict,
    n_folds: int = 5,
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Train CatBoost K-fold CV, reloading X_train per fold to save memory.

    Returns (oof_preds, test_preds, fold_rmses).
    """
    n_train = len(y_all)
    oof_preds = np.zeros(n_train, dtype=np.float32)
    test_preds = np.zeros(len(X_test), dtype=np.float32)
    fold_rmses: List[float] = []

    test_pool = Pool(X_test)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    splits = list(kf.split(np.empty(n_train)))  # pre-compute splits

    for fold_idx, (tr_idx, va_idx) in enumerate(splits, 1):
        print(f"\n  ── Fold {fold_idx}/{n_folds} "
              f"(train={len(tr_idx):,}, val={len(va_idx):,}) ──")

        # Reload X_train for this fold (frees previous fold's data)
        print(f"  Loading X_train for fold {fold_idx} …")
        X_train = _read_parquet_cols(X_TRAIN_PATH, columns)
        print(f"  X_train: {X_train.shape}")

        # Build CatBoost Pools from slices
        train_pool = Pool(X_train[tr_idx], y_all[tr_idx])
        val_pool = Pool(X_train[va_idx], y_all[va_idx])

        # Free the big numpy array before training
        del X_train
        gc.collect()

        model = CatBoostRegressor(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            verbose=params.get("verbose", 100),
        )

        # OOF predictions
        va_preds = np.clip(model.predict(val_pool), 1.0, 5.0)
        oof_preds[va_idx] = va_preds
        fold_rmse = float(np.sqrt(np.mean((va_preds - y_all[va_idx]) ** 2)))
        fold_rmses.append(fold_rmse)
        print(f"  Fold {fold_idx} RMSE: {fold_rmse:.5f}")

        # Test predictions (accumulate for averaging)
        test_preds += np.clip(model.predict(test_pool), 1.0, 5.0) / n_folds

        del train_pool, val_pool, model, va_preds
        gc.collect()

    del test_pool
    gc.collect()
    return oof_preds, test_preds, fold_rmses


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("CatBoost base model for stacking (T18)")
    print("=" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Determine columns & load static data ──────────────────────────
    print("\n[1/4] Loading metadata …")
    pf = pq.ParquetFile(str(X_TRAIN_PATH))
    all_cols = pf.schema_arrow.names
    use_cols = _select_columns(all_cols)
    print(f"  Features: {len(use_cols)} / {len(all_cols)} columns "
          f"(dropped {len(all_cols) - len(use_cols)} TF-IDF)")

    y_train = load_y()
    print(f"  y_train: {y_train.shape}")

    print("  Loading X_test …")
    X_test = load_test(use_cols)
    print(f"  X_test: {X_test.shape}")

    # 2. 5-fold CV training ────────────────────────────────────────────
    print("\n[2/4] Training CatBoost (5-fold CV) …")
    start_train = time.perf_counter()

    oof_preds, test_preds, fold_rmses = train_catboost_oof(
        use_cols, y_train, X_test, CB_PARAMS, n_folds=N_FOLDS,
    )

    train_time = time.perf_counter() - start_train
    print(f"\n  Training completed in {train_time:.1f}s")

    # Compute OOF RMSE while y_train is in memory
    oof_preds = np.clip(oof_preds, 1.0, 5.0)
    oof_rmse = float(np.sqrt(np.mean((oof_preds - y_train) ** 2)))

    del X_test, y_train
    gc.collect()

    mean_fold_rmse = float(np.mean(fold_rmses))
    print(f"\n  Fold RMSEs: {[f'{r:.5f}' for r in fold_rmses]}")
    print(f"  Mean fold RMSE: {mean_fold_rmse:.5f}")
    print(f"  Overall OOF RMSE: {oof_rmse:.5f}")

    # 3. Save predictions ──────────────────────────────────────────────
    print("\n[3/4] Saving predictions …")
    np.save(str(OOF_PATH), oof_preds)
    print(f"  OOF → {OOF_PATH}  shape={oof_preds.shape}")

    np.save(str(TEST_PATH), test_preds)
    print(f"  Test → {TEST_PATH}  shape={test_preds.shape}")

    # 4. Update metrics.json ───────────────────────────────────────────
    print("\n[4/4] Updating metrics …")
    metrics_update = {
        "stages": {
            "catboost": {
                "oof_rmse": round(oof_rmse, 5),
                "mean_fold_rmse": round(mean_fold_rmse, 5),
                "fold_rmses": [round(r, 5) for r in fold_rmses],
                "train_time_sec": round(train_time, 2),
                "model": "catboost",
                "features_used": len(use_cols),
                "params": {
                    "iterations": CB_PARAMS["iterations"],
                    "learning_rate": CB_PARAMS["learning_rate"],
                    "depth": CB_PARAMS["depth"],
                    "n_folds": N_FOLDS,
                },
            }
        }
    }
    write_metrics(str(METRICS_PATH), metrics_update)
    print(f"  Metrics → {METRICS_PATH}")

    # Summary
    delta = 0.550 - oof_rmse
    print(f"\n  CatBoost OOF RMSE: {oof_rmse:.5f}  (Stage 2 LGB: 0.550)")
    print(f"  Δ vs Stage 2: {delta:+.5f}  "
          f"{'✅ improved' if delta > 0 else '⚠️  worse'}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

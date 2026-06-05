#!/usr/bin/env python
"""Stage 2: ALL features (TF-IDF + stats + embeddings) + LightGBM → Kaggle submission.

Loads pre-assembled feature matrices:
- X_train.parquet  (3,007,439 × 5927 features, 14 GB compressed)
- X_test.parquet   (10,000     × 5927 features)
- y_train.npy      (3,007,439,) target ratings

Memory budget: 64 GB process limit.  Strategy:
  • Read parquet row-group by row-group directly (no memmap)
  • CV on 1M-row subsample (~24 GB float32)
  • Final model via incremental batch training (7 row groups, init_model)
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from typing import List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import KFold

# ── path setup ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.utils.timer import StageTimer, timed, write_metrics

# ── constants ──────────────────────────────────────────────────────────
FEAT_DIR = ROOT / "artifacts" / "features"
X_TRAIN_PATH = FEAT_DIR / "X_train.parquet"
X_TEST_PATH = FEAT_DIR / "X_test.parquet"
Y_TRAIN_PATH = FEAT_DIR / "y_train.npy"

SUBMISSION_PATH = ROOT / "output" / "submission-stage2.csv"
METRICS_PATH = ROOT / "docs" / "changelog" / "metrics.json"

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "n_estimators": 500,
    "verbose": -1,
}

RANDOM_SEED = 42
N_ROUNDS = LGB_PARAMS["n_estimators"]


# ── helpers ────────────────────────────────────────────────────────────
def _read_row_group_to_numpy(path: Path, rg: int) -> Tuple[np.ndarray, List[str]]:
    """Read a single row group from parquet → float32 numpy array."""
    pf = pq.ParquetFile(str(path))
    table = pf.read_row_group(rg)
    df = table.to_pandas()
    col_names = list(df.columns)
    arr = df.values.astype(np.float32)
    del df, table
    return arr, col_names


def _read_parquet_schema(path: Path) -> Tuple[List[str], int, int]:
    """Read schema and row-group count without loading data."""
    pf = pq.ParquetFile(str(path))
    col_names = pf.schema_arrow.names
    n_rows = pf.metadata.num_rows
    n_groups = pf.metadata.num_row_groups
    return col_names, n_rows, n_groups


def _load_test(path: Path, etl_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load the small test set (10K rows) fully into RAM."""
    table = pq.read_table(str(path))
    df = table.to_pandas()
    col_names = list(df.columns)
    X_test = df.values.astype(np.float32)
    del df, table

    if etl_path.exists():
        test_ids = pd.read_parquet(etl_path, columns=["id"])["id"].values
    else:
        test_ids = np.arange(len(X_test))

    return X_test, test_ids, col_names


# ── data assembly ──────────────────────────────────────────────────────
def load_metadata():
    """Load metadata, test set, and y_train. Returns everything needed for
    later row-group-by-row-group processing."""
    train_cols, n_train, n_train_rg = _read_parquet_schema(X_TRAIN_PATH)
    test_cols, n_test, _ = _read_parquet_schema(X_TEST_PATH)

    print(f"  X_train: {n_train:,} rows × {len(train_cols)} cols ({n_train_rg} row groups)")
    print(f"  X_test:  {n_test:,} rows × {len(test_cols)} cols")

    # Column alignment
    common_cols = sorted(set(train_cols) & set(test_cols))
    if len(common_cols) < len(train_cols) or len(common_cols) < len(test_cols):
        print(f"  Common features: {len(common_cols)} "
              f"(dropped {len(train_cols) - len(common_cols)} train-only, "
              f"{len(test_cols) - len(common_cols)} test-only)")
    train_col_idx = np.array([train_cols.index(c) for c in common_cols])
    test_col_idx = np.array([test_cols.index(c) for c in common_cols])

    # Load y_train
    y_train = np.load(str(Y_TRAIN_PATH)).astype(np.float32)
    print(f"  y_train: {y_train.shape}")

    # Load test (small)
    etl_test = ROOT / "artifacts" / "etl" / "test.parquet"
    X_test_raw, test_ids, _ = _load_test(X_TEST_PATH, etl_test)
    X_test = X_test_raw[:, test_col_idx].copy()
    del X_test_raw
    gc.collect()

    print(f"  X_test aligned: {X_test.shape}")
    return common_cols, train_col_idx, test_col_idx, X_test, test_ids, y_train, n_train_rg


# ── cross-validation (on subsample from row groups) ───────────────────
def cv_rmse(
    train_path: Path,
    train_col_idx: np.ndarray,
    y_all: np.ndarray,
    n_train_rg: int,
    n_splits: int = 3,
    n_sample_rows: int = 1_000_000,
) -> float:
    """3-fold CV on a random subsample loaded from parquet row groups.

    Loads row groups into memory, subsamples, runs CV, frees everything.
    Peak memory: ~n_sample_rows × D × 4 bytes for the subsample array.
    """
    print(f"  Loading row groups for CV subsample ({n_sample_rows:,} rows) …")

    # Load row groups until we have enough rows
    chunks: List[np.ndarray] = []
    total_rows = 0
    pf = pq.ParquetFile(str(train_path))
    rg = 0
    while total_rows < n_sample_rows and rg < pf.metadata.num_row_groups:
        table = pf.read_row_group(rg)
        df = table.to_pandas()
        arr = df.values.astype(np.float32)
        del df, table
        # Select common columns
        arr_sel = arr[:, train_col_idx].copy()
        del arr
        chunks.append(arr_sel)
        total_rows += len(arr_sel)
        rg += 1
        print(f"    row group {rg}: +{len(arr_sel):,} rows (total: {total_rows:,})")
        gc.collect()

    # Stack and subsample
    X_pool = np.vstack(chunks)
    del chunks
    gc.collect()

    if len(X_pool) > n_sample_rows:
        rng = np.random.RandomState(RANDOM_SEED)
        idx = np.sort(rng.choice(len(X_pool), size=n_sample_rows, replace=False))
        X_sub = X_pool[idx]
        y_sub = y_all[idx]  # y_all is ordered same as parquet rows
        del X_pool
    else:
        X_sub = X_pool
        y_sub = y_all[:len(X_pool)]
        del X_pool
    gc.collect()
    print(f"  CV subsample: {X_sub.shape}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    rmses: List[float] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_sub), 1):
        ds_tr = lgb.Dataset(X_sub[tr_idx], y_sub[tr_idx], free_raw_data=True)
        ds_va = lgb.Dataset(X_sub[va_idx], y_sub[va_idx], free_raw_data=True)
        # Use fewer rounds for CV to keep wall-time manageable
        cv_rounds = min(N_ROUNDS, 200)
        model = lgb.train(
            LGB_PARAMS, ds_tr,
            num_boost_round=cv_rounds,
            valid_sets=[ds_va],
        )
        preds = np.clip(model.predict(X_sub[va_idx]), 1.0, 5.0)
        rmse = float(np.sqrt(np.mean((preds - y_sub[va_idx]) ** 2)))
        rmses.append(rmse)
        print(f"  fold {fold}: RMSE = {rmse:.5f}")
        del ds_tr, ds_va, model, preds
        gc.collect()

    del X_sub, y_sub
    gc.collect()

    mean_rmse = float(np.mean(rmses))
    print(f"  mean CV RMSE = {mean_rmse:.5f}")
    return mean_rmse


# ── incremental batch training ─────────────────────────────────────────
@timed("stage_2", "train_time_sec")
def _train_full_incremental(
    train_path: Path,
    train_col_idx: np.ndarray,
    y_all: np.ndarray,
    n_train_rg: int,
) -> lgb.Booster:
    """Train on all rows via row-group batches with init_model.

    Each row group is ~430K rows × 5927 × 4 ≈ 10.2 GB → LGB binned ≈ 3.4 GB.
    Peak per batch ≈ 14 GB.  Well within 64 GB.
    """
    rounds_per_rg = max(1, N_ROUNDS // n_train_rg)
    extra = N_ROUNDS - rounds_per_rg * n_train_rg

    pf = pq.ParquetFile(str(train_path))
    row_offset = 0
    model = None

    for rg in range(n_train_rg):
        n_rounds = rounds_per_rg + (extra if rg == n_train_rg - 1 else 0)

        print(f"  Row group {rg + 1}/{n_train_rg}: loading …")
        table = pf.read_row_group(rg)
        df = table.to_pandas()
        arr = df.values.astype(np.float32)
        del df, table
        X_batch = arr[:, train_col_idx].copy()
        del arr
        n_chunk = len(X_batch)
        y_batch = y_all[row_offset:row_offset + n_chunk]
        row_offset += n_chunk

        print(f"    {n_chunk:,} rows → training {n_rounds} rounds …")
        ds = lgb.Dataset(X_batch, y_batch, free_raw_data=True)
        model = lgb.train(
            LGB_PARAMS, ds,
            num_boost_round=n_rounds,
            init_model=model,
        )
        del X_batch, y_batch, ds
        gc.collect()
        print(f"    done — total trees: {model.num_trees()}")

    return model


@timed("stage_2", "inference_time_sec")
def _predict_and_save(
    model: lgb.Booster,
    X_test: np.ndarray,
    test_ids: np.ndarray,
    output_path: str,
) -> pd.DataFrame:
    preds = np.clip(model.predict(X_test), 1.0, 5.0)
    submission = pd.DataFrame({"id": test_ids, "rating": preds})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("Stage 2: ALL features (multimodal) + LightGBM")
    print("=" * 60)

    # 1. Load metadata & test ───────────────────────────────────────────
    print("\n[1/5] Loading metadata & test set …")
    t0 = time.perf_counter()
    col_names, train_col_idx, test_col_idx, X_test, test_ids, y_train, n_train_rg = \
        load_metadata()
    load_time = time.perf_counter() - t0
    print(f"  Prepared in {load_time:.1f}s")

    # 2. Cross-validation ───────────────────────────────────────────────
    print("\n[2/5] 3-fold cross-validation (1M-row subsample) …")
    mean_rmse = cv_rmse(
        X_TRAIN_PATH, train_col_idx, y_train, n_train_rg,
        n_splits=3, n_sample_rows=500_000,
    )

    # 3. Train full model (incremental row-group batches) ───────────────
    print("\n[3/5] Training full model (incremental, per row group) …")
    timer = StageTimer()
    model = _train_full_incremental(
        X_TRAIN_PATH, train_col_idx, y_train, n_train_rg, stage_timer=timer,
    )

    # 4. Predict + save submission ──────────────────────────────────────
    print("\n[4/5] Predicting on test set …")
    submission = _predict_and_save(
        model, X_test, test_ids, str(SUBMISSION_PATH), stage_timer=timer,
    )
    print(f"  Submission → {SUBMISSION_PATH}  ({len(submission):,} rows)")

    # 5. Update metrics.json ────────────────────────────────────────────
    print("\n[5/5] Updating metrics …")
    timings = timer.to_dict().get("stage_2", {})
    metrics_update = {
        "stages": {
            "2": {
                "rmse": round(mean_rmse, 5),
                "train_time_sec": round(timings.get("train_time_sec", 0.0), 2),
                "inference_time_sec": round(timings.get("inference_time_sec", 0.0), 2),
                "model": "lgb_multimodal",
                "features": [
                    "all",
                    "tfidf",
                    "user_stats",
                    "prod_stats",
                    "temporal",
                    "text_length",
                    "te_user",
                    "te_prod",
                    "bert_embeddings",
                    "price_features",
                    "category_stats",
                ],
            }
        }
    }
    write_metrics(str(METRICS_PATH), metrics_update)
    print(f"  Metrics → {METRICS_PATH}")

    # ── feature importance ─────────────────────────────────────────────
    print("\n  Top-20 Feature Importance (gain):")
    importance = model.feature_importance(importance_type="gain")
    imp_df = (
        pd.DataFrame({"feature": col_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    for _, row in imp_df.iterrows():
        print(f"    {row['feature']:40s} {row['importance']:>12,.0f}")

    # Save importance for the report
    imp_path = ROOT / "artifacts" / "features" / "stage2_feature_importance.csv"
    imp_df_full = (
        pd.DataFrame({"feature": col_names, "importance": importance})
        .sort_values("importance", ascending=False)
    )
    imp_df_full.to_csv(imp_path, index=False)
    print(f"  Full importance → {imp_path}")

    print(f"\n  stage_2 RMSE: {mean_rmse:.5f}  (stage_0: 1.17626  stage_1: 0.54975)")
    delta_vs_s1 = 0.54975 - mean_rmse
    print(f"  Δ vs stage_1: {delta_vs_s1:+.5f}  {'✅ improved' if delta_vs_s1 > 0 else '⚠️  worse'}")
    delta_vs_s0 = 1.17626 - mean_rmse
    print(f"  Δ vs stage_0: {delta_vs_s0:+.5f}  {'✅ improved' if delta_vs_s0 > 0 else '⚠️  worse'}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

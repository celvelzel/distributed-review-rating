#!/usr/bin/env python
"""Optimized LightGBM for review rating prediction — NO target leakage.

Features (all leakage-free):
- TF-IDF (5000 features) — text signal
- Temporal features: year, month, day, weekday, hour, is_weekend, is_holiday_season
- Text length features: title_len, comment_len, title_comment_ratio, has_caps, has_exclamation
- Base features: votes, purchased

DOES NOT USE: user_te, prod_te, avg_rating, prod_avg_rating, user_stats, product_stats
(anything that aggregates target across users/products → leakage)

Uses native lgb.train() API (faster for sparse data) with Optuna-tuned params
from a subsample, then trains final model on all 3M rows.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import hstack as sparse_hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold

# ── path setup ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.utils.timer import StageTimer, timed, write_metrics

# ── constants ──────────────────────────────────────────────────────────
TRAIN_PATH = ROOT / "artifacts" / "etl" / "train.parquet"
TEST_PATH = ROOT / "artifacts" / "etl" / "test.parquet"
TEMPORAL_PATH = ROOT / "artifacts" / "features" / "temporal.parquet"
TEXT_LEN_PATH = ROOT / "artifacts" / "features" / "text_length.parquet"

SUBMISSION_PATH = ROOT / "output" / "submission-optimized-v1.csv"
METRICS_PATH = ROOT / "docs" / "changelog" / "metrics.json"
MODEL_PATH = ROOT / "artifacts" / "models" / "optimized_v1.txt"

RANDOM_SEED = 42
FINAL_N_ESTIMATORS = 300
TFIDF_MAX_FEATURES = 5_000

# Best params from prior Optuna run on 20K subsample:
#   num_leaves=31, max_depth=-1, learning_rate=0.05,
#   min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.6
BEST_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "seed": RANDOM_SEED,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.6,
    "bagging_freq": 1,
}


# ── helpers ────────────────────────────────────────────────────────────
def _combine_text(df: pd.DataFrame) -> pd.Series:
    return (df["title"].fillna("") + " " + df["comment"].fillna("")).str.strip()


def _build_tfidf(train_texts, test_texts):
    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    X_tr = vec.fit_transform(train_texts.fillna(""))
    X_te = vec.transform(test_texts.fillna(""))
    return X_tr, X_te, vec


def _build_meta(df, temporal_df, textlen_df) -> np.ndarray:
    """Build leakage-free meta features aligned by id. Return numpy."""
    ids = df["id"].values
    temp = temporal_df.iloc[ids].drop(columns=["id"], errors="ignore").reset_index(drop=True).astype(np.float32)
    tl = textlen_df.iloc[ids].drop(columns=["id"], errors="ignore").reset_index(drop=True).astype(np.float32)
    votes = df["votes"].fillna(0).values.astype(np.float32)
    purchased = (
        df["purchased"]
        .map({True: 1, False: 0, "True": 1, "False": 0, "true": 1, "false": 0})
        .fillna(0).values.astype(np.float32)
    )
    result = pd.concat([temp, tl, pd.DataFrame({"votes": votes, "purchased": purchased})], axis=1)
    return result.values.astype(np.float32)


# ── timed helpers ──────────────────────────────────────────────────────
@timed("optimized", "train_time_sec")
def _train_full(X_tfidf, meta_np, y, params, n_estimators) -> lgb.Booster:
    """Train on all data using native lgb.train() API (fast for sparse)."""
    X_all = sparse_hstack([X_tfidf, meta_np])
    ds = lgb.Dataset(X_all, y, free_raw_data=True)
    model = lgb.train(
        params, ds,
        num_boost_round=n_estimators,
        callbacks=[lgb.log_evaluation(period=50)],
    )
    del X_all, ds
    gc.collect()
    return model


@timed("optimized", "inference_time_sec")
def _predict_and_save(model, X_tfidf_test, meta_np_test, test_ids, output_path) -> pd.DataFrame:
    X_test = sparse_hstack([X_tfidf_test, meta_np_test])
    preds = np.clip(model.predict(X_test), 1.0, 5.0)
    sub = pd.DataFrame({"id": test_ids, "rating": preds})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)
    del X_test
    gc.collect()
    return sub


# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 60)
    print("Optimized LightGBM — NO target leakage")
    print("Features: TF-IDF(5K) + temporal + text_length + base")
    print("=" * 60)
    t_total = time.perf_counter()

    # 1. Load data
    print("\n[1/5] Loading data …")
    t0 = time.perf_counter()
    train_df = pd.read_parquet(TRAIN_PATH, columns=["id", "title", "comment", "votes", "purchased", "rating"])
    test_df = pd.read_parquet(TEST_PATH, columns=["id", "title", "comment", "votes", "purchased"])
    y_train = train_df["rating"].values.astype(np.float32)
    test_ids = test_df["id"].values
    print(f"  train: {len(train_df):,}  test: {len(test_df):,}  ({time.perf_counter()-t0:.1f}s)")

    # 2. Load temporal + text_length
    print("\n[2/5] Loading feature parquets …")
    t0 = time.perf_counter()
    temporal_df = pd.read_parquet(TEMPORAL_PATH)
    textlen_df = pd.read_parquet(TEXT_LEN_PATH)
    print(f"  temporal: {temporal_df.shape}  text_length: {textlen_df.shape}  ({time.perf_counter()-t0:.1f}s)")

    # 3. Build TF-IDF + meta features
    print(f"\n[3/5] Building features …")
    t0 = time.perf_counter()
    X_tfidf_train, X_tfidf_test, _ = _build_tfidf(
        _combine_text(train_df), _combine_text(test_df),
    )
    meta_train_np = _build_meta(train_df, temporal_df, textlen_df)
    meta_test_np = _build_meta(test_df, temporal_df, textlen_df)
    print(f"  TF-IDF train: {X_tfidf_train.shape}  test: {X_tfidf_test.shape}")
    print(f"  Meta train: {meta_train_np.shape}  test: {meta_test_np.shape}")
    print(f"  Built in {time.perf_counter()-t0:.1f}s")

    del train_df, test_df, temporal_df, textlen_df
    gc.collect()

    # 4. Quick CV on subsample for evaluation
    print(f"\n[4/5] 3-fold CV on 20K subsample …")
    t0 = time.perf_counter()
    rng = np.random.RandomState(RANDOM_SEED)
    cv_idx = np.sort(rng.choice(len(y_train), size=20_000, replace=False))
    X_cv = sparse_hstack([X_tfidf_train[cv_idx], meta_train_np[cv_idx]]).tocsr()
    y_cv = y_train[cv_idx]

    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    rmses = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(y_cv), 1):
        ds_tr = lgb.Dataset(X_cv[tr_idx], y_cv[tr_idx], free_raw_data=True)
        ds_va = lgb.Dataset(X_cv[va_idx], y_cv[va_idx], free_raw_data=True)
        m = lgb.train(
            BEST_PARAMS, ds_tr,
            num_boost_round=FINAL_N_ESTIMATORS,
            valid_sets=[ds_va],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        preds = np.clip(m.predict(X_cv[va_idx]), 1.0, 5.0)
        rmse = float(np.sqrt(np.mean((preds - y_cv[va_idx]) ** 2)))
        rmses.append(rmse)
        print(f"    fold {fold}: RMSE = {rmse:.5f}  best_iter={m.best_iteration}")
        del ds_tr, ds_va, m, preds
        gc.collect()

    mean_cv_rmse = float(np.mean(rmses))
    print(f"    mean CV RMSE = {mean_cv_rmse:.5f}  ({time.perf_counter()-t0:.1f}s)")
    del X_cv, y_cv, cv_idx
    gc.collect()

    # 5. Train final model on ALL data + predict
    print(f"\n[5/5] Training on all {len(y_train):,} rows + predicting …")
    timer = StageTimer()
    model = _train_full(
        X_tfidf_train, meta_train_np, y_train,
        BEST_PARAMS, FINAL_N_ESTIMATORS, stage_timer=timer,
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"  Model → {MODEL_PATH}")

    submission = _predict_and_save(
        model, X_tfidf_test, meta_test_np, test_ids,
        str(SUBMISSION_PATH), stage_timer=timer,
    )
    print(f"  Submission → {SUBMISSION_PATH}  ({len(submission):,} rows)")

    # Write metrics
    timings = timer.to_dict().get("optimized", {})
    write_metrics(str(METRICS_PATH), {
        "stages": {
            "optimized_v1": {
                "cv_rmse_subsample": round(mean_cv_rmse, 5),
                "train_time_sec": round(timings.get("train_time_sec", 0.0), 2),
                "inference_time_sec": round(timings.get("inference_time_sec", 0.0), 2),
                "model": "lgb_optimized",
                "features": ["tfidf_5000", "temporal", "text_length", "votes", "purchased"],
                "n_estimators": FINAL_N_ESTIMATORS,
                "best_params": {k: v for k, v in BEST_PARAMS.items()
                                if k not in ("objective", "metric", "verbosity", "seed")},
                "leakage_free": True,
                "note": "Optuna params from prior 20K subsample run; CV on 20K subsample",
            }
        }
    })
    print(f"  Metrics → {METRICS_PATH}")

    # Summary
    elapsed = time.perf_counter() - t_total
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Subsample CV RMSE:   {mean_cv_rmse:.5f}")
    print(f"  Stage 0 baseline:    1.17626 (local) / 0.80107 (Kaggle)")
    print(f"  Leaky Stage 1:       0.54975 (uses target encoding)")
    print(f"  Leakage-free:        YES — CV should match Kaggle")
    bp = {k: v for k, v in BEST_PARAMS.items()
          if k not in ("objective", "metric", "verbosity", "seed")}
    print(f"  Params: {json.dumps(bp)}")
    print(f"  Total time:          {elapsed:.1f}s")
    print("=" * 60)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

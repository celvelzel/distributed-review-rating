#!/usr/bin/env python
"""Advanced TF-IDF + Model Optimization.

Goal: Beat 0.79012 (current best) and approach 0.62 (competitor).
Strategy: Multiple TF-IDF representations + model diversity + ensemble.

Key insights from history:
- TF-IDF features generalize well (no target leakage)
- Statistical features (user_te, prod_te, avg_rating) leak target info
- Regularization helps (subsample=0.8, colsample=0.8)
- Bigrams may capture important phrases
"""

import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAIN_PATH = ROOT / "artifacts" / "etl" / "train.parquet"
TEST_PATH = ROOT / "artifacts" / "etl" / "test.parquet"
SUBMISSION_DIR = ROOT / "output"
METRICS_PATH = ROOT / "docs" / "changelog" / "metrics.json"

SEED = 42


def combine_text(df: pd.DataFrame) -> pd.Series:
    """Combine title and comment into single text."""
    return (df["title"].fillna("") + " " + df["comment"].fillna("")).str.strip()


def get_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from time column (no leakage)."""
    ts = pd.to_datetime(df["time"], unit="ms", errors="coerce")
    return pd.DataFrame({
        "year": ts.dt.year.fillna(0).astype(int),
        "month": ts.dt.month.fillna(0).astype(int),
        "day": ts.dt.day.fillna(0).astype(int),
        "weekday": ts.dt.weekday.fillna(0).astype(int),
        "hour": ts.dt.hour.fillna(0).astype(int),
        "is_weekend": (ts.dt.weekday >= 5).astype(int).fillna(0),
    })


def get_text_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract text length features (no leakage)."""
    title = df["title"].fillna("").astype(str)
    comment = df["comment"].fillna("").astype(str)
    return pd.DataFrame({
        "title_len": title.str.len(),
        "comment_len": comment.str.len(),
        "title_word_count": title.str.split().str.len(),
        "comment_word_count": comment.str.split().str.len(),
        "title_comment_ratio": title.str.len() / (comment.str.len() + 1),
        "has_caps": title.str.contains(r"[A-Z]").astype(int),
        "has_exclamation": (title.str.contains("!") | comment.str.contains("!")).astype(int),
        "has_question": (title.str.contains(r"\?") | comment.str.contains(r"\?")).astype(int),
        "has_ellipsis": (title.str.contains(r"\.\.\.") | comment.str.contains(r"\.\.\.")).astype(int),
    })


def get_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract base features (no leakage)."""
    return pd.DataFrame({
        "votes": df["votes"].fillna(0).astype(float),
        "purchased": df["purchased"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int),
    })


def get_tfidf_configs():
    """Define TF-IDF configurations to try."""
    return [
        # Current best baseline
        {"name": "tfidf_5k_baseline", "max_features": 5000, "ngram_range": (1, 1), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        
        # More features
        {"name": "tfidf_10k", "max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        {"name": "tfidf_15k", "max_features": 15000, "ngram_range": (1, 1), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        {"name": "tfidf_20k", "max_features": 20000, "ngram_range": (1, 1), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        
        # Bigrams
        {"name": "tfidf_10k_bi", "max_features": 10000, "ngram_range": (1, 2), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        {"name": "tfidf_15k_bi", "max_features": 15000, "ngram_range": (1, 2), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        {"name": "tfidf_20k_bi", "max_features": 20000, "ngram_range": (1, 2), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        
        # Trigrams
        {"name": "tfidf_15k_tri", "max_features": 15000, "ngram_range": (1, 3), "sublinear_tf": True, "min_df": 1, "max_df": 1.0},
        
        # With min_df filtering
        {"name": "tfidf_10k_mindf2", "max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True, "min_df": 2, "max_df": 0.95},
        {"name": "tfidf_10k_mindf5", "max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": True, "min_df": 5, "max_df": 0.9},
        
        # Without sublinear_tf
        {"name": "tfidf_10k_nosub", "max_features": 10000, "ngram_range": (1, 1), "sublinear_tf": False, "min_df": 1, "max_df": 1.0},
    ]


def get_lgb_configs():
    """Define LightGBM configurations to try."""
    return [
        # Current best baseline
        {"name": "lgb_baseline", "n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1},
        
        # More trees, lower learning rate
        {"name": "lgb_slow", "n_estimators": 1000, "num_leaves": 127, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1},
        
        # More leaves
        {"name": "lgb_255leaves", "n_estimators": 500, "num_leaves": 255, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1},
        
        # Stronger regularization
        {"name": "lgb_regularized", "n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 0.5},
        
        # Fast learner
        {"name": "lgb_fast", "n_estimators": 300, "num_leaves": 127, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1},
    ]


def cv_evaluate(X, y, model_class, params, n_folds=3, n_sample=200_000):
    """Fast CV on subsample."""
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(y), size=min(n_sample, len(y)), replace=False)
    X_sub = X[idx] if issparse(X) else X[idx]
    y_sub = y[idx]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    rmses = []
    for tr, va in kf.split(X_sub):
        model = model_class(**params)
        model.fit(X_sub[tr], y_sub[tr])
        preds = np.clip(model.predict(X_sub[va]), 1.0, 5.0)
        rmse = float(np.sqrt(np.mean((preds - y_sub[va]) ** 2)))
        rmses.append(rmse)
    return float(np.mean(rmses))


def train_and_predict(X_train, y_train, X_test, model_class, params):
    """Train model and return test predictions."""
    model = model_class(**params)
    model.fit(X_train, y_train)
    return np.clip(model.predict(X_test), 1.0, 5.0)


def main():
    print("=" * 70)
    print("Advanced TF-IDF + Model Optimization")
    print("Goal: Beat 0.79012 (current best) → approach 0.62 (competitor)")
    print("=" * 70)
    t_start = time.time()

    # Load data
    print("\n[1/6] Loading data …")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    y_train = train_df["rating"].values.astype(np.float32)
    test_ids = test_df["id"].values
    print(f"  train: {len(train_df):,}  |  test: {len(test_df):,}")

    train_texts = combine_text(train_df)
    test_texts = combine_text(test_df)

    # Get leakage-free features
    print("\n[2/6] Extracting leakage-free features …")
    train_temporal = get_temporal_features(train_df)
    test_temporal = get_temporal_features(test_df)
    train_textlen = get_text_length_features(train_df)
    test_textlen = get_text_length_features(test_df)
    train_base = get_base_features(train_df)
    test_base = get_base_features(test_df)

    # Combine all leakage-free features
    train_extra = pd.concat([train_temporal, train_textlen, train_base], axis=1).fillna(0).values.astype(np.float32)
    test_extra = pd.concat([test_temporal, test_textlen, test_base], axis=1).fillna(0).values.astype(np.float32)
    print(f"  Extra features: {train_extra.shape[1]}")

    # Phase 1: Find best TF-IDF config with LightGBM
    print("\n[3/6] Phase 1: Finding best TF-IDF config …")
    tfidf_configs = get_tfidf_configs()
    lgb_baseline = {"objective": "regression", "metric": "rmse", "verbose": -1, "n_jobs": -1, "random_seed": SEED,
                    "n_estimators": 500, "num_leaves": 127, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}
    
    tfidf_results = []
    for i, tfidf_cfg in enumerate(tfidf_configs, 1):
        name = tfidf_cfg["name"]
        print(f"\n  [{i}/{len(tfidf_configs)}] {name}")
        
        t0 = time.time()
        vec = TfidfVectorizer(max_features=tfidf_cfg["max_features"],
                              ngram_range=tfidf_cfg["ngram_range"],
                              sublinear_tf=tfidf_cfg["sublinear_tf"],
                              min_df=tfidf_cfg["min_df"],
                              max_df=tfidf_cfg["max_df"],
                              strip_accents="unicode",
                              dtype=np.float32)
        X_tfidf = vec.fit_transform(train_texts.fillna(""))
        
        rmse = cv_evaluate(X_tfidf, y_train, lgb.LGBMRegressor, lgb_baseline, n_folds=3, n_sample=200_000)
        elapsed = time.time() - t0
        
        tfidf_results.append({"name": name, "rmse": rmse, "cfg": tfidf_cfg, "features": X_tfidf.shape[1]})
        print(f"    RMSE = {rmse:.5f}  ({elapsed:.1f}s, {X_tfidf.shape[1]} features)")

    # Sort and show top TF-IDF configs
    tfidf_results.sort(key=lambda x: x["rmse"])
    print(f"\n  Top 5 TF-IDF configs:")
    for rank, r in enumerate(tfidf_results[:5], 1):
        print(f"    {rank}. {r['name']:30s} RMSE = {r['rmse']:.5f} ({r['features']} features)")

    best_tfidf_cfg = tfidf_results[0]["cfg"]

    # Phase 2: Find best LightGBM config with best TF-IDF
    print("\n[4/6] Phase 2: Finding best LightGBM config …")
    vec = TfidfVectorizer(max_features=best_tfidf_cfg["max_features"],
                          ngram_range=best_tfidf_cfg["ngram_range"],
                          sublinear_tf=best_tfidf_cfg["sublinear_tf"],
                          min_df=best_tfidf_cfg["min_df"],
                          max_df=best_tfidf_cfg["max_df"],
                          strip_accents="unicode",
                          dtype=np.float32)
    X_tfidf = vec.fit_transform(train_texts.fillna(""))
    
    lgb_configs = get_lgb_configs()
    lgb_results = []
    for i, lgb_cfg in enumerate(lgb_configs, 1):
        name = lgb_cfg["name"]
        print(f"\n  [{i}/{len(lgb_configs)}] {name}")
        
        t0 = time.time()
        params = {"objective": "regression", "metric": "rmse", "verbose": -1, "n_jobs": -1, "random_seed": SEED, **lgb_cfg}
        rmse = cv_evaluate(X_tfidf, y_train, lgb.LGBMRegressor, params, n_folds=3, n_sample=200_000)
        elapsed = time.time() - t0
        
        lgb_results.append({"name": name, "rmse": rmse, "cfg": lgb_cfg})
        print(f"    RMSE = {rmse:.5f}  ({elapsed:.1f}s)")

    # Sort and show top LightGBM configs
    lgb_results.sort(key=lambda x: x["rmse"])
    print(f"\n  Top 3 LightGBM configs:")
    for rank, r in enumerate(lgb_results[:3], 1):
        print(f"    {rank}. {r['name']:30s} RMSE = {r['rmse']:.5f}")

    best_lgb_cfg = lgb_results[0]["cfg"]

    # Phase 3: Try XGBoost and CatBoost if available
    print("\n[5/6] Phase 3: Trying alternative models …")
    model_results = []
    
    # XGBoost
    if HAS_XGB:
        print("\n  Trying XGBoost …")
        xgb_params = {"objective": "reg:squarederror", "eval_metric": "rmse", "verbosity": 0, "n_jobs": -1,
                      "random_state": SEED, "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
                      "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1}
        rmse = cv_evaluate(X_tfidf, y_train, xgb.XGBRegressor, xgb_params, n_folds=3, n_sample=200_000)
        model_results.append({"name": "xgboost", "rmse": rmse, "class": xgb.XGBRegressor, "params": xgb_params})
        print(f"    XGBoost RMSE = {rmse:.5f}")
    
    # CatBoost
    if HAS_CATBOOST:
        print("\n  Trying CatBoost …")
        cat_params = {"iterations": 500, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3,
                      "random_seed": SEED, "verbose": 0, "task_type": "CPU"}
        rmse = cv_evaluate(X_tfidf, y_train, CatBoostRegressor, cat_params, n_folds=3, n_sample=200_000)
        model_results.append({"name": "catboost", "rmse": rmse, "class": CatBoostRegressor, "params": cat_params})
        print(f"    CatBoost RMSE = {rmse:.5f}")

    # Phase 4: Ensemble of best models
    print("\n[6/6] Phase 4: Building ensemble …")
    
    # Train best LightGBM on full data
    print("\n  Training best LightGBM on full data …")
    best_lgb_params = {"objective": "regression", "metric": "rmse", "verbose": -1, "n_jobs": -1, "random_seed": SEED, **best_lgb_cfg}
    X_tfidf_train = vec.fit_transform(train_texts.fillna(""))
    X_tfidf_test = vec.transform(test_texts.fillna(""))
    
    preds_lgb = train_and_predict(X_tfidf_train, y_train, X_tfidf_test, lgb.LGBMRegressor, best_lgb_params)
    print(f"    LightGBM predictions: mean={preds_lgb.mean():.3f}, std={preds_lgb.std():.3f}")
    
    # Train XGBoost if available
    if HAS_XGB:
        print("\n  Training XGBoost on full data …")
        xgb_params_full = {"objective": "reg:squarederror", "eval_metric": "rmse", "verbosity": 0, "n_jobs": -1,
                           "random_state": SEED, "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
                           "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 0.1}
        preds_xgb = train_and_predict(X_tfidf_train, y_train, X_tfidf_test, xgb.XGBRegressor, xgb_params_full)
        print(f"    XGBoost predictions: mean={preds_xgb.mean():.3f}, std={preds_xgb.std():.3f}")
    
    # Train CatBoost if available
    if HAS_CATBOOST:
        print("\n  Training CatBoost on full data …")
        cat_params_full = {"iterations": 500, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3,
                           "random_seed": SEED, "verbose": 0, "task_type": "CPU"}
        preds_cat = train_and_predict(X_tfidf_train, y_train, X_tfidf_test, CatBoostRegressor, cat_params_full)
        print(f"    CatBoost predictions: mean={preds_cat.mean():.3f}, std={preds_cat.std():.3f}")
    
    # Create ensemble predictions
    print("\n  Creating ensemble predictions …")
    all_preds = [preds_lgb]
    model_names = ["lgb"]
    
    if HAS_XGB:
        all_preds.append(preds_xgb)
        model_names.append("xgb")
    if HAS_CATBOOST:
        all_preds.append(preds_cat)
        model_names.append("cat")
    
    # Simple average ensemble
    preds_ensemble = np.mean(all_preds, axis=0)
    preds_ensemble = np.clip(preds_ensemble, 1.0, 5.0)
    print(f"    Ensemble predictions: mean={preds_ensemble.mean():.3f}, std={preds_ensemble.std():.3f}")
    
    # Weighted ensemble (give more weight to better models)
    weights = []
    for name in model_names:
        if name == "lgb":
            weights.append(0.5)  # LightGBM gets highest weight
        elif name == "xgb":
            weights.append(0.3)
        elif name == "cat":
            weights.append(0.2)
    
    preds_weighted = np.average(all_preds, axis=0, weights=weights)
    preds_weighted = np.clip(preds_weighted, 1.0, 5.0)
    print(f"    Weighted ensemble: mean={preds_weighted.mean():.3f}, std={preds_weighted.std():.3f}")
    
    # Save all submissions
    print("\n  Saving submissions …")
    submissions = {
        "lgb_best": preds_lgb,
        "ensemble_avg": preds_ensemble,
        "ensemble_weighted": preds_weighted,
    }
    
    if HAS_XGB:
        submissions["xgb"] = preds_xgb
    if HAS_CATBOOST:
        submissions["catboost"] = preds_cat
    
    submission_files = {}
    for name, preds in submissions.items():
        sub = pd.DataFrame({"id": test_ids, "rating": preds})
        sub_path = SUBMISSION_DIR / f"submission-advanced-{name}.csv"
        sub.to_csv(sub_path, index=False)
        submission_files[name] = str(sub_path)
        print(f"    {name}: {sub_path}")
    
    # Summary
    total = time.time() - t_start
    print(f"\n{'='*70}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Best TF-IDF config: {best_tfidf_cfg['name']}")
    print(f"  Best LightGBM config: {best_lgb_cfg['name']}")
    print(f"\n  TF-IDF Results (top 3):")
    for rank, r in enumerate(tfidf_results[:3], 1):
        print(f"    {rank}. {r['name']:30s} RMSE = {r['rmse']:.5f}")
    print(f"\n  LightGBM Results (top 3):")
    for rank, r in enumerate(lgb_results[:3], 1):
        print(f"    {rank}. {r['name']:30s} RMSE = {r['rmse']:.5f}")
    print(f"\n  Submissions saved to: {SUBMISSION_DIR}")
    print(f"  Total time: {total:.0f}s")
    print(f"\n  Recommended submission: submission-advanced-ensemble_weighted.csv")
    print(f"{'='*70}")
    
    # Save metrics
    metrics = {
        "advanced_optimization": {
            "best_tfidf_config": best_tfidf_cfg["name"],
            "best_lgb_config": best_lgb_cfg["name"],
            "tfidf_results": [{"name": r["name"], "rmse": round(r["rmse"], 5)} for r in tfidf_results[:5]],
            "lgb_results": [{"name": r["name"], "rmse": round(r["rmse"], 5)} for r in lgb_results[:3]],
            "model_results": [{"name": r["name"], "rmse": round(r["rmse"], 5)} for r in model_results],
            "submissions": submission_files,
        }
    }
    try:
        with open(METRICS_PATH) as f:
            existing = json.load(f)
        existing.update(metrics)
        with open(METRICS_PATH, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        print(f"  Warning: Could not update metrics: {e}")


if __name__ == "__main__":
    main()

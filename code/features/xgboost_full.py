"""
XGBoost on full data for graph features.
Optimized for speed to avoid timeouts.
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

FEAT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "features")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def load_data():
    """Load all available features."""
    train_df = pd.read_parquet(os.path.join(PROJECT_ROOT, "artifacts", "etl", "train.parquet"))
    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "test.csv"))
    y = train_df['rating'].values.astype(np.float32)

    exp_train = pd.read_parquet(os.path.join(FEAT_DIR, "expanded_graph_train.parquet"))
    exp_test = pd.read_parquet(os.path.join(FEAT_DIR, "expanded_graph_test.parquet"))
    us = pd.read_parquet(os.path.join(FEAT_DIR, "user_stats_kfold.parquet"))
    ps = pd.read_parquet(os.path.join(FEAT_DIR, "product_stats_kfold.parquet"))

    us_dedup = us.groupby('id').first().reset_index()
    ps_dedup = ps.groupby('parent_prod_id').first().reset_index()
    us_dict = us_dedup.set_index('id')
    ps_dict = ps_dedup.set_index('parent_prod_id')

    expanded_cols = [c for c in exp_train.columns if c not in ['id', 'parent_prod_id']]
    features_train = {}
    features_test = {}

    for col in expanded_cols:
        features_train[col] = exp_train[col].values.astype(np.float32)
        features_test[col] = exp_test[col].values.astype(np.float32)

    for col in ['avg_rating', 'num_reviews', 'rating_std']:
        if col in us_dict.columns:
            features_train[f'user_{col}'] = train_df['user_id'].map(us_dict[col]).fillna(0).values.astype(np.float32)
            features_test[f'user_{col}'] = test_df['user_id'].map(us_dict[col]).fillna(0).values.astype(np.float32)

    for col in ['prod_avg_rating', 'prod_num_reviews']:
        if col in ps_dict.columns:
            features_train[col] = train_df['parent_prod_id'].map(ps_dict[col]).fillna(0).values.astype(np.float32)
            features_test[col] = test_df['parent_prod_id'].map(ps_dict[col]).fillna(0).values.astype(np.float32)

    X_train = pd.DataFrame(features_train)
    X_test = pd.DataFrame(features_test)

    for X in [X_train, X_test]:
        X['leniency_x_reviews'] = X['user_leniency'] * X['user_num_reviews_oof']
        X['cat_dev_x_reviews'] = X['user_cat_deviation'] * X['user_cat_review_count']
        X['user_prod_diff'] = X['user_avg_rating'] - X['prod_avg_rating']

    return X_train, X_test, y, train_df, test_df


def main():
    t_start = time.time()
    print("=" * 70)
    print("XGBoost on Full Data (CPU)")
    print("=" * 70)

    X_train, X_test, y, train_df, test_df = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Features: {list(X_train.columns)}")
    print()

    # XGBoost with fast settings
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
        'tree_method': 'hist',
        'verbosity': 0,
    }

    print("Training XGBoost (5 folds)...")
    for fold, (tr, va) in enumerate(kf.split(X_train)):
        fold_start = time.time()

        dtrain = xgb.DMatrix(X_train.iloc[tr], label=y[tr])
        dval = xgb.DMatrix(X_train.iloc[va], label=y[va])
        dtest = xgb.DMatrix(X_test)

        model = xgb.train(params, dtrain, num_boost_round=300,
                         evals=[(dval, 'val')],
                         early_stopping_rounds=30, verbose_eval=False)

        oof[va] = model.predict(dval)
        test_preds += model.predict(dtest) / 5

        fold_rmse = np.sqrt(np.mean((y[va] - oof[va]) ** 2))
        print(f"  Fold {fold+1}: RMSE = {fold_rmse:.6f} ({time.time()-fold_start:.1f}s)")

    rmse = np.sqrt(np.mean((y - oof) ** 2))
    print(f"\n  XGBoost OOF RMSE = {rmse:.6f}")
    print(f"  Test predictions: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")
    print()

    # Save predictions
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(os.path.join(FEAT_DIR, "xgboost_expanded_oof.npy"), oof)
    np.save(os.path.join(FEAT_DIR, "xgboost_expanded_test.npy"), test_preds)

    submission = pd.DataFrame({
        "id": test_df["id"].values,
        "rating": np.clip(test_preds, 1.0, 5.0)
    })
    sub_path = os.path.join(OUTPUT_DIR, "xgboost_expanded_features.csv")
    submission.to_csv(sub_path, index=False)

    print(f"  Saved OOF: {os.path.join(FEAT_DIR, 'xgboost_expanded_oof.npy')}")
    print(f"  Saved test: {os.path.join(FEAT_DIR, 'xgboost_expanded_test.npy')}")
    print(f"  Saved submission: {sub_path}")
    print()

    # Compare with Ridge
    print("=" * 70)
    print("Comparison")
    print("=" * 70)
    ridge_oof = np.load(os.path.join(FEAT_DIR, "ridge_expanded_oof.npy"))
    ridge_rmse = np.sqrt(np.mean((y - ridge_oof) ** 2))
    print(f"  Ridge OOF RMSE: {ridge_rmse:.6f}")
    print(f"  XGBoost OOF RMSE: {rmse:.6f}")
    print(f"  Improvement: {(ridge_rmse - rmse) / ridge_rmse * 100:.2f}%")
    print()

    # Create blend with DeBERTa
    print("=" * 70)
    print("Creating blends with DeBERTa")
    print("=" * 70)

    deberta_ve = np.load(os.path.join(PROJECT_ROOT, "artifacts", "models", "deberta_base_ensemble_ve.npy"))
    test_ids = test_df["id"].values

    for xgb_weight in [0.05, 0.10, 0.15, 0.20]:
        blended = deberta_ve * (1 - xgb_weight) + test_preds * xgb_weight
        blended = np.clip(blended, 1.0, 5.0)

        name = f"deberta_ve{int((1-xgb_weight)*100)}_xgb{int(xgb_weight*100)}"
        submission = pd.DataFrame({"id": test_ids, "rating": blended})
        filepath = os.path.join(OUTPUT_DIR, f"{name}.csv")
        submission.to_csv(filepath, index=False)
        print(f"  {name}: mean={blended.mean():.4f}, std={blended.std():.4f}")

    print()
    print(f"Total time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()

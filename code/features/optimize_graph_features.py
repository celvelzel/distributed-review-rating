"""
Optimize graph features RMSE using CPU-only models.
All runs are 5-fold OOF. No GPU usage.
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

FEAT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "features")


def load_data():
    """Load all available features."""
    train_df = pd.read_parquet(os.path.join(PROJECT_ROOT, "artifacts", "etl", "train.parquet"))
    y = train_df['rating'].values.astype(np.float32)

    # Expanded features
    exp_train = pd.read_parquet(os.path.join(FEAT_DIR, "expanded_graph_train.parquet"))
    expanded_cols = [c for c in exp_train.columns if c not in ['id', 'parent_prod_id']]
    
    # Kfold stats
    us = pd.read_parquet(os.path.join(FEAT_DIR, "user_stats_kfold.parquet"))
    ps = pd.read_parquet(os.path.join(FEAT_DIR, "product_stats_kfold.parquet"))
    cs = pd.read_parquet(os.path.join(FEAT_DIR, "category_stats_kfold.parquet"))

    # Deduplicate stats
    us_dedup = us.groupby('id').first().reset_index()
    ps_dedup = ps.groupby('parent_prod_id').first().reset_index()
    cs_dedup = cs.groupby('main_category').first().reset_index()
    
    us_dict = us_dedup.set_index('id')
    ps_dict = ps_dedup.set_index('parent_prod_id')
    cs_dict = cs_dedup.set_index('main_category')

    # Build feature matrix
    features = {}
    
    # Expanded features
    for col in expanded_cols:
        features[col] = exp_train[col].values.astype(np.float32)
    
    # User stats
    for col in ['avg_rating', 'num_reviews', 'rating_std', 'avg_votes', 'purchased_rate']:
        if col in us_dict.columns:
            features[f'user_{col}'] = train_df['user_id'].map(us_dict[col]).fillna(0).values.astype(np.float32)
    
    # Product stats
    for col in ['prod_avg_rating', 'prod_num_reviews', 'prod_rating_number', 'prod_price']:
        if col in ps_dict.columns:
            features[col] = train_df['parent_prod_id'].map(ps_dict[col]).fillna(0).values.astype(np.float32)
    
    # Category stats
    prod_cat = ps_dedup[['parent_prod_id', 'main_category']].set_index('parent_prod_id')['main_category']
    train_cat = train_df['parent_prod_id'].map(prod_cat).fillna('Unknown')
    for col in ['cat_avg_rating', 'cat_rating_std', 'cat_avg_price']:
        if col in cs_dict.columns:
            features[col] = train_cat.map(cs_dict[col]).fillna(0).values.astype(np.float32)

    X = pd.DataFrame(features)
    return X, y, train_df


def evaluate_model(X, y, model_fn, name, n_splits=5):
    """Evaluate model with 5-fold OOF."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    
    for fold, (tr, va) in enumerate(kf.split(X)):
        model = model_fn()
        model.fit(X.iloc[tr], y[tr])
        oof[va] = model.predict(X.iloc[va])
    
    rmse = np.sqrt(np.mean((y - oof) ** 2))
    return rmse, oof


def main():
    t_start = time.time()
    print("=" * 70)
    print("Graph Features RMSE Optimization (CPU-only)")
    print("=" * 70)
    
    X, y, train_df = load_data()
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {list(X.columns)}")
    print()
    
    results = []
    
    # ============================================================
    # Section 1: Ridge with different feature sets
    # ============================================================
    print("-" * 70)
    print("Section 1: Ridge Regression - Feature Sets")
    print("-" * 70)
    
    feature_sets = {
        'expanded_only': ['store_product_count', 'store_avg_rating_number', 'store_total_rating_number', 
                          'store_has_name', 'user_leniency', 'user_harshness', 'user_num_reviews_oof',
                          'user_cat_avg_rating', 'user_cat_review_count', 'user_cat_deviation'],
        'expanded+user_stats': ['store_product_count', 'store_avg_rating_number', 'store_total_rating_number',
                                'store_has_name', 'user_leniency', 'user_harshness', 'user_num_reviews_oof',
                                'user_cat_avg_rating', 'user_cat_review_count', 'user_cat_deviation',
                                'user_avg_rating', 'user_num_reviews', 'user_rating_std'],
        'expanded+prod_stats': ['store_product_count', 'store_avg_rating_number', 'store_total_rating_number',
                                'store_has_name', 'user_leniency', 'user_harshness', 'user_num_reviews_oof',
                                'user_cat_avg_rating', 'user_cat_review_count', 'user_cat_deviation',
                                'prod_avg_rating', 'prod_num_reviews', 'prod_rating_number'],
        'all_features': list(X.columns),
    }
    
    for name, cols in feature_sets.items():
        available_cols = [c for c in cols if c in X.columns]
        X_sub = X[available_cols]
        rmse, _ = evaluate_model(X_sub, y, lambda: Ridge(alpha=1.0), name)
        results.append(('Ridge', name, len(available_cols), rmse))
        print(f"  {name} ({len(available_cols)}d): RMSE = {rmse:.6f}")
    
    print()
    
    # ============================================================
    # Section 2: Different models on all features
    # ============================================================
    print("-" * 70)
    print("Section 2: Different Models (all features)")
    print("-" * 70)
    
    models = {
        'Ridge(alpha=1)': lambda: Ridge(alpha=1.0),
        'Ridge(alpha=0.1)': lambda: Ridge(alpha=0.1),
        'Lasso(alpha=0.001)': lambda: Lasso(alpha=0.001),
        'ElasticNet(alpha=0.001)': lambda: ElasticNet(alpha=0.001, l1_ratio=0.5),
    }
    
    X_all = X.copy()
    
    for name, model_fn in models.items():
        rmse, _ = evaluate_model(X_all, y, model_fn, name)
        results.append((name, 'all_features', X_all.shape[1], rmse))
        print(f"  {name}: RMSE = {rmse:.6f}")
    
    print()
    
    # ============================================================
    # Section 3: Feature engineering
    # ============================================================
    print("-" * 70)
    print("Section 3: Feature Engineering")
    print("-" * 70)
    
    # Key interactions
    X_eng = X_all.copy()
    
    # User leniency * user review count
    X_eng['leniency_x_reviews'] = X_eng['user_leniency'] * X_eng['user_num_reviews_oof']
    
    # User cat deviation * user cat review count
    X_eng['cat_dev_x_reviews'] = X_eng['user_cat_deviation'] * X_eng['user_cat_review_count']
    
    # User avg rating - product avg rating
    if 'user_avg_rating' in X_eng.columns and 'prod_avg_rating' in X_eng.columns:
        X_eng['user_prod_diff'] = X_eng['user_avg_rating'] - X_eng['prod_avg_rating']
        X_eng['user_prod_ratio'] = X_eng['user_avg_rating'] / (X_eng['prod_avg_rating'] + 1e-6)
    
    # Log transforms
    for col in ['user_num_reviews_oof', 'user_cat_review_count', 'store_product_count']:
        if col in X_eng.columns:
            X_eng[f'{col}_log'] = np.log1p(X_eng[col].clip(lower=0))
    
    rmse, _ = evaluate_model(X_eng, y, lambda: Ridge(alpha=1.0), 'engineered')
    results.append(('Ridge', 'engineered', X_eng.shape[1], rmse))
    print(f"  Engineered features ({X_eng.shape[1]}d): RMSE = {rmse:.6f}")
    
    print()
    
    # ============================================================
    # Section 4: Gradient Boosting (CPU)
    # ============================================================
    print("-" * 70)
    print("Section 4: Gradient Boosting (CPU)")
    print("-" * 70)
    
    # Sample for speed (GBM is slow on 3M rows)
    sample_size = 500000
    idx = np.random.RandomState(42).choice(len(y), sample_size, replace=False)
    X_sample = X_all.iloc[idx]
    y_sample = y[idx]
    
    gbm_params = [
        ('GBM(d=3,lr=0.1)', GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42)),
        ('GBM(d=4,lr=0.05)', GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)),
    ]
    
    for name, model in gbm_params:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(y_sample))
        for tr, va in kf.split(X_sample):
            from sklearn.base import clone
            m = clone(model)
            m.fit(X_sample.iloc[tr], y_sample[tr])
            oof[va] = m.predict(X_sample.iloc[va])
        rmse = np.sqrt(np.mean((y_sample - oof) ** 2))
        results.append((name, 'sample_500k', X_sample.shape[1], rmse))
        print(f"  {name} (500k sample): RMSE = {rmse:.6f}")
    
    print()
    
    # ============================================================
    # Section 5: Try LightGBM if available
    # ============================================================
    print("-" * 70)
    print("Section 5: LightGBM (CPU)")
    print("-" * 70)
    
    try:
        import lightgbm as lgb
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        
        for fold, (tr, va) in enumerate(kf.split(X_all)):
            train_set = lgb.Dataset(X_all.iloc[tr], label=y[tr])
            val_set = lgb.Dataset(X_all.iloc[va], label=y[va], reference=train_set)
            
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'learning_rate': 0.05,
                'num_leaves': 63,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'seed': 42,
                'device': 'cpu',
                'n_jobs': -1,
            }
            
            model = lgb.train(params, train_set, num_boost_round=1000,
                            valid_sets=[val_set], 
                            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(50)])
            oof[va] = model.predict(X_all.iloc[va])
            print(f"  Fold {fold+1}: RMSE = {np.sqrt(np.mean((y[va] - oof[va])**2)):.6f}")
        
        rmse = np.sqrt(np.mean((y - oof) ** 2))
        results.append(('LightGBM', 'all_features', X_all.shape[1], rmse))
        print(f"  LightGBM OOF RMSE = {rmse:.6f}")
        
        # Feature importance
        print("\n  Top 10 features by importance:")
        importance = model.feature_importance(importance_type='gain')
        feat_imp = pd.DataFrame({'feature': X_all.columns, 'importance': importance})
        feat_imp = feat_imp.sort_values('importance', ascending=False)
        for _, row in feat_imp.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.0f}")
        
    except ImportError:
        print("  LightGBM not installed, skipping")
    
    print()
    
    # ============================================================
    # Section 6: Try XGBoost if available
    # ============================================================
    print("-" * 70)
    print("Section 6: XGBoost (CPU)")
    print("-" * 70)
    
    try:
        import xgboost as xgb
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(y))
        
        for fold, (tr, va) in enumerate(kf.split(X_all)):
            dtrain = xgb.DMatrix(X_all.iloc[tr], label=y[tr])
            dval = xgb.DMatrix(X_all.iloc[va], label=y[va])
            
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'learning_rate': 0.05,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'seed': 42,
                'tree_method': 'hist',
                'verbosity': 0,
            }
            
            model = xgb.train(params, dtrain, num_boost_round=1000,
                             evals=[(dval, 'val')], 
                             early_stopping_rounds=50, verbose_eval=False)
            oof[va] = model.predict(dval)
            print(f"  Fold {fold+1}: RMSE = {np.sqrt(np.mean((y[va] - oof[va])**2)):.6f}")
        
        rmse = np.sqrt(np.mean((y - oof) ** 2))
        results.append(('XGBoost', 'all_features', X_all.shape[1], rmse))
        print(f"  XGBoost OOF RMSE = {rmse:.6f}")
        
    except ImportError:
        print("  XGBoost not installed, skipping")
    
    print()
    
    # ============================================================
    # Summary
    # ============================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    results_df = pd.DataFrame(results, columns=['Model', 'Features', 'Dims', 'RMSE'])
    results_df = results_df.sort_values('RMSE')
    
    print("\nAll results (sorted by RMSE):")
    print("-" * 60)
    for _, row in results_df.iterrows():
        print(f"  {row['Model']:25s} | {row['Features']:20s} | {row['Dims']:3d}d | RMSE = {row['RMSE']:.6f}")
    
    best = results_df.iloc[0]
    print(f"\n  BEST: {best['Model']} with {best['Features']} ({best['Dims']}d) -> RMSE = {best['RMSE']:.6f}")
    
    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

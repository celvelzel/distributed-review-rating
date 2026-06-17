"""
Optimized Ridge model for graph features.
Goal: Minimize OOF RMSE while preventing overfitting.
Final output will be blended with DeBERTa VE 90% + Ridge 10%.
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

FEAT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "features")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def load_data():
    """Load all available features."""
    train_df = pd.read_parquet(os.path.join(PROJECT_ROOT, "artifacts", "etl", "train.parquet"))
    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "test.csv"))
    y = train_df['rating'].values.astype(np.float32)

    # Expanded features
    exp_train = pd.read_parquet(os.path.join(FEAT_DIR, "expanded_graph_train.parquet"))
    exp_test = pd.read_parquet(os.path.join(FEAT_DIR, "expanded_graph_test.parquet"))
    
    # Kfold stats
    us = pd.read_parquet(os.path.join(FEAT_DIR, "user_stats_kfold.parquet"))
    ps = pd.read_parquet(os.path.join(FEAT_DIR, "product_stats_kfold.parquet"))

    # Deduplicate stats
    us_dedup = us.groupby('id').first().reset_index()
    ps_dedup = ps.groupby('parent_prod_id').first().reset_index()
    
    us_dict = us_dedup.set_index('id')
    ps_dict = ps_dedup.set_index('parent_prod_id')

    # Build features for train
    expanded_cols = [c for c in exp_train.columns if c not in ['id', 'parent_prod_id']]
    features_train = {}
    features_test = {}
    
    for col in expanded_cols:
        features_train[col] = exp_train[col].values.astype(np.float32)
        features_test[col] = exp_test[col].values.astype(np.float32)
    
    # User stats
    for col in ['avg_rating', 'num_reviews', 'rating_std']:
        if col in us_dict.columns:
            features_train[f'user_{col}'] = train_df['user_id'].map(us_dict[col]).fillna(0).values.astype(np.float32)
            features_test[f'user_{col}'] = test_df['user_id'].map(us_dict[col]).fillna(0).values.astype(np.float32)
    
    # Product stats
    for col in ['prod_avg_rating', 'prod_num_reviews']:
        if col in ps_dict.columns:
            features_train[col] = train_df['parent_prod_id'].map(ps_dict[col]).fillna(0).values.astype(np.float32)
            features_test[col] = test_df['parent_prod_id'].map(ps_dict[col]).fillna(0).values.astype(np.float32)

    X_train = pd.DataFrame(features_train)
    X_test = pd.DataFrame(features_test)
    
    # Add interaction features (careful - use only robust ones)
    for X in [X_train, X_test]:
        X['leniency_x_reviews'] = X['user_leniency'] * X['user_num_reviews_oof']
        X['cat_dev_x_reviews'] = X['user_cat_deviation'] * X['user_cat_review_count']
        X['user_prod_diff'] = X['user_avg_rating'] - X['prod_avg_rating']
    
    return X_train, X_test, y, train_df, test_df


def evaluate_model(X, y, alpha=1.0, n_splits=5):
    """Evaluate model with 5-fold OOF."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    
    for tr, va in kf.split(X):
        model = Ridge(alpha=alpha)
        model.fit(X.iloc[tr], y[tr])
        oof[va] = model.predict(X.iloc[va])
    
    rmse = np.sqrt(np.mean((y - oof) ** 2))
    return rmse, oof


def train_final_model(X_train, y, X_test, alpha=1.0, n_splits=5):
    """Train final model with 5-fold for test predictions."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    test_preds = np.zeros(len(X_test))
    
    for tr, va in kf.split(X_train):
        model = Ridge(alpha=alpha)
        model.fit(X_train.iloc[tr], y[tr])
        oof[va] = model.predict(X_train.iloc[va])
        test_preds += model.predict(X_test) / n_splits
    
    rmse = np.sqrt(np.mean((y - oof) ** 2))
    return rmse, oof, test_preds


def main():
    t_start = time.time()
    print("=" * 70)
    print("Optimized Ridge Model for Graph Features")
    print("=" * 70)
    
    # Load data
    X_train, X_test, y, train_df, test_df = load_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Features: {list(X_train.columns)}")
    print()
    
    # ============================================================
    # Step 1: Find optimal alpha
    # ============================================================
    print("-" * 70)
    print("Step 1: Hyperparameter tuning (Ridge alpha)")
    print("-" * 70)
    
    best_alpha = None
    best_rmse = float('inf')
    
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        rmse, _ = evaluate_model(X_train, y, alpha=alpha)
        print(f"  alpha={alpha:8.2f}: RMSE = {rmse:.6f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
    
    print(f"\n  Best alpha: {best_alpha} (RMSE = {best_rmse:.6f})")
    print()
    
    # ============================================================
    # Step 2: Feature importance analysis
    # ============================================================
    print("-" * 70)
    print("Step 2: Feature importance (correlation with target)")
    print("-" * 70)
    
    corrs = X_train.corrwith(pd.Series(y))
    corrs_abs = corrs.abs().sort_values(ascending=False)
    print("\n  Feature correlations:")
    for feat, corr in corrs_abs.items():
        print(f"    {feat:30s} | corr = {corr:.4f}")
    print()
    
    # ============================================================
    # Step 3: Try feature subsets to prevent overfitting
    # ============================================================
    print("-" * 70)
    print("Step 3: Feature subset analysis (prevent overfitting)")
    print("-" * 70)
    
    # Top features by correlation
    top_features = corrs_abs.head(10).index.tolist()
    rmse_top, _ = evaluate_model(X_train[top_features], y, alpha=best_alpha)
    print(f"  Top 10 features: RMSE = {rmse_top:.6f}")
    
    # All features
    rmse_all, _ = evaluate_model(X_train, y, alpha=best_alpha)
    print(f"  All {X_train.shape[1]} features: RMSE = {rmse_all:.6f}")
    
    # Use all features if better, otherwise use top features
    if rmse_all < rmse_top:
        selected_features = list(X_train.columns)
        print(f"\n  Using all features (better)")
    else:
        selected_features = top_features
        print(f"\n  Using top 10 features (prevent overfitting)")
    print()
    
    # ============================================================
    # Step 4: Train final model
    # ============================================================
    print("-" * 70)
    print("Step 4: Training final model")
    print("-" * 70)
    
    rmse_final, oof, test_preds = train_final_model(
        X_train[selected_features], y, X_test[selected_features], alpha=best_alpha
    )
    print(f"  Final OOF RMSE: {rmse_final:.6f}")
    print(f"  Test predictions: mean={test_preds.mean():.4f}, std={test_preds.std():.4f}")
    print()
    
    # ============================================================
    # Step 5: Save predictions
    # ============================================================
    print("-" * 70)
    print("Step 5: Saving predictions")
    print("-" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save OOF
    oof_path = os.path.join(FEAT_DIR, "ridge_expanded_oof.npy")
    np.save(oof_path, oof)
    print(f"  Saved OOF: {oof_path}")
    
    # Save test predictions
    test_path = os.path.join(FEAT_DIR, "ridge_expanded_test.npy")
    np.save(test_path, test_preds)
    print(f"  Saved test: {test_path}")
    
    # Save submission
    submission = pd.DataFrame({
        "id": test_df["id"].values,
        "rating": np.clip(test_preds, 1.0, 5.0)
    })
    sub_path = os.path.join(OUTPUT_DIR, "ridge_expanded_features.csv")
    submission.to_csv(sub_path, index=False)
    print(f"  Saved submission: {sub_path}")
    print()
    
    # ============================================================
    # Summary
    # ============================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Features used: {len(selected_features)}")
    print(f"  Best alpha: {best_alpha}")
    print(f"  OOF RMSE: {rmse_final:.6f}")
    print(f"  vs former Ridge (0.8076): {(0.8076 - rmse_final) / 0.8076 * 100:.1f}% improvement")
    print()
    print(f"  Total time: {time.time()-t_start:.1f}s")
    print()
    print("  Next steps:")
    print("  1. Wait for DeBERTa training to complete")
    print("  2. Blend DeBERTa VE 90% + Ridge expanded 10%")
    print("  3. Submit to Kaggle")


if __name__ == "__main__":
    main()

"""
Retrain CatBoost with K-Fold features (no target leakage).
"""

import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def train_catboost_kfold():
    t_start = time.time()
    
    # Load features
    log.info("Loading K-Fold features...")
    X = pq.read_table("artifacts/features/X_train_kfold.parquet").to_pandas()
    y = np.load("artifacts/features/y_train.npy")
    X_test = pq.read_table("artifacts/features/X_test_kfold.parquet").to_pandas()
    
    log.info(f"  X_train: {X.shape}, y_train: {y.shape}, X_test: {X_test.shape}")
    
    # Identify categorical columns (main_category if present)
    cat_cols = [c for c in X.columns if c in ["main_category"]]
    
    # K-Fold CV
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_rmses = []
    
    params = {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "RMSE",
        "verbose": 100,
        "random_seed": 42,
        "early_stopping_rounds": 50,
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        log.info(f"\n{'='*60}")
        log.info(f"Fold {fold + 1}/{n_folds}")
        log.info(f"{'='*60}")
        
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y[val_idx]
        
        # Create CatBoost pools
        train_pool = Pool(X_train_fold, y_train_fold, cat_features=cat_cols)
        val_pool = Pool(X_val_fold, y_val_fold, cat_features=cat_cols)
        
        # Train
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        
        # OOF predictions
        oof_preds[val_idx] = model.predict(X_val_fold)
        
        # Test predictions (average across folds)
        test_preds += model.predict(X_test) / n_folds
        
        # Fold RMSE
        fold_rmse = np.sqrt(np.mean((oof_preds[val_idx] - y_val_fold) ** 2))
        fold_rmses.append(fold_rmse)
        log.info(f"  Fold {fold + 1} RMSE: {fold_rmse:.5f}")
        
        # Save model
        os.makedirs("artifacts/models", exist_ok=True)
        model.save_model(f"artifacts/models/catboost_kfold_fold{fold}.cbm")
    
    # Overall OOF RMSE
    oof_rmse = np.sqrt(np.mean((oof_preds - y) ** 2))
    log.info(f"\n{'='*60}")
    log.info(f"Overall OOF RMSE: {oof_rmse:.5f}")
    log.info(f"Fold RMSEs: {[f'{r:.5f}' for r in fold_rmses]}")
    log.info(f"Mean Fold RMSE: {np.mean(fold_rmses):.5f}")
    
    # Save predictions
    np.save("artifacts/models/catboost_kfold_oof.npy", oof_preds)
    np.save("artifacts/models/catboost_kfold_test.npy", test_preds)
    
    # Update metrics.json
    metrics_path = "docs/changelog/metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    metrics["catboost_kfold"] = {
        "oof_rmse": round(oof_rmse, 5),
        "mean_fold_rmse": round(np.mean(fold_rmses), 5),
        "fold_rmses": [round(r, 5) for r in fold_rmses],
        "train_time_sec": round(time.time() - t_start, 2),
        "model": "catboost_kfold",
        "features": "kfold_stats + temporal + text_length + te + bert + lightgcn",
        "note": "Fixed target leakage - user/product/category stats now use K-Fold"
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    log.info(f"\n✅ Training complete in {time.time() - t_start:.1f}s")
    log.info(f"   OOF RMSE: {oof_rmse:.5f}")
    log.info(f"   Saved: catboost_kfold_oof.npy, catboost_kfold_test.npy")


if __name__ == "__main__":
    train_catboost_kfold()

"""
Audit dependencies for various scripts.
Checks if required input files exist for:
- train_graph_models.py
- expand_graph_features.py
- stacking_v3.py
"""
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Check all dependencies for train_graph_models.py
deps = {
    "train.parquet": os.path.join(PROJECT_ROOT, "artifacts", "etl", "train.parquet"),
    "test.csv": os.path.join(PROJECT_ROOT, "data", "test.csv"),
    "expanded_graph_train.parquet": os.path.join(PROJECT_ROOT, "artifacts", "features", "expanded_graph_train.parquet"),
    "expanded_graph_test.parquet": os.path.join(PROJECT_ROOT, "artifacts", "features", "expanded_graph_test.parquet"),
    "user_stats_kfold.parquet": os.path.join(PROJECT_ROOT, "artifacts", "features", "user_stats_kfold.parquet"),
    "product_stats_kfold.parquet": os.path.join(PROJECT_ROOT, "artifacts", "features", "product_stats_kfold.parquet"),
}

# Also check alternative paths
alt_deps = {
    "user_stats_pandas.parquet": os.path.join(PROJECT_ROOT, "artifacts", "features", "user_stats_pandas.parquet"),
    "product_stats_pandas.parquet": os.path.join(PROJECT_ROOT, "artifacts", "features", "product_stats_pandas.parquet"),
}

print("=== train_graph_models.py Dependencies ===")
for name, path in deps.items():
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    print(f"  {'OK' if exists else 'MISSING':7s} {name:40s} {path}")
    if exists:
        print(f"          Size: {size/1024/1024:.2f} MB")

print("\n=== Alternative paths (not used by train_graph_models.py) ===")
for name, path in alt_deps.items():
    exists = os.path.exists(path)
    print(f"  {'OK' if exists else 'MISSING':7s} {name:40s}")

# Check expand_graph_features.py dependencies
print("\n=== expand_graph_features.py Dependencies ===")
exp_deps = {
    "train.parquet (ETL)": os.path.join(PROJECT_ROOT, "artifacts", "etl", "train.parquet"),
    "prodinfo.parquet (ETL)": os.path.join(PROJECT_ROOT, "artifacts", "etl", "prodinfo.parquet"),
    "test.csv": os.path.join(PROJECT_ROOT, "data", "test.csv"),
    "prodInfo.csv": os.path.join(PROJECT_ROOT, "data", "prodInfo.csv"),
}
for name, path in exp_deps.items():
    exists = os.path.exists(path)
    print(f"  {'OK' if exists else 'MISSING':7s} {name:40s} {path}")

# Check stacking_v3.py dependencies
print("\n=== stacking_v3.py Dependencies ===")
model_dir = os.path.join(PROJECT_ROOT, "artifacts", "models")
v3_deps = [
    "lgb_tfidf_oof.npy", "lgb_tfidf_test.npy",
    "xgboost_oof.npy", "xgboost_test.npy",
    "mlp_oof.npy", "mlp_test.npy",
    "lgb_safe_dense_oof.npy", "lgb_safe_dense_test.npy",
    "xgboost_safe_oof.npy", "xgboost_safe_test.npy",
    "catboost_safe_oof.npy", "catboost_safe_test.npy",
    "ensemble_diverse_oof.npy", "ensemble_diverse_test.npy",
    "xgb_graph_safe_oof.npy", "xgb_graph_safe_test.npy",
    "lgb_graph_safe_oof.npy", "lgb_graph_safe_test.npy",
    "stacking_v2_test.npy",
    "deberta_lora_fold1_test.npy",
]
for dep in v3_deps:
    path = os.path.join(model_dir, dep)
    exists = os.path.exists(path)
    print(f"  {'OK' if exists else 'MISSING':7s} {dep}")

# Check test.parquet (needed by stacking_v3.py for test_ids)
print("\n=== test_ids source ===")
test_parquet = os.path.join(PROJECT_ROOT, "artifacts", "etl", "test.parquet")
print(f"  {'OK' if os.path.exists(test_parquet) else 'MISSING':7s} test.parquet (ETL)")

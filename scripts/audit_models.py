"""
Audit model artifacts.
Loads and prints shape/mean/std/RMSE for OOF predictions of base models.
Also checks for test-only artifacts.
"""
import numpy as np
y = np.load("artifacts/features/y_train.npy").astype(np.float32)
print(f"y_train: shape={y.shape}, mean={y.mean():.4f}, std={y.std():.4f}")
models = ["lgb_tfidf", "xgboost", "mlp", "lgb_safe_dense", "xgboost_safe", "catboost_safe", "ensemble_diverse"]
for m in models:
    try:
        arr = np.load(f"artifacts/models/{m}_oof.npy").astype(np.float32)
        rmse = np.sqrt(np.mean((arr - y) ** 2))
        print(f"{m:20s}: shape={arr.shape}, mean={arr.mean():.4f}, std={arr.std():.4f}, OOF_RMSE={rmse:.5f}")
    except Exception as e:
        print(f"{m:20s}: ERROR - {e}")
# Also check key test-only artifacts
test_only = ["stacking_v2_test", "deberta_lora_fold1_test", "deberta_base_full_test"]
for m in test_only:
    try:
        arr = np.load(f"artifacts/models/{m}.npy").astype(np.float32)
        print(f"{m:30s}: shape={arr.shape}, mean={arr.mean():.4f}, std={arr.std():.4f}")
    except Exception as e:
        print(f"{m:30s}: ERROR - {e}")

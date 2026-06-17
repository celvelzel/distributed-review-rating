"""
Blend DeBERTa predictions with expanded graph feature Ridge predictions.

Since DeBERTa full training is still running, use existing fold1+fold2 ensemble.
"""

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_DIR = os.path.join(PROJECT_ROOT, "artifacts", "models")


def main():
    # Load expanded Ridge predictions
    ridge_path = os.path.join(OUTPUT_DIR, "ridge_expanded_features_only.csv")
    if not os.path.exists(ridge_path):
        print("ERROR: ridge_expanded_features_only.csv not found. Run test_expanded_features.py first.")
        return
    
    ridge_df = pd.read_csv(ridge_path)
    ridge_preds = ridge_df["rating"].values
    print(f"Ridge expanded: mean={ridge_preds.mean():.4f}, std={ridge_preds.std():.4f}")

    # Load DeBERTa predictions (use existing ensemble)
    deberta_path = os.path.join(MODEL_DIR, "deberta_base_ensemble_test.npy")
    if not os.path.exists(deberta_path):
        print("ERROR: deberta_base_ensemble_test.npy not found.")
        return
    
    deberta_preds = np.load(deberta_path)
    print(f"DeBERTa ensemble: mean={deberta_preds.mean():.4f}, std={deberta_preds.std():.4f}")

    # Apply variance expansion to DeBERTa
    target_std = 1.422  # Training set std
    deberta_ve = (deberta_preds - deberta_preds.mean()) * (target_std / deberta_preds.std()) + 3.941
    deberta_ve = np.clip(deberta_ve, 1.0, 5.0)
    print(f"DeBERTa VE: mean={deberta_ve.mean():.4f}, std={deberta_ve.std():.4f}")

    # Generate blend variants
    print("\nGenerating blend submissions...")
    blends = [
        ("deberta_ve90_ridge_expanded10", 0.90, 0.10),
        ("deberta_ve85_ridge_expanded15", 0.85, 0.15),
        ("deberta_ve80_ridge_expanded20", 0.80, 0.20),
        ("deberta_ve95_ridge_expanded5", 0.95, 0.05),
    ]

    for name, w_deberta, w_ridge in blends:
        blended = deberta_ve * w_deberta + ridge_preds * w_ridge
        blended = np.clip(blended, 1.0, 5.0)
        
        submission = pd.DataFrame({
            "id": ridge_df["id"].values,
            "rating": blended
        })
        filepath = os.path.join(OUTPUT_DIR, f"{name}.csv")
        submission.to_csv(filepath, index=False)
        print(f"  {name}: mean={blended.mean():.4f}, std={blended.std():.4f}")

    print("\nDone! Submit the most promising variant to Kaggle.")


if __name__ == "__main__":
    main()

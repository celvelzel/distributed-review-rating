"""
Blend best existing submission with graph-based Ridge predictions.

Since DeBERTa .npy predictions are on HPC, we use the best submission CSV
as the base and blend with graph feature Ridge predictions.
"""

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def main():
    # Load best existing submission (use submission-final as proxy for DeBERTa)
    best_path = os.path.join(OUTPUT_DIR, "submission-final.csv")
    best_df = pd.read_csv(best_path)
    best_preds = best_df["rating"].values

    # Load Ridge graph features predictions
    ridge_path = os.path.join(OUTPUT_DIR, "ridge_graph_features_only.csv")
    ridge_df = pd.read_csv(ridge_path)
    ridge_preds = ridge_df["rating"].values

    print(f"Best submission: mean={best_preds.mean():.4f}, std={best_preds.std():.4f}")
    print(f"Ridge graph: mean={ridge_preds.mean():.4f}, std={ridge_preds.std():.4f}")

    # Generate blend variants
    blends = [
        ("graph_blend_95_5", 0.95, 0.05),
        ("graph_blend_90_10", 0.90, 0.10),
        ("graph_blend_85_15", 0.85, 0.15),
        ("graph_blend_80_20", 0.80, 0.20),
        ("graph_blend_70_30", 0.70, 0.30),
    ]

    for name, w_best, w_ridge in blends:
        blended = best_preds * w_best + ridge_preds * w_ridge
        blended = np.clip(blended, 1.0, 5.0)

        submission = pd.DataFrame({
            "id": best_df["id"].values,
            "rating": blended
        })
        filepath = os.path.join(OUTPUT_DIR, f"{name}.csv")
        submission.to_csv(filepath, index=False)
        print(f"  {name}: mean={blended.mean():.4f}, std={blended.std():.4f}")

    # Also try graph-calibrated approach
    # Use Ridge predictions as anchor points
    calibrated = best_preds * 0.85 + ridge_preds * 0.15
    # Apply variance expansion to calibrated predictions
    target_std = 1.422
    pred_std = calibrated.std()
    scale = target_std / pred_std
    calibrated_ve = (calibrated - calibrated.mean()) * scale + calibrated.mean()
    calibrated_ve = np.clip(calibrated_ve, 1.0, 5.0)

    submission = pd.DataFrame({
        "id": best_df["id"].values,
        "rating": calibrated_ve
    })
    filepath = os.path.join(OUTPUT_DIR, "graph_calibrated_ve.csv")
    submission.to_csv(filepath, index=False)
    print(f"  graph_calibrated_ve: mean={calibrated_ve.mean():.4f}, std={calibrated_ve.std():.4f}")

    print("\nDone! Submit the most promising variant to Kaggle.")


if __name__ == "__main__":
    main()

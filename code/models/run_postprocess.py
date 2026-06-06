"""
Post-processing runner: optimal rounding + clipping on stacking OOF predictions.
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from code.models.postprocess import optimal_round, clip_15


def main():
    t0 = time.time()

    # Paths
    stacking_oof_path = "artifacts/models/stacking_oof.npy"
    y_train_path = "artifacts/features/y_train.npy"
    output_path = "artifacts/models/final_predictions.npy"

    # Load data
    print("Loading stacking OOF predictions...")
    stacking_oof = np.load(stacking_oof_path)
    print(f"  stacking_oof shape: {stacking_oof.shape}")

    print("Loading y_train...")
    y_train = np.load(y_train_path)
    print(f"  y_train shape: {y_train.shape}")

    # Report raw RMSE
    rmse_raw = np.sqrt(np.mean((stacking_oof - y_train) ** 2))
    print(f"\nRaw stacking OOF RMSE: {rmse_raw:.6f}")

    # Optimal rounding
    print("\nRunning optimal rounding...")
    rounded_preds, best_name, rmse_rounded = optimal_round(stacking_oof, y_train)
    print(f"  Best rounding strategy: {best_name}")
    print(f"  RMSE after rounding: {rmse_rounded:.6f}")
    print(f"  RMSE improvement: {rmse_raw - rmse_rounded:+.6f}")

    # Clip to [1, 5]
    print("\nClipping to [1, 5]...")
    final_preds = clip_15(rounded_preds)
    rmse_final = np.sqrt(np.mean((final_preds - y_train) ** 2))
    print(f"  RMSE after clip: {rmse_final:.6f}")

    # Sanity checks
    print(f"\nFinal predictions range: [{final_preds.min()}, {final_preds.max()}]")
    print(f"Final predictions shape: {final_preds.shape}")
    unique_vals = np.unique(final_preds)
    print(f"Unique values ({len(unique_vals)}): {unique_vals[:20]}")

    # Save
    np.save(output_path, final_preds)
    print(f"\nSaved: {output_path}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"\n=== SUMMARY ===")
    print(f"Rounding strategy: {best_name}")
    print(f"RMSE before: {rmse_raw:.6f}")
    print(f"RMSE after:  {rmse_final:.6f}")


if __name__ == "__main__":
    main()

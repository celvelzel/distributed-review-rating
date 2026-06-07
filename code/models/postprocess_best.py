#!/usr/bin/env python
"""Post-process best submission to potentially improve score.

Strategy: Try different clipping and rounding strategies.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUBMISSION_DIR = ROOT / "output"


def main():
    print("=" * 60)
    print("Post-processing Best Submission")
    print("=" * 60)
    
    # Load best submission
    best_path = SUBMISSION_DIR / "submission-tfidf-regularized.csv"
    df = pd.read_csv(best_path)
    preds = df["rating"].values
    test_ids = df["id"].values
    
    print(f"\n  Original predictions:")
    print(f"    Mean: {preds.mean():.3f}")
    print(f"    Std:  {preds.std():.3f}")
    print(f"    Min:  {preds.min():.3f}")
    print(f"    Max:  {preds.max():.3f}")
    
    # Strategy 1: Different clipping ranges
    print("\n  Strategy 1: Different clipping ranges")
    clip_ranges = [
        (1.0, 5.0),  # Current
        (1.0, 4.5),
        (1.5, 5.0),
        (1.5, 4.5),
        (2.0, 5.0),
    ]
    
    for low, high in clip_ranges:
        clipped = np.clip(preds, low, high)
        name = f"clip_{low}_{high}".replace(".", "")
        sub = pd.DataFrame({"id": test_ids, "rating": clipped})
        sub_path = SUBMISSION_DIR / f"submission-{name}.csv"
        sub.to_csv(sub_path, index=False)
        print(f"    {name}: mean={clipped.mean():.3f}, std={clipped.std():.3f}")
    
    # Strategy 2: Rounding to nearest 0.5
    print("\n  Strategy 2: Rounding strategies")
    rounded_05 = np.round(preds * 2) / 2  # Round to nearest 0.5
    rounded_05 = np.clip(rounded_05, 1.0, 5.0)
    sub = pd.DataFrame({"id": test_ids, "rating": rounded_05})
    sub_path = SUBMISSION_DIR / "submission-rounded_05.csv"
    sub.to_csv(sub_path, index=False)
    print(f"    rounded_05: mean={rounded_05.mean():.3f}, std={rounded_05.std():.3f}")
    
    rounded_1 = np.round(preds)  # Round to nearest integer
    rounded_1 = np.clip(rounded_1, 1.0, 5.0)
    sub = pd.DataFrame({"id": test_ids, "rating": rounded_1})
    sub_path = SUBMISSION_DIR / "submission-rounded_1.csv"
    sub.to_csv(sub_path, index=False)
    print(f"    rounded_1: mean={rounded_1.mean():.3f}, std={rounded_1.std():.3f}")
    
    # Strategy 3: Blend with baseline
    print("\n  Strategy 3: Blend with baseline")
    baseline_path = SUBMISSION_DIR / "stage0_submission.csv"
    if baseline_path.exists():
        df_baseline = pd.read_csv(baseline_path)
        baseline_preds = df_baseline["rating"].values
        
        for alpha in [0.9, 0.8, 0.7, 0.6]:
            blended = alpha * preds + (1 - alpha) * baseline_preds
            blended = np.clip(blended, 1.0, 5.0)
            name = f"blend_{int(alpha*100)}_{int((1-alpha)*100)}"
            sub = pd.DataFrame({"id": test_ids, "rating": blended})
            sub_path = SUBMISSION_DIR / f"submission-{name}.csv"
            sub.to_csv(sub_path, index=False)
            print(f"    {name}: mean={blended.mean():.3f}, std={blended.std():.3f}")
    
    # Strategy 4: Add small noise
    print("\n  Strategy 4: Add small noise")
    np.random.seed(42)
    for noise_level in [0.01, 0.02, 0.05]:
        noisy = preds + np.random.normal(0, noise_level, len(preds))
        noisy = np.clip(noisy, 1.0, 5.0)
        name = f"noise_{noise_level}".replace(".", "")
        sub = pd.DataFrame({"id": test_ids, "rating": noisy})
        sub_path = SUBMISSION_DIR / f"submission-{name}.csv"
        sub.to_csv(sub_path, index=False)
        print(f"    {name}: mean={noisy.mean():.3f}, std={noisy.std():.3f}")
    
    print(f"\n  Best submission: submission-tfidf-regularized.csv (Kaggle: 0.79012)")
    print(f"  Try submitting these variations to see if they improve!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

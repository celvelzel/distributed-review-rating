#!/usr/bin/env python
"""Create ensemble of existing submissions.

Strategy: Combine multiple submissions to potentially improve score.
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
    print("Creating Ensemble of Existing Submissions")
    print("=" * 60)
    
    # Load existing submissions
    submissions = {}
    
    # Best model (0.79012)
    best_path = SUBMISSION_DIR / "submission-tfidf-regularized.csv"
    if best_path.exists():
        df = pd.read_csv(best_path)
        submissions["best"] = df["rating"].values
        print(f"  Loaded best: {best_path.name}")
    
    # Baseline (0.80107)
    baseline_path = SUBMISSION_DIR / "stage0_submission.csv"
    if baseline_path.exists():
        df = pd.read_csv(baseline_path)
        submissions["baseline"] = df["rating"].values
        print(f"  Loaded baseline: {baseline_path.name}")
    
    # Other submissions
    for name in ["submission-blend_80_20.csv", "submission-ensemble-weighted.csv", 
                  "submission-ensemble.csv", "submission-tfidf-v2.csv"]:
        path = SUBMISSION_DIR / name
        if path.exists():
            df = pd.read_csv(path)
            key = name.replace("submission-", "").replace(".csv", "")
            submissions[key] = df["rating"].values
            print(f"  Loaded: {name}")
    
    if len(submissions) < 2:
        print("  Error: Need at least 2 submissions for ensemble")
        return
    
    # Create different ensemble strategies
    print("\n  Creating ensemble strategies …")
    
    # Strategy 1: Average of all
    all_preds = np.array(list(submissions.values()))
    ensemble_avg = np.mean(all_preds, axis=0)
    ensemble_avg = np.clip(ensemble_avg, 1.0, 5.0)
    
    # Strategy 2: Weighted average (best gets more weight)
    weights = []
    for key in submissions.keys():
        if key == "best":
            weights.append(0.5)
        elif key == "baseline":
            weights.append(0.2)
        else:
            weights.append(0.3 / (len(submissions) - 2))
    
    weights = np.array(weights) / sum(weights)  # Normalize
    ensemble_weighted = np.average(all_preds, axis=0, weights=weights)
    ensemble_weighted = np.clip(ensemble_weighted, 1.0, 5.0)
    
    # Strategy 3: Median ensemble
    ensemble_median = np.median(all_preds, axis=0)
    ensemble_median = np.clip(ensemble_median, 1.0, 5.0)
    
    # Strategy 4: Best + baseline blend (80/20)
    if "best" in submissions and "baseline" in submissions:
        blend_80_20 = 0.8 * submissions["best"] + 0.2 * submissions["baseline"]
        blend_80_20 = np.clip(blend_80_20, 1.0, 5.0)
    else:
        blend_80_20 = ensemble_avg
    
    # Load IDs from best submission
    df_ids = pd.read_csv(best_path)
    test_ids = df_ids["id"].values
    
    # Save ensemble submissions
    print("\n  Saving ensemble submissions …")
    
    ensembles = {
        "ensemble_avg": ensemble_avg,
        "ensemble_weighted": ensemble_weighted,
        "ensemble_median": ensemble_median,
        "blend_80_20": blend_80_20,
    }
    
    for name, preds in ensembles.items():
        sub = pd.DataFrame({"id": test_ids, "rating": preds})
        sub_path = SUBMISSION_DIR / f"submission-{name}.csv"
        sub.to_csv(sub_path, index=False)
        print(f"    {name}: {sub_path.name}")
    
    # Print statistics
    print("\n  Prediction statistics:")
    for name, preds in ensembles.items():
        print(f"    {name:20s}: mean={preds.mean():.3f}, std={preds.std():.3f}, min={preds.min():.3f}, max={preds.max():.3f}")
    
    print(f"\n  Best submission: submission-tfidf-regularized.csv (Kaggle: 0.79012)")
    print(f"  Try submitting the ensemble versions to see if they improve!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

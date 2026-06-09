"""Generate final Kaggle submission from diverse ensemble test predictions.

Loads the diverse ensemble test predictions (LGB=0.09, XGB=0.05, MLP=0.86),
clips to [1, 5], joins with test IDs, and writes output/submission-final.csv.

Ensemble OOF RMSE: 1.12938
Optimal weights found via scipy.optimize.minimize on OOF predictions.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# Ensure project root is on sys.path so `code.*` imports work
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from code.utils.timer import timed

ROOT = Path(__file__).resolve().parents[2]
ENSEMBLE_NPY = ROOT / "artifacts" / "models" / "ensemble_diverse_test.npy"
TEST_CSV = ROOT / "data" / "test.csv"
OUTPUT_CSV = ROOT / "output" / "submission-final.csv"


def _load_test_ids(csv_path: Path) -> list[int]:
    """Load test IDs from CSV (first column, skip header)."""
    ids: list[int] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        id_idx = header.index("id")
        for row in reader:
            ids.append(int(row[id_idx]))
    return ids


@timed("final_submission", "generate_submission")
def generate_submission() -> None:
    """Load ensemble predictions, clip, join IDs, write CSV."""
    # --- Load diverse ensemble predictions ---
    preds = np.load(ENSEMBLE_NPY)
    print(f"Loaded ensemble predictions: shape={preds.shape}, "
          f"min={preds.min():.4f}, max={preds.max():.4f}, mean={preds.mean():.4f}")

    # --- Clip to [1, 5] ---
    preds = np.clip(preds, 1.0, 5.0)
    print(f"After clip: min={preds.min():.4f}, max={preds.max():.4f}")

    # --- Load test IDs ---
    ids = _load_test_ids(TEST_CSV)
    print(f"Loaded test IDs: {len(ids)} rows")

    assert len(ids) == len(preds), (
        f"ID/prediction length mismatch: {len(ids)} vs {len(preds)}"
    )

    # --- Write submission CSV ---
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "rating"])
        for id_, rating in zip(ids, preds):
            writer.writerow([id_, f"{rating:.6f}"])

    line_count = sum(1 for _ in open(OUTPUT_CSV, encoding="utf-8"))
    print(f"Saved submission: {OUTPUT_CSV}  ({line_count} lines)")


if __name__ == "__main__":
    generate_submission()

"""Generate final submission from stacking ensemble test predictions.

Loads the stacking meta-learner test predictions, clips to [1, 5],
joins with test IDs, and writes output/submission-final.csv.
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
import pyarrow.parquet as pq

from code.utils.timer import timed

ROOT = Path(__file__).resolve().parents[2]
STACKING_NPY = ROOT / "artifacts" / "models" / "stacking_test.npy"
TEST_PARQUET = ROOT / "artifacts" / "etl" / "test.parquet"
OUTPUT_CSV = ROOT / "output" / "submission-final.csv"


@timed("predict", "generate_submission")
def generate_submission() -> None:
    """Load predictions, clip, join IDs, write CSV."""
    # --- 加载 stacking 元学习器预测 ---
    preds = np.load(STACKING_NPY)
    print(f"Loaded stacking predictions: shape={preds.shape}, "
          f"min={preds.min():.4f}, max={preds.max():.4f}")

    # --- 裁剪到 [1, 5] 评分范围（安全措施，上游已裁剪）---
    preds = np.clip(preds, 1.0, 5.0)
    print(f"After clip: min={preds.min():.4f}, max={preds.max():.4f}")

    # --- Load test IDs ---
    test_table = pq.read_table(TEST_PARQUET, columns=["id"])
    ids = test_table["id"].to_pylist()
    print(f"Loaded test IDs: {len(ids)} rows")

    # 断言 ID 数量与预测数量一致，防止错位
    assert len(ids) == len(preds), (
        f"ID/prediction length mismatch: {len(ids)} vs {len(preds)}"
    )

    # --- 写出提交 CSV: id,rating 格式，评分保留 6 位小数 ---
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

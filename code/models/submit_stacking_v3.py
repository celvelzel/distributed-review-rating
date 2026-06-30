#!/usr/bin/env python
"""
Generate Kaggle submissions blending DeBERTa 1M fold1 VE with stacking v3.

Uses the proven 1M DeBERTa fold1 predictions (deberta_lora_fold1_test.npy)
that achieved the best Kaggle score of 0.61734 with stacking v2.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "artifacts" / "models"
FEAT_DIR = ROOT / "artifacts" / "features"
ETL_DIR = ROOT / "artifacts" / "etl"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 60)
    print("Submit: DeBERTa 1M fold1 VE + Stacking v3 blends")
    print("=" * 60)

    # ── Load data ──
    y_train = np.load(str(FEAT_DIR / "y_train.npy")).astype(np.float32)
    test_ids = pd.read_parquet(ETL_DIR / "test.parquet", columns=["id"])["id"].values

    # DeBERTa 1M fold1 test predictions (the one that gave 0.61734)
    deb_test_path = MODEL_DIR / "deberta_lora_fold1_test.npy"
    if not deb_test_path.exists():
        print(f"ERROR: {deb_test_path} not found. Run deberta_lora_1m.py first.")
        return
    deb_test = np.load(str(deb_test_path)).astype(np.float32)
    print(f"  DeBERTa 1M fold1: mean={deb_test.mean():.4f}, std={deb_test.std():.4f}")

    # Stacking v3 test predictions
    stack_v3_path = MODEL_DIR / "stacking_v3_test.npy"
    if not stack_v3_path.exists():
        print(f"ERROR: {stack_v3_path} not found. Run stacking_v3.py first.")
        return
    stack_v3_test = np.load(str(stack_v3_path)).astype(np.float32)
    print(f"  Stacking v3:      mean={stack_v3_test.mean():.4f}, std={stack_v3_test.std():.4f}")

    # Stacking v2 (baseline for comparison)
    stack_v2_path = MODEL_DIR / "stacking_v2_test.npy"
    stack_v2_test = None
    if stack_v2_path.exists():
        stack_v2_test = np.load(str(stack_v2_path)).astype(np.float32)
        print(f"  Stacking v2:      mean={stack_v2_test.mean():.4f}, std={stack_v2_test.std():.4f}")

    # ── 对 DeBERTa 预测做方差扩展 (Variance Expansion) ──
    # VE 公式: ve = (pred - pred_mean) * (target_std / pred_std) + target_mean
    # 1. 中心化: 减去预测均值，消除模型偏差
    # 2. 缩放: 乘以 target_std/pred_std，拉伸到与训练目标相同的方差
    # 3. 平移: 加上 target_mean，恢复正确的中心位置
    # 4. 裁剪到 [1, 5] 合法评分范围
    # DeBERTa 预测方差偏小（欠分散），VE 拉伸后能改善 RMSE
    target_std = y_train.std()
    target_mean = y_train.mean()
    pred_std = deb_test.std()
    scale = target_std / pred_std
    deb_ve = np.clip((deb_test - deb_test.mean()) * scale + target_mean, 1.0, 5.0)
    print(f"\n  VE applied: scale={scale:.4f}  (pred_std {pred_std:.4f} → {deb_ve.std():.4f})")
    print(f"  DeBERTa VE: mean={deb_ve.mean():.4f}, std={deb_ve.std():.4f}")

    # ── Generate submissions ──
    print(f"\n{'=' * 60}")
    print("Generating submissions...\n")

    submissions = {}

    # 1. Stacking v3 standalone (diagnostic — shows how v3 performs alone)
    submissions["stacking-v3-standalone"] = np.clip(stack_v3_test, 1.0, 5.0)

    # 2. DeBERTa VE + Stacking v3 混合 — 主要提交候选
    # 混合公式: blend = w_deb% * deb_ve + (100-w_deb)% * stack_v3
    # 90/10 配比复现了 0.61734 的最佳 Kaggle 成绩（DeBERTa 90% + Stacking v2 10%）
    for w_deb in [95, 90, 85, 80, 75]:
        w_stack = 100 - w_deb
        name = f"deb1m-ve{w_deb}-sv3-{w_stack}"
        blend = np.clip(w_deb / 100 * deb_ve + w_stack / 100 * stack_v3_test, 1.0, 5.0)
        submissions[name] = blend
        print(f"  {name}: mean={blend.mean():.4f}, std={blend.std():.4f}")

    # 3. Baseline: DeBERTa VE + Stacking v2 (the 0.61734 recipe) — included
    # as a sanity check that should reproduce the known-good Kaggle score.
    if stack_v2_test is not None:
        for w_deb in [90, 85]:
            w_stack = 100 - w_deb
            name = f"deb1m-ve{w_deb}-sv2-{w_stack}"
            blend = np.clip(w_deb / 100 * deb_ve + w_stack / 100 * stack_v2_test, 1.0, 5.0)
            submissions[name] = blend
            print(f"  {name}: mean={blend.mean():.4f}, std={blend.std():.4f}  (baseline)")

    # 4. DeBERTa VE only (no blend) — pure DeBERTa baseline after variance expansion
    submissions["deb1m-ve-only"] = deb_ve
    print(f"  deb1m-ve-only: mean={deb_ve.mean():.4f}, std={deb_ve.std():.4f}")

    # ── 保存所有提交 CSV (id, rating 两列) ──
    print(f"\n{'=' * 60}")
    print("Saving CSVs...")
    for name, preds in submissions.items():
        sub = pd.DataFrame({"id": test_ids, "rating": preds})
        path = OUTPUT_DIR / f"submission-{name}.csv"
        sub.to_csv(path, index=False)
        print(f"  → {path.name}")

    print(f"\n  Total: {len(submissions)} submissions saved")

    # ── Recommended submission order (for Kaggle) ──
    print(f"\n{'=' * 60}")
    print("Recommended Kaggle submission order:")
    print("  1. submission-stacking-v3-standalone.csv    (diagnostic: v3 alone)")
    print("  2. submission-deb1m-ve90-sv3-10.csv         (primary: mirrors 0.617 recipe)")
    print("  3. submission-deb1m-ve85-sv3-15.csv         (if v3 is significantly better than v2)")
    print("  4. submission-deb1m-ve90-sv2-10.csv         (baseline: should reproduce ~0.617)")
    print("=" * 60)


if __name__ == "__main__":
    main()

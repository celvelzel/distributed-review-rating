#!/usr/bin/env python
"""
Verify stacking v3 improvement over v2.

Run this AFTER stacking_v3.py has completed. It will:
1. Load stacking v3 results and compare with stacking v2
2. Compute detailed OOF metrics (RMSE, mean, std, distribution)
3. Simulate DeBERTa 1M VE blends at multiple ratios
4. Generate a verification report with pass/fail criteria
5. Save everything to docs/changelog/stacking-v3-verification.md

Usage:
    python code/models/verify_stacking_v3.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "artifacts" / "models"
FEAT_DIR = ROOT / "artifacts" / "features"
ETL_DIR = ROOT / "artifacts" / "etl"
DOCS_DIR = ROOT / "docs" / "changelog"


def load_optional(path):
    """Load .npy file if it exists, else return None."""
    if path.exists():
        return np.load(str(path)).astype(np.float32)
    return None


def main():
    t_start = time.perf_counter()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load ground truth ──
    y_train = np.load(str(FEAT_DIR / "y_train.npy")).astype(np.float32)
    print(f"y_train: shape={y_train.shape}, mean={y_train.mean():.4f}, std={y_train.std():.4f}\n")

    # ── Load stacking v3 results ──
    v3_oof = load_optional(MODEL_DIR / "stacking_v3_oof.npy")
    v3_test = load_optional(MODEL_DIR / "stacking_v3_test.npy")
    v3_json_path = MODEL_DIR / "stacking_v3_results.json"
    v3_json = None
    if v3_json_path.exists():
        with open(v3_json_path) as f:
            v3_json = json.load(f)

    if v3_oof is None:
        print("ERROR: stacking_v3_oof.npy not found. Run stacking_v3.py first.")
        return

    # ── Load stacking v2 (baseline) ──
    v2_oof = load_optional(MODEL_DIR / "stacking_v2_oof.npy")
    v2_test = load_optional(MODEL_DIR / "stacking_v2_test.npy")

    # ── Load all meta-learner variants ──
    v3_variants = {}
    for variant in ["ridge", "lgb", "catboost", "elasticnet", "ridge+lgb"]:
        oof = load_optional(MODEL_DIR / f"stacking_v3_{variant}_oof.npy")
        test = load_optional(MODEL_DIR / f"stacking_v3_{variant}_test.npy")
        if oof is not None:
            v3_variants[variant] = {"oof": oof, "test": test}

    # ── Load DeBERTa 1M fold1 ──
    deb_test = load_optional(MODEL_DIR / "deberta_lora_fold1_test.npy")

    # ═══════════════════════════════════════════════════════════════
    # Analysis
    # ═══════════════════════════════════════════════════════════════

    lines = []
    def p(msg=""):
        print(msg)
        lines.append(msg)

    p(f"# Stacking v3 Verification Report")
    p(f"\n**Date**: {timestamp}\n")

    # ── 1. v3 OOF quality ──
    p("## 1. Stacking v3 OOF Quality\n")
    v3_rmse = float(np.sqrt(np.mean((v3_oof - y_train) ** 2)))
    p(f"| Metric | Value |")
    p(f"|--------|-------|")
    p(f"| OOF RMSE | {v3_rmse:.5f} |")
    p(f"| Mean | {v3_oof.mean():.4f} |")
    p(f"| Std | {v3_oof.std():.4f} |")
    p(f"| Min | {v3_oof.min():.2f} |")
    p(f"| Max | {v3_oof.max():.2f} |")
    p()

    # ── 2. v3 vs v2 comparison ──
    p("## 2. Stacking v3 vs v2\n")

    verdict = "UNKNOWN"
    if v2_oof is not None:
        v2_rmse = float(np.sqrt(np.mean((v2_oof - y_train) ** 2)))
        improvement = v2_rmse - v3_rmse
        pct = improvement / v2_rmse * 100
        p(f"| Metric | v2 | v3 | Delta |")
        p(f"|--------|----|----|-------|")
        p(f"| OOF RMSE | {v2_rmse:.5f} | {v3_rmse:.5f} | {improvement:+.5f} ({pct:+.2f}%) |")
        p(f"| Mean | {v2_oof.mean():.4f} | {v3_oof.mean():.4f} | {v3_oof.mean()-v2_oof.mean():+.4f} |")
        p(f"| Std | {v2_oof.std():.4f} | {v3_oof.std():.4f} | {v3_oof.std()-v2_oof.std():+.4f} |")
        p()

        if improvement > 0.001:
            verdict = "PASS"
            p(f"**Verdict**: PASS — v3 OOF RMSE improved by {improvement:.5f} ({pct:.2f}%)\n")
        elif improvement > -0.001:
            verdict = "NEUTRAL"
            p(f"**Verdict**: NEUTRAL — v3 and v2 are essentially identical (delta < 0.001)\n")
        else:
            verdict = "FAIL"
            p(f"**Verdict**: FAIL — v3 is WORSE than v2 by {abs(improvement):.5f}\n")

        # Per-row difference distribution
        diff = v3_oof - v2_oof
        p(f"Row-level diff (v3 - v2): mean={diff.mean():.5f}, std={diff.std():.5f}, |diff|>0.1: {np.sum(np.abs(diff)>0.1)} rows\n")
    else:
        p("*stacking_v2_oof.npy not found — cannot compute direct OOF comparison*\n")
        if v2_test is not None and v3_test is not None:
            test_diff = np.abs(v3_test - v2_test)
            p(f"Test prediction comparison (v3 vs v2):")
            p(f"- Mean |diff|: {test_diff.mean():.5f}")
            p(f"- Max |diff|: {test_diff.max():.5f}")
            p(f"- Rows with |diff| > 0.05: {np.sum(test_diff > 0.05)} / {len(test_diff)}")
            p(f"- Correlation: {np.corrcoef(v3_test, v2_test)[0,1]:.6f}")
            p()
            if test_diff.mean() > 0.01:
                verdict = "CHANGED"
                p(f"**Verdict**: CHANGED — test predictions differ significantly from v2\n")
            else:
                verdict = "NEUTRAL"
                p(f"**Verdict**: NEUTRAL — test predictions nearly identical to v2\n")

    # ── 3. Meta-learner breakdown ──
    p("## 3. Meta-Learner Breakdown\n")
    p(f"| Meta-Learner | OOF RMSE | vs Best |")
    p(f"|-------------|----------|---------|")
    variant_rmses = {}
    for name, data in v3_variants.items():
        rmse = float(np.sqrt(np.mean((data["oof"] - y_train) ** 2)))
        variant_rmses[name] = rmse
    best_variant_rmse = min(variant_rmses.values()) if variant_rmses else v3_rmse
    for name, rmse in sorted(variant_rmses.items(), key=lambda x: x[1]):
        marker = " ★" if rmse == best_variant_rmse else ""
        delta = rmse - best_variant_rmse
        p(f"| {name} | {rmse:.5f}{marker} | +{delta:.5f} |")
    p()

    # ── 4. DeBERTa blend simulation ──
    p("## 4. DeBERTa 1M Blend Simulation\n")
    if deb_test is not None:
        target_std, target_mean = y_train.std(), y_train.mean()
        scale = target_std / deb_test.std()
        deb_ve = np.clip((deb_test - deb_test.mean()) * scale + target_mean, 1.0, 5.0)

        deb_oof_path = MODEL_DIR / "deberta_v3base_1m_oof.npy"
        deb_oof = load_optional(deb_oof_path)

        p(f"DeBERTa 1M fold1 test: mean={deb_test.mean():.4f}, std={deb_test.std():.4f}")
        p(f"After VE: mean={deb_ve.mean():.4f}, std={deb_ve.std():.4f}, scale={scale:.4f}\n")

        p(f"| Blend Ratio | Test Mean | Test Std | Notes |")
        p(f"|-------------|-----------|----------|-------|")

        # With stacking v3
        for w_deb in [95, 90, 85, 80, 75]:
            w_stk = 100 - w_deb
            blend_v3 = np.clip(w_deb / 100 * deb_ve + w_stk / 100 * v3_test, 1.0, 5.0)
            note = ""
            if w_deb == 90:
                note = "← mirrors 0.617 recipe"
            p(f"| {w_deb}% DeBERTa_VE + {w_stk}% v3 | {blend_v3.mean():.4f} | {blend_v3.std():.4f} | {note} |")

        # Baseline: with stacking v2
        if v2_test is not None:
            p(f"\n**Baseline (v2):**")
            p(f"| Blend Ratio | Test Mean | Test Std | Notes |")
            p(f"|-------------|-----------|----------|-------|")
            for w_deb in [95, 90, 85, 80]:
                w_stk = 100 - w_deb
                blend_v2 = np.clip(w_deb / 100 * deb_ve + w_stk / 100 * v2_test, 1.0, 5.0)
                note = "← current best 0.61734" if w_deb == 90 else ""
                p(f"| {w_deb}% DeBERTa_VE + {w_stk}% v2 | {blend_v2.mean():.4f} | {blend_v2.std():.4f} | {note} |")

        # Per-variant blend comparison at 90/10
        if len(v3_variants) > 1:
            p(f"\n**90/10 blend per meta-learner variant:**")
            p(f"| Variant | Test Mean | Test Std |")
            p(f"|---------|-----------|----------|")
            for name, data in v3_variants.items():
                blend = np.clip(0.9 * deb_ve + 0.1 * data["test"], 1.0, 5.0)
                p(f"| {name} | {blend.mean():.4f} | {blend.std():.4f} |")
        p()

        # OOF-based blend estimation (if DeBERTa OOF exists)
        if deb_oof is not None:
            deb_oof_ve_scale = target_std / deb_oof.std()
            deb_oof_ve = np.clip((deb_oof - deb_oof.mean()) * deb_oof_ve_scale + target_mean, 1.0, 5.0)
            deb_oof_rmse = float(np.sqrt(np.mean((deb_oof_ve - y_train) ** 2)))
            p(f"**OOF RMSE estimates (DeBERTa VE OOF):**")
            p(f"- DeBERTa VE alone: {deb_oof_rmse:.5f}")
            blend_90_v3 = np.clip(0.9 * deb_oof_ve + 0.1 * v3_oof, 1.0, 5.0)
            blend_90_v3_rmse = float(np.sqrt(np.mean((blend_90_v3 - y_train) ** 2)))
            p(f"- 90% DeBERTa_VE + 10% v3: {blend_90_v3_rmse:.5f}")
            if v2_oof is not None:
                blend_90_v2 = np.clip(0.9 * deb_oof_ve + 0.1 * v2_oof, 1.0, 5.0)
                blend_90_v2_rmse = float(np.sqrt(np.mean((blend_90_v2 - y_train) ** 2)))
                p(f"- 90% DeBERTa_VE + 10% v2: {blend_90_v2_rmse:.5f}")
                delta_blend = blend_90_v2_rmse - blend_90_v3_rmse
                p(f"- Improvement: {delta_blend:+.5f}")
            p()
    else:
        p("*deberta_lora_fold1_test.npy not found — cannot simulate blends*\n")

    # ── 5. Base model contribution analysis ──
    p("## 5. Base Model Contribution\n")
    if v3_json and "meta_learner_details" in v3_json:
        ridge_detail = v3_json["meta_learner_details"].get("ridge", {})
        if "avg_coefs" in ridge_detail:
            p("**Ridge coefficients (positive = helpful, negative = harmful):**\n")
            p("| Model | Coefficient | Signal Type |")
            p("|-------|-------------|-------------|")
            signal_types = {
                "lgb_tfidf": "Text TF-IDF",
                "xgboost": "Text TF-IDF",
                "mlp": "DeBERTa embedding",
                "lgb_safe_dense": "Sentiment+Metadata",
                "xgboost_safe": "Sentiment+Metadata",
                "catboost_safe": "Sentiment+Metadata",
                "ensemble_diverse": "Mixed ensemble",
                "xgb_graph_safe": "Graph features (NEW)",
                "lgb_graph_safe": "Graph features (NEW)",
            }
            for name, coef in sorted(ridge_detail["avg_coefs"].items(), key=lambda x: -x[1]):
                signal = signal_types.get(name, "Unknown")
                p(f"| {name} | {coef:.4f} | {signal} |")
            p()

            # Check if graph models contribute
            graph_coefs = {k: v for k, v in ridge_detail["avg_coefs"].items()
                          if "graph" in k}
            if graph_coefs:
                pos_graph = {k: v for k, v in graph_coefs.items() if v > 0.01}
                neg_graph = {k: v for k, v in graph_coefs.items() if v < -0.01}
                if pos_graph:
                    p(f"**Graph models have POSITIVE Ridge weights**: {pos_graph}")
                    p("→ Graph features contribute useful signal to the meta-learner.\n")
                elif neg_graph:
                    p(f"**Graph models have NEGATIVE Ridge weights**: {neg_graph}")
                    p("→ Graph features may be redundant or harmful. Consider removing.\n")
                else:
                    p("**Graph models have near-zero Ridge weights** → redundant signal.\n")

    # ── 6. Final recommendation ──
    p("## 6. Recommendation\n")
    if verdict == "PASS":
        p("Stacking v3 shows **measurable OOF improvement** over v2.")
        p("Proceed with Kaggle submissions in this order:")
        p("1. `submission-stacking-v3.csv` — standalone (diagnostic)")
        p("2. `submission-deb1m-ve90-sv3-10.csv` — primary")
        p("3. `submission-deb1m-ve85-sv3-15.csv` — explore higher stacking weight\n")
    elif verdict == "CHANGED":
        p("Stacking v3 test predictions differ from v2 but OOF comparison unavailable.")
        p("Submit to Kaggle to determine if the change is beneficial:")
        p("1. `submission-stacking-v3.csv` — standalone (diagnostic)")
        p("2. `submission-deb1m-ve90-sv3-10.csv` — primary\n")
    elif verdict == "NEUTRAL":
        p("Stacking v3 is essentially identical to v2. No improvement detected.")
        p("Consider adding more diverse base models or switching to Route A/B.\n")
    elif verdict == "FAIL":
        p("Stacking v3 is **worse** than v2 on OOF. Possible causes:")
        p("- New base models are too correlated with existing ones")
        p("- Graph features contain leakage that hurts meta-learner generalization")
        p("Revert to stacking v2 and investigate.\n")
    else:
        p("Cannot determine improvement. Submit both v2 and v3 to Kaggle for comparison.\n")

    p(f"---\n*Verification completed in {time.perf_counter()-t_start:.1f}s*")

    # ── Save report ──
    report_path = DOCS_DIR / "stacking-v3-verification.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()

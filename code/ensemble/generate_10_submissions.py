#!/usr/bin/env python3
"""
Generate 10 competitive Kaggle submissions for 2026-06-22
Target: Beat 2nd place (0.47361), Current best: 0.59770

Strategies:
1. VE ratio variations around current best
2. Include DeBERTa-v3-large predictions
3. Different blending strategies
4. Weighted combinations
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# ============================================================
# 1. Load all available predictions
# ============================================================

print("=" * 60)
print("Loading predictions...")
print("=" * 60)

# DeBERTa-v3-base (1M, 5f×5e) - Current best
base_1m = np.load("artifacts/models/deberta_lora_fold1_test.npy")
print(f"✓ DeBERTa-v3-base (1M): {base_1m.shape}, std={base_1m.std():.4f}")

# DeBERTa-v3-large (3M, 3f×3e) - New
try:
    large_3m = np.load("artifacts/models/deberta_large_fold1_test.npy")
    print(f"✓ DeBERTa-v3-large (3M): {large_3m.shape}, std={large_3m.std():.4f}")
except:
    print("✗ DeBERTa-v3-large not available, using base as placeholder")
    large_3m = base_1m.copy()

# Stacking V3 predictions
stacking_rlg = np.load("artifacts/models/stacking_v3_ridge+lgb_test.npy")
stacking_ridge = np.load("artifacts/models/stacking_v3_ridge_test.npy")
stacking_lgb = np.load("artifacts/models/stacking_v3_lgb_test.npy")
stacking_elasticnet = np.load("artifacts/models/stacking_v3_elasticnet_test.npy")
stacking_catboost = np.load("artifacts/models/stacking_v3_catboost_test.npy")

print(f"✓ Stacking V3 ridge+lgb: {stacking_rlg.shape}, std={stacking_rlg.std():.4f}")
print(f"✓ Stacking V3 ridge: {stacking_ridge.shape}, std={stacking_ridge.std():.4f}")
print(f"✓ Stacking V3 lgb: {stacking_lgb.shape}, std={stacking_lgb.std():.4f}")
print(f"✓ Stacking V3 elasticnet: {stacking_elasticnet.shape}, std={stacking_elasticnet.std():.4f}")
print(f"✓ Stacking V3 catboost: {stacking_catboost.shape}, std={stacking_catboost.std():.4f}")

# Load test IDs
test_ids = np.load("artifacts/models/test_tokens.npz", allow_pickle=True)["ids"]
print(f"✓ Test IDs: {len(test_ids)} samples")

# ============================================================
# 2. Helper functions
# ============================================================

def variance_expansion(pred, target_std=1.422, target_mean=3.941):
    """Apply variance expansion to predictions"""
    ve = (pred - pred.mean()) / pred.std() * target_std + target_mean
    return np.clip(ve, 1.0, 5.0)

def create_submission(pred, filename, description):
    """Create submission CSV file"""
    df = pd.DataFrame({
        "id": range(len(pred)),
        "rating": pred
    })
    df.to_csv(f"output/{filename}.csv", index=False)
    print(f"✓ Created: {filename}.csv ({description})")
    return filename

# ============================================================
# 3. Generate 10 competitive submissions
# ============================================================

print("\n" + "=" * 60)
print("Generating 10 competitive submissions...")
print("=" * 60)

submissions = []

# ----------------------------------------------------------
# Strategy 1: NEW VE ratio variations (not yet submitted)
# Based on doc: best is 60/40, next steps are 65/35 and 70/30
# Already tried: 30%, 50%, 55%, 60%, 85%, 88%, 90%
# ----------------------------------------------------------

# Submission 1: VE 65% + Stacking V3 rlg 35% (文档建议)
ve65 = variance_expansion(base_1m)
blend1 = 0.65 * ve65 + 0.35 * stacking_rlg
blend1 = np.clip(blend1, 1.0, 5.0)
submissions.append(create_submission(
    blend1, 
    "sub-20260622-01-ve65-rlg35",
    "VE 65% + Stacking V3 ridge+lgb 35% (文档建议)"
))

# Submission 2: VE 70% + Stacking V3 rlg 30% (文档建议)
blend2 = 0.70 * ve65 + 0.30 * stacking_rlg
blend2 = np.clip(blend2, 1.0, 5.0)
submissions.append(create_submission(
    blend2,
    "sub-20260622-02-ve70-rlg30",
    "VE 70% + Stacking V3 ridge+lgb 30% (文档建议)"
))

# Submission 3: VE 75% + Stacking V3 rlg 25% (更高 VE)
blend3 = 0.75 * ve65 + 0.25 * stacking_rlg
blend3 = np.clip(blend3, 1.0, 5.0)
submissions.append(create_submission(
    blend3,
    "sub-20260622-03-ve75-rlg25",
    "VE 75% + Stacking V3 ridge+lgb 25% (更高 VE)"
))

# ----------------------------------------------------------
# Strategy 2: Include DeBERTa-v3-large predictions
# ----------------------------------------------------------

# Submission 5: Large model + VE + Stacking
ve_large = variance_expansion(large_3m)
blend5 = 0.40 * ve65 + 0.30 * ve_large + 0.30 * stacking_rlg
blend5 = np.clip(blend5, 1.0, 5.0)
submissions.append(create_submission(
    blend5,
    "sub-20260622-05-base40-large30-rlg30",
    "Base VE 40% + Large VE 30% + Stacking V3 rlg 30%"
))

# Submission 6: Large model dominant
blend6 = 0.30 * ve65 + 0.40 * ve_large + 0.30 * stacking_rlg
blend6 = np.clip(blend6, 1.0, 5.0)
submissions.append(create_submission(
    blend6,
    "sub-20260622-06-base30-large40-rlg30",
    "Base VE 30% + Large VE 40% + Stacking V3 rlg 30%"
))

# ----------------------------------------------------------
# Strategy 3: Different stacking combinations
# ----------------------------------------------------------

# Submission 7: VE + Multiple stacking models
blend7 = 0.55 * ve65 + 0.20 * stacking_rlg + 0.15 * stacking_ridge + 0.10 * stacking_lgb
blend7 = np.clip(blend7, 1.0, 5.0)
submissions.append(create_submission(
    blend7,
    "sub-20260622-07-ve55-multi-stack",
    "VE 55% + rlg 20% + ridge 15% + lgb 10%"
))

# Submission 8: VE + All stacking models
blend8 = 0.50 * ve65 + 0.15 * stacking_rlg + 0.10 * stacking_ridge + 0.10 * stacking_lgb + 0.10 * stacking_elasticnet + 0.05 * stacking_catboost
blend8 = np.clip(blend8, 1.0, 5.0)
submissions.append(create_submission(
    blend8,
    "sub-20260622-08-ve50-all-stack",
    "VE 50% + All stacking models weighted"
))

# ----------------------------------------------------------
# Strategy 4: Advanced blending
# ----------------------------------------------------------

# Submission 9: Geometric mean blending
geo_blend = np.power(ve65 * stacking_rlg, 0.5)
geo_blend = (geo_blend - geo_blend.mean()) / geo_blend.std() * 1.422 + 3.941
geo_blend = np.clip(geo_blend, 1.0, 5.0)
submissions.append(create_submission(
    geo_blend,
    "sub-20260622-09-geometric",
    "Geometric mean of VE and Stacking V3 rlg"
))

# Submission 10: Harmonic mean blending
harm_blend = 2 * (ve65 * stacking_rlg) / (ve65 + stacking_rlg + 1e-8)
harm_blend = (harm_blend - harm_blend.mean()) / harm_blend.std() * 1.422 + 3.941
harm_blend = np.clip(harm_blend, 1.0, 5.0)
submissions.append(create_submission(
    harm_blend,
    "sub-20260622-10-harmonic",
    "Harmonic mean of VE and Stacking V3 rlg"
))

# ============================================================
# 4. Summary
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("\n📊 Submission Statistics:")
for i, sub_name in enumerate(submissions, 1):
    pred = pd.read_csv(f"output/{sub_name}.csv")["rating"]
    print(f"{i:2d}. {sub_name}")
    print(f"    Mean: {pred.mean():.4f}, Std: {pred.std():.4f}, Min: {pred.min():.4f}, Max: {pred.max():.4f}")

print("\n📁 Files created in output/")
print("\n💡 Recommendations:")
print("1. Start with sub-20260622-01-ve60-rlg40 (current best configuration)")
print("2. Try sub-20260622-05-base40-large30-rlg30 (includes Large model)")
print("3. Try sub-20260622-09-geometric (geometric mean blending)")
print("\n🎯 Target: Beat 2nd place (0.47361)")
print("=" * 60)

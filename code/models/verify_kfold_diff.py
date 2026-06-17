#!/usr/bin/env python
"""Verify that old (manual shuffle) and new (sklearn KFold) produce different fold assignments."""

import numpy as np
from sklearn.model_selection import KFold

N_FOLDS = 3

# Use a small sample for quick verification
n = 100_000  # representative subset
print(f"Testing with n={n}, N_FOLDS={N_FOLDS}")
print("=" * 60)

# === Old method (deberta_lora.py) ===
rng = np.random.RandomState(42)
indices_old = np.arange(n)
rng.shuffle(indices_old)
fold_sizes = np.full(N_FOLDS, n // N_FOLDS, dtype=int)
fold_sizes[: n % N_FOLDS] += 1
old_folds = np.split(indices_old, np.cumsum(fold_sizes)[:-1])

# === New method (deberta_base_full.py) ===
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
new_folds = [va_idx for _, va_idx in kf.split(np.arange(n))]

# === Compare ===
print("\nFold sizes:")
for i in range(N_FOLDS):
    overlap = len(set(old_folds[i]) & set(new_folds[i]))
    jaccard = overlap / len(set(old_folds[i]) | set(new_folds[i]))
    print(f"  Fold {i+1}: old={len(old_folds[i]):,}, new={len(new_folds[i]):,}, "
          f"overlap={overlap:,}, Jaccard={jaccard:.4f}")

# Check if ANY sample is in the same fold
total_same_fold = 0
for i in range(N_FOLDS):
    total_same_fold += len(set(old_folds[i]) & set(new_folds[i]))
print(f"\nTotal samples in same fold: {total_same_fold:,} / {n:,} ({total_same_fold/n*100:.2f}%)")

if total_same_fold / n < 0.5:
    print("\n❌ CONFIRMED: Old and New KFold produce DIFFERENT fold assignments!")
    print("   This is the root cause of the 3×3 degradation.")
else:
    print("\n✅ Old and New KFold produce similar fold assignments.")
    print("   The root cause must be something else.")

# Also check with full 3M size
print("\n" + "=" * 60)
print("Checking with full 3M size (n=3,007,439)...")
n_full = 3_007_439

rng_full = np.random.RandomState(42)
indices_full = np.arange(n_full)
rng_full.shuffle(indices_full)
fold_sizes_full = np.full(N_FOLDS, n_full // N_FOLDS, dtype=int)
fold_sizes_full[: n_full % N_FOLDS] += 1
old_folds_full = np.split(indices_full, np.cumsum(fold_sizes_full)[:-1])

kf_full = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
new_folds_full = [va_idx for _, va_idx in kf_full.split(np.arange(n_full))]

print("\nFold sizes (full 3M):")
for i in range(N_FOLDS):
    overlap = len(set(old_folds_full[i]) & set(new_folds_full[i]))
    jaccard = overlap / len(set(old_folds_full[i]) | set(new_folds_full[i]))
    print(f"  Fold {i+1}: old={len(old_folds_full[i]):,}, new={len(new_folds_full[i]):,}, "
          f"overlap={overlap:,}, Jaccard={jaccard:.6f}")

total_same = sum(len(set(old_folds_full[i]) & set(new_folds_full[i])) for i in range(N_FOLDS))
print(f"\nTotal samples in same fold: {total_same:,} / {n_full:,} ({total_same/n_full*100:.4f}%)")

if total_same / n_full < 0.5:
    print("\n❌ CONFIRMED: KFold difference is the root cause!")
else:
    print("\n⚠️  KFold assignments are similar. Investigate other causes.")

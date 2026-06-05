#!/usr/bin/env python3.8
"""
Adversarial Validation for Distribution Shift Detection (T11)

Trains a LightGBM binary classifier to distinguish train vs test data.
If AUC ≈ 0.5 → distributions are similar (good).
If AUC > 0.6 → distribution shift detected (warning).

Uses a 10K+10K sample for speed. Does NOT modify original data.
"""

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "artifacts" / "etl" / "train.parquet"
TEST_PATH = ROOT / "etl" / "test.parquet"
# Also check artifacts path
if not TEST_PATH.exists():
    TEST_PATH = ROOT / "artifacts" / "etl" / "test.parquet"
REPORT_PATH = ROOT / "docs" / "changelog" / "adversarial-validation.md"

SAMPLE_SIZE = 10_000
SEED = 42


# ── Feature Engineering ────────────────────────────────────────────────
def _engineer_features(df: pd.DataFrame, has_rating: bool = False) -> pd.DataFrame:
    """Create numeric features common to both train and test."""
    out = pd.DataFrame(index=df.index)

    # votes (numeric)
    out["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0)

    # purchased → binary 0/1
    out["purchased"] = df["purchased"].astype(str).str.lower().map(
        {"true": 1, "1": 1, "yes": 1}
    ).fillna(0).astype(int)

    # title_len and comment_len
    out["title_len"] = df["title"].astype(str).str.len()
    out["comment_len"] = df["comment"].astype(str).str.len()

    # price (numeric, fill NaN with 0)
    if "price" in df.columns:
        out["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    else:
        out["price"] = 0

    # rating_number (numeric)
    if "rating_number" in df.columns:
        out["rating_number"] = pd.to_numeric(df["rating_number"], errors="coerce").fillna(0)
    else:
        out["rating_number"] = 0

    # time-based features
    for col in ["review_year", "review_month", "review_weekday", "review_hour"]:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if "is_weekend" in df.columns:
        out["is_weekend"] = pd.to_numeric(df["is_weekend"], errors="coerce").fillna(0)

    # rating — only available in train, include if present
    if has_rating and "rating" in df.columns:
        out["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)

    return out


# ── Core Functions ─────────────────────────────────────────────────────
def adversarial_validate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list = None,
    sample_size: int = SAMPLE_SIZE,
    n_folds: int = 5,
    seed: int = SEED,
) -> dict:
    """
    Adversarial validation: train a classifier to distinguish train vs test.

    Parameters
    ----------
    train_df : DataFrame with training data (has 'rating')
    test_df  : DataFrame with test data (no 'rating')
    feature_cols : list of feature column names (auto-detected if None)
    sample_size : rows to sample from each split
    n_folds  : CV folds
    seed     : random seed

    Returns
    -------
    dict with keys: auc, feature_importance, top_features, recommendation
    """
    print("=" * 60)
    print("ADVERSARIAL VALIDATION — Distribution Shift Detection")
    print("=" * 60)

    # ── Engineer features ──────────────────────────────────────────────
    print("\n[1/5] Engineering features...")
    train_feats = _engineer_features(train_df, has_rating=True)
    test_feats = _engineer_features(test_df, has_rating=False)

    # Use only common columns
    common_cols = sorted(set(train_feats.columns) & set(test_feats.columns))
    print(f"  Common features ({len(common_cols)}): {common_cols}")

    if feature_cols is not None:
        common_cols = [c for c in feature_cols if c in common_cols]

    train_feats = train_feats[common_cols]
    test_feats = test_feats[common_cols]

    # ── Sample for speed ───────────────────────────────────────────────
    print(f"\n[2/5] Sampling {sample_size} rows from each split...")
    n_train = min(sample_size, len(train_feats))
    n_test = min(sample_size, len(test_feats))

    train_sample = train_feats.sample(n=n_train, random_state=seed)
    test_sample = test_feats.sample(n=n_test, random_state=seed)

    # Label: train=0, test=1
    train_sample = train_sample.copy()
    test_sample = test_sample.copy()
    train_sample["label"] = 0
    test_sample["label"] = 1

    # Combine and shuffle
    data = pd.concat([train_sample, test_sample], ignore_index=True)
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    X = data[common_cols].values
    y = data["label"].values

    print(f"  Combined shape: {X.shape}, label distribution: "
          f"train={sum(y==0)}, test={sum(y==1)}")

    # ── Train LightGBM with CV ─────────────────────────────────────────
    print(f"\n[3/5] Training LightGBM ({n_folds}-fold CV)...")
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": seed,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
    }

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds = np.zeros(len(X))
    fold_importances = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=common_cols)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=common_cols, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=300,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        oof_preds[val_idx] = model.predict(X_val)
        fold_importances.append(model.feature_importance(importance_type="gain"))

        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        print(f"  Fold {fold_idx + 1}: AUC = {fold_auc:.4f}")

    # ── Aggregate Results ──────────────────────────────────────────────
    overall_auc = roc_auc_score(y, oof_preds)
    mean_importance = np.mean(fold_importances, axis=0)
    importance_df = pd.DataFrame({
        "feature": common_cols,
        "importance": mean_importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print(f"\n[4/5] Overall OOF AUC = {overall_auc:.4f}")

    # ── Top-10 features ────────────────────────────────────────────────
    top10 = importance_df.head(10)
    print("\n  Top-10 Features by Importance:")
    for _, row in top10.iterrows():
        print(f"    {row['feature']:20s}  {row['importance']:>12.1f}")

    # ── Recommendation ─────────────────────────────────────────────────
    if overall_auc > 0.6:
        shift_features = top10[top10["importance"] > 0]["feature"].tolist()
        recommendation = (
            f"⚠️ DISTRIBUTION SHIFT DETECTED (AUC={overall_auc:.4f} > 0.6). "
            f"Features showing shift: {shift_features}. "
            "Consider feature alignment, domain adaptation, or re-sampling."
        )
        warning = True
    else:
        recommendation = (
            f"✅ No significant distribution shift (AUC={overall_auc:.4f} ≈ 0.5). "
            "Train and test distributions are similar. Proceed with confidence."
        )
        warning = False

    print(f"\n[5/5] {recommendation}")

    return {
        "auc": overall_auc,
        "feature_importance": importance_df.to_dict(orient="records"),
        "top_features": top10["feature"].tolist(),
        "warning": warning,
        "recommendation": recommendation,
    }


def identify_distribution_shift(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    threshold_auc: float = 0.6,
    sample_size: int = SAMPLE_SIZE,
) -> dict:
    """
    High-level wrapper: run adversarial validation and flag shifts.

    Parameters
    ----------
    train_df, test_df : raw DataFrames
    threshold_auc : AUC above this triggers a warning
    sample_size : rows to sample per split

    Returns
    -------
    dict with auc, shifted_features, recommendation
    """
    result = adversarial_validate(train_df, test_df, sample_size=sample_size)

    shifted = []
    if result["auc"] > threshold_auc:
        # Features contributing most to the shift
        shifted = [
            f["feature"]
            for f in result["feature_importance"]
            if f["importance"] > 0
        ][:10]

    return {
        "auc": result["auc"],
        "shifted_features": shifted,
        "recommendation": result["recommendation"],
    }


# ── Report Generation ──────────────────────────────────────────────────
def generate_report(result: dict, output_path: Path = REPORT_PATH) -> None:
    """Write adversarial validation report to markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    top10 = result["feature_importance"][:10]

    lines = [
        "# Adversarial Validation Report — T11",
        "",
        "## Objective",
        "Detect distribution shift between train and test sets using a LightGBM binary classifier.",
        "If AUC ≈ 0.5 → distributions are similar (good). If AUC > 0.6 → shift detected.",
        "",
        "## Method",
        "- Labeled train=0, test=1",
        "- Sampled 10K rows from each split for speed",
        "- Trained LightGBM binary classifier with 5-fold stratified CV",
        "- Measured OOF AUC and feature importance (gain)",
        "",
        "## Results",
        "",
        f"**Overall OOF AUC: {result['auc']:.4f}**",
        "",
    ]

    if result["warning"]:
        lines.append(f"> ⚠️ **Distribution shift detected** (AUC > 0.6)")
    else:
        lines.append(f"> ✅ **No significant distribution shift** (AUC ≈ 0.5)")

    lines += [
        "",
        "## Top-10 Features by Importance",
        "",
        "| Rank | Feature | Importance (gain) |",
        "|------|---------|-------------------|",
    ]
    for i, f in enumerate(top10, 1):
        lines.append(f"| {i} | {f['feature']} | {f['importance']:.1f} |")

    lines += [
        "",
        "## Recommendation",
        "",
        result["recommendation"],
        "",
        "## Notes",
        "- This analysis is exploratory and does NOT modify the original data.",
        "- Features used: votes, purchased, title_len, comment_len, price, rating_number, time features.",
        "- `rating` is excluded from the classifier since it exists only in train.",
        f"- AUC ≈ 0.5 means train/test are indistinguishable (desired outcome).",
    ]

    output_path.write_text("\n".join(lines) + "\n")
    print(f"\n📄 Report saved to: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────
def main():
    print(f"Loading data from:\n  train: {TRAIN_PATH}\n  test:  {TEST_PATH}\n")

    # Load with pandas (small enough for 10K samples)
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    # Run adversarial validation
    result = adversarial_validate(train_df, test_df)

    # Generate report
    generate_report(result)

    # Also run identify_distribution_shift for completeness
    print("\n" + "=" * 60)
    print("identify_distribution_shift() wrapper result:")
    shift_result = identify_distribution_shift(train_df, test_df, threshold_auc=0.6)
    print(f"  AUC: {shift_result['auc']:.4f}")
    print(f"  Shifted features: {shift_result['shifted_features']}")
    print(f"  Recommendation: {shift_result['recommendation']}")

    return result


if __name__ == "__main__":
    result = main()
    sys.exit(0)

"""
K-Fold Target Encoding with leakage prevention.

For training data: uses K-Fold cross-validation so each row's encoding
is computed from the OTHER folds only (no leakage).

For test data: uses the FULL training mean per group.

Bayesian smoothing formula:
    smoothed_te = (count * group_mean + smoothing * global_mean) / (count + smoothing)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def kf_target_encode(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str,
    target_col: str = "rating",
    n_splits: int = 5,
    smoothing: float = 1.0,
) -> tuple:
    """
    K-Fold target encoding with Bayesian smoothing and leak prevention.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with group_col and target_col.
    test_df : pd.DataFrame
        Test data with group_col (no target_col needed).
    group_col : str
        Column to group by (e.g., 'user_id', 'prod_id').
    target_col : str
        Target column name (default 'rating').
    n_splits : int
        Number of K-Fold splits.
    smoothing : float
        Bayesian smoothing weight (higher = more shrinkage toward global mean).

    Returns
    -------
    train_encoded : pd.Series
        Target-encoded values for training set (leak-free).
    test_encoded : pd.Series
        Target-encoded values for test set (using full train mean).
    """
    global_mean = train_df[target_col].mean()

    # --- Train encoding via K-Fold (no leakage) ---
    train_encoded = pd.Series(np.nan, index=train_df.index, dtype=float)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(train_df):
        fold_train = train_df.iloc[train_idx]

        # Compute per-group stats on OTHER folds only
        group_stats = fold_train.groupby(group_col)[target_col].agg(["mean", "count"])
        group_stats.columns = ["group_mean", "group_count"]

        # Bayesian smoothing
        group_stats["smoothed"] = (
            group_stats["group_count"] * group_stats["group_mean"]
            + smoothing * global_mean
        ) / (group_stats["group_count"] + smoothing)

        # Map to validation fold
        val_groups = train_df.iloc[val_idx][group_col]
        mapped = val_groups.map(group_stats["smoothed"])

        # Fallback for unseen groups in validation fold
        mapped = mapped.fillna(global_mean)
        train_encoded.iloc[val_idx] = mapped.values

    # --- Test encoding using FULL train mean ---
    full_stats = train_df.groupby(group_col)[target_col].agg(["mean", "count"])
    full_stats.columns = ["group_mean", "group_count"]

    full_stats["smoothed"] = (
        full_stats["group_count"] * full_stats["group_mean"]
        + smoothing * global_mean
    ) / (full_stats["group_count"] + smoothing)

    test_encoded = test_df[group_col].map(full_stats["smoothed"])
    test_encoded = test_encoded.fillna(global_mean)

    return train_encoded, test_encoded


def generate_target_encodings(
    train_path: str = "artifacts/etl/train.parquet",
    test_path: str = "artifacts/etl/test.parquet",
    output_dir: str = "artifacts/features",
    n_splits: int = 5,
    smoothing: float = 1.0,
):
    """
    Generate target encoding features for user_id and prod_id.

    Outputs:
        - artifacts/features/te_user.parquet  (id, user_te)
        - artifacts/features/te_prod.parquet  (id, prod_te)
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # Mapping: group_col -> output column name
    col_map = {"user_id": "user_te", "prod_id": "prod_te"}

    for group_col, output_name in [("user_id", "te_user"), ("prod_id", "te_prod")]:
        print(f"Computing target encoding for {group_col}...")

        train_te, test_te = kf_target_encode(
            train_df=train_df,
            test_df=test_df,
            group_col=group_col,
            target_col="rating",
            n_splits=n_splits,
            smoothing=smoothing,
        )

        te_col = col_map[group_col]

        # Combine train and test into single DataFrame with id
        result = pd.concat([
            pd.DataFrame({"id": train_df["id"], te_col: train_te}),
            pd.DataFrame({"id": test_df["id"], te_col: test_te}),
        ], ignore_index=True)

        out_path = os.path.join(output_dir, f"{output_name}.parquet")
        result.to_parquet(out_path, index=False)
        print(f"  Saved {out_path} ({len(result)} rows)")

    print("Done: target encoding features generated.")


if __name__ == "__main__":
    generate_target_encodings()

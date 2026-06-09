"""
Comprehensive leakage verification tests for all K-Fold feature implementations.

Tests verify that:
1. Train features use K-Fold (stats from OTHER folds only, not current row)
2. Test features use full train stats (correct behavior, no leakage possible)
3. No feature uses its own target value

Leakage Mechanism (from audit):
  - Original user_stats/product_stats/category_stats compute groupBy avg(rating)
    which INCLUDES the row's own rating → leakage
  - K-Fold fixes compute stats on OTHER folds only → no leakage

Test Strategy (Gold Standard):
  - Replicate the exact K-Fold split used by each implementation
  - For each validation fold, manually compute expected stats from OTHER folds
  - Compare actual output against manually-computed expected values
  - This is the ONLY reliable way to verify no leakage
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from code.features.target_encoding import kf_target_encode
from code.features.user_stats_kfold import compute_user_stats_kfold, compute_user_stats_full
from code.features.product_stats_kfold import compute_product_stats_kfold
from code.features.category_stats_kfold import compute_category_stats_kfold


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def user_data():
    """Dataset for user stats testing.

    User 0: 5 reviews with ratings [1.0, 2.0, 4.0, 5.0, 3.0] (mean=3.0)
    User 1: 3 reviews with ratings [1.0, 3.0, 5.0] (mean=3.0)
    User 2: 2 reviews with ratings [2.0, 4.0] (mean=3.0)
    User 3: 1 review with rating [5.0]
    """
    train = pd.DataFrame({
        "id": list(range(11)),
        "user_id": [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3],
        "rating": [1.0, 2.0, 4.0, 5.0, 3.0, 1.0, 3.0, 5.0, 2.0, 4.0, 5.0],
        "votes": [10, 20, 30, 40, 50, 15, 25, 35, 5, 15, 10],
        "purchased": ["True", "False", "True", "False", "True",
                       "True", "False", "True", "False", "True", "True"],
    })
    return train


@pytest.fixture
def product_data():
    """Dataset for product stats testing.

    Product A: 5 reviews with ratings [1.0, 2.0, 4.0, 5.0, 3.0]
    Product B: 3 reviews with ratings [1.0, 3.0, 5.0]
    Product C: 2 reviews with ratings [2.0, 4.0]
    Product D: 1 review with rating [5.0]
    """
    train = pd.DataFrame({
        "id": list(range(11)),
        "parent_prod_id": ["A", "A", "A", "A", "A", "B", "B", "B", "C", "C", "D"],
        "rating": [1.0, 2.0, 4.0, 5.0, 3.0, 1.0, 3.0, 5.0, 2.0, 4.0, 5.0],
    })
    prodinfo = pd.DataFrame({
        "parent_prod_id": ["A", "B", "C", "D"],
        "price": [10.0, 20.0, 30.0, 40.0],
        "rating_number": [100, 200, 50, 25],
        "main_category": ["Electronics", "Books", "Electronics", "Books"],
    })
    return train, prodinfo


@pytest.fixture
def category_data():
    """Dataset for category stats testing.

    Electronics: reviews from products A, C → ratings [1.0, 2.0, 4.0, 5.0, 3.0, 2.0, 4.0]
    Books: reviews from products B, D → ratings [1.0, 3.0, 5.0, 5.0]
    """
    train = pd.DataFrame({
        "id": list(range(11)),
        "parent_prod_id": ["A", "A", "A", "A", "A", "B", "B", "B", "C", "C", "D"],
        "rating": [1.0, 2.0, 4.0, 5.0, 3.0, 1.0, 3.0, 5.0, 2.0, 4.0, 5.0],
    })
    prodinfo = pd.DataFrame({
        "parent_prod_id": ["A", "B", "C", "D"],
        "price": [10.0, 20.0, 30.0, 40.0],
        "main_category": ["Electronics", "Books", "Electronics", "Books"],
    })
    return train, prodinfo


@pytest.fixture
def target_encoding_data():
    """Dataset for target encoding testing.

    User 0: 5 reviews with ratings [1.0, 2.0, 4.0, 5.0, 3.0]
    User 1: 3 reviews with ratings [1.0, 3.0, 5.0]
    User 2: 2 reviews with ratings [2.0, 4.0]
    User 3: 1 review with rating [5.0]
    """
    train = pd.DataFrame({
        "id": list(range(11)),
        "user_id": [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3],
        "prod_id": [0, 1, 2, 3, 4, 0, 1, 2, 0, 1, 0],
        "rating": [1.0, 2.0, 4.0, 5.0, 3.0, 1.0, 3.0, 5.0, 2.0, 4.0, 5.0],
    })
    test = pd.DataFrame({
        "id": range(100, 105),
        "user_id": [0, 1, 2, 3, 99],
        "prod_id": [0, 0, 0, 0, 0],
    })
    return train, test


# ---------------------------------------------------------------------------
# Helper: Gold-Standard Leakage Verification
# ---------------------------------------------------------------------------

def verify_user_stats_no_leakage(train_pdf, result, n_splits):
    """Gold-standard verification: replicate K-Fold split, check each fold.

    For each validation fold, manually compute user stats from OTHER folds,
    then compare against the actual output. This catches ANY leakage.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Need purchased_bin for full verification
    train_copy = train_pdf.copy()
    train_copy["_purchased_bin"] = (
        train_copy["purchased"]
        .map({"True": 1.0, "False": 0.0, True: 1.0, False: 0.0})
        .fillna(0.0)
    )

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_pdf)):
        other_df = train_copy.iloc[train_idx]
        val_ids = train_pdf.iloc[val_idx]

        # Compute expected stats from other folds
        expected_stats = other_df.groupby("user_id").agg(
            exp_avg_rating=("rating", "mean"),
            exp_num_reviews=("rating", "count"),
            exp_avg_votes=("votes", "mean"),
            exp_purchased_rate=("_purchased_bin", "mean"),
            exp_rating_std=("rating", "std"),
        )
        expected_stats["exp_rating_std"] = expected_stats["exp_rating_std"].fillna(0.0)

        for idx in val_idx:
            row = train_pdf.iloc[idx]
            uid = row["user_id"]
            rid = row["id"]

            actual_row = result[result["id"] == rid]
            assert len(actual_row) == 1, f"Row id={rid} not found in result"

            if uid in expected_stats.index:
                exp = expected_stats.loc[uid]

                # Verify avg_rating (most critical - this is where leakage occurs)
                actual_avg = actual_row["avg_rating"].values[0]
                assert abs(actual_avg - exp["exp_avg_rating"]) < 1e-10, (
                    f"LEAKAGE DETECTED: Row id={rid} (user {uid}): "
                    f"actual avg_rating={actual_avg}, "
                    f"expected from other folds={exp['exp_avg_rating']}"
                )

                # Verify num_reviews
                actual_count = actual_row["num_reviews"].values[0]
                assert actual_count == exp["exp_num_reviews"], (
                    f"Row id={rid}: actual num_reviews={actual_count}, "
                    f"expected={exp['exp_num_reviews']}"
                )

                # Verify avg_votes
                actual_votes = actual_row["avg_votes"].values[0]
                assert abs(actual_votes - exp["exp_avg_votes"]) < 1e-10, (
                    f"Row id={rid}: actual avg_votes={actual_votes}, "
                    f"expected={exp['exp_avg_votes']}"
                )
            else:
                # User not in other folds → should be NaN
                actual_avg = actual_row["avg_rating"].values[0]
                assert np.isnan(actual_avg), (
                    f"Row id={rid} (user {uid}): expected NaN (user not in other folds), "
                    f"got {actual_avg}"
                )


def verify_product_stats_no_leakage(train_pdf, prodinfo_pdf, oof_result, n_splits):
    """Gold-standard verification for product stats."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_pdf)):
        other_df = train_pdf.iloc[train_idx]
        val_rows = train_pdf.iloc[val_idx]

        # Compute expected prod_avg_rating from other folds
        expected = other_df.groupby("parent_prod_id")["rating"].agg(["mean", "count"])
        expected.columns = ["exp_avg", "exp_count"]

        for idx in val_idx:
            row = train_pdf.iloc[idx]
            prod = row["parent_prod_id"]
            rid = row["id"]

            actual_row = oof_result[oof_result["id"] == rid]
            assert len(actual_row) == 1, f"Row id={rid} not found in result"

            if prod in expected.index:
                actual_avg = actual_row["prod_avg_rating"].values[0]
                exp_avg = expected.loc[prod, "exp_avg"]
                assert abs(actual_avg - exp_avg) < 1e-10, (
                    f"LEAKAGE DETECTED: Row id={rid} (product {prod}): "
                    f"actual prod_avg_rating={actual_avg}, "
                    f"expected from other folds={exp_avg}"
                )
            else:
                actual_avg = actual_row["prod_avg_rating"].values[0]
                assert np.isnan(actual_avg), (
                    f"Row id={rid} (product {prod}): expected NaN, got {actual_avg}"
                )


def verify_category_stats_no_leakage(train_pdf, prodinfo_pdf, oof_result, n_splits):
    """Gold-standard verification for category stats."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    prod_cat = prodinfo_pdf[["parent_prod_id", "main_category"]].drop_duplicates(
        subset=["parent_prod_id"]
    )
    train_with_cat = train_pdf.merge(prod_cat, on="parent_prod_id", how="left")

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_pdf)):
        other_df = train_pdf.iloc[train_idx]

        # Compute expected category stats from other folds
        other_with_cat = other_df[["parent_prod_id", "rating"]].merge(
            prod_cat, on="parent_prod_id", how="left"
        )
        expected = other_with_cat.groupby("main_category")["rating"].agg(["mean", "count"])
        expected.columns = ["exp_avg", "exp_count"]

        for idx in val_idx:
            cat = train_with_cat.iloc[idx]["main_category"]
            rid = train_pdf.iloc[idx]["id"]

            actual_row = oof_result[oof_result["id"] == rid]
            assert len(actual_row) == 1, f"Row id={rid} not found in result"

            if cat in expected.index:
                actual_avg = actual_row["cat_avg_rating"].values[0]
                exp_avg = expected.loc[cat, "exp_avg"]
                assert abs(actual_avg - exp_avg) < 1e-10, (
                    f"LEAKAGE DETECTED: Row id={rid} (category {cat}): "
                    f"actual cat_avg_rating={actual_avg}, "
                    f"expected from other folds={exp_avg}"
                )
            else:
                actual_avg = actual_row["cat_avg_rating"].values[0]
                assert np.isnan(actual_avg), (
                    f"Row id={rid} (category {cat}): expected NaN, got {actual_avg}"
                )


# ---------------------------------------------------------------------------
# User Stats K-Fold Tests
# ---------------------------------------------------------------------------

class TestUserStatsKfold:
    """Leakage tests for user_stats_kfold.compute_user_stats_kfold."""

    def test_no_leakage_gold_standard(self, user_data):
        """GOLD STANDARD: Manually verify each fold's stats come from OTHER folds only.

        This is the definitive leakage test. We replicate the exact K-Fold split
        and verify that each validation row's avg_rating matches the mean of
        that user's ratings in the OTHER folds (excluding the current fold).
        """
        result = compute_user_stats_kfold(user_data, n_splits=5)
        verify_user_stats_no_leakage(user_data, result, n_splits=5)

    def test_no_leakage_3fold(self, user_data):
        """Verify no leakage with 3-fold split."""
        result = compute_user_stats_kfold(user_data, n_splits=3)
        verify_user_stats_no_leakage(user_data, result, n_splits=3)

    def test_no_leakage_2fold(self, user_data):
        """Verify no leakage with 2-fold split."""
        result = compute_user_stats_kfold(user_data, n_splits=2)
        verify_user_stats_no_leakage(user_data, result, n_splits=2)

    def test_kfold_stats_differ_from_full_mean(self, user_data):
        """K-Fold stats should NOT all equal the full-group mean for a multi-review user.

        This is a probabilistic test: with enough reviews and varied ratings,
        it's virtually impossible for all K-Fold folds to produce the same mean
        as the full-group mean.
        """
        result = compute_user_stats_kfold(user_data, n_splits=5)

        # User 0 has 5 reviews with ratings [1,2,4,5,3], full mean=3.0
        # With 5 folds, user 0's 5 reviews span at most 5 folds.
        # Each fold's "other" data has 4 of user 0's reviews.
        # The 4-review subsets have different means (they exclude different ratings).
        user0_ids = user_data[user_data["user_id"] == 0]["id"].values
        user0_result = result[result["id"].isin(user0_ids)]["avg_rating"]
        full_mean = user_data[user_data["user_id"] == 0]["rating"].mean()

        # At least one row should differ from full mean
        all_same = all(abs(v - full_mean) < 1e-6 for v in user0_result.values)
        assert not all_same, (
            f"All user 0 K-Fold stats equal full mean {full_mean}. "
            f"Values: {list(user0_result.values)}. "
            f"With 5 varied ratings, this should not happen."
        )

    def test_unseen_user_gets_nan(self, user_data):
        """Users with only 1 review in val fold (not in other folds) get NaN."""
        result = compute_user_stats_kfold(user_data, n_splits=5)

        # User 3 has only 1 review (id=10). In whatever fold it lands,
        # the other folds have no user 3 data → NaN
        user3_stat = result[result["id"] == 10]["avg_rating"].values[0]
        assert np.isnan(user3_stat), (
            f"User 3 (single review) got avg_rating={user3_stat}, expected NaN"
        )

    def test_output_shape_and_types(self, user_data):
        """Output must have correct columns, length, and types."""
        result = compute_user_stats_kfold(user_data, n_splits=5)

        expected_cols = {"id", "avg_rating", "num_reviews", "avg_votes",
                         "purchased_rate", "rating_std"}
        assert expected_cols.issubset(set(result.columns)), (
            f"Missing columns: {expected_cols - set(result.columns)}"
        )
        assert len(result) == len(user_data), (
            f"Row count mismatch: {len(result)} vs {len(user_data)}"
        )
        assert result["avg_rating"].dtype in [np.float64, np.float32]
        assert result["rating_std"].dtype in [np.float64, np.float32]

    def test_test_set_uses_full_stats(self, user_data):
        """compute_user_stats_full should return FULL stats (for test set mapping)."""
        full_stats = compute_user_stats_full(user_data)

        # User 0: full mean of [1,2,4,5,3] = 3.0
        assert full_stats.loc[0, "avg_rating"] == pytest.approx(3.0, abs=1e-6)
        assert full_stats.loc[0, "num_reviews"] == 5

        # User 2: full mean of [2,4] = 3.0
        assert full_stats.loc[2, "avg_rating"] == pytest.approx(3.0, abs=1e-6)
        assert full_stats.loc[2, "num_reviews"] == 2

    def test_deterministic_output(self, user_data):
        """Same input + same random_state → identical output."""
        result1 = compute_user_stats_kfold(user_data, n_splits=5)
        result2 = compute_user_stats_kfold(user_data, n_splits=5)
        pd.testing.assert_frame_equal(result1, result2)

    def test_single_review_user_always_nan_in_val(self, user_data):
        """A user with exactly 1 review: it's always alone in its fold's val set,
        and the other folds have 0 reviews for that user → always NaN."""
        result = compute_user_stats_kfold(user_data, n_splits=3)

        # User 3 has 1 review (id=10). No matter which fold, other folds
        # won't have user 3 → NaN.
        user3_stat = result[result["id"] == 10]["avg_rating"].values[0]
        assert np.isnan(user3_stat), (
            f"Single-review user got {user3_stat}, expected NaN"
        )


# ---------------------------------------------------------------------------
# Product Stats K-Fold Tests
# ---------------------------------------------------------------------------

class TestProductStatsKfold:
    """Leakage tests for product_stats_kfold.compute_product_stats_kfold."""

    def test_no_leakage_gold_standard(self, product_data):
        """GOLD STANDARD: Verify each fold's product stats come from OTHER folds."""
        train, prodinfo = product_data
        oof_result, _ = compute_product_stats_kfold(train, prodinfo, n_splits=5)
        verify_product_stats_no_leakage(train, prodinfo, oof_result, n_splits=5)

    def test_no_leakage_3fold(self, product_data):
        """Verify no leakage with 3-fold split."""
        train, prodinfo = product_data
        oof_result, _ = compute_product_stats_kfold(train, prodinfo, n_splits=3)
        verify_product_stats_no_leakage(train, prodinfo, oof_result, n_splits=3)

    def test_no_leakage_2fold(self, product_data):
        """Verify no leakage with 2-fold split."""
        train, prodinfo = product_data
        oof_result, _ = compute_product_stats_kfold(train, prodinfo, n_splits=2)
        verify_product_stats_no_leakage(train, prodinfo, oof_result, n_splits=2)

    def test_kfold_stats_differ_from_full_mean(self, product_data):
        """K-Fold stats should NOT all equal the full-group mean."""
        train, prodinfo = product_data
        oof_result, _ = compute_product_stats_kfold(train, prodinfo, n_splits=5)

        # Product A: ratings [1,2,4,5,3], full mean=3.0
        prod_a_ids = train[train["parent_prod_id"] == "A"]["id"].values
        kfold_a = oof_result[oof_result["id"].isin(prod_a_ids)]["prod_avg_rating"]
        full_mean_a = train[train["parent_prod_id"] == "A"]["rating"].mean()

        all_same = all(abs(v - full_mean_a) < 1e-6 for v in kfold_a.values)
        assert not all_same, (
            f"All Product A K-Fold stats equal full mean {full_mean_a}. "
            f"Values: {list(kfold_a.values)}"
        )

    def test_full_stats_correct(self, product_data):
        """Full stats (for test set) should be correct."""
        train, prodinfo = product_data
        _, full_stats = compute_product_stats_kfold(train, prodinfo, n_splits=5)

        # Product A: mean of [1,2,4,5,3] = 3.0
        assert full_stats.loc["A", "prod_avg_rating"] == pytest.approx(3.0, abs=1e-6)
        assert full_stats.loc["A", "prod_num_reviews"] == 5

        # Product B: mean of [1,3,5] = 3.0
        assert full_stats.loc["B", "prod_avg_rating"] == pytest.approx(3.0, abs=1e-6)

    def test_output_shape_and_types(self, product_data):
        """Output must have correct columns and length."""
        train, prodinfo = product_data
        oof_result, _ = compute_product_stats_kfold(train, prodinfo, n_splits=5)

        expected_cols = {"id", "prod_avg_rating", "prod_num_reviews",
                         "prod_price", "prod_rating_number", "main_category"}
        assert expected_cols.issubset(set(oof_result.columns)), (
            f"Missing columns: {expected_cols - set(oof_result.columns)}"
        )
        assert len(oof_result) == len(train), "Row count mismatch"

    def test_single_review_product_nan(self, product_data):
        """Product with 1 review: NaN when it's in val fold (not in other folds)."""
        train, prodinfo = product_data
        oof_result, _ = compute_product_stats_kfold(train, prodinfo, n_splits=5)

        # Product D has 1 review (id=10)
        prod_d_stat = oof_result[oof_result["id"] == 10]["prod_avg_rating"].values[0]
        assert np.isnan(prod_d_stat), (
            f"Single-review product got {prod_d_stat}, expected NaN"
        )


# ---------------------------------------------------------------------------
# Category Stats K-Fold Tests
# ---------------------------------------------------------------------------

class TestCategoryStatsKfold:
    """Leakage tests for category_stats_kfold.compute_category_stats_kfold."""

    def test_no_leakage_gold_standard(self, category_data):
        """GOLD STANDARD: Verify each fold's category stats come from OTHER folds."""
        train, prodinfo = category_data
        oof_result, _ = compute_category_stats_kfold(train, prodinfo, n_splits=5)
        verify_category_stats_no_leakage(train, prodinfo, oof_result, n_splits=5)

    def test_no_leakage_3fold(self, category_data):
        """Verify no leakage with 3-fold split."""
        train, prodinfo = category_data
        oof_result, _ = compute_category_stats_kfold(train, prodinfo, n_splits=3)
        verify_category_stats_no_leakage(train, prodinfo, oof_result, n_splits=3)

    def test_no_leakage_2fold(self, category_data):
        """Verify no leakage with 2-fold split."""
        train, prodinfo = category_data
        oof_result, _ = compute_category_stats_kfold(train, prodinfo, n_splits=2)
        verify_category_stats_no_leakage(train, prodinfo, oof_result, n_splits=2)

    def test_full_stats_correct(self, category_data):
        """Full stats (for test set) should be correct."""
        train, prodinfo = category_data
        _, full_stats = compute_category_stats_kfold(train, prodinfo, n_splits=5)

        # Electronics: products A, C → ratings [1,2,4,5,3,2,4] → mean = 21/7 = 3.0
        assert full_stats.loc["Electronics", "cat_avg_rating"] == pytest.approx(3.0, abs=1e-6)

    def test_output_shape_and_types(self, category_data):
        """Output must have correct columns and length."""
        train, prodinfo = category_data
        oof_result, _ = compute_category_stats_kfold(train, prodinfo, n_splits=5)

        expected_cols = {"id", "cat_avg_rating", "cat_avg_price", "cat_rating_std"}
        assert expected_cols.issubset(set(oof_result.columns)), (
            f"Missing columns: {expected_cols - set(oof_result.columns)}"
        )
        assert len(oof_result) == len(train), "Row count mismatch"

    def test_kfold_stats_differ_from_full_mean(self, category_data):
        """K-Fold stats should NOT all equal the full-group mean for a category."""
        train, prodinfo = category_data
        oof_result, _ = compute_category_stats_kfold(train, prodinfo, n_splits=5)

        prod_cat = prodinfo[["parent_prod_id", "main_category"]].drop_duplicates(
            subset=["parent_prod_id"]
        )
        train_with_cat = train.merge(prod_cat, on="parent_prod_id", how="left")

        # Electronics reviews
        elec_ids = train_with_cat[
            train_with_cat["main_category"] == "Electronics"
        ]["id"].values
        kfold_elec = oof_result[oof_result["id"].isin(elec_ids)]["cat_avg_rating"]
        full_mean_elec = train_with_cat[
            train_with_cat["main_category"] == "Electronics"
        ]["rating"].mean()

        all_same = all(abs(v - full_mean_elec) < 1e-6 for v in kfold_elec.values)
        assert not all_same, (
            f"All Electronics K-Fold stats equal full mean {full_mean_elec}. "
            f"Values: {list(kfold_elec.values)}"
        )


# ---------------------------------------------------------------------------
# Target Encoding K-Fold Tests
# ---------------------------------------------------------------------------

class TestTargetEncodingKfold:
    """Leakage tests for target_encoding.kf_target_encode."""

    def test_no_leakage_gold_standard(self, target_encoding_data):
        """GOLD STANDARD: Manually verify each fold's TE comes from OTHER folds.

        Replicate the K-Fold split, compute expected TE from other folds,
        and compare against actual output.
        """
        train, test = target_encoding_data
        train_te, _ = kf_target_encode(
            train, test, group_col="user_id", target_col="rating",
            n_splits=5, smoothing=1.0,
        )

        global_mean = train["rating"].mean()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(train):
            fold_train = train.iloc[train_idx]

            # Compute expected TE from other folds
            group_stats = fold_train.groupby("user_id")["rating"].agg(["mean", "count"])
            group_stats.columns = ["group_mean", "group_count"]
            group_stats["smoothed"] = (
                group_stats["group_count"] * group_stats["group_mean"]
                + 1.0 * global_mean
            ) / (group_stats["group_count"] + 1.0)

            for idx in val_idx:
                row = train.iloc[idx]
                uid = row["user_id"]
                rid = row["id"]
                actual_te = train_te.iloc[idx]

                if uid in group_stats.index:
                    expected_te = group_stats.loc[uid, "smoothed"]
                    assert abs(actual_te - expected_te) < 1e-10, (
                        f"LEAKAGE DETECTED: Row id={rid} (user {uid}): "
                        f"actual TE={actual_te}, expected from other folds={expected_te}"
                    )
                else:
                    assert abs(actual_te - global_mean) < 1e-10, (
                        f"Row id={rid} (user {uid}): expected global_mean={global_mean}, "
                        f"got {actual_te}"
                    )

    def test_no_leakage_3fold(self, target_encoding_data):
        """Verify no leakage with 3-fold split."""
        train, test = target_encoding_data
        train_te, _ = kf_target_encode(
            train, test, group_col="user_id", target_col="rating",
            n_splits=3, smoothing=1.0,
        )

        global_mean = train["rating"].mean()
        kf = KFold(n_splits=3, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(train):
            fold_train = train.iloc[train_idx]
            group_stats = fold_train.groupby("user_id")["rating"].agg(["mean", "count"])
            group_stats.columns = ["group_mean", "group_count"]
            group_stats["smoothed"] = (
                group_stats["group_count"] * group_stats["group_mean"]
                + 1.0 * global_mean
            ) / (group_stats["group_count"] + 1.0)

            for idx in val_idx:
                uid = train.iloc[idx]["user_id"]
                actual_te = train_te.iloc[idx]
                if uid in group_stats.index:
                    expected_te = group_stats.loc[uid, "smoothed"]
                    assert abs(actual_te - expected_te) < 1e-10, (
                        f"Row {train.iloc[idx]['id']}: TE={actual_te}, expected={expected_te}"
                    )

    def test_train_te_not_equal_to_raw_target(self, target_encoding_data):
        """Train TE must NOT equal raw target for every row."""
        train, test = target_encoding_data

        train_te, _ = kf_target_encode(
            train, test, group_col="user_id", target_col="rating",
            n_splits=5, smoothing=1.0,
        )

        exact_match = (train_te.values == train["rating"].values).all()
        assert not exact_match, (
            "Target encoding equals raw target for every row — likely leakage!"
        )

    def test_test_uses_full_train_stats(self, target_encoding_data):
        """Test set encoding should use FULL train stats."""
        train, test = target_encoding_data

        _, test_te = kf_target_encode(
            train, test, group_col="user_id", target_col="rating",
            n_splits=5, smoothing=1.0,
        )

        global_mean = train["rating"].mean()
        stats = train.groupby("user_id")["rating"].agg(["mean", "count"])
        expected = (stats["count"] * stats["mean"] + 1.0 * global_mean) / (
            stats["count"] + 1.0
        )
        mapped = test["user_id"].map(expected).fillna(global_mean)
        np.testing.assert_array_almost_equal(test_te.values, mapped.values, decimal=10)

    def test_unseen_user_gets_global_mean(self, target_encoding_data):
        """Test rows with unseen users should get the global mean."""
        train, test = target_encoding_data

        _, test_te = kf_target_encode(
            train, test, group_col="user_id", target_col="rating",
            n_splits=5, smoothing=1.0,
        )

        global_mean = train["rating"].mean()
        # Last row (user 99) should get global_mean
        np.testing.assert_almost_equal(test_te.iloc[-1], global_mean, decimal=10)

    def test_output_length_and_type(self, target_encoding_data):
        """Output series must match input length and be numeric."""
        train, test = target_encoding_data

        train_te, test_te = kf_target_encode(
            train, test, group_col="user_id", target_col="rating",
            n_splits=5, smoothing=1.0,
        )

        assert len(train_te) == len(train)
        assert len(test_te) == len(test)
        assert train_te.dtype in [np.float64, np.float32]
        assert test_te.dtype in [np.float64, np.float32]

    def test_train_te_values_in_range(self, target_encoding_data):
        """TE values should be bounded within rating range (with smoothing)."""
        train, test = target_encoding_data

        train_te, _ = kf_target_encode(
            train, test, group_col="user_id", target_col="rating",
            n_splits=5, smoothing=1.0,
        )

        assert train_te.min() >= 0.5, f"Min TE {train_te.min()} below expected floor"
        assert train_te.max() <= 5.5, f"Max TE {train_te.max()} above expected ceiling"


# ---------------------------------------------------------------------------
# Cross-Feature Consistency Tests
# ---------------------------------------------------------------------------

class TestCrossFeatureConsistency:
    """Tests that verify consistent behavior across all feature types."""

    def test_deterministic_output_user(self, user_data):
        """Same input + same random_state → identical output (user stats)."""
        r1 = compute_user_stats_kfold(user_data, n_splits=5)
        r2 = compute_user_stats_kfold(user_data, n_splits=5)
        pd.testing.assert_frame_equal(r1, r2)

    def test_deterministic_output_product(self, product_data):
        """Same input + same random_state → identical output (product stats)."""
        train, prodinfo = product_data
        o1, _ = compute_product_stats_kfold(train, prodinfo, n_splits=5)
        o2, _ = compute_product_stats_kfold(train, prodinfo, n_splits=5)
        pd.testing.assert_frame_equal(o1, o2)

    def test_deterministic_output_category(self, category_data):
        """Same input + same random_state → identical output (category stats)."""
        train, prodinfo = category_data
        o1, _ = compute_category_stats_kfold(train, prodinfo, n_splits=5)
        o2, _ = compute_category_stats_kfold(train, prodinfo, n_splits=5)
        pd.testing.assert_frame_equal(o1, o2)

    def test_deterministic_output_target_encoding(self, target_encoding_data):
        """Same input + same random_state → identical output (target encoding)."""
        train, test = target_encoding_data
        t1, _ = kf_target_encode(train, test, "user_id", "rating", 5, 1.0)
        t2, _ = kf_target_encode(train, test, "user_id", "rating", 5, 1.0)
        np.testing.assert_array_equal(t1.values, t2.values)

    def test_more_folds_changes_distribution(self, user_data):
        """Different n_splits should produce different fold assignments → different stats."""
        r2 = compute_user_stats_kfold(user_data, n_splits=2)
        r5 = compute_user_stats_kfold(user_data, n_splits=5)

        assert len(r2) == len(r5) == len(user_data)

        # At least some values should differ
        all_2 = sorted(r2["avg_rating"].dropna().values)
        all_5 = sorted(r5["avg_rating"].dropna().values)
        assert not np.allclose(all_2, all_5, atol=1e-6), (
            "2-fold and 5-fold produce identical results — unexpected"
        )


# ---------------------------------------------------------------------------
# Regression: Verify Leakage Would Be Detected
# ---------------------------------------------------------------------------

class TestLeakageDetection:
    """Meta-tests that verify our gold-standard tests would catch leakage.

    Simulate leaky implementations and verify the gold-standard check fails.
    """

    def test_leaky_user_stats_detected(self, user_data):
        """A leaky implementation (using full mean) should be detected."""
        # Simulate leaky: full mean per user applied to all rows
        full_means = user_data.groupby("user_id")["rating"].mean()
        leaky_result = user_data[["id"]].copy()
        leaky_result["avg_rating"] = user_data["user_id"].map(full_means)
        leaky_result["num_reviews"] = 1  # dummy
        leaky_result["avg_votes"] = 0.0
        leaky_result["purchased_rate"] = 0.0
        leaky_result["rating_std"] = 0.0

        # The gold-standard check should catch that leaky stats don't match
        # the expected "other folds" stats
        with pytest.raises(AssertionError, match="LEAKAGE DETECTED|actual avg_rating"):
            verify_user_stats_no_leakage(user_data, leaky_result, n_splits=5)

    def test_leaky_product_stats_detected(self, product_data):
        """A leaky product implementation should be detected."""
        train, prodinfo = product_data

        # Simulate leaky: full mean per product
        full_means = train.groupby("parent_prod_id")["rating"].mean()
        leaky_result = train[["id"]].copy()
        leaky_result["prod_avg_rating"] = train["parent_prod_id"].map(full_means)
        leaky_result["prod_num_reviews"] = 1

        with pytest.raises(AssertionError, match="LEAKAGE DETECTED|actual prod_avg"):
            verify_product_stats_no_leakage(train, prodinfo, leaky_result, n_splits=5)

    def test_leaky_category_stats_detected(self, category_data):
        """A leaky category implementation should be detected."""
        train, prodinfo = category_data

        prod_cat = prodinfo[["parent_prod_id", "main_category"]].drop_duplicates(
            subset=["parent_prod_id"]
        )
        train_with_cat = train.merge(prod_cat, on="parent_prod_id", how="left")
        full_means = train_with_cat.groupby("main_category")["rating"].mean()

        leaky_result = train[["id"]].copy()
        leaky_result["cat_avg_rating"] = train_with_cat["main_category"].map(full_means).values
        leaky_result["cat_avg_price"] = 0.0
        leaky_result["cat_rating_std"] = 0.0

        with pytest.raises(AssertionError, match="LEAKAGE DETECTED|actual cat_avg"):
            verify_category_stats_no_leakage(train, prodinfo, leaky_result, n_splits=5)

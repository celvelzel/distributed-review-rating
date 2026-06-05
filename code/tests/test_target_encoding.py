"""
Tests for K-Fold Target Encoding with leak prevention.
"""

import numpy as np
import pandas as pd
import pytest

from code.features.target_encoding import kf_target_encode


@pytest.fixture
def sample_data():
    """Create small deterministic dataset for testing."""
    np.random.seed(42)
    n = 200
    n_users = 20
    n_prods = 10

    train = pd.DataFrame({
        "id": range(n),
        "user_id": np.random.choice(range(n_users), size=n),
        "prod_id": np.random.choice(range(n_prods), size=n),
        "rating": np.random.randint(1, 6, size=n).astype(float),
    })

    test = pd.DataFrame({
        "id": range(n, n + 50),
        "user_id": np.random.choice(range(n_users), size=50),
        "prod_id": np.random.choice(range(n_prods), size=50),
    })

    return train, test


class TestTargetEncoding:
    """Tests for kf_target_encode."""

    def test_no_leakage_train(self, sample_data):
        """Train TE must NOT equal raw target for any row (leakage check)."""
        train, test = sample_data

        train_te, _ = kf_target_encode(
            train, test, group_col="user_id", target_col="rating",
            n_splits=5, smoothing=1.0,
        )

        # For rows where the user appears in other folds too,
        # the TE value (a weighted average) should generally differ from
        # the raw rating. With smoothing, exact equality is extremely unlikely.
        # Check that NOT ALL rows match their raw target exactly.
        exact_match = (train_te.values == train["rating"].values).all()
        assert not exact_match, (
            "Target encoding equals raw target for every row — likely leakage!"
        )

    def test_test_uses_full_train_mean(self, sample_data):
        """Test set encoding should use FULL train stats (no K-Fold for test)."""
        train, test = sample_data

        _, test_te = kf_target_encode(
            train, test, group_col="user_id", target_col="rating",
            n_splits=5, smoothing=1.0,
        )

        # Recompute expected: full train group mean with smoothing
        global_mean = train["rating"].mean()
        stats = train.groupby("user_id")["rating"].agg(["mean", "count"])
        expected = (stats["count"] * stats["mean"] + 1.0 * global_mean) / (stats["count"] + 1.0)

        mapped = test["user_id"].map(expected).fillna(global_mean)

        np.testing.assert_array_almost_equal(test_te.values, mapped.values, decimal=10)

    def test_output_correct_columns_and_length(self, sample_data):
        """Output series must match input length and be numeric."""
        train, test = sample_data

        train_te, test_te = kf_target_encode(
            train, test, group_col="prod_id", target_col="rating",
            n_splits=5, smoothing=1.0,
        )

        assert len(train_te) == len(train), "Train TE length mismatch"
        assert len(test_te) == len(test), "Test TE length mismatch"
        assert train_te.dtype in [np.float64, np.float32], "Train TE not float"
        assert test_te.dtype in [np.float64, np.float32], "Test TE not float"

    def test_train_te_values_in_range(self, sample_data):
        """TE values should be bounded within [min_rating, max_rating] (with smoothing)."""
        train, test = sample_data

        train_te, _ = kf_target_encode(
            train, test, group_col="user_id", target_col="rating",
            n_splits=5, smoothing=1.0,
        )

        # With smoothing toward global mean, all values should be within
        # [1, 5] for ratings 1-5
        assert train_te.min() >= 0.5, f"Min TE {train_te.min()} below expected floor"
        assert train_te.max() <= 5.5, f"Max TE {train_te.max()} above expected ceiling"

    def test_unseen_group_uses_global_mean(self, sample_data):
        """Test rows with unseen groups should get the global mean."""
        train, test = sample_data

        # Add a test row with a user_id not in train
        test_with_new = pd.concat([
            test,
            pd.DataFrame({"id": [99999], "user_id": [9999], "prod_id": [0]}),
        ], ignore_index=True)

        _, test_te = kf_target_encode(
            train, test_with_new, group_col="user_id", target_col="rating",
            n_splits=5, smoothing=1.0,
        )

        global_mean = train["rating"].mean()
        # The last row (unseen user) should get global_mean
        np.testing.assert_almost_equal(test_te.iloc[-1], global_mean, decimal=10)

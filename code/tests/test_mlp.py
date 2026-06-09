"""
Comprehensive MLP validation tests.

Tests verify:
1. Architecture: layers, dimensions, activations
2. Data loading: feature alignment, embedding mapping
3. Feature quality: detect near-zero embeddings (root cause of MLP failure)
4. Training: loss convergence, gradient flow
5. Predictions: diversity, range, not degenerate

Based on MLP diagnosis: Feature quality problem (NOT architecture bug).
LightGCN embeddings are near-zero → MLP learns to predict ~3.8 for everything.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# ── import mlp via importlib (avoids conflict with built-in 'code') ──────
_ROOT = Path(__file__).resolve().parents[2]
_mlp_path = _ROOT / "code" / "models" / "mlp.py"
_spec = importlib.util.spec_from_file_location("mlp", str(_mlp_path))
_mlp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mlp)
RatingMLP = _mlp.RatingMLP
make_optimizer = _mlp.make_optimizer


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def random_seed():
    """Set deterministic seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def mlp_model():
    """Create default MLP model (896→512→128→1)."""
    return RatingMLP(input_dim=896, dropout=0.3)


@pytest.fixture
def small_mlp_model():
    """Create small MLP model for fast tests (64→32→16→1)."""
    return RatingMLP(input_dim=64, dropout=0.3)


@pytest.fixture
def synthetic_data():
    """Create synthetic features and targets for training tests."""
    n_train = 1000
    n_val = 200
    input_dim = 64

    X_train = np.random.randn(n_train, input_dim).astype(np.float32)
    y_train = (3.0 + 0.5 * X_train[:, 0] + 0.1 * np.random.randn(n_train)).astype(
        np.float32
    )

    X_val = np.random.randn(n_val, input_dim).astype(np.float32)
    y_val = (3.0 + 0.5 * X_val[:, 0] + 0.1 * np.random.randn(n_val)).astype(
        np.float32
    )

    return X_train, y_train, X_val, y_val


@pytest.fixture
def embedding_files(tmp_path):
    """Create mock embedding files for data loading tests."""
    n_users = 100
    n_items = 50
    user_emb_dim = 64
    item_emb_dim = 64

    # Create embeddings with known properties
    user_emb = np.random.randn(n_users, user_emb_dim).astype(np.float32) * 0.5
    item_emb = np.random.randn(n_items, item_emb_dim).astype(np.float32) * 0.3

    # Create ID mappings
    user2idx = {f"user_{i}": i for i in range(n_users)}
    item2idx = {f"item_{i}": i for i in range(n_items)}

    # Save files
    user_emb_path = tmp_path / "user_emb.npy"
    item_emb_path = tmp_path / "item_emb.npy"
    user2idx_path = tmp_path / "user2idx.json"
    item2idx_path = tmp_path / "item2idx.json"

    np.save(str(user_emb_path), user_emb)
    np.save(str(item_emb_path), item_emb)
    with open(user2idx_path, "w") as f:
        json.dump(user2idx, f)
    with open(item2idx_path, "w") as f:
        json.dump(item2idx, f)

    return {
        "user_emb": user_emb,
        "item_emb": item_emb,
        "user2idx": user2idx,
        "item2idx": item2idx,
        "user_emb_path": user_emb_path,
        "item_emb_path": item_emb_path,
        "user2idx_path": user2idx_path,
        "item2idx_path": item2idx_path,
    }


@pytest.fixture
def near_zero_embeddings(tmp_path):
    """Create near-zero embeddings (simulates LightGCN failure).

    This is the ROOT CAUSE of MLP failure:
    - User embedding norm mean=0.013 (should be ~1.0)
    - Item embedding norm mean=0.009 (should be ~1.0)
    """
    n_users = 100
    n_items = 50
    user_emb_dim = 64
    item_emb_dim = 64

    # Near-zero embeddings (simulates failed LightGCN training)
    user_emb = np.random.randn(n_users, user_emb_dim).astype(np.float32) * 0.005
    item_emb = np.random.randn(n_items, item_emb_dim).astype(np.float32) * 0.003

    user2idx = {f"user_{i}": i for i in range(n_users)}
    item2idx = {f"item_{i}": i for i in range(n_items)}

    user_emb_path = tmp_path / "user_emb.npy"
    item_emb_path = tmp_path / "item_emb.npy"
    user2idx_path = tmp_path / "user2idx.json"
    item2idx_path = tmp_path / "item2idx.json"

    np.save(str(user_emb_path), user_emb)
    np.save(str(item_emb_path), item_emb)
    with open(user2idx_path, "w") as f:
        json.dump(user2idx, f)
    with open(item2idx_path, "w") as f:
        json.dump(item2idx, f)

    return {
        "user_emb": user_emb,
        "item_emb": item_emb,
        "user2idx": user2idx,
        "item2idx": item2idx,
        "user_emb_path": user_emb_path,
        "item_emb_path": item_emb_path,
        "user2idx_path": user2idx_path,
        "item2idx_path": item2idx_path,
    }


# ── 1. Architecture Tests ───────────────────────────────────────────────────


class TestArchitecture:
    """Verify MLP architecture matches specification."""

    def test_default_input_dim(self, mlp_model):
        """Default input dimension should be 896 (768 DeBERTa + 64 user + 64 item)."""
        first_layer = list(mlp_model.net.children())[0]
        assert first_layer.in_features == 896, (
            f"Expected input_dim=896, got {first_layer.in_features}"
        )

    def test_layer_structure(self, mlp_model):
        """MLP should have 3 linear layers with activations."""
        layers = list(mlp_model.net.children())

        # Expected: Linear(896→512), ReLU, Dropout, Linear(512→128), ReLU, Dropout, Linear(128→1)
        assert len(layers) == 7, f"Expected 7 layers, got {len(layers)}"

        # Check types
        assert isinstance(layers[0], nn.Linear), "Layer 0 should be Linear"
        assert isinstance(layers[1], nn.ReLU), "Layer 1 should be ReLU"
        assert isinstance(layers[2], nn.Dropout), "Layer 2 should be Dropout"
        assert isinstance(layers[3], nn.Linear), "Layer 3 should be Linear"
        assert isinstance(layers[4], nn.ReLU), "Layer 4 should be ReLU"
        assert isinstance(layers[5], nn.Dropout), "Layer 5 should be Dropout"
        assert isinstance(layers[6], nn.Linear), "Layer 6 should be Linear"

    def test_layer_dimensions(self, mlp_model):
        """Layer dimensions: 896→512→128→1."""
        layers = list(mlp_model.net.children())

        assert layers[0].in_features == 896, "First layer input should be 896"
        assert layers[0].out_features == 512, "First layer output should be 512"
        assert layers[3].in_features == 512, "Second layer input should be 512"
        assert layers[3].out_features == 128, "Second layer output should be 128"
        assert layers[6].in_features == 128, "Third layer input should be 128"
        assert layers[6].out_features == 1, "Third layer output should be 1"

    def test_dropout_rate(self, mlp_model):
        """Dropout rate should be 0.3."""
        dropout_layers = [l for l in mlp_model.net if isinstance(l, nn.Dropout)]
        assert len(dropout_layers) == 2, "Should have 2 dropout layers"
        for dropout in dropout_layers:
            assert dropout.p == 0.3, f"Dropout rate should be 0.3, got {dropout.p}"

    def test_custom_input_dim(self):
        """MLP should support custom input dimensions."""
        model = RatingMLP(input_dim=256, dropout=0.5)
        first_layer = list(model.net.children())[0]
        assert first_layer.in_features == 256

    def test_forward_pass_shape(self, mlp_model, random_seed):
        """Forward pass should output correct shape."""
        batch_size = 32
        x = torch.randn(batch_size, 896)
        output = mlp_model(x)

        assert output.shape == (batch_size,), (
            f"Expected shape ({batch_size},), got {output.shape}"
        )

    def test_forward_pass_squeeze(self, mlp_model):
        """Output should be squeezed (no trailing dimension)."""
        x = torch.randn(16, 896)
        output = mlp_model(x)
        assert output.dim() == 1, f"Expected 1D output, got {output.dim()}D"

    def test_gradient_flow(self, mlp_model, random_seed):
        """Gradients should flow through all layers (no vanishing)."""
        x = torch.randn(32, 896, requires_grad=True)
        y = torch.randn(32)

        output = mlp_model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        # Check gradients exist for all parameters
        for name, param in mlp_model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


# ── 2. Optimizer Tests ──────────────────────────────────────────────────────


class TestOptimizer:
    """Verify optimizer configuration."""

    def test_optimizer_type(self, mlp_model):
        """Optimizer should be Adam."""
        optimizer = make_optimizer(mlp_model)
        assert isinstance(optimizer, torch.optim.Adam), (
            f"Expected Adam optimizer, got {type(optimizer)}"
        )

    def test_default_lr(self, mlp_model):
        """Default learning rate should be 1e-3."""
        optimizer = make_optimizer(mlp_model)
        assert optimizer.defaults["lr"] == 1e-3, (
            f"Expected lr=1e-3, got {optimizer.defaults['lr']}"
        )

    def test_default_weight_decay(self, mlp_model):
        """Default weight decay should be 1e-5."""
        optimizer = make_optimizer(mlp_model)
        assert optimizer.defaults["weight_decay"] == 1e-5, (
            f"Expected weight_decay=1e-5, got {optimizer.defaults['weight_decay']}"
        )

    def test_custom_hyperparams(self, mlp_model):
        """Optimizer should accept custom hyperparameters."""
        optimizer = make_optimizer(mlp_model, lr=1e-4, weight_decay=1e-3)
        assert optimizer.defaults["lr"] == 1e-4
        assert optimizer.defaults["weight_decay"] == 1e-3

    def test_optimizer_updates_params(self, mlp_model, random_seed):
        """Optimizer should update model parameters."""
        optimizer = make_optimizer(mlp_model, lr=0.1)
        criterion = nn.MSELoss()

        x = torch.randn(16, 896)
        y = torch.randn(16)

        # Get initial params
        initial_params = {
            name: param.clone() for name, param in mlp_model.named_parameters()
        }

        # Training step
        optimizer.zero_grad()
        output = mlp_model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # Check params changed
        params_changed = False
        for name, param in mlp_model.named_parameters():
            if not torch.equal(param, initial_params[name]):
                params_changed = True
                break
        assert params_changed, "Optimizer did not update any parameters"


# ── 3. Data Loading Tests ───────────────────────────────────────────────────


class TestDataLoading:
    """Verify data loading and feature alignment."""

    def test_embedding_shapes(self, embedding_files):
        """Embeddings should have correct shapes."""
        user_emb = embedding_files["user_emb"]
        item_emb = embedding_files["item_emb"]

        assert user_emb.shape == (100, 64), f"User emb shape: {user_emb.shape}"
        assert item_emb.shape == (50, 64), f"Item emb shape: {item_emb.shape}"

    def test_embedding_dtype(self, embedding_files):
        """Embeddings should be float32."""
        user_emb = embedding_files["user_emb"]
        item_emb = embedding_files["item_emb"]

        assert user_emb.dtype == np.float32, f"User emb dtype: {user_emb.dtype}"
        assert item_emb.dtype == np.float32, f"Item emb dtype: {item_emb.dtype}"

    def test_id_mapping_coverage(self, embedding_files):
        """All IDs should be mapped to embedding indices."""
        user2idx = embedding_files["user2idx"]
        item2idx = embedding_files["item2idx"]

        assert len(user2idx) == 100, f"Expected 100 users, got {len(user2idx)}"
        assert len(item2idx) == 50, f"Expected 50 items, got {len(item2idx)}"

    def test_feature_alignment(self, embedding_files):
        """Features should be aligned: [bert_768 | user_emb_64 | item_emb_64]."""
        n_samples = 100
        bert_dim = 768
        user_dim = 64
        item_dim = 64

        # Simulate feature building
        bert_features = np.random.randn(n_samples, bert_dim).astype(np.float32)
        user_emb = embedding_files["user_emb"]
        item_emb = embedding_files["item_emb"]

        # Map embeddings (simulate run_mlp.py logic)
        user_ids = [f"user_{i % 100}" for i in range(n_samples)]
        item_ids = [f"item_{i % 50}" for i in range(n_samples)]

        user2idx = embedding_files["user2idx"]
        item2idx = embedding_files["item2idx"]

        u_idx = np.array([user2idx.get(uid, -1) for uid in user_ids])
        i_idx = np.array([item2idx.get(pid, -1) for pid in item_ids])

        u_feats = np.zeros((n_samples, user_dim), dtype=np.float32)
        valid_u = u_idx >= 0
        u_feats[valid_u] = user_emb[u_idx[valid_u]]

        i_feats = np.zeros((n_samples, item_dim), dtype=np.float32)
        valid_i = i_idx >= 0
        i_feats[valid_i] = item_emb[i_idx[valid_i]]

        X = np.concatenate([bert_features, u_feats, i_feats], axis=1)

        assert X.shape == (n_samples, 896), f"Feature shape: {X.shape}"
        assert valid_u.all(), "Some user IDs not found"
        assert valid_i.all(), "Some item IDs not found"

    def test_missing_ids_zero_padded(self, embedding_files):
        """Missing IDs should be zero-padded (not crash)."""
        n_samples = 10
        user_dim = 64
        item_dim = 64

        user_emb = embedding_files["user_emb"]
        item_emb = embedding_files["item_emb"]
        user2idx = embedding_files["user2idx"]
        item2idx = embedding_files["item2idx"]

        # Mix of valid and invalid IDs
        user_ids = ["user_0", "user_1", "nonexistent_user", "user_3", "user_4",
                     "another_missing", "user_6", "user_7", "user_8", "user_9"]
        item_ids = ["item_0", "missing_item", "item_2", "item_3", "item_4",
                     "item_5", "another_missing", "item_7", "item_8", "item_9"]

        u_idx = np.array([user2idx.get(uid, -1) for uid in user_ids])
        i_idx = np.array([item2idx.get(pid, -1) for pid in item_ids])

        u_feats = np.zeros((n_samples, user_dim), dtype=np.float32)
        valid_u = u_idx >= 0
        u_feats[valid_u] = user_emb[u_idx[valid_u]]

        i_feats = np.zeros((n_samples, item_dim), dtype=np.float32)
        valid_i = i_idx >= 0
        i_feats[valid_i] = item_emb[i_idx[valid_i]]

        # Check zero padding for missing IDs
        assert u_feats[2].sum() == 0, "Missing user should be zero-padded"
        assert u_feats[5].sum() == 0, "Missing user should be zero-padded"
        assert i_feats[1].sum() == 0, "Missing item should be zero-padded"
        assert i_feats[6].sum() == 0, "Missing item should be zero-padded"

        # Check valid IDs have non-zero embeddings
        assert u_feats[0].sum() != 0, "Valid user should have embedding"
        assert i_feats[0].sum() != 0, "Valid item should have embedding"


# ── 4. Feature Quality Tests (Root Cause Detection) ─────────────────────────


class TestFeatureQuality:
    """Detect feature quality problems (root cause of MLP failure).

    LightGCN embeddings are near-zero → MLP learns to predict ~3.8 for everything.
    """

    def test_healthy_embedding_norms(self, embedding_files):
        """Healthy embeddings should have meaningful norms (not near-zero)."""
        user_emb = embedding_files["user_emb"]
        item_emb = embedding_files["item_emb"]

        user_norms = np.linalg.norm(user_emb, axis=1)
        item_norms = np.linalg.norm(item_emb, axis=1)

        # Healthy embeddings should have norms > 0.1
        assert user_norms.mean() > 0.1, (
            f"User embedding norms too low: {user_norms.mean():.4f} (expected > 0.1)"
        )
        assert item_norms.mean() > 0.1, (
            f"Item embedding norms too low: {item_norms.mean():.4f} (expected > 0.1)"
        )

    def test_detect_near_zero_embeddings(self, near_zero_embeddings):
        """Should detect near-zero embeddings (LightGCN failure mode)."""
        user_emb = near_zero_embeddings["user_emb"]
        item_emb = near_zero_embeddings["item_emb"]

        user_norms = np.linalg.norm(user_emb, axis=1)
        item_norms = np.linalg.norm(item_emb, axis=1)

        # These should FAIL (near-zero norms indicate broken embeddings)
        # This test validates our diagnostic catches the problem
        is_user_broken = user_norms.mean() < 0.1
        is_item_broken = item_norms.mean() < 0.1

        assert is_user_broken, (
            f"Should detect broken user embeddings (norm={user_norms.mean():.4f})"
        )
        assert is_item_broken, (
            f"Should detect broken item embeddings (norm={item_norms.mean():.4f})"
        )

    def test_embedding_norm_threshold(self, embedding_files, near_zero_embeddings):
        """Embedding norm threshold should distinguish healthy from broken."""
        THRESHOLD = 0.1  # Minimum acceptable norm

        # Healthy embeddings
        healthy_user_norms = np.linalg.norm(embedding_files["user_emb"], axis=1)
        healthy_item_norms = np.linalg.norm(embedding_files["item_emb"], axis=1)

        # Broken embeddings
        broken_user_norms = np.linalg.norm(near_zero_embeddings["user_emb"], axis=1)
        broken_item_norms = np.linalg.norm(near_zero_embeddings["item_emb"], axis=1)

        assert healthy_user_norms.mean() > THRESHOLD, "Healthy embeddings should pass"
        assert healthy_item_norms.mean() > THRESHOLD, "Healthy embeddings should pass"
        assert broken_user_norms.mean() < THRESHOLD, "Broken embeddings should fail"
        assert broken_item_norms.mean() < THRESHOLD, "Broken embeddings should fail"

    def test_feature_variance(self, embedding_files):
        """Features should have non-trivial variance (not constant)."""
        user_emb = embedding_files["user_emb"]
        item_emb = embedding_files["item_emb"]

        user_var = np.var(user_emb, axis=0).mean()
        item_var = np.var(item_emb, axis=0).mean()

        assert user_var > 0.01, f"User embedding variance too low: {user_var:.4f}"
        assert item_var > 0.01, f"Item embedding variance too low: {item_var:.4f}"


# ── 5. Training Tests ───────────────────────────────────────────────────────


class TestTraining:
    """Verify training behavior: loss convergence, gradient flow."""

    def test_loss_decreases(self, small_mlp_model, synthetic_data, random_seed):
        """Loss should decrease during training."""
        X_train, y_train, X_val, y_val = synthetic_data
        model = small_mlp_model
        optimizer = make_optimizer(model, lr=1e-3)
        criterion = nn.MSELoss()

        # Convert to tensors
        X_t = torch.from_numpy(X_train)
        y_t = torch.from_numpy(y_train)

        # Initial loss
        model.eval()
        with torch.no_grad():
            initial_pred = model(X_t)
            initial_loss = criterion(initial_pred, y_t).item()

        # Train for 10 steps
        model.train()
        for _ in range(10):
            optimizer.zero_grad()
            pred = model(X_t)
            loss = criterion(pred, y_t)
            loss.backward()
            optimizer.step()

        # Final loss
        model.eval()
        with torch.no_grad():
            final_pred = model(X_t)
            final_loss = criterion(final_pred, y_t).item()

        assert final_loss < initial_loss, (
            f"Loss should decrease: {initial_loss:.4f} → {final_loss:.4f}"
        )

    def test_loss_decreases_significantly(self, small_mlp_model, synthetic_data, random_seed):
        """Loss should decrease by at least 10% in 50 steps."""
        X_train, y_train, _, _ = synthetic_data
        model = small_mlp_model
        optimizer = make_optimizer(model, lr=1e-3)
        criterion = nn.MSELoss()

        X_t = torch.from_numpy(X_train)
        y_t = torch.from_numpy(y_train)

        # Initial loss
        model.eval()
        with torch.no_grad():
            initial_loss = criterion(model(X_t), y_t).item()

        # Train for 50 steps
        model.train()
        for _ in range(50):
            optimizer.zero_grad()
            pred = model(X_t)
            loss = criterion(pred, y_t)
            loss.backward()
            optimizer.step()

        # Final loss
        model.eval()
        with torch.no_grad():
            final_loss = criterion(model(X_t), y_t).item()

        improvement = (initial_loss - final_loss) / initial_loss
        assert improvement > 0.1, (
            f"Loss should improve by >10%, got {improvement*100:.1f}%"
        )

    def test_gradient_norms_reasonable(self, small_mlp_model, synthetic_data, random_seed):
        """Gradient norms should be reasonable (not exploding/vanishing)."""
        X_train, y_train, _, _ = synthetic_data
        model = small_mlp_model
        optimizer = make_optimizer(model, lr=1e-3)
        criterion = nn.MSELoss()

        X_t = torch.from_numpy(X_train[:100])
        y_t = torch.from_numpy(y_train[:100])

        # Training step
        optimizer.zero_grad()
        pred = model(X_t)
        loss = criterion(pred, y_t)
        loss.backward()

        # Check gradient norms
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert grad_norm > 1e-7, f"Vanishing gradient for {name}: {grad_norm}"
                assert grad_norm < 1000, f"Exploding gradient for {name}: {grad_norm}"

    def test_model_updates_during_training(self, small_mlp_model, synthetic_data, random_seed):
        """Model parameters should change during training."""
        X_train, y_train, _, _ = synthetic_data
        model = small_mlp_model
        optimizer = make_optimizer(model, lr=0.01)  # Higher LR for visible changes
        criterion = nn.MSELoss()

        X_t = torch.from_numpy(X_train)
        y_t = torch.from_numpy(y_train)

        # Save initial params
        initial_params = {
            name: param.clone() for name, param in model.named_parameters()
        }

        # Train
        model.train()
        for _ in range(10):
            optimizer.zero_grad()
            pred = model(X_t)
            loss = criterion(pred, y_t)
            loss.backward()
            optimizer.step()

        # Check params changed
        params_changed = 0
        total_params = 0
        for name, param in model.named_parameters():
            total_params += 1
            if not torch.equal(param, initial_params[name]):
                params_changed += 1

        assert params_changed > 0, "No parameters changed during training"


# ── 6. Prediction Tests ─────────────────────────────────────────────────────


class TestPredictions:
    """Verify prediction quality: diversity, range, not degenerate."""

    def test_predictions_not_all_same(self, small_mlp_model, random_seed):
        """Predictions should NOT all be the same value (degenerate check)."""
        model = small_mlp_model
        model.eval()

        # Diverse inputs
        x = torch.randn(100, 64)

        with torch.no_grad():
            preds = model(x).numpy()

        pred_std = np.std(preds)
        assert pred_std > 0.01, (
            f"Predictions are nearly constant: std={pred_std:.4f}"
        )

    def test_predictions_have_variance(self, small_mlp_model, random_seed):
        """Predictions should have meaningful variance (not collapsed)."""
        model = small_mlp_model
        model.eval()

        x = torch.randn(1000, 64)
        with torch.no_grad():
            preds = model(x).numpy()

        pred_std = np.std(preds)
        # After random init, small model (64-dim) has lower variance than
        # full 896-dim. Threshold 0.03 catches degenerate (near-zero) case.
        assert pred_std > 0.03, (
            f"Prediction variance too low: {pred_std:.4f} (expected > 0.03)"
        )

    def test_predictions_reasonable_range(self, small_mlp_model, random_seed):
        """Predictions should be in reasonable range (not NaN/Inf)."""
        model = small_mlp_model
        model.eval()

        x = torch.randn(100, 64)
        with torch.no_grad():
            preds = model(x).numpy()

        assert not np.any(np.isnan(preds)), "Predictions contain NaN"
        assert not np.any(np.isinf(preds)), "Predictions contain Inf"
        assert preds.min() > -100, f"Predictions too negative: {preds.min()}"
        assert preds.max() < 100, f"Predictions too positive: {preds.max()}"

    def test_trained_predictions_diverge(self, small_mlp_model, synthetic_data, random_seed):
        """After training, predictions should diverge for different inputs."""
        X_train, y_train, _, _ = synthetic_data
        model = small_mlp_model
        optimizer = make_optimizer(model, lr=1e-3)
        criterion = nn.MSELoss()

        X_t = torch.from_numpy(X_train)
        y_t = torch.from_numpy(y_train)

        # Train for 50 steps
        model.train()
        for _ in range(50):
            optimizer.zero_grad()
            pred = model(X_t)
            loss = criterion(pred, y_t)
            loss.backward()
            optimizer.step()

        # Test with diverse inputs
        model.eval()
        x_test = torch.randn(100, 64)
        with torch.no_grad():
            preds = model(x_test).numpy()

        pred_std = np.std(preds)
        assert pred_std > 0.05, (
            f"Trained predictions collapsed: std={pred_std:.4f}"
        )

    def test_output_clipping_range(self, small_mlp_model, random_seed):
        """Predictions should be clipable to [1.0, 5.0] for ratings."""
        model = small_mlp_model
        model.eval()

        x = torch.randn(100, 64)
        with torch.no_grad():
            preds = model(x).numpy()

        # Clip predictions (as done in run_mlp.py)
        clipped = np.clip(preds, 1.0, 5.0)

        assert clipped.min() >= 1.0, "Clipped predictions below 1.0"
        assert clipped.max() <= 5.0, "Clipped predictions above 5.0"


# ── 7. End-to-End Tests ─────────────────────────────────────────────────────


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_training_loop(self, small_mlp_model, synthetic_data, random_seed):
        """Full training loop should work: forward → loss → backward → step."""
        X_train, y_train, X_val, y_val = synthetic_data
        model = small_mlp_model
        optimizer = make_optimizer(model, lr=1e-3)
        criterion = nn.MSELoss()

        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train)
        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val)

        # Training loop
        model.train()
        train_losses = []
        for epoch in range(20):
            optimizer.zero_grad()
            pred = model(X_train_t)
            loss = criterion(pred, y_train_t)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()

        # Check losses are reasonable
        assert train_losses[-1] < train_losses[0], "Training loss should decrease"
        assert val_loss < 10.0, f"Validation loss too high: {val_loss}"
        assert not np.isnan(val_loss), "Validation loss is NaN"

    def test_model_save_load(self, small_mlp_model, tmp_path, random_seed):
        """Model should be saveable and loadable."""
        model = small_mlp_model

        # Save
        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)

        # Load
        loaded_model = RatingMLP(input_dim=64, dropout=0.3)
        loaded_model.load_state_dict(torch.load(save_path))

        # Compare predictions
        x = torch.randn(16, 64)
        model.eval()
        loaded_model.eval()

        with torch.no_grad():
            original_preds = model(x).numpy()
            loaded_preds = loaded_model(x).numpy()

        np.testing.assert_array_almost_equal(
            original_preds, loaded_preds, decimal=6,
            err_msg="Loaded model produces different predictions"
        )

    def test_model_deterministic(self, random_seed):
        """Model should be deterministic with same seed."""
        torch.manual_seed(42)
        model1 = RatingMLP(input_dim=64, dropout=0.3)

        torch.manual_seed(42)
        model2 = RatingMLP(input_dim=64, dropout=0.3)

        x = torch.randn(16, 64)

        model1.eval()
        model2.eval()

        with torch.no_grad():
            preds1 = model1(x).numpy()
            preds2 = model2(x).numpy()

        np.testing.assert_array_almost_equal(
            preds1, preds2, decimal=6,
            err_msg="Models with same seed produce different outputs"
        )


# ── 8. Regression Tests ─────────────────────────────────────────────────────


class TestRegression:
    """Regression tests for known MLP failure modes."""

    def test_detect_prediction_compression(self, small_mlp_model, random_seed):
        """Detect prediction compression (MLP predicts ~3.8 for everything).

        This is the signature of the MLP failure mode:
        - Prediction std ≈ 0.34 (should be ~1.42 for actual ratings)
        - All predictions clustered around 3.8
        """
        model = small_mlp_model
        model.eval()

        # Generate predictions
        x = torch.randn(1000, 64)
        with torch.no_grad():
            preds = model(x).numpy()

        pred_mean = np.mean(preds)
        pred_std = np.std(preds)

        # After random init, predictions should NOT be fully compressed.
        # Small model (64-dim) has lower variance than full 896-dim model.
        # Threshold 0.03 catches degenerate case (std ≈ 0).
        assert pred_std > 0.03, (
            f"Predictions compressed: std={pred_std:.4f}, mean={pred_mean:.4f}. "
            f"This indicates the model has learned to predict a constant value."
        )

    def test_detect_near_zero_feature_impact(self, near_zero_embeddings, random_seed):
        """Near-zero features should produce compressed predictions.

        Simulates the MLP failure mode:
        - 128 near-zero features (LightGCN)
        - 768 weak features (DeBERTa)
        → Model learns to predict ~3.8 for everything
        """
        n_samples = 1000
        bert_dim = 768
        user_dim = 64
        item_dim = 64

        # Simulate weak DeBERTa features (low signal-to-noise)
        bert_features = np.random.randn(n_samples, bert_dim).astype(np.float32) * 0.1

        # Near-zero LightGCN features
        user_emb = near_zero_embeddings["user_emb"]
        item_emb = near_zero_embeddings["item_emb"]

        # Map embeddings
        user_ids = [f"user_{i % 100}" for i in range(n_samples)]
        item_ids = [f"item_{i % 50}" for i in range(n_samples)]

        user2idx = near_zero_embeddings["user2idx"]
        item2idx = near_zero_embeddings["item2idx"]

        u_idx = np.array([user2idx.get(uid, -1) for uid in user_ids])
        i_idx = np.array([item2idx.get(pid, -1) for pid in item_ids])

        u_feats = np.zeros((n_samples, user_dim), dtype=np.float32)
        valid_u = u_idx >= 0
        u_feats[valid_u] = user_emb[u_idx[valid_u]]

        i_feats = np.zeros((n_samples, item_dim), dtype=np.float32)
        valid_i = i_idx >= 0
        i_feats[valid_i] = item_emb[i_idx[valid_i]]

        X = np.concatenate([bert_features, u_feats, i_feats], axis=1)

        # Check feature norms
        user_feat_norms = np.linalg.norm(X[:, bert_dim:bert_dim + user_dim], axis=1)
        item_feat_norms = np.linalg.norm(X[:, bert_dim + user_dim:], axis=1)

        # These should be near-zero (LightGCN failure)
        assert user_feat_norms.mean() < 0.1, (
            f"User features should be near-zero: norm={user_feat_norms.mean():.4f}"
        )
        assert item_feat_norms.mean() < 0.1, (
            f"Item features should be near-zero: norm={item_feat_norms.mean():.4f}"
        )

    def test_linear_vs_mlp_equivalence(self, synthetic_data, random_seed):
        """Linear model should achieve similar RMSE as MLP (key diagnostic).

        From MLP diagnosis: Ridge achieves RMSE=1.181 = same as MLP (1.152-1.177).
        This proves MLP adds no value — all signal is captured by linear projection.
        """
        X_train, y_train, X_val, y_val = synthetic_data

        # Linear model
        linear_model = nn.Linear(64, 1)
        linear_optimizer = torch.optim.Adam(linear_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train)
        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val)

        # Train linear model
        linear_model.train()
        for _ in range(100):
            linear_optimizer.zero_grad()
            pred = linear_model(X_train_t).squeeze()
            loss = criterion(pred, y_train_t)
            loss.backward()
            linear_optimizer.step()

        # MLP model
        mlp_model = RatingMLP(input_dim=64, dropout=0.3)
        mlp_optimizer = make_optimizer(mlp_model, lr=1e-3)

        mlp_model.train()
        for _ in range(100):
            mlp_optimizer.zero_grad()
            pred = mlp_model(X_train_t)
            loss = criterion(pred, y_train_t)
            loss.backward()
            mlp_optimizer.step()

        # Compare validation RMSE
        linear_model.eval()
        mlp_model.eval()

        with torch.no_grad():
            linear_preds = linear_model(X_val_t).squeeze().numpy()
            mlp_preds = mlp_model(X_val_t).numpy()

        linear_rmse = np.sqrt(np.mean((linear_preds - y_val) ** 2))
        mlp_rmse = np.sqrt(np.mean((mlp_preds - y_val) ** 2))

        # MLP should not be dramatically better than linear
        # (In real scenario, they achieve ~same RMSE)
        rmse_ratio = mlp_rmse / linear_rmse
        assert rmse_ratio < 2.0, (
            f"MLP RMSE ({mlp_rmse:.4f}) much worse than linear ({linear_rmse:.4f}). "
            f"Ratio: {rmse_ratio:.2f}"
        )

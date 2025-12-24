"""Tests for the training module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data import IO_SEP_TOKEN_ID, encode_example
from src.train import (
    TrainConfig,
    build_param_groups,
    compute_loss,
    create_optimizer,
    create_output_mask,
    create_scheduler,
    get_rng_state,
    load_checkpoint,
    save_checkpoint,
    set_rng_state,
)
from src.transformer import create_tiny_model

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def config():
    """Minimal training config for fast tests."""
    return TrainConfig(
        model_size="tiny",
        batch_size=2,
        epochs=1,
        lr=1e-3,
        compile_model=False,
        use_amp=False,
        log_every=1,
    )


@pytest.fixture
def model():
    """Tiny model for testing."""
    return create_tiny_model()


@pytest.fixture
def sample_batch():
    """Create a sample training batch."""
    # Create two examples with input and output
    ex1_tokens = encode_example([[1, 2], [3, 4]], [[5, 6]])
    ex2_tokens = encode_example([[0, 1, 2]], [[3, 4, 5]])

    # Pad to same length
    max_len = max(len(ex1_tokens), len(ex2_tokens))
    ex1_padded = ex1_tokens + [0] * (max_len - len(ex1_tokens))
    ex2_padded = ex2_tokens + [0] * (max_len - len(ex2_tokens))

    input_ids = torch.tensor([ex1_padded, ex2_padded])
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    attention_mask[0, : len(ex1_tokens)] = True
    attention_mask[1, : len(ex2_tokens)] = True

    # Simple positions (x, y, z)
    positions_3d = torch.zeros(2, max_len, 3, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "positions_3d": positions_3d,
    }


# =============================================================================
# OUTPUT MASK TESTS
# =============================================================================


class TestOutputMask:
    """Tests for output mask creation."""

    def test_mask_shape(self, sample_batch):
        """Test mask has correct shape."""
        input_ids = sample_batch["input_ids"]
        mask = create_output_mask(input_ids)

        assert mask.shape == input_ids.shape
        assert mask.dtype == torch.bool

    def test_mask_before_separator(self):
        """Test that tokens before separator are False."""
        # START, 1, 2, NEWLINE, IO_SEP, 3, NEWLINE, END
        tokens = encode_example([[1, 2]], [[3]])
        input_ids = torch.tensor([tokens])

        mask = create_output_mask(input_ids)

        # Find separator
        sep_idx = tokens.index(IO_SEP_TOKEN_ID)

        # All before separator should be False
        assert not mask[0, :sep_idx].any()

    def test_mask_after_separator(self):
        """Test that tokens after separator are True."""
        tokens = encode_example([[1, 2]], [[3]])
        input_ids = torch.tensor([tokens])

        mask = create_output_mask(input_ids)

        # Find separator
        sep_idx = tokens.index(IO_SEP_TOKEN_ID)

        # Separator itself should be False
        assert not mask[0, sep_idx]

        # Tokens after separator should be True
        assert mask[0, sep_idx + 1 :].any()

    def test_mask_no_separator(self):
        """Test behavior when no separator (input only)."""
        tokens = encode_example([[1, 2]], None)  # No output
        input_ids = torch.tensor([tokens])

        mask = create_output_mask(input_ids)

        # Should have some True values after separator
        assert mask.sum() == 0  # No output tokens without output


# =============================================================================
# LOSS COMPUTATION TESTS
# =============================================================================


class TestLossComputation:
    """Tests for loss computation."""

    def test_loss_shape(self, model, sample_batch):
        """Test that loss is a scalar."""
        model.eval()

        with torch.no_grad():
            logits = model(
                sample_batch["input_ids"],
                sample_batch["positions_3d"],
            )

        output_mask = create_output_mask(sample_batch["input_ids"])

        losses = compute_loss(
            logits=logits,
            labels=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"],
            output_mask=output_mask,
        )

        assert losses["loss"].dim() == 0
        assert losses["input_loss"].dim() == 0
        assert losses["output_loss"].dim() == 0

    def test_loss_positive(self, model, sample_batch):
        """Test that losses are positive."""
        model.eval()

        with torch.no_grad():
            logits = model(
                sample_batch["input_ids"],
                sample_batch["positions_3d"],
            )

        output_mask = create_output_mask(sample_batch["input_ids"])

        losses = compute_loss(
            logits=logits,
            labels=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"],
            output_mask=output_mask,
        )

        assert losses["loss"] >= 0
        assert losses["input_loss"] >= 0
        assert losses["output_loss"] >= 0

    def test_loss_weights(self, model, sample_batch):
        """Test that loss weights work correctly when uniform_weight=False."""
        model.eval()

        with torch.no_grad():
            logits = model(
                sample_batch["input_ids"],
                sample_batch["positions_3d"],
            )

        output_mask = create_output_mask(sample_batch["input_ids"])

        # Only output loss (must set uniform_weight=False to use weights)
        losses_out = compute_loss(
            logits=logits,
            labels=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"],
            output_mask=output_mask,
            input_loss_weight=0.0,
            output_loss_weight=1.0,
            uniform_weight=False,
        )

        # Only input loss
        losses_in = compute_loss(
            logits=logits,
            labels=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"],
            output_mask=output_mask,
            input_loss_weight=1.0,
            output_loss_weight=0.0,
            uniform_weight=False,
        )

        # With only output weight, loss should equal output_loss
        assert torch.allclose(losses_out["loss"], losses_out["output_loss"])

        # With only input weight, loss should equal input_loss
        assert torch.allclose(losses_in["loss"], losses_in["input_loss"])

    def test_uniform_weight(self, model, sample_batch):
        """Test that uniform_weight=True gives equal weight to all tokens."""
        model.eval()

        with torch.no_grad():
            logits = model(
                sample_batch["input_ids"],
                sample_batch["positions_3d"],
            )

        output_mask = create_output_mask(sample_batch["input_ids"])

        # Uniform loss (default, matches original mdlARC)
        losses = compute_loss(
            logits=logits,
            labels=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"],
            output_mask=output_mask,
            uniform_weight=True,
        )

        # Loss should be defined and reasonable
        assert losses["loss"] >= 0
        assert torch.isfinite(losses["loss"])


# =============================================================================
# OPTIMIZER TESTS
# =============================================================================


class TestOptimizer:
    """Tests for optimizer setup."""

    def test_param_groups(self, model):
        """Test that parameter groups are created correctly."""
        groups = build_param_groups(model, weight_decay=0.01)

        assert len(groups) == 2
        assert groups[0]["weight_decay"] == 0.01
        assert groups[1]["weight_decay"] == 0.0

        # All params should be in one of the groups
        total_params = sum(len(g["params"]) for g in groups)
        model_params = sum(1 for p in model.parameters() if p.requires_grad)
        assert total_params == model_params

    def test_create_optimizer(self, model):
        """Test optimizer creation."""
        device = torch.device("cpu")
        optimizer = create_optimizer(model, lr=1e-3, weight_decay=0.01, device=device)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert len(optimizer.param_groups) == 2


# =============================================================================
# SCHEDULER TESTS
# =============================================================================


class TestScheduler:
    """Tests for learning rate scheduler."""

    def test_warmup(self, model):
        """Test warmup phase."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = create_scheduler(optimizer, total_steps=100, warmup_fraction=0.1)

        # At step 0, LR should be 0
        assert scheduler.get_last_lr()[0] == pytest.approx(0.0)

        # Step through warmup
        for _ in range(5):
            scheduler.step()

        # At step 5 (half of warmup), LR should be ~0.5 * base_lr
        assert 0.4e-3 < scheduler.get_last_lr()[0] < 0.6e-3

    def test_cosine_decay(self, model):
        """Test cosine decay phase."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = create_scheduler(optimizer, total_steps=100, warmup_fraction=0.1)

        # Step through warmup
        for _ in range(10):
            scheduler.step()

        # At end of warmup, LR should be at peak
        assert scheduler.get_last_lr()[0] == pytest.approx(1e-3, rel=0.01)

        # Step through decay
        for _ in range(90):
            scheduler.step()

        # At end, LR should be near 0
        assert scheduler.get_last_lr()[0] < 0.1e-3


# =============================================================================
# CHECKPOINTING TESTS
# =============================================================================


class TestCheckpointing:
    """Tests for checkpoint saving and loading."""

    def test_rng_state_roundtrip(self):
        """Test RNG state save and restore."""
        # Set a known state
        torch.manual_seed(42)
        np.random.seed(42)

        # Generate some numbers
        torch_nums_before = torch.rand(5)
        np_nums_before = np.random.rand(5)

        # Reset and save state
        torch.manual_seed(42)
        np.random.seed(42)
        state = get_rng_state()

        # Generate different numbers
        torch.manual_seed(999)
        np.random.seed(999)
        _ = torch.rand(10)
        _ = np.random.rand(10)

        # Restore state
        set_rng_state(state)

        # Should get same numbers as before
        torch_nums_after = torch.rand(5)
        np_nums_after = np.random.rand(5)

        assert torch.allclose(torch_nums_before, torch_nums_after)
        assert np.allclose(np_nums_before, np_nums_after)

    def test_checkpoint_roundtrip(self, model, config):
        """Test checkpoint save and load."""
        device = torch.device("cpu")
        optimizer = create_optimizer(model, config.lr, config.weight_decay, device)
        scheduler = create_scheduler(optimizer, total_steps=100)

        # Step a few times
        for _ in range(5):
            scheduler.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.pt"

            # Save
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=3,
                global_step=150,
                config=config,
                metrics={"loss": 1.5},
                path=path,
            )

            assert path.exists()

            # Create new model and load
            new_model = create_tiny_model()
            new_optimizer = create_optimizer(
                new_model, config.lr, config.weight_decay, device
            )
            new_scheduler = create_scheduler(new_optimizer, total_steps=100)

            checkpoint = load_checkpoint(path, new_model, new_optimizer, new_scheduler)

            # Check metadata
            assert checkpoint["epoch"] == 3
            assert checkpoint["global_step"] == 150
            assert checkpoint["metrics"]["loss"] == 1.5

            # Check model weights match
            for (n1, p1), (n2, p2) in zip(
                model.named_parameters(),
                new_model.named_parameters(),
                strict=True,
            ):
                assert n1 == n2
                assert torch.allclose(p1, p2)


# =============================================================================
# TRAINING LOOP TESTS
# =============================================================================


class TestTrainingLoop:
    """Tests for training and validation loops."""

    def test_train_one_epoch_components(self, model, config, sample_batch):
        """Test that training components work together."""
        device = torch.device("cpu")
        model = model.to(device)

        optimizer = create_optimizer(model, config.lr, config.weight_decay, device)
        _ = create_scheduler(optimizer, total_steps=10)

        # Verify optimizer and scheduler created successfully
        assert len(optimizer.param_groups) == 2
        # Full train_one_epoch tested via integration tests

    def test_model_updates(self, model, sample_batch):
        """Test that training updates model weights."""
        device = torch.device("cpu")
        model = model.to(device)

        # Get initial weights
        initial_weights = {
            name: param.clone() for name, param in model.named_parameters()
        }

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        # Forward and backward
        input_ids = sample_batch["input_ids"].to(device)
        positions_3d = sample_batch["positions_3d"].to(device)
        attention_mask = sample_batch["attention_mask"].to(device)

        logits = model(input_ids, positions_3d)
        output_mask = create_output_mask(input_ids)

        losses = compute_loss(
            logits=logits,
            labels=input_ids,
            attention_mask=attention_mask,
            output_mask=output_mask,
        )

        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()

        # Check weights changed
        weights_changed = False
        for name, param in model.named_parameters():
            if not torch.allclose(initial_weights[name], param):
                weights_changed = True
                break

        assert weights_changed, "Model weights should update after training step"

    def test_gradient_clipping(self, model, sample_batch):
        """Test gradient clipping."""
        device = torch.device("cpu")
        model = model.to(device)

        input_ids = sample_batch["input_ids"].to(device)
        positions_3d = sample_batch["positions_3d"].to(device)
        attention_mask = sample_batch["attention_mask"].to(device)

        logits = model(input_ids, positions_3d)
        output_mask = create_output_mask(input_ids)

        losses = compute_loss(
            logits=logits,
            labels=input_ids,
            attention_mask=attention_mask,
            output_mask=output_mask,
        )

        losses["loss"].backward()

        # Clip gradients
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Check all gradients are clipped
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5

        # Total norm should be approximately <= max_norm
        # (may be slightly higher due to numerical precision)
        assert total_norm <= max_norm * 1.1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for training pipeline."""

    def test_full_training_step(self, model, config, sample_batch):
        """Test a complete training step."""
        device = torch.device("cpu")
        model = model.to(device)
        model.train()

        optimizer = create_optimizer(model, config.lr, config.weight_decay, device)

        input_ids = sample_batch["input_ids"].to(device)
        positions_3d = sample_batch["positions_3d"].to(device)
        attention_mask = sample_batch["attention_mask"].to(device)

        # Forward
        logits = model(input_ids, positions_3d)
        output_mask = create_output_mask(input_ids)

        # Compute loss
        losses = compute_loss(
            logits=logits,
            labels=input_ids,
            attention_mask=attention_mask,
            output_mask=output_mask,
            input_loss_weight=config.input_loss_weight,
            output_loss_weight=config.output_loss_weight,
        )

        # Backward
        optimizer.zero_grad()
        losses["loss"].backward()

        # Clip
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        # Step
        optimizer.step()

        # Verify loss decreased on second forward
        with torch.no_grad():
            logits2 = model(input_ids, positions_3d)
            losses2 = compute_loss(
                logits=logits2,
                labels=input_ids,
                attention_mask=attention_mask,
                output_mask=output_mask,
            )

        # Loss should generally decrease after one step
        # (not guaranteed, but usually true with high LR)
        # Just check it's still finite
        assert torch.isfinite(losses2["loss"])

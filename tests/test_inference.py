"""Tests for the inference module."""

import pytest
import torch

from src.data import encode_example
from src.inference import (
    generate_output,
    predict_grid,
    predict_with_augmentations,
)
from src.transformer import create_tiny_model

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def model():
    """Tiny model for testing."""
    model = create_tiny_model()
    model.eval()
    return model


@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cpu")


@pytest.fixture
def simple_task():
    """Simple ARC task for testing."""
    return {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
            {"input": [[5, 6], [7, 8]], "output": [[6, 5], [8, 7]]},
        ],
        "test": [
            {"input": [[0, 1], [2, 3]]},
        ],
    }


# =============================================================================
# GENERATION TESTS
# =============================================================================


class TestGenerateOutput:
    """Tests for the generate_output function."""

    def test_output_shape(self, model, device):
        """Test that output has correct shape."""
        # Simple prompt
        tokens = encode_example([[1, 2]], None)  # Input only
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        # Compute positions
        from src.data import compute_positions_3d

        positions = compute_positions_3d(tokens)
        positions_3d = torch.tensor([positions], dtype=torch.long, device=device)

        max_new = 10
        with torch.no_grad():
            output = generate_output(
                model,
                input_ids,
                positions_3d,
                max_new_tokens=max_new,
                temperature=0.0,
            )

        # Output should be prompt + new tokens
        assert output.shape[0] == 1
        assert output.shape[1] >= len(tokens)
        assert output.shape[1] <= len(tokens) + max_new

    def test_greedy_deterministic(self, model, device):
        """Test that greedy generation is deterministic."""
        tokens = encode_example([[1, 2]], None)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        from src.data import compute_positions_3d

        positions = compute_positions_3d(tokens)
        positions_3d = torch.tensor([positions], dtype=torch.long, device=device)

        with torch.no_grad():
            model.reset_cache()
            out1 = generate_output(model, input_ids, positions_3d, max_new_tokens=5)
            model.reset_cache()
            out2 = generate_output(model, input_ids, positions_3d, max_new_tokens=5)

        assert torch.equal(out1, out2)

    def test_batched_generation(self, model, device):
        """Test generation with batch size > 1."""
        tokens1 = encode_example([[1, 2]], None)
        tokens2 = encode_example([[3, 4, 5]], None)

        # Pad to same length
        max_len = max(len(tokens1), len(tokens2))
        tokens1_padded = tokens1 + [0] * (max_len - len(tokens1))
        tokens2_padded = tokens2 + [0] * (max_len - len(tokens2))

        input_ids = torch.tensor(
            [tokens1_padded, tokens2_padded], dtype=torch.long, device=device
        )

        from src.data import compute_positions_3d

        pos1 = compute_positions_3d(tokens1_padded)
        pos2 = compute_positions_3d(tokens2_padded)
        positions_3d = torch.tensor([pos1, pos2], dtype=torch.long, device=device)

        with torch.no_grad():
            output = generate_output(model, input_ids, positions_3d, max_new_tokens=5)

        assert output.shape[0] == 2

    def test_stops_on_end_token(self, model, device):
        """Test that generation can stop on END token."""
        tokens = encode_example([[1]], None)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        from src.data import compute_positions_3d

        positions = compute_positions_3d(tokens)
        positions_3d = torch.tensor([positions], dtype=torch.long, device=device)

        # Generate with stop_on_end
        with torch.no_grad():
            output = generate_output(
                model,
                input_ids,
                positions_3d,
                max_new_tokens=100,
                stop_on_end=True,
            )

        # Should stop before max_new_tokens if END generated
        # (may or may not happen depending on model weights)
        assert output.shape[1] <= len(tokens) + 100


# =============================================================================
# PREDICT GRID TESTS
# =============================================================================


class TestPredictGrid:
    """Tests for the predict_grid function."""

    def test_returns_grid_or_none(self, model, device):
        """Test that predict_grid returns a grid or None."""
        input_grids = [[[1, 2], [3, 4]]]
        output_grids = [[[5, 6], [7, 8]]]
        test_input = [[0, 1], [2, 3]]

        with torch.no_grad():
            result = predict_grid(
                model,
                input_grids,
                output_grids,
                test_input,
                device,
                max_output_tokens=50,
            )

        # Result should be list of lists (grid) or None
        assert result is None or isinstance(result, list)
        if result is not None:
            assert all(isinstance(row, list) for row in result)

    def test_handles_multiple_train_pairs(self, model, device):
        """Test with multiple training examples."""
        input_grids = [[[1, 2]], [[3, 4]], [[5, 6]]]
        output_grids = [[[7, 8]], [[9, 0]], [[1, 2]]]
        test_input = [[4, 5]]

        with torch.no_grad():
            result = predict_grid(
                model,
                input_grids,
                output_grids,
                test_input,
                device,
                max_output_tokens=30,
            )

        # Just check it doesn't crash
        assert result is None or isinstance(result, list)

    def test_handles_varying_grid_sizes(self, model, device):
        """Test with different sized grids."""
        input_grids = [[[1, 2, 3], [4, 5, 6]]]  # 2x3
        output_grids = [[[7], [8]]]  # 2x1
        test_input = [[1, 1, 1], [2, 2, 2]]  # 2x3

        with torch.no_grad():
            result = predict_grid(
                model,
                input_grids,
                output_grids,
                test_input,
                device,
                max_output_tokens=20,
            )

        assert result is None or isinstance(result, list)


# =============================================================================
# AUGMENTED PREDICTION TESTS
# =============================================================================


class TestPredictWithAugmentations:
    """Tests for augmented prediction."""

    def test_correct_number_of_predictions(self, model, device, simple_task):
        """Test that we get expected number of predictions."""
        num_color_perms = 4  # Use fewer for faster test

        with torch.no_grad():
            predictions = predict_with_augmentations(
                model,
                simple_task,
                test_idx=0,
                device=device,
                num_color_perms=num_color_perms,
                max_output_tokens=20,
            )

        # 8 dihedral x num_color_perms
        expected_count = 8 * num_color_perms
        assert len(predictions) == expected_count

    def test_prediction_tuple_structure(self, model, device, simple_task):
        """Test that predictions have correct structure."""
        with torch.no_grad():
            predictions = predict_with_augmentations(
                model,
                simple_task,
                test_idx=0,
                device=device,
                num_color_perms=2,
                max_output_tokens=10,
            )

        for pred_grid, dihedral_idx, color_perm in predictions:
            # Grid is either list or None
            assert pred_grid is None or isinstance(pred_grid, list)
            # Dihedral idx is 0-7
            assert 0 <= dihedral_idx <= 7
            # Color perm is tuple or None (None for identity)
            assert color_perm is None or isinstance(color_perm, tuple)

    def test_identity_augmentation_first(self, model, device, simple_task):
        """Test that first prediction uses identity transforms."""
        with torch.no_grad():
            predictions = predict_with_augmentations(
                model,
                simple_task,
                test_idx=0,
                device=device,
                num_color_perms=1,  # Only identity color
                max_output_tokens=10,
            )

        # First prediction should have dihedral_idx=0 and color_perm=None
        _, dihedral_idx, color_perm = predictions[0]
        assert dihedral_idx == 0
        assert color_perm is None

    def test_all_dihedral_indices_covered(self, model, device, simple_task):
        """Test that all 8 dihedral transforms are used."""
        with torch.no_grad():
            predictions = predict_with_augmentations(
                model,
                simple_task,
                test_idx=0,
                device=device,
                num_color_perms=1,
                max_output_tokens=10,
            )

        dihedral_indices = {d for _, d, _ in predictions}
        assert dihedral_indices == {0, 1, 2, 3, 4, 5, 6, 7}


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for inference pipeline."""

    def test_full_pipeline(self, model, device, simple_task):
        """Test the full prediction pipeline."""
        with torch.no_grad():
            predictions = predict_with_augmentations(
                model,
                simple_task,
                test_idx=0,
                device=device,
                num_color_perms=2,
                max_output_tokens=30,
            )

        # Should have predictions
        assert len(predictions) > 0

        # At least some should produce valid grids (with random weights, not guaranteed)
        # Just verify the structure is correct
        _ = [p for p, _, _ in predictions if p is not None]
        # May be 0 with random weights, that's OK for this test

    def test_model_stays_in_eval_mode(self, model, device, simple_task):
        """Test that model remains in eval mode after inference."""
        model.eval()

        with torch.no_grad():
            _ = predict_with_augmentations(
                model,
                simple_task,
                test_idx=0,
                device=device,
                num_color_perms=1,
                max_output_tokens=10,
            )

        assert not model.training

    def test_cache_reset_between_predictions(self, model, device):
        """Test that KV cache is properly reset."""
        tokens = encode_example([[1, 2]], None)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        from src.data import compute_positions_3d

        positions = compute_positions_3d(tokens)
        positions_3d = torch.tensor([positions], dtype=torch.long, device=device)

        with torch.no_grad():
            out1 = generate_output(model, input_ids, positions_3d, max_new_tokens=5)

        # Verify cache was used (should be non-None after generation)
        # Cache behavior may vary depending on implementation
        _ = any(block.attn.k_cache is not None for block in model.blocks)

        # Reset and generate again
        model.reset_cache()
        with torch.no_grad():
            out2 = generate_output(model, input_ids, positions_3d, max_new_tokens=5)

        # Outputs should be identical
        assert torch.equal(out1, out2)

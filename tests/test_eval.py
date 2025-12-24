"""Tests for the evaluation module."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from src.eval import (
    evaluate_dataset,
    evaluate_task,
    generate_submission,
    grids_equal,
    load_model_from_checkpoint,
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
        ],
        "test": [
            {"input": [[5, 6], [7, 8]]},
        ],
    }


@pytest.fixture
def simple_challenges():
    """Dictionary of simple challenges."""
    return {
        "task1": {
            "train": [{"input": [[1]], "output": [[2]]}],
            "test": [{"input": [[3]]}],
        },
        "task2": {
            "train": [{"input": [[4]], "output": [[5]]}],
            "test": [{"input": [[6]]}],
        },
    }


@pytest.fixture
def simple_solutions():
    """Solutions for simple challenges."""
    return {
        "task1": [[[4]]],  # Solution for task1's test
        "task2": [[[7]]],  # Solution for task2's test
    }


# =============================================================================
# GRIDS EQUAL TESTS
# =============================================================================


class TestGridsEqual:
    """Tests for grid equality check."""

    def test_equal_grids(self):
        """Test that identical grids are equal."""
        g1 = [[1, 2], [3, 4]]
        g2 = [[1, 2], [3, 4]]
        assert grids_equal(g1, g2)

    def test_different_values(self):
        """Test that grids with different values are not equal."""
        g1 = [[1, 2], [3, 4]]
        g2 = [[1, 2], [3, 5]]
        assert not grids_equal(g1, g2)

    def test_different_rows(self):
        """Test that grids with different row counts are not equal."""
        g1 = [[1, 2], [3, 4]]
        g2 = [[1, 2]]
        assert not grids_equal(g1, g2)

    def test_different_cols(self):
        """Test that grids with different column counts are not equal."""
        g1 = [[1, 2], [3, 4]]
        g2 = [[1, 2, 3], [4, 5, 6]]
        assert not grids_equal(g1, g2)

    def test_empty_grids(self):
        """Test that empty grids are equal."""
        assert grids_equal([], [])

    def test_single_cell(self):
        """Test single cell grids."""
        assert grids_equal([[1]], [[1]])
        assert not grids_equal([[1]], [[2]])


# =============================================================================
# EVALUATE TASK TESTS
# =============================================================================


class TestEvaluateTask:
    """Tests for single task evaluation."""

    def test_returns_correct_structure(self, model, device, simple_task):
        """Test that evaluate_task returns correct structure."""
        result = evaluate_task(
            model,
            simple_task,
            "test_task",
            device,
            num_color_perms=1,  # Fast
            max_output_tokens=20,
        )

        assert "task_id" in result
        assert result["task_id"] == "test_task"
        assert "test_outputs" in result
        assert "correct_at_1" in result
        assert "correct_at_2" in result
        assert "fractional_at_1" in result
        assert "fractional_at_2" in result
        assert "voting_stats" in result

    def test_correct_number_of_outputs(self, model, device, simple_task):
        """Test that we get outputs for each test example."""
        result = evaluate_task(
            model,
            simple_task,
            "test_task",
            device,
            num_color_perms=1,
            max_output_tokens=20,
        )

        num_tests = len(simple_task["test"])
        assert len(result["test_outputs"]) == num_tests
        assert len(result["correct_at_1"]) == num_tests
        assert len(result["correct_at_2"]) == num_tests

    def test_with_solutions(self, model, device, simple_task):
        """Test evaluation with solutions provided."""
        solutions = [[[9, 9], [9, 9]]]  # Wrong answer

        result = evaluate_task(
            model,
            simple_task,
            "test_task",
            device,
            solutions=solutions,
            num_color_perms=1,
            max_output_tokens=20,
        )

        # Correctness should be determined
        assert result["correct_at_1"][0] is not None
        assert result["correct_at_2"][0] is not None

    def test_without_solutions(self, model, device, simple_task):
        """Test evaluation without solutions."""
        result = evaluate_task(
            model,
            simple_task,
            "test_task",
            device,
            solutions=None,
            num_color_perms=1,
            max_output_tokens=20,
        )

        # Correctness should be None
        assert result["correct_at_1"][0] is None
        assert result["correct_at_2"][0] is None


# =============================================================================
# EVALUATE DATASET TESTS
# =============================================================================


class TestEvaluateDataset:
    """Tests for dataset evaluation."""

    def test_returns_correct_structure(self, model, device, simple_challenges):
        """Test that evaluate_dataset returns correct structure."""
        result = evaluate_dataset(
            model,
            simple_challenges,
            device,
            num_color_perms=1,
            max_output_tokens=10,
            show_progress=False,
        )

        assert "total_tasks" in result
        assert "correct_at_1" in result
        assert "correct_at_2" in result
        assert "accuracy_at_1" in result
        assert "accuracy_at_2" in result
        assert "arc_score_at_1" in result
        assert "arc_score_at_2" in result
        assert "per_task_results" in result

    def test_counts_all_tasks(self, model, device, simple_challenges):
        """Test that all tasks are counted."""
        result = evaluate_dataset(
            model,
            simple_challenges,
            device,
            num_color_perms=1,
            max_output_tokens=10,
            show_progress=False,
        )

        assert result["total_tasks"] == len(simple_challenges)
        assert len(result["per_task_results"]) == len(simple_challenges)

    def test_with_solutions(self, model, device, simple_challenges, simple_solutions):
        """Test dataset evaluation with solutions."""
        result = evaluate_dataset(
            model,
            simple_challenges,
            device,
            solutions=simple_solutions,
            num_color_perms=1,
            max_output_tokens=10,
            show_progress=False,
        )

        # Accuracy should be between 0 and 1
        assert 0.0 <= result["accuracy_at_1"] <= 1.0
        assert 0.0 <= result["accuracy_at_2"] <= 1.0


# =============================================================================
# GENERATE SUBMISSION TESTS
# =============================================================================


class TestGenerateSubmission:
    """Tests for submission generation."""

    def test_creates_file(self, model, device, simple_challenges):
        """Test that submission file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "submission.json"

            generate_submission(
                model,
                simple_challenges,
                output_path,
                device,
                num_color_perms=1,
                max_output_tokens=10,
                show_progress=False,
            )

            assert output_path.exists()

    def test_valid_json(self, model, device, simple_challenges):
        """Test that submission is valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "submission.json"

            generate_submission(
                model,
                simple_challenges,
                output_path,
                device,
                num_color_perms=1,
                max_output_tokens=10,
                show_progress=False,
            )

            with open(output_path) as f:
                submission = json.load(f)

            assert isinstance(submission, dict)

    def test_correct_structure(self, model, device, simple_challenges):
        """Test submission has correct structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "submission.json"

            generate_submission(
                model,
                simple_challenges,
                output_path,
                device,
                num_color_perms=1,
                max_output_tokens=10,
                show_progress=False,
            )

            with open(output_path) as f:
                submission = json.load(f)

            # Should have entry for each task
            assert set(submission.keys()) == set(simple_challenges.keys())

            # Each task should have list of test entries
            for task_id, entries in submission.items():
                task = simple_challenges[task_id]
                assert len(entries) == len(task["test"])

                for entry in entries:
                    assert "attempt_1" in entry
                    assert "attempt_2" in entry
                    assert isinstance(entry["attempt_1"], list)
                    assert isinstance(entry["attempt_2"], list)


# =============================================================================
# MODEL LOADING TESTS
# =============================================================================


class TestLoadModel:
    """Tests for model loading from checkpoints."""

    def test_load_with_state_dict(self, model, device):
        """Test loading model with model_state_dict key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint.pt"

            # Save checkpoint with model config so loader knows dimensions
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": 14,
                    "d_model": 256,
                    "n_heads": 4,
                    "n_layers": 4,
                    "d_ff": 704,
                },
                },
                ckpt_path,
            )

            # Load
            loaded = load_model_from_checkpoint(ckpt_path, device)

            assert loaded is not None
            assert loaded.num_parameters == model.num_parameters

    def test_load_with_train_config(self, model, device):
        """Test loading model with TrainConfig."""
        from src.train import TrainConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint.pt"

            config = TrainConfig(model_size="tiny")

            # Need to create a model matching the tiny config
            from src.transformer import create_tiny_model

            tiny_model = create_tiny_model()

            torch.save(
                {
                    "model_state_dict": tiny_model.state_dict(),
                    "config": config,
                },
                ckpt_path,
            )

            loaded = load_model_from_checkpoint(ckpt_path, device)
            assert loaded is not None
            assert loaded.num_parameters == tiny_model.num_parameters


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for full evaluation pipeline."""

    def test_full_pipeline(self, model, device, simple_challenges, simple_solutions):
        """Test the complete evaluation pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "submission.json"

            # Generate submission
            generate_submission(
                model,
                simple_challenges,
                output_path,
                device,
                num_color_perms=1,
                max_output_tokens=10,
                show_progress=False,
            )

            # Evaluate
            results = evaluate_dataset(
                model,
                simple_challenges,
                device,
                solutions=simple_solutions,
                num_color_perms=1,
                max_output_tokens=10,
                show_progress=False,
            )

            # Verify results
            assert output_path.exists()
            assert results["total_tasks"] == 2
            assert 0.0 <= results["accuracy_at_1"] <= 1.0

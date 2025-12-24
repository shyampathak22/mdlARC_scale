"""Tests for the ARC data pipeline."""


import numpy as np
import pytest
import torch

from src.data import (
    # Dihedral
    DIHEDRAL_TRANSFORMS,
    END_TOKEN_ID,
    IO_SEP_TOKEN_ID,
    NEWLINE_TOKEN_ID,
    START_TOKEN_ID,
    # Tokenization
    VOCAB_SIZE,
    # Dataset
    ARCDataset,
    LengthBucketSampler,
    apply_color_perm_to_tokens,
    apply_dihedral_transform,
    collate_fn,
    # Positions
    compute_positions_3d,
    create_dataloader,
    decode_output_grid,
    encode_example,
    flip_h,
    flip_v,
    # Color augmentation
    generate_color_permutations,
    grid_to_tokens,
    identity,
    inverse_color_perm,
    inverse_dihedral,
    rot90,
    rot180,
    rot270,
    tokens_to_grid,
)

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def simple_grid():
    """A simple 2x3 grid."""
    return [[1, 2, 3], [4, 5, 6]]


@pytest.fixture
def square_grid():
    """A 2x2 grid for testing transforms."""
    return [[1, 2], [3, 4]]


@pytest.fixture
def sample_challenges(tmp_path):
    """Create a temporary challenges JSON file."""
    import json

    challenges = {
        "task_001": {
            "train": [
                {"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 8]]},
                {"input": [[0, 1]], "output": [[1, 0]]},
            ],
            "test": [
                {"input": [[2, 3], [4, 5]]},  # No output - this is test data
            ],
        },
        "task_002": {
            "train": [
                {"input": [[1]], "output": [[2]]},
            ],
            "test": [
                {"input": [[3]]},
            ],
        },
    }

    path = tmp_path / "challenges.json"
    with open(path, "w") as f:
        json.dump(challenges, f)

    return path


# =============================================================================
# TOKENIZATION TESTS
# =============================================================================

class TestTokenization:
    """Tests for grid tokenization."""

    def test_vocab_size(self):
        """Test vocabulary size constant."""
        assert VOCAB_SIZE == 14  # 0-9 colors + 4 special tokens

    def test_special_tokens(self):
        """Test special token IDs."""
        assert START_TOKEN_ID == 10
        assert NEWLINE_TOKEN_ID == 11
        assert IO_SEP_TOKEN_ID == 12
        assert END_TOKEN_ID == 13

    def test_grid_to_tokens(self, simple_grid):
        """Test grid to token conversion."""
        tokens = grid_to_tokens(simple_grid)

        # Should be: 1, 2, 3, NEWLINE, 4, 5, 6, NEWLINE
        assert tokens == [1, 2, 3, NEWLINE_TOKEN_ID, 4, 5, 6, NEWLINE_TOKEN_ID]

    def test_tokens_to_grid(self, simple_grid):
        """Test token to grid conversion."""
        tokens = grid_to_tokens(simple_grid)
        recovered = tokens_to_grid(tokens)

        assert recovered == simple_grid

    def test_encode_example_with_output(self, simple_grid):
        """Test encoding a complete example."""
        output = [[7, 8], [9, 0]]
        tokens = encode_example(simple_grid, output)

        assert tokens[0] == START_TOKEN_ID
        assert IO_SEP_TOKEN_ID in tokens
        assert tokens[-1] == END_TOKEN_ID

    def test_encode_example_without_output(self, simple_grid):
        """Test encoding input only (test data)."""
        tokens = encode_example(simple_grid, None)

        assert tokens[0] == START_TOKEN_ID
        assert tokens[-1] == IO_SEP_TOKEN_ID
        assert END_TOKEN_ID not in tokens

    def test_decode_output_grid(self):
        """Test extracting output grid from tokens."""
        input_grid = [[1, 2], [3, 4]]
        output_grid = [[5, 6], [7, 8]]

        tokens = encode_example(input_grid, output_grid)
        decoded = decode_output_grid(tokens)

        assert decoded == output_grid

    def test_decode_no_output(self, simple_grid):
        """Test decoding when no output present."""
        tokens = encode_example(simple_grid, None)
        decoded = decode_output_grid(tokens)

        assert decoded is None

    def test_roundtrip_various_sizes(self):
        """Test roundtrip for various grid sizes."""
        test_cases = [
            [[0]],  # 1x1
            [[1, 2, 3, 4, 5]],  # 1x5
            [[i] for i in range(5)],  # 5x1
            [[i + j for j in range(10)] for i in range(10)],  # 10x10
        ]

        for grid in test_cases:
            # Normalize values to 0-9
            grid = [[v % 10 for v in row] for row in grid]
            tokens = grid_to_tokens(grid)
            recovered = tokens_to_grid(tokens)
            assert recovered == grid, f"Failed for grid of shape {len(grid)}x{len(grid[0])}"


# =============================================================================
# POSITION TESTS
# =============================================================================

class TestPositions:
    """Tests for 3D position computation."""

    def test_position_shape(self, simple_grid):
        """Test position array shape."""
        tokens = encode_example(simple_grid, [[0]])
        positions = compute_positions_3d(tokens)

        assert positions.shape == (len(tokens), 3)
        assert positions.dtype == np.int64

    def test_x_coordinate(self):
        """Test x-coordinate increments across columns (0-indexed)."""
        grid = [[1, 2, 3, 4, 5]]  # Single row
        tokens = encode_example(grid, None)
        positions = compute_positions_3d(tokens)

        # After START token, x should increment: 0, 1, 2, 3, 4
        # tokens: START, 1, 2, 3, 4, 5, NEWLINE, IO_SEP
        x_coords = positions[1:6, 0]  # Grid tokens only
        assert list(x_coords) == [0, 1, 2, 3, 4]

    def test_y_coordinate(self):
        """Test y-coordinate increments across rows."""
        grid = [[1], [2], [3]]  # Single column, 3 rows
        tokens = encode_example(grid, None)
        positions = compute_positions_3d(tokens)

        # Check y coords for the grid value tokens
        # START(0), 1(1), NL(2), 2(3), NL(4), 3(5), NL(6), IO_SEP(7)
        y_at_1 = positions[1, 1]  # First value
        y_at_2 = positions[3, 1]  # Second value
        y_at_3 = positions[5, 1]  # Third value

        assert y_at_1 == 0
        assert y_at_2 == 1
        assert y_at_3 == 2

    def test_z_coordinate_input_vs_output(self):
        """Test z-coordinate distinguishes input from output (matches original mdlARC)."""
        input_grid = [[1]]
        output_grid = [[2]]
        tokens = encode_example(input_grid, output_grid)
        positions = compute_positions_3d(tokens)

        # Find IO separator position
        sep_idx = tokens.index(IO_SEP_TOKEN_ID)

        # START token should have z=0
        start_z = positions[0, 2]
        assert start_z == 0

        # Input grid tokens should have z=1
        input_z = positions[1, 2]  # First input token (color)
        assert input_z == 1

        # Separator should have z=2
        sep_z = positions[sep_idx, 2]
        assert sep_z == 2

        # Output grid tokens should have z=3
        output_z = positions[sep_idx + 1, 2]  # First output token (color)
        assert output_z == 3

        # END token should have z=4
        end_idx = tokens.index(END_TOKEN_ID)
        end_z = positions[end_idx, 2]
        assert end_z == 4


# =============================================================================
# DIHEDRAL TRANSFORM TESTS
# =============================================================================

class TestDihedralTransforms:
    """Tests for the 8 dihedral transformations."""

    def test_all_unique(self, square_grid):
        """Test that all 8 transforms produce unique results."""
        results = []
        for transform in DIHEDRAL_TRANSFORMS:
            result = transform(square_grid)
            results.append(tuple(tuple(row) for row in result))

        assert len(set(results)) == 8, "All 8 transforms should be unique"

    def test_identity(self, square_grid):
        """Test identity transform."""
        assert identity(square_grid) == square_grid

    def test_rot90(self, square_grid):
        """Test 90° rotation."""
        result = rot90(square_grid)
        assert result == [[3, 1], [4, 2]]

    def test_rot180(self, square_grid):
        """Test 180° rotation."""
        result = rot180(square_grid)
        assert result == [[4, 3], [2, 1]]

    def test_rot270(self, square_grid):
        """Test 270° rotation."""
        result = rot270(square_grid)
        assert result == [[2, 4], [1, 3]]

    def test_rot360_identity(self, square_grid):
        """Test that 4 rotations = identity."""
        result = rot90(rot90(rot90(rot90(square_grid))))
        assert result == square_grid

    def test_flip_h(self, square_grid):
        """Test horizontal flip."""
        result = flip_h(square_grid)
        assert result == [[2, 1], [4, 3]]

    def test_flip_v(self, square_grid):
        """Test vertical flip."""
        result = flip_v(square_grid)
        assert result == [[3, 4], [1, 2]]

    def test_double_flip_identity(self, square_grid):
        """Test that double flip = identity."""
        assert flip_h(flip_h(square_grid)) == square_grid
        assert flip_v(flip_v(square_grid)) == square_grid

    def test_apply_dihedral_transform(self, square_grid):
        """Test applying transform to input-output pair."""
        input_grid = square_grid
        output_grid = [[5, 6], [7, 8]]

        aug_in, aug_out = apply_dihedral_transform(input_grid, output_grid, 1)

        assert aug_in == rot90(input_grid)
        assert aug_out == rot90(output_grid)

    def test_inverse_dihedral(self, square_grid):
        """Test inverse transforms recover original."""
        for i in range(8):
            transformed = DIHEDRAL_TRANSFORMS[i](square_grid)
            recovered = inverse_dihedral(transformed, i)
            assert recovered == square_grid, f"Inverse failed for transform {i}"


# =============================================================================
# COLOR AUGMENTATION TESTS
# =============================================================================

class TestColorAugmentation:
    """Tests for color permutation augmentation."""

    def test_permutation_shape(self):
        """Test permutation array shape."""
        perms = generate_color_permutations(10)
        assert perms.shape == (10, 10)

    def test_first_is_identity(self):
        """Test that first permutation is identity."""
        perms = generate_color_permutations(5)
        assert list(perms[0]) == list(range(10))

    def test_color_zero_preserved(self):
        """Test that color 0 (background) is always preserved."""
        perms = generate_color_permutations(100, seed=12345)
        assert all(perms[:, 0] == 0)

    def test_permutations_valid(self):
        """Test that all permutations are valid."""
        perms = generate_color_permutations(50)
        for perm in perms:
            assert set(perm) == set(range(10))

    def test_deterministic(self):
        """Test that same seed produces same permutations."""
        perms1 = generate_color_permutations(10, seed=42)
        perms2 = generate_color_permutations(10, seed=42)
        assert np.array_equal(perms1, perms2)

    def test_apply_color_perm(self):
        """Test applying permutation to tokens."""
        tokens = torch.tensor([[0, 1, 2, 3, 10, 11]])  # Include special tokens
        perm = torch.tensor([0, 5, 6, 7, 4, 1, 2, 3, 8, 9])

        result = apply_color_perm_to_tokens(tokens, perm)

        # Colors should be permuted
        assert result[0, 1].item() == 5  # 1 -> 5
        assert result[0, 2].item() == 6  # 2 -> 6
        # Special tokens unchanged
        assert result[0, 4].item() == 10
        assert result[0, 5].item() == 11
        # Zero unchanged
        assert result[0, 0].item() == 0

    def test_inverse_color_perm(self):
        """Test inverse permutation."""
        perm = np.array([0, 5, 6, 7, 4, 1, 2, 3, 8, 9])
        inv = inverse_color_perm(perm)

        # Applying perm then inv should give identity
        test = np.arange(10)
        permuted = perm[test]
        recovered = inv[permuted]
        assert np.array_equal(recovered, test)


# =============================================================================
# DATASET TESTS
# =============================================================================

class TestARCDataset:
    """Tests for the ARCDataset class."""

    def test_load_train_split(self, sample_challenges):
        """Test loading train split only."""
        dataset = ARCDataset(sample_challenges, splits=("train",))

        # task_001 has 2 train pairs, task_002 has 1
        assert len(dataset) == 3

    def test_load_test_split(self, sample_challenges):
        """Test loading test split only."""
        dataset = ARCDataset(sample_challenges, splits=("test",))

        # task_001 has 1 test pair, task_002 has 1
        assert len(dataset) == 2

    def test_load_both_splits(self, sample_challenges):
        """Test loading both splits."""
        dataset = ARCDataset(sample_challenges, splits=("train", "test"))

        assert len(dataset) == 5

    def test_dihedral_augmentation(self, sample_challenges):
        """Test 8x expansion with dihedral augmentation."""
        dataset_no_aug = ARCDataset(sample_challenges, splits=("train",), apply_dihedral=False)
        dataset_aug = ARCDataset(sample_challenges, splits=("train",), apply_dihedral=True)

        assert len(dataset_aug) == len(dataset_no_aug) * 8

    def test_example_structure(self, sample_challenges):
        """Test structure of returned examples."""
        dataset = ARCDataset(sample_challenges, splits=("train",))
        example = dataset[0]

        assert "tokens" in example
        assert "positions" in example
        assert "task_id" in example
        assert "split" in example
        assert "has_output" in example

    def test_train_has_output(self, sample_challenges):
        """Test that train examples have output."""
        dataset = ARCDataset(sample_challenges, splits=("train",))

        for i in range(len(dataset)):
            assert dataset[i]["has_output"] is True

    def test_test_no_output(self, sample_challenges):
        """Test that test examples don't have output (by default)."""
        dataset = ARCDataset(sample_challenges, splits=("test",))

        for i in range(len(dataset)):
            assert dataset[i]["has_output"] is False

    def test_sorted_by_length(self, sample_challenges):
        """Test that examples are sorted by length."""
        dataset = ARCDataset(sample_challenges, splits=("train",))

        lengths = [len(dataset[i]["tokens"]) for i in range(len(dataset))]
        assert lengths == sorted(lengths)

    def test_max_seq_len_drop(self, sample_challenges):
        """Test dropping sequences exceeding max length."""
        dataset_all = ARCDataset(sample_challenges, max_seq_len=1000)
        dataset_short = ARCDataset(sample_challenges, max_seq_len=10, drop_long=True)

        assert len(dataset_short) <= len(dataset_all)


# =============================================================================
# BATCHING TESTS
# =============================================================================

class TestBatching:
    """Tests for batching and collation."""

    def test_length_bucket_sampler(self):
        """Test length-bucketed sampling."""
        lengths = [10, 20, 15, 25, 12, 22]
        sampler = LengthBucketSampler(lengths, batch_size=2, shuffle=False)

        batches = list(sampler)

        # Should have 3 batches
        assert len(batches) == 3
        # Each batch should have similar lengths
        for batch in batches:
            assert len(batch) == 2

    def test_sampler_epoch(self):
        """Test epoch-based shuffling."""
        lengths = list(range(100))
        sampler = LengthBucketSampler(lengths, batch_size=10, shuffle=True)

        sampler.set_epoch(0)
        batches_e0 = [tuple(b) for b in sampler]

        sampler.set_epoch(1)
        batches_e1 = [tuple(b) for b in sampler]

        # Different epochs should give different order
        assert batches_e0 != batches_e1

    def test_collate_fn(self, sample_challenges):
        """Test collation of a batch."""
        dataset = ARCDataset(sample_challenges, splits=("train",))

        batch = [dataset[i] for i in range(min(3, len(dataset)))]
        collated = collate_fn(batch)

        assert "input_ids" in collated
        assert "attention_mask" in collated
        assert "positions_3d" in collated
        assert collated["input_ids"].dim() == 2
        assert collated["attention_mask"].dim() == 2
        assert collated["positions_3d"].dim() == 3

    def test_collate_padding(self, sample_challenges):
        """Test that padding is applied correctly."""
        dataset = ARCDataset(sample_challenges, splits=("train",))

        batch = [dataset[i] for i in range(min(3, len(dataset)))]
        collated = collate_fn(batch)

        # All sequences should be padded to same length
        batch_size, max_len = collated["input_ids"].shape
        assert batch_size == len(batch)

        # Attention mask should match
        assert collated["attention_mask"].shape == (batch_size, max_len)

    def test_collate_color_aug(self, sample_challenges):
        """Test color augmentation in collation."""
        dataset = ARCDataset(sample_challenges, splits=("train",))

        batch = [dataset[0]]
        perms = torch.tensor(generate_color_permutations(5))

        # Without augmentation
        collated_orig = collate_fn(batch, color_perms=None)

        # With augmentation (perm 1)
        collated_aug = collate_fn(batch, color_perms=perms, color_perm_idx=1)

        # Tokens should be different
        assert not torch.equal(collated_orig["input_ids"], collated_aug["input_ids"])

    def test_create_dataloader(self, sample_challenges):
        """Test dataloader creation."""
        dataset = ARCDataset(sample_challenges, splits=("train",))
        loader = create_dataloader(dataset, batch_size=2)

        batch = next(iter(loader))

        assert isinstance(batch, dict)
        assert batch["input_ids"].shape[0] <= 2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the full data pipeline."""

    def test_full_pipeline(self, sample_challenges):
        """Test complete data pipeline from load to batch."""
        # Load dataset
        dataset = ARCDataset(
            sample_challenges,
            splits=("train",),
            apply_dihedral=True,
        )

        # Create dataloader
        loader = create_dataloader(dataset, batch_size=4)

        # Get a batch
        batch = next(iter(loader))

        # Verify batch structure
        assert batch["input_ids"].dtype == torch.long
        assert batch["positions_3d"].dtype == torch.long
        assert batch["attention_mask"].dtype == torch.bool

        # Verify shapes match
        batch_size, seq_len = batch["input_ids"].shape
        assert batch["positions_3d"].shape == (batch_size, seq_len, 3)
        assert batch["attention_mask"].shape == (batch_size, seq_len)

    def test_data_transformer_compatibility(self, sample_challenges):
        """Test that data pipeline output works with transformer."""
        from src.transformer import create_tiny_model

        # Load data
        dataset = ARCDataset(sample_challenges, splits=("train",))
        loader = create_dataloader(dataset, batch_size=2)
        batch = next(iter(loader))

        # Create model
        model = create_tiny_model()
        model.eval()

        # Forward pass should work
        with torch.no_grad():
            logits = model(
                batch["input_ids"],
                batch["positions_3d"],
            )

        assert logits.shape == (*batch["input_ids"].shape, 14)

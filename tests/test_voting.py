"""Tests for the AAIVR voting module."""

import pytest

from src.data import DIHEDRAL_TRANSFORMS
from src.voting import (
    aaivr_predict,
    aggregate_votes,
    compute_voting_stats,
    grid_to_tuple,
    invert_color_permutation,
    invert_dihedral,
    select_top_k,
    tuple_to_grid,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_grid():
    """Simple 2x2 grid for testing."""
    return [[1, 2], [3, 4]]


@pytest.fixture
def identity_perm():
    """Identity color permutation."""
    return tuple(range(10))


@pytest.fixture
def swap_perm():
    """Color permutation that swaps 1<->2 and 3<->4."""
    return (0, 2, 1, 4, 3, 5, 6, 7, 8, 9)


# =============================================================================
# INVERT DIHEDRAL TESTS
# =============================================================================


class TestInvertDihedral:
    """Tests for dihedral inversion."""

    def test_identity_inverse(self, simple_grid):
        """Test that identity inverse is identity."""
        result = invert_dihedral(simple_grid, 0)
        assert result == simple_grid

    def test_rot90_inverse(self, simple_grid):
        """Test that rot90 inverse works."""
        # Apply rot90 then invert
        rotated = DIHEDRAL_TRANSFORMS[1](simple_grid)
        recovered = invert_dihedral(rotated, 1)
        assert recovered == simple_grid

    def test_rot180_inverse(self, simple_grid):
        """Test that rot180 inverse works."""
        rotated = DIHEDRAL_TRANSFORMS[2](simple_grid)
        recovered = invert_dihedral(rotated, 2)
        assert recovered == simple_grid

    def test_rot270_inverse(self, simple_grid):
        """Test that rot270 inverse works."""
        rotated = DIHEDRAL_TRANSFORMS[3](simple_grid)
        recovered = invert_dihedral(rotated, 3)
        assert recovered == simple_grid

    def test_flip_h_inverse(self, simple_grid):
        """Test that horizontal flip inverse works."""
        flipped = DIHEDRAL_TRANSFORMS[4](simple_grid)
        recovered = invert_dihedral(flipped, 4)
        assert recovered == simple_grid

    def test_flip_v_inverse(self, simple_grid):
        """Test that vertical flip inverse works."""
        flipped = DIHEDRAL_TRANSFORMS[5](simple_grid)
        recovered = invert_dihedral(flipped, 5)
        assert recovered == simple_grid

    def test_flip_diag_inverse(self, simple_grid):
        """Test that diagonal flip inverse works."""
        flipped = DIHEDRAL_TRANSFORMS[6](simple_grid)
        recovered = invert_dihedral(flipped, 6)
        assert recovered == simple_grid

    def test_flip_anti_diag_inverse(self, simple_grid):
        """Test that anti-diagonal flip inverse works."""
        flipped = DIHEDRAL_TRANSFORMS[7](simple_grid)
        recovered = invert_dihedral(flipped, 7)
        assert recovered == simple_grid

    def test_all_inverses_roundtrip(self, simple_grid):
        """Test all 8 dihedral transforms have correct inverses."""
        for idx in range(8):
            transformed = DIHEDRAL_TRANSFORMS[idx](simple_grid)
            recovered = invert_dihedral(transformed, idx)
            assert recovered == simple_grid, f"Inverse failed for transform {idx}"


# =============================================================================
# INVERT COLOR PERMUTATION TESTS
# =============================================================================


class TestInvertColorPermutation:
    """Tests for color permutation inversion."""

    def test_identity_perm(self, simple_grid, identity_perm):
        """Test that identity permutation is a no-op."""
        result = invert_color_permutation(simple_grid, identity_perm)
        assert result == simple_grid

    def test_swap_perm_inverse(self, swap_perm):
        """Test that swap permutation inverts correctly."""
        grid = [[1, 2], [3, 4]]
        # Apply permutation: 1->2, 2->1, 3->4, 4->3
        permuted = [[2, 1], [4, 3]]

        recovered = invert_color_permutation(permuted, swap_perm)
        assert recovered == grid

    def test_preserves_background(self):
        """Test that color 0 (background) is preserved."""
        grid = [[0, 1], [2, 0]]
        perm = (0, 5, 6, 3, 4, 1, 2, 7, 8, 9)  # 1<->5, 2<->6

        # Apply perm
        permuted = [[0, 5], [6, 0]]

        recovered = invert_color_permutation(permuted, perm)
        assert recovered == grid

    def test_complex_permutation(self):
        """Test a more complex permutation."""
        # Permutation: 1->3, 2->5, 3->1, 5->2
        perm = (0, 3, 5, 1, 4, 2, 6, 7, 8, 9)

        grid = [[1, 2, 3], [4, 5, 6]]
        # After applying perm: 1->3, 2->5, 3->1, 5->2
        permuted = [[3, 5, 1], [4, 2, 6]]

        recovered = invert_color_permutation(permuted, perm)
        assert recovered == grid


# =============================================================================
# GRID TUPLE CONVERSION TESTS
# =============================================================================


class TestGridTupleConversion:
    """Tests for grid <-> tuple conversion."""

    def test_grid_to_tuple(self, simple_grid):
        """Test grid to tuple conversion."""
        t = grid_to_tuple(simple_grid)
        assert t == ((1, 2), (3, 4))
        assert isinstance(t, tuple)
        assert all(isinstance(row, tuple) for row in t)

    def test_tuple_to_grid(self):
        """Test tuple to grid conversion."""
        t = ((1, 2), (3, 4))
        g = tuple_to_grid(t)
        assert g == [[1, 2], [3, 4]]
        assert isinstance(g, list)
        assert all(isinstance(row, list) for row in g)

    def test_roundtrip(self, simple_grid):
        """Test grid -> tuple -> grid roundtrip."""
        t = grid_to_tuple(simple_grid)
        g = tuple_to_grid(t)
        assert g == simple_grid

    def test_hashable(self, simple_grid):
        """Test that tuple is hashable (can be used as dict key)."""
        t = grid_to_tuple(simple_grid)
        d = {t: 1}
        assert d[t] == 1


# =============================================================================
# AGGREGATE VOTES TESTS
# =============================================================================


class TestAggregateVotes:
    """Tests for vote aggregation."""

    def test_empty_predictions(self):
        """Test with empty predictions list."""
        result = aggregate_votes([])
        assert result == []

    def test_all_none_predictions(self):
        """Test with all None predictions."""
        predictions = [(None, 0, None), (None, 1, None), (None, 2, None)]
        result = aggregate_votes(predictions)
        assert result == []

    def test_single_prediction(self, simple_grid):
        """Test with single prediction."""
        predictions = [(simple_grid, 0, None)]
        result = aggregate_votes(predictions)

        assert len(result) == 1
        grid, count = result[0]
        assert grid == simple_grid
        assert count == 1

    def test_majority_vote(self, simple_grid):
        """Test that majority vote wins."""
        other_grid = [[5, 6], [7, 8]]

        predictions = [
            (simple_grid, 0, None),
            (simple_grid, 0, None),
            (simple_grid, 0, None),
            (other_grid, 0, None),
        ]
        result = aggregate_votes(predictions)

        assert len(result) == 2
        # simple_grid should be first with 3 votes
        assert result[0][0] == simple_grid
        assert result[0][1] == 3
        # other_grid second with 1 vote
        assert result[1][0] == other_grid
        assert result[1][1] == 1

    def test_inverts_dihedral_before_counting(self):
        """Test that dihedral is inverted before counting votes."""
        grid = [[1, 2], [3, 4]]

        # Apply rot90 to get a different representation
        rotated = DIHEDRAL_TRANSFORMS[1](grid)

        predictions = [
            (grid, 0, None),  # Identity
            (rotated, 1, None),  # Rotated - should map back to same
        ]
        result = aggregate_votes(predictions)

        # Both should count as same grid
        assert len(result) == 1
        assert result[0][1] == 2  # 2 votes for same canonical grid

    def test_inverts_color_before_counting(self):
        """Test that color perm is inverted before counting votes."""
        grid = [[1, 2], [3, 4]]
        perm = (0, 2, 1, 4, 3, 5, 6, 7, 8, 9)  # Swap 1<->2, 3<->4

        # Grid after applying perm
        permuted = [[2, 1], [4, 3]]

        predictions = [
            (grid, 0, None),  # No color aug
            (permuted, 0, perm),  # With color aug - should map back
        ]
        result = aggregate_votes(predictions)

        assert len(result) == 1
        assert result[0][1] == 2

    def test_mixed_valid_invalid(self, simple_grid):
        """Test with mix of valid and None predictions."""
        # Apply rot180 to get the transformed version
        rot180_grid = DIHEDRAL_TRANSFORMS[2](simple_grid)

        predictions = [
            (simple_grid, 0, None),  # Identity
            (None, 1, None),  # Invalid
            (rot180_grid, 2, None),  # Rot180 applied, should map back
            (None, 3, None),  # Invalid
        ]
        result = aggregate_votes(predictions)

        # Only valid predictions counted, both should map to same canonical
        assert len(result) == 1
        assert result[0][1] == 2


# =============================================================================
# SELECT TOP K TESTS
# =============================================================================


class TestSelectTopK:
    """Tests for top-k selection."""

    def test_select_top_1(self):
        """Test selecting top 1."""
        grid1 = [[1, 2]]
        grid2 = [[3, 4]]
        votes = [(grid1, 5), (grid2, 3)]

        result = select_top_k(votes, k=1)
        assert len(result) == 1
        assert result[0] == grid1

    def test_select_top_2(self):
        """Test selecting top 2."""
        grid1 = [[1, 2]]
        grid2 = [[3, 4]]
        grid3 = [[5, 6]]
        votes = [(grid1, 5), (grid2, 3), (grid3, 1)]

        result = select_top_k(votes, k=2)
        assert len(result) == 2
        assert result[0] == grid1
        assert result[1] == grid2

    def test_fewer_than_k(self):
        """Test when fewer predictions than k."""
        grid1 = [[1, 2]]
        votes = [(grid1, 5)]

        result = select_top_k(votes, k=3)
        assert len(result) == 1
        assert result[0] == grid1

    def test_empty_votes(self):
        """Test with empty votes."""
        result = select_top_k([], k=2)
        assert result == []


# =============================================================================
# AAIVR PREDICT TESTS
# =============================================================================


class TestAAIVRPredict:
    """Tests for the full AAIVR pipeline."""

    def test_basic_aaivr(self):
        """Test basic AAIVR prediction."""
        grid = [[1, 2], [3, 4]]

        predictions = [
            (grid, 0, None),
            (grid, 0, None),
            ([[5, 6]], 0, None),
        ]

        result = aaivr_predict(predictions, top_k=2)

        assert len(result) == 2
        assert result[0] == grid  # Most votes

    def test_aaivr_with_augmentations(self):
        """Test AAIVR with various augmentations."""
        grid = [[1, 2], [3, 4]]

        # Create predictions under different augmentations
        # All should map back to same canonical form
        predictions = []
        for dihedral_idx in range(8):
            transformed = DIHEDRAL_TRANSFORMS[dihedral_idx](grid)
            predictions.append((transformed, dihedral_idx, None))

        result = aaivr_predict(predictions, top_k=1)

        assert len(result) == 1
        assert result[0] == grid  # All 8 should vote for same grid


# =============================================================================
# VOTING STATS TESTS
# =============================================================================


class TestVotingStats:
    """Tests for voting statistics."""

    def test_empty_stats(self):
        """Test stats with no predictions."""
        stats = compute_voting_stats([])
        assert stats["total_predictions"] == 0
        assert stats["unique_grids"] == 0
        assert stats["confidence"] == 0.0

    def test_all_invalid_stats(self):
        """Test stats with all invalid predictions."""
        predictions = [(None, 0, None), (None, 1, None)]
        stats = compute_voting_stats(predictions)

        assert stats["total_predictions"] == 2
        assert stats["valid_predictions"] == 0
        assert stats["invalid_predictions"] == 2
        assert stats["confidence"] == 0.0

    def test_valid_stats(self):
        """Test stats with valid predictions."""
        grid = [[1, 2]]
        predictions = [
            (grid, 0, None),
            (grid, 0, None),
            (grid, 0, None),
            ([[5, 6]], 0, None),
            (None, 0, None),
        ]
        stats = compute_voting_stats(predictions)

        assert stats["total_predictions"] == 5
        assert stats["valid_predictions"] == 4
        assert stats["invalid_predictions"] == 1
        assert stats["unique_grids"] == 2
        assert stats["top_vote_count"] == 3
        assert stats["confidence"] == 3 / 4  # 3 votes out of 4 valid

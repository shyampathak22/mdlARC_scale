"""
AAIVR (Augmentation-Aware Inverse Voting) for ARC predictions.

Given predictions under multiple augmentations, this module:
1. Inverts each augmentation to get canonical predictions
2. Aggregates votes for each unique prediction
3. Returns top-k predictions by vote count
"""

import random
from collections import Counter

from src.data import DIHEDRAL_TRANSFORMS

# Inverse dihedral mapping: DIHEDRAL_TRANSFORMS[inverse_idx] undoes DIHEDRAL_TRANSFORMS[idx]
INVERSE_DIHEDRAL = [0, 3, 2, 1, 4, 5, 6, 7]


def invert_dihedral(grid: list[list[int]], dihedral_idx: int) -> list[list[int]]:
    """
    Apply inverse of a dihedral transform to recover original orientation.

    Args:
        grid: The transformed grid
        dihedral_idx: Which dihedral transform was applied (0-7)

    Returns:
        Grid in original orientation
    """
    inverse_idx = INVERSE_DIHEDRAL[dihedral_idx]
    return DIHEDRAL_TRANSFORMS[inverse_idx](grid)


def invert_color_permutation(
    grid: list[list[int]],
    perm: tuple[int, ...],
) -> list[list[int]]:
    """
    Apply inverse color permutation to grid.

    Args:
        grid: Grid with permuted colors
        perm: The color permutation that was applied (10 elements, mapping old -> new)

    Returns:
        Grid with original colors restored
    """
    # Build inverse permutation
    inv_perm = [0] * 10
    for old_color, new_color in enumerate(perm):
        inv_perm[new_color] = old_color

    # Apply to grid
    return [[inv_perm[cell] for cell in row] for row in grid]


def grid_to_tuple(grid: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    """Convert grid to hashable tuple for vote counting."""
    return tuple(tuple(row) for row in grid)


def tuple_to_grid(t: tuple[tuple[int, ...], ...]) -> list[list[int]]:
    """Convert tuple back to grid."""
    return [list(row) for row in t]


def is_rectangular_grid(grid: list[list[int]]) -> bool:
    """Return True if grid is non-empty and rectangular."""
    if not grid or not grid[0]:
        return False
    row_len = len(grid[0])
    return all(len(row) == row_len for row in grid)


def _is_valid_grid(
    grid: list[list[int]],
    test_input: list[list[int]] | None = None,
    discard_input_copies: bool = True,
) -> bool:
    """
    Check if a grid is valid for voting.

    Filters out:
    - Empty grids
    - Malformed grids (jagged rows)
    - Input-copy predictions (if test_input provided)
    """
    if not is_rectangular_grid(grid):
        return False

    # Filter input-copy (indicates model failure)
    if discard_input_copies and test_input is not None and grid == test_input:
        return False

    return True


def _canonicalize_prediction(
    pred_grid: list[list[int]],
    dihedral_idx: int,
    color_perm: tuple[int, ...] | None,
) -> list[list[int]]:
    """Invert augmentations to get canonical prediction."""
    canonical = invert_dihedral(pred_grid, dihedral_idx)
    if color_perm is not None:
        canonical = invert_color_permutation(canonical, color_perm)
    return canonical


def aggregate_votes(
    predictions: list[tuple[list[list[int]] | None, int, tuple[int, ...] | None]],
    test_input: list[list[int]] | None = None,
    discard_input_copies: bool = True,
    rng: random.Random | None = None,
) -> list[tuple[list[list[int]], int]]:
    """
    Aggregate predictions by inverting augmentations and counting votes.

    For each prediction:
    1. Filter invalid predictions (None, empty, malformed, input-copy)
    2. Apply inverse dihedral transform
    3. Apply inverse color permutation (if color aug was used)
    4. Convert to canonical form and count votes

    Args:
        predictions: List of (predicted_grid, dihedral_idx, color_perm) tuples.
                    predicted_grid may be None if decoding failed.
        test_input: Optional test input grid to filter input-copy predictions.
        discard_input_copies: If True, drop predictions identical to test_input.
        rng: Optional RNG for randomized tie-breaking.

    Returns:
        List of (canonical_grid, vote_count) sorted by votes descending.
    """
    vote_counter: Counter[tuple[tuple[int, ...], ...]] = Counter()

    for pred_grid, dihedral_idx, color_perm in predictions:
        # Filter None predictions
        if pred_grid is None:
            continue

        # Filter empty, malformed, and input-copy predictions
        if not _is_valid_grid(pred_grid, test_input, discard_input_copies):
            continue

        # Step 1: Invert augmentations
        canonical = _canonicalize_prediction(pred_grid, dihedral_idx, color_perm)

        # Step 3: Convert to hashable and count
        canonical_tuple = grid_to_tuple(canonical)
        vote_counter[canonical_tuple] += 1

    # Sort by vote count descending with randomized tie-breaks
    if not vote_counter:
        return []

    if rng is None:
        rng = random.Random()

    items = list(vote_counter.items())
    tie_breakers = {grid: rng.random() for grid, _ in items}
    items.sort(key=lambda item: (-item[1], tie_breakers[item[0]]))

    # Convert back to grids
    return [(tuple_to_grid(t), count) for t, count in items]


def select_top_k(
    votes: list[tuple[list[list[int]], int]],
    k: int = 2,
) -> list[list[list[int]]]:
    """
    Select top k predictions by vote count.

    Args:
        votes: List of (grid, vote_count) sorted by votes descending
        k: Number of predictions to return

    Returns:
        List of up to k grids (may be fewer if not enough unique predictions)
    """
    return [grid for grid, _ in votes[:k]]


def aaivr_predict(
    predictions: list[tuple[list[list[int]] | None, int, tuple[int, ...] | None]],
    top_k: int = 2,
    test_input: list[list[int]] | None = None,
    discard_input_copies: bool = True,
    rng: random.Random | None = None,
) -> list[list[list[int]]]:
    """
    Full AAIVR pipeline: aggregate votes and select top-k.

    This is the main entry point for AAIVR voting.

    Args:
        predictions: List of (predicted_grid, dihedral_idx, color_perm) tuples
        top_k: Number of predictions to return
        test_input: Optional test input grid to filter input-copy predictions
        discard_input_copies: If True, drop predictions identical to test_input
        rng: Optional RNG for randomized tie-breaking

    Returns:
        List of top-k grids by vote count
    """
    votes = aggregate_votes(
        predictions,
        test_input=test_input,
        discard_input_copies=discard_input_copies,
        rng=rng,
    )
    return select_top_k(votes, k=top_k)


def compute_voting_stats(
    predictions: list[tuple[list[list[int]] | None, int, tuple[int, ...] | None]],
    test_input: list[list[int]] | None = None,
    discard_input_copies: bool = True,
) -> dict:
    """
    Compute statistics about the voting results.

    Useful for debugging and analysis.

    Args:
        predictions: List of (predicted_grid, dihedral_idx, color_perm) tuples
        test_input: Optional test input grid to filter input-copy predictions
        discard_input_copies: If True, drop predictions identical to test_input

    Returns:
        Dictionary with voting statistics
    """
    votes = aggregate_votes(
        predictions,
        test_input=test_input,
        discard_input_copies=discard_input_copies,
        rng=random.Random(0),
    )

    # Count valid predictions (not None and passes validation)
    total_valid = sum(
        1
        for p, _, _ in predictions
        if p is not None and _is_valid_grid(p, test_input, discard_input_copies)
    )
    total_invalid = len(predictions) - total_valid

    if not votes:
        return {
            "total_predictions": len(predictions),
            "valid_predictions": total_valid,
            "invalid_predictions": total_invalid,
            "unique_grids": 0,
            "top_vote_count": 0,
            "vote_distribution": [],
            "confidence": 0.0,
        }

    vote_counts = [count for _, count in votes]
    top_count = vote_counts[0] if vote_counts else 0

    # Confidence: top votes / total valid predictions
    confidence = top_count / total_valid if total_valid > 0 else 0.0

    return {
        "total_predictions": len(predictions),
        "valid_predictions": total_valid,
        "invalid_predictions": total_invalid,
        "unique_grids": len(votes),
        "top_vote_count": top_count,
        "vote_distribution": vote_counts[:10],  # Top 10
        "confidence": confidence,
    }

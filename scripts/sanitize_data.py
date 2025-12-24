#!/usr/bin/env python3
"""
Data sanitization script to prevent evaluation data leakage.

This script ensures that evaluation solutions are NEVER present during training.
It should be run before any training to guarantee fair evaluation.

Usage:
    uv run python scripts/sanitize_data.py --data-dir data/
    uv run python scripts/sanitize_data.py --data-dir data/ --restore  # For scoring only
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Files that MUST be deleted before training
FORBIDDEN_FILES = [
    "arc-agi_evaluation_solutions.json",
    "arc-agi_test_solutions.json",  # Private test set (shouldn't exist locally)
]

# Files needed for training
REQUIRED_TRAINING_FILES = [
    "arc-agi_training_challenges.json",
    "arc-agi_training_solutions.json",
    "arc-agi_evaluation_challenges.json",  # Inputs only, no outputs
]

# Known file hashes for verification (SHA256)
# These help detect if someone accidentally mixed in solutions
KNOWN_HASHES = {
    # ARC-AGI official files (update these if dataset changes)
    "arc-agi_training_challenges.json": None,  # Will be computed on first run
    "arc-agi_training_solutions.json": None,
    "arc-agi_evaluation_challenges.json": None,
}


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_for_leakage(data_dir: Path) -> list[str]:
    """
    Check for potential data leakage issues.

    Returns list of warning messages.
    """
    warnings = []

    # Check for forbidden files
    for filename in FORBIDDEN_FILES:
        filepath = data_dir / filename
        if filepath.exists():
            warnings.append(f"CRITICAL: Found forbidden file: {filepath}")

    # Check if training solutions accidentally contain eval task IDs
    train_solutions = data_dir / "arc-agi_training_solutions.json"
    eval_challenges = data_dir / "arc-agi_evaluation_challenges.json"

    if train_solutions.exists() and eval_challenges.exists():
        with open(train_solutions) as f:
            sol_ids = set(json.load(f).keys())
        with open(eval_challenges) as f:
            eval_ids = set(json.load(f).keys())

        overlap = sol_ids & eval_ids
        if overlap:
            warnings.append(
                f"CRITICAL: Training solutions contain {len(overlap)} eval task IDs! "
                f"This is data leakage. First few: {list(overlap)[:5]}"
            )

    return warnings


def sanitize(data_dir: Path, verbose: bool = True) -> bool:
    """
    Remove all evaluation solution files to prevent leakage.

    Returns True if sanitization was successful.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        return False

    deleted = []

    # Delete forbidden files
    for filename in FORBIDDEN_FILES:
        filepath = data_dir / filename
        if filepath.exists():
            filepath.unlink()
            deleted.append(filename)
            if verbose:
                print(f"Deleted: {filepath}")

    # Check for leakage
    warnings = check_for_leakage(data_dir)
    for warning in warnings:
        print(f"WARNING: {warning}")

    # Verify required files exist
    missing = []
    for filename in REQUIRED_TRAINING_FILES:
        if not (data_dir / filename).exists():
            missing.append(filename)

    if missing:
        print(f"Missing required files: {missing}")
        print("Run: uv run python -c \"from src.data import download_dataset; download_dataset('arc2')\"")
        return False

    if verbose:
        if deleted:
            print(f"\nSanitization complete. Deleted {len(deleted)} file(s).")
        else:
            print("\nData directory is already clean.")
        print("Safe to proceed with training.")

    return len(warnings) == 0


def restore_for_scoring(data_dir: Path, verbose: bool = True) -> bool:
    """
    Re-download evaluation solutions for scoring only.

    This should ONLY be called after training/inference is complete.
    """
    data_dir = Path(data_dir)

    # Download from ARC-AGI repo
    eval_solutions_url = (
        "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/"
        "data/evaluation/arc-agi_evaluation_solutions.json"
    )

    output_path = data_dir / "arc-agi_evaluation_solutions.json"

    if verbose:
        print(f"Downloading evaluation solutions for scoring...")

    try:
        result = subprocess.run(
            ["curl", "-fsSL", "-o", str(output_path), eval_solutions_url],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            # Try with wget
            result = subprocess.run(
                ["wget", "-q", "-O", str(output_path), eval_solutions_url],
                capture_output=True,
                text=True,
            )

        if result.returncode != 0:
            print(f"Error downloading solutions: {result.stderr}")
            return False

        if verbose:
            print(f"Downloaded: {output_path}")

        return True

    except FileNotFoundError:
        print("Error: curl or wget not found. Please install one of them.")
        return False


def verify_clean(data_dir: Path) -> bool:
    """Verify the data directory is clean for training."""
    data_dir = Path(data_dir)

    # Check no forbidden files
    for filename in FORBIDDEN_FILES:
        if (data_dir / filename).exists():
            return False

    # Check no leakage
    warnings = check_for_leakage(data_dir)
    return len(warnings) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Sanitize data directory to prevent evaluation data leakage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Before training (REQUIRED):
  python scripts/sanitize_data.py --data-dir data/

  # Verify data is clean:
  python scripts/sanitize_data.py --data-dir data/ --verify

  # After inference, restore solutions for scoring:
  python scripts/sanitize_data.py --data-dir data/ --restore
        """,
    )

    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        default=Path("data"),
        help="Path to data directory",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore evaluation solutions (for scoring only)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify data is clean without modifying",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()

    if args.verify:
        if verify_clean(args.data_dir):
            if not args.quiet:
                print("Data directory is clean. Safe for training.")
            sys.exit(0)
        else:
            print("ERROR: Data directory contains forbidden files or has leakage!")
            sys.exit(1)

    if args.restore:
        success = restore_for_scoring(args.data_dir, verbose=not args.quiet)
        sys.exit(0 if success else 1)

    # Default: sanitize
    success = sanitize(args.data_dir, verbose=not args.quiet)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

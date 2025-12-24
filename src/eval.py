"""
Evaluation script for ARC models.

Provides:
- Single task evaluation
- Full dataset evaluation with metrics
- Submission JSON generation for ARC-AGI
- CLI interface
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from src.inference import predict_with_augmentations
from src.train import TrainConfig
from src.transformer import (
    TinyTransformer,
    create_large_model,
    create_medium_model,
    create_model,
    create_small_model,
    create_tiny_model,
)
from src.voting import aaivr_predict, compute_voting_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def grids_equal(grid1: list[list[int]], grid2: list[list[int]]) -> bool:
    """Check if two grids are equal."""
    if len(grid1) != len(grid2):
        return False
    for row1, row2 in zip(grid1, grid2, strict=True):
        if row1 != row2:
            return False
    return True


def evaluate_task(
    model: TinyTransformer,
    task: dict,
    task_id: str,
    device: torch.device,
    solutions: list[list[list[int]]] | None = None,
    num_color_perms: int = 8,
    top_k: int = 2,
    max_output_tokens: int = 900,
) -> dict:
    """
    Evaluate model on a single ARC task.

    Args:
        model: The transformer model
        task: ARC task dict with "train" and "test" keys
        task_id: Task identifier
        device: Device to run on
        solutions: Optional list of ground truth solutions for test examples
        num_color_perms: Number of color permutations for AAIVR
        top_k: Number of predictions to return
        max_output_tokens: Maximum tokens per prediction

    Returns:
        Dictionary with evaluation results
    """
    model.eval()

    results = {
        "task_id": task_id,
        "test_outputs": [],
        "correct_at_1": [],
        "correct_at_2": [],
        "voting_stats": [],
    }

    num_tests = len(task["test"])

    for test_idx in range(num_tests):
        # Get the test input for filtering input-copy predictions
        test_input = task["test"][test_idx]["input"]

        # Get predictions under all augmentations
        predictions = predict_with_augmentations(
            model,
            task,
            test_idx,
            device,
            num_color_perms=num_color_perms,
            max_output_tokens=max_output_tokens,
        )

        # Compute voting stats (with input-copy filtering)
        stats = compute_voting_stats(predictions, test_input=test_input)
        results["voting_stats"].append(stats)

        # Get top-k predictions via AAIVR (with input-copy filtering)
        top_predictions = aaivr_predict(predictions, top_k=top_k, test_input=test_input)
        results["test_outputs"].append(top_predictions)

        # Check correctness if solutions provided
        if solutions is not None and test_idx < len(solutions):
            ground_truth = solutions[test_idx]

            # Pass@1: first prediction correct
            correct_at_1 = (
                len(top_predictions) > 0 and grids_equal(top_predictions[0], ground_truth)
            )
            results["correct_at_1"].append(correct_at_1)

            # Pass@2: any of top 2 predictions correct
            correct_at_2 = any(
                grids_equal(pred, ground_truth) for pred in top_predictions[:2]
            )
            results["correct_at_2"].append(correct_at_2)
        else:
            results["correct_at_1"].append(None)
            results["correct_at_2"].append(None)

    # Aggregate task-level metrics
    valid_at_1 = [c for c in results["correct_at_1"] if c is not None]
    valid_at_2 = [c for c in results["correct_at_2"] if c is not None]

    results["pass_at_1"] = all(valid_at_1) if valid_at_1 else None
    results["pass_at_2"] = all(valid_at_2) if valid_at_2 else None

    return results


def evaluate_dataset(
    model: TinyTransformer,
    challenges: dict[str, dict],
    device: torch.device,
    solutions: dict[str, list[list[list[int]]]] | None = None,
    num_color_perms: int = 8,
    top_k: int = 2,
    max_output_tokens: int = 900,
    show_progress: bool = True,
) -> dict:
    """
    Evaluate model on full dataset.

    Args:
        model: The transformer model
        challenges: Dict mapping task_id -> task
        device: Device to run on
        solutions: Optional dict mapping task_id -> list of solutions
        num_color_perms: Number of color permutations for AAIVR
        top_k: Number of predictions per test
        max_output_tokens: Maximum tokens per prediction
        show_progress: Whether to show progress bar

    Returns:
        Dictionary with aggregated metrics and per-task results
    """
    model.eval()

    results = {
        "total_tasks": len(challenges),
        "correct_at_1": 0,
        "correct_at_2": 0,
        "total_test_examples": 0,
        "per_task_results": {},
    }

    task_iterator = challenges.items()
    if show_progress:
        task_iterator = tqdm(task_iterator, desc="Evaluating", total=len(challenges))

    for task_id, task in task_iterator:
        task_solutions = solutions.get(task_id) if solutions else None

        task_result = evaluate_task(
            model,
            task,
            task_id,
            device,
            solutions=task_solutions,
            num_color_perms=num_color_perms,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
        )

        results["per_task_results"][task_id] = task_result
        results["total_test_examples"] += len(task["test"])

        if task_result["pass_at_1"] is True:
            results["correct_at_1"] += 1
        if task_result["pass_at_2"] is True:
            results["correct_at_2"] += 1

    # Compute accuracy
    total = results["total_tasks"]
    results["accuracy_at_1"] = results["correct_at_1"] / total if total > 0 else 0.0
    results["accuracy_at_2"] = results["correct_at_2"] / total if total > 0 else 0.0

    return results


def generate_submission(
    model: TinyTransformer,
    challenges: dict[str, dict],
    output_path: Path,
    device: torch.device,
    num_color_perms: int = 8,
    top_k: int = 2,
    max_output_tokens: int = 900,
    show_progress: bool = True,
) -> None:
    """
    Generate submission.json for ARC-AGI evaluation.

    Format:
    {
        "task_id": [
            {"attempt_1": [[grid]], "attempt_2": [[grid]]},
            ...  # one per test input
        ]
    }

    Args:
        model: The transformer model
        challenges: Dict mapping task_id -> task
        output_path: Path to write submission JSON
        device: Device to run on
        num_color_perms: Number of color permutations
        top_k: Number of attempts per test (max 2)
        max_output_tokens: Maximum tokens per prediction
        show_progress: Whether to show progress bar
    """
    model.eval()

    submission = {}

    task_iterator = challenges.items()
    if show_progress:
        task_iterator = tqdm(task_iterator, desc="Generating submission", total=len(challenges))

    for task_id, task in task_iterator:
        task_submissions = []

        for test_idx in range(len(task["test"])):
            test_input = task["test"][test_idx]["input"]

            predictions = predict_with_augmentations(
                model,
                task,
                test_idx,
                device,
                num_color_perms=num_color_perms,
                max_output_tokens=max_output_tokens,
            )

            top_predictions = aaivr_predict(
                predictions, top_k=min(top_k, 2), test_input=test_input
            )

            # Build submission entry
            entry = {}
            if len(top_predictions) > 0:
                entry["attempt_1"] = top_predictions[0]
            else:
                entry["attempt_1"] = [[0]]  # Fallback empty grid

            if len(top_predictions) > 1:
                entry["attempt_2"] = top_predictions[1]
            else:
                entry["attempt_2"] = entry["attempt_1"]  # Duplicate first

            task_submissions.append(entry)

        submission[task_id] = task_submissions

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(submission, f)

    logger.info(f"Submission saved to {output_path}")


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> TinyTransformer:
    """
    Load a model from a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint .pt file
        device: Device to load model on

    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint
    if "config" in checkpoint:
        config = checkpoint["config"]
        if isinstance(config, TrainConfig):
            # Create model from train config using factory functions
            model_factories = {
                "tiny": create_tiny_model,
                "small": create_small_model,
                "medium": create_medium_model,
                "large": create_large_model,
            }
            factory = model_factories.get(config.model_size, create_small_model)
            model = factory()
        elif isinstance(config, dict):
            # Assume it's a TransformerConfig dict or similar
            model = create_model(**config)
        else:
            model = create_model()
    else:
        # Default model
        model = create_model()

    # Load weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        # Try loading directly
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Model parameters: {model.num_parameters:,}")

    return model


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate ARC model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--challenges",
        type=Path,
        required=True,
        help="Path to challenges JSON file",
    )
    parser.add_argument(
        "--solutions",
        type=Path,
        default=None,
        help="Path to solutions JSON file (optional, for scoring)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write submission JSON (optional)",
    )
    parser.add_argument(
        "--num-color-perms",
        type=int,
        default=8,
        help="Number of color permutations for AAIVR (default: 8)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Number of predictions to generate (default: 2)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=900,
        help="Maximum output tokens per prediction (default: 900)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress bars",
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load model
    model = load_model_from_checkpoint(args.checkpoint, device)

    # Load challenges
    with open(args.challenges) as f:
        challenges = json.load(f)
    logger.info(f"Loaded {len(challenges)} challenges")

    # Load solutions if provided
    solutions = None
    if args.solutions:
        with open(args.solutions) as f:
            solutions = json.load(f)
        logger.info(f"Loaded solutions for {len(solutions)} tasks")

    # Generate submission if output path provided
    if args.output:
        generate_submission(
            model,
            challenges,
            args.output,
            device,
            num_color_perms=args.num_color_perms,
            top_k=args.top_k,
            max_output_tokens=args.max_tokens,
            show_progress=not args.quiet,
        )

    # Evaluate if solutions provided
    if solutions:
        results = evaluate_dataset(
            model,
            challenges,
            device,
            solutions=solutions,
            num_color_perms=args.num_color_perms,
            top_k=args.top_k,
            max_output_tokens=args.max_tokens,
            show_progress=not args.quiet,
        )

        # Print results
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Total tasks: {results['total_tasks']}")
        print(f"Total test examples: {results['total_test_examples']}")
        print(f"Correct @ 1: {results['correct_at_1']} ({results['accuracy_at_1']:.2%})")
        print(f"Correct @ 2: {results['correct_at_2']} ({results['accuracy_at_2']:.2%})")
        print("=" * 50)

        # Save detailed results
        if args.output:
            results_path = args.output.with_suffix(".results.json")
            with open(results_path, "w") as f:
                # Remove per-task details for summary file
                summary = {k: v for k, v in results.items() if k != "per_task_results"}
                json.dump(summary, f, indent=2)
            logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()

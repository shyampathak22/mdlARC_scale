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

# Default paths - ARC-AGI 1 evaluation set (400 tasks)
DEFAULT_EVAL_CHALLENGES = Path("data/arc-agi_evaluation_challenges.json")
DEFAULT_EVAL_SOLUTIONS = Path("data/arc-agi_evaluation_solutions.json")


def ensure_eval_data_exists() -> tuple[Path, Path]:
    """
    Build evaluation challenges and solutions files if they don't exist.
    Downloads ARC-AGI if needed and extracts from individual task files.

    Returns:
        Tuple of (challenges_path, solutions_path)
    """
    challenges_path = DEFAULT_EVAL_CHALLENGES
    solutions_path = DEFAULT_EVAL_SOLUTIONS

    if challenges_path.exists() and solutions_path.exists():
        return challenges_path, solutions_path

    from src.data import download_dataset

    data_dir = challenges_path.parent
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download ARC-1 if needed
    arc1_data_dir = data_dir / "ARC-AGI-master" / "data"
    if not arc1_data_dir.exists():
        logger.info("Downloading ARC-1 dataset...")
        download_dataset("arc1", output_dir=data_dir)

    # Build from individual task files
    eval_dir = arc1_data_dir / "evaluation"
    if not eval_dir.exists():
        raise FileNotFoundError(f"Evaluation directory not found: {eval_dir}")

    logger.info("Building evaluation challenges and solutions...")
    challenges = {}
    solutions = {}

    for f in sorted(eval_dir.glob("*.json")):
        with open(f) as fp:
            task = json.load(fp)
        task_id = f.stem
        challenges[task_id] = task
        # Extract solutions (test outputs)
        solutions[task_id] = [t["output"] for t in task["test"]]

    with open(challenges_path, "w") as fp:
        json.dump(challenges, fp)
    with open(solutions_path, "w") as fp:
        json.dump(solutions, fp)

    logger.info(f"Built evaluation data: {len(challenges)} tasks")
    return challenges_path, solutions_path


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
    example_id_map: dict[str, int] | None = None,
    use_train_examples: bool = True,
    solutions: list[list[list[int]]] | None = None,
    num_color_perms: int = 8,
    top_k: int = 2,
    max_output_tokens: int = 900,
    num_samples: int = 1,
    sample_temperature: float = 0.0,
    sample_top_k: int | None = None,
    constrain_decoding: bool = False,
    palette_penalty: float = 2.0,
    shape_penalty: float = 2.0,
) -> dict:
    """
    Evaluate model on a single ARC task.

    Args:
        model: The transformer model
        task: ARC task dict with "train" and "test" keys
        task_id: Task identifier
        device: Device to run on
        example_id_map: Optional task_id -> example_id mapping for conditioning
        use_train_examples: If False, prompt uses only the test input (per-pair prompt)
        solutions: Optional list of ground truth solutions for test examples
        num_color_perms: Number of color permutations for AAIVR
        top_k: Number of predictions to return
        max_output_tokens: Maximum tokens per prediction
        num_samples: Number of samples per augmentation (self-consistency)
        sample_temperature: Sampling temperature for self-consistency
        sample_top_k: If set and temperature > 0, sample from top-k logits
        constrain_decoding: If True, apply soft palette/shape constraints
        palette_penalty: Logit penalty for disallowed colors
        shape_penalty: Logit penalty for shape rule violations

    Returns:
        Dictionary with evaluation results (includes fractional_at_1/2 for ARC-style scoring)
    """
    model.eval()

    results = {
        "task_id": task_id,
        "test_outputs": [],
        "correct_at_1": [],
        "correct_at_2": [],
        "voting_stats": [],
    }

    example_id = example_id_map.get(task_id, 0) if example_id_map else 0
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
            example_id=example_id,
            use_train_examples=use_train_examples,
            num_color_perms=num_color_perms,
            max_output_tokens=max_output_tokens,
            num_samples=num_samples,
            sample_temperature=sample_temperature,
            sample_top_k=sample_top_k,
            constrain_decoding=constrain_decoding,
            palette_penalty=palette_penalty,
            shape_penalty=shape_penalty,
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
    if len(valid_at_1) == len(results["correct_at_1"]) and valid_at_1:
        results["fractional_at_1"] = sum(valid_at_1) / len(valid_at_1)
    else:
        results["fractional_at_1"] = None
    if len(valid_at_2) == len(results["correct_at_2"]) and valid_at_2:
        results["fractional_at_2"] = sum(valid_at_2) / len(valid_at_2)
    else:
        results["fractional_at_2"] = None

    return results


def evaluate_dataset(
    model: TinyTransformer,
    challenges: dict[str, dict],
    device: torch.device,
    example_id_map: dict[str, int] | None = None,
    use_train_examples: bool = True,
    solutions: dict[str, list[list[list[int]]]] | None = None,
    num_color_perms: int = 8,
    top_k: int = 2,
    max_output_tokens: int = 900,
    num_samples: int = 1,
    sample_temperature: float = 0.0,
    sample_top_k: int | None = None,
    constrain_decoding: bool = False,
    palette_penalty: float = 2.0,
    shape_penalty: float = 2.0,
    show_progress: bool = True,
    rank: int = 0,
    world_size: int = 1,
) -> dict:
    """
    Evaluate model on full dataset.

    Args:
        model: The transformer model
        challenges: Dict mapping task_id -> task
        device: Device to run on
        example_id_map: Optional task_id -> example_id mapping for conditioning
        use_train_examples: If False, prompt uses only the test input (per-pair prompt)
        solutions: Optional dict mapping task_id -> list of solutions
        num_color_perms: Number of color permutations for AAIVR
        top_k: Number of predictions per test
        max_output_tokens: Maximum tokens per prediction
        num_samples: Number of samples per augmentation (self-consistency)
        sample_temperature: Sampling temperature for self-consistency
        sample_top_k: If set and temperature > 0, sample from top-k logits
        constrain_decoding: If True, apply soft palette/shape constraints
        palette_penalty: Logit penalty for disallowed colors
        shape_penalty: Logit penalty for shape rule violations
        show_progress: Whether to show progress bar
        rank: Current process rank (for distributed eval)
        world_size: Total number of processes

    Returns:
        Dictionary with aggregated metrics and per-task results (includes arc_score_at_1/2)
    """
    model.eval()

    # Split tasks across GPUs
    all_task_ids = sorted(challenges.keys())
    my_task_ids = all_task_ids[rank::world_size]  # Round-robin assignment

    results = {
        "total_tasks": len(challenges),
        "my_tasks": len(my_task_ids),
        "correct_at_1": 0,
        "correct_at_2": 0,
        "total_test_examples": 0,
        "arc_score_at_1": 0.0,
        "arc_score_at_2": 0.0,
        "arc_tasks_scored": 0,
        "per_task_results": {},
    }

    task_iterator = [(tid, challenges[tid]) for tid in my_task_ids]
    if show_progress and rank == 0:
        task_iterator = tqdm(task_iterator, desc=f"Evaluating (GPU {rank})", total=len(my_task_ids))
    elif show_progress:
        task_iterator = tqdm(task_iterator, desc=f"GPU {rank}", total=len(my_task_ids), position=rank)

    correct_so_far = 0
    incorrect_so_far = 0
    evaluated_so_far = 0

    for task_id, task in task_iterator:
        task_solutions = solutions.get(task_id) if solutions else None

        task_result = evaluate_task(
            model,
            task,
            task_id,
            device,
            example_id_map=example_id_map,
            use_train_examples=use_train_examples,
            solutions=task_solutions,
            num_color_perms=num_color_perms,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            num_samples=num_samples,
            sample_temperature=sample_temperature,
            sample_top_k=sample_top_k,
            constrain_decoding=constrain_decoding,
            palette_penalty=palette_penalty,
            shape_penalty=shape_penalty,
        )

        results["per_task_results"][task_id] = task_result
        results["total_test_examples"] += len(task["test"])

        if task_result["pass_at_1"] is True:
            results["correct_at_1"] += 1
            correct_so_far += 1
            evaluated_so_far += 1
        elif task_result["pass_at_1"] is False:
            incorrect_so_far += 1
            evaluated_so_far += 1
        if task_result["pass_at_2"] is True:
            results["correct_at_2"] += 1

        if task_result["fractional_at_1"] is not None and task_result["fractional_at_2"] is not None:
            results["arc_score_at_1"] += task_result["fractional_at_1"]
            results["arc_score_at_2"] += task_result["fractional_at_2"]
            results["arc_tasks_scored"] += 1

        if show_progress and hasattr(task_iterator, "set_postfix"):
            total = evaluated_so_far if evaluated_so_far > 0 else 1
            acc = correct_so_far / total
            task_iterator.set_postfix(
                correct=correct_so_far,
                incorrect=incorrect_so_far,
                acc=f"{acc:.2%}",
            )

    # Compute accuracy (will be aggregated across ranks later)
    my_total = len(my_task_ids)
    results["accuracy_at_1"] = results["correct_at_1"] / my_total if my_total > 0 else 0.0
    results["accuracy_at_2"] = results["correct_at_2"] / my_total if my_total > 0 else 0.0
    if results["arc_tasks_scored"] > 0:
        results["arc_score_at_1"] = results["arc_score_at_1"] / results["arc_tasks_scored"]
        results["arc_score_at_2"] = results["arc_score_at_2"] / results["arc_tasks_scored"]
    else:
        results["arc_score_at_1"] = None
        results["arc_score_at_2"] = None

    return results


def generate_submission(
    model: TinyTransformer,
    challenges: dict[str, dict],
    output_path: Path,
    device: torch.device,
    example_id_map: dict[str, int] | None = None,
    use_train_examples: bool = True,
    num_color_perms: int = 8,
    top_k: int = 2,
    max_output_tokens: int = 900,
    num_samples: int = 1,
    sample_temperature: float = 0.0,
    sample_top_k: int | None = None,
    constrain_decoding: bool = False,
    palette_penalty: float = 2.0,
    shape_penalty: float = 2.0,
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
        example_id_map: Optional task_id -> example_id mapping for conditioning
        use_train_examples: If False, prompt uses only the test input (per-pair prompt)
        num_color_perms: Number of color permutations
        top_k: Number of attempts per test (max 2)
        max_output_tokens: Maximum tokens per prediction
        num_samples: Number of samples per augmentation (self-consistency)
        sample_temperature: Sampling temperature for self-consistency
        sample_top_k: If set and temperature > 0, sample from top-k logits
        constrain_decoding: If True, apply soft palette/shape constraints
        palette_penalty: Logit penalty for disallowed colors
        shape_penalty: Logit penalty for shape rule violations
        show_progress: Whether to show progress bar
    """
    model.eval()

    submission = {}

    task_iterator = challenges.items()
    if show_progress:
        task_iterator = tqdm(task_iterator, desc="Generating submission", total=len(challenges))

    for task_id, task in task_iterator:
        task_submissions = []
        example_id = example_id_map.get(task_id, 0) if example_id_map else 0

        for test_idx in range(len(task["test"])):
            test_input = task["test"][test_idx]["input"]

            predictions = predict_with_augmentations(
                model,
                task,
                test_idx,
                device,
                example_id=example_id,
                use_train_examples=use_train_examples,
                num_color_perms=num_color_perms,
                max_output_tokens=max_output_tokens,
                num_samples=num_samples,
                sample_temperature=sample_temperature,
                sample_top_k=sample_top_k,
                constrain_decoding=constrain_decoding,
                palette_penalty=palette_penalty,
                shape_penalty=shape_penalty,
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
    return_task_id_map: bool = False,
) -> TinyTransformer | tuple[TinyTransformer, dict[str, int] | None]:
    """
    Load a model from a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint .pt file
        device: Device to load model on
        return_task_id_map: If True, return task_id -> example_id mapping

    Returns:
        Loaded model (and optional example_id mapping)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    task_id_to_example_id = checkpoint.get("task_id_to_example_id")

    # Get model config from checkpoint - prefer model_config (TransformerConfig)
    if "model_config" in checkpoint:
        # Direct TransformerConfig dict - use it
        model = create_model(**checkpoint["model_config"])
    elif "config" in checkpoint:
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
        elif isinstance(config, dict) and "model_size" in config:
            # TrainConfig-style dict with model_size
            model_factories = {
                "tiny": create_tiny_model,
                "small": create_small_model,
                "medium": create_medium_model,
                "large": create_large_model,
            }
            factory = model_factories.get(config["model_size"], create_small_model)
            model = factory()
        else:
            model = create_model()
    else:
        # Default model
        model = create_model()

    # Load weights (check various key names used by different checkpoints)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        # Try loading directly (checkpoint IS the state dict)
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Model parameters: {model.num_parameters:,}")

    if return_task_id_map:
        return model, task_id_to_example_id
    return model


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Evaluate ARC model")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/small/best.pt"),
        help="Path to model checkpoint (default: checkpoints/small/best.pt)",
    )
    parser.add_argument(
        "--challenges",
        type=Path,
        default=None,
        help="Path to challenges JSON (default: ARC-AGI eval set)",
    )
    parser.add_argument(
        "--solutions",
        type=Path,
        default=None,
        help="Path to solutions JSON (default: ARC-AGI eval set)",
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
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples per augmentation (default: 1)",
    )
    parser.add_argument(
        "--sample-temp",
        type=float,
        default=0.0,
        help="Sampling temperature for self-consistency (default: 0.0)",
    )
    parser.add_argument(
        "--sample-top-k",
        type=int,
        default=0,
        help="Top-k sampling cutoff (default: 0, disabled)",
    )
    parser.add_argument(
        "--constrained-decoding",
        action="store_true",
        help="Enable soft palette/shape constraints during decoding",
    )
    parser.add_argument(
        "--palette-penalty",
        type=float,
        default=2.0,
        help="Logit penalty for disallowed colors (default: 2.0)",
    )
    parser.add_argument(
        "--shape-penalty",
        type=float,
        default=2.0,
        help="Logit penalty for shape rule violations (default: 2.0)",
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
    parser.add_argument(
        "--per-pair-prompt",
        action="store_true",
        help="Use only the test input as the prompt (no train examples)",
    )

    args = parser.parse_args()
    sample_top_k = args.sample_top_k if args.sample_top_k > 0 else None

    # Distributed setup (auto-detect torchrun)
    import os

    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    distributed = local_rank >= 0

    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        is_main = rank == 0
    else:
        rank = 0
        world_size = 1
        device = torch.device(args.device)
        is_main = True

    if is_main:
        logger.info(f"Using device: {device}" + (f" (distributed: {world_size} GPUs)" if distributed else ""))

    # Ensure eval data exists and get default paths (only on main rank)
    challenges_path = args.challenges
    solutions_path = args.solutions

    if challenges_path is None or solutions_path is None:
        if is_main:
            default_challenges, default_solutions = ensure_eval_data_exists()
        if distributed:
            dist.barrier()
        # All ranks use the default paths
        if challenges_path is None:
            challenges_path = DEFAULT_EVAL_CHALLENGES
        if solutions_path is None:
            solutions_path = DEFAULT_EVAL_SOLUTIONS

    # Load model
    model, task_id_to_example_id = load_model_from_checkpoint(
        args.checkpoint, device, return_task_id_map=True
    )

    # Load challenges
    with open(challenges_path) as f:
        challenges = json.load(f)
    if is_main:
        logger.info(f"Loaded {len(challenges)} challenges")

    # Load solutions
    with open(solutions_path) as f:
        solutions = json.load(f)
    if is_main:
        logger.info(f"Loaded {len(solutions)} solutions")

    # Generate submission if output path provided (single GPU only for now)
    if args.output and not distributed:
        generate_submission(
            model,
            challenges,
            args.output,
            device,
            example_id_map=task_id_to_example_id,
            use_train_examples=not args.per_pair_prompt,
            num_color_perms=args.num_color_perms,
            top_k=args.top_k,
            max_output_tokens=args.max_tokens,
            num_samples=args.num_samples,
            sample_temperature=args.sample_temp,
            sample_top_k=sample_top_k,
            constrain_decoding=args.constrained_decoding,
            palette_penalty=args.palette_penalty,
            shape_penalty=args.shape_penalty,
            show_progress=not args.quiet,
        )

    # Evaluate if solutions provided
    if solutions:
        results = evaluate_dataset(
            model,
            challenges,
            device,
            example_id_map=task_id_to_example_id,
            use_train_examples=not args.per_pair_prompt,
            solutions=solutions,
            num_color_perms=args.num_color_perms,
            top_k=args.top_k,
            max_output_tokens=args.max_tokens,
            num_samples=args.num_samples,
            sample_temperature=args.sample_temp,
            sample_top_k=sample_top_k,
            constrain_decoding=args.constrained_decoding,
            palette_penalty=args.palette_penalty,
            shape_penalty=args.shape_penalty,
            show_progress=not args.quiet,
            rank=rank,
            world_size=world_size,
        )

        # Gather results from all ranks
        if distributed:
            dist.barrier()
            # Gather counts from all ranks
            correct_1 = torch.tensor([results["correct_at_1"]], device=device)
            correct_2 = torch.tensor([results["correct_at_2"]], device=device)
            test_examples = torch.tensor([results["total_test_examples"]], device=device)
            arc_score_1 = torch.tensor(
                [(results["arc_score_at_1"] or 0.0) * results["arc_tasks_scored"]],
                device=device,
            )
            arc_score_2 = torch.tensor(
                [(results["arc_score_at_2"] or 0.0) * results["arc_tasks_scored"]],
                device=device,
            )
            arc_tasks = torch.tensor([results["arc_tasks_scored"]], device=device)

            dist.all_reduce(correct_1, op=dist.ReduceOp.SUM)
            dist.all_reduce(correct_2, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_examples, op=dist.ReduceOp.SUM)
            dist.all_reduce(arc_score_1, op=dist.ReduceOp.SUM)
            dist.all_reduce(arc_score_2, op=dist.ReduceOp.SUM)
            dist.all_reduce(arc_tasks, op=dist.ReduceOp.SUM)

            results["correct_at_1"] = correct_1.item()
            results["correct_at_2"] = correct_2.item()
            results["total_test_examples"] = test_examples.item()
            results["accuracy_at_1"] = results["correct_at_1"] / results["total_tasks"]
            results["accuracy_at_2"] = results["correct_at_2"] / results["total_tasks"]
            results["arc_tasks_scored"] = arc_tasks.item()
            if results["arc_tasks_scored"] > 0:
                results["arc_score_at_1"] = arc_score_1.item() / results["arc_tasks_scored"]
                results["arc_score_at_2"] = arc_score_2.item() / results["arc_tasks_scored"]
            else:
                results["arc_score_at_1"] = None
                results["arc_score_at_2"] = None

        # Print results (main rank only)
        if is_main:
            print("\n" + "=" * 50)
            print("EVALUATION RESULTS")
            print("=" * 50)
            print(f"Total tasks: {results['total_tasks']}")
            print(f"Total test examples: {results['total_test_examples']}")
            print(f"Correct @ 1: {results['correct_at_1']} ({results['accuracy_at_1']:.2%})")
            print(f"Correct @ 2: {results['correct_at_2']} ({results['accuracy_at_2']:.2%})")
            if results.get("arc_score_at_1") is not None and results.get("arc_score_at_2") is not None:
                print(f"ARC Score @ 1: {results['arc_score_at_1']:.2%}")
                print(f"ARC Score @ 2: {results['arc_score_at_2']:.2%}")
            print("=" * 50)

            # Save detailed results
            if args.output:
                results_path = args.output.with_suffix(".results.json")
                with open(results_path, "w") as f:
                    # Remove per-task details for summary file
                    summary = {k: v for k, v in results.items() if k != "per_task_results"}
                    json.dump(summary, f, indent=2)
                logger.info(f"Results saved to {results_path}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

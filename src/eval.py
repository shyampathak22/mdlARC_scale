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

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data import END_TOKEN_ID, compute_positions_3d, encode_example
from src.inference import predict_with_augmentations
from src.train import TrainConfig, compute_loss, create_output_mask
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


def _build_tta_batch(
    task: dict,
    device: torch.device,
    example_id: int,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    tokens_list = []
    positions_list = []

    for pair in task.get("train", []):
        tokens = encode_example(pair["input"], pair["output"])
        if len(tokens) > max_seq_len:
            continue
        tokens_list.append(tokens)
        positions_list.append(compute_positions_3d(tokens))

    if not tokens_list:
        return None

    max_len = max(len(tokens) for tokens in tokens_list)
    batch_size = len(tokens_list)

    input_ids = torch.full(
        (batch_size, max_len),
        END_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    positions_3d = torch.zeros(batch_size, max_len, 3, dtype=torch.long, device=device)
    example_ids = torch.full((batch_size,), example_id, dtype=torch.long, device=device)

    for i, tokens in enumerate(tokens_list):
        seq_len = len(tokens)
        input_ids[i, :seq_len] = torch.tensor(tokens, dtype=torch.long, device=device)
        attention_mask[i, :seq_len] = True
        positions_3d[i, :seq_len] = torch.tensor(positions_list[i], dtype=torch.long, device=device)

    return input_ids, positions_3d, attention_mask, example_ids


def _adapt_example_embedding(
    model: TinyTransformer,
    task: dict,
    device: torch.device,
    example_id: int,
    steps: int,
    lr: float,
    weight_decay: float,
) -> torch.Tensor | None:
    if steps <= 0:
        return None
    if not getattr(model.config, "use_example_embedding", True):
        return None
    if not hasattr(model, "example_embed"):
        return None
    if example_id < 0 or example_id >= model.example_embed.num_embeddings:
        return None

    batch = _build_tta_batch(task, device, example_id, model.config.max_seq_len)
    if batch is None:
        return None
    input_ids, positions_3d, attention_mask, example_ids = batch

    original = model.example_embed.weight.data[example_id].detach().clone()
    params = list(model.parameters())
    requires_grad = [p.requires_grad for p in params]

    for p in params:
        p.requires_grad_(False)
    model.example_embed.weight.requires_grad_(True)

    was_training = model.training
    model.eval()

    optimizer = torch.optim.Adam([model.example_embed.weight], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(
            input_ids,
            positions_3d,
            example_ids=example_ids,
            attention_mask=attention_mask,
        )
        output_mask = create_output_mask(input_ids)
        losses = compute_loss(
            logits=logits,
            labels=input_ids,
            attention_mask=attention_mask,
            output_mask=output_mask,
            input_loss_weight=0.0,
            output_loss_weight=1.0,
            uniform_weight=False,
        )
        loss = losses["loss"]
        loss.backward()
        optimizer.step()

        if weight_decay > 0.0:
            with torch.no_grad():
                decay = 1.0 - lr * weight_decay
                model.example_embed.weight.data[example_id].mul_(decay)

    for p, req in zip(params, requires_grad):
        p.requires_grad_(req)

    if was_training:
        model.train()
    else:
        model.eval()

    return original


def _build_tta_batch_for_tasks(
    task_items: list[tuple[str, dict]],
    example_id_map: dict[str, int] | None,
    device: torch.device,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    tokens_list: list[list[int]] = []
    positions_list: list[np.ndarray] = []
    example_ids_list: list[int] = []

    if example_id_map is None:
        return None

    for task_id, task in task_items:
        example_id = example_id_map.get(task_id)
        if example_id is None:
            continue
        for pair in task.get("train", []):
            tokens = encode_example(pair["input"], pair["output"])
            if len(tokens) > max_seq_len:
                continue
            tokens_list.append(tokens)
            positions_list.append(compute_positions_3d(tokens))
            example_ids_list.append(example_id)

    if not tokens_list:
        return None

    max_len = max(len(tokens) for tokens in tokens_list)
    batch_size = len(tokens_list)

    input_ids = torch.full(
        (batch_size, max_len),
        END_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    positions_3d = torch.zeros(batch_size, max_len, 3, dtype=torch.long, device=device)
    example_ids = torch.tensor(example_ids_list, dtype=torch.long, device=device)

    for i, tokens in enumerate(tokens_list):
        seq_len = len(tokens)
        input_ids[i, :seq_len] = torch.tensor(tokens, dtype=torch.long, device=device)
        attention_mask[i, :seq_len] = True
        positions_3d[i, :seq_len] = torch.tensor(positions_list[i], dtype=torch.long, device=device)

    return input_ids, positions_3d, attention_mask, example_ids


def _compute_tta_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    output_mask: torch.Tensor,
    example_ids: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len, vocab_size = logits.shape

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:]
    shift_output_mask = output_mask[:, 1:]

    valid_mask = shift_mask & shift_output_mask
    flat_logits = shift_logits.view(-1, vocab_size)
    flat_labels = shift_labels.view(-1)
    per_token_loss = F.cross_entropy(flat_logits, flat_labels, reduction="none").view(batch_size, seq_len - 1)
    per_token_loss = per_token_loss * valid_mask.float()

    token_sum = per_token_loss.sum(dim=1)
    token_count = valid_mask.sum(dim=1).float()

    unique_ids, inverse = torch.unique(example_ids, return_inverse=True)
    sum_per_task = torch.zeros_like(unique_ids, dtype=token_sum.dtype)
    count_per_task = torch.zeros_like(unique_ids, dtype=token_count.dtype)
    sum_per_task.scatter_add_(0, inverse, token_sum)
    count_per_task.scatter_add_(0, inverse, token_count)

    task_loss = sum_per_task / count_per_task.clamp(min=1.0)
    return task_loss.mean()


def _adapt_example_embeddings_batched(
    model: TinyTransformer,
    task_items: list[tuple[str, dict]],
    example_id_map: dict[str, int] | None,
    device: torch.device,
    steps: int,
    lr: float,
    weight_decay: float,
) -> dict[int, torch.Tensor]:
    if steps <= 0:
        return {}
    if not getattr(model.config, "use_example_embedding", True):
        return {}
    if not hasattr(model, "example_embed"):
        return {}

    batch = _build_tta_batch_for_tasks(task_items, example_id_map, device, model.config.max_seq_len)
    if batch is None:
        return {}
    input_ids, positions_3d, attention_mask, example_ids = batch

    unique_ids = torch.unique(example_ids)
    original = {int(idx): model.example_embed.weight.data[idx].detach().clone() for idx in unique_ids}

    params = list(model.parameters())
    requires_grad = [p.requires_grad for p in params]

    for p in params:
        p.requires_grad_(False)
    model.example_embed.weight.requires_grad_(True)

    was_training = model.training
    model.eval()

    optimizer = torch.optim.Adam([model.example_embed.weight], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(
            input_ids,
            positions_3d,
            example_ids=example_ids,
            attention_mask=attention_mask,
        )
        output_mask = create_output_mask(input_ids)
        loss = _compute_tta_loss(logits, input_ids, attention_mask, output_mask, example_ids)
        loss.backward()
        optimizer.step()

        if weight_decay > 0.0:
            with torch.no_grad():
                decay = 1.0 - lr * weight_decay
                model.example_embed.weight.data[unique_ids].mul_(decay)

    for p, req in zip(params, requires_grad):
        p.requires_grad_(req)

    if was_training:
        model.train()
    else:
        model.eval()

    return original


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
    tta_steps: int = 0,
    tta_lr: float = 0.0,
    tta_weight_decay: float = 0.0,
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
        tta_steps: Number of test-time adaptation steps (0 disables)
        tta_lr: Learning rate for test-time adaptation
        tta_weight_decay: Weight decay for test-time adaptation

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

    example_id = example_id_map.get(task_id) if example_id_map else 0
    if example_id is None:
        example_id = 0
    num_tests = len(task["test"])

    original_embed = None
    if tta_steps > 0 and example_id_map and example_id_map.get(task_id) is not None:
        original_embed = _adapt_example_embedding(
            model=model,
            task=task,
            device=device,
            example_id=example_id,
            steps=tta_steps,
            lr=tta_lr,
            weight_decay=tta_weight_decay,
        )

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

    if original_embed is not None:
        with torch.no_grad():
            model.example_embed.weight.data[example_id].copy_(original_embed)

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
    tta_steps: int = 0,
    tta_lr: float = 0.0,
    tta_weight_decay: float = 0.0,
    tta_batch_tasks: int = 1,
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
        tta_steps: Number of test-time adaptation steps (0 disables)
        tta_lr: Learning rate for test-time adaptation
        tta_weight_decay: Weight decay for test-time adaptation
        tta_batch_tasks: Number of tasks to adapt in a single TTA batch
        show_progress: Whether to show progress bar
        rank: Current process rank (for distributed eval)
        world_size: Total number of processes

    Returns:
        Dictionary with aggregated metrics and per-task results (includes arc_score_at_1/2)
    """
    model.eval()
    tta_batch_tasks = max(1, tta_batch_tasks)

    # Split tasks across GPUs
    all_task_ids = sorted(challenges.keys())
    my_task_ids = all_task_ids[rank::world_size]  # Round-robin assignment

    if tta_steps > 0:
        local_map = dict(example_id_map) if example_id_map else {}
        next_id = max(local_map.values(), default=-1) + 1
        max_examples = getattr(model.config, "num_examples", None)
        for task_id in my_task_ids:
            if task_id in local_map:
                continue
            if max_examples is None or next_id < max_examples:
                local_map[task_id] = next_id
                next_id += 1
            else:
                local_map[task_id] = None
        example_id_map = local_map

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

    task_items = [(tid, challenges[tid]) for tid in my_task_ids]
    pbar = None
    if show_progress:
        desc = f"Evaluating (GPU {rank})" if rank == 0 else f"GPU {rank}"
        pbar_kwargs = {"desc": desc, "total": len(task_items)}
        if rank != 0:
            pbar_kwargs["position"] = rank
        pbar = tqdm(**pbar_kwargs)

    correct_so_far = 0
    incorrect_so_far = 0
    evaluated_so_far = 0

    def _record_task(task_id: str, task: dict, tta_steps_override: int) -> None:
        nonlocal correct_so_far, incorrect_so_far, evaluated_so_far

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
            tta_steps=tta_steps_override,
            tta_lr=tta_lr,
            tta_weight_decay=tta_weight_decay,
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

        if pbar is not None:
            total = evaluated_so_far if evaluated_so_far > 0 else 1
            acc = correct_so_far / total
            pbar.set_postfix(
                correct=correct_so_far,
                incorrect=incorrect_so_far,
                acc=f"{acc:.2%}",
            )
            pbar.update(1)

    if tta_steps > 0 and tta_batch_tasks > 1:
        for idx in range(0, len(task_items), tta_batch_tasks):
            batch_items = task_items[idx : idx + tta_batch_tasks]
            original_embeds = _adapt_example_embeddings_batched(
                model=model,
                task_items=batch_items,
                example_id_map=example_id_map,
                device=device,
                steps=tta_steps,
                lr=tta_lr,
                weight_decay=tta_weight_decay,
            )
            for task_id, task in batch_items:
                _record_task(task_id, task, tta_steps_override=0)
            if original_embeds:
                with torch.no_grad():
                    for example_id, embed in original_embeds.items():
                        model.example_embed.weight.data[example_id].copy_(embed)
    else:
        for task_id, task in task_items:
            _record_task(task_id, task, tta_steps_override=tta_steps)

    if pbar is not None:
        pbar.close()

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
    tta_steps: int = 0,
    tta_lr: float = 0.0,
    tta_weight_decay: float = 0.0,
    tta_batch_tasks: int = 1,
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
        tta_steps: Number of test-time adaptation steps (0 disables)
        tta_lr: Learning rate for test-time adaptation
        tta_weight_decay: Weight decay for test-time adaptation
        tta_batch_tasks: Number of tasks to adapt in a single TTA batch
        show_progress: Whether to show progress bar
    """
    model.eval()
    tta_batch_tasks = max(1, tta_batch_tasks)

    submission = {}

    if tta_steps > 0:
        local_map = dict(example_id_map) if example_id_map else {}
        next_id = max(local_map.values(), default=-1) + 1
        max_examples = getattr(model.config, "num_examples", None)
        for task_id in challenges.keys():
            if task_id in local_map:
                continue
            if max_examples is None or next_id < max_examples:
                local_map[task_id] = next_id
                next_id += 1
            else:
                local_map[task_id] = None
        example_id_map = local_map

    task_items = list(challenges.items())
    pbar = None
    if show_progress:
        pbar = tqdm(total=len(task_items), desc="Generating submission")

    def _generate_task(task_id: str, task: dict, use_tta: bool) -> None:
        task_submissions = []
        example_id = example_id_map.get(task_id) if example_id_map else 0
        if example_id is None:
            example_id = 0

        original_embed = None
        if use_tta and tta_steps > 0 and example_id_map and example_id_map.get(task_id) is not None:
            original_embed = _adapt_example_embedding(
                model=model,
                task=task,
                device=device,
                example_id=example_id,
                steps=tta_steps,
                lr=tta_lr,
                weight_decay=tta_weight_decay,
            )

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

        if original_embed is not None:
            with torch.no_grad():
                model.example_embed.weight.data[example_id].copy_(original_embed)
        if pbar is not None:
            pbar.update(1)

    if tta_steps > 0 and tta_batch_tasks > 1:
        for idx in range(0, len(task_items), tta_batch_tasks):
            batch_items = task_items[idx : idx + tta_batch_tasks]
            original_embeds = _adapt_example_embeddings_batched(
                model=model,
                task_items=batch_items,
                example_id_map=example_id_map,
                device=device,
                steps=tta_steps,
                lr=tta_lr,
                weight_decay=tta_weight_decay,
            )
            for task_id, task in batch_items:
                _generate_task(task_id, task, use_tta=False)
            if original_embeds:
                with torch.no_grad():
                    for example_id, embed in original_embeds.items():
                        model.example_embed.weight.data[example_id].copy_(embed)
    else:
        for task_id, task in task_items:
            _generate_task(task_id, task, use_tta=True)

    if pbar is not None:
        pbar.close()

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
        "--tta-steps",
        type=int,
        default=0,
        help="Test-time adaptation steps per task (default: 0)",
    )
    parser.add_argument(
        "--tta-lr",
        type=float,
        default=0.1,
        help="Learning rate for test-time adaptation (default: 0.1)",
    )
    parser.add_argument(
        "--tta-weight-decay",
        type=float,
        default=0.0,
        help="Weight decay for test-time adaptation (default: 0.0)",
    )
    parser.add_argument(
        "--tta-batch-tasks",
        type=int,
        default=1,
        help="Number of tasks to adapt in a single TTA batch (default: 1)",
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
            tta_steps=args.tta_steps,
            tta_lr=args.tta_lr,
            tta_weight_decay=args.tta_weight_decay,
            tta_batch_tasks=args.tta_batch_tasks,
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
            tta_steps=args.tta_steps,
            tta_lr=args.tta_lr,
            tta_weight_decay=args.tta_weight_decay,
            tta_batch_tasks=args.tta_batch_tasks,
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

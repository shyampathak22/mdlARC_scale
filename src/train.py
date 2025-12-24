"""
Training loop for mdlARC reproduction.

Key features:
- Separate input/output loss tracking
- AdamW with selective weight decay
- Linear warmup + cosine decay LR schedule
- BF16 autocast on CUDA
- torch.compile() for performance
- Checkpointing with RNG state for reproducibility
"""

import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.data import (
    IO_SEP_TOKEN_ID,
    ARCDataset,
    ColorAugmentor,
    apply_color_perm_to_tokens,
    create_dataloader,
    generate_color_permutations,
)
from src.transformer import (
    TinyTransformer,
    create_large_model,
    create_medium_model,
    create_small_model,
    create_tiny_model,
)

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class TrainConfig:
    """Training configuration."""

    # Model
    model_size: str = "small"  # tiny, small, medium, large
    rope_type: str = "3d"  # 3d, 1d, none
    norm_type: str = "rmsnorm"  # rmsnorm, layernorm
    ffn_type: str = "swiglu"  # swiglu, gelu
    use_example_embedding: bool = True

    # Data
    data_path: str = "data/challenges.json"
    # WARNING: solutions_path should ONLY contain solutions for the TRAINING set.
    # If this file contains eval set solutions, they will leak into training!
    # Use arc_training_solutions.json, NOT arc_public_evaluation_solutions.json
    solutions_path: str | None = None  # Solutions for training set test pairs
    batch_size: int = 8  # Per-GPU batch size (lower for memory)
    effective_batch_size: int = 32  # Target batch size (via gradient accumulation)
    val_batch_size: int = 16
    max_seq_len: int = 1863  # Matches original mdlARC
    num_workers: int = 4
    include_concept: bool = True  # Include ConceptARC in training data
    train_on_test_split: bool = True  # Include train-task test pairs in training

    # Augmentation
    apply_dihedral: bool = True
    num_color_perms: int = 100
    color_seed: int = 42

    # Optimization
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_fraction: float = 0.05

    # Training
    epochs: int = 100
    log_every: int = 10  # steps
    val_every: int = 1  # epochs

    # Hardware
    compile_model: bool = True
    use_amp: bool = True  # BF16 autocast

    # Distributed training
    use_ddp: bool = False  # Enable DistributedDataParallel
    local_rank: int = -1  # Set by torchrun

    # Checkpointing
    save_dir: str = "checkpoints"
    save_every: int = 10  # epochs
    resume_from: str | None = None

    # Loss weighting
    uniform_loss_weight: bool = True  # If True, all tokens weighted equally (original mdlARC)
    input_loss_weight: float = 1.0  # Weight for input portion (if uniform_loss_weight=False)
    output_loss_weight: float = 1.0  # Weight for output portion (if uniform_loss_weight=False)

    # Logging
    use_wandb: bool = True  # Enable W&B logging
    wandb_project: str = "mdlARC_scale"  # W&B project name
    wandb_run_name: str | None = None  # Custom run name (auto-generated if None)
    wandb_log_every: int = 50  # Log to wandb every N steps
    ema_decay: float = 0.99  # EMA decay for smoothed loss

    # MFU tracking
    # Peak TFLOPS for GPU (BF16 performance)
    peak_tflops: float = 23.7  # Default for RTX 5060 Ti (BF16)

    # Misc
    seed: int = 42

    def get_run_name(self) -> str:
        """Generate a run name from hyperparameters."""
        if self.wandb_run_name is not None:
            return self.wandb_run_name
        return f"{self.model_size}_bs{self.effective_batch_size}_lr{self.lr:.0e}"

    @property
    def grad_accum_steps(self) -> int:
        """Compute gradient accumulation steps to reach effective batch size."""
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        return max(1, self.effective_batch_size // (self.batch_size * world_size))

    @classmethod
    def from_toml(cls, path: str | Path) -> "TrainConfig":
        """Load config from a TOML file."""
        import tomllib

        path = Path(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Flatten nested TOML structure to match dataclass fields
        flat = {}

        # [model] section
        if "model" in data:
            flat["model_size"] = data["model"].get("size", "small")
            flat["rope_type"] = data["model"].get("rope_type", "3d")
            flat["norm_type"] = data["model"].get("norm_type", "rmsnorm")
            flat["ffn_type"] = data["model"].get("ffn_type", "swiglu")
            flat["use_example_embedding"] = data["model"].get("use_example_embedding", True)

        # [data] section
        if "data" in data:
            flat["data_path"] = data["data"].get("path", "data/challenges.json")
            flat["solutions_path"] = data["data"].get("solutions_path")
            flat["max_seq_len"] = data["data"].get("max_seq_len", 1863)
            flat["include_concept"] = data["data"].get("include_concept", True)
            flat["train_on_test_split"] = data["data"].get("train_on_test_split", True)

        # [training] section
        if "training" in data:
            t = data["training"]
            flat["batch_size"] = t.get("batch_size", 8)
            flat["effective_batch_size"] = t.get("effective_batch_size", 32)
            flat["val_batch_size"] = t.get("val_batch_size", 16)
            flat["epochs"] = t.get("epochs", 100)
            flat["lr"] = t.get("lr", 3e-4)
            flat["weight_decay"] = t.get("weight_decay", 0.01)
            flat["grad_clip"] = t.get("grad_clip", 1.0)
            flat["warmup_fraction"] = t.get("warmup_fraction", 0.05)

        # [augmentation] section
        if "augmentation" in data:
            a = data["augmentation"]
            flat["apply_dihedral"] = a.get("apply_dihedral", True)
            flat["num_color_perms"] = a.get("num_color_perms", 100)
            flat["color_seed"] = a.get("color_seed", 42)

        # [hardware] section
        if "hardware" in data:
            h = data["hardware"]
            flat["compile_model"] = h.get("compile_model", True)
            flat["use_amp"] = h.get("use_amp", True)
            flat["num_workers"] = h.get("num_workers", 4)

        # [checkpointing] section
        if "checkpointing" in data:
            c = data["checkpointing"]
            flat["save_dir"] = c.get("save_dir", "checkpoints")
            flat["save_every"] = c.get("save_every", 10)
            flat["resume_from"] = c.get("resume_from")

        # [logging] section
        if "logging" in data:
            lg = data["logging"]
            flat["use_wandb"] = lg.get("use_wandb", True)
            flat["wandb_project"] = lg.get("wandb_project", "mdlARC_scale")
            flat["wandb_run_name"] = lg.get("wandb_run_name")
            flat["log_every"] = lg.get("log_every", 10)
            flat["wandb_log_every"] = lg.get("wandb_log_every", 50)
            flat["ema_decay"] = lg.get("ema_decay", 0.99)

        # [loss] section
        if "loss" in data:
            lo = data["loss"]
            flat["uniform_loss_weight"] = lo.get("uniform_loss_weight", True)
            flat["input_loss_weight"] = lo.get("input_loss_weight", 1.0)
            flat["output_loss_weight"] = lo.get("output_loss_weight", 1.0)

        # [misc] section
        if "misc" in data:
            m = data["misc"]
            flat["seed"] = m.get("seed", 42)
            flat["peak_tflops"] = m.get("peak_tflops", 23.7)

        return cls(**flat)


# =============================================================================
# LOSS COMPUTATION
# =============================================================================


def create_output_mask(
    input_ids: torch.Tensor,
    io_sep_token_id: int = IO_SEP_TOKEN_ID,
) -> torch.Tensor:
    """
    Create a mask that is True for tokens AFTER the IO separator.

    This is used to separate input loss from output loss during training.
    We primarily care about the model's ability to predict output tokens.

    Args:
        input_ids: [batch, seq] token IDs
        io_sep_token_id: The separator token ID

    Returns:
        output_mask: [batch, seq] bool tensor (True = output token)
    """
    # Find separator positions
    is_sep = input_ids == io_sep_token_id

    # Cumsum marks everything at/after separator
    cumsum = is_sep.long().cumsum(dim=1)

    # Output mask: after separator (cumsum > 0) but not the separator itself
    output_mask = (cumsum > 0) & ~is_sep

    return output_mask


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    output_mask: torch.Tensor,
    input_loss_weight: float = 1.0,
    output_loss_weight: float = 1.0,
    uniform_weight: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Compute cross-entropy loss with separate input/output tracking.

    Uses next-token prediction: logits[:, t] predicts labels[:, t+1]

    The original mdlARC uses uniform token weighting (all tokens contribute equally
    to the loss), which is controlled by `uniform_weight=True`.

    Args:
        logits: [batch, seq, vocab] model predictions
        labels: [batch, seq] target tokens (same as input_ids for LM)
        attention_mask: [batch, seq] valid token mask
        output_mask: [batch, seq] mask for output tokens
        input_loss_weight: Weight for input portion (only used if uniform_weight=False)
        output_loss_weight: Weight for output portion (only used if uniform_weight=False)
        uniform_weight: If True, all tokens weighted equally (matches original mdlARC)

    Returns:
        Dict with 'loss', 'input_loss', 'output_loss'
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Shift for next-token prediction
    # logits[:, :-1] predicts labels[:, 1:]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].float()
    shift_output_mask = output_mask[:, 1:].float()

    # Flatten for loss computation
    flat_logits = shift_logits.view(-1, vocab_size)
    flat_labels = shift_labels.view(-1)
    flat_mask = shift_mask.view(-1)
    flat_output_mask = shift_output_mask.view(-1)

    # Per-token cross-entropy loss
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fct(flat_logits, flat_labels)

    # Apply attention mask (ignore padding)
    per_token_loss = per_token_loss * flat_mask

    # Separate input and output losses (for monitoring)
    input_mask = flat_mask * (1 - flat_output_mask)  # Valid tokens that are NOT output
    output_mask_final = flat_mask * flat_output_mask  # Valid tokens that ARE output

    # Compute mean losses (avoid div by zero)
    input_token_count = input_mask.sum().clamp(min=1)
    output_token_count = output_mask_final.sum().clamp(min=1)
    total_token_count = flat_mask.sum().clamp(min=1)

    input_loss = (per_token_loss * input_mask).sum() / input_token_count
    output_loss = (per_token_loss * output_mask_final).sum() / output_token_count

    # Combined loss
    if uniform_weight:
        # Original mdlARC: all tokens weighted equally
        total_loss = per_token_loss.sum() / total_token_count
    else:
        # Weighted combination of input/output losses
        total_loss = input_loss_weight * input_loss + output_loss_weight * output_loss

    return {
        "loss": total_loss,
        "input_loss": input_loss,
        "output_loss": output_loss,
    }


# =============================================================================
# OPTIMIZER SETUP
# =============================================================================


def build_param_groups(
    model: nn.Module,
    weight_decay: float,
) -> list[dict]:
    """
    Create parameter groups with selective weight decay.

    Weight decay is applied to:
    - Linear layer weights (excluding QKV projections in attention)

    No weight decay for:
    - Biases
    - LayerNorm parameters
    - Embedding weights
    - Attention-related parameters

    Args:
        model: The model
        weight_decay: Weight decay value for applicable parameters

    Returns:
        List of parameter group dicts for optimizer
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No decay for biases
        if "bias" in name:
            no_decay_params.append(param)
        # No decay for LayerNorm
        elif "ln" in name or "layer_norm" in name or "LayerNorm" in name:
            no_decay_params.append(param)
        # No decay for embeddings
        elif "embed" in name:
            no_decay_params.append(param)
        # No decay for attention projections (qkv, out)
        elif "attn" in name or "attention" in name:
            no_decay_params.append(param)
        # Decay for other weights (FFN, LM head)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def create_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> AdamW:
    """Create AdamW optimizer with selective weight decay."""
    param_groups = build_param_groups(model, weight_decay)

    # Use fused AdamW on CUDA for performance
    use_fused = device.type == "cuda" and "fused" in AdamW.__init__.__code__.co_varnames

    return AdamW(param_groups, lr=lr, fused=use_fused)


# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================


def create_scheduler(
    optimizer: AdamW,
    total_steps: int,
    warmup_fraction: float = 0.05,
) -> LambdaLR:
    """
    Create linear warmup + cosine decay learning rate scheduler.

    Args:
        optimizer: The optimizer
        total_steps: Total number of training steps
        warmup_fraction: Fraction of steps for linear warmup (default 5%)

    Returns:
        LambdaLR scheduler
    """
    warmup_steps = int(total_steps * warmup_fraction)

    def lr_lambda(current_step: int) -> float:
        # Linear warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Cosine decay
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# TRAINING LOOP
# =============================================================================


def estimate_model_flops(model: nn.Module) -> int:
    """
    Estimate FLOPs per token for the model.

    For transformers, the approximate FLOPs per token are:
    - 2 * num_params (multiply-add operations)

    For training (forward + backward + gradient update):
    - ~6 * num_params per token

    Returns FLOPs per token for training (6x model params).
    """
    # Get base model (unwrap DDP/compile)
    base_model = model
    if hasattr(model, "_orig_mod"):
        base_model = model._orig_mod
    if hasattr(base_model, "module"):
        base_model = base_model.module

    num_params = sum(p.numel() for p in base_model.parameters())
    # 6x for training: 2x forward, 2x backward, 2x gradient
    return 6 * num_params


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
    epoch: int,
    config: TrainConfig,
    color_augmentor: ColorAugmentor | None = None,
    scaler: torch.amp.GradScaler | None = None,
    wandb_run: object | None = None,  # wandb.sdk.wandb_run.Run
    global_step: int = 0,
) -> dict[str, float]:
    """
    Train for one epoch with gradient accumulation and MFU tracking.

    Args:
        model: The model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch: Current epoch number
        config: Training configuration
        color_augmentor: Optional ColorAugmentor for deterministic per-epoch augmentation
        scaler: Optional GradScaler for mixed precision
        wandb_run: Optional wandb run for step-level logging
        global_step: Current global step for wandb logging

    Returns:
        Dict with training metrics
    """
    model.train()

    total_loss = 0.0
    total_input_loss = 0.0
    total_output_loss = 0.0
    num_batches = 0
    num_tokens = 0

    # Accuracy tracking
    total_output_correct = 0
    total_output_tokens = 0

    # EMA tracking
    ema_loss = None
    ema_output_loss = None
    ema_accuracy = None
    ema_decay = config.ema_decay

    start_time = time.perf_counter()

    use_amp = config.use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    # Gradient accumulation setup
    grad_accum_steps = config.grad_accum_steps
    is_rank_zero = not config.use_ddp or config.local_rank == 0

    # MFU calculation setup
    flops_per_token = estimate_model_flops(model) if config.peak_tflops > 0 else 0
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    peak_flops = config.peak_tflops * 1e12 * world_size  # Total peak across all GPUs

    active_perm = None
    if color_augmentor is not None and color_augmentor.num_permutations > 0:
        active_perm = color_augmentor.perms[color_augmentor.current_index].to(device)

    # Zero gradients at start
    optimizer.zero_grad()

    # Progress bar (only on rank 0)
    pbar = None
    if is_rank_zero:
        pbar = tqdm(
            total=len(dataloader),
            desc=f"Epoch {epoch + 1}",
            unit="batch",
            leave=True,
            dynamic_ncols=True,
        )

    # For MFU tracking
    mfu_window_tokens = 0
    mfu_window_start = time.perf_counter()
    current_mfu = 0.0

    for step, batch in enumerate(dataloader):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        positions_3d = batch["positions_3d"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        example_ids = batch["example_ids"].to(device)

        # Apply deterministic color augmentation (per-epoch schedule)
        if active_perm is not None:
            splits = batch.get("splits")
            if splits is not None:
                should_aug = torch.tensor(
                    [
                        (color_augmentor.apply_to_test_split or split != "test")
                        for split in splits
                    ],
                    device=device,
                    dtype=torch.bool,
                )
                if should_aug.any():
                    permuted = apply_color_perm_to_tokens(input_ids, active_perm)
                    input_ids = torch.where(should_aug.unsqueeze(1), permuted, input_ids)
            else:
                input_ids = apply_color_perm_to_tokens(input_ids, active_perm)

        # Create output mask for loss separation
        output_mask = create_output_mask(input_ids)

        # Determine if this is an accumulation step (don't sync gradients)
        is_accumulating = (step + 1) % grad_accum_steps != 0

        # Context manager for gradient sync control in DDP
        # Only sync gradients on the last accumulation step
        if config.use_ddp and is_accumulating:
            ctx = model.no_sync()
        else:
            ctx = nullcontext()

        with ctx:
            # Forward pass with autocast
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(
                    input_ids, positions_3d,
                    example_ids=example_ids,
                    attention_mask=attention_mask,
                )

                losses = compute_loss(
                    logits=logits,
                    labels=input_ids,
                    attention_mask=attention_mask,
                    output_mask=output_mask,
                    input_loss_weight=config.input_loss_weight,
                    output_loss_weight=config.output_loss_weight,
                    uniform_weight=config.uniform_loss_weight,
                )
                # Scale loss by accumulation steps
                loss = losses["loss"] / grad_accum_steps

            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # Accumulate metrics (use unscaled loss for logging)
        batch_tokens = attention_mask.sum().item()
        batch_loss = losses["loss"].item()
        batch_output_loss = losses["output_loss"].item()

        # Compute per-pixel accuracy on output tokens
        with torch.no_grad():
            preds = logits.argmax(dim=-1)  # [B, S]
            output_valid = output_mask & attention_mask  # Only count valid output tokens
            correct = (preds == input_ids) & output_valid
            batch_output_correct = correct.sum().item()
            batch_output_tokens = output_valid.sum().item()
            batch_accuracy = batch_output_correct / max(batch_output_tokens, 1)

        total_loss += batch_loss * batch_tokens
        total_input_loss += losses["input_loss"].item() * batch_tokens
        total_output_loss += batch_output_loss * batch_tokens
        total_output_correct += batch_output_correct
        total_output_tokens += batch_output_tokens
        num_tokens += batch_tokens
        num_batches += 1
        mfu_window_tokens += batch_tokens

        # Update EMA
        if ema_loss is None:
            ema_loss = batch_loss
            ema_output_loss = batch_output_loss
            ema_accuracy = batch_accuracy
        else:
            ema_loss = ema_decay * ema_loss + (1 - ema_decay) * batch_loss
            ema_output_loss = ema_decay * ema_output_loss + (1 - ema_decay) * batch_output_loss
            ema_accuracy = ema_decay * ema_accuracy + (1 - ema_decay) * batch_accuracy

        # Optimizer step after accumulating enough gradients
        if not is_accumulating:
            if scaler is not None:
                if config.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

        # Update progress bar
        if pbar is not None:
            # Calculate MFU every log_every steps for stable readings
            if (step + 1) % config.log_every == 0 and flops_per_token > 0:
                elapsed = time.perf_counter() - mfu_window_start
                if elapsed > 0:
                    achieved_flops = flops_per_token * mfu_window_tokens / elapsed
                    current_mfu = 100.0 * achieved_flops / peak_flops
                mfu_window_tokens = 0
                mfu_window_start = time.perf_counter()

            # Build postfix dict (show EMA loss in progress bar)
            postfix = {
                "loss": f"{ema_loss:.3f}" if ema_loss else f"{batch_loss:.3f}",
                "out": f"{ema_output_loss:.3f}" if ema_output_loss else f"{batch_output_loss:.3f}",
                "acc": f"{100*ema_accuracy:.1f}%" if ema_accuracy else f"{100*batch_accuracy:.1f}%",
                "lr": f"{scheduler.get_last_lr()[0]:.1e}",
            }
            if flops_per_token > 0:
                postfix["MFU"] = f"{current_mfu:.1f}%"

            pbar.set_postfix(postfix)
            pbar.update(1)

        # Wandb step-level logging
        current_step = global_step + step + 1
        if wandb_run is not None and (step + 1) % config.wandb_log_every == 0:
            wandb_run.log(
                {
                    "step/loss": batch_loss,
                    "step/output_loss": batch_output_loss,
                    "step/ema_loss": ema_loss,
                    "step/ema_output_loss": ema_output_loss,
                    "step/accuracy": batch_accuracy,
                    "step/ema_accuracy": ema_accuracy,
                    "step/lr": scheduler.get_last_lr()[0],
                    "step/mfu": current_mfu if flops_per_token > 0 else 0,
                },
                step=current_step,
            )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Handle remaining gradients if not divisible by grad_accum_steps
    remaining = len(dataloader) % grad_accum_steps
    if remaining != 0:
        if scaler is not None:
            if config.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Compute epoch metrics
    epoch_time = time.perf_counter() - start_time
    avg_loss = total_loss / num_tokens
    avg_input_loss = total_input_loss / num_tokens
    avg_output_loss = total_output_loss / num_tokens
    avg_accuracy = total_output_correct / max(total_output_tokens, 1)

    # Compute average MFU for the epoch
    avg_mfu = 0.0
    if flops_per_token > 0 and epoch_time > 0:
        achieved_flops = flops_per_token * num_tokens / epoch_time
        avg_mfu = 100.0 * achieved_flops / peak_flops

    return {
        "loss": avg_loss,
        "input_loss": avg_input_loss,
        "output_loss": avg_output_loss,
        "accuracy": avg_accuracy,
        "ema_loss": ema_loss if ema_loss else avg_loss,
        "ema_output_loss": ema_output_loss if ema_output_loss else avg_output_loss,
        "ema_accuracy": ema_accuracy if ema_accuracy else avg_accuracy,
        "tokens_per_sec": num_tokens / epoch_time,
        "epoch_time": epoch_time,
        "mfu": avg_mfu,
    }


# =============================================================================
# VALIDATION
# =============================================================================


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: TrainConfig,
) -> dict[str, float]:
    """
    Run validation and compute metrics.

    Returns:
        Dict with validation metrics
    """
    model.eval()

    total_loss = 0.0
    total_input_loss = 0.0
    total_output_loss = 0.0
    total_output_correct = 0
    total_output_tokens = 0
    num_tokens = 0

    use_amp = config.use_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        positions_3d = batch["positions_3d"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        example_ids = batch["example_ids"].to(device)

        output_mask = create_output_mask(input_ids)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(
                input_ids, positions_3d,
                example_ids=example_ids,
                attention_mask=attention_mask,
            )

            losses = compute_loss(
                logits=logits,
                labels=input_ids,
                attention_mask=attention_mask,
                output_mask=output_mask,
                input_loss_weight=config.input_loss_weight,
                output_loss_weight=config.output_loss_weight,
                uniform_weight=config.uniform_loss_weight,
            )

        # Compute accuracy
        preds = logits.argmax(dim=-1)
        output_valid = output_mask & attention_mask
        correct = (preds == input_ids) & output_valid
        total_output_correct += correct.sum().item()
        total_output_tokens += output_valid.sum().item()

        batch_tokens = attention_mask.sum().item()
        total_loss += losses["loss"].item() * batch_tokens
        total_input_loss += losses["input_loss"].item() * batch_tokens
        total_output_loss += losses["output_loss"].item() * batch_tokens
        num_tokens += batch_tokens

    return {
        "val_loss": total_loss / num_tokens,
        "val_input_loss": total_input_loss / num_tokens,
        "val_output_loss": total_output_loss / num_tokens,
        "val_accuracy": total_output_correct / max(total_output_tokens, 1),
    }


# =============================================================================
# CHECKPOINTING
# =============================================================================


def get_rng_state() -> dict:
    """Capture RNG states for reproducibility."""
    state = {
        "python": None,  # Not easily capturable
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    """Restore RNG states."""
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    epoch: int,
    global_step: int,
    config: TrainConfig,
    metrics: dict,
    path: str | Path,
    task_id_to_example_id: dict[str, int] | None = None,
) -> None:
    """Save training checkpoint with example_id mapping for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Get base model if compiled/DDP wrapped
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    if hasattr(base_model, "module"):
        base_model = base_model.module

    checkpoint = {
        "model_state": base_model.state_dict(),
        "model_config": asdict(base_model.config),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": asdict(config),
        "metrics": metrics,
        "rng_state": get_rng_state(),
        # Store example_id mapping for reproducibility
        "task_id_to_example_id": task_id_to_example_id,
    }

    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: AdamW | None = None,
    scheduler: LambdaLR | None = None,
    current_task_id_mapping: dict[str, int] | None = None,
) -> dict:
    """
    Load checkpoint and restore state.

    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        current_task_id_mapping: Current dataset's task_id_to_example_id for verification

    Returns:
        Checkpoint dict with metadata
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Get base model if compiled/DDP wrapped
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    if hasattr(base_model, "module"):
        base_model = base_model.module
    base_model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    if "rng_state" in checkpoint:
        set_rng_state(checkpoint["rng_state"])

    # Verify example_id mapping matches if both are available
    saved_mapping = checkpoint.get("task_id_to_example_id")
    if saved_mapping is not None and current_task_id_mapping is not None:
        if saved_mapping != current_task_id_mapping:
            print(
                "WARNING: task_id_to_example_id mapping differs between checkpoint and "
                "current dataset. Example embeddings may be misaligned!"
            )
            # Show which tasks differ
            saved_tasks = set(saved_mapping.keys())
            current_tasks = set(current_task_id_mapping.keys())
            missing = saved_tasks - current_tasks
            new = current_tasks - saved_tasks
            if missing:
                print(f"  Tasks in checkpoint but not in dataset: {len(missing)}")
            if new:
                print(f"  Tasks in dataset but not in checkpoint: {len(new)}")

    print(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")

    return checkpoint


# =============================================================================
# MODEL FACTORY
# =============================================================================


def create_model_from_config(config: TrainConfig) -> TinyTransformer:
    """Create model based on config.model_size."""
    model_factories = {
        "tiny": create_tiny_model,
        "small": create_small_model,
        "medium": create_medium_model,
        "large": create_large_model,
    }

    if config.model_size not in model_factories:
        raise ValueError(f"Unknown model size: {config.model_size}")

    return model_factories[config.model_size](
        max_seq_len=config.max_seq_len,
        rope_type=config.rope_type,
        norm_type=config.norm_type,
        ffn_type=config.ffn_type,
        use_example_embedding=config.use_example_embedding,
    )


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================


def setup_ddp(local_rank: int) -> None:
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)


def cleanup_ddp() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def ensure_data_exists(data_path: str, solutions_path: str | None, include_concept: bool = True) -> None:
    """
    Check if required data files exist, download and build them if not.

    Downloads ARC-1 and optionally ConceptARC to match original mdlARC training data.
    Called automatically before training starts (only on rank 0).
    """
    from src.data import (
        build_dataset,
        build_training_solutions,
        download_dataset,
    )

    data_path = Path(data_path)
    data_dir = data_path.parent
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check if we need to download/build data
    need_challenges = not data_path.exists()
    need_solutions = solutions_path and not Path(solutions_path).exists()

    if not need_challenges and not need_solutions:
        return

    # Download ARC-1 dataset if needed
    arc1_data_dir = data_dir / "ARC-AGI-master" / "data"
    if not arc1_data_dir.exists():
        print("Downloading ARC-1 dataset...")
        download_dataset("arc1", output_dir=data_dir)

    concept_corpus_dir = data_dir / "ConceptARC-main" / "corpus"
    if include_concept and not concept_corpus_dir.exists():
        print("Downloading ConceptARC dataset...")
        download_dataset("concept", output_dir=data_dir)

    # Build challenges JSON (ARC-1 + ConceptARC by default)
    if need_challenges:
        sources = [("arc1", arc1_data_dir)]
        if include_concept:
            sources.append(("concept", concept_corpus_dir))
        source_label = "ARC-1 + ConceptARC" if include_concept else "ARC-1 only"
        print(f"Building {data_path} ({source_label})...")
        build_dataset(
            sources=sources,
            output_path=data_path,
            strip_eval_outputs=True,  # No eval test outputs (prevent leakage)
            prefix_task_ids=False,  # Don't prefix, keep original task IDs
        )

    # Build training solutions JSON (ARC-1 training tasks only)
    if need_solutions:
        print(f"Building {solutions_path}...")
        build_training_solutions(arc1_data_dir, solutions_path)


def sanitize_data_dir(data_path: str) -> None:
    """
    Remove evaluation solution files to prevent data leakage.

    Called automatically before training starts.
    """
    data_dir = Path(data_path).parent

    # Files that MUST NOT exist during training
    forbidden_files = [
        "arc-agi_evaluation_solutions.json",
        "arc-agi_test_solutions.json",
    ]

    for filename in forbidden_files:
        filepath = data_dir / filename
        if filepath.exists():
            filepath.unlink()
            print(f"Deleted eval solutions to prevent leakage: {filepath}")


def train(config: TrainConfig) -> TinyTransformer:
    """
    Main training function with DDP and gradient accumulation support.

    Args:
        config: Training configuration

    Returns:
        Trained model
    """
    # DDP setup
    if config.use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        config.local_rank = local_rank
        setup_ddp(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        is_rank_zero = local_rank == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        is_rank_zero = True

    if is_rank_zero:
        print(f"Using device: {device}")
        if config.use_ddp:
            print(f"DDP enabled with {world_size} GPUs")
        print(f"Gradient accumulation steps: {config.grad_accum_steps}")
        print(f"Effective batch size: {config.batch_size * world_size * config.grad_accum_steps}")

        # Auto-download and build data if missing
        ensure_data_exists(config.data_path, config.solutions_path, config.include_concept)

        # Auto-delete eval solutions to prevent data leakage
        sanitize_data_dir(config.data_path)

    # Synchronize after data download (important for DDP)
    if config.use_ddp:
        dist.barrier()

    # Initialize wandb (only on rank 0)
    wandb_run = None
    if config.use_wandb and is_rank_zero:
        try:
            import wandb

            wandb_run = wandb.init(
                project=config.wandb_project,
                name=config.get_run_name(),
                config=asdict(config),
                resume="allow" if config.resume_from else None,
            )
            print(f"W&B run: {wandb_run.name} ({wandb_run.url})")
        except ImportError:
            print("Warning: wandb not installed, skipping logging")
            wandb_run = None

    # Set seed for reproducibility
    # Use different seeds per rank for data diversity but same for model init
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Create model
    if is_rank_zero:
        print(f"Creating {config.model_size} model...")
    model = create_model_from_config(config)
    model = model.to(device)
    if is_rank_zero:
        print(f"Model parameters: {model.num_parameters:,}")

    # Wrap model in DDP before compile
    if config.use_ddp:
        model = DDP(
            model,
            device_ids=[config.local_rank],
        )

    # Compile if requested (after DDP wrapping)
    if config.compile_model and device.type == "cuda":
        if is_rank_zero:
            print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Load datasets
    # Original mdlARC trains on BOTH train and test pairs (since we know all solutions
    # for training tasks). This is not cheating - we're using all available data.
    if is_rank_zero:
        print("Loading training data...")
    train_splits = ("train", "test") if config.train_on_test_split else ("train",)
    include_test_outputs = config.train_on_test_split
    train_dataset = ARCDataset(
        config.data_path,
        splits=train_splits,
        include_test_outputs=include_test_outputs,
        solutions_path=config.solutions_path if include_test_outputs else None,
        apply_dihedral=config.apply_dihedral,
        max_seq_len=config.max_seq_len,
    )
    if is_rank_zero:
        print(f"Training examples: {len(train_dataset)}")

    # Validation dataset (same data, no augmentation, for loss tracking)
    val_dataset = None
    if config.solutions_path is not None:
        if is_rank_zero:
            print("Loading validation data...")
        val_dataset = ARCDataset(
            config.data_path,
            splits=("test",),  # Only test pairs for validation
            include_test_outputs=True,
            solutions_path=config.solutions_path,
            apply_dihedral=False,  # No augmentation for validation
            max_seq_len=config.max_seq_len,
        )
        if is_rank_zero:
            print(f"Validation examples: {len(val_dataset)}")

    # Color augmentor for deterministic per-epoch color augmentation
    color_augmentor = None
    if config.num_color_perms > 0:
        perms = torch.tensor(
            generate_color_permutations(config.num_color_perms, seed=config.color_seed),
            dtype=torch.long,
        )
        color_augmentor = ColorAugmentor(
            perms,
            apply_to_test_split=False,
            seed=config.color_seed,
        )

    # Create dataloaders with DistributedSampler for DDP
    train_sampler = None
    if config.use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=config.local_rank,
            shuffle=True,
            seed=config.seed,
        )

    train_loader = create_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),  # Don't shuffle if using sampler
        num_workers=config.num_workers,
        sampler=train_sampler,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = create_dataloader(
            val_dataset,
            batch_size=config.val_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

    # Optimizer and scheduler
    # Compute steps accounting for gradient accumulation
    steps_per_epoch = len(train_loader) // config.grad_accum_steps
    total_steps = steps_per_epoch * config.epochs
    optimizer = create_optimizer(model, config.lr, config.weight_decay, device)
    scheduler = create_scheduler(optimizer, total_steps, config.warmup_fraction)

    # GradScaler for mixed precision (only for fp16, not bf16)
    scaler = None  # bf16 doesn't need scaler on modern GPUs

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if config.resume_from is not None:
        # Get base model for loading (unwrap DDP/compile)
        base_model = model
        if hasattr(model, "_orig_mod"):
            base_model = model._orig_mod
        if hasattr(base_model, "module"):
            base_model = base_model.module

        checkpoint = load_checkpoint(
            config.resume_from,
            base_model,
            optimizer,
            scheduler,
            current_task_id_mapping=train_dataset.task_id_to_example_id,
        )
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]

    # Training loop
    if is_rank_zero:
        print(f"\nStarting training for {config.epochs} epochs...")
        print(f"Total optimizer steps: {total_steps}")
        print(f"Warmup steps: {int(total_steps * config.warmup_fraction)}")
        print("-" * 60)

    best_val_loss = float("inf")

    try:
        for epoch in range(start_epoch, config.epochs):
            if is_rank_zero:
                print(f"\nEpoch {epoch + 1}/{config.epochs}")

            # Set epoch for DistributedSampler (ensures different shuffling per epoch)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # Set deterministic color permutation for this epoch
            if color_augmentor is not None and color_augmentor.num_permutations > 0:
                color_augmentor.set_index(epoch)
                if is_rank_zero:
                    print(
                        f"Using color permutation {color_augmentor.current_index + 1}"
                        f"/{color_augmentor.num_permutations} for this epoch."
                    )

            # Train
            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                epoch=epoch,
                config=config,
                color_augmentor=color_augmentor,
                scaler=scaler,
                wandb_run=wandb_run,
                global_step=global_step,
            )

            if is_rank_zero:
                mfu_str = f" | MFU: {train_metrics['mfu']:.1f}%" if train_metrics.get("mfu", 0) > 0 else ""
                print(
                    f"  Train Loss: {train_metrics['loss']:.4f} | "
                    f"Output Loss: {train_metrics['output_loss']:.4f} | "
                    f"Time: {train_metrics['epoch_time']:.1f}s{mfu_str}"
                )

            # Validation (only on rank 0 for simplicity)
            val_metrics = {}
            if val_loader is not None and (epoch + 1) % config.val_every == 0:
                val_metrics = validate(model, val_loader, device, config)
                if is_rank_zero:
                    print(
                        f"  Val Loss: {val_metrics['val_loss']:.4f} | "
                        f"Val Output Loss: {val_metrics['val_output_loss']:.4f}"
                    )

                    if val_metrics["val_output_loss"] < best_val_loss:
                        best_val_loss = val_metrics["val_output_loss"]
                        save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            global_step=global_step + steps_per_epoch,
                            config=config,
                            metrics={**train_metrics, **val_metrics},
                            path=Path(config.save_dir) / "best.pt",
                            task_id_to_example_id=train_dataset.task_id_to_example_id,
                        )

            # Log to wandb (epoch-level)
            if wandb_run is not None:
                log_dict = {
                    "epoch": epoch + 1,
                    "train/loss": train_metrics["loss"],
                    "train/input_loss": train_metrics["input_loss"],
                    "train/output_loss": train_metrics["output_loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "train/ema_loss": train_metrics["ema_loss"],
                    "train/ema_output_loss": train_metrics["ema_output_loss"],
                    "train/ema_accuracy": train_metrics["ema_accuracy"],
                    "train/tokens_per_sec": train_metrics["tokens_per_sec"],
                    "train/epoch_time": train_metrics["epoch_time"],
                    "lr": scheduler.get_last_lr()[0],
                }
                if train_metrics.get("mfu", 0) > 0:
                    log_dict["train/mfu"] = train_metrics["mfu"]
                if val_metrics:
                    log_dict.update(
                        {
                            "val/loss": val_metrics["val_loss"],
                            "val/input_loss": val_metrics["val_input_loss"],
                            "val/output_loss": val_metrics["val_output_loss"],
                            "val/accuracy": val_metrics.get("val_accuracy", 0),
                        }
                    )
                wandb_run.log(log_dict, step=global_step + len(train_loader))

            # Periodic checkpoint (only on rank 0)
            if is_rank_zero and (epoch + 1) % config.save_every == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step + steps_per_epoch,
                    config=config,
                    metrics={**train_metrics, **val_metrics},
                    path=Path(config.save_dir) / f"epoch_{epoch + 1}.pt",
                    task_id_to_example_id=train_dataset.task_id_to_example_id,
                )

            global_step += len(train_loader)  # Count batches, not optimizer steps

            # Synchronize all processes before next epoch
            if config.use_ddp:
                dist.barrier()

        # Save final checkpoint (only on rank 0)
        if is_rank_zero:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=config.epochs - 1,
                global_step=global_step,
                config=config,
                metrics={**train_metrics, **val_metrics},
                path=Path(config.save_dir) / "final.pt",
                task_id_to_example_id=train_dataset.task_id_to_example_id,
            )

            print("\nTraining complete!")

    finally:
        # Clean up wandb
        if wandb_run is not None:
            wandb_run.finish()

        # Clean up DDP
        if config.use_ddp:
            cleanup_ddp()

    # Return base model (unwrap DDP and compile)
    base_model = model
    if hasattr(model, "_orig_mod"):
        base_model = model._orig_mod
    if hasattr(base_model, "module"):
        base_model = base_model.module
    return base_model


# =============================================================================
# CLI
# =============================================================================

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def main():
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train mdlARC model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run train --config small           # Use configs/small.toml
  uv run train --config configs/custom.toml  # Use custom config file
  uv run train --config small --epochs 50    # Override epochs
  uv run train --config small --no-wandb     # Disable wandb
        """,
    )

    # Config file (primary way to configure)
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="small",
        help="Config name (tiny/small/medium/large) or path to TOML file",
    )

    # Overrides (applied after loading config)
    parser.add_argument("--model-size", type=str, choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--solutions-path", type=str)
    parser.add_argument("--batch-size", type=int, help="Per-GPU batch size")
    parser.add_argument("--effective-batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--ddp", action="store_true", help="Enable DistributedDataParallel")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-run-name", type=str)
    parser.add_argument("--peak-tflops", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rope-type", type=str, choices=["3d", "1d", "none"])
    parser.add_argument("--norm-type", type=str, choices=["rmsnorm", "layernorm"])
    parser.add_argument("--ffn-type", type=str, choices=["swiglu", "gelu"])
    parser.add_argument("--no-example-embedding", action="store_true", help="Disable example embeddings")

    # Ablation-specific overrides
    parser.add_argument("--no-apply-dihedral", action="store_true", help="Disable dihedral augmentation")
    parser.add_argument("--num-color-perms", type=int, help="Number of color permutations (0 to disable)")
    parser.add_argument("--input-loss-weight", type=float, help="Weight for input loss (0 for output-only)")
    parser.add_argument("--output-loss-weight", type=float, help="Weight for output loss")
    parser.add_argument("--uniform-loss-weight", dest="uniform_loss_weight", action="store_true")
    parser.add_argument("--no-uniform-loss-weight", dest="uniform_loss_weight", action="store_false")
    parser.add_argument("--no-concept", action="store_true", help="Exclude ConceptARC from training data")
    parser.add_argument("--train-split-only", action="store_true", help="Train only on train split pairs")
    parser.set_defaults(uniform_loss_weight=None)

    args = parser.parse_args()

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.exists():
        # Try as a preset name
        config_path = CONFIGS_DIR / f"{args.config}.toml"
        if not config_path.exists():
            print(f"Error: Config not found: {args.config}")
            print(f"Available presets: {', '.join(p.stem for p in CONFIGS_DIR.glob('*.toml'))}")
            return

    print(f"Loading config from: {config_path}")
    config = TrainConfig.from_toml(config_path)

    # Apply CLI overrides
    if args.model_size is not None:
        config.model_size = args.model_size
    if args.data_path is not None:
        config.data_path = args.data_path
    if args.solutions_path is not None:
        config.solutions_path = args.solutions_path
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.effective_batch_size is not None:
        config.effective_batch_size = args.effective_batch_size
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.lr is not None:
        config.lr = args.lr
    if args.save_dir is not None:
        config.save_dir = args.save_dir
    if args.resume is not None:
        config.resume_from = args.resume
    if args.no_compile:
        config.compile_model = False
    if args.no_amp:
        config.use_amp = False
    if args.ddp or os.environ.get("LOCAL_RANK") is not None:
        # Auto-detect torchrun by checking for LOCAL_RANK env var
        config.use_ddp = True
    if args.no_wandb:
        config.use_wandb = False
    if args.wandb_project is not None:
        config.wandb_project = args.wandb_project
    if args.wandb_run_name is not None:
        config.wandb_run_name = args.wandb_run_name
    if args.peak_tflops is not None:
        config.peak_tflops = args.peak_tflops
    if args.seed is not None:
        config.seed = args.seed
    if args.rope_type is not None:
        config.rope_type = args.rope_type
    if args.norm_type is not None:
        config.norm_type = args.norm_type
    if args.ffn_type is not None:
        config.ffn_type = args.ffn_type
    if args.no_example_embedding:
        config.use_example_embedding = False

    # Ablation overrides
    if args.no_apply_dihedral:
        config.apply_dihedral = False
    if args.num_color_perms is not None:
        config.num_color_perms = args.num_color_perms
    if args.input_loss_weight is not None:
        config.input_loss_weight = args.input_loss_weight
        if args.uniform_loss_weight is None:
            config.uniform_loss_weight = False
    if args.output_loss_weight is not None:
        config.output_loss_weight = args.output_loss_weight
        if args.uniform_loss_weight is None:
            config.uniform_loss_weight = False
    if args.uniform_loss_weight is not None:
        config.uniform_loss_weight = args.uniform_loss_weight
    if args.no_concept:
        config.include_concept = False
    if args.train_split_only:
        config.train_on_test_split = False

    train(config)


if __name__ == "__main__":
    main()

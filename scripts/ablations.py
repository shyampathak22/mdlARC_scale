#!/usr/bin/env python3
"""
Ablation study runner for mdlARC.

Systematically tests different components to understand what makes mdlARC work.

Usage:
    uv run ablations --size tiny --epochs 20 --gpus 2
    uv run ablations --size tiny --epochs 20 --gpus 1 --ablations rope,aug
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch


@dataclass
class AblationConfig:
    """Configuration for a single ablation run."""
    name: str
    description: str
    overrides: dict  # Config overrides to apply


# Define all ablations to test
ABLATIONS = {
    # Baseline
    "baseline": AblationConfig(
        name="baseline",
        description="Full mdlARC system (control)",
        overrides={},
    ),

    # RoPE ablations
    "rope_1d": AblationConfig(
        name="rope_1d",
        description="1D RoPE over sequence positions",
        overrides={"rope_type": "1d"},
    ),
    "rope_none": AblationConfig(
        name="rope_none",
        description="No RoPE positional encoding",
        overrides={"rope_type": "none"},
    ),

    # Augmentation ablations (implemented)
    "no_dihedral": AblationConfig(
        name="no_dihedral",
        description="Disable dihedral (geometric) augmentation",
        overrides={"no_apply_dihedral": True},
    ),
    "no_color": AblationConfig(
        name="no_color",
        description="Disable color permutation augmentation",
        overrides={"num_color_perms": 0},
    ),
    "no_aug": AblationConfig(
        name="no_aug",
        description="No augmentation at all",
        overrides={"no_apply_dihedral": True, "num_color_perms": 0},
    ),

    # Loss ablations (implemented)
    "output_only_loss": AblationConfig(
        name="output_only_loss",
        description="Only compute loss on output tokens",
        overrides={"input_loss_weight": 0.0, "uniform_loss_weight": False},
    ),

    # Architecture ablations
    "layernorm": AblationConfig(
        name="layernorm",
        description="Swap RMSNorm for LayerNorm",
        overrides={"norm_type": "layernorm"},
    ),
    "gelu": AblationConfig(
        name="gelu",
        description="Swap SwiGLU for GELU FFN",
        overrides={"ffn_type": "gelu"},
    ),
    "no_example_embedding": AblationConfig(
        name="no_example_embedding",
        description="Disable example embeddings",
        overrides={"no_example_embedding": True},
    ),

    # Sweep-ish variants (not pure ablations)
    "color_10": AblationConfig(
        name="color_10",
        description="Only 10 color permutations (vs 100)",
        overrides={"num_color_perms": 10},
    ),
    "color_50": AblationConfig(
        name="color_50",
        description="50 color permutations (vs 100)",
        overrides={"num_color_perms": 50},
    ),
    "low_input_loss": AblationConfig(
        name="low_input_loss",
        description="Reduce input loss weight to 0.1",
        overrides={"input_loss_weight": 0.1, "uniform_loss_weight": False},
    ),

    # Data composition ablations
    "no_concept": AblationConfig(
        name="no_concept",
        description="Train on ARC-1 only (no ConceptARC)",
        overrides={
            "no_concept": True,
            "data_path": "data/arc1_training_challenges.json",
            "solutions_path": "data/arc1_training_solutions.json",
        },
    ),
    "train_split_only": AblationConfig(
        name="train_split_only",
        description="Train only on train split pairs (no test-pair outputs)",
        overrides={"train_split_only": True},
    ),
}

# Ablation groups for quick runs
ABLATION_GROUPS = {
    "rope": ["baseline", "rope_1d", "rope_none"],
    "aug": ["baseline", "no_dihedral", "no_color", "no_aug"],
    "color": ["baseline", "no_color", "color_10", "color_50"],
    "color_sweep": ["baseline", "color_10", "color_50"],
    "data": ["baseline", "no_concept", "train_split_only"],
    "arch": ["baseline", "layernorm", "gelu"],
    "core": ["baseline", "no_example_embedding"],
    "loss": ["baseline", "output_only_loss"],
    "sweep": ["baseline", "color_10", "color_50", "low_input_loss"],
    "all": [
        "baseline",
        "rope_1d",
        "rope_none",
        "no_dihedral",
        "no_color",
        "no_aug",
        "output_only_loss",
        "layernorm",
        "gelu",
        "no_concept",
        "train_split_only",
        "no_example_embedding",
    ],
    "all_with_sweeps": list(ABLATIONS.keys()),
    "quick": ["baseline", "no_aug", "rope_none"],  # Fast sanity check
}


def run_ablation(
    ablation: AblationConfig,
    base_config: str,
    epochs: int,
    gpus: int,
    output_dir: Path,
    no_wandb: bool = False,
) -> dict:
    """Run a single ablation and return results."""

    save_dir = output_dir / ablation.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build command
    if gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={gpus}",
            "-m", "src.train",
        ]
    else:
        cmd = [sys.executable, "-m", "src.train"]

    cmd.extend([
        "--config", base_config,
        "--epochs", str(epochs),
        "--save-dir", str(save_dir),
        "--wandb-run-name", f"ablation_{ablation.name}",
    ])

    if no_wandb:
        cmd.append("--no-wandb")

    # Add overrides (these need to be implemented in train.py)
    for key, value in ablation.overrides.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key.replace('_', '-')}")
            else:
                cmd.append(f"--no-{key.replace('_', '-')}")
        else:
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    print(f"\n{'='*60}")
    print(f"Running ablation: {ablation.name}")
    print(f"Description: {ablation.description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    # Run training
    start_time = datetime.now()
    result = subprocess.run(cmd, capture_output=False)
    end_time = datetime.now()

    # Try to load final metrics
    metrics = {}
    metrics_file = save_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
    else:
        for ckpt_name in ("best.pt", "final.pt"):
            ckpt_path = save_dir / ckpt_name
            if not ckpt_path.exists():
                continue
            try:
                checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                metrics = checkpoint.get("metrics", {}) or {}
                if metrics:
                    break
            except Exception:
                continue

    return {
        "name": ablation.name,
        "description": ablation.description,
        "overrides": ablation.overrides,
        "returncode": result.returncode,
        "duration_seconds": (end_time - start_time).total_seconds(),
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run mdlARC ablation studies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ablation groups:
  rope    - Test 3D RoPE vs 1D vs none
  aug     - Test augmentation strategies
  color   - Color ablations + small sweeps
  color_sweep - Color permutation count sweep
  data    - Test data composition
  arch    - Test architecture choices
  core    - Core conditioning ablations
  loss    - Test loss weighting ablations
  sweep   - Hyperparam-ish sweeps (not pure ablations)
  quick   - Fast sanity check (3 runs)
  all     - Run pure ablations only
  all_with_sweeps - Run everything (ablations + sweeps)

Examples:
  uv run ablations --size tiny --epochs 20 --gpus 2
  uv run ablations --size tiny --epochs 10 --ablations rope,aug
  uv run ablations --size tiny --epochs 20 --list baseline,no_aug,rope_none
        """,
    )
    parser.add_argument(
        "--size", "-s",
        choices=["tiny", "small", "medium", "large"],
        default="tiny",
        help="Base model size (default: tiny)",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=20,
        help="Epochs per ablation (default: 20)",
    )
    parser.add_argument(
        "--gpus", "-g",
        type=int,
        default=1,
        help="Number of GPUs (default: 1)",
    )
    parser.add_argument(
        "--ablations", "-a",
        help="Comma-separated ablation groups to run (e.g., rope,aug)",
    )
    parser.add_argument(
        "--list", "-l",
        help="Comma-separated list of specific ablations to run",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="checkpoints/ablations",
        help="Output directory (default: checkpoints/ablations)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show available ablations and exit",
    )

    args = parser.parse_args()

    if args.show:
        print("Available ablations:")
        print("-" * 60)
        for name, config in ABLATIONS.items():
            print(f"  {name:20s} - {config.description}")
        print("\nAblation groups:")
        print("-" * 60)
        for group, members in ABLATION_GROUPS.items():
            print(f"  {group:10s} - {', '.join(members)}")
        return

    # Determine which ablations to run
    ablation_names = set()

    if args.list:
        ablation_names.update(args.list.split(","))
    elif args.ablations:
        for group in args.ablations.split(","):
            if group in ABLATION_GROUPS:
                ablation_names.update(ABLATION_GROUPS[group])
            else:
                print(f"Warning: Unknown group '{group}', skipping")
    else:
        # Default to quick check
        ablation_names.update(ABLATION_GROUPS["quick"])

    # Validate ablation names
    invalid = ablation_names - set(ABLATIONS.keys())
    if invalid:
        print(f"Error: Unknown ablations: {invalid}")
        print(f"Available: {list(ABLATIONS.keys())}")
        sys.exit(1)

    # Sort for consistent ordering (baseline first)
    ablation_list = sorted(ablation_names, key=lambda x: (x != "baseline", x))

    print(f"Running {len(ablation_list)} ablations: {ablation_list}")
    print(f"Base config: {args.size}")
    print(f"Epochs per run: {args.epochs}")
    print(f"GPUs: {args.gpus}")

    # Run ablations
    output_dir = Path(args.output_dir) / f"{args.size}_{args.epochs}ep"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in ablation_list:
        ablation = ABLATIONS[name]
        result = run_ablation(
            ablation=ablation,
            base_config=f"configs/{args.size}.toml",
            epochs=args.epochs,
            gpus=args.gpus,
            output_dir=output_dir,
            no_wandb=args.no_wandb,
        )
        results.append(result)

        # Save intermediate results
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)

    for r in results:
        status = "OK" if r["returncode"] == 0 else "FAILED"
        duration = r["duration_seconds"] / 60
        metrics_str = ""
        if r["metrics"]:
            val_loss = r["metrics"].get("val_output_loss", "N/A")
            metrics_str = f" | val_out_loss: {val_loss}"
        print(f"{r['name']:20s} [{status:6s}] {duration:5.1f}min{metrics_str}")

    print(f"\nFull results saved to: {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()

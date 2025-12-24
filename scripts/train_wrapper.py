#!/usr/bin/env python3
"""
Wrapper script for training that handles DDP/torchrun automatically.

Usage:
    uv run train --config configs/small.toml --gpus 2
    uv run train --config tiny --gpus 1
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Train mdlARC model")
    parser.add_argument("--config", "-c", required=True, help="Config file or preset name")
    parser.add_argument("--gpus", "-g", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--epochs", "-e", type=int, help="Override epochs")
    parser.add_argument("--save-dir", "-s", help="Override save directory")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb")
    parser.add_argument("--resume", "-r", help="Resume from checkpoint")

    args, extra_args = parser.parse_known_args()

    # Build the command
    if args.gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={args.gpus}",
            "-m", "src.train",
        ]
    else:
        cmd = [sys.executable, "-m", "src.train"]

    # Add config
    cmd.extend(["--config", args.config])

    # Add optional overrides
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.save_dir:
        cmd.extend(["--save-dir", args.save_dir])
    if args.no_wandb:
        cmd.append("--no-wandb")
    if args.resume:
        cmd.extend(["--resume", args.resume])

    # Pass through any extra args
    cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")

    # Execute
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

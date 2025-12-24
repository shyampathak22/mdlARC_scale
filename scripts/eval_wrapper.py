#!/usr/bin/env python3
"""
Wrapper script for evaluation that handles multi-GPU automatically.

Usage:
    uv run evaluate --gpus 2
    uv run evaluate --checkpoint checkpoints/small/best.pt --gpus 1
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Evaluate mdlARC model")
    parser.add_argument("--gpus", "-g", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--checkpoint", "-c", type=str, default="checkpoints/small/best.pt",
                        help="Path to checkpoint")
    parser.add_argument("--challenges", type=str, help="Path to challenges JSON")
    parser.add_argument("--solutions", type=str, help="Path to solutions JSON")
    parser.add_argument("--output", "-o", type=str, help="Output path for submission")
    parser.add_argument("--num-color-perms", type=int, default=8,
                        help="Number of color permutations for AAIVR")
    parser.add_argument("--quiet", "-q", action="store_true", help="Disable progress bars")

    args, extra_args = parser.parse_known_args()

    # Build the command
    if args.gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={args.gpus}",
            "-m", "src.eval",
        ]
    else:
        cmd = [sys.executable, "-m", "src.eval"]

    # Add arguments
    cmd.extend(["--checkpoint", args.checkpoint])

    if args.challenges:
        cmd.extend(["--challenges", args.challenges])
    if args.solutions:
        cmd.extend(["--solutions", args.solutions])
    if args.output:
        cmd.extend(["--output", args.output])

    cmd.extend(["--num-color-perms", str(args.num_color_perms)])

    if args.quiet:
        cmd.append("--quiet")

    # Pass through any extra args
    cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")

    # Execute
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

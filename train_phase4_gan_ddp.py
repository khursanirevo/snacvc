#!/usr/bin/env python3
"""
DDP wrapper for Phase 4 GAN training.
Uses GPU 1 and 2 for distributed training.

Usage:
    uv run python train_phase4_gan_ddp.py --config configs/phase4_gan.json
"""

import os
import sys
import subprocess

def main():
    # Get config path
    config_path = "configs/phase4_gan.json"
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]
            break

    # DDP setup
    n_gpus_per_node = 2
    world_size = n_gpus_per_node

    # GPUs to use: 1 and 2
    visible_devices = "1,2"

    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_devices

    # Build torchrun command
    cmd = [
        "uv", "run", "python", "-m", "torch.distributed.run",
        f"--nproc_per_node={n_gpus_per_node}",
        "--master_port=29500",
        "train_phase4_gan.py",
        "--config", config_path,
        "--ddp"
    ]

    print(f"Launching DDP training on GPUs {visible_devices}")
    print(f"World size: {world_size}")
    print(f"Command: {' '.join(cmd)}")
    print("")

    # Launch training
    result = subprocess.run(cmd, env=env, check=False)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run ablation studies on Phase 4 GAN training.

Ablations to test:
1. No contrastive loss
2. No GAN loss
3. Different negative sampling strategies
4. Unfrozen speaker encoder
5. Different segment lengths
6. Different conditioning methods
"""

import sys
import torch
import json
from pathlib import Path
import subprocess
import shutil
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


ABLATION_CONFIGS = {
    "baseline": {
        "description": "Full model with all losses (baseline)",
        "config_override": {},
    },

    "no_contrastive": {
        "description": "Remove contrastive speaker loss",
        "config_override": {
            "contrastive_weight": 0.0,
        },
    },

    "no_gan": {
        "description": "Remove GAN loss (reconstruction only)",
        "config_override": {
            "gan_weight": 0.0,
            "lambda_adv": 0.0,
            "lambda_fm": 0.0,
        },
    },

    "low_contrastive": {
        "description": "Lower contrastive loss weight (0.1 instead of 0.5)",
        "config_override": {
            "contrastive_weight": 0.1,
        },
    },

    "high_contrastive": {
        "description": "Higher contrastive loss weight (1.0 instead of 0.5)",
        "config_override": {
            "contrastive_weight": 1.0,
        },
    },

    "no_hard_negative": {
        "description": "Random negative sampling instead of hard negative mining",
        "config_override": {
            "use_hard_negative_mining": False,
        },
    },

    "more_negatives": {
        "description": "Increase max negatives from 6 to 12",
        "config_override": {
            "max_negatives": 12,
        },
    },

    "segment_1s": {
        "description": "Shorter segment length (1s instead of 2s)",
        "config_override": {
            "segment_length": 1.0,
        },
    },

    "segment_4s": {
        "description": "Longer segment length (4s instead of 2s)",
        "config_override": {
            "segment_length": 4.0,
        },
    },

    "batch_4": {
        "description": "Smaller batch size (4 instead of 8)",
        "config_override": {
            "batch_size": 4,
        },
    },

    "batch_16": {
        "description": "Larger batch size (16 if VRAM allows)",
        "config_override": {
            "batch_size": 16,
        },
    },
}


def create_ablation_config(base_config_path, ablation_name, output_dir):
    """
    Create ablation config by modifying base config.

    Args:
        base_config_path: Path to base phase4_gan.json
        ablation_name: Name of ablation from ABLATION_CONFIGS
        output_dir: Where to save ablation config

    Returns:
        - ablation_config_path: Path to created config
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        base_config = json.load(f)

    # Get ablation override
    ablation = ABLATION_CONFIGS[ablation_name]
    override = ablation["config_override"]

    # Apply override
    ablation_config = base_config.copy()
    ablation_config.update(override)

    # Update experiment name
    ablation_config["experiment_name"] = f"phase4_ablation_{ablation_name}"
    ablation_config["output_dir"] = str(output_dir / f"checkpoints_{ablation_name}")

    # Save ablation config
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation_config_path = output_dir / f"config_{ablation_name}.json"
    with open(ablation_config_path, 'w') as f:
        json.dump(ablation_config, f, indent=2)

    print(f"Created ablation config: {ablation_config_path}")
    print(f"  Description: {ablation['description']}")

    return ablation_config_path


def run_ablation(ablation_name, base_config_path, output_dir,
                 gpu_id=1, use_ddp=False, epochs=10):
    """
    Run a single ablation study.

    Args:
        ablation_name: Name of ablation from ABLATION_CONFIGS
        base_config_path: Path to base phase4_gan.json
        output_dir: Base directory for all ablations
        gpu_id: GPU to use
        use_ddp: Whether to use DDP
        epochs: Number of epochs to run (default: 10 for quick ablation)
    """
    print(f"\n{'='*70}")
    print(f"Running Ablation: {ablation_name}")
    print(f"{'='*70}")

    # Create ablation config
    ablation_config_path = create_ablation_config(
        base_config_path, ablation_name, output_dir
    )

    # Build command
    if use_ddp:
        cmd = [
            "uv", "run", "python", "-m", "torch.distributed.run",
            f"--nproc_per_node=2",
            "--master_port=29500",
            "train_phase4_gan.py",
            "--config", str(ablation_config_path),
            "--ddp",
        ]
        # Set GPU env var
        env_cmd = f"CUDA_VISIBLE_DEVICES={gpu_id},{gpu_id+1}"
    else:
        cmd = [
            "uv", "run", "python",
            "train_phase4_gan.py",
            "--config", str(ablation_config_path),
            "--device", str(gpu_id),
        ]
        env_cmd = None

    # Override epochs for quick ablation
    # (Modify config to have fewer epochs)
    with open(ablation_config_path, 'r') as f:
        config = json.load(f)
    config['num_epochs'] = epochs
    config['save_every'] = 2  # Save more frequently
    with open(ablation_config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Command: {' '.join(cmd)}")
    print(f"Output: {output_dir / f'logs_{ablation_name}.log'}")

    # Run training
    log_path = output_dir / f"logs_{ablation_name}.log"

    with open(log_path, 'w') as log_file:
        if env_cmd:
            # Need to run with env var
            full_cmd = f"{env_cmd} {' '.join(cmd)}"
            result = subprocess.run(
                full_cmd,
                shell=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )
        else:
            result = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )

    if result.returncode == 0:
        print(f"✅ Ablation '{ablation_name}' completed successfully")
    else:
        print(f"❌ Ablation '{ablation_name}' failed with return code {result.returncode}")

    return result.returncode == 0


def run_all_ablations(ablation_names, base_config_path, output_dir,
                      gpu_ids=[1, 3], use_ddp=False, epochs=10):
    """
    Run multiple ablations in sequence.

    Args:
        ablation_names: List of ablation names to run
        base_config_path: Path to base config
        output_dir: Base directory for all ablations
        gpu_ids: List of GPU IDs to use (round-robin)
        use_ddp: Whether to use DDP for each ablation
        epochs: Number of epochs per ablation
    """
    print(f"\n{'='*70}")
    print(f"Running {len(ablation_names)} ablations")
    print(f"{'='*70}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for idx, ablation_name in enumerate(ablation_names):
        # Select GPU (round-robin)
        gpu_id = gpu_ids[idx % len(gpu_ids)]

        print(f"\n[{idx+1}/{len(ablation_names)}] Running: {ablation_name} on GPU {gpu_id}")

        success = run_ablation(
            ablation_name=ablation_name,
            base_config_path=base_config_path,
            output_dir=output_dir,
            gpu_id=gpu_id,
            use_ddp=use_ddp,
            epochs=epochs
        )

        results[ablation_name] = {
            'success': success,
            'timestamp': datetime.now().isoformat(),
        }

    # Save results summary
    results_path = output_dir / "ablation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Ablation Study Complete!")
    print(f"Results saved to: {results_path}")
    print(f"{'='*70}")

    # Print summary
    print("\nResults Summary:")
    for ablation_name, result in results.items():
        status = "✅ Success" if result['success'] else "❌ Failed"
        print(f"  {ablation_name}: {status}")

    return results


def compare_ablation_results(ablation_output_dir):
    """
    Compare results from multiple ablations.

    Generates:
    - Loss curves comparison
    - Final metrics comparison table
    - Speaker similarity comparison
    """
    import matplotlib.pyplot as plt
    import numpy as np

    ablation_output_dir = Path(ablation_output_dir)

    print(f"\n{'='*70}")
    print("Comparing Ablation Results")
    print(f"{'='*70}")

    # Collect results from all ablations
    all_metrics = {}

    for config_path in ablation_output_dir.glob("config_*.json"):
        ablation_name = config_path.stem.replace("config_", "")

        # Read log file to extract final metrics
        log_path = ablation_output_dir / f"logs_{ablation_name}.log"

        if not log_path.exists():
            continue

        # Parse log for metrics
        metrics = parse_training_log(log_path)
        all_metrics[ablation_name] = metrics

    # Plot comparison
    if all_metrics:
        plot_ablation_comparison(all_metrics, ablation_output_dir)

    print(f"\n✅ Comparison complete! Plots saved to {ablation_output_dir}")


def parse_training_log(log_path):
    """Parse training log to extract metrics."""
    metrics = {
        'final_losses': {},
        'loss_history': [],
    }

    with open(log_path, 'r') as f:
        for line in f:
            # Look for loss lines like: "Train - G: 1.23, D: 2.34, ..."
            if "Train - G:" in line:
                try:
                    # Parse generator loss
                    g_start = line.find("G:") + 2
                    g_end = line.find(",", g_start)
                    g_loss = float(line[g_start:g_end])

                    # Parse discriminator loss
                    d_start = line.find("D:") + 2
                    d_end = line.find(",", d_start)
                    d_loss = float(line[d_start:d_end])

                    metrics['loss_history'].append({
                        'gen': g_loss,
                        'disc': d_loss,
                    })

                    metrics['final_losses']['gen'] = g_loss
                    metrics['final_losses']['disc'] = d_loss

                except (ValueError, AttributeError):
                    pass

    return metrics


def plot_ablation_comparison(all_metrics, output_dir):
    """Plot comparison of ablation results."""

    import matplotlib.pyplot as plt

    # 1. Final losses bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ablation_names = list(all_metrics.keys())

    gen_losses = [all_metrics[name].get('final_losses', {}).get('gen', 0)
                  for name in ablation_names]
    disc_losses = [all_metrics[name].get('final_losses', {}).get('disc', 0)
                   for name in ablation_names]

    x = np.arange(len(ablation_names))
    width = 0.35

    axes[0].bar(x - width/2, gen_losses, width, label='Generator', alpha=0.8)
    axes[0].bar(x + width/2, disc_losses, width, label='Discriminator', alpha=0.8)

    axes[0].set_xlabel('Ablation')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Final Losses by Ablation')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ablation_names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2. Training curves
    for ablation_name in ablation_names:
        history = all_metrics[ablation_name].get('loss_history', [])
        if history:
            steps = range(len(history))
            gen_losses = [h['gen'] for h in history]
            axes[1].plot(steps, gen_losses, label=ablation_name, alpha=0.7)

    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Generator Loss')
    axes[1].set_title('Training Curves Comparison')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "ablation_comparison.png", dpi=150)
    plt.close()

    print(f"Saved: {output_dir / 'ablation_comparison.png'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--base-config", type=str,
                        default="configs/phase4_gan.json",
                        help="Path to base config")
    parser.add_argument("--output-dir", type=str,
                        default="ablations/phase4",
                        help="Output directory for ablations")
    parser.add_argument("--ablations", type=str, nargs='+',
                        default=["baseline", "no_contrastive", "no_gan"],
                        help="Ablations to run")
    parser.add_argument("--gpu-ids", type=int, nargs='+',
                        default=[1, 3],
                        help="GPU IDs to use (round-robin)")
    parser.add_argument("--ddp", action="store_true",
                        help="Use DDP for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs per ablation")
    parser.add_argument("--compare-only", action="store_true",
                        help="Only compare existing results, don't run new ablations")

    args = parser.parse_args()

    if args.compare_only:
        compare_ablation_results(args.output_dir)
    else:
        run_all_ablations(
            ablation_names=args.ablations,
            base_config_path=args.base_config,
            output_dir=args.output_dir,
            gpu_ids=args.gpu_ids,
            use_ddp=args.ddp,
            epochs=args.epochs
        )

        # Compare results after running
        compare_ablation_results(args.output_dir)

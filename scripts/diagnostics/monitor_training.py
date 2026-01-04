#!/usr/bin/env python3
"""
Real-time training diagnostics for Phase 4 GAN training.

Monitors:
- Loss curves and stability
- Gradient norms
- Speaker similarity metrics
- Parameter norm evolution
- GAN training health
"""

import sys
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from snac import SNACWithSpeakerConditioning
from snac.discriminators import MultiPeriodDiscriminator, MultiResolutionSTFTDiscriminator


def compute_gradient_norm(model):
    """Compute total gradient L2 norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def compute_parameter_norm(model):
    """Compute total parameter L2 norm."""
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def compute_speaker_similarity(model, real_audio, fake_audio, device):
    """
    Compute speaker similarity between real and generated audio.

    Returns:
        - mean_cosine_sim: Average cosine similarity
        - std_cosine_sim: Standard deviation of similarity
        - speaker_acc: Verification accuracy (>0.85 threshold)
    """
    model.eval()
    with torch.no_grad():
        # Get model_base (handle DDP)
        if hasattr(model, 'module'):
            model_base = model.module
        else:
            model_base = model

        # Extract embeddings
        real_embs = model_base.extract_speaker_embedding(real_audio)
        fake_embs = model_base.extract_speaker_embedding(fake_audio)

        # Normalize
        real_embs = torch.nn.functional.normalize(real_embs, p=2, dim=-1)
        fake_embs = torch.nn.functional.normalize(fake_embs, p=2, dim=-1)

        # Cosine similarity
        cosine_sim = (real_embs * fake_embs).sum(dim=-1)

        mean_sim = cosine_sim.mean().item()
        std_sim = cosine_sim.std().item()

        # Verification accuracy (threshold 0.85)
        acc = (cosine_sim > 0.85).float().mean().item()

    model.train()
    return mean_sim, std_sim, acc


def check_gan_health(gen_loss, disc_loss, prev_gen_loss=None, prev_disc_loss=None):
    """
    Check GAN training health.

    Returns:
        - health_status: 'healthy', 'warning', or 'critical'
        - issues: List of detected issues
    """
    issues = []

    # Discriminator too strong
    if disc_loss > 10.0:
        issues.append("Discriminator too strong (D loss > 10)")

    # Generator too weak
    if gen_loss > 10.0:
        issues.append("Generator loss too high (G loss > 10)")

    # Mode collapse detector
    if prev_disc_loss is not None:
        disc_change = abs(disc_loss - prev_disc_loss)
        if disc_change < 0.01:
            issues.append("Discriminator not learning (loss not changing)")

    # Loss ratio check
    ratio = disc_loss / (gen_loss + 1e-8)
    if ratio > 10:
        issues.append(f"Loss imbalance: D/G ratio = {ratio:.2f}")

    # Determine status
    if len(issues) == 0:
        status = "healthy"
    elif len(issues) <= 2:
        status = "warning"
    else:
        status = "critical"

    return status, issues


def generate_diagnostics_report(model, mpd, mrd, batch, device, losses,
                                prev_losses=None, step=0, output_dir=None):
    """
    Generate comprehensive diagnostics report.

    Returns:
        - metrics: Dict of diagnostic metrics
        - status: Overall health status
    """
    metrics = {
        'step': step,
    }

    # 1. Loss tracking
    metrics['losses'] = {
        'gen': losses['gen'],
        'disc': losses['disc'],
        'recon': losses['recon'],
        'contrast': losses['contrast'],
        'adv': losses['adv'],
        'fm': losses['fm'],
    }

    # 2. Gradient norms
    metrics['grad_norms'] = {
        'generator': compute_gradient_norm(model),
        'mpd': compute_gradient_norm(mpd),
        'mrd': compute_gradient_norm(mrd),
    }

    # 3. Parameter norms
    metrics['param_norms'] = {
        'generator': compute_parameter_norm(model),
        'mpd': compute_parameter_norm(mpd),
        'mrd': compute_parameter_norm(mrd),
    }

    # 4. GAN health check
    gen_loss = losses['gen']
    disc_loss = losses['disc']

    prev_gen_loss = prev_losses['gen'] if prev_losses else None
    prev_disc_loss = prev_losses['disc'] if prev_losses else None

    status, issues = check_gan_health(gen_loss, disc_loss, prev_gen_loss, prev_disc_loss)
    metrics['gan_health'] = {
        'status': status,
        'issues': issues,
    }

    # 5. Speaker similarity (if validation batch provided)
    if 'real_audio' in batch and 'fake_audio' in batch:
        real_audio = batch['real_audio'].to(device)
        fake_audio = batch['fake_audio'].to(device)

        mean_sim, std_sim, acc = compute_speaker_similarity(model, real_audio, fake_audio, device)

        metrics['speaker_similarity'] = {
            'mean': mean_sim,
            'std': std_sim,
            'accuracy': acc,
        }

    # 6. Loss trends
    if prev_losses:
        for key in ['gen', 'disc', 'recon', 'contrast']:
            change = losses[key] - prev_losses[key]
            metrics['losses'][f'{key}_change'] = change

    # Print report
    print_diagnostics_report(metrics)

    # Save to file
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Append to JSONL file
        log_file = output_dir / "diagnostics.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

    return metrics, status


def print_diagnostics_report(metrics):
    """Print formatted diagnostics report."""

    print("\n" + "="*70)
    print(f"DIAGNOSTICS REPORT - Step {metrics['step']}")
    print("="*70)

    # Losses
    print("\n[Losses]")
    for key, value in metrics['losses'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")

    # Gradient norms
    print("\n[Gradient Norms]")
    for key, value in metrics['grad_norms'].items():
        print(f"  {key}: {value:.4f}")

    # Parameter norms
    print("\n[Parameter Norms]")
    for key, value in metrics['param_norms'].items():
        print(f"  {key}: {value:.4f}")

    # GAN health
    print("\n[GAN Training Health]")
    gan_health = metrics['gan_health']
    status_emoji = {
        'healthy': '✅',
        'warning': '⚠️',
        'critical': '❌'
    }
    print(f"  Status: {status_emoji[gan_health['status']]} {gan_health['status'].upper()}")
    if gan_health['issues']:
        for issue in gan_health['issues']:
            print(f"  - {issue}")

    # Speaker similarity
    if 'speaker_similarity' in metrics:
        print("\n[Speaker Similarity]")
        sim = metrics['speaker_similarity']
        print(f"  Mean: {sim['mean']:.4f}")
        print(f"  Std:  {sim['std']:.4f}")
        print(f"  Acc:  {sim['accuracy']:.4f} (threshold=0.85)")

    print("="*70 + "\n")


if __name__ == "__main__":
    print("Training Diagnostics Module")
    print("Import this module in your training script to use:")
    print("  from scripts.diagnostics.monitor_training import generate_diagnostics_report")

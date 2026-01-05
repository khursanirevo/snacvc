"""
Phase 9: Simple Full SNAC Fine-tuning

No voice conversion, no adapters, no fancy losses.
Just fine-tune the entire SNAC model on custom data to improve reconstruction quality.

Goal: Preserve original capabilities while optimizing for our data distribution.
"""

import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from snac import SNAC

# Import utilities
from snac.dataset import OptimizedAudioDataset as SimpleAudioDataset, variable_length_collate, curriculum_collate


def get_curriculum_config(epoch, config):
    """
    Get segment length and batch size for current epoch based on curriculum.

    Returns:
        segment_length: Length in seconds for this epoch
        batch_size: Batch size for this epoch (scaled by segment length)
    """
    curriculum = config.get('curriculum', None)

    if curriculum is None:
        # No curriculum, use fixed values
        segment_length = config.get('segment_length', 2.0)
        if isinstance(segment_length, list):
            segment_length = segment_length[0]  # Use first if list
        batch_size = config['batch_size']
        return segment_length, batch_size

    # Curriculum format: {"1": [length, batch_multiplier], "2-3": [length, batch_multiplier], ...}
    # Or simplified: {"epochs": [1,2], "length": 1.0, "batch_multiplier": 2.0}
    current_epoch = epoch + 1  # 1-indexed for matching config

    # Check each curriculum entry
    for entry in curriculum:
        epochs = entry['epochs']
        length = entry['length']
        batch_mult = entry.get('batch_multiplier', 1.0)

        # Parse epoch range
        if isinstance(epochs, list):
            if current_epoch >= epochs[0] and current_epoch <= epochs[1]:
                base_batch = config['batch_size']
                batch_size = int(base_batch * batch_mult)
                return length, batch_size
        else:
            if current_epoch == epochs:
                base_batch = config['batch_size']
                batch_size = int(base_batch * batch_mult)
                return length, batch_size

    # Default: use last curriculum entry
    last_entry = curriculum[-1]
    length = last_entry['length']
    batch_mult = last_entry.get('batch_multiplier', 1.0)
    base_batch = config['batch_size']
    batch_size = int(base_batch * batch_mult)
    return length, batch_size


def create_dataloaders(train_dataset, val_dataset, segment_length, batch_size, num_workers, curriculum_mode=False):
    """Create dataloaders with appropriate collate function."""
    if curriculum_mode:
        # Fixed length per epoch
        collate_fn = curriculum_collate(train_dataset, segment_length)
        print(f"  Curriculum mode: {segment_length}s segments, batch_size={batch_size}")
    elif isinstance(segment_length, list):
        # Variable length per batch
        collate_fn = variable_length_collate(train_dataset)
        print(f"  Variable length mode: {segment_length}s (random per batch), batch_size={batch_size}")
    else:
        # Fixed length
        collate_fn = None
        print(f"  Fixed length mode: {segment_length}s, batch_size={batch_size}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def reconstruction_loss(audio, audio_hat, config):
    """Combined L1 + multi-scale STFT loss."""
    l1_weight = config.get('l1_weight', 1.0)
    stft_weight = config.get('stft_weight', 1.0)
    n_ffts = config.get('n_ffts', [1024, 2048, 4096])

    # L1 loss
    loss_l1 = F.l1_loss(audio_hat, audio)

    # Multi-scale STFT loss
    loss_stft = 0.0
    for n_fft in n_ffts:
        audio_stft = torch.stft(audio.squeeze(1), n_fft=n_fft, return_complex=True)
        audio_hat_stft = torch.stft(audio_hat.squeeze(1), n_fft=n_fft, return_complex=True)
        loss_stft += F.l1_loss(audio_hat_stft.abs(), audio_stft.abs())

    loss_stft = loss_stft / len(n_ffts)

    return l1_weight * loss_l1 + stft_weight * loss_stft, loss_l1, loss_stft


def train_epoch(model, dataloader, optimizer, device, config, epoch):
    """Simple training loop - just reconstruction."""
    model.train()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    total_loss = 0.0
    total_loss_l1 = 0.0
    total_loss_stft = 0.0
    num_batches = 0

    # EMA smoothing (fastai-style)
    ema_loss = None
    ema_l1 = None
    ema_stft = None
    smoothing = 0.98  # FastAI default

    for batch in pbar:
        audio = batch['audio'].to(device)
        audio = audio.unsqueeze(1)

        optimizer.zero_grad()

        # Forward
        audio_hat, codes = model(audio)

        # Reconstruction loss (decomposed)
        loss, loss_l1, loss_stft = reconstruction_loss(audio, audio_hat, config)

        # Backward
        loss.backward()

        # Gradient clipping
        if config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

        optimizer.step()

        # Metrics
        total_loss += loss.item()
        total_loss_l1 += loss_l1.item()
        total_loss_stft += loss_stft.item()
        num_batches += 1

        # Update EMA (smoothed losses for display)
        if ema_loss is None:
            ema_loss = loss.item()
            ema_l1 = loss_l1.item()
            ema_stft = loss_stft.item()
        else:
            ema_loss = smoothing * ema_loss + (1 - smoothing) * loss.item()
            ema_l1 = smoothing * ema_l1 + (1 - smoothing) * loss_l1.item()
            ema_stft = smoothing * ema_stft + (1 - smoothing) * loss_stft.item()

        # Progress (EMA-smoothed, decomposed)
        postfix = {
            'loss': ema_loss,
            'l1': ema_l1,
            'stft': ema_stft
        }
        pbar.set_postfix(postfix)

    return {
        'loss': total_loss / num_batches,
        'l1': total_loss_l1 / num_batches,
        'stft': total_loss_stft / num_batches,
    }


def validate(model, dataloader, device, config):
    """Simple validation with EMA-smoothed display."""
    model.eval()

    total_loss = 0.0
    total_loss_l1 = 0.0
    total_loss_stft = 0.0
    num_batches = 0

    # EMA smoothing
    ema_loss = None
    ema_l1 = None
    ema_stft = None
    smoothing = 0.98

    pbar = tqdm(dataloader, desc="Validation")

    with torch.no_grad():
        for batch in pbar:
            audio = batch['audio'].to(device).unsqueeze(1)

            audio_hat, codes = model(audio)
            loss, loss_l1, loss_stft = reconstruction_loss(audio, audio_hat, config)

            total_loss += loss.item()
            total_loss_l1 += loss_l1.item()
            total_loss_stft += loss_stft.item()
            num_batches += 1

            # Update EMA
            if ema_loss is None:
                ema_loss = loss.item()
                ema_l1 = loss_l1.item()
                ema_stft = loss_stft.item()
            else:
                ema_loss = smoothing * ema_loss + (1 - smoothing) * loss.item()
                ema_l1 = smoothing * ema_l1 + (1 - smoothing) * loss_l1.item()
                ema_stft = smoothing * ema_stft + (1 - smoothing) * loss_stft.item()

            # Progress (EMA-smoothed)
            pbar.set_postfix({
                'loss': ema_loss,
                'l1': ema_l1,
                'stft': ema_stft
            })

    return {
        'val_loss': total_loss / num_batches,
        'val_l1': total_loss_l1 / num_batches,
        'val_stft': total_loss_stft / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Phase 9: Full SNAC Fine-tuning")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output dir
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained SNAC model
    print("\nLoading pretrained SNAC model...")
    model = SNAC.from_pretrained(config['pretrained_model']).to(device)

    # Freeze layers if specified
    freeze_encoder = config.get('freeze_encoder', False)
    freeze_vq = config.get('freeze_vq', False)

    if freeze_encoder:
        print("Freezing ENCODER...")
        for param in model.encoder.parameters():
            param.requires_grad = False

    if freeze_vq:
        print("Freezing VQ (quantizer)...")
        for param in model.quantizer.parameters():
            param.requires_grad = False

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")

    # Optimizer (only trainable parameters)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=config['weight_decay']
    )

    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=config['learning_rate'] * config.get('lr_min_ratio', 0.1)
    )

    # Datasets
    print("\nLoading datasets...")
    # Determine min segment length for filtering
    segment_length_config = config.get('segment_length', 2.0)
    if 'curriculum' in config:
        # Use minimum length from curriculum for filtering
        min_length = min(entry['length'] for entry in config['curriculum'])
    elif isinstance(segment_length_config, list):
        min_length = min(segment_length_config)
    else:
        min_length = segment_length_config

    train_dataset = SimpleAudioDataset(
        config['train_data'],
        sampling_rate=24000,
        segment_length=min_length,  # Use min length for filtering
        augment=True,
    )
    val_dataset = SimpleAudioDataset(
        config['val_data'],
        sampling_rate=24000,
        segment_length=min_length,  # Use min length for filtering
        augment=False,
    )

    # Check if curriculum mode
    curriculum_mode = 'curriculum' in config
    if curriculum_mode:
        print(f"Curriculum learning enabled:")
        for entry in config['curriculum']:
            epochs = entry['epochs']
            length = entry['length']
            batch_mult = entry.get('batch_multiplier', 1.0)
            if isinstance(epochs, list):
                print(f"  Epochs {epochs[0]}-{epochs[1]}: {length}s, batch_mult={batch_mult}x")
            else:
                print(f"  Epoch {epochs}: {length}s, batch_mult={batch_mult}x")

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    print("\n" + "="*70)
    print("Phase 9: Full SNAC Fine-tuning")
    print("  Goal: Better reconstruction on our data")
    print("  Method: Fine-tune ALL parameters")
    print("  Loss: Simple reconstruction (L1 + STFT)")
    print("="*70 + "\n")

    # Create initial dataloaders
    segment_length, batch_size = get_curriculum_config(start_epoch, config)
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, segment_length, batch_size,
        config['num_workers'], curriculum_mode=curriculum_mode
    )

    # IMPORTANT: Run validation BEFORE training to establish baseline
    print("📊 Establishing BASELINE validation metrics (before any training)...")
    val_metrics_baseline = validate(model, val_loader, device, config)
    print(f"\n✅ BASELINE (pre-training):")
    print(f"  Val loss: {val_metrics_baseline['val_loss']:.4f}")
    print(f"    - L1: {val_metrics_baseline['val_l1']:.4f}")
    print(f"    - STFT: {val_metrics_baseline['val_stft']:.4f}")

    # Cache baseline
    baseline_path = output_dir / "baseline_metrics.json"
    with open(baseline_path, 'w') as f:
        json.dump(val_metrics_baseline, f, indent=2)
    print(f"  Cached: {baseline_path}")

    print("\n" + "="*70)
    print("Starting training...")
    print("="*70 + "\n")

    for epoch in range(start_epoch, config['num_epochs']):
        # Update dataloaders for curriculum learning
        if curriculum_mode:
            segment_length, batch_size = get_curriculum_config(epoch, config)
            print(f"\n🔄 Creating dataloaders for epoch {epoch+1}:")
            train_loader, val_loader = create_dataloaders(
                train_dataset, val_dataset, segment_length, batch_size,
                config['num_workers'], curriculum_mode=True
            )

        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.2e}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, config, epoch
        )

        # Validate
        val_metrics = validate(model, val_loader, device, config)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train: loss={train_metrics['loss']:.4f} (l1={train_metrics['l1']:.4f}, stft={train_metrics['stft']:.4f})")
        print(f"  Val:   loss={val_metrics['val_loss']:.4f} (l1={val_metrics['val_l1']:.4f}, stft={val_metrics['val_stft']:.4f})")
        print(f"  vs Baseline: Δ={val_metrics['val_loss'] - val_metrics_baseline['val_loss']:.4f}")

        # Step scheduler
        scheduler.step()

        # Save checkpoint
        val_loss = val_metrics['val_loss']
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if (epoch + 1) % config.get('save_every', 2) == 0 or is_best:
            checkpoint_path = output_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
            }, checkpoint_path)
            print(f"Saved: {checkpoint_path}")

            if is_best:
                best_path = output_dir / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'config': config,
                }, best_path)
                print(f"Saved best model: {best_path}")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

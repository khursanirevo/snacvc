"""
Phase 10: Simple Full SNAC Fine-tuning

No voice conversion, no adapters, no fancy losses.
Just fine-tune the entire SNAC model on custom data to improve reconstruction quality.

Goal: Preserve original capabilities while optimizing for our data distribution.
"""

import os
import json
import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from snac import SNAC

# Import utilities
from snac.dataset import OptimizedAudioDataset as SimpleAudioDataset, variable_length_collate, curriculum_collate


def setup_logging(config):
    """Setup logging to file based on experiment name."""
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Use experiment_name from config, fallback to a timestamp
    experiment_name = config.get('experiment_name', config.get('output_dir', 'training').split('/')[-1])

    # Create experiment-specific log directory
    log_dir = logs_dir / experiment_name
    log_dir.mkdir(exist_ok=True)

    # Setup log file
    log_file = log_dir / "training.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )

    return logging.getLogger(__name__)


def get_curriculum_config(epoch, config):
    """
    Get segment length and batch size for current epoch based on curriculum.

    Returns:
        segment_length: Length in seconds for this epoch
        batch_size: Batch size for this epoch (scaled by segment length)
        batch_multiplier: The multiplier used (for scaling validation batch size)
    """
    curriculum = config.get('curriculum', None)

    if curriculum is None:
        # No curriculum, use fixed values
        segment_length = config.get('segment_length', 2.0)
        if isinstance(segment_length, list):
            segment_length = segment_length[0]  # Use first if list
        batch_size = config['batch_size']
        return segment_length, batch_size, 1.0

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
                return length, batch_size, batch_mult
        else:
            if current_epoch == epochs:
                base_batch = config['batch_size']
                batch_size = int(base_batch * batch_mult)
                return length, batch_size, batch_mult

    # Default: use last curriculum entry
    last_entry = curriculum[-1]
    length = last_entry['length']
    batch_mult = last_entry.get('batch_multiplier', 1.0)
    base_batch = config['batch_size']
    batch_size = int(base_batch * batch_mult)
    return length, batch_size, batch_mult


def update_dataloader_batch_size(dataloader, new_batch_size):
    """Update the batch size of a dataloader by recreating its batch sampler."""
    from torch.utils.data.sampler import BatchSampler

    # Get the original sampler (likely a RandomSampler)
    base_sampler = dataloader.batch_sampler.sampler

    # Create new batch sampler with updated batch size
    new_batch_sampler = BatchSampler(
        base_sampler,
        batch_size=new_batch_size,
        drop_last=True
    )

    # Replace the batch sampler
    dataloader.batch_sampler = new_batch_sampler


def create_dataloaders(train_dataset, val_dataset, segment_length, batch_size, num_workers, curriculum_mode=False, persistent_workers=False, eval_batch_size=None):
    """Create dataloaders with appropriate collate function."""
    # Use eval_batch_size for validation if specified, otherwise use batch_size
    val_batch_size = eval_batch_size if eval_batch_size is not None else batch_size

    # Calculate workers: half of batch size, max 8
    actual_workers = min(batch_size // 2, 8)

    if curriculum_mode:
        # Fixed length per epoch
        collate_fn = curriculum_collate(train_dataset, segment_length)
        print(f"  Curriculum mode: {segment_length}s segments, batch_size={batch_size}, val_batch_size={val_batch_size}, workers={actual_workers}")
    elif isinstance(segment_length, list):
        # Variable length per batch
        collate_fn = variable_length_collate(train_dataset)
        print(f"  Variable length mode: {segment_length}s (random per batch), batch_size={batch_size}, val_batch_size={val_batch_size}, workers={actual_workers}")
    else:
        # Fixed length
        collate_fn = None
        print(f"  Fixed length mode: {segment_length}s, batch_size={batch_size}, val_batch_size={val_batch_size}, workers={actual_workers}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=actual_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers if actual_workers > 0 else False,
        drop_last=True,  # Drop incomplete batches to ensure consistent batch size
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=actual_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=persistent_workers if actual_workers > 0 else False,
        drop_last=True,  # Drop incomplete batches to ensure consistent batch size
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


def train_epoch(model, dataloader, optimizer, scheduler, device, config, epoch, global_batch_count=0,
                 output_dir=None, last_checkpoint_time=None, best_val_loss=None):
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

    # Checkpoint settings
    save_every_batches = config.get('save_every_batches', None)
    save_every_minutes = config.get('save_every_minutes', None)

    for batch in pbar:
        audio = batch['audio']

        # Skip empty batches (from corrupted files)
        if audio.numel() == 0 or audio.shape[0] == 0:
            continue

        audio = audio.to(device)
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

        # Increment global batch count
        global_batch_count += 1
        num_batches += 1

        # Checkpoint: save every N batches
        if save_every_batches and global_batch_count % save_every_batches == 0 and output_dir:
            try:
                checkpoint_path = output_dir / f"checkpoint_batch{global_batch_count}.pt"
                checkpoint_state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'config': config,
                    'global_batch_count': global_batch_count,
                }
                torch.save(checkpoint_state, checkpoint_path)
                pbar.write(f"  💾 Saved checkpoint at batch {global_batch_count}: {checkpoint_path}")
                del checkpoint_state
            except Exception as e:
                pbar.write(f"  ⚠️  Failed to save checkpoint at batch {global_batch_count}: {e}")

        # Checkpoint: save every N minutes
        if save_every_minutes and output_dir:
            current_time = time.time()
            if current_time - last_checkpoint_time >= save_every_minutes * 60:
                try:
                    # Use unique filename with both time and batch count to prevent collisions
                    checkpoint_path = output_dir / f"checkpoint_time{int(current_time)}_batch{global_batch_count}.pt"
                    checkpoint_state = {
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'config': config,
                        'global_batch_count': global_batch_count,
                    }
                    torch.save(checkpoint_state, checkpoint_path)
                    pbar.write(f"  💾 Saved time-based checkpoint: {checkpoint_path}")
                    del checkpoint_state
                    last_checkpoint_time = current_time
                except Exception as e:
                    pbar.write(f"  ⚠️  Failed to save time-based checkpoint: {e}")

        # Metrics
        total_loss += loss.item()
        total_loss_l1 += loss_l1.item()
        total_loss_stft += loss_stft.item()

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

    # Handle edge case where all batches were skipped
    if num_batches == 0:
        pbar.write("⚠️  Warning: No valid batches processed in this epoch!")
        return {
            'loss': float('inf'),
            'l1': float('inf'),
            'stft': float('inf'),
        }, global_batch_count, last_checkpoint_time

    return {
        'loss': total_loss / num_batches,
        'l1': total_loss_l1 / num_batches,
        'stft': total_loss_stft / num_batches,
    }, global_batch_count, last_checkpoint_time


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
            audio = batch['audio']

            # Skip empty batches (from corrupted files)
            if audio.numel() == 0 or audio.shape[0] == 0:
                continue

            audio = audio.to(device).unsqueeze(1)

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

    # Handle edge case where all batches were skipped
    if num_batches == 0:
        pbar.write("⚠️  Warning: No valid batches in validation!")
        return {
            'val_loss': float('inf'),
            'val_l1': float('inf'),
            'val_stft': float('inf'),
        }

    return {
        'val_loss': total_loss / num_batches,
        'val_l1': total_loss_l1 / num_batches,
        'val_stft': total_loss_stft / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Phase 10: Full SNAC Fine-tuning")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    # Setup logging FIRST (before any logger usage)
    logger = setup_logging(config)
    logger.info(f"Config: {args.config}")
    logger.info(f"Experiment: {config.get('experiment_name', 'unknown')}")

    # Set random seed for reproducibility
    random_seed = config.get('random_seed', 42)
    import random
    import numpy as np
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    logger.info(f"Random seed set to: {random_seed}")

    # Validate config parameters
    assert config.get('num_epochs', 1) > 0, "num_epochs must be positive"
    assert config.get('learning_rate', 0) > 0, "learning_rate must be positive"
    assert config.get('batch_size', 1) > 0, "batch_size must be positive"
    assert config.get('weight_decay', 0) >= 0, "weight_decay must be non-negative"
    logger.info("✅ Config validation passed")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output dir
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained SNAC model
    logger.info("\nLoading pretrained SNAC model...")
    model = SNAC.from_pretrained(config['pretrained_model']).to(device)

    # Freeze layers if specified
    freeze_encoder = config.get('freeze_encoder', False)
    freeze_vq = config.get('freeze_vq', False)

    if freeze_encoder:
        logger.info("Freezing ENCODER...")
        for param in model.encoder.parameters():
            param.requires_grad = False

    if freeze_vq:
        logger.info("Freezing VQ (quantizer)...")
        for param in model.quantizer.parameters():
            param.requires_grad = False

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    logger.info(f"Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")

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
    logger.info("\nLoading datasets...")

    # Validate dataset paths
    train_path = Path(config['train_data'])
    val_path = Path(config['val_data'])

    if not train_path.exists():
        raise FileNotFoundError(f"Training data path does not exist: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data path does not exist: {val_path}")

    # Check for empty directories
    train_files = list(train_path.glob('*.wav')) + list(train_path.glob('*.mp3')) + list(train_path.glob('*.flac'))
    val_files = list(val_path.glob('*.wav')) + list(val_path.glob('*.mp3')) + list(val_path.glob('*.flac'))

    if len(train_files) == 0:
        raise ValueError(f"Training data directory is empty (no audio files found): {train_path}")
    if len(val_files) == 0:
        raise ValueError(f"Validation data directory is empty (no audio files found): {val_path}")

    logger.info(f"✅ Train data path: {train_path} ({len(train_files)} files)")
    logger.info(f"✅ Val data path: {val_path} ({len(val_files)} files)")


    # Check if curriculum mode (BEFORE creating datasets)
    curriculum_mode = 'curriculum' in config

    # Determine min segment length for filtering
    segment_length_config = config.get('segment_length', 2.0)
    if curriculum_mode:
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
        curriculum_mode=curriculum_mode,  # Enable idx returns for curriculum
    )
    val_dataset = SimpleAudioDataset(
        config['val_data'],
        sampling_rate=24000,
        segment_length=min_length,  # Use min length for filtering
        augment=False,
        curriculum_mode=curriculum_mode,  # Enable idx returns for curriculum
    )
    if curriculum_mode:
        logger.info(f"Curriculum learning enabled:")
        for entry in config['curriculum']:
            epochs = entry['epochs']
            length = entry['length']
            batch_mult = entry.get('batch_multiplier', 1.0)
            if isinstance(epochs, list):
                logger.info(f"  Epochs {epochs[0]}-{epochs[1]}: {length}s, batch_mult={batch_mult}x")
            else:
                logger.info(f"  Epoch {epochs}: {length}s, batch_mult={batch_mult}x")

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    val_metrics_baselines = {}  # Track separate baselines per curriculum stage (segment_length)
    global_batch_count = 0  # Track total batches across all epochs
    last_checkpoint_time = time.time()  # Track time for time-based checkpointing

    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Validate checkpoint structure
            required_keys = ['model', 'optimizer', 'scheduler', 'epoch', 'config']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                raise ValueError(f"Checkpoint missing required keys: {missing_keys}")
            
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1  # Resume from NEXT epoch
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            global_batch_count = checkpoint.get('global_batch_count', 0)  # Resume batch count
            logger.info(f"Resumed from epoch {checkpoint['epoch']}, starting epoch {start_epoch}")
            logger.info(f"Resumed from global batch {global_batch_count}")
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint from {args.resume}: {e}")
            raise

        # Try to load original baselines from cache
        baseline_path = output_dir / "baseline_metrics.json"
        if baseline_path.exists():
            try:
                with open(baseline_path, 'r') as f:
                    val_metrics_baselines = json.load(f)
                # Validate structure
                if not isinstance(val_metrics_baselines, dict):
                    raise ValueError("Baseline cache is not a dict")
                for seg_key, metrics in val_metrics_baselines.items():
                    if not isinstance(metrics, dict) or 'val_loss' not in metrics:
                        raise ValueError(f"Invalid baseline structure for {seg_key}")
                logger.info(f"✅ Loaded baselines from cache: {baseline_path}")
                for seg_len, metrics in val_metrics_baselines.items():
                    logger.info(f"  {seg_len}s baseline val_loss: {metrics['val_loss']:.4f}")
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"⚠️  Failed to load baseline cache ({e}), will calculate new baselines")
                val_metrics_baselines = {}
        else:
            logger.info("⚠️  Warning: baseline_metrics.json not found, will calculate new baselines")
            logger.info("     (Δ vs baseline will be calculated per curriculum stage)")

    # Training loop
    logger.info("\n" + "="*70)
    logger.info("Phase 10: Full SNAC Fine-tuning")
    logger.info("  Goal: Better reconstruction on our data")
    logger.info("  Method: Fine-tune ALL parameters")
    logger.info("  Loss: Simple reconstruction (L1 + STFT)")
    logger.info("="*70 + "\n")

    # Create initial dataloaders (will be reused across epochs)
    segment_length, batch_size, _ = get_curriculum_config(start_epoch, config)
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset, segment_length, batch_size,
        config['num_workers'], curriculum_mode=curriculum_mode,
        persistent_workers=not curriculum_mode,  # Disable persistent workers with curriculum (collate_fn changes)
        eval_batch_size=config.get('eval_batch_size')  # Use separate batch size for validation
    )

    # IMPORTANT: Run validation BEFORE training to establish baseline
    # For both curriculum and non-curriculum modes
    seg_key = f"{segment_length}s"
    if not val_metrics_baselines:
        logger.info(f"📊 Establishing BASELINE validation metrics for {seg_key} segments (before any training)...")
        val_metrics_baselines[seg_key] = validate(model, val_loader, device, config)
        logger.info(f"\n✅ BASELINE ({seg_key} segments, pre-training):")
        logger.info(f"  Val loss: {val_metrics_baselines[seg_key]['val_loss']:.4f}")
        logger.info(f"    - L1: {val_metrics_baselines[seg_key]['val_l1']:.4f}")
        logger.info(f"    - STFT: {val_metrics_baselines[seg_key]['val_stft']:.4f}")

        # Cache baselines (save all stages)
        baseline_path = output_dir / "baseline_metrics.json"
        with open(baseline_path, 'w') as f:
            json.dump(val_metrics_baselines, f, indent=2)
        logger.info(f"  Cached: {baseline_path}")

    logger.info("\n" + "="*70)
    logger.info("Starting training...")
    logger.info("="*70 + "\n")

    for epoch in range(start_epoch, config['num_epochs']):
        # Get curriculum config (for both curriculum and non-curriculum modes)
        segment_length, batch_size, batch_mult = get_curriculum_config(epoch, config)

        # Update dataloaders for curriculum learning (just update collate, don't recreate!)
        if curriculum_mode:

            # Update collate functions (fast, no worker restart)
            train_loader.collate_fn = curriculum_collate(train_dataset, segment_length)
            val_loader.collate_fn = curriculum_collate(val_dataset, segment_length)

            # Update batch size (properly recreate batch sampler)
            # Scale eval_batch_size by same multiplier to maintain consistency
            eval_batch_size = config.get('eval_batch_size', batch_size)
            val_batch_size = int(eval_batch_size * batch_mult) if batch_mult > 0 else eval_batch_size
            update_dataloader_batch_size(train_loader, batch_size)
            update_dataloader_batch_size(val_loader, val_batch_size)

            logger.info(f"  Curriculum mode: {segment_length}s segments, batch_size={batch_size}, val_batch_size={val_batch_size}, workers={config['num_workers']}")

            # Calculate baseline for this curriculum stage if not yet done
            seg_key = f"{segment_length}s"
            if seg_key not in val_metrics_baselines:
                logger.info(f"📊 Establishing BASELINE for {seg_key} segments...")
                val_metrics_baselines[seg_key] = validate(model, val_loader, device, config)
                logger.info(f"  {seg_key} baseline val_loss: {val_metrics_baselines[seg_key]['val_loss']:.4f}")

                # Save updated baselines
                baseline_path = output_dir / "baseline_metrics.json"
                with open(baseline_path, 'w') as f:
                    json.dump(val_metrics_baselines, f, indent=2)
                logger.info(f"  Cached: {baseline_path}")

        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        logger.info(f"Learning rate: {current_lr:.2e}")

        # Train
        train_metrics, global_batch_count, last_checkpoint_time = train_epoch(
            model, train_loader, optimizer, scheduler, device, config, epoch,
            global_batch_count=global_batch_count,
            output_dir=output_dir,
            last_checkpoint_time=last_checkpoint_time,
            best_val_loss=best_val_loss
        )

        # Validate
        val_metrics = validate(model, val_loader, device, config)

        # Update scheduler
        scheduler.step()

        # Calculate improvement over baseline (use correct baseline for current curriculum stage)
        if curriculum_mode:
            seg_key = f"{segment_length}s"
            val_baseline = val_metrics_baselines[seg_key]
            baseline_info = f"({seg_key} baseline)"
        else:
            # Non-curriculum: use first baseline (calculated before training loop)
            seg_key = list(val_metrics_baselines.keys())[0]
            val_baseline = val_metrics_baselines[seg_key]
            baseline_info = f"(baseline)"

        improvement = val_baseline['val_loss'] - val_metrics['val_loss']
        # Avoid division by zero
        if val_baseline['val_loss'] == 0:
            improvement_pct = 0.0 if improvement == 0 else float('inf')
        else:
            improvement_pct = (improvement / val_baseline['val_loss']) * 100

        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Train loss: {train_metrics['loss']:.4f} (L1: {train_metrics['l1']:.4f}, STFT: {train_metrics['stft']:.4f})")
        logger.info(f"  Val loss:   {val_metrics['val_loss']:.4f} (L1: {val_metrics['val_l1']:.4f}, STFT: {val_metrics['val_stft']:.4f})")
        logger.info(f"  Δ vs baseline: {improvement:+.4f} ({improvement_pct:+.2f}%) {baseline_info}")

        # Save checkpoint
        if (epoch + 1) % config.get('save_every', 2) == 0:
            try:
                checkpoint_path = output_dir / f"checkpoint_epoch{epoch}.pt"
                checkpoint_state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'global_batch_count': global_batch_count,
                }
                torch.save(checkpoint_state, checkpoint_path)
                logger.info(f"  Saved: {checkpoint_path}")
                del checkpoint_state
            except Exception as e:
                logger.error(f"  ⚠️  Failed to save epoch checkpoint: {e}")

        # Save best model
        # Validate val_loss is finite before comparing
        if torch.isfinite(val_metrics['val_loss']) and val_metrics['val_loss'] < best_val_loss:
            old_best_val_loss = best_val_loss
            try:
                best_val_loss = val_metrics['val_loss']
                best_path = output_dir / "best_model.pt"
                checkpoint_state = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'global_batch_count': global_batch_count,
                }
                torch.save(checkpoint_state, best_path)
                logger.info(f"  ✅ New best model! ({best_val_loss:.4f})")
                del checkpoint_state
            except Exception as e:
                logger.error(f"  ⚠️  Failed to save best model checkpoint: {e}")
                # Revert best_val_loss if save failed
                best_val_loss = old_best_val_loss
        elif not torch.isfinite(val_metrics['val_loss']):
            logger.warning(f"  ⚠️  Warning: Validation loss is {val_metrics['val_loss']}, skipping best model check")

        # Cleanup GPU memory to prevent fragmentation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final summary
    logger.info("="*70)
    logger.info("Training complete!")
    logger.info("="*70)
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

    # Show improvement for each curriculum stage baseline
    logger.info(f"\nImprovement vs baselines:")
    for seg_key, baseline in val_metrics_baselines.items():
        improvement = baseline['val_loss'] - best_val_loss
        # Avoid division by zero
        if baseline['val_loss'] == 0:
            improvement_pct = 0.0 if improvement == 0 else float('inf')
        else:
            improvement_pct = (improvement / baseline['val_loss']) * 100
        logger.info(f"  vs {seg_key} baseline: {improvement:+.4f} ({improvement_pct:+.2f}%)")

    logger.info(f"\nCheckpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()

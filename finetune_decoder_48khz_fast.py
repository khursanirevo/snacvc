"""
Phase 11: SNAC Decoder 48kHz Training (Optimized with Pre-computed Data)

Fast version that uses:
1. Pre-computed quantized codes (no encoder+VQ forward pass)
2. Pre-computed 48kHz audio targets (no SIDON forward pass)

Benefits:
- 3-5x faster training (no encoder/VQ/SIDON forward passes)
- Less memory (don't load encoder/VQ/SIDON)
- Higher batch size possible
- Smaller checkpoints
- Reproducible (fixed precomputed data)

Prerequisites:
1. Run precompute_codes.py to generate quantized codes from 24kHz audio
2. Run precompute_48khz_audio.py to generate 48kHz audio using SIDON

Usage:
    python finetune_decoder_48khz_fast.py \
        --config configs/phase11_decoder_48khz.json \
        --codes_dir /mnt/data/codes_phase11 \
        --audio_48khz_dir /mnt/data/combine/train/audio_48khz \
        --device 0
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import torchaudio
import numpy as np
from tqdm import tqdm

from snac import SNAC
from snac.layers import Decoder, DecoderBlock, Snake1d
from snac.codes_dataset import PrecomputedCodesDataset, collate_fn


class Decoder48kHz(nn.Module):
    """
    Decoder that outputs 48kHz audio from quantized codes.

    Architecture:
        Quantized codes → Decoder (pretrained) → 24kHz features (96 channels)
                                                  ↓
                                            NEW 2x Upsampler
                                                  ↓
                                            Final Conv → 48kHz audio
    """
    def __init__(self, pretrained_decoder, quantizer, device='cuda'):
        super().__init__()
        self.pretrained_decoder = pretrained_decoder
        self.quantizer = quantizer
        self.device = device

        # NEW 2x upsampler: 64 → 64 (upsample by 2x in time)
        # Use transposed convolution for upsampling
        self.upsampler2x = nn.ConvTranspose1d(
            in_channels=64,  # Matches pretrained decoder output
            out_channels=64,
            kernel_size=4,  # 2x upsampling
            stride=2,
            padding=1
        )

        # Initialize with sensible weights
        nn.init.xavier_uniform_(self.upsampler2x.weight)
        if self.upsampler2x.bias is not None:
            nn.init.zeros_(self.upsampler2x.bias)

        # Final conv: 64 → 1
        self.final_conv = nn.Conv1d(64, 1, kernel_size=7, padding=3)

        # Initialize final conv
        nn.init.xavier_uniform_(self.final_conv.weight)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)

        print("\n✅ 48kHz upsampler initialized (random init, will be trained from scratch)")

    def forward(self, z_q):
        """Forward pass: quantized latent → 48kHz audio."""
        # Run pretrained decoder up to block 3 (before Snake1d and final conv)
        x = z_q
        for i, layer in enumerate(self.pretrained_decoder.model[:-2]):
            x = layer(x)

        # x is now (B, 64, T) at 24kHz rate

        # Apply new 2x upsampler
        x = self.upsampler2x(x)  # (B, 64, T*2) at 48kHz rate

        # Final conv to 1 channel
        audio = self.final_conv(x)  # (B, 1, T*2)

        # Apply tanh to clip to [-1, 1]
        audio = torch.tanh(audio)

        return audio

    def get_new_layers(self):
        """Return the new layers for warmup training."""
        return [self.upsampler2x, self.final_conv]


def reconstruction_loss(pred, target, config):
    """Compute reconstruction loss."""
    l1_loss = F.l1_loss(pred, target)

    stft_loss = 0.0
    n_ffts = config.get('n_ffts', [1024, 2048, 4096, 8192])

    for n_fft in n_ffts:
        pred_stft = torch.stft(pred.squeeze(1), n_fft=n_fft, return_complex=True)
        target_stft = torch.stft(target.squeeze(1), n_fft=n_fft, return_complex=True)
        stft_loss += F.l1_loss(pred_stft.abs(), target_stft.abs())

    stft_loss /= len(n_ffts)

    loss = config['l1_weight'] * l1_loss + config['stft_weight'] * stft_loss
    return loss, l1_loss, stft_loss


def train_epoch(model, train_loader, optimizer, scheduler, device, config, epoch):
    """Train for one epoch."""
    model.train()

    epoch_loss = 0.0
    epoch_l1 = 0.0
    epoch_stft = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for codes_batch, audio_48k_target in pbar:
        codes_batch = [c.to(device) for c in codes_batch]
        audio_48k_target = audio_48k_target.to(device)

        # Decode from codes (no encoder/VQ needed!)
        z_q = model.quantizer.from_codes(codes_batch)
        audio_48k_pred = model(z_q)

        # Trim to match lengths
        min_len = min(audio_48k_pred.shape[-1], audio_48k_target.shape[-1])
        audio_48k_pred = audio_48k_pred[..., :min_len]
        audio_48k_target = audio_48k_target[..., :min_len]

        # Compute loss
        loss, l1_loss, stft_loss = reconstruction_loss(audio_48k_pred, audio_48k_target, config)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        if config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

        optimizer.step()

        epoch_loss += loss.item()
        epoch_l1 += l1_loss.item()
        epoch_stft += stft_loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'l1': f'{l1_loss.item():.4f}', 'stft': f'{stft_loss.item():.4f}'})

    return {'loss': epoch_loss / num_batches, 'l1': epoch_l1 / num_batches, 'stft': epoch_stft / num_batches}


@torch.no_grad()
def validate(model, val_loader, device, config):
    """Validate the model."""
    model.eval()

    val_loss = 0.0
    val_l1 = 0.0
    val_stft = 0.0
    num_batches = 0

    for codes_batch, audio_48k_target in tqdm(val_loader, desc="Validation"):
        codes_batch = [c.to(device) for c in codes_batch]
        audio_48k_target = audio_48k_target.to(device)

        z_q = model.quantizer.from_codes(codes_batch)
        audio_48k_pred = model(z_q)

        min_len = min(audio_48k_pred.shape[-1], audio_48k_target.shape[-1])
        audio_48k_pred = audio_48k_pred[..., :min_len]
        audio_48k_target = audio_48k_target[..., :min_len]

        loss, l1_loss, stft_loss = reconstruction_loss(audio_48k_pred, audio_48k_target, config)

        val_loss += loss.item()
        val_l1 += l1_loss.item()
        val_stft += stft_loss.item()
        num_batches += 1

    return {'val_loss': val_loss / num_batches, 'val_l1': val_l1 / num_batches, 'val_stft': val_stft / num_batches}


def setup_logging(config):
    """Setup logging."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    experiment_name = config.get('experiment_name', 'training')
    log_dir = logs_dir / experiment_name
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / "training.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train 48kHz decoder with precomputed data")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--codes_dir", type=str, required=True,
                        help="Directory with precomputed codes")
    parser.add_argument("--audio_48khz_dir", type=str, required=True,
                        help="Directory with precomputed 48kHz audio")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    logger = setup_logging(config)
    logger.info(f"Config: {args.config}")
    logger.info(f"Experiment: {config.get('experiment_name', 'unknown')}")

    random_seed = config.get('random_seed', 42)
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained SNAC 24kHz (decoder only!)
    logger.info("\nLoading pretrained SNAC 24kHz decoder...")
    pretrained_model = SNAC.from_pretrained(config['pretrained_model'])
    pretrained_decoder = pretrained_model.decoder.to(device)

    logger.info("✓ Decoder loaded")

    # Create 48kHz decoder wrapper with quantizer
    logger.info("\nCreating 48kHz decoder with smart initialization...")
    model_48k = Decoder48kHz(pretrained_decoder, pretrained_model.quantizer, device).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model_48k.parameters())
    trainable_params = sum(p.numel() for p in model_48k.parameters() if p.requires_grad)

    logger.info(f"\nModel parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    # Datasets with precomputed data and random segment extraction at code level
    logger.info("\nLoading precomputed datasets with code-level random segments...")
    segment_length = config.get('segment_length', 4.0)

    train_dataset = PrecomputedCodesDataset(
        f"{args.codes_dir}/train",
        f"{args.audio_48khz_dir}",  # No /train/ suffix - files are already here
        segment_length_sec=segment_length,
    )
    val_dataset = PrecomputedCodesDataset(
        f"{args.codes_dir}/val",
        f"{args.audio_48khz_dir}".replace("train", "valid"),  # Use valid directory
        segment_length_sec=segment_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('eval_batch_size', config['batch_size']),
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Training config
    warmup_epochs = config.get('warmup_epochs', 3)
    warmup_lr = config.get('warmup_learning_rate', 5e-5)
    main_lr = config['learning_rate']
    total_epochs = config['num_epochs']

    logger.info("\n" + "="*70)
    logger.info("Phase 11: Two-Phase Training with Precomputed Data")
    logger.info("="*70)
    logger.info(f"\nPhase 1 - Warmup (epochs 1-{warmup_epochs}):")
    logger.info(f"  Train: New 2x upsampler + final conv only")
    logger.info(f"  Learning rate: {warmup_lr:.2e}")
    logger.info(f"\nPhase 2 - Main (epochs {warmup_epochs+1}-{total_epochs}):")
    logger.info(f"  Train: Entire decoder")
    logger.info(f"  Learning rate: {main_lr:.2e}")
    logger.info("="*70 + "\n")

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model_48k.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Resumed from epoch {checkpoint['epoch']}")

    for epoch in range(start_epoch, total_epochs):
        # Phase management
        if epoch < warmup_epochs:
            if epoch == 0:
                logger.info("\n" + "="*70)
                logger.info("PHASE 1: WARMUP - Training new layers only")
                logger.info("="*70 + "\n")

                # Freeze pretrained decoder
                for param in model_48k.pretrained_decoder.parameters():
                    param.requires_grad = False

                optimizer = AdamW(
                    filter(lambda p: p.requires_grad, model_48k.parameters()),
                    lr=warmup_lr,
                    betas=(0.9, 0.999),
                    weight_decay=config['weight_decay']
                )

                # Create warmup scheduler (runs for warmup_epochs)
                scheduler = CosineAnnealingLR(optimizer, T_max=warmup_epochs, eta_min=warmup_lr * 0.1)

            phase = "Warmup"
        elif epoch == warmup_epochs:
            logger.info("\n" + "="*70)
            logger.info("PHASE 2: MAIN - Unfreezing entire decoder")
            logger.info("="*70 + "\n")

            # Unfreeze entire decoder
            for param in model_48k.parameters():
                param.requires_grad = True

            optimizer = AdamW(
                model_48k.parameters(),
                lr=main_lr,
                betas=(0.9, 0.999),
                weight_decay=config['weight_decay']
            )

            # Create main scheduler (runs for remaining epochs)
            scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=main_lr * 0.1)

            phase = "Main"

        logger.info(f"\nEpoch {epoch + 1}/{total_epochs} [{phase}]")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        train_metrics = train_epoch(model_48k, train_loader, optimizer, scheduler, device, config, epoch)
        val_metrics = validate(model_48k, val_loader, device, config)

        scheduler.step()

        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Train loss: {train_metrics['loss']:.4f} (L1: {train_metrics['l1']:.4f}, STFT: {train_metrics['stft']:.4f})")
        logger.info(f"  Val loss:   {val_metrics['val_loss']:.4f} (L1: {val_metrics['val_l1']:.4f}, STFT: {val_metrics['val_stft']:.4f})")

        # Save checkpoint
        if (epoch + 1) % config.get('save_every', 2) == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model': model_48k.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
            }, checkpoint_path)
            logger.info(f"  Saved: {checkpoint_path}")

        # Save best
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            best_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model': model_48k.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
            }, best_path)
            logger.info(f"  ✅ New best model! ({best_val_loss:.4f})")

    logger.info("\n" + "="*70)
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()

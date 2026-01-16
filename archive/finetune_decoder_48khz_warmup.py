"""
Phase 11: SNAC Decoder 48kHz Output with Smart Initialization

Architecture:
  Old Decoder (frozen, pretrained): [8,8,4,2] → 24kHz output
                                              ↓
                                    NEW 2x Upsampler (copied weights)
                                              ↓
                                    Final Conv → 48kHz output

Two-phase training:
1. Warmup: Train only the new 2x upsampler + final conv
2. Main: Unfreeze entire decoder for end-to-end fine-tuning
"""

import os
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchaudio
import numpy as np
from tqdm import tqdm

from snac import SNAC
from snac.layers import Decoder, DecoderBlock, Snake1d
from finetune_decoder_48khz import SIDONUpsampler


def batch_sidon(sidon, audio_batch: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Process a batch of audio through SIDON (which doesn't support batching).

    Args:
        sidon: SIDONUpsampler instance
        audio_batch: (B, 1, T) tensor
        sample_rate: Original sample rate

    Returns:
        (B, 1, T_48k) tensor of 48kHz audio
    """
    batch_size = audio_batch.shape[0]
    outputs = []

    for i in range(batch_size):
        # Extract single sample: (B, 1, T) -> (1, T) -> (1, 1, T)
        audio_single = audio_batch[i].unsqueeze(0)  # (1, T)

        # Process through SIDON
        _, audio_48k = sidon(audio_single, sample_rate=sample_rate)  # Returns (1, 1, T_48k)

        # Remove extra batch dim added by SIDON: (1, 1, T) -> (1, T)
        outputs.append(audio_48k.squeeze(0))

    # Stack back to batch: list of (1, T) -> (B, 1, T)
    return torch.stack(outputs, dim=0)


def collate_fn(batch):
    """Collate function that filters out None values (short/corrupted files)."""
    # Filter out None values
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        # Return empty batch (should be rare)
        return {'audio': torch.empty(0)}

    # Stack audio tensors and add channel dimension (B, T) -> (B, 1, T)
    audio_batch = torch.stack([item['audio'] for item in batch])
    audio_batch = audio_batch.unsqueeze(1)  # Add channel dimension
    return {'audio': audio_batch}


class Decoder48kHz(nn.Module):
    """
    Wrapper that adds a 2x upsampler to the pretrained 24kHz decoder.

    Architecture:
        pretrained_decoder (frozen) → 24kHz features → 2x upsampler → 48kHz audio
    """
    def __init__(self, pretrained_snac, device='cuda'):
        super().__init__()
        self.device = device

        # Store references to encoder, decoder, and quantizer from full SNAC model
        self.encoder = pretrained_snac.encoder
        self.quantizer = pretrained_snac.quantizer
        self.pretrained_decoder = pretrained_snac.decoder

        # The decoder structure (for 24kHz SNAC):
        # Layer 5 (last DecoderBlock): input_dim=128, output_dim=64, stride=3
        # Layer 6: Snake1d
        # Layer 7: Conv 64->1
        # Layer 8: Tanh
        #
        # We extract features after layer 5 (before Snake1d and final conv)
        # which gives us 64 channels at 24kHz rate
        #
        # We'll add:
        # 1. A 2x upsampler: 64 → 64 (upsample by 2x in time)
        # 2. A final Conv1d: 64 → 1 for mono audio

        # New 2x upsampler: 64 → 64 (upsample by 2x)
        self.upsampler2x = DecoderBlock(
            input_dim=64,      # Match last decoder block output
            output_dim=64,     # Keep same channel count
            stride=2,          # 2x upsampling
            noise=False,       # No noise for upsampler
            groups=1
        )

        # Final conv: 64 → 1 (for mono audio)
        self.final_conv = nn.Conv1d(64, 1, kernel_size=7, padding=3)

        # Snake1d activation for after upsampler
        self.snake1d = Snake1d(64)

        # Copy weights from pretrained decoder's last DecoderBlock (layer 5)
        print("\nCopying weights from pretrained decoder (layer 5)...")
        pretrained_last_block = pretrained_snac.decoder.model[5]  # Last DecoderBlock

        # Copy conv transpose weights
        if hasattr(pretrained_last_block, 'block'):
            for sublayer in pretrained_last_block.block:
                if hasattr(sublayer, 'in_channels') and hasattr(sublayer, 'out_channels'):
                    if sublayer.in_channels == 128 and sublayer.out_channels == 64:
                        # Found the conv transpose layer
                        if hasattr(self.upsampler2x, 'block'):
                            for new_sublayer in self.upsampler2x.block:
                                if hasattr(new_sublayer, 'weight'):
                                    # Can't directly copy due to dimension mismatch (stride 3 vs 2)
                                    # Initialize with xavier instead
                                    nn.init.xavier_uniform_(new_sublayer.weight)
                                    if hasattr(new_sublayer, 'bias') and new_sublayer.bias is not None:
                                        nn.init.zeros_(new_sublayer.bias)
                        break

        # Initialize upsampler and final conv
        nn.init.xavier_uniform_(self.final_conv.weight)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)

        print("✅ 48kHz upsampler initialized (random init, will be trained from scratch)")

    def forward(self, z_q):
        """
        Forward pass through decoder.

        Args:
            z_q: Quantized latent (B, latent_dim, T)

        Returns:
            Audio at 48kHz (B, 1, T*2)
        """
        # Get 24kHz features from pretrained decoder
        # Decoder structure: 0-5=DecoderBlocks, 6=Snake1d, 7=Conv, 8=Tanh
        # Run through layers 0-5 only (stop before Snake1d and final conv)
        x = z_q
        for layer in self.pretrained_decoder.model[:6]:  # Layers 0-5 (DecoderBlocks)
            x = layer(x)

        # x is now (B, 64, T) at 24kHz rate

        # Apply our new 2x upsampler
        x = self.upsampler2x(x)  # (B, 64, T*2) at 48kHz rate

        # Apply activation (Snake1d)
        x = self.snake1d(x)

        # Final conv to 1 channel
        audio = self.final_conv(x)  # (B, 1, T*2)

        # Apply tanh to clip to [-1, 1]
        audio = torch.tanh(audio)

        return audio

    def get_new_layers(self):
        """Return the new layers for warmup training."""
        return [self.upsampler2x, self.final_conv]

    def get_pretrained_decoder(self):
        """Return the pretrained decoder."""
        return self.pretrained_decoder


def freeze_new_model(model):
    """Freeze everything, prepare for warmup."""
    # Freeze pretrained decoder
    for param in model.pretrained_decoder.parameters():
        param.requires_grad = False

    # Unfreeze new layers
    for param in model.upsampler2x.parameters():
        param.requires_grad = True
    for param in model.final_conv.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen pretrained decoder")
    print(f"Trainable (new layers only): {trainable:,} params")


def unfreeze_decoder(model):
    """Unfreeze entire decoder for main training."""
    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters())
    print(f"Trainable (full decoder): {trainable:,} params")


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


def train_epoch(model, train_loader, sidon, optimizer, scheduler, device, config, epoch):
    """Train for one epoch."""
    model.train()
    sidon.fe.eval()
    sidon.decoder.eval()

    epoch_loss = 0.0
    epoch_l1 = 0.0
    epoch_stft = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for batch in pbar:
        # Handle empty batches (all files in batch were None)
        if batch['audio'].numel() == 0:
            continue

        audio_24k = batch['audio'].to(device)

        # Generate 48kHz target using SIDON (process batch one-by-one)
        with torch.no_grad():
            audio_48k_target = batch_sidon(sidon, audio_24k, sample_rate=24000)

        # Move to device if needed (batch_sidon returns on same device as input)
        if audio_48k_target.device != device:
            audio_48k_target = audio_48k_target.to(device)

        # Forward through SNAC (encoder + VQ)
        with torch.no_grad():
            z = model.encoder(audio_24k)
            z_q, codes = model.quantizer(z)

        # Decode to 48kHz
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
def validate(model, val_loader, sidon, device, config):
    """Validate the model."""
    model.eval()
    sidon.fe.eval()
    sidon.decoder.eval()

    val_loss = 0.0
    val_l1 = 0.0
    val_stft = 0.0
    num_batches = 0

    for batch in tqdm(val_loader, desc="Validation"):
        # Handle empty batches (all files in batch were None)
        if batch['audio'].numel() == 0:
            continue

        audio_24k = batch['audio'].to(device)

        # Generate 48kHz target using SIDON (process batch one-by-one)
        audio_48k_target = batch_sidon(sidon, audio_24k, sample_rate=24000)

        # Move to device if needed (batch_sidon returns on same device as input)
        if audio_48k_target.device != device:
            audio_48k_target = audio_48k_target.to(device)

        with torch.no_grad():
            z = model.encoder(audio_24k)
            z_q, codes = model.quantizer(z)

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
    parser = argparse.ArgumentParser(description="Train 48kHz decoder with smart init")
    parser.add_argument("--config", type=str, required=True)
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

    # Initialize SIDON
    logger.info("\nLoading SIDON upsampler...")
    sidon = SIDONUpsampler(device)
    logger.info("✓ SIDON loaded")

    # Load pretrained SNAC 24kHz
    logger.info("\nLoading pretrained SNAC 24kHz model...")
    pretrained_model = SNAC.from_pretrained(config['pretrained_model']).to(device)

    # Freeze encoder and VQ
    logger.info("Freezing encoder and VQ...")
    for param in pretrained_model.encoder.parameters():
        param.requires_grad = False
    for param in pretrained_model.quantizer.parameters():
        param.requires_grad = False
    logger.info("✓ Encoder and VQ frozen")

    # Create 48kHz decoder wrapper (pass full SNAC model)
    logger.info("\nCreating 48kHz decoder with smart initialization...")
    model_48k = Decoder48kHz(pretrained_model, device).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model_48k.parameters())
    frozen_params = sum(p.numel() for p in model_48k.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model_48k.parameters() if p.requires_grad)

    logger.info(f"\nModel parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    logger.info(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")

    # Datasets
    logger.info("\nLoading datasets...")
    from snac.dataset import OptimizedAudioDataset

    train_dataset = OptimizedAudioDataset(
        config['train_data'],
        sampling_rate=24000,
        segment_length=config.get('segment_length', 4.0),
        augment=True,
    )
    val_dataset = OptimizedAudioDataset(
        config['val_data'],
        sampling_rate=24000,
        segment_length=config.get('segment_length', 4.0),
        augment=False,
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
    logger.info("Phase 11: Two-Phase Training")
    logger.info("="*70)
    logger.info(f"\nPhase 1 - Warmup (epochs 1-{warmup_epochs}):")
    logger.info(f"  Train: New 2x upsampler + final conv only")
    logger.info(f"  Frozen: Pretrained decoder (all 4 blocks)")
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
                freeze_new_model(model_48k)

                # Optimizer for warmup
                optimizer = AdamW(
                    filter(lambda p: p.requires_grad, model_48k.parameters()),
                    lr=warmup_lr,
                    betas=(0.9, 0.999),
                    weight_decay=config['weight_decay']
                )

            phase = "Warmup"
        elif epoch == warmup_epochs:
            logger.info("\n" + "="*70)
            logger.info("PHASE 2: MAIN - Unfreezing entire decoder")
            logger.info("="*70 + "\n")
            unfreeze_decoder(model_48k)

            # Optimizer for main training
            optimizer = AdamW(
                model_48k.parameters(),
                lr=main_lr,
                betas=(0.9, 0.999),
                weight_decay=config['weight_decay']
            )

            phase = "Main"

        # Scheduler
        if epoch < warmup_epochs:
            scheduler = CosineAnnealingLR(optimizer, T_max=warmup_epochs, eta_min=warmup_lr * 0.1)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=main_lr * 0.1)

        logger.info(f"\nEpoch {epoch + 1}/{total_epochs} [{phase}]")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        train_metrics = train_epoch(model_48k, train_loader, sidon, optimizer, scheduler, device, config, epoch)
        val_metrics = validate(model_48k, val_loader, sidon, device, config)

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

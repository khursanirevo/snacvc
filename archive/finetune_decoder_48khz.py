"""
Phase 11: SNAC Decoder Fine-tuning for 48kHz Output

Train SNAC decoder to output 48kHz audio while keeping encoder and VQ frozen.

Architecture:
  Input (24kHz) → Encoder (frozen) → VQ (frozen) → Decoder (trainable) → Output (48kHz)

Training target: SIDON upsampler's 48kHz output
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from snac import SNAC
from snac.layers import Decoder
from snac.dataset import OptimizedAudioDataset

# SIDON upsampler (will be imported conditionally)
try:
    import torchaudio
    import transformers
    from huggingface_hub import hf_hub_download
    SIDON_AVAILABLE = True
except ImportError:
    SIDON_AVAILABLE = False
    print("Warning: SIDON dependencies not available. Install with: pip install torchaudio transformers huggingface_hub")


def setup_logging(config):
    """Setup logging to file based on experiment name."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    experiment_name = config.get('experiment_name', config.get('output_dir', 'training').split('/')[-1])
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


class SIDONUpsampler:
    """
    SIDON speech upsampler from sarulab-speech/sidon-v0.1

    Takes audio at any sample rate, outputs 48kHz enhanced audio.
    Uses CUDA versions for better performance.
    """
    def __init__(self, device='cuda'):
        if not SIDON_AVAILABLE:
            raise RuntimeError("SIDON dependencies not available")

        self.device = device

        # Clear CUDA cache to avoid device conflicts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load SIDON CUDA models directly (like sidon_recon.py)
        fe_path = hf_hub_download("sarulab-speech/sidon-v0.1", filename="feature_extractor_cuda.pt")
        decoder_path = hf_hub_download("sarulab-speech/sidon-v0.1", filename="decoder_cuda.pt")

        self.fe = torch.jit.load(fe_path, map_location='cuda').to('cuda')
        self.fe.eval()

        self.decoder = torch.jit.load(decoder_path, map_location='cuda').to('cuda')
        self.decoder.eval()

        self.preprocessor = transformers.SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )

    @torch.inference_mode()
    def __call__(self, audio: torch.Tensor, sample_rate: int) -> Tuple[int, torch.Tensor]:
        """
        Upsample audio to 48kHz.

        Args:
            audio: (B, 1, T) or (1, T) or (T,) tensor at any sample rate
            sample_rate: Original sample rate

        Returns:
            (48000, audio_48k) tuple
        """
        # Normalize
        audio = 0.9 * (audio / audio.abs().max())

        # Ensure tensor format
        if not isinstance(audio, torch.Tensor):
            audio = torch.tensor(audio, dtype=torch.float32, device=self.device)

        # Convert to mono if stereo
        if audio.ndim > 1 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0)

        # Add batch dimension if needed
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.ndim == 2:
            audio = audio.unsqueeze(0)

        # Highpass filter
        audio = torchaudio.functional.highpass_biquad(
            audio.squeeze(0), sample_rate=sample_rate, cutoff_freq=50
        ).unsqueeze(0)

        # Resample to 16kHz for SIDON
        audio_16k = torchaudio.functional.resample(
            audio.squeeze(0), orig_freq=sample_rate, new_freq=16_000
        )

        # Pad for SIDON processing
        audio_16k = F.pad(audio_16k, (0, 24000))

        # Process chunks
        restoreds = []
        feature_cache = None

        for chunk in audio_16k.view(-1).split(16000 * 60):
            # Preprocessor needs CPU input (uses numpy), then move to CUDA
            chunk_cpu = chunk.cpu() if chunk.is_cuda else chunk
            inputs = self.preprocessor(
                F.pad(chunk_cpu, (40, 40)), sampling_rate=16_000, return_tensors="pt"
            ).to(self.device)

            feature = self.fe(inputs["input_features"])["last_hidden_state"]

            if feature_cache is not None:
                feature = torch.cat([feature_cache, feature], dim=1)
                restored = self.decoder(feature.transpose(1, 2))
                restored = restored[:, :, 4800:]
            else:
                restored = self.decoder(feature.transpose(1, 2))
                restored = restored[:, :, 50 * 3:]

            feature_cache = feature[:, -5:, :]
            restoreds.append(restored.cpu())

        restored = torch.cat(restoreds, dim=-1)

        # Trim to target length (2x upsampling from original sample rate)
        target_samples = int(48_000 / sample_rate * audio.shape[-1])
        restored = restored[..., :target_samples]

        return 48_000, restored.squeeze(0).unsqueeze(0)  # (B, 1, T)


def create_decoder_48khz(latent_dim: int, decoder_dim: int) -> Decoder:
    """
    Create a decoder that outputs 48kHz from 24kHz input.

    Original decoder: [8, 8, 4, 2] = 512x upsampling (24kHz input → 24kHz output)
    New decoder:      [8, 8, 4, 2, 2] = 1024x upsampling (24kHz input → 48kHz output)

    Args:
        latent_dim: Input latent dimension
        decoder_dim: Base decoder dimension

    Returns:
        Decoder configured for 48kHz output
    """
    return Decoder(
        input_channel=latent_dim,
        channels=decoder_dim,
        rates=[8, 8, 4, 2, 2],  # Added extra 2x upsampling for 48kHz
        noise=True,
        depthwise=True,
        attn_window_size=32,
    )


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, config: dict) -> Tuple:
    """
    Compute reconstruction loss between predicted and target audio.

    Args:
        pred: (B, 1, T) predicted audio at 48kHz
        target: (B, 1, T) target audio at 48kHz
        config: Training configuration

    Returns:
        (loss, l1_loss, stft_loss) tuple
    """
    # L1 loss
    l1_loss = F.l1_loss(pred, target)

    # Multi-scale STFT loss
    stft_loss = 0.0
    n_ffts = config.get('n_ffts', [1024, 2048, 4096])

    for n_fft in n_ffts:
        # STFT
        pred_stft = torch.stft(pred.squeeze(1), n_fft=n_fft, return_complex=True)
        target_stft = torch.stft(target.squeeze(1), n_fft=n_fft, return_complex=True)

        # Magnitude loss
        stft_loss += F.l1_loss(pred_stft.abs(), target_stft.abs())

    stft_loss /= len(n_ffts)

    # Combined loss
    loss = config['l1_weight'] * l1_loss + config['stft_weight'] * stft_loss

    return loss, l1_loss, stft_loss


def train_epoch(model, sidon, train_loader, optimizer, scheduler, device, config, epoch):
    """Train for one epoch."""
    model.train()
    sidon.fe.eval()  # SIDON in inference mode
    sidon.decoder.eval()

    epoch_loss = 0.0
    epoch_l1 = 0.0
    epoch_stft = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for batch_idx, audio_batch in enumerate(pbar):
        # Audio is at 24kHz
        audio_24k = audio_batch.to(device)

        # Generate 48kHz target using SIDON
        with torch.no_grad():
            _, audio_48k_target = sidon(audio_24k, sample_rate=24000)

        # Trim to same length
        min_len = min(audio_24k.shape[-1], audio_48k_target.shape[-1])
        audio_24k = audio_24k[..., :min_len]
        audio_48k_target = audio_48k_target[..., :min_len]

        # Forward through SNAC
        z = model.encoder(audio_24k)
        z_q, codes = model.quantizer(z)
        audio_48k_pred = model.decoder(z_q)

        # Trim prediction to match target
        audio_48k_pred = audio_48k_pred[..., :audio_48k_target.shape[-1]]

        # Compute loss
        loss, l1_loss, stft_loss = reconstruction_loss(audio_48k_pred, audio_48k_target, config)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

        optimizer.step()

        # Update metrics
        epoch_loss += loss.item()
        epoch_l1 += l1_loss.item()
        epoch_stft += stft_loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'l1': f'{l1_loss.item():.4f}',
            'stft': f'{stft_loss.item():.4f}'
        })

    # Epoch averages
    avg_loss = epoch_loss / num_batches
    avg_l1 = epoch_l1 / num_batches
    avg_stft = epoch_stft / num_batches

    return {'loss': avg_loss, 'l1': avg_l1, 'stft': avg_stft}


@torch.no_grad()
def validate(model, sidon, val_loader, device, config):
    """Validate the model."""
    model.eval()
    sidon.fe.eval()
    sidon.decoder.eval()

    val_loss = 0.0
    val_l1 = 0.0
    val_stft = 0.0
    num_batches = 0

    for audio_batch in tqdm(val_loader, desc="Validation"):
        audio_24k = audio_batch.to(device)

        # Generate 48kHz target using SIDON
        _, audio_48k_target = sidon(audio_24k, sample_rate=24000)

        # Trim to same length
        min_len = min(audio_24k.shape[-1], audio_48k_target.shape[-1])
        audio_24k = audio_24k[..., :min_len]
        audio_48k_target = audio_48k_target[..., :min_len]

        # Forward through SNAC
        z = model.encoder(audio_24k)
        z_q, codes = model.quantizer(z)
        audio_48k_pred = model.decoder(z_q)

        # Trim prediction
        audio_48k_pred = audio_48k_pred[..., :audio_48k_target.shape[-1]]

        # Compute loss
        loss, l1_loss, stft_loss = reconstruction_loss(audio_48k_pred, audio_48k_target, config)

        val_loss += loss.item()
        val_l1 += l1_loss.item()
        val_stft += stft_loss.item()
        num_batches += 1

    avg_loss = val_loss / num_batches
    avg_l1 = val_l1 / num_batches
    avg_stft = val_stft / num_batches

    return {'val_loss': avg_loss, 'val_l1': avg_l1, 'val_stft': avg_stft}


def main():
    parser = argparse.ArgumentParser(description="Train SNAC decoder for 48kHz output")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Config: {args.config}")
    logger.info(f"Experiment: {config.get('experiment_name', 'unknown')}")

    # Set random seed
    random_seed = config.get('random_seed', 42)
    import random
    import numpy as np
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    logger.info(f"Random seed: {random_seed}")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Create output dir
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize SIDON upsampler
    logger.info("\nLoading SIDON upsampler...")
    sidon = SIDONUpsampler(device)
    logger.info("✓ SIDON loaded")

    # Load pretrained SNAC 24kHz model
    logger.info("\nLoading pretrained SNAC 24kHz model...")
    model = SNAC.from_pretrained(config['pretrained_model']).to(device)

    # Freeze encoder and VQ
    logger.info("Freezing encoder and VQ...")
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.quantizer.parameters():
        param.requires_grad = False

    # Replace decoder with 48kHz decoder
    logger.info("Creating 48kHz decoder...")
    old_decoder = model.decoder
    model.decoder = create_decoder_48khz(model.latent_dim, model.decoder_dim).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"\nModel parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable (48kHz decoder): {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    logger.info(f"  Frozen (encoder + VQ): {total_params - trainable_params:,}")

    # Optimizer (only decoder parameters)
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

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get('eval_batch_size', config['batch_size']),
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    # Training loop
    logger.info("\n" + "="*70)
    logger.info("Phase 11: SNAC Decoder Fine-tuning for 48kHz Output")
    logger.info("  Input: 24kHz")
    logger.info("  Output: 48kHz")
    logger.info("  Target: SIDON upsampler")
    logger.info("="*70 + "\n")

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Resumed from epoch {checkpoint['epoch']}")

    for epoch in range(start_epoch, config['num_epochs']):
        logger.info(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Train
        train_metrics = train_epoch(model, sidon, train_loader, optimizer, scheduler, device, config, epoch)

        # Validate
        val_metrics = validate(model, sidon, val_loader, device, config)

        # Update scheduler
        scheduler.step()

        # Log metrics
        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Train loss: {train_metrics['loss']:.4f} (L1: {train_metrics['l1']:.4f}, STFT: {train_metrics['stft']:.4f})")
        logger.info(f"  Val loss:   {val_metrics['val_loss']:.4f} (L1: {val_metrics['val_l1']:.4f}, STFT: {val_metrics['val_stft']:.4f})")

        # Save checkpoint
        if (epoch + 1) % config.get('save_every', 2) == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
            }, checkpoint_path)
            logger.info(f"  Saved: {checkpoint_path}")

        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            best_path = output_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
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

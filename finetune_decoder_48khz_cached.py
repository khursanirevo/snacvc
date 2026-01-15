"""
Phase 11: SNAC Decoder 48kHz Training with Code Caching

Strategy:
1. First epoch: Pre-compute all codes and 48kHz targets, cache to disk
2. Subsequent epochs: Load random segments from cache (much faster!)

Benefits:
- 3-5x faster after first epoch (no encoder/VQ/SIDON forward passes)
- Reproducible training with fixed cache
- Can resume from cache without recomputing

Usage:
    python finetune_decoder_48khz_cached.py \
        --config configs/phase11_decoder_48khz.json \
        --cache_dir /mnt/data/codes_phase11_full \
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
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from snac import SNAC
from snac.layers import Decoder, DecoderBlock, Snake1d
from finetune_decoder_48khz import SIDONUpsampler


def collate_fn(batch):
    """Collate function that filters out None values."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {'codes': [], 'audio': torch.empty(0)}

    # Codes is a list of 3 scales
    codes_batch = [[] for _ in range(3)]
    audio_batch = []

    max_audio_len = max(item['audio'].shape[-1] for item in batch)

    for item in batch:
        # Pad audio
        audio = item['audio']
        if audio.shape[-1] < max_audio_len:
            audio = F.pad(audio, (0, max_audio_len - audio.shape[-1]), mode='constant')
        audio_batch.append(audio)

        # Collect codes
        for scale in range(3):
            codes_batch[scale].append(item['codes'][scale])

    # Stack
    audio_batch = torch.stack(audio_batch)  # (B, 1, T)
    codes_batch = [torch.stack(codes_batch[scale]) for scale in range(3)]

    return {'codes': codes_batch, 'audio': audio_batch}


class Decoder48kHz(nn.Module):
    """Wrapper that adds a 2x upsampler to the pretrained 24kHz decoder."""
    def __init__(self, pretrained_snac, device='cuda'):
        super().__init__()
        self.device = device
        self.encoder = pretrained_snac.encoder
        self.quantizer = pretrained_snac.quantizer
        self.pretrained_decoder = pretrained_snac.decoder

        # New 2x upsampler: 64 → 64 (upsample by 2x)
        self.upsampler2x = DecoderBlock(
            input_dim=64,
            output_dim=64,
            stride=2,
            noise=False,
            groups=1
        )

        self.final_conv = nn.Conv1d(64, 1, kernel_size=7, padding=3)
        self.snake1d = Snake1d(64)

        # Initialize
        nn.init.xavier_uniform_(self.final_conv.weight)
        if self.final_conv.bias is not None:
            nn.init.zeros_(self.final_conv.bias)

        print("✅ 48kHz upsampler initialized")

    def forward(self, z_q):
        # Run decoder layers 0-5
        x = z_q
        for layer in self.pretrained_decoder.model[:6]:
            x = layer(x)

        # 2x upsampler
        x = self.upsampler2x(x)
        x = self.snake1d(x)
        audio = self.final_conv(x)
        audio = torch.tanh(audio)
        return audio

    def get_new_layers(self):
        return [self.upsampler2x, self.final_conv]


class CodeCacheDataset(Dataset):
    """
    Dataset that loads pre-computed codes with random segment extraction.

    Token-to-time mapping (SNAC 24kHz):
    - Hop length: 512 samples
    - VQ strides: [4, 2, 1] for the 3 codebooks
    - Encoder stride: 3*3*7*7 = 441

    For 4 seconds at 24kHz:
    - Input: 96000 samples
    - After encoder: 96000 / 441 ≈ 218 timesteps
    - Scale 0 codes (stride 4): ~54 codes
    - Scale 1 codes (stride 2): ~109 codes
    - Scale 2 codes (stride 1): ~218 codes
    """
    def __init__(
        self,
        cache_dir: str,
        audio_48khz_dir: str,
        segment_length_sec: float = 4.0,
        sampling_rate: int = 24000,
        hop_length: int = 512,
    ):
        self.cache_dir = Path(cache_dir)
        self.audio_48khz_dir = Path(audio_48khz_dir)
        self.segment_length_sec = segment_length_sec
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length

        # Load metadata
        metadata_file = self.cache_dir / "metadata.json"
        if not metadata_file.exists():
            raise ValueError(f"Metadata not found: {metadata_file}")

        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        self.vq_strides = self.metadata['vq_strides']  # [4, 2, 1]
        self.num_scales = len(self.vq_strides)

        # Calculate segment length in codes
        segment_samples = int(segment_length_sec * sampling_rate)
        # Encoder total stride = 441 (3*3*7*7)
        encoder_stride = 441
        base_length = segment_samples // encoder_stride

        self.segment_lengths = [
            base_length * (self.vq_strides[0] // self.vq_strides[i])
            for i in range(len(self.vq_strides))
        ]

        print(f"Segment length: {segment_length_sec}s = {segment_samples} samples")
        print(f"  Scale 0: {self.segment_lengths[0]} codes")
        print(f"  Scale 1: {self.segment_lengths[1]} codes")
        print(f"  Scale 2: {self.segment_lengths[2]} codes")

        # Get all parquet files
        self.parquet_files = sorted(self.cache_dir.glob("codes_batch_*.parquet"))
        print(f"Found {len(self.parquet_files)} code files")

        self._build_index()

    def _build_index(self):
        """Build index of all files with their code lengths."""
        self.file_index = []
        self.cumulative_segments = [0]

        for file_idx, parquet_file in enumerate(self.parquet_files):
            try:
                df = pd.read_parquet(parquet_file, columns=['shape_scale_0'])
                num_codes = df['shape_scale_0'].iloc[0][0]
                num_valid_segments = max(0, num_codes - self.segment_lengths[0] + 1)

                if num_valid_segments > 0:
                    self.file_index.append({
                        'file_idx': file_idx,
                        'parquet_file': parquet_file,
                        'num_codes': num_codes,
                        'num_segments': num_valid_segments,
                    })
            except Exception as e:
                print(f"Warning: Error reading {parquet_file}: {e}")
                continue

        for info in self.file_index:
            self.cumulative_segments.append(self.cumulative_segments[-1] + info['num_segments'])

        self.total_segments = self.cumulative_segments[-1]
        print(f"Total segments: {self.total_segments:,}")

    def __len__(self):
        return self.total_segments

    def __getitem__(self, idx):
        """Get a random segment from codes."""
        import bisect
        file_idx = bisect.bisect_right(self.cumulative_segments, idx) - 1
        segment_idx_in_file = idx - self.cumulative_segments[file_idx]

        file_info = self.file_index[file_idx]
        df = pd.read_parquet(file_info['parquet_file'])

        # Select a random row
        row_idx = torch.randint(0, len(df), (1,)).item()
        row = df.iloc[row_idx]

        # Get max start position
        max_starts = [
            row[f'shape_scale_{i}'][0] - self.segment_lengths[i]
            for i in range(self.num_scales)
        ]
        max_start_scale_0 = max_starts[0]

        if max_start_scale_0 <= 0:
            start_pos = 0
        else:
            start_pos = torch.randint(0, max_start_scale_0 + 1, (1,)).item()

        # Extract codes for each scale
        codes = []
        for scale in range(self.num_scales):
            scale_start = start_pos * self.vq_strides[scale]
            segment_codes = row[f'codes_scale_{scale}'][scale_start:scale_start + self.segment_lengths[scale]]

            if len(segment_codes) < self.segment_lengths[scale]:
                pad_length = self.segment_lengths[scale] - len(segment_codes)
                segment_codes = np.pad(segment_codes, (0, pad_length), mode='constant')

            codes.append(torch.tensor(segment_codes, dtype=torch.long))

        # Get corresponding 48kHz audio
        audio_file_name = Path(row['file_path']).name
        audio_path = self.audio_48khz_dir / audio_file_name

        # Convert start position to 48kHz audio samples
        start_sample_24k = start_pos * self.hop_length * self.vq_strides[0]
        start_sample_48k = start_sample_24k * 2
        segment_samples = int(self.segment_length_sec * 48000)

        try:
            audio_48k, sr = torchaudio.load(str(audio_path))
            if audio_48k.shape[0] > 1:
                audio_48k = audio_48k.mean(dim=0, keepdim=True)
            if sr != 48000:
                resampler = torchaudio.transforms.Resample(sr, 48000)
                audio_48k = resampler(audio_48k)

            if audio_48k.shape[-1] >= start_sample_48k + segment_samples:
                audio_segment = audio_48k[..., start_sample_48k:start_sample_48k + segment_samples]
            else:
                audio_segment = audio_48k[..., start_sample_48k:]
                if audio_segment.shape[-1] < segment_samples:
                    pad_length = segment_samples - audio_segment.shape[-1]
                    audio_segment = F.pad(audio_segment, (0, pad_length), mode='constant')
        except Exception as e:
            audio_segment = torch.zeros(1, segment_samples)

        return {'codes': codes, 'audio': audio_segment.squeeze(0)}


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

    for batch in pbar:
        if batch['audio'].numel() == 0:
            continue

        codes_batch = [c.to(device) for c in batch['codes']]
        audio_48k_target = batch['audio'].to(device)

        # Decode from codes
        z_q = model.quantizer.from_codes(codes_batch)
        audio_48k_pred = model(z_q)

        min_len = min(audio_48k_pred.shape[-1], audio_48k_target.shape[-1])
        audio_48k_pred = audio_48k_pred[..., :min_len]
        audio_48k_target = audio_48k_target[..., :min_len]

        loss, l1_loss, stft_loss = reconstruction_loss(audio_48k_pred, audio_48k_target, config)

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

    for batch in tqdm(val_loader, desc="Validation"):
        if batch['audio'].numel() == 0:
            continue

        codes_batch = [c.to(device) for c in batch['codes']]
        audio_48k_target = batch['audio'].to(device)

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


def precompute_cache(train_dataset, val_dataset, sidon, snac_model, cache_dir, device, config):
    """Pre-compute codes and 48kHz targets for all audio files."""
    print("\n" + "="*70)
    print("PRE-COMPUTING CODES AND 48kHz TARGETS")
    print("="*70)

    cache_dir = Path(cache_dir)
    train_cache_dir = cache_dir / "train"
    val_cache_dir = cache_dir / "val"
    train_cache_dir.mkdir(parents=True, exist_ok=True)
    val_cache_dir.mkdir(parents=True, exist_ok=True)

    for split_name, dataset, output_dir in [("train", train_dataset, train_cache_dir),
                                              ("val", val_dataset, val_cache_dir)]:
        print(f"\nProcessing {split_name} set ({len(dataset)} files)...")

        batch_size = 1000
        all_data = []
        success_count = 0
        error_count = 0

        for idx in tqdm(range(len(dataset)), desc=f"Caching {split_name}"):
            try:
                audio = dataset[idx]['audio']  # (T,)
                # Reshape to (1, 1, T) for encoder
                audio_tensor = audio.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, T)

                # Skip if too short
                if audio_tensor.shape[-1] < 24000:  # At least 1 second
                    continue

                # Generate 48kHz target
                sample_rate, audio_48k = sidon(audio_tensor, sample_rate=24000)

                # Check if SIDON returned valid audio
                if audio_48k is None or audio_48k.numel() == 0:
                    print(f"Error processing {dataset.samples[idx].name}: SIDON returned None")
                    continue

                audio_48k = audio_48k.to(device)

                # Encode with SNAC
                with torch.no_grad():
                    z = snac_model.encoder(audio_tensor)
                    _, codes = snac_model.quantizer(z)

                # Store data - convert codes to list for parquet compatibility
                all_data.append({
                    'file_path': dataset.samples[idx].name,
                    'codes_scale_0': codes[0].cpu().numpy().tolist(),
                    'codes_scale_1': codes[1].cpu().numpy().tolist(),
                    'codes_scale_2': codes[2].cpu().numpy().tolist(),
                    'shape_scale_0': (codes[0].shape[0],),
                    'shape_scale_1': (codes[1].shape[0],),
                    'shape_scale_2': (codes[2].shape[0],),
                })
                success_count += 1

                # Print status every 1000 files
                if (idx + 1) % 1000 == 0:
                    print(f"Progress: {idx + 1}/{len(dataset)} | Success: {success_count} | Errors: {error_count} | In batch: {len(all_data)}")

                # Save batch
                if len(all_data) >= batch_size:
                    print(f"\n✓ Saving batch with {len(all_data)} files...")
                    df = pd.DataFrame(all_data)
                    batch_num = len(list(output_dir.glob("*.parquet")))
                    output_path = output_dir / f"codes_batch_{batch_num}.parquet"
                    df.to_parquet(output_path, index=False)
                    print(f"✓ Saved {output_path}")
                    all_data = []

            except Exception as e:
                error_count += 1
                if error_count <= 10:  # Only print first 10 errors to avoid spam
                    print(f"Error processing {dataset.samples[idx].name}: {e}")
                continue

        # Save remaining
        if all_data:
            print(f"\nSaving final batch with {len(all_data)} files...")
            df = pd.DataFrame(all_data)
            batch_num = len(list(output_dir.glob("*.parquet")))
            output_path = output_dir / f"codes_batch_{batch_num}.parquet"
            df.to_parquet(output_path, index=False)
            print(f"✓ Saved {output_path}")

        # Save metadata
        metadata = {
            'total_samples': len(dataset),
            'num_codebooks': 3,
            'codebook_size': 4096,
            'vq_strides': [4, 2, 1],
            'latent_dim': 768,
            'hop_length': 512,
            'sampling_rate': 24000,
        }
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ {split_name} cache saved to {output_dir}")

    print("\n" + "="*70)
    print("PRE-COMPUTATION COMPLETE")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Train 48kHz decoder with cached codes")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="/mnt/data/codes_phase11_full")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--force_recache", action="store_true")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
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

    # Load pretrained SNAC
    logger.info("\nLoading pretrained SNAC 24kHz model...")
    pretrained_model = SNAC.from_pretrained(config['pretrained_model']).to(device)

    for param in pretrained_model.encoder.parameters():
        param.requires_grad = False
    for param in pretrained_model.quantizer.parameters():
        param.requires_grad = False
    logger.info("✓ Encoder and VQ frozen")

    # Create 48kHz decoder
    logger.info("\nCreating 48kHz decoder...")
    model_48k = Decoder48kHz(pretrained_model, device).to(device)

    total_params = sum(p.numel() for p in model_48k.parameters())
    trainable_params = sum(p.numel() for p in model_48k.parameters() if p.requires_grad)
    logger.info(f"Model parameters: Total={total_params:,}, Trainable={trainable_params:,}")

    # Check if cache exists
    train_cache_dir = Path(args.cache_dir) / "train"
    val_cache_dir = Path(args.cache_dir) / "val"
    cache_exists = (train_cache_dir / "metadata.json").exists() and (val_cache_dir / "metadata.json").exists()

    if args.force_recache or not cache_exists:
        # Need to precompute cache
        logger.info("\nLoading audio datasets for caching...")
        from snac.dataset import OptimizedAudioDataset

        train_dataset = OptimizedAudioDataset(
            config['train_data'],
            sampling_rate=24000,
            segment_length=config.get('segment_length', 4.0),
            augment=False,  # No augmentation for caching
        )
        val_dataset = OptimizedAudioDataset(
            config['val_data'],
            sampling_rate=24000,
            segment_length=config.get('segment_length', 4.0),
            augment=False,
        )

        # Pre-compute cache
        precompute_cache(train_dataset, val_dataset, sidon, pretrained_model, args.cache_dir, device, config)

    # Load cached datasets
    logger.info("\nLoading cached datasets...")
    train_dataset = CodeCacheDataset(
        f"{args.cache_dir}/train",
        f"{config['audio_48khz_train_dir']}",
        segment_length_sec=config.get('segment_length', 4.0),
    )
    val_dataset = CodeCacheDataset(
        f"{args.cache_dir}/val",
        f"{config['audio_48khz_val_dir']}",
        segment_length_sec=config.get('segment_length', 4.0),
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
    logger.info("Phase 11: Two-Phase Training with Cached Codes")
    logger.info("="*70)
    logger.info(f"\nPhase 1 - Warmup (epochs 1-{warmup_epochs}):")
    logger.info(f"  Learning rate: {warmup_lr:.2e}")
    logger.info(f"\nPhase 2 - Main (epochs {warmup_epochs+1}-{total_epochs}):")
    logger.info(f"  Learning rate: {main_lr:.2e}")
    logger.info("="*70 + "\n")

    start_epoch = 0
    best_val_loss = float('inf')

    for epoch in range(start_epoch, total_epochs):
        if epoch < warmup_epochs:
            if epoch == 0:
                logger.info("\n" + "="*70)
                logger.info("PHASE 1: WARMUP")
                logger.info("="*70 + "\n")
                for param in model_48k.pretrained_decoder.parameters():
                    param.requires_grad = False
                optimizer = AdamW(
                    filter(lambda p: p.requires_grad, model_48k.parameters()),
                    lr=warmup_lr,
                    betas=(0.9, 0.999),
                    weight_decay=config['weight_decay']
                )
                scheduler = CosineAnnealingLR(optimizer, T_max=warmup_epochs, eta_min=warmup_lr * 0.1)
            phase = "Warmup"
        elif epoch == warmup_epochs:
            logger.info("\n" + "="*70)
            logger.info("PHASE 2: MAIN")
            logger.info("="*70 + "\n")
            for param in model_48k.parameters():
                param.requires_grad = True
            optimizer = AdamW(
                model_48k.parameters(),
                lr=main_lr,
                betas=(0.9, 0.999),
                weight_decay=config['weight_decay']
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=main_lr * 0.1)
            phase = "Main"

        logger.info(f"\nEpoch {epoch + 1}/{total_epochs} [{phase}]")
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        train_metrics = train_epoch(model_48k, train_loader, optimizer, scheduler, device, config, epoch)
        val_metrics = validate(model_48k, val_loader, device, config)

        scheduler.step()

        logger.info(f"\nEpoch {epoch + 1} Summary:")
        logger.info(f"  Train loss: {train_metrics['loss']:.4f}")
        logger.info(f"  Val loss:   {val_metrics['val_loss']:.4f}")

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

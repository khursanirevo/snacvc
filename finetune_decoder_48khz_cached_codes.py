"""
Phase 11: SNAC Decoder 48kHz with Cached Codes

Uses pre-computed encoder+VQ codes to skip expensive forward pass.
Still runs SIDON on-the-fly for 48kHz target generation.
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
from torch.utils.data import DataLoader, Dataset
import torchaudio
import numpy as np
from tqdm import tqdm
import pandas as pd

from snac import SNAC
from snac.layers import Decoder, DecoderBlock, Snake1d
from snac.dataset import OptimizedAudioDataset
from finetune_decoder_48khz import SIDONUpsampler


def batch_sidon(sidon, audio_batch: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Process a batch of audio through SIDON (which doesn't support batching)."""
    batch_size = audio_batch.shape[0]
    outputs = []

    for i in range(batch_size):
        audio_single = audio_batch[i].unsqueeze(0)
        _, audio_48k = sidon(audio_single, sample_rate=sample_rate)
        outputs.append(audio_48k.squeeze(0))

    return torch.stack(outputs, dim=0)


class CachedCodesDataset(Dataset):
    """
    Dataset that loads pre-computed VQ codes and original audio paths.
    Skips encoder+VQ forward pass for speed.
    """
    def __init__(self, cache_dir: str, audio_dir: str, sampling_rate: int = 24000, segment_length: float = 4.0):
        self.cache_dir = Path(cache_dir)
        self.audio_dir = Path(audio_dir)
        self.sampling_rate = sampling_rate
        self.segment_samples = int(sampling_rate * segment_length)

        # Load metadata
        metadata_file = self.cache_dir / "metadata.json"
        if not metadata_file.exists():
            raise ValueError(f"No metadata found at {metadata_file}")

        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        # Get all parquet files
        self.parquet_files = sorted(self.cache_dir.glob("codes_batch_*.parquet"))
        print(f"Found {len(self.parquet_files)} code files")

        # Build index: (parquet_file, row_idx) for each segment
        self.index = []
        for file_idx, parquet_file in enumerate(self.parquet_files):
            try:
                df = pd.read_parquet(parquet_file, columns=['file_path'])
                for row_idx in range(len(df)):
                    self.index.append({
                        'parquet_file': parquet_file,
                        'row_idx': row_idx
                    })
            except Exception as e:
                print(f"Warning: Error reading {parquet_file}: {e}")
                continue

        print(f"Total segments: {len(self.index)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_info = self.index[idx]

        # Load codes from parquet
        df = pd.read_parquet(
            file_info['parquet_file'],
            columns=['file_path', 'codes_scale_0', 'codes_scale_1', 'codes_scale_2',
                    'shape_scale_0', 'shape_scale_1', 'shape_scale_2']
        )
        row = df.iloc[file_info['row_idx']]

        # Get file path
        file_path = row['file_path']

        # Load codes as tensors (convert from object arrays)
        def to_tensor(obj):
            if isinstance(obj, np.ndarray):
                if obj.dtype == object:
                    # Nested array, convert to list then tensor
                    return torch.tensor(obj.tolist(), dtype=torch.long)
                else:
                    return torch.from_numpy(obj).long()
            elif isinstance(obj, list):
                return torch.tensor(obj, dtype=torch.long)
            else:
                return torch.tensor([obj], dtype=torch.long)

        codes_scale_0 = to_tensor(row['codes_scale_0'])
        codes_scale_1 = to_tensor(row['codes_scale_1'])
        codes_scale_2 = to_tensor(row['codes_scale_2'])

        codes = [codes_scale_0, codes_scale_1, codes_scale_2]

        # Load original audio for SIDON
        audio_path = self.audio_dir / file_path
        try:
            audio, sr = torchaudio.load(audio_path)
            if sr != self.sampling_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sampling_rate)

            # Convert to mono if needed
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # Trim or pad to segment length
            if audio.shape[-1] < self.segment_samples:
                return None  # Skip short files

            audio = audio[:, :self.segment_samples].squeeze(0)

            return {
                'codes': codes,
                'audio': audio,
                'file_path': file_path
            }

        except Exception as e:
            return None


def collate_fn(batch):
    """Collate function that handles codes and audio separately."""
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return {'codes': None, 'audio': torch.empty(0)}

    # Stack audio tensors
    audio_batch = torch.stack([item['audio'] for item in batch])
    audio_batch = audio_batch.unsqueeze(1)  # (B, 1, T)

    # Codes stay as list of tensors (each batch element has different code lengths)
    codes_batch = [item['codes'] for item in batch]

    return {'codes': codes_batch, 'audio': audio_batch}


class Decoder48kHz(nn.Module):
    """Wrapper that adds 2x upsampler to pretrained 24kHz decoder."""
    def __init__(self, pretrained_snac, device='cuda'):
        super().__init__()
        self.device = device

        self.encoder = pretrained_snac.encoder
        self.quantizer = pretrained_snac.quantizer
        self.pretrained_decoder = pretrained_snac.decoder

        # New 2x upsampler
        self.upsampler2x = DecoderBlock(
            input_dim=64,
            output_dim=64,
            stride=2,
            noise=False,
            groups=1
        )

        self.final_conv = nn.Conv1d(64, 1, kernel_size=7, padding=3)
        self.snake1d = Snake1d(64)

        # Initialize randomly (will be trained from scratch in warmup phase)
        print("\nInitializing 2x upsampler randomly (will train from scratch)")
        nn.init.xavier_uniform_(self.upsampler2x.block[1].weight)
        nn.init.zeros_(self.upsampler2x.block[1].bias)
        nn.init.xavier_uniform_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)
        print("✅ 2x upsampler initialized (random init, will be trained from scratch)")

    def forward(self, z_q):
        """
        Forward pass through decoder.

        Args:
            z_q: Quantized latent (B, 512, T')

        Returns:
            Audio at 48kHz: (B, 1, T*2)
        """
        # Decode through pretrained decoder (frozen or unfrozen)
        h = self.pretrained_decoder(z_q)

        # h is now (B, 64, T_24k) at 24kHz rate
        # Apply Snake1d
        h = self.snake1d(h)

        # Upsample 2x to 48kHz
        h = self.upsampler2x(h)

        # Final conv to mono
        audio = self.final_conv(h)

        # Tanh for [-1, 1] range
        audio = torch.tanh(audio)

        return audio


def reconstruction_loss(pred, target, config):
    """Compute reconstruction loss."""
    l1_weight = config.get('l1_weight', 1.0)
    stft_weight = config.get('stft_weight', 1.0)
    n_ffts = config.get('n_ffts', [1024, 2048, 4096, 8192])

    # L1 loss
    l1_loss = F.l1_loss(pred, target)

    # Multi-scale STFT loss
    stft_loss = 0.0
    for n_fft in n_ffts:
        pred_stft = torch.stft(pred.squeeze(1), n_fft=n_fft, return_complex=True)
        target_stft = torch.stft(target.squeeze(1), n_fft=n_fft, return_complex=True)
        stft_loss += F.l1_loss(pred_stft.abs(), target_stft.abs())

    stft_loss = stft_loss / len(n_ffts)

    loss = l1_weight * l1_loss + stft_weight * stft_loss
    return loss, l1_loss, stft_loss


def codes_to_latent(codes, vq_strides, device):
    """
    Convert hierarchical codes to latent tensor.

    Args:
        codes: list of 3 tensors [codes_s0, codes_s1, codes_s2]
        vq_strides: [4, 2, 1]
        device: cuda

    Returns:
        z_q: (B, 512, T) quantized latent
    """
    codes_s0, codes_s1, codes_s2 = codes
    stride_0, stride_1, stride_2 = vq_strides

    # Start with coarsest level (scale 0)
    z = F.one_hot(codes_s0, 4096).float() * (4096**0.5)
    z = z.repeat_interleave(stride_0, dim=-1)

    # Add scale 1 residual
    z_1 = F.one_hot(codes_s1, 4096).float() * (4096**0.5)
    z_1 = z_1.repeat_interleave(stride_1, dim=-1)
    z = z + z_1

    # Add scale 2 residual
    z_2 = F.one_hot(codes_s2, 4096).float() * (4096**0.5)
    z_2 = z_2.repeat_interleave(stride_2, dim=-1)
    z = z + z_2

    # Reshape to (512, T)
    T = z.shape[-1] // 512
    z = z.reshape(512, T).unsqueeze(0)

    return z.to(device)


@torch.no_grad()
def validate(model, val_loader, sidon, device, config, vq_strides):
    """Validate the model."""
    model.eval()

    val_loss = 0.0
    val_l1 = 0.0
    val_stft = 0.0
    num_batches = 0

    for batch in tqdm(val_loader, desc="Validation"):
        if batch['codes'] is None or batch['audio'].numel() == 0:
            continue

        codes_batch = batch['codes']
        audio_24k = batch['audio'].to(device)

        # Generate 48kHz target using SIDON
        audio_48k_target = batch_sidon(sidon, audio_24k, sample_rate=24000)

        if audio_48k_target.device != device:
            audio_48k_target = audio_48k_target.to(device)

        # Convert codes to latent for each sample in batch
        audio_48k_preds = []
        for codes in codes_batch:
            z_q = codes_to_latent(codes, vq_strides, device)
            audio_48k_pred = model(z_q)
            audio_48k_preds.append(audio_48k_pred)

        # Stack predictions
        audio_48k_pred = torch.cat(audio_48k_preds, dim=0)

        # Trim to match lengths
        min_len = min(audio_48k_pred.shape[-1], audio_48k_target.shape[-1])
        audio_48k_pred = audio_48k_pred[..., :min_len]
        audio_48k_target = audio_48k_target[..., :min_len]

        # Compute loss
        loss, l1_loss, stft_loss = reconstruction_loss(audio_48k_pred, audio_48k_target, config)

        val_loss += loss.item()
        val_l1 += l1_loss.item()
        val_stft += stft_loss.item()
        num_batches += 1

    return {'loss': val_loss / num_batches, 'l1': val_l1 / num_batches, 'stft': val_stft / num_batches}


def train_epoch(model, train_loader, sidon, optimizer, device, config, vq_strides, epoch):
    """Train for one epoch."""
    model.train()

    epoch_loss = 0.0
    epoch_l1 = 0.0
    epoch_stft = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for batch in pbar:
        if batch['codes'] is None or batch['audio'].numel() == 0:
            continue

        codes_batch = batch['codes']
        audio_24k = batch['audio'].to(device)

        # Generate 48kHz target using SIDON
        audio_48k_target = batch_sidon(sidon, audio_24k, sample_rate=24000)

        if audio_48k_target.device != device:
            audio_48k_target = audio_48k_target.to(device)

        # Convert codes to latent and decode for each sample
        optimizer.zero_grad()
        total_loss = 0.0
        total_l1 = 0.0
        total_stft = 0.0

        for codes in codes_batch:
            z_q = codes_to_latent(codes, vq_strides, device)
            audio_48k_pred = model(z_q)

            # Trim to match target
            min_len = min(audio_48k_pred.shape[-1], audio_48k_target.shape[-1])
            audio_48k_pred = audio_48k_pred[..., :min_len]
            target_trimmed = audio_48k_target[..., :min_len]

            loss, l1_loss, stft_loss = reconstruction_loss(audio_48k_pred, target_trimmed, config)

            total_loss += loss
            total_l1 += l1_loss
            total_stft += stft_loss

        # Average over batch and backward
        avg_loss = total_loss / len(codes_batch)
        avg_loss.backward()

        if config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

        optimizer.step()

        epoch_loss += avg_loss.item()
        epoch_l1 += (total_l1 / len(codes_batch)).item()
        epoch_stft += (total_stft / len(codes_batch)).item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{avg_loss.item():.4f}'})

    return {'loss': epoch_loss / num_batches, 'l1': epoch_l1 / num_batches, 'stft': epoch_stft / num_batches}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, required=True)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    device = torch.device(f'cuda:{args.device}')

    print("\n" + "="*70)
    print("PHASE 11: SNAC DECODER 48kHz WITH CACHED CODES")
    print("="*70)

    # Load SIDON
    print("\nLoading SIDON upsampler...")
    sidon = SIDONUpsampler(device)
    print("✓ SIDON loaded")

    # Load pretrained SNAC
    print("\nLoading pretrained SNAC 24kHz model...")
    pretrained_model = SNAC.from_pretrained(config.get('pretrained_model', 'hubertsiuzdak/snac_24khz')).to(device)

    # Freeze encoder and VQ
    print("Freezing encoder and VQ...")
    for param in pretrained_model.encoder.parameters():
        param.requires_grad = False
    for param in pretrained_model.quantizer.parameters():
        param.requires_grad = False
    print("✓ Encoder and VQ frozen")

    # Create 48kHz decoder
    print("\nCreating 48kHz decoder with smart initialization...")
    model = Decoder48kHz(pretrained_model, device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"  Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")

    # VQ strides
    vq_strides = config.get('vq_strides', [4, 2, 1])

    # Load cached datasets
    print("\nLoading cached datasets...")
    train_dataset = CachedCodesDataset(
        f"{args.cache_dir}/train",
        config['train_data'],
        sampling_rate=config.get('sampling_rate', 24000),
        segment_length=config.get('segment_length', 4.0)
    )

    val_dataset = CachedCodesDataset(
        f"{args.cache_dir}/val",
        config['val_data'],
        sampling_rate=config.get('sampling_rate', 24000),
        segment_length=config.get('segment_length', 4.0)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    batch_size = config.get('batch_size', 32)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.get('learning_rate', 1e-5))
    scheduler = CosineAnnealingLR(optimizer, T_max=config.get('num_epochs', 15))

    print("\n" + "="*70)
    print("TRAINING STARTED")
    print("="*70)

    best_val_loss = float('inf')

    for epoch in range(config.get('num_epochs', 15)):
        print(f"\nEpoch {epoch + 1}/{config.get('num_epochs', 15)}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Determine which params to train based on epoch
        warmup_epochs = config.get('warmup_epochs', 3)

        if epoch < warmup_epochs:
            print("\n" + "="*70)
            print(f"PHASE 1: WARMUP - Training new layers only")
            print("="*70)

            # Freeze pretrained decoder
            for param in model.pretrained_decoder.parameters():
                param.requires_grad = False

            # Unfreeze new layers
            for param in model.upsampler2x.parameters():
                param.requires_grad = True
            for param in model.final_conv.parameters():
                param.requires_grad = True

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable (new layers only): {trainable:,} params")
        else:
            print("\n" + "="*70)
            print(f"PHASE 2: MAIN - Training entire decoder")
            print("="*70)

            # Unfreeze all decoder params
            for param in model.pretrained_decoder.parameters():
                param.requires_grad = True
            for param in model.upsampler2x.parameters():
                param.requires_grad = True
            for param in model.final_conv.parameters():
                param.requires_grad = True

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable (entire decoder): {trainable:,} params")

        # Train
        train_metrics = train_epoch(model, train_loader, sidon, optimizer, device, config, vq_strides, epoch)
        print(f"Train - Loss: {train_metrics['loss']:.4f}, L1: {train_metrics['l1']:.4f}, STFT: {train_metrics['stft']:.4f}")

        # Validate
        val_metrics = validate(model, val_loader, sidon, device, config, vq_strides)
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, L1: {val_metrics['l1']:.4f}, STFT: {val_metrics['stft']:.4f}")

        # Save checkpoint
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            output_dir = Path(config.get('output_dir', 'checkpoints/phase11_decoder_48khz'))
            output_dir.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
            }

            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"✓ Saved best model (val_loss: {val_metrics['loss']:.4f})")

        scheduler.step()

    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)


if __name__ == '__main__':
    main()

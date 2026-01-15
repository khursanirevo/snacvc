"""
Phase 11: 48kHz Decoder Fine-tuning with Cached Codes (Simple Version)
Uses cached codes + SIDON on-the-fly for 48kHz targets.
Based on verified test_cached_codes_training.py
"""
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd
import torchaudio
from tqdm import tqdm
import numpy as np
import math
import random

os.environ['HF_HOME'] = '/mnt/data/work/snac/.hf_cache'
os.environ['HF_HUB_CACHE'] = '/mnt/data/work/snac/.hf_cache/hub'

from snac import SNAC
from snac.layers import Decoder, DecoderBlock, Snake1d


class CachedCodesDataset(Dataset):
    """Dataset that loads pre-computed codes from parquet files and pre-generated 48kHz audio."""
    def __init__(self, cache_dir, audio_48k_dir, segment_length=4.0, limit=None):
        self.cache_dir = Path(cache_dir)
        self.audio_48k_dir = Path(audio_48k_dir)
        self.parquet_files = sorted(self.cache_dir.glob("codes_batch_gpu*.parquet"))
        self.segment_length = segment_length  # in seconds

        # SNAC 24kHz parameters (original model before 48kHz upsampling)
        # Encoder downsampling: [3, 3, 7, 7] = 441x total
        # 24kHz / 441 = ~54.4 Hz latent frame rate
        # VQ strides: [8, 4, 2, 1] for hierarchical codes
        # So scale 0 has 8x stride, scale 1 has 4x, scale 2 has 2x, scale 3 has 1x
        self.latent_frame_rate = 24000 / 441  # ~54.4 Hz

        print(f"Found {len(self.parquet_files)} parquet files")

        # Build index: list of (parquet_file, row_idx)
        self.index = []
        for file_idx, parquet_file in enumerate(self.parquet_files):
            try:
                df = pd.read_parquet(parquet_file, columns=['file_path'])
                for row_idx in range(len(df)):
                    self.index.append({'parquet_file': parquet_file, 'row_idx': row_idx})
            except Exception as e:
                print(f"Warning: Failed to read {parquet_file}: {e}")
                continue

        # Apply limit if specified (for testing)
        if limit is not None and limit > 0:
            self.index = self.index[:limit]
            print(f"Limited dataset to {len(self.index)} samples")

        print(f"Total samples: {len(self.index)}")
        print(f"Segment length: {self.segment_length}s")
        self._update_segment_info()

    def _update_segment_info(self):
        """Calculate token counts for current segment length.

        CRITICAL: All scales must upsample to the SAME number of latent frames.
        Formula: num_tokens × vq_stride = num_latent_frames for all scales.

        Solution: Round num_latent_frames down to multiple of 8 (coarsest stride).
        """
        num_latent_frames = int(self.segment_length * self.latent_frame_rate)

        # Round DOWN to multiple of 8 (coarsest VQ stride)
        # This ensures all scales will align perfectly
        num_latent_frames = (num_latent_frames // 8) * 8

        # Now divide by strides - all will produce integer multiples
        self.num_frames_scale0 = num_latent_frames // 8
        self.num_frames_scale1 = num_latent_frames // 4
        self.num_frames_scale2 = num_latent_frames // 2

        self.num_audio_samples = int(self.segment_length * 48000)
        print(f"Segment {self.segment_length}s: {self.num_frames_scale0}/{self.num_frames_scale1}/{self.num_frames_scale2} tokens, {self.num_audio_samples} audio samples")

    def set_segment_length(self, segment_length):
        """Update segment length during training."""
        self.segment_length = segment_length
        self._update_segment_info()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_info = self.index[idx]

        try:
            df = pd.read_parquet(
                file_info['parquet_file'],
                columns=['file_path', 'codes_scale_0', 'codes_scale_1', 'codes_scale_2']
            )
            row = df.iloc[file_info['row_idx']]
        except Exception as e:
            return None

        def to_tensor(obj):
            if isinstance(obj, np.ndarray):
                if obj.dtype == object:
                    return torch.tensor(obj.tolist(), dtype=torch.long)
                else:
                    return torch.from_numpy(obj).long()
            elif isinstance(obj, list):
                return torch.tensor(obj, dtype=torch.long)
            else:
                return torch.tensor([obj], dtype=torch.long)

        codes = [
            to_tensor(row['codes_scale_0']),
            to_tensor(row['codes_scale_1']),
            to_tensor(row['codes_scale_2'])
        ]

        # Calculate random start position for slicing
        # Get full code lengths
        full_len_scale0 = codes[0].shape[0]
        full_len_scale1 = codes[1].shape[0]
        full_len_scale2 = codes[2].shape[0]

        # Calculate maximum valid start positions for each scale
        max_start_scale0 = max(0, full_len_scale0 - self.num_frames_scale0)
        max_start_scale1 = max(0, full_len_scale1 - self.num_frames_scale1)
        max_start_scale2 = max(0, full_len_scale2 - self.num_frames_scale2)

        # Use minimum to ensure all scales have enough tokens
        max_start_pos = min(max_start_scale0, max_start_scale1, max_start_scale2)

        # Generate random start position
        if max_start_pos > 0:
            start_pos = random.randint(0, max_start_pos)
        else:
            start_pos = 0  # File too short, use from start

        # CRITICAL: Multi-scale alignment for random slicing
        # vq_strides = [8, 4, 2] for SNAC 24kHz model
        # Formula: scale_start = start_pos * (vq_strides[0] // vq_strides[scale])
        # This ensures all 3 scales extract codes from the SAME temporal position
        vq_strides = [8, 4, 2]

        # Calculate start and end positions for each scale
        starts = [
            start_pos * (vq_strides[0] // vq_strides[0]),
            start_pos * (vq_strides[0] // vq_strides[1]),
            start_pos * (vq_strides[0] // vq_strides[2]),
        ]
        ends = [
            starts[0] + self.num_frames_scale0,
            starts[1] + self.num_frames_scale1,
            starts[2] + self.num_frames_scale2,
        ]

        # Slice codes, clamping to available tokens
        codes = [
            codes[0][starts[0]:min(ends[0], full_len_scale0)],
            codes[1][starts[1]:min(ends[1], full_len_scale1)],
            codes[2][starts[2]:min(ends[2], full_len_scale2)],
        ]

        # CRITICAL: If file is shorter than expected, pad to expected size
        # This ensures all batches can be stacked properly
        if codes[0].shape[0] < self.num_frames_scale0:
            pad_size = self.num_frames_scale0 - codes[0].shape[0]
            codes[0] = F.pad(codes[0], (0, pad_size), value=0)
        if codes[1].shape[0] < self.num_frames_scale1:
            pad_size = self.num_frames_scale1 - codes[1].shape[0]
            codes[1] = F.pad(codes[1], (0, pad_size), value=0)
        if codes[2].shape[0] < self.num_frames_scale2:
            pad_size = self.num_frames_scale2 - codes[2].shape[0]
            codes[2] = F.pad(codes[2], (0, pad_size), value=0)

        # Load pre-generated 48kHz audio directly
        audio_path = self.audio_48k_dir / row['file_path']
        try:
            audio, sr = torchaudio.load(audio_path)
            if sr != 48000:
                audio = torchaudio.functional.resample(audio, sr, 48000)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # Calculate aligned random start position for audio
            # hop_length = 441 (from encoder: 3*3*7*7)
            # start_sample_24k = start_pos * hop_length * vq_strides[0]
            # start_sample_48k = start_sample_24k * 2 (for 48kHz)
            hop_length = 441
            vq_stride_0 = vq_strides[0]
            start_sample_24k = start_pos * hop_length * vq_stride_0
            start_sample_48k = start_sample_24k * 2  # Convert to 48kHz

            # Slice audio from same random position
            end_sample_48k = start_sample_48k + self.num_audio_samples

            if end_sample_48k <= audio.shape[-1]:
                # Audio long enough, slice exact segment
                audio = audio[:, start_sample_48k:end_sample_48k].squeeze(0)
            else:
                # Audio too short, take from start_pos to end, then pad
                audio_slice = audio[:, start_sample_48k:].squeeze(0)
                padding = end_sample_48k - audio_slice.shape[-1]
                audio = F.pad(audio_slice, (0, padding))

            # CRITICAL: Ensure audio is exactly num_audio_samples
            if audio.shape[-1] != self.num_audio_samples:
                # Truncate if too long, pad if too short
                if audio.shape[-1] > self.num_audio_samples:
                    audio = audio[:self.num_audio_samples]
                else:
                    padding = self.num_audio_samples - audio.shape[-1]
                    audio = F.pad(audio, (0, padding))

            return {'codes': codes, 'audio': audio}
        except Exception as e:
            return None


def collate_fn(batch):
    """Collate function that filters out None values."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {'codes': None, 'audio': torch.empty(0)}

    audio_batch = torch.stack([item['audio'] for item in batch])
    audio_batch = audio_batch.unsqueeze(1)
    codes_batch = [item['codes'] for item in batch]
    return {'codes': codes_batch, 'audio': audio_batch}


def reconstruction_loss(pred, target):
    """L1 + multi-scale STFT loss."""
    l1_loss = F.l1_loss(pred, target)
    stft_loss = 0.0
    for n_fft in [1024, 2048, 4096]:
        pred_stft = torch.stft(pred.squeeze(1), n_fft=n_fft, return_complex=True)
        target_stft = torch.stft(target.squeeze(1), n_fft=n_fft, return_complex=True)
        stft_loss += F.l1_loss(pred_stft.abs(), target_stft.abs())
    stft_loss = stft_loss / 3
    loss = l1_loss + stft_loss
    return loss, l1_loss, stft_loss


class Decoder48kHz(nn.Module):
    """48kHz decoder using pretrained SNAC decoder + 2x upsampler."""
    def __init__(self, pretrained_snac):
        super().__init__()
        # Use only blocks 0-5 (upsampling path, outputs 64 channels at 24kHz)
        self.pretrained_decoder_upsamp = nn.Sequential(*list(pretrained_snac.decoder.model.children())[:6])

        # Get pretrained block 5 (128→64, stride=2) for smart weight initialization
        pretrained_block5 = list(pretrained_snac.decoder.model.children())[5]  # DecoderBlock

        # New layers for 48kHz output
        self.snake1d = Snake1d(64)
        self.upsampler2x = DecoderBlock(64, 64, stride=2, noise=False, groups=1)
        self.final_conv = nn.Conv1d(64, 1, kernel_size=7, padding=3)

        # Smart weight initialization: Adapt pretrained block5 weights
        # Pretrained: (64, 128, kernel_size) → New: (64, 64, kernel_size)
        try:
            with torch.no_grad():
                # Adapt ConvTranspose1d weights (average 128→64 input channels)
                pretrained_conv_weight = pretrained_block5.block[1].weight  # (64, 128, kernel_size)
                # Fastai approach: Average across input channel dimension
                adapted_weight = pretrained_conv_weight.mean(dim=1, keepdim=True).repeat(1, 64, 1)
                self.upsampler2x.block[1].weight.copy_(adapted_weight)

                # Copy bias directly (same shape: 64)
                if pretrained_block5.block[1].bias is not None:
                    self.upsampler2x.block[1].bias.copy_(pretrained_block5.block[1].bias)
            print("✓ Smart weight initialization successful")
        except Exception as e:
            print(f"⚠ Smart weight initialization failed: {e}")
            print("  Using random initialization instead")
            # Random initialization as fallback
            nn.init.xavier_uniform_(self.upsampler2x.block[1].weight)
            if self.upsampler2x.block[1].bias is not None:
                nn.init.zeros_(self.upsampler2x.block[1].bias)

        # Random init for final conv
        nn.init.xavier_uniform_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, z_q):
        # Get 64-channel features at 24kHz from frozen decoder
        h = self.pretrained_decoder_upsamp(z_q)
        # Upsample to 48kHz
        h = self.snake1d(h)
        h = self.upsampler2x(h)
        # Final audio output
        audio = self.final_conv(h)
        audio = torch.tanh(audio)
        return audio


def train_epoch(model, dataloader, pretrained_model, optimizer, device, epoch, scheduler=None, save_checkpoint_fn=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_l1 = 0
    total_stft = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        if batch['codes'] is None or batch['audio'].numel() == 0:
            continue

        codes_batch = batch['codes']
        audio_48k_target = batch['audio'].to(device)

        # Reorganize codes by scale for batch processing
        codes_batch_0 = torch.stack([c[0] for c in codes_batch]).to(device)
        codes_batch_1 = torch.stack([c[1] for c in codes_batch]).to(device)
        codes_batch_2 = torch.stack([c[2] for c in codes_batch]).to(device)

        # Convert codes to latent z_q
        with torch.no_grad():
            z_q = pretrained_model.quantizer.from_codes([codes_batch_0, codes_batch_1, codes_batch_2])

        # Forward pass
        audio_48k_pred = model(z_q)

        # Trim to match
        min_len = min(audio_48k_pred.shape[-1], audio_48k_target.shape[-1])
        audio_48k_pred = audio_48k_pred[..., :min_len]
        audio_48k_target = audio_48k_target[..., :min_len]

        # Compute loss and backward
        optimizer.zero_grad()
        loss, l1_loss, stft_loss = reconstruction_loss(audio_48k_pred, audio_48k_target)
        loss.backward()
        optimizer.step()

        # Step scheduler if provided
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_l1 += l1_loss.item()
        total_stft += stft_loss.item()
        num_batches += 1

        # Show current LR in progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'l1': f'{l1_loss.item():.4f}', 'stft': f'{stft_loss.item():.4f}', 'lr': f'{current_lr:.2e}'})

        # Checkpoint every 5000 iterations
        if save_checkpoint_fn and (batch_idx + 1) % 5000 == 0:
            save_checkpoint_fn(epoch, batch_idx + 1, loss.item())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_l1 = total_l1 / num_batches if num_batches > 0 else 0
    avg_stft = total_stft / num_batches if num_batches > 0 else 0

    return avg_loss, avg_l1, avg_stft


def validate(model, dataloader, pretrained_model, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_l1 = 0
    total_stft = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if batch['codes'] is None or batch['audio'].numel() == 0:
                continue

            codes_batch = batch['codes']
            audio_48k_target = batch['audio'].to(device)

            # Reorganize codes by scale for batch processing
            codes_batch_0 = torch.stack([c[0] for c in codes_batch]).to(device)
            codes_batch_1 = torch.stack([c[1] for c in codes_batch]).to(device)
            codes_batch_2 = torch.stack([c[2] for c in codes_batch]).to(device)

            # Convert codes to latent z_q
            z_q = pretrained_model.quantizer.from_codes([codes_batch_0, codes_batch_1, codes_batch_2])

            # Forward pass
            audio_48k_pred = model(z_q)

            # Trim to match
            min_len = min(audio_48k_pred.shape[-1], audio_48k_target.shape[-1])
            audio_48k_pred = audio_48k_pred[..., :min_len]
            audio_48k_target = audio_48k_target[..., :min_len]

            # Compute loss
            loss, l1_loss, stft_loss = reconstruction_loss(audio_48k_pred, audio_48k_target)
            total_loss += loss.item()
            total_l1 += l1_loss.item()
            total_stft += stft_loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_l1 = total_l1 / num_batches if num_batches > 0 else 0
    avg_stft = total_stft / num_batches if num_batches > 0 else 0

    return avg_loss, avg_l1, avg_stft


def reconstruction_loss(pred, target):
    """L1 + multi-scale STFT loss."""
    l1_loss = F.l1_loss(pred, target)
    stft_loss = 0.0
    for n_fft in [1024, 2048, 4096]:
        pred_stft = torch.stft(pred.squeeze(1), n_fft=n_fft, return_complex=True)
        target_stft = torch.stft(target.squeeze(1), n_fft=n_fft, return_complex=True)
        stft_loss += F.l1_loss(pred_stft.abs(), target_stft.abs())
    stft_loss = stft_loss / 3
    loss = l1_loss + stft_loss
    return loss, l1_loss, stft_loss


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Warmup learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Number of warmup epochs (frozen decoder)')
    parser.add_argument('--main_lr', type=float, default=1e-5, help='Main phase learning rate (unfrozen decoder)')
    parser.add_argument('--segment_schedule', type=str, default='1,2,3,4', help='Segment length schedule (e.g., "1,2,3,4" = 1s epochs 1-3, 2s epochs 4-6, etc.)')
    parser.add_argument('--cache_dir', type=str, default='/mnt/data/codes_phase11/train', help='Cache directory')
    parser.add_argument('--audio_48k_dir', type=str, default='/mnt/data/combine/train/audio_48khz', help='48kHz audio directory')
    parser.add_argument('--val_cache_dir', type=str, default='/mnt/data/codes_phase11/val', help='Validation cache directory')
    parser.add_argument('--val_audio_48k_dir', type=str, default='/mnt/data/combine/valid/audio_48khz', help='Validation 48kHz audio directory')
    parser.add_argument('--output_dir', type=str, default='checkpoints/phase11_decoder_48khz_cached', help='Output directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Optional checkpoint to load instead of pretrained model')
    parser.add_argument('--resume', type=str, default=None, help='Resume training from checkpoint (checkpoint_epochN.pt or best_model.pt)')
    parser.add_argument('--limit', type=int, default=None, help='Limit training to N rows (for testing)')
    parser.add_argument('--val_limit', type=int, default=None, help='Limit validation to N rows (for testing)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}')

    # Parse segment schedule
    segment_schedule = [float(x) for x in args.segment_schedule.split(',')]
    print(f"Segment schedule: {segment_schedule}")

    # Determine initial segment length (will be overridden if resuming)
    initial_segment = segment_schedule[0]

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = output_dir / 'training.log'
    log_handle = open(log_file, 'w')
    log_handle.write(f"Training log for Phase 11 (Cached Codes + Pre-generated 48kHz audio)\n")
    log_handle.write(f"Device: cuda:{args.device}\n")
    log_handle.write(f"Batch size: {args.batch_size}\n")
    log_handle.write(f"Warmup learning rate: {args.lr}\n")
    log_handle.write(f"Warmup epochs: {args.warmup_epochs}\n")
    log_handle.write(f"Main learning rate: {args.main_lr}\n")
    log_handle.write(f"Segment schedule: {segment_schedule}\n")
    log_handle.write(f"Total epochs: {args.epochs}\n")
    log_handle.write(f"Cache dir: {args.cache_dir}\n")
    log_handle.write(f"48kHz audio dir: {args.audio_48k_dir}\n\n")
    log_handle.flush()

    print("="*70)
    print("PHASE 11: 48kHz DECODER FINE-TUNING (CACHED CODES + 48kHz AUDIO)")
    print("="*70)

    # Load dataset with initial segment length
    print("\nLoading dataset...")
    initial_segment = segment_schedule[0]
    dataset = CachedCodesDataset(args.cache_dir, args.audio_48k_dir, segment_length=initial_segment, limit=args.limit)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=4, persistent_workers=False)
    print(f"Dataset size: {len(dataset)}")

    # Load validation dataset
    print("\nLoading validation dataset...")
    val_dataset = CachedCodesDataset(args.val_cache_dir, args.val_audio_48k_dir, segment_length=initial_segment, limit=args.val_limit)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_fn, num_workers=4, persistent_workers=False)
    print(f"Validation dataset size: {len(val_dataset)}")

    # Load pretrained SNAC
    print("\nLoading pretrained SNAC...")
    if args.checkpoint:
        print(f"Loading from checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        pretrained_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        pretrained_model.load_state_dict(checkpoint['model'])
        pretrained_model = pretrained_model.to(device)
        print("✓ Loaded checkpoint successfully")
    else:
        pretrained_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
    pretrained_model.eval()

    # Create 48kHz decoder
    print("\nCreating 48kHz decoder...")
    model = Decoder48kHz(pretrained_model).to(device)

    # Setup initial training state (will be overridden if resuming)
    start_epoch = 1
    scheduler = None
    current_phase = "WARMUP"

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n▶ Resuming from checkpoint: {args.resume}")
        resume_checkpoint = torch.load(args.resume, map_location='cpu')

        # Load model state
        model.load_state_dict(resume_checkpoint['model_state_dict'])

        # Determine current segment length from resumed epoch
        resume_epoch = resume_checkpoint['epoch']
        segment_idx = min(resume_epoch - 1, len(segment_schedule) - 1)
        initial_segment = segment_schedule[segment_idx]

        # Rebuild optimizer and scheduler based on phase
        if resume_epoch <= args.warmup_epochs:
            # Still in warmup phase
            print(f"  Resuming in WARMUP phase at epoch {resume_epoch}/{args.warmup_epochs}")
            for param in model.pretrained_decoder_upsamp.parameters():
                param.requires_grad = False

            optimizer = AdamW([
                {'params': model.upsampler2x.parameters(), 'lr': args.lr},
                {'params': model.final_conv.parameters(), 'lr': args.lr}
            ])
            scheduler = None
            current_phase = "WARMUP"
        else:
            # In main phase - create temp dataset to get batches_per_epoch
            print(f"  Resuming in MAIN phase at epoch {resume_epoch}")
            for param in model.pretrained_decoder_upsamp.parameters():
                param.requires_grad = True

            optimizer = AdamW([
                {'params': model.pretrained_decoder_upsamp.parameters(), 'lr': args.main_lr},
                {'params': model.upsampler2x.parameters(), 'lr': args.main_lr},
                {'params': model.final_conv.parameters(), 'lr': args.main_lr}
            ])

            current_phase = "MAIN"
            scheduler = None  # Will be created after dataset is loaded

        # Restore optimizer state
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        print("  ✓ Optimizer state restored")

        start_epoch = resume_epoch + 1  # Continue from next epoch
        print(f"  ✓ Starting from epoch {start_epoch}")
    else:
        # Fresh start - freeze pretrained decoder
        for param in model.pretrained_decoder_upsamp.parameters():
            param.requires_grad = False

        # Only train new layers
        optimizer = AdamW([
            {'params': model.upsampler2x.parameters(), 'lr': args.lr},
            {'params': model.final_conv.parameters(), 'lr': args.lr}
        ])

    # Load dataset with correct initial segment length
    print("\nLoading dataset...")
    dataset = CachedCodesDataset(args.cache_dir, args.audio_48k_dir, segment_length=initial_segment, limit=args.limit)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=4, persistent_workers=False)
    print(f"Dataset size: {len(dataset)}")

    # If resuming in MAIN phase, recreate scheduler now that we have dataset size
    if args.resume and resume_epoch > args.warmup_epochs:
        print("\nRecreating OneCycleLR scheduler...")
        batches_per_epoch = len(dataloader)
        total_steps = (args.epochs - args.warmup_epochs) * batches_per_epoch
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.main_lr * 10,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1e4
        )

        # Restore scheduler state if available
        if 'scheduler_state_dict' in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
            print("  ✓ Scheduler state restored")
        else:
            print("  ⚠ No scheduler state in checkpoint, starting fresh")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,} total, {trainable_params:,} trainable")

    # Training loop with warmup scheduler + one-cycle LR + segment length schedule
    print("\n" + "="*70)
    if args.resume:
        print(f"RESUMING TRAINING FROM EPOCH {start_epoch}")
    else:
        print("STARTING TRAINING WITH WARMUP + ONE-CYCLE LR + SEGMENT SCHEDULE")
    print("="*70)
    print(f"Phase 1 (Warmup): Epochs 1-{args.warmup_epochs}, LR={args.lr}, Frozen decoder")
    print(f"Phase 2 (Main): Epochs {args.warmup_epochs+1}-{args.epochs}, OneCycle LR, Unfrozen decoder")
    print(f"Segment schedule: {segment_schedule}")
    print("="*70)

    best_loss = float('inf')
    last_checkpoint_time = time.time()

    # Helper function to save intermediate checkpoints
    def save_intermediate_checkpoint(epoch, iteration, loss, phase, checkpoint_type="iter"):
        """Save intermediate checkpoint (5000 iterations or time-based)."""
        checkpoint = {
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'phase': phase,
            'warmup_epochs': args.warmup_epochs,
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if checkpoint_type == "iter":
            filename = output_dir / f'checkpoint_epoch{epoch}_iter{iteration}.pt'
        else:  # time-based
            filename = output_dir / f'checkpoint_epoch{epoch}_time_{int(time.time())}.pt'

        torch.save(checkpoint, filename)
        print(f"  → Intermediate checkpoint saved: {filename.name}")
        log_handle.write(f"  → Intermediate checkpoint saved: {filename.name}\n")
        log_handle.flush()

    for epoch in range(start_epoch, args.epochs + 1):
        # Update segment length based on schedule
        segment_idx = min(epoch - 1, len(segment_schedule) - 1)
        new_segment_length = segment_schedule[segment_idx]

        # Check if segment length changed
        if new_segment_length != dataset.segment_length:
            print(f"\n📏 Epoch {epoch}: Increasing segment length to {new_segment_length}s")
            dataset.set_segment_length(new_segment_length)
            val_dataset.set_segment_length(new_segment_length)
            # Recreate dataloader with new segment length
            # CRITICAL: persistent_workers=False ensures workers restart with new dataset state
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                    collate_fn=collate_fn, num_workers=4, persistent_workers=False)
            val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                       collate_fn=collate_fn, num_workers=4, persistent_workers=False)
            log_handle.write(f"Epoch {epoch}: Segment length → {new_segment_length}s\n")
            log_handle.flush()

        # Check if we need to transition to Phase 2
        if epoch == args.warmup_epochs + 1:
            print("\n" + "="*70)
            print("PHASE 2: UNFREEZING DECODER + ONE-CYCLE LR SCHEDULER")
            print("="*70)

            # Unfreeze pretrained decoder
            for param in model.pretrained_decoder_upsamp.parameters():
                param.requires_grad = True

            # Recreate optimizer with all parameters
            optimizer = AdamW([
                {'params': model.pretrained_decoder_upsamp.parameters(), 'lr': args.main_lr},
                {'params': model.upsampler2x.parameters(), 'lr': args.main_lr},
                {'params': model.final_conv.parameters(), 'lr': args.main_lr}
            ])

            # Create OneCycleLR scheduler for main phase
            # Total steps = (epochs - warmup_epochs) * batches_per_epoch
            batches_per_epoch = len(dataloader)
            total_steps = (args.epochs - args.warmup_epochs) * batches_per_epoch
            scheduler = OneCycleLR(
                optimizer,
                max_lr=args.main_lr * 10,  # Peak LR = 10x base LR
                total_steps=total_steps,
                pct_start=0.3,  # 30% warmup, 70% annealing
                anneal_strategy='cos',
                div_factor=25,  # Initial LR = max_lr / 25
                final_div_factor=1e4  # Final LR = max_lr / 1e4
            )

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Unfrozen! Now training {trainable_params:,} parameters")
            print(f"OneCycle LR: max_lr={args.main_lr * 10:.2e}, total_steps={total_steps}")
            log_handle.write(f"\n=== PHASE 2 STARTED: Unfrozen decoder, {trainable_params:,} trainable params ===\n")
            log_handle.write(f"OneCycle LR: max_lr={args.main_lr * 10:.2e}, total_steps={total_steps}\n")
            log_handle.flush()

        # Create checkpoint function for this epoch
        phase = "WARMUP" if epoch <= args.warmup_epochs else "MAIN"
        def make_checkpoint_fn(epoch, phase):
            return lambda iteration, loss: save_intermediate_checkpoint(epoch, iteration, loss, phase, "iter")

        # Train for one epoch
        avg_loss, avg_l1, avg_stft = train_epoch(model, dataloader, pretrained_model, optimizer, device, epoch, scheduler, make_checkpoint_fn(epoch, phase))

        # Log with phase indicator and segment length
        log_msg = f"Epoch {epoch} [{phase}, {dataset.segment_length}s]: Loss={avg_loss:.4f}, L1={avg_l1:.4f}, STFT={avg_stft:.4f}"
        print(log_msg)
        log_handle.write(log_msg + "\n")
        log_handle.flush()

        # Validate
        print("\nRunning validation...")
        val_loss, val_l1, val_stft = validate(model, val_dataloader, pretrained_model, device)
        val_msg = f"Validation: Loss={val_loss:.4f}, L1={val_l1:.4f}, STFT={val_stft:.4f}"
        print(val_msg)
        log_handle.write(val_msg + "\n")
        log_handle.flush()

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'val_loss': val_loss,
            'phase': phase,
            'warmup_epochs': args.warmup_epochs,
        }

        # Save scheduler state if available
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(checkpoint, output_dir / f'checkpoint_epoch{epoch}.pt')

        # Save best model based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(checkpoint, output_dir / 'best_model.pt')
            print(f"  → New best model saved (val_loss: {val_loss:.4f})")
            log_handle.write(f"  → New best model saved (val_loss: {val_loss:.4f})\n")
            log_handle.flush()

        # Time-based checkpointing (every 30 minutes)
        elapsed_time = time.time() - last_checkpoint_time
        if elapsed_time >= 1800:  # 30 minutes
            save_intermediate_checkpoint(epoch, 0, avg_loss, phase, "time")
            last_checkpoint_time = time.time()

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    log_handle.close()


if __name__ == '__main__':
    main()

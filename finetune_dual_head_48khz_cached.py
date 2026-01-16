"""
Phase 12: Dual-Head Decoder Fine-tuning (24kHz + 48kHz)
Uses cached codes + dual-head architecture with separate outputs for 24kHz and 48kHz.
- 24kHz head: Continues from Phase 10 weights (already trained)
- 48kHz head: New 2x upsampler + final conv
- Training strategy: Different LRs for different components, weighted dual loss
"""
import os
import time
import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd
import torchaudio
from tqdm import tqdm
import numpy as np
import random

os.environ['HF_HOME'] = '/mnt/data/work/snac/.hf_cache'
os.environ['HF_HUB_CACHE'] = '/mnt/data/work/snac/.hf_cache/hub'

from snac import SNAC
from snac.layers import Decoder, DecoderBlock, Snake1d


def calculate_num_workers(batch_size, max_workers=16):
    """Calculate optimal number of workers based on batch size (Phase 10 formula)."""
    return min(batch_size // 2, max_workers)


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
        """Calculate token counts for current segment length."""
        num_latent_frames = int(self.segment_length * self.latent_frame_rate)

        # Round DOWN to multiple of 8 (coarsest VQ stride)
        num_latent_frames = (num_latent_frames // 8) * 8

        # Now divide by strides - all will produce integer multiples
        self.num_frames_scale0 = num_latent_frames // 8
        self.num_frames_scale1 = num_latent_frames // 4
        self.num_frames_scale2 = num_latent_frames // 2

        self.num_audio_samples_48k = int(self.segment_length * 48000)
        self.num_audio_samples_24k = int(self.segment_length * 24000)
        print(f"Segment {self.segment_length}s: {self.num_frames_scale0}/{self.num_frames_scale1}/{self.num_frames_scale2} tokens, {self.num_audio_samples_48k} audio samples (48kHz)")

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
                    return torch.from_numpy(obj.copy()).long()
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
        full_len_scale0 = codes[0].shape[0]
        full_len_scale1 = codes[1].shape[0]
        full_len_scale2 = codes[2].shape[0]

        max_start_scale0 = max(0, full_len_scale0 - self.num_frames_scale0)
        max_start_scale1 = max(0, full_len_scale1 - self.num_frames_scale1)
        max_start_scale2 = max(0, full_len_scale2 - self.num_frames_scale2)

        max_start_pos = min(max_start_scale0, max_start_scale1, max_start_scale2)

        if max_start_pos > 0:
            start_pos = random.randint(0, max_start_pos)
        else:
            start_pos = 0

        # Multi-scale alignment for random slicing
        vq_strides = [8, 4, 2]

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

        codes = [
            codes[0][starts[0]:min(ends[0], full_len_scale0)],
            codes[1][starts[1]:min(ends[1], full_len_scale1)],
            codes[2][starts[2]:min(ends[2], full_len_scale2)],
        ]

        # Pad if needed
        if codes[0].shape[0] < self.num_frames_scale0:
            pad_size = self.num_frames_scale0 - codes[0].shape[0]
            codes[0] = F.pad(codes[0], (0, pad_size), value=0)
        if codes[1].shape[0] < self.num_frames_scale1:
            pad_size = self.num_frames_scale1 - codes[1].shape[0]
            codes[1] = F.pad(codes[1], (0, pad_size), value=0)
        if codes[2].shape[0] < self.num_frames_scale2:
            pad_size = self.num_frames_scale2 - codes[2].shape[0]
            codes[2] = F.pad(codes[2], (0, pad_size), value=0)

        # Load 48kHz audio
        audio_path = self.audio_48k_dir / row['file_path']
        try:
            audio_48k, sr = torchaudio.load(audio_path)
            if sr != 48000:
                audio_48k = torchaudio.functional.resample(audio_48k, sr, 48000)
            if audio_48k.shape[0] > 1:
                audio_48k = torch.mean(audio_48k, dim=0, keepdim=True)

            hop_length = 441
            vq_stride_0 = vq_strides[0]
            start_sample_24k = start_pos * hop_length * vq_stride_0
            start_sample_48k = start_sample_24k * 2

            end_sample_48k = start_sample_48k + self.num_audio_samples_48k

            if end_sample_48k <= audio_48k.shape[-1]:
                audio_48k = audio_48k[:, start_sample_48k:end_sample_48k].squeeze(0)
            else:
                audio_slice = audio_48k[:, start_sample_48k:].squeeze(0)
                padding = end_sample_48k - audio_slice.shape[-1]
                audio_48k = F.pad(audio_slice, (0, padding))

            if audio_48k.shape[-1] != self.num_audio_samples_48k:
                if audio_48k.shape[-1] > self.num_audio_samples_48k:
                    audio_48k = audio_48k[:self.num_audio_samples_48k]
                else:
                    padding = self.num_audio_samples_48k - audio_48k.shape[-1]
                    audio_48k = F.pad(audio_48k, (0, padding))

            # Downsample to 24kHz for 24kHz target
            audio_24k = torchaudio.functional.resample(audio_48k, 48000, 24000)

            return {'codes': codes, 'audio_48k': audio_48k, 'audio_24k': audio_24k}
        except Exception as e:
            return None


def collate_fn(batch):
    """Collate function that filters out None values."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {'codes': None, 'audio_48k': torch.empty(0), 'audio_24k': torch.empty(0)}

    audio_48k_batch = torch.stack([item['audio_48k'] for item in batch])
    audio_48k_batch = audio_48k_batch.unsqueeze(1)
    audio_24k_batch = torch.stack([item['audio_24k'] for item in batch])
    audio_24k_batch = audio_24k_batch.unsqueeze(1)
    codes_batch = [item['codes'] for item in batch]
    return {'codes': codes_batch, 'audio_48k': audio_48k_batch, 'audio_24k': audio_24k_batch}


class DualHeadDecoder(nn.Module):
    """Dual-head decoder with 24kHz and 48kHz outputs.

    Architecture:
        Shared Decoder (blocks 0-5) → 64-channel features @ 24kHz
                                        ├→ 24kHz Final Conv → 24kHz output
                                        └→ 2x Upsampler → 48kHz Final Conv → 48kHz output
    """
    def __init__(self, pretrained_snac, phase10_checkpoint_path=None):
        super().__init__()

        # Shared: pretrained decoder blocks 0-5 (outputs 64 channels @ 24kHz)
        # Using pretrained SNAC decoder to match the pretrained SNAC encoder
        # (Phase 10 decoder is trained with Phase 10 encoder, not compatible)
        self.pretrained_decoder_upsamp = nn.Sequential(*list(pretrained_snac.decoder.model.children())[:6])
        print("✓ Using pretrained SNAC decoder (matches pretrained SNAC encoder)")

        # Get pretrained block 5 for smart weight initialization of 48kHz upsampler
        pretrained_block5 = list(pretrained_snac.decoder.model.children())[5]

        # ===== 24kHz Head (from pretrained SNAC) =====
        # Using pretrained SNAC final conv to match pretrained SNAC decoder
        # (Phase 10 final conv is trained with Phase 10 decoder, not compatible)
        self.final_conv_24k = nn.Conv1d(64, 1, kernel_size=7, padding=3)

        try:
            with torch.no_grad():
                pretrained_final_conv = list(pretrained_snac.decoder.model.children())[7]
                self.final_conv_24k.weight.data.copy_(pretrained_final_conv.weight)
                if pretrained_final_conv.bias is not None:
                    self.final_conv_24k.bias.data.copy_(pretrained_final_conv.bias)
            print("✓ Loaded 24kHz final conv weights from pretrained SNAC (matches pretrained SNAC decoder)")
        except Exception as e:
            print(f"⚠ Failed to load pretrained SNAC final conv: {e}")
            print("  Using random initialization for 24kHz final conv")
            nn.init.xavier_uniform_(self.final_conv_24k.weight)
            nn.init.zeros_(self.final_conv_24k.bias)

        # ===== 48kHz Head (new) =====
        self.snake1d = Snake1d(64)
        self.upsampler2x = DecoderBlock(64, 64, stride=2, noise=False, groups=1)
        self.final_conv_48k = nn.Conv1d(64, 1, kernel_size=7, padding=3)

        # Smart weight initialization for 48kHz upsampler
        # Pretrained block 5: ConvTranspose1d [128, 64, 4] (128 out channels)
        # New upsampler: ConvTranspose1d [64, 64, 4] (64 out channels)
        try:
            with torch.no_grad():
                pretrained_conv_weight = pretrained_block5.block[1].weight  # [128, 64, 4]
                # Take first 64 channels from pretrained 128 channels
                adapted_weight = pretrained_conv_weight[:64, :, :]  # [64, 64, 4]
                self.upsampler2x.block[1].weight.copy_(adapted_weight)

                if pretrained_block5.block[1].bias is not None:
                    # Take first 64 bias values
                    self.upsampler2x.block[1].bias.copy_(pretrained_block5.block[1].bias[:64])
            print("✓ Smart weight initialization for 48kHz upsampler successful")
        except Exception as e:
            print(f"⚠ Smart weight initialization failed: {e}")
            print("  Using random initialization instead")
            nn.init.xavier_uniform_(self.upsampler2x.block[1].weight)
            if self.upsampler2x.block[1].bias is not None:
                nn.init.zeros_(self.upsampler2x.block[1].bias)

        # Random init for 48kHz final conv
        nn.init.xavier_uniform_(self.final_conv_48k.weight)
        nn.init.zeros_(self.final_conv_48k.bias)

    def forward(self, z_q, output_24k=True, output_48k=True):
        """
        Args:
            z_q: Quantized latent codes [B, 64, T]
            output_24k: Whether to output 24kHz
            output_48k: Whether to output 48kHz

        Returns:
            Tuple of (audio_24k, audio_48k)
            Each is None if corresponding output flag is False
        """
        # Shared decoder: get 64-channel features at 24kHz
        h = self.pretrained_decoder_upsamp(z_q)  # [B, 64, T] @ 24kHz

        audio_24k = None
        audio_48k = None

        # 24kHz output path
        if output_24k:
            audio_24k = self.final_conv_24k(h)
            audio_24k = torch.tanh(audio_24k)

        # 48kHz output path
        if output_48k:
            h_upsampled = self.snake1d(h)
            h_upsampled = self.upsampler2x(h_upsampled)
            audio_48k = self.final_conv_48k(h_upsampled)
            audio_48k = torch.tanh(audio_48k)

        return audio_24k, audio_48k


def reconstruction_loss(pred, target, n_ffts=[1024, 2048, 4096, 8192]):
    """Compute reconstruction loss: L1 + multi-scale STFT."""
    # Trim to match
    min_len = min(pred.shape[-1], target.shape[-1])
    pred = pred[..., :min_len]
    target = target[..., :min_len]

    # L1 loss
    l1_loss = F.l1_loss(pred, target)

    # Multi-scale STFT loss
    stft_loss = 0.0
    for n_fft in n_ffts:
        pred_stft = torch.stft(pred.squeeze(1), n_fft=n_fft, return_complex=True)
        target_stft = torch.stft(target.squeeze(1), n_fft=n_fft, return_complex=True)
        stft_loss += F.l1_loss(pred_stft.real, target_stft.real) + \
                    F.l1_loss(pred_stft.imag, target_stft.imag)
    stft_loss = stft_loss / len(n_ffts)

    total_loss = l1_loss + stft_loss
    return total_loss, l1_loss, stft_loss


def train_epoch(model, dataloader, pretrained_model, optimizer, device, epoch,
                scheduler=None, loss_weights_24k=0.0, loss_weights_48k=1.0,
                save_checkpoint_fn=None, training_interrupted=None):
    """Train for one epoch with dual-head loss."""
    model.train()
    total_loss = 0
    total_loss_24k = 0
    total_loss_48k = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        if training_interrupted is not None and training_interrupted[0]:
            print("\n⚠ Training interrupted mid-epoch! Breaking training loop...")
            break

        if batch['codes'] is None or batch['audio_48k'].numel() == 0:
            continue

        codes_batch = batch['codes']
        audio_24k_target = batch['audio_24k'].to(device)
        audio_48k_target = batch['audio_48k'].to(device)

        # Reorganize codes by scale
        codes_batch_0 = torch.stack([c[0] for c in codes_batch]).to(device)
        codes_batch_1 = torch.stack([c[1] for c in codes_batch]).to(device)
        codes_batch_2 = torch.stack([c[2] for c in codes_batch]).to(device)

        # Convert codes to latent z_q
        with torch.no_grad():
            z_q = pretrained_model.quantizer.from_codes([codes_batch_0, codes_batch_1, codes_batch_2])

        # Forward pass (dual output)
        audio_24k_pred, audio_48k_pred = model(z_q, output_24k=True, output_48k=True)

        # Compute losses for both heads
        loss_24k, l1_24k, stft_24k = reconstruction_loss(audio_24k_pred, audio_24k_target, n_ffts=[1024, 2048, 4096])
        loss_48k, l1_48k, stft_48k = reconstruction_loss(audio_48k_pred, audio_48k_target, n_ffts=[1024, 2048, 4096, 8192])

        # Weighted combined loss
        loss = loss_weights_24k * loss_24k + loss_weights_48k * loss_48k

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_loss_24k += loss_24k.item()
        total_loss_48k += loss_48k.item()
        num_batches += 1

        # Show current LR and losses in progress bar
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            '24k': f'{loss_24k.item():.4f}',
            '48k': f'{loss_48k.item():.4f}',
            'lr': f'{current_lr:.2e}'
        })

        # Checkpoint every 5000 iterations
        if save_checkpoint_fn and (batch_idx + 1) % 5000 == 0:
            save_checkpoint_fn(batch_idx + 1, loss.item())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_loss_24k = total_loss_24k / num_batches if num_batches > 0 else 0
    avg_loss_48k = total_loss_48k / num_batches if num_batches > 0 else 0

    return avg_loss, avg_loss_24k, avg_loss_48k


def validate(model, dataloader, pretrained_model, device):
    """Validate the dual-head model."""
    model.eval()
    random.seed(0)  # Fixed seed for consistent validation

    total_loss_24k = 0
    total_loss_48k = 0
    total_l1_24k = 0
    total_l1_48k = 0
    total_stft_24k = 0
    total_stft_48k = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if batch['codes'] is None or batch['audio_48k'].numel() == 0:
                continue

            codes_batch = batch['codes']
            audio_24k_target = batch['audio_24k'].to(device)
            audio_48k_target = batch['audio_48k'].to(device)

            codes_batch_0 = torch.stack([c[0] for c in codes_batch]).to(device)
            codes_batch_1 = torch.stack([c[1] for c in codes_batch]).to(device)
            codes_batch_2 = torch.stack([c[2] for c in codes_batch]).to(device)

            z_q = pretrained_model.quantizer.from_codes([codes_batch_0, codes_batch_1, codes_batch_2])

            audio_24k_pred, audio_48k_pred = model(z_q, output_24k=True, output_48k=True)

            # Compute both losses
            loss_24k, l1_24k, stft_24k = reconstruction_loss(audio_24k_pred, audio_24k_target, n_ffts=[1024, 2048, 4096])
            loss_48k, l1_48k, stft_48k = reconstruction_loss(audio_48k_pred, audio_48k_target, n_ffts=[1024, 2048, 4096, 8192])

            total_loss_24k += loss_24k.item()
            total_loss_48k += loss_48k.item()
            total_l1_24k += l1_24k.item()
            total_l1_48k += l1_48k.item()
            total_stft_24k += stft_24k.item()
            total_stft_48k += stft_48k.item()
            num_batches += 1

    avg_loss_24k = total_loss_24k / num_batches if num_batches > 0 else 0
    avg_loss_48k = total_loss_48k / num_batches if num_batches > 0 else 0
    avg_l1_24k = total_l1_24k / num_batches if num_batches > 0 else 0
    avg_l1_48k = total_l1_48k / num_batches if num_batches > 0 else 0
    avg_stft_24k = total_stft_24k / num_batches if num_batches > 0 else 0
    avg_stft_48k = total_stft_48k / num_batches if num_batches > 0 else 0

    return avg_loss_24k, avg_loss_48k, avg_l1_24k, avg_l1_48k, avg_stft_24k, avg_stft_48k


def save_checkpoint(model, optimizer, scheduler, epoch, batch_idx, loss_24k, loss_48k,
                    output_dir, filename_prefix="checkpoint"):
    """Save training checkpoint."""
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, f"{filename_prefix}_epoch{epoch}_batch{batch_idx}.pt")
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'loss_24k': loss_24k,
        'loss_48k': loss_48k,
    }, checkpoint_path)
    print(f"  → Checkpoint saved: {checkpoint_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Phase 12: Dual-Head Decoder Fine-tuning (24kHz + 48kHz)")
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--batch_size', type=int, default=96, help='Base batch size')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='Number of warmup epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Warmup learning rate')
    parser.add_argument('--main_lr', type=float, default=5e-5, help='Main phase learning rate')
    parser.add_argument('--lr_24k_final_conv', type=float, default=1e-5, help='Learning rate for 24kHz final conv (main phase)')
    parser.add_argument('--segment_schedule', type=str, default='1.0,2.0,3.0,4.0', help='Segment length schedule')
    parser.add_argument('--batch_multiplier', type=str, default='2.0,1.0,0.6,0.45', help='Batch multipliers')
    parser.add_argument('--epoch_ranges', type=str, default='1-2,3-4,5-6,7-15', help='Epoch ranges for each segment (e.g., "1-2,3-4,5-6,7-15")')
    parser.add_argument('--cache_dir', type=str, default='/mnt/data/codes_phase11/train', help='Cached codes directory')
    parser.add_argument('--audio_48k_dir', type=str, default='/mnt/data/combine/train/audio_48khz', help='48kHz audio directory')
    parser.add_argument('--val_cache_dir', type=str, default='/mnt/data/codes_phase11/val', help='Validation cache dir')
    parser.add_argument('--val_audio_48k_dir', type=str, default='/mnt/data/combine/valid/audio_48khz', help='Validation 48kHz audio dir')
    parser.add_argument('--output_dir', type=str, default='checkpoints/phase12_dual_head', help='Output directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--resume', type=str, default=None, help='Resume from specific checkpoint')
    parser.add_argument('--limit', type=int, default=None, help='Limit dataset size (for testing)')
    parser.add_argument('--val_limit', type=int, default=None, help='Limit validation dataset size')
    parser.add_argument('--phase10_checkpoint', type=str, default='checkpoints/phase10_revolab_all/best_model.pt',
                        help='Path to Phase 10 checkpoint for 24kHz head initialization')
    parser.add_argument('--loss_weight_24k', type=float, default=0.3, help='Loss weight for 24kHz (main phase)')
    parser.add_argument('--loss_weight_48k', type=float, default=1.0, help='Loss weight for 48kHz')

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Parse segment schedule and batch multipliers
    segment_schedule = [float(x) for x in args.segment_schedule.split(',')]
    batch_multipliers = [float(x) for x in args.batch_multiplier.split(',')]

    # Parse epoch ranges (e.g., "1-2,3-4,5-6,7-15")
    epoch_ranges = []
    for range_str in args.epoch_ranges.split(','):
        start, end = map(int, range_str.split('-'))
        epoch_ranges.append((start, end))

    if len(batch_multipliers) != len(segment_schedule):
        raise ValueError(f"Batch multiplier count ({len(batch_multipliers)}) must match segment schedule count ({len(segment_schedule)})")

    if len(epoch_ranges) != len(segment_schedule):
        raise ValueError(f"Epoch range count ({len(epoch_ranges)}) must match segment schedule count ({len(segment_schedule)})")

    print(f"Segment schedule: {segment_schedule}")
    print(f"Batch multipliers: {batch_multipliers}")
    print(f"Epoch ranges: {epoch_ranges}")
    print(f"Dynamic batch sizes: {[int(args.batch_size * m) for m in batch_multipliers]}")

    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    print("Random seed set to 42 for reproducibility")

    print("\n" + "="*70)
    print("PHASE 12: DUAL-HEAD DECODER FINE-TUNING (24kHz + 48kHz)")
    print("="*70)

    print("\nLoading pretrained SNAC...")
    pretrained_snac = SNAC.from_pretrained('hubertsiuzdak/snac_24khz')
    pretrained_snac.eval()
    for param in pretrained_snac.parameters():
        param.requires_grad = False
    pretrained_snac = pretrained_snac.to(device)  # Move to device for quantizer

    print("\nCreating dual-head decoder...")
    model = DualHeadDecoder(pretrained_snac, phase10_checkpoint_path=args.phase10_checkpoint).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,} total, {trainable_params:,} trainable")

    print("\nLoading dataset...")
    dataset = CachedCodesDataset(args.cache_dir, args.audio_48k_dir, segment_length=segment_schedule[0], limit=args.limit)

    val_dataset = CachedCodesDataset(args.val_cache_dir, args.val_audio_48k_dir,
                                     segment_length=segment_schedule[0], limit=args.val_limit)

    print(f"\nDataset size: {len(dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Create dataloaders
    initial_batch_multiplier = batch_multipliers[0]
    initial_batch_size = int(args.batch_size * initial_batch_multiplier)
    num_workers = calculate_num_workers(initial_batch_size, max_workers=16)

    print(f"Initial batch size: {args.batch_size} × {initial_batch_multiplier} = {initial_batch_size}")
    print(f"Workers: {num_workers} (dynamic: min(batch_size//2, 16))")

    dataloader = DataLoader(dataset, batch_size=initial_batch_size, shuffle=True,
                           collate_fn=collate_fn, num_workers=num_workers, persistent_workers=False)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                               collate_fn=collate_fn, num_workers=num_workers, persistent_workers=False)

    # Training state
    start_epoch = 1
    best_val_loss_48k = float('inf')
    best_val_loss_24k = float('inf')
    best_combined_loss = float('inf')

    # Resume logic
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss_48k = checkpoint.get('best_val_loss_48k', float('inf'))
        best_val_loss_24k = checkpoint.get('best_val_loss_24k', float('inf'))
        best_combined_loss = checkpoint.get('best_combined_loss', float('inf'))
        print(f"Resumed from epoch {checkpoint['epoch']}, best 24k: {best_val_loss_24k:.4f}, best 48k: {best_val_loss_48k:.4f}")

    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"Total epochs: {args.epochs}")
    print(f"Warmup epochs: {args.warmup_epochs}")
    print(f"Warmup LR: {args.lr}")
    print(f"Main LR: {args.main_lr}")
    print(f"24kHz final conv LR: {args.lr_24k_final_conv}")
    print(f"Loss weights: 24kHz={args.loss_weight_24k}, 48kHz={args.loss_weight_48k}")
    print(f"Base batch size: {args.batch_size}")
    print(f"Segment schedule: {segment_schedule}")
    print("="*70)

    # Training interrupted flag for signal handling
    training_interrupted = [False]

    def signal_handler(sig, frame):
        print("\n\n⚠ Signal received! Setting interrupt flag...")
        training_interrupted[0] = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    def create_save_checkpoint_fn(epoch, batch_idx):
        def save_fn(batch_idx, loss):
            # Determine which loss to use for checkpointing
            save_checkpoint(model, None, None, epoch, batch_idx, None, loss,
                          args.output_dir, filename_prefix="checkpoint_intermediate")
        return save_fn

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        # Update segment length based on epoch ranges (like Phase 10)
        current_segment_idx = None
        for idx, (start, end) in enumerate(epoch_ranges):
            if start <= epoch <= end:
                current_segment_idx = idx
                break

        if current_segment_idx is None:
            # If epoch is beyond all ranges, use the last segment
            current_segment_idx = len(segment_schedule) - 1

        current_segment_length = segment_schedule[current_segment_idx]

        if epoch > 1 and current_segment_length != dataset.segment_length:
            print(f"\n📏 Epoch {epoch}: Increasing segment length to {current_segment_length}s")
            dataset.set_segment_length(current_segment_length)
            val_dataset.set_segment_length(current_segment_length)

            # Recreate dataloaders with new segment length
            batch_multiplier = batch_multipliers[current_segment_idx]
            new_batch_size = int(args.batch_size * batch_multiplier)
            print(f"📦 Batch size: {args.batch_size} × {batch_multiplier} = {new_batch_size}")

            new_num_workers = calculate_num_workers(new_batch_size, max_workers=16)
            print(f"👷 Workers: {new_num_workers} (dynamic: min({new_batch_size}//2, 16))")

            dataloader = DataLoader(dataset, batch_size=new_batch_size, shuffle=True,
                                   collate_fn=collate_fn, num_workers=new_num_workers, persistent_workers=False)

        # Determine phase
        is_warmup = epoch <= args.warmup_epochs

        if is_warmup:
            print("\n" + "="*70)
            print(f"PHASE 1: WARMUP (Epoch {epoch}/{args.warmup_epochs})")
            print("="*70)
            print("Frozen: Shared decoder + 24kHz final conv")
            print("Trainable: 48kHz upsampler + 48kHz final conv")
            print(f"LR: {args.lr}")
            print(f"Loss: 48kHz only (24kHz for monitoring)")

            # Freeze shared decoder and 24kHz final conv
            for param in model.pretrained_decoder_upsamp.parameters():
                param.requires_grad = False
            model.final_conv_24k.requires_grad_(False)

            # Unfreeze 48kHz components
            for param in model.snake1d.parameters():
                param.requires_grad = True
            for param in model.upsampler2x.parameters():
                param.requires_grad = True
            model.final_conv_48k.requires_grad_(True)

            # Optimizer for warmup (only 48kHz components)
            optimizer = AdamW([
                {'params': model.snake1d.parameters(), 'lr': args.lr},
                {'params': model.upsampler2x.parameters(), 'lr': args.lr},
                {'params': model.final_conv_48k.parameters(), 'lr': args.lr},
            ])

            # No scheduler in warmup (constant LR)
            scheduler = None

            # Loss weights: 48kHz only
            loss_weight_24k = 0.0
            loss_weight_48k = 1.0

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable params: {trainable:,}")

        else:
            print("\n" + "="*70)
            print(f"PHASE 2: MAIN TRAINING (Epoch {epoch})")
            print("="*70)
            print("Unfrozen: Shared decoder + 24kHz final conv + 48kHz components")
            print(f"Shared decoder LR: {args.main_lr}")
            print(f"48kHz components LR: {args.main_lr}")
            print(f"24kHz final conv LR: {args.lr_24k_final_conv} (smaller for fine-tuning)")
            print(f"Loss weights: 24kHz={args.loss_weight_24k}, 48kHz={args.loss_weight_48k}")

            # Unfreeze shared decoder
            for param in model.pretrained_decoder_upsamp.parameters():
                param.requires_grad = True

            # Unfreeze 24kHz final conv with smaller LR
            model.final_conv_24k.requires_grad_(True)

            # Keep 48kHz components unfrozen
            for param in model.snake1d.parameters():
                param.requires_grad = True
            for param in model.upsampler2x.parameters():
                param.requires_grad = True
            model.final_conv_48k.requires_grad_(True)

            # Optimizer with gradient groups (different LRs)
            num_steps_per_epoch = len(dataloader)
            total_steps = num_steps_per_epoch * (args.epochs - args.warmup_epochs)

            optimizer = AdamW([
                {'params': model.pretrained_decoder_upsamp.parameters(), 'lr': args.main_lr},
                {'params': model.snake1d.parameters(), 'lr': args.main_lr},
                {'params': model.upsampler2x.parameters(), 'lr': args.main_lr},
                {'params': model.final_conv_48k.parameters(), 'lr': args.main_lr},
                {'params': model.final_conv_24k.parameters(), 'lr': args.lr_24k_final_conv},
            ])

            # OneCycleLR scheduler for main phase
            scheduler = OneCycleLR(
                optimizer,
                max_lr=args.main_lr * 10,  # Peak LR
                total_steps=total_steps,
                pct_start=0.3,  # 30% warmup
                anneal_strategy='cos',
                final_div_factor=1000,
            )

            loss_weight_24k = args.loss_weight_24k
            loss_weight_48k = args.loss_weight_48k

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable params: {trainable:,}")
            print(f"OneCycle LR: max_lr={args.main_lr * 10:.2e}, total_steps={total_steps}")

        print("\nTraining...")
        avg_loss, avg_loss_24k, avg_loss_48k = train_epoch(
            model, dataloader, pretrained_snac, optimizer, device, epoch,
            scheduler=scheduler,
            loss_weights_24k=loss_weight_24k,
            loss_weights_48k=loss_weight_48k,
            save_checkpoint_fn=create_save_checkpoint_fn(epoch, 0) if is_warmup else None,
            training_interrupted=training_interrupted
        )

        print(f"\nEpoch {epoch} [Train]: Loss={avg_loss:.4f}, 24kHz={avg_loss_24k:.4f}, 48kHz={avg_loss_48k:.4f}")

        if training_interrupted[0]:
            print("\n⚠ Saving emergency checkpoint before exit...")
            save_checkpoint(model, optimizer, scheduler, epoch, len(dataloader),
                          avg_loss_24k, avg_loss_48k, args.output_dir,
                          filename_prefix="checkpoint_interrupted")
            break

        # Validation
        print("\nRunning validation...")
        val_loss_24k, val_loss_48k, val_l1_24k, val_l1_48k, val_stft_24k, val_stft_48k = validate(
            model, val_dataloader, pretrained_snac, device
        )

        print(f"\nEpoch {epoch} [Val]:")
        print(f"  24kHz: Loss={val_loss_24k:.4f} (L1={val_l1_24k:.4f}, STFT={val_stft_24k:.4f})")
        print(f"  48kHz: Loss={val_loss_48k:.4f} (L1={val_l1_48k:.4f}, STFT={val_stft_48k:.4f})")

        # Save best models
        combined_loss = loss_weight_24k * val_loss_24k + loss_weight_48k * val_loss_48k

        is_best_24k = val_loss_24k < best_val_loss_24k
        is_best_48k = val_loss_48k < best_val_loss_48k
        is_best_combined = combined_loss < best_combined_loss

        if is_best_24k:
            best_val_loss_24k = val_loss_24k
            print("  → New best 24kHz model!")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss_24k': val_loss_24k,
                'val_loss_48k': val_loss_48k,
            }, os.path.join(args.output_dir, 'best_model_24k.pt'))

        if is_best_48k:
            best_val_loss_48k = val_loss_48k
            print("  → New best 48kHz model!")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss_24k': val_loss_24k,
                'val_loss_48k': val_loss_48k,
            }, os.path.join(args.output_dir, 'best_model_48k.pt'))

        if is_best_combined:
            best_combined_loss = combined_loss
            print("  → New best combined model!")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss_24k': val_loss_24k,
                'val_loss_48k': val_loss_48k,
                'best_val_loss_24k': best_val_loss_24k,
                'best_val_loss_48k': best_val_loss_48k,
                'best_combined_loss': best_combined_loss,
            }, os.path.join(args.output_dir, 'best_model.pt'))

        # Save epoch checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, len(dataloader),
                       val_loss_24k, val_loss_48k, args.output_dir,
                       filename_prefix=f"checkpoint_epoch{epoch}")

    print("\nTraining complete!")
    print(f"\nFinal Results:")
    print(f"  Best 24kHz val loss: {best_val_loss_24k:.4f}")
    print(f"  Best 48kHz val loss: {best_val_loss_48k:.4f}")


if __name__ == '__main__':
    main()

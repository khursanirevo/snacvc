"""
Quick training test with cached codes.
Tests a few iterations to verify the full training pipeline works.
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd
import torchaudio

os.environ['HF_HOME'] = '/mnt/data/work/snac/.hf_cache'
os.environ['HF_HUB_CACHE'] = '/mnt/data/work/snac/.hf_cache/hub'

from snac import SNAC
from snac.layers import Decoder, DecoderBlock, Snake1d
from finetune_decoder_48khz import SIDONUpsampler

device = torch.device('cuda:0')

print("="*70)
print("QUICK TRAINING TEST: Cached Codes (First 3 Batches)")
print("="*70)

# Dataset
class CachedCodesDataset(Dataset):
    def __init__(self, cache_dir, max_files=3):
        self.cache_dir = Path(cache_dir)
        self.audio_dir = Path("/mnt/data/combine/train/audio")
        self.parquet_files = sorted(self.cache_dir.glob("codes_batch_gpu*.parquet"))[:max_files]

        self.index = []
        for file_idx, parquet_file in enumerate(self.parquet_files):
            df = pd.read_parquet(parquet_file, columns=['file_path'])
            for row_idx in range(len(df)):
                self.index.append({'parquet_file': parquet_file, 'row_idx': row_idx})

        print(f"Loaded {len(self.index)} samples from {len(self.parquet_files)} parquet files")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_info = self.index[idx]
        df = pd.read_parquet(
            file_info['parquet_file'],
            columns=['file_path', 'codes_scale_0', 'codes_scale_1', 'codes_scale_2']
        )
        row = df.iloc[file_info['row_idx']]

        def to_tensor(obj):
            import numpy as np
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

        # Load audio for SIDON
        audio_path = self.audio_dir / row['file_path']
        try:
            audio, sr = torchaudio.load(audio_path)
            if sr != 24000:
                audio = torchaudio.functional.resample(audio, sr, 24000)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            audio = audio[:, :96000].squeeze(0)
            return {'codes': codes, 'audio': audio}
        except:
            return None


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {'codes': None, 'audio': torch.empty(0)}

    audio_batch = torch.stack([item['audio'] for item in batch])
    audio_batch = audio_batch.unsqueeze(1)
    codes_batch = [item['codes'] for item in batch]
    return {'codes': codes_batch, 'audio': audio_batch}


def batch_sidon(sidon, audio_batch, sample_rate=24000):
    """Process audio through SIDON."""
    batch_size = audio_batch.shape[0]
    outputs = []
    for i in range(batch_size):
        audio_single = audio_batch[i].unsqueeze(0)
        _, audio_48k = sidon(audio_single, sample_rate=sample_rate)
        outputs.append(audio_48k.squeeze(0))
    return torch.stack(outputs, dim=0)


def reconstruction_loss(pred, target):
    l1_loss = F.l1_loss(pred, target)
    stft_loss = 0.0
    for n_fft in [1024, 2048, 4096]:
        pred_stft = torch.stft(pred.squeeze(1), n_fft=n_fft, return_complex=True)
        target_stft = torch.stft(target.squeeze(1), n_fft=n_fft, return_complex=True)
        stft_loss += F.l1_loss(pred_stft.abs(), target_stft.abs())
    stft_loss = stft_loss / 3
    loss = l1_loss + stft_loss
    return loss, l1_loss, stft_loss


# Load data
print("\nLoading dataset...")
dataset = CachedCodesDataset('/mnt/data/codes_phase11/train', max_files=3)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
print(f"Dataset size: {len(dataset)}")

# Load SIDON
print("\nLoading SIDON...")
sidon = SIDONUpsampler(device)
print("✓ SIDON loaded")

# Load pretrained SNAC (for decoder)
print("\nLoading pretrained SNAC...")
pretrained_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)

# Create 48kHz decoder
class Decoder48kHz(nn.Module):
    def __init__(self, pretrained_snac):
        super().__init__()
        # Use only blocks 0-5 (upsampling path, outputs 64 channels at 24kHz)
        # NOT blocks 6-8 (which produce the final 1-channel audio)
        self.pretrained_decoder_upsamp = nn.Sequential(*list(pretrained_snac.decoder.model.children())[:6])

        # New layers for 48kHz output
        self.snake1d = Snake1d(64)
        self.upsampler2x = DecoderBlock(64, 64, stride=2, noise=False, groups=1)
        self.final_conv = nn.Conv1d(64, 1, kernel_size=7, padding=3)

        # Random initialization for now
        nn.init.xavier_uniform_(self.upsampler2x.block[1].weight)
        nn.init.zeros_(self.upsampler2x.block[1].bias)
        nn.init.xavier_uniform_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(self, z_q):
        # Get 64-channel features at 24kHz from frozen decoder
        h = self.pretrained_decoder_upsamp(z_q)  # (B, 64, T_24k)
        # Upsample to 48kHz
        h = self.snake1d(h)
        h = self.upsampler2x(h)  # (B, 64, T_48k)
        # Final audio output
        audio = self.final_conv(h)  # (B, 1, T_48k)
        audio = torch.tanh(audio)
        return audio

model = Decoder48kHz(pretrained_model).to(device)

# Freeze pretrained decoder
for param in model.pretrained_decoder_upsamp.parameters():
    param.requires_grad = False

# Only train new layers
optimizer = AdamW([
    {'params': model.upsampler2x.parameters(), 'lr': 5e-5},
    {'params': model.final_conv.parameters(), 'lr': 5e-5}
])

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel params: {total_params:,} total, {trainable_params:,} trainable")

# Test training
print("\n" + "="*70)
print("TESTING: 5 Training Iterations")
print("="*70)

model.train()
for iteration, batch in enumerate(dataloader):
    if batch['codes'] is None or batch['audio'].numel() == 0:
        continue

    if iteration >= 5:
        break

    codes_batch = batch['codes']
    audio_24k = batch['audio'].to(device)

    # Generate 48kHz target with SIDON
    audio_48k_target = batch_sidon(sidon, audio_24k, sample_rate=24000)
    audio_48k_target = audio_48k_target.to(device)

    # Reorganize codes by scale for batch processing
    codes_batch_0 = torch.stack([c[0] for c in codes_batch]).to(device)
    codes_batch_1 = torch.stack([c[1] for c in codes_batch]).to(device)
    codes_batch_2 = torch.stack([c[2] for c in codes_batch]).to(device)

    with torch.no_grad():
        # Use SNAC's built-in method to convert hierarchical codes to latent z_q
        z_q = pretrained_model.quantizer.from_codes([codes_batch_0, codes_batch_1, codes_batch_2])

    # Pass z_q through custom 48kHz decoder
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

    print(f"Iteration {iteration+1}: Loss={loss.item():.4f}, L1={l1_loss.item():.4f}, STFT={stft_loss.item():.4f}")

print("\n" + "="*70)
print("✓ Training test complete!")
print("="*70)
print("\nConclusion: Cached codes training works! Ready for full training.")

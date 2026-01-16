"""
Quick test of cached codes training.
Uses first few cached batches to verify everything works.
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

os.environ['HF_HOME'] = '/mnt/data/work/snac/.hf_cache'
os.environ['HF_HUB_CACHE'] = '/mnt/data/work/snac/.hf_cache/hub'

from snac import SNAC
from snac.layers import Decoder, DecoderBlock, Snake1d
from finetune_decoder_48khz import SIDONUpsampler

device = torch.device('cuda:0')

print("="*70)
print("QUICK TEST: Cached Codes Training")
print("="*70)

# Test dataset - only use first 5 parquet files
class TestCachedCodesDataset(Dataset):
    def __init__(self, cache_dir, max_files=5):
        self.cache_dir = Path(cache_dir)
        self.parquet_files = sorted(self.cache_dir.glob("codes_batch_gpu*.parquet"))[:max_files]
        print(f"Loading {len(self.parquet_files)} parquet files...")

        self.index = []
        for file_idx, parquet_file in enumerate(self.parquet_files):
            df = pd.read_parquet(parquet_file, columns=['file_path'])
            for row_idx in range(len(df)):
                self.index.append({'parquet_file': parquet_file, 'row_idx': row_idx})

        print(f"Total samples: {len(self.index)}")

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
            if isinstance(obj, np.ndarray):
                if obj.dtype == object:
                    return torch.tensor(obj.tolist(), dtype=torch.long)
                else:
                    return torch.from_numpy(obj).long()
            elif isinstance(obj, list):
                return torch.tensor(obj, dtype=torch.long)
            else:
                return torch.tensor([obj], dtype=torch.long)

        import numpy as np
        codes = [
            to_tensor(row['codes_scale_0']),
            to_tensor(row['codes_scale_1']),
            to_tensor(row['codes_scale_2'])
        ]

        return {'codes': codes, 'file_path': row['file_path']}


def batch_sidon(sidon, audio_batch, sample_rate=24000):
    """Process audio through SIDON one by one."""
    batch_size = audio_batch.shape[0]
    outputs = []

    for i in range(batch_size):
        audio_single = audio_batch[i].unsqueeze(0)
        _, audio_48k = sidon(audio_single, sample_rate=sample_rate)
        outputs.append(audio_48k.squeeze(0))

    return torch.stack(outputs, dim=0)


def codes_to_latent(codes, vq_strides, device):
    """
    Convert hierarchical codes to latent tensor using SNAC's quantizer.
    This is the CORRECT way - use VQ encoder directly instead of manual reconstruction.
    """
    codes_s0, codes_s1, codes_s2 = codes

    # For each scale, embed codes using VQ codebook
    # Then combine using the proper residual structure

    # Actually, simpler approach: use the model's decode method directly
    # But we need to convert codes to the right format first

    # The SNAC model expects codes as [batch, num_levels, time]
    # We need to properly interleave them

    # Let's use the quantizer's embedding directly
    pass  # We'll test a different approach below


# Load data
print("\nLoading test dataset...")
test_dataset = TestCachedCodesDataset('/mnt/data/codes_phase11/train', max_files=3)
print(f"Dataset size: {len(test_dataset)}")

# Test loading one sample
print("\nTesting sample loading...")
sample = test_dataset[0]
print(f"Sample codes shapes:")
for i, codes in enumerate(sample['codes']):
    print(f"  Scale {i}: {codes.shape}")

# Try loading audio and running through full pipeline
print("\n" + "="*70)
print("Testing full pipeline with one sample...")
print("="*70)

import torchaudio

# Load original audio for this sample
audio_path = Path("/mnt/data/combine/train/audio") / sample['file_path']
audio, sr = torchaudio.load(audio_path)
if sr != 24000:
    audio = torchaudio.functional.resample(audio, sr, 24000)
if audio.shape[0] > 1:
    audio = torch.mean(audio, dim=0, keepdim=True)
audio = audio[:, :96000].to(device)  # 4 seconds

print(f"Audio shape: {audio.shape}")

# Load model and run through encoder+VQ to get expected codes
print("\nLoading SNAC model...")
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
model.eval()

with torch.no_grad():
    z = model.encoder(audio)
    print(f"Encoder output shape: {z.shape}")

    _, expected_codes = model.quantizer(z)
    print(f"Expected codes shapes:")
    for i, code in enumerate(expected_codes):
        print(f"  Scale {i}: {code.shape}")

# Compare with cached codes
print(f"\nCached codes shapes:")
for i, code in enumerate(sample['codes']):
    print(f"  Scale {i}: {code.shape}")

# Test decode
print("\nTesting decode from cached codes...")
try:
    # Use model's decode method
    with torch.no_grad():
        # Stack codes into proper format
        codes_batch = [c.unsqueeze(0).to(device) for c in sample['codes']]
        audio_recon = model.decode(codes_batch)
        print(f"Reconstructed audio shape: {audio_recon.shape}")
        print("✓ Decode successful!")

except Exception as e:
    print(f"✗ Decode failed: {e}")

print("\n" + "="*70)
print("Test complete!")
print("="*70)

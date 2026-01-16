"""
Simple test: Verify cached codes can decode correctly.
"""
import torch
import pandas as pd
from pathlib import Path

device = torch.device('cuda:0')

print("="*70)
print("TEST: Decode from Cached Codes")
print("="*70)

# Load one cached sample
cache_dir = Path('/mnt/data/codes_phase11/train')
parquet_file = sorted(cache_dir.glob("codes_batch_gpu*.parquet"))[0]

df = pd.read_parquet(parquet_file, columns=['file_path', 'codes_scale_0', 'codes_scale_1', 'codes_scale_2'])
row = df.iloc[0]

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
    to_tensor(row['codes_scale_0']).unsqueeze(0).to(device),
    to_tensor(row['codes_scale_1']).unsqueeze(0).to(device),
    to_tensor(row['codes_scale_2']).unsqueeze(0).to(device),
]

print(f"\nLoaded cached codes:")
for i, c in enumerate(codes):
    print(f"  Scale {i}: {c.shape}")

# Load SNAC model
print("\nLoading SNAC model...")
import os
os.environ['HF_HOME'] = '/mnt/data/work/snac/.hf_cache'
os.environ['HF_HUB_CACHE'] = '/mnt/data/work/snac/.hf_cache/hub'

from snac import SNAC
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
model.eval()

print(f"Model loaded")

# Test decode
print("\nTesting decode...")
with torch.no_grad():
    try:
        audio_recon = model.decode(codes)
        print(f"✓ Decode successful!")
        print(f"  Reconstructed audio shape: {audio_recon.shape}")
        print(f"  Audio length: {audio_recon.shape[-1]} samples ({audio_recon.shape[-1]/24000:.2f} seconds)")
        print(f"  Audio range: [{audio_recon.min():.3f}, {audio_recon.max():.3f}]")
    except Exception as e:
        print(f"✗ Decode failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("Test complete!")
print("="*70)

# Test with a batch of samples
print("\n" + "="*70)
print("TEST: Batch Decode from Cached Codes")
print("="*70)

# Load first 10 samples
batch_codes = []
for idx in range(min(10, len(df))):
    row = df.iloc[idx]
    codes = [
        to_tensor(row['codes_scale_0']),
        to_tensor(row['codes_scale_1']),
        to_tensor(row['codes_scale_2']),
    ]
    batch_codes.append(codes)

print(f"\nLoaded {len(batch_codes)} samples")

# Reorganize into batched format by scale
batch_codes_0 = torch.stack([c[0] for c in batch_codes]).to(device)
batch_codes_1 = torch.stack([c[1] for c in batch_codes]).to(device)
batch_codes_2 = torch.stack([c[2] for c in batch_codes]).to(device)

print(f"Batch shapes:")
print(f"  Scale 0: {batch_codes_0.shape}")
print(f"  Scale 1: {batch_codes_1.shape}")
print(f"  Scale 2: {batch_codes_2.shape}")

# Decode batch
print("\nTesting batch decode...")
with torch.no_grad():
    try:
        audio_batch = model.decode([batch_codes_0, batch_codes_1, batch_codes_2])
        print(f"✓ Batch decode successful!")
        print(f"  Batch shape: {audio_batch.shape}")
        print(f"  Sample 0 length: {audio_batch.shape[-1]} samples ({audio_batch.shape[-1]/24000:.2f} seconds)")
    except Exception as e:
        print(f"✗ Batch decode failed: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("All tests complete!")
print("="*70)

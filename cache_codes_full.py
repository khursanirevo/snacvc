"""
Cache encoder+VQ codes for full dataset (2.8M files).
Skips expensive encoder+VQ forward pass during training.
"""
import os
import json
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import pandas as pd

os.environ['HF_HOME'] = '/mnt/data/work/snac/.hf_cache'
os.environ['HF_HUB_CACHE'] = '/mnt/data/work/snac/.hf_cache/hub'

from snac import SNAC

device = torch.device('cuda:0')

print("Loading SNAC model...")
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
for param in model.encoder.parameters():
    param.requires_grad = False
for param in model.quantizer.parameters():
    param.requires_grad = False

print("\n" + "="*70)
print("CACHING CODES FOR FULL DATASET")
print("="*70)

# Cache training set
print("\nProcessing training set...")
train_dir = Path("/mnt/data/combine/train/audio")
cache_dir = Path("/mnt/data/codes_phase11/train")
cache_dir.mkdir(parents=True, exist_ok=True)

# Get all audio files
audio_files = list(train_dir.glob("*.wav")) + list(train_dir.glob("*.mp3"))
print(f"Found {len(audio_files)} audio files")

all_data = []
batch_size = 1000

for idx, audio_path in enumerate(tqdm(audio_files, desc="Caching train")):
    try:
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        if sr != 24000:
            audio = torchaudio.functional.resample(audio, sr, 24000)

        # Convert to mono
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        # Trim to 4 seconds
        segment_samples = 96000  # 4s @ 24kHz
        if audio.shape[-1] < segment_samples:
            continue  # Skip short files

        audio = audio[:, :segment_samples].to(device)

        # Encode
        with torch.no_grad():
            z = model.encoder(audio)
            _, codes = model.quantizer(z)

        all_data.append({
            'file_path': audio_path.name,
            'codes_scale_0': codes[0].cpu().numpy().tolist(),
            'codes_scale_1': codes[1].cpu().numpy().tolist(),
            'codes_scale_2': codes[2].cpu().numpy().tolist(),
            'shape_scale_0': (codes[0].shape[0],),
            'shape_scale_1': (codes[1].shape[0],),
            'shape_scale_2': (codes[2].shape[0],),
        })

        # Save batch
        if len(all_data) >= batch_size:
            df = pd.DataFrame(all_data)
            batch_num = len(list(cache_dir.glob("*.parquet")))
            df.to_parquet(cache_dir / f"codes_batch_{batch_num}.parquet", index=False)
            print(f"✓ Saved batch {batch_num} ({len(all_data)} files)")
            all_data = []

    except Exception as e:
        continue

# Save remaining
if all_data:
    df = pd.DataFrame(all_data)
    batch_num = len(list(cache_dir.glob("*.parquet")))
    df.to_parquet(cache_dir / f"codes_batch_{batch_num}.parquet", index=False)
    print(f"✓ Saved final batch {batch_num} ({len(all_data)} files)")

# Metadata
metadata = {
    'total_samples': len(list(cache_dir.glob("*.parquet"))),
    'num_codebooks': 3,
    'codebook_size': 4096,
    'vq_strides': [4, 2, 1],
    'hop_length': 512,
}
with open(cache_dir / "metadata.json", 'w') as f:
    json.dump(metadata, f)

print(f"\n✅ Done! Cached codes to {cache_dir}")
print(f"Total batches: {len(list(cache_dir.glob('*.parquet')))}")

# Cache validation set
print("\nProcessing validation set...")
val_dir = Path("/mnt/data/combine/valid/audio")
cache_dir_val = Path("/mnt/data/codes_phase11/val")
cache_dir_val.mkdir(parents=True, exist_ok=True)

audio_files_val = list(val_dir.glob("*.wav")) + list(val_dir.glob("*.mp3"))
print(f"Found {len(audio_files_val)} audio files")

all_data = []

for idx, audio_path in enumerate(tqdm(audio_files_val, desc="Caching val")):
    try:
        audio, sr = torchaudio.load(audio_path)
        if sr != 24000:
            audio = torchaudio.functional.resample(audio, sr, 24000)

        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        segment_samples = 96000
        if audio.shape[-1] < segment_samples:
            continue

        audio = audio[:, :segment_samples].to(device)

        with torch.no_grad():
            z = model.encoder(audio)
            _, codes = model.quantizer(z)

        all_data.append({
            'file_path': audio_path.name,
            'codes_scale_0': codes[0].cpu().numpy().tolist(),
            'codes_scale_1': codes[1].cpu().numpy().tolist(),
            'codes_scale_2': codes[2].cpu().numpy().tolist(),
            'shape_scale_0': (codes[0].shape[0],),
            'shape_scale_1': (codes[1].shape[0],),
            'shape_scale_2': (codes[2].shape[0],),
        })

        if len(all_data) >= batch_size:
            df = pd.DataFrame(all_data)
            batch_num = len(list(cache_dir_val.glob("*.parquet")))
            df.to_parquet(cache_dir_val / f"codes_batch_{batch_num}.parquet", index=False)
            print(f"✓ Saved batch {batch_num} ({len(all_data)} files)")
            all_data = []

    except Exception as e:
        continue

if all_data:
    df = pd.DataFrame(all_data)
    batch_num = len(list(cache_dir_val.glob("*.parquet")))
    df.to_parquet(cache_dir_val / f"codes_batch_{batch_num}.parquet", index=False)
    print(f"✓ Saved final batch {batch_num} ({len(all_data)} files)")

metadata = {
    'total_samples': len(list(cache_dir_val.glob("*.parquet"))),
    'num_codebooks': 3,
    'codebook_size': 4096,
    'vq_strides': [4, 2, 1],
    'hop_length': 512,
}
with open(cache_dir_val / "metadata.json", 'w') as f:
    json.dump(metadata, f)

print(f"\n✅ Done! Cached codes to {cache_dir_val}")
print(f"Total batches: {len(list(cache_dir_val.glob('*.parquet')))}")

print("\n" + "="*70)
print("CACHING COMPLETE")
print("="*70)

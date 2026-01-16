"""
Cache encoder+VQ codes for full dataset (2.8M files) - BATCHED VERSION.
Processes multiple audio files per GPU batch for 10-20x speedup.
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
model.eval()
for param in model.encoder.parameters():
    param.requires_grad = False
for param in model.quantizer.parameters():
    param.requires_grad = False

print("\n" + "="*70)
print("CACHING CODES FOR FULL DATASET (BATCHED)")
print("="*70)

def load_audio_batch(file_paths, target_sr=24000, target_samples=96000):
    """
    Load multiple audio files into a single batch.

    Returns:
        audio_batch: (B, 1, T) tensor
        valid_paths: list of successfully loaded paths
    """
    audio_list = []
    valid_paths = []

    for path in file_paths:
        try:
            audio, sr = torchaudio.load(path)

            # Resample if needed
            if sr != target_sr:
                audio = torchaudio.functional.resample(audio, sr, target_sr)

            # Convert to mono
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # Skip if too short
            if audio.shape[-1] < target_samples:
                continue

            # Trim to exact length
            audio = audio[:, :target_samples]
            audio_list.append(audio)
            valid_paths.append(path)

        except Exception as e:
            continue

    if len(audio_list) == 0:
        return None, []

    # Stack into batch
    return torch.stack(audio_list, dim=0), valid_paths


def process_split(split_dir, cache_dir, gpu_batch_size=32, save_batch_size=1000):
    """
    Process a dataset split with batching.

    Args:
        split_dir: Path to audio directory
        cache_dir: Path to cache directory
        gpu_batch_size: Number of files to process per GPU batch
        save_batch_size: Number of files to save per parquet file
    """
    split_dir = Path(split_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Get all audio files
    audio_files = list(split_dir.glob("*.wav")) + list(split_dir.glob("*.mp3"))
    print(f"Found {len(audio_files)} audio files")

    all_data = []
    total_processed = 0

    # Process in GPU batches
    for start_idx in tqdm(range(0, len(audio_files), gpu_batch_size), desc=f"Caching {cache_dir.name}"):
        batch_files = audio_files[start_idx:start_idx + gpu_batch_size]

        # Load batch
        audio_batch, valid_paths = load_audio_batch(batch_files)

        if audio_batch is None:
            continue

        # Move to GPU and encode
        audio_batch = audio_batch.to(device)

        with torch.no_grad():
            z = model.encoder(audio_batch)
            _, codes = model.quantizer(z)

        # Store results
        for i, path in enumerate(valid_paths):
            all_data.append({
                'file_path': path.name,
                'codes_scale_0': codes[0][i].cpu().numpy().tolist(),
                'codes_scale_1': codes[1][i].cpu().numpy().tolist(),
                'codes_scale_2': codes[2][i].cpu().numpy().tolist(),
                'shape_scale_0': (codes[0][i].shape[0],),
                'shape_scale_1': (codes[1][i].shape[0],),
                'shape_scale_2': (codes[2][i].shape[0],),
            })

        total_processed += len(valid_paths)

        # Save to disk periodically
        if len(all_data) >= save_batch_size:
            df = pd.DataFrame(all_data)
            batch_num = len(list(cache_dir.glob("*.parquet")))
            df.to_parquet(cache_dir / f"codes_batch_{batch_num}.parquet", index=False)
            print(f"✓ Saved batch {batch_num} ({len(all_data)} files, {total_processed} total)")
            all_data = []

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

    print(f"\n✅ Done! Cached {total_processed} files to {cache_dir}")
    print(f"Total batches: {len(list(cache_dir.glob('*.parquet')))}")


# Process training set
print("\nProcessing training set...")
process_split(
    split_dir="/mnt/data/combine/train/audio",
    cache_dir="/mnt/data/codes_phase11/train",
    gpu_batch_size=32,  # Process 32 files at once on GPU
    save_batch_size=1000  # Save every 1000 files
)

# Process validation set
print("\n" + "="*70)
print("\nProcessing validation set...")
process_split(
    split_dir="/mnt/data/combine/valid/audio",
    cache_dir="/mnt/data/codes_phase11/val",
    gpu_batch_size=32,
    save_batch_size=1000
)

print("\n" + "="*70)
print("CACHING COMPLETE")
print("="*70)

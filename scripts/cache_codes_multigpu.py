"""
Cache FULL encoder+VQ codes using ALL 4 GPUs in parallel.
Each GPU processes 1/4 of the dataset for ~4x speedup.

IMPORTANT: Stores FULL audio codes (not pre-sliced) for random segment slicing during training.
"""
import os
import sys
import json
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
import pandas as pd

os.environ['HF_HOME'] = '/mnt/data/work/snac/.hf_cache'
os.environ['HF_HUB_CACHE'] = '/mnt/data/work/snac/.hf_cache/hub'

from snac import SNAC

def load_audio_batch(file_paths, target_sr=24000, min_samples=24000):
    """Load multiple audio files into a single batch.

    NOTE: Loads FULL audio (no slicing) for random segment slicing during training.
    """
    audio_list = []
    valid_paths = []

    for path in file_paths:
        try:
            audio, sr = torchaudio.load(path)

            if sr != target_sr:
                audio = torchaudio.functional.resample(audio, sr, target_sr)

            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # Only filter out very short files (< 1 second)
            if audio.shape[-1] < min_samples:
                continue

            # Keep FULL audio (no slicing!)
            audio_list.append(audio)
            valid_paths.append(path)

        except Exception as e:
            continue

    if len(audio_list) == 0:
        return None, []

    return torch.stack(audio_list, dim=0), valid_paths


def process_split(split_dir, cache_dir, gpu_id, gpu_batch_size=1, save_batch_size=1000):
    """
    Process a dataset split on a specific GPU.
    Stores FULL audio codes (no slicing) for random segment slicing during training.

    Args:
        split_dir: Path to audio directory
        cache_dir: Path to cache directory
        gpu_id: GPU device ID
        gpu_batch_size: NOT USED (process individually due to variable lengths)
        save_batch_size: Number of files to save per parquet file
    """
    device = torch.device(f'cuda:{gpu_id}')

    print(f"\n[GPU {gpu_id}] Loading SNAC model...")
    model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
    model.eval()
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.quantizer.parameters():
        param.requires_grad = False

    split_dir = Path(split_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Get all audio files
    audio_files = list(split_dir.glob("*.wav")) + list(split_dir.glob("*.mp3"))
    print(f"[GPU {gpu_id}] Found {len(audio_files)} audio files")

    # Shard files based on GPU ID (round-robin assignment)
    # GPU 0: files[0::4], GPU 1: files[1::4], etc.
    sharded_files = audio_files[gpu_id::4]
    print(f"[GPU {gpu_id}] Processing {len(sharded_files)} files (shard {gpu_id}/4)")

    all_data = []
    total_processed = 0

    # Process files individually (variable lengths, can't batch)
    for audio_path in tqdm(sharded_files, desc=f"GPU {gpu_id}"):
        try:
            # Load single audio file
            audio, sr = torchaudio.load(audio_path)

            if sr != 24000:
                audio = torchaudio.functional.resample(audio, sr, 24000)

            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            # Skip very short files
            if audio.shape[-1] < 24000:
                continue

            # Encode to codes
            audio = audio.to(device)
            with torch.no_grad():
                z = model.encoder(audio.unsqueeze(0))
                _, codes = model.quantizer(z)

            # Store results
            all_data.append({
                'file_path': audio_path.name,
                'codes_scale_0': codes[0][0].cpu().numpy().tolist(),
                'codes_scale_1': codes[1][0].cpu().numpy().tolist(),
                'codes_scale_2': codes[2][0].cpu().numpy().tolist(),
                'audio_length': audio.shape[-1],
                'shape_scale_0': (codes[0][0].shape[0],),
                'shape_scale_1': (codes[1][0].shape[0],),
                'shape_scale_2': (codes[2][0].shape[0],),
            })

            total_processed += 1

            # Save to disk periodically
            if len(all_data) >= save_batch_size:
                df = pd.DataFrame(all_data)
                batch_num = len(list(cache_dir.glob(f"codes_batch_gpu{gpu_id}_*.parquet")))
                df.to_parquet(cache_dir / f"codes_batch_gpu{gpu_id}_{batch_num}.parquet", index=False)
                print(f"[GPU {gpu_id}] ✓ Saved batch {batch_num} ({len(all_data)} files, {total_processed} total)")
                all_data = []

        except Exception as e:
            continue

    # Save remaining
    if all_data:
        df = pd.DataFrame(all_data)
        batch_num = len(list(cache_dir.glob(f"codes_batch_gpu{gpu_id}_*.parquet")))
        df.to_parquet(cache_dir / f"codes_batch_gpu{gpu_id}_{batch_num}.parquet", index=False)
        print(f"[GPU {gpu_id}] ✓ Saved final batch {batch_num} ({len(all_data)} files)")

    print(f"\n[GPU {gpu_id}] ✅ Done! Cached {total_processed} files to {cache_dir}")

    return total_processed


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True, help='GPU ID (0-3)')
    parser.add_argument('--split', type=str, required=True, help='Dataset split (train/val)')
    args = parser.parse_args()

    gpu_id = args.gpu
    split = args.split

    print(f"\n{'='*70}")
    print(f"MULTI-GPU CODE CACHING - GPU {gpu_id}")
    print(f"{'='*70}")

    if split == 'train':
        split_dir = "/mnt/data/combine/train/audio"
        cache_dir = "/mnt/data/codes_phase11/train"
    else:
        split_dir = "/mnt/data/combine/valid/audio"
        cache_dir = "/mnt/data/codes_phase11/val"

    total = process_split(
        split_dir=split_dir,
        cache_dir=cache_dir,
        gpu_id=gpu_id,
        gpu_batch_size=32,
        save_batch_size=1000
    )

    # Last GPU writes metadata
    if gpu_id == 3:
        # Wait a bit for other GPUs to finish
        import time
        time.sleep(5)

        # Count total files
        if split == 'train':
            cache_dir_train = Path("/mnt/data/codes_phase11/train")
            total_batches = len(list(cache_dir_train.glob("*.parquet")))
        else:
            cache_dir_val = Path("/mnt/data/codes_phase11/val")
            total_batches = len(list(cache_dir_val.glob("*.parquet")))

        metadata = {
            'total_samples': total_batches,
            'num_codebooks': 3,
            'codebook_size': 4096,
            'vq_strides': [8, 4, 2, 1],  # Hierarchical strides
            'full_codes': True,  # IMPORTANT: Storing full audio codes, not pre-sliced
            'random_slicing': True,  # For random segment slicing during training
            'multi_gpu': True,
        }

        cache_path = Path(cache_dir)
        with open(cache_path / "metadata.json", 'w') as f:
            json.dump(metadata, f)

        print(f"\n[GPU {gpu_id}] Wrote metadata to {cache_path / 'metadata.json'}")

    print(f"\n[GPU {gpu_id}] {'='*70}")
    print(f"[GPU {gpu_id}] COMPLETE - Processed {total} files")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()

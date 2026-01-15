"""
Pre-compute SNAC latent codes for Phase 11 training

Since encoder and VQ are frozen, we can pre-compute all quantized codes once
and reuse them for every epoch. This saves:
- Time: No encoder/VQ forward pass during training
- Memory: Don't need to load encoder/VQ
- Disk: Smaller checkpoints (decoder only)

Output format: Parquet files with quantized codes
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

from snac import SNAC
from snac.dataset import OptimizedAudioDataset


def compute_codes(model: SNAC, audio: torch.Tensor, device: str) -> List[torch.Tensor]:
    """
    Compute quantized codes from audio using frozen encoder+VQ.

    Args:
        model: SNAC model
        audio: (B, 1, T) audio tensor at 24kHz
        device: Device to run on

    Returns:
        List of 4 code tensors at different scales
    """
    audio = audio.to(device)

    # Preprocess
    length = audio.shape[-1]
    audio_padded = model.preprocess(audio)

    # Encode
    with torch.no_grad():
        z = model.encoder(audio_padded)
        _, codes = model.quantizer(z)

    # Trim codes to original length
    # Each scale has different temporal resolution
    # Scale 0: stride 8, Scale 1: stride 4, Scale 2: stride 2, Scale 3: stride 1
    hop_length = model.hop_length
    expected_length = (length + hop_length - 1) // hop_length

    trimmed_codes = []
    for i, code in enumerate(codes):
        stride = model.vq_strides[i]
        expected_len = (length + stride * hop_length - 1) // (stride * hop_length)
        trimmed_codes.append(code[..., :expected_len].cpu())

    return trimmed_codes


def collate_filter_none(batch):
    """Custom collate function that filters out None values (short files)."""
    # Filter out None values (short files)
    batch = [item for item in batch if item is not None]

    if not batch:
        return None  # Empty batch

    # Extract audio tensors from dicts and ensure correct shape
    audio_tensors = []
    for item in batch:
        if isinstance(item, dict):
            audio = item['audio']  # Dict with 'audio' key
        else:
            audio = item  # Raw tensor

        # Ensure shape is (1, T) - mono audio
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # (T,) -> (1, T)
        elif audio.dim() == 2 and audio.shape[0] > 1:
            # Convert stereo to mono by averaging
            audio = audio.mean(dim=0, keepdim=True)

        audio_tensors.append(audio)

    # Stack tensors: (B, 1, T)
    return torch.stack(audio_tensors, dim=0)


def process_dataset(
    model: SNAC,
    dataset: OptimizedAudioDataset,
    output_dir: Path,
    batch_size: int,
    device: str,
):
    """
    Process entire dataset and save quantized codes.

    Args:
        model: SNAC model (encoder+VQ frozen)
        dataset: Audio dataset
        output_dir: Directory to save codes
        batch_size: Batch size for processing
        device: Device to run on
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_filter_none,
    )

    # Process batches
    all_data = []
    sample_count = 0

    for batch_idx, audio_batch in enumerate(tqdm(dataloader, desc="Computing codes")):
        if audio_batch is None:
            continue  # Skip empty batches

        try:
            # Compute codes
            codes = compute_codes(model, audio_batch, device)

            # Store for each sample in batch
            batch_size_actual = audio_batch.shape[0]
            for i in range(batch_size_actual):
                current_idx = sample_count  # Track before incrementing
                sample_codes = [code[i].numpy() for code in codes]

                # Validate we have all 3 scales (SNAC has 3 codebooks)
                if len(sample_codes) != 3:
                    print(f"Warning: Sample {current_idx} ({dataset.samples[current_idx].name}) has {len(sample_codes)} scales, expected 3, skipping...")
                    sample_count += 1
                    continue

                all_data.append({
                    'file_path': dataset.samples[current_idx].as_posix(),
                    'codes_scale_0': sample_codes[0],  # Coarsest
                    'codes_scale_1': sample_codes[1],
                    'codes_scale_2': sample_codes[2],  # Finest
                    'shape_scale_0': sample_codes[0].shape,
                    'shape_scale_1': sample_codes[1].shape,
                    'shape_scale_2': sample_codes[2].shape,
                })
                sample_count += 1

            # Save batch to disk periodically
            if len(all_data) >= 1000:
                df = pd.DataFrame(all_data)

                # Convert arrays to lists for Parquet serialization
                for scale in range(3):  # SNAC has 3 codebooks
                    df[f'codes_scale_{scale}'] = df[f'codes_scale_{scale}'].apply(lambda x: x.tolist())
                    df[f'shape_scale_{scale}'] = df[f'shape_scale_{scale}'].apply(lambda x: list(x))

                batch_file = output_dir / f"codes_batch_{len(all_data)//1000:04d}.parquet"
                df.to_parquet(batch_file, index=False)
                all_data = []

        except Exception as e:
            print(f"Warning: Error processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save remaining
    if all_data:
        df = pd.DataFrame(all_data)
        for scale in range(3):  # SNAC has 3 codebooks
            df[f'codes_scale_{scale}'] = df[f'codes_scale_{scale}'].apply(lambda x: x.tolist())
            df[f'shape_scale_{scale}'] = df[f'shape_scale_{scale}'].apply(lambda x: list(x))

        batch_file = output_dir / f"codes_batch_{(len(all_data)//1000)+1:04d}.parquet"
        df.to_parquet(batch_file, index=False)

    # Save metadata
    metadata = {
        'total_samples': int(sample_count),
        'num_codebooks': int(len(model.vq_strides)),
        'codebook_size': int(model.codebook_size),
        'vq_strides': [int(x) for x in model.vq_strides],
        'latent_dim': int(model.latent_dim),
        'hop_length': int(model.hop_length),
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Saved codes to: {output_dir}")
    print(f"   Total samples processed: {sample_count}")
    print(f"   Metadata: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute SNAC codes for Phase 11")
    parser.add_argument("--pretrained_model", type=str, default="hubertsiuzdak/snac_24khz")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--segment_length", type=float, default=4.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit to N samples for testing (default: all)")
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Start file index (for parallel processing)")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End file index (for parallel processing)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load SNAC model
    print(f"\nLoading SNAC model: {args.pretrained_model}")
    model = SNAC.from_pretrained(args.pretrained_model).to(device)
    model.eval()

    print(f"  Encoder rates: {model.encoder_rates}")
    print(f"  Decoder rates: {model.decoder_rates}")
    print(f"  VQ strides: {model.vq_strides}")
    print(f"  Hop length: {model.hop_length}")
    print(f"  Latent dim: {model.latent_dim}")

    # Create dataset
    print(f"\nLoading dataset from: {args.data_dir}")
    dataset = OptimizedAudioDataset(
        args.data_dir,
        sampling_rate=24000,
        segment_length=args.segment_length,
        augment=False,
    )

    # Apply file range for parallel processing
    start_offset = args.start_idx if args.start_idx is not None else 0
    end_offset = args.end_idx if args.end_idx is not None else len(dataset)

    # Limit samples for testing
    if args.max_samples:
        print(f"  Limiting to {args.max_samples} samples for testing")
        dataset.samples = dataset.samples[:args.max_samples]
        start_offset = 0
        end_offset = len(dataset)
    else:
        dataset.samples = dataset.samples[start_offset:end_offset]

    print(f"  Processing files {start_offset} to {end_offset} ({len(dataset)} files)")

    # Process dataset
    print(f"\nComputing codes...")
    output_dir = Path(args.output_dir)
    process_dataset(model, dataset, output_dir, args.batch_size, str(device))


if __name__ == "__main__":
    main()

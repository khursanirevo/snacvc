"""
Preprocessing script to split audio files into fixed-length segments.

This converts the dataset from:
  - 79K audio files, each providing 1 random segment per epoch
To:
  - ~11M segments, all utilized every epoch

Usage:
    python split_audio.py --input data/train --output data/train_split --segment-length 4.0
    python split_audio.py --input data/val --output data/val_split --segment-length 4.0
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torchaudio
import torch


def split_audio_file(
    audio_path,
    output_dir,
    segment_length=4.0,
    sampling_rate=24000,
    min_length=1.0,
):
    """
    Split a single audio file into fixed-length segments.

    Args:
        audio_path: Path to input audio file
        output_dir: Directory to save segments
        segment_length: Length of each segment in seconds
        sampling_rate: Target sampling rate
        min_length: Minimum segment length in seconds (skip if shorter)

    Returns:
        Number of segments created
    """
    try:
        # Load audio
        audio, sr = torchaudio.load(str(audio_path))

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, sampling_rate)
            audio = resampler(audio)

        audio = audio.squeeze(0)  # (T,)
        duration = audio.shape[-1] / sampling_rate

        # Skip if too short
        if duration < min_length:
            return 0

        # Calculate segment length in samples
        segment_samples = int(segment_length * sampling_rate)

        # Create output filename base
        audio_name = Path(audio_path).stem
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Split into segments
        num_segments = 0
        for start_idx in range(0, audio.shape[-1], segment_samples):
            end_idx = start_idx + segment_samples

            # Get segment (pad last segment if needed)
            if end_idx <= audio.shape[-1]:
                segment = audio[start_idx:end_idx]
            else:
                # Last segment - pad with zeros
                segment = torch.zeros(segment_samples, dtype=audio.dtype)
                segment[:audio.shape[-1] - start_idx] = audio[start_idx:]

            # Save segment
            segment_name = f"{audio_name}_seg{num_segments:04d}.wav"
            segment_path = output_dir / segment_name
            torchaudio.save(
                str(segment_path),
                segment.unsqueeze(0),
                sampling_rate,
            )

            num_segments += 1

        return num_segments

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return 0


def main(
    input_dir,
    output_dir,
    segment_length=4.0,
    sampling_rate=24000,
    min_length=1.0,
    num_workers=4,
):
    """
    Split all audio files in input directory into fixed-length segments.

    Args:
        input_dir: Directory containing audio files
        output_dir: Directory to save segments
        segment_length: Length of each segment in seconds
        sampling_rate: Target sampling rate
        min_length: Minimum segment length in seconds
        num_workers: Number of parallel workers
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all audio files
    audio_extensions = ['.wav', '.WAV', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(list(input_dir.glob(f'*{ext}')))

    print(f"Found {len(audio_files)} audio files in {input_dir}")

    if len(audio_files) == 0:
        print(f"No audio files found in {input_dir}")
        return

    # Split all files
    total_segments = 0
    skipped = 0

    for audio_path in tqdm(audio_files, desc="Splitting audio"):
        num_segments = split_audio_file(
            audio_path,
            output_dir,
            segment_length=segment_length,
            sampling_rate=sampling_rate,
            min_length=min_length,
        )

        if num_segments > 0:
            total_segments += num_segments
        else:
            skipped += 1

    print(f"\n=== Splitting Complete ===")
    print(f"Input files: {len(audio_files)}")
    print(f"Output segments: {total_segments}")
    print(f"Skipped files: {skipped}")
    print(f"Output directory: {output_dir}")

    # Save statistics
    stats = {
        'input_files': len(audio_files),
        'output_segments': total_segments,
        'skipped_files': skipped,
        'segment_length': segment_length,
        'sampling_rate': sampling_rate,
    }

    stats_path = output_dir / 'splitting_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"Statistics saved to: {stats_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Split audio files into fixed-length segments'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing audio files'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for segments'
    )
    parser.add_argument(
        '--segment-length',
        type=float,
        default=4.0,
        help='Segment length in seconds (default: 4.0)'
    )
    parser.add_argument(
        '--sampling-rate',
        type=int,
        default=24000,
        help='Target sampling rate (default: 24000)'
    )
    parser.add_argument(
        '--min-length',
        type=float,
        default=1.0,
        help='Minimum audio length in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )

    args = parser.parse_args()

    main(
        input_dir=args.input,
        output_dir=args.output,
        segment_length=args.segment_length,
        sampling_rate=args.sampling_rate,
        min_length=args.min_length,
        num_workers=args.num_workers,
    )

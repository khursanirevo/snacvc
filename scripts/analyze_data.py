#!/usr/bin/env python3
"""
Analyze audio dataset statistics before training.

Shows:
- Number of files per folder
- Total duration
- Audio format distribution
- Sample rate distribution
"""

import os
from pathlib import Path
import torch
import torchaudio
from collections import defaultdict
import tqdm

SOURCE_DIR = Path("/mnt/data/processed")

def analyze_audio_file(audio_path):
    """Analyze a single audio file."""
    try:
        metadata = torchaudio.info(str(audio_path))
        return {
            'sample_rate': metadata.sample_rate,
            'num_frames': metadata.num_frames,
            'duration': metadata.num_frames / metadata.sample_rate,
            'channels': metadata.num_channels,
        }
    except Exception as e:
        return None

def analyze_dataset():
    """Analyze all audio files in the dataset."""

    print("Scanning for audio files...")
    audio_extensions = ['.wav', '.WAV', '.mp3', '.flac', '.ogg', '.m4a']

    # Get all subdirectories (except audio/text metadata folders)
    folders = [d for d in SOURCE_DIR.iterdir()
               if d.is_dir() and d.name not in ['audio', 'text', 'meta', 'metadata']]

    print(f"\nFound {len(folders)} folders to analyze\n")

    # Statistics per folder
    folder_stats = {}
    total_files = 0
    total_duration = 0

    sample_rates = defaultdict(int)
    channels = defaultdict(int)
    extensions = defaultdict(int)

    for folder in sorted(folders):
        print(f"Analyzing: {folder.name}")

        # Find all audio files in this folder
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(folder.glob(f'**/*{ext}'))
            audio_files.extend(folder.glob(f'**/*{ext.upper()}'))

        # Remove duplicates
        seen = set()
        unique_files = []
        for f in audio_files:
            if f.name not in seen:
                seen.add(f.name)
                unique_files.append(f)

        audio_files = unique_files

        if len(audio_files) == 0:
            print(f"  No audio files found\n")
            continue

        # Analyze files
        durations = []
        sr_count = defaultdict(int)

        for audio_file in audio_files:
            ext = audio_file.suffix.lower()
            extensions[ext] += 1

            info = analyze_audio_file(audio_file)
            if info:
                durations.append(info['duration'])
                sample_rates[info['sample_rate']] += 1
                channels[info['num_channels']] += 1
                sr_count[info['sample_rate']] += 1

        # Compute statistics
        total_folder_duration = sum(durations)
        avg_duration = sum(durations) / len(durations) if durations else 0

        folder_stats[folder.name] = {
            'num_files': len(audio_files),
            'total_duration_hours': total_folder_duration / 3600,
            'avg_duration_seconds': avg_duration,
            'sample_rates': dict(sr_count),
        }

        total_files += len(audio_files)
        total_duration += total_folder_duration

        print(f"  Files: {len(audio_files):,}")
        print(f"  Total duration: {total_folder_duration / 3600:.2f} hours")
        print(f"  Avg duration: {avg_duration:.2f} seconds")
        print(f"  Sample rates: {dict(sr_count)}\n")

    # Overall summary
    print("=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"\nTotal folders analyzed: {len(folder_stats)}")
    print(f"Total audio files: {total_files:,}")
    print(f"Total duration: {total_duration / 3600:.2f} hours ({total_duration / 86400:.2f} days)")

    print(f"\nSample rate distribution:")
    for sr, count in sorted(sample_rates.items()):
        print(f"  {sr} Hz: {count:,} files ({100*count/total_files:.1f}%)")

    print(f"\nChannel distribution:")
    for ch, count in sorted(channels.items()):
        print(f"  {ch} channel(s): {count:,} files ({100*count/total_files:.1f}%)")

    print(f"\nFile format distribution:")
    for ext, count in sorted(extensions.items()):
        print(f"  {ext}: {count:,} files ({100*count/total_files:.1f}%)")

    # Per-folder summary table
    print("\n" + "=" * 60)
    print("PER-FOLDER SUMMARY")
    print("=" * 60)
    print(f"\n{'Folder':<30} {'Files':>10} {'Hours':>10} {'Avg (s)':>10}")
    print("-" * 62)

    for folder_name, stats in sorted(folder_stats.items(),
                                    key=lambda x: x[1]['num_files'],
                                    reverse=True):
        print(f"{folder_name:<30} {stats['num_files']:>10,} "
              f"{stats['total_duration_hours']:>10.2f} "
              f"{stats['avg_duration_seconds']:>10.2f}")

    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)

    if total_files < 10000:
        samples = total_files
    else:
        samples = min(50000, total_files)

    val_samples = int(samples * 0.2)
    train_samples = samples - val_samples

    print(f"\nSuggested training setup:")
    print(f"  Total samples to use: {samples:,}")
    print(f"  Train: {train_samples:,} files")
    print(f"  Val: {val_samples:,} files")
    print(f"  Batch size: 8 (adjust based on GPU memory)")
    print(f"  Epochs: Start with 10-20 to test convergence")

if __name__ == '__main__':
    analyze_dataset()

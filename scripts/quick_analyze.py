#!/usr/bin/env python3
"""
Quick analysis of audio dataset without loading audio files.
"""

import os
from pathlib import Path
from collections import defaultdict

SOURCE_DIR = Path("/mnt/data/processed")

def analyze_dataset():
    """Analyze all audio files in the dataset."""

    print("Scanning for audio files...")
    audio_extensions = ['.wav', '.WAV', '.mp3', '.flac', '.ogg', '.m4a']

    # Get all subdirectories (except audio/text metadata folders)
    folders = [d for d in SOURCE_DIR.iterdir()
               if d.is_dir() and d.name not in ['audio', 'text', 'meta', 'metadata', '.git']]

    print(f"\nFound {len(folders)} folders to analyze\n")

    # Statistics per folder
    folder_stats = {}
    total_files = 0
    extensions = defaultdict(int)

    for folder in sorted(folders):
        # Skip if it's a symlink
        if folder.is_symlink():
            target = str(folder.resolve())
            print(f"Skipping symlink: {folder.name} -> {target}")
            continue

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

        # Count files by extension
        ext_count = defaultdict(int)
        for f in audio_files:
            ext = f.suffix.lower()
            ext_count[ext] += 1
            extensions[ext] += 1

        folder_stats[folder.name] = {
            'num_files': len(audio_files),
            'extensions': dict(ext_count),
        }

        total_files += len(audio_files)

        print(f"  Files: {len(audio_files):,}")
        print(f"  Extensions: {dict(ext_count)}\n")

    # Overall summary
    print("=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"\nTotal folders analyzed: {len(folder_stats)}")
    print(f"Total audio files: {total_files:,}")

    print(f"\nFile format distribution:")
    for ext, count in sorted(extensions.items()):
        print(f"  {ext}: {count:,} files ({100*count/total_files if total_files > 0 else 0:.1f}%)")

    # Per-folder summary table
    print("\n" + "=" * 60)
    print("PER-FOLDER SUMMARY (Top 20)")
    print("=" * 60)
    print(f"\n{'Folder':<40} {'Files':>12}")
    print("-" * 54)

    for folder_name, stats in sorted(folder_stats.items(),
                                    key=lambda x: x[1]['num_files'],
                                    reverse=True)[:20]:
        print(f"{folder_name:<40} {stats['num_files']:>12,}")

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

if __name__ == '__main__':
    analyze_dataset()

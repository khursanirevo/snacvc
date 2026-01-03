#!/usr/bin/env python3
"""
Prepare dataset for SNAC speaker conditioning training.

This version doesn't require speaker labels - the model learns speaker
characteristics from the audio itself using ECAPA-TDNN embeddings.

For ablation study, we can:
1. Train without speaker labels
2. Extract embeddings and cluster to evaluate if speaker groups emerge
"""

import os
import shutil
from pathlib import Path
import random

# Configuration
SOURCE_DIR = Path("/mnt/data/processed")
OUTPUT_DIR = Path("/mnt/data/work/snac/data")
TOTAL_SAMPLES = 50000  # Total audio files to use
VAL_SPLIT = 0.2  # 20% for validation
RANDOM_SEED = 42

def find_audio_files(directory):
    """Find all audio files in a directory recursively."""
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(directory.glob(f'**/*{ext}'))
        audio_files.extend(directory.glob(f'**/*{ext.upper()}'))

    return audio_files

def prepare_dataset():
    """Prepare train/val datasets - no speaker labels needed!"""

    # Create output directories
    train_dir = OUTPUT_DIR / 'train'
    val_dir = OUTPUT_DIR / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print("Finding all audio files in /mnt/data/processed...")
    audio_files = find_audio_files(SOURCE_DIR)

    # Remove duplicates and filter
    seen = set()
    unique_files = []
    for f in audio_files:
        if f.name not in seen:
            seen.add(f.name)
            unique_files.append(f)

    audio_files = unique_files
    print(f"Found {len(audio_files)} unique audio files")

    # Sample files
    if len(audio_files) > TOTAL_SAMPLES:
        print(f"Sampling {TOTAL_SAMPLES} files from {len(audio_files)} total")
        audio_files = random.sample(audio_files, TOTAL_SAMPLES)

    # Shuffle and split
    random.shuffle(audio_files)
    val_count = int(len(audio_files) * VAL_SPLIT)
    val_files = audio_files[:val_count]
    train_files = audio_files[val_count:]

    print(f"\nTrain: {len(train_files)} files")
    print(f"Val: {len(val_files)} files")

    # Copy/symlink files (flat structure, no speaker folders)
    print("\nCopying files to train directory...")
    for i, audio_file in enumerate(train_files):
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(train_files)}")

        dest = train_dir / audio_file.name
        if not dest.exists():
            try:
                # Symlink to save disk space
                os.symlink(audio_file, dest)
            except:
                # Fallback to copy if symlink fails
                shutil.copy2(audio_file, dest)

    print("\nCopying files to val directory...")
    for i, audio_file in enumerate(val_files):
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(val_files)}")

        dest = val_dir / audio_file.name
        if not dest.exists():
            try:
                os.symlink(audio_file, dest)
            except:
                shutil.copy2(audio_file, dest)

    # Summary
    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("="*60)
    print(f"\nTrain: {len(train_files)} audio files")
    print(f"Val: {len(val_files)} audio files")
    print(f"\nNote: No speaker labels needed!")
    print(f"The model learns speaker characteristics from embeddings.")

if __name__ == '__main__':
    prepare_dataset()

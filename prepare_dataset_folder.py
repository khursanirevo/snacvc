#!/usr/bin/env python3
"""
Prepare a dataset folder in the Revolab combined format.

Creates the following structure:
    dataset_name/
    ├── manifest.json
    ├── train/
    │   ├── audio/       # Audio files (.wav, .mp3, etc.)
    │   └── json/        # Optional metadata JSON files
    └── valid/
        ├── audio/
        └── json/

Usage:
    # Create empty structure
    python prepare_dataset_folder.py --name my_dataset --output /mnt/data/processed/my_dataset

    # Organize existing audio files with train/val split (symlinks by default)
    python prepare_dataset_folder.py \
        --name my_dataset \
        --output /mnt/data/processed/my_dataset \
        --source /path/to/audio/files \
        --val_split 0.1

    # Copy files instead of symlinks
    python prepare_dataset_folder.py \
        --name my_dataset \
        --output /mnt/data/processed/my_dataset \
        --source /path/to/audio/files \
        --val_split 0.1 \
        --copy

    # Organize from existing train/valid folders
    python prepare_dataset_folder.py \
        --name my_dataset \
        --output /mnt/data/processed/my_dataset \
        --train_dir /path/to/train \
        --valid_dir /path/to/valid

    # Combine multiple source directories with stratified split
    python prepare_dataset_folder.py \
        --name my_dataset \
        --output /mnt/data/processed/my_dataset \
        --sources /path/to/dir1 /path/to/dir2 \
        --val_split 0.1
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import Tuple

# Global variable for symlink mode (needed for multiprocessing)
USE_SYMLINK = True
COLLISION_LOG = []


def log_collision(src_file: Path, existing_file: Path, dst_dir: Path):
    """Log file collision for review."""
    global COLLISION_LOG
    collision_info = {
        "new_file": str(src_file),
        "existing_file": str(existing_file),
        "destination_dir": str(dst_dir)
    }
    COLLISION_LOG.append(collision_info)
    print(f"⚠️  Collision: {src_file.name}")
    print(f"   Existing: {existing_file}")
    print(f"   Skipped:  {src_file}")
    return 1


def create_folder_structure(output_dir: Path):
    """Create the standard folder structure."""
    folders = [
        output_dir / "train" / "audio",
        output_dir / "train" / "json",
        output_dir / "valid" / "audio",
        output_dir / "valid" / "json",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    print(f"✅ Created folder structure in: {output_dir}")
    return folders


def copy_file(src_dst: Tuple[Path, Path]):
    """Copy or symlink a single file - designed for multiprocessing."""
    global USE_SYMLINK
    src, dst = src_dst
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Check for collision
        if dst.exists() or dst.is_symlink():
            # Resolve existing symlink if it is one
            if dst.is_symlink():
                existing_target = Path(os.readlink(dst))
                src_abs = src.resolve()
                # If symlink points to same file, it's a true duplicate
                if existing_target == src_abs or existing_target == src:
                    return 0  # Skip duplicate
            # Log collision and skip
            log_collision(src, dst, dst.parent)
            return 0

        if USE_SYMLINK:
            # Create absolute symlink to save space
            src_abs = src.resolve()
            dst.symlink_to(src_abs)
        else:
            shutil.copy2(src, dst)
        return 1
    except Exception as e:
        print(f"Error {'symlinking' if USE_SYMLINK else 'copying'} {src}: {e}")
        return 0


def create_manifest(output_dir: Path, name: str, train_count: int = 0, valid_count: int = 0,
                    source_datasets: list = None):
    """Create manifest.json file."""
    manifest = {
        "name": name,
        "train_samples": train_count,
        "valid_samples": valid_count,
        "total_samples": train_count + valid_count,
        "source_datasets": len(source_datasets) if source_datasets else 0,
        "datasets": source_datasets or []
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Created manifest.json")
    print(f"   Train samples: {train_count}")
    print(f"   Valid samples: {valid_count}")
    print(f"   Total: {train_count + valid_count}")


def organize_from_single_source(source_dir: Path, output_dir: Path, val_split: float = 0.1,
                                 seed: int = 42):
    """Organize audio files from a single directory into train/valid splits."""
    random.seed(seed)

    # Find all audio files
    audio_extensions = ['.wav', '.WAV', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(list(source_dir.glob(f'*{ext}')))

    if len(audio_files) == 0:
        print(f"⚠️  No audio files found in {source_dir}")
        return 0, 0

    print(f"Found {len(audio_files)} audio files")

    # Shuffle and split
    random.shuffle(audio_files)
    split_idx = int(len(audio_files) * (1 - val_split))

    train_files = audio_files[:split_idx]
    valid_files = audio_files[split_idx:]

    print(f"Split: {len(train_files)} train, {len(valid_files)} valid")

    # Parallel copy
    num_workers = min(cpu_count(), 32)

    # Prepare training files
    print(f"\nCopying training files (using {num_workers} workers)...")
    train_copy_tasks = []
    for src in train_files:
        dst = output_dir / "train" / "audio" / src.name
        train_copy_tasks.append((src, dst))

    # Check for JSON files
    for src in train_files:
        json_src = src.with_suffix('.json')
        if json_src.exists():
            json_dst = output_dir / "train" / "json" / json_src.name
            train_copy_tasks.append((json_src, json_dst))

    with Pool(num_workers) as pool:
        list(tqdm(
            pool.imap_unordered(copy_file, train_copy_tasks),
            total=len(train_copy_tasks),
            desc="Train"
        ))

    # Prepare validation files
    print(f"\nCopying validation files (using {num_workers} workers)...")
    valid_copy_tasks = []
    for src in valid_files:
        dst = output_dir / "valid" / "audio" / src.name
        valid_copy_tasks.append((src, dst))

    # Check for JSON files
    for src in valid_files:
        json_src = src.with_suffix('.json')
        if json_src.exists():
            json_dst = output_dir / "valid" / "json" / json_src.name
            valid_copy_tasks.append((json_src, json_dst))

    with Pool(num_workers) as pool:
        list(tqdm(
            pool.imap_unordered(copy_file, valid_copy_tasks),
            total=len(valid_copy_tasks),
            desc="Valid"
        ))

    return len(train_files), len(valid_files)


def organize_from_multiple_sources(source_dirs: list, output_dir: Path, val_split: float = 0.1,
                                    seed: int = 42):
    """
    Organize audio files from multiple directories with stratified sampling.

    Each source directory maintains its own 90/10 split, then all are combined.
    """
    random.seed(seed)

    audio_extensions = ['.wav', '.WAV', '.mp3', '.flac', '.ogg', '.m4a']
    all_train_files = []
    all_valid_files = []
    source_dataset_names = []

    print(f"\n{'='*70}")
    print(f"Stratified sampling from {len(source_dirs)} source directories")
    print(f"{'='*70}\n")

    for source_dir in source_dirs:
        source_path = Path(source_dir)
        dataset_name = source_path.name

        # Check if source_dir has 'audio' subdirectory
        if (source_path / "audio").exists():
            audio_search_path = source_path / "audio"
        else:
            audio_search_path = source_path

        # Find all audio files in this source
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(list(audio_search_path.glob(f'*{ext}')))

        if len(audio_files) == 0:
            print(f"⚠️  No audio files found in {source_dir}, skipping...")
            continue

        print(f"📁 {dataset_name}: {len(audio_files)} audio files")

        # Shuffle and split this dataset
        random.shuffle(audio_files)
        split_idx = int(len(audio_files) * (1 - val_split))

        train_files = audio_files[:split_idx]
        valid_files = audio_files[split_idx:]

        print(f"   → Train: {len(train_files)}, Valid: {len(valid_files)}")

        # Add to combined lists
        all_train_files.extend([(f, dataset_name) for f in train_files])
        all_valid_files.extend([(f, dataset_name) for f in valid_files])
        source_dataset_names.append(dataset_name)

    print(f"\n{'='*70}")
    print(f"Total: {len(all_train_files)} train, {len(all_valid_files)} valid")
    print(f"{'='*70}\n")

    if len(all_train_files) == 0:
        print("⚠️  No audio files found in any source directory!")
        return 0, 0, []

    # Prepare copy tasks for parallel processing
    num_workers = min(cpu_count(), 32)  # Use up to 32 workers

    # Copy training files in parallel
    print(f"Copying training files (using {num_workers} workers)...")
    train_copy_tasks = []
    train_json_tasks = []

    for src, dataset_name in all_train_files:
        # Use original filename (no prefix) to detect duplicates
        dst = output_dir / "train" / "audio" / src.name
        train_copy_tasks.append((src, dst))

        # Check for JSON files
        json_src = src.with_suffix('.json')
        if json_src.exists():
            json_dst = output_dir / "train" / "json" / json_src.name
            train_json_tasks.append((json_src, json_dst))

    # Parallel copy training audio files
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(copy_file, train_copy_tasks),
            total=len(train_copy_tasks),
            desc="Train audio"
        ))

    # Copy JSON files for training (usually fewer, can use same pool)
    if train_json_tasks:
        with Pool(num_workers) as pool:
            list(tqdm(
                pool.imap_unordered(copy_file, train_json_tasks),
                total=len(train_json_tasks),
                desc="Train JSON"
            ))

    # Copy validation files in parallel
    print(f"\nCopying validation files (using {num_workers} workers)...")
    valid_copy_tasks = []
    valid_json_tasks = []

    for src, dataset_name in all_valid_files:
        # Use original filename (no prefix) to detect duplicates
        dst = output_dir / "valid" / "audio" / src.name
        valid_copy_tasks.append((src, dst))

        # Check for JSON files
        json_src = src.with_suffix('.json')
        if json_src.exists():
            json_dst = output_dir / "valid" / "json" / json_src.name
            valid_json_tasks.append((json_src, json_dst))

    # Parallel copy validation audio files
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(copy_file, valid_copy_tasks),
            total=len(valid_copy_tasks),
            desc="Valid audio"
        ))

    # Copy JSON files for validation
    if valid_json_tasks:
        with Pool(num_workers) as pool:
            list(tqdm(
                pool.imap_unordered(copy_file, valid_json_tasks),
                total=len(valid_json_tasks),
                desc="Valid JSON"
            ))

    # Save collision log
    global COLLISION_LOG
    if COLLISION_LOG:
        collision_file = output_dir / "collisions.jsonl"
        with open(collision_file, 'w') as f:
            for collision in COLLISION_LOG:
                f.write(json.dumps(collision) + '\n')
        print(f"\n⚠️  Found {len(COLLISION_LOG)} collisions")
        print(f"   Logged to: {collision_file}")
        COLLISION_LOG = []  # Reset for next run

    return len(all_train_files), len(all_valid_files), source_dataset_names


def organize_from_split_sources(train_dir: Path, valid_dir: Path, output_dir: Path):
    """Organize audio files from existing train/valid directories."""

    def copy_audio_files(source_dir: Path, target_split: str):
        """Copy or symlink audio files from source to target directory."""
        global USE_SYMLINK
        audio_extensions = ['.wav', '.WAV', '.mp3', '.flac', '.ogg', '.m4a']
        audio_files = []

        for ext in audio_extensions:
            audio_files.extend(list(source_dir.glob(f'*{ext}')))

        if len(audio_files) == 0:
            print(f"⚠️  No audio files found in {source_dir}")
            return 0

        print(f"Found {len(audio_files)} {target_split} files")

        target_audio_dir = output_dir / target_split / "audio"
        target_json_dir = output_dir / target_split / "json"

        for src in tqdm(audio_files, desc=target_split.capitalize()):
            dst = target_audio_dir / src.name
            if USE_SYMLINK:
                src_abs = src.resolve()
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(src_abs)
            else:
                shutil.copy2(src, dst)

            # Also copy/symlink JSON if it exists
            json_src = src.with_suffix('.json')
            if json_src.exists():
                json_dst = target_json_dir / json_src.name
                if USE_SYMLINK:
                    json_abs = json_src.resolve()
                    if json_dst.is_symlink() or json_dst.exists():
                        json_dst.unlink()
                    json_dst.symlink_to(json_abs)
                else:
                    shutil.copy2(json_src, json_dst)

        return len(audio_files)

    train_count = copy_audio_files(train_dir, "train")
    valid_count = copy_audio_files(valid_dir, "valid")

    return train_count, valid_count


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a dataset folder in Revolab combined format"
    )
    parser.add_argument("--name", type=str, required=True,
                        help="Dataset name (e.g., 'my_dataset')")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory path")
    parser.add_argument("--source", type=str, default=None,
                        help="Source directory with audio files (will split into train/valid)")
    parser.add_argument("--sources", type=str, nargs='*', default=[],
                        help="Multiple source directories (stratified 90/10 split from each)")
    parser.add_argument("--train_dir", type=str, default=None,
                        help="Existing training directory")
    parser.add_argument("--valid_dir", type=str, default=None,
                        help="Existing validation directory")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1 = 10%%)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files instead of creating symlinks (default: symlinks to save space)")

    args = parser.parse_args()

    global USE_SYMLINK
    USE_SYMLINK = not args.copy

    if USE_SYMLINK:
        print("🔗 Using SYMLINKS to save disk space")
    else:
        print("📋 COPYING files (will use more disk space)")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create folder structure
    create_folder_structure(output_dir)

    # Organize files
    train_count = 0
    valid_count = 0
    source_datasets = []

    if args.sources:
        # Multiple source directories with stratified sampling
        train_count, valid_count, source_datasets = organize_from_multiple_sources(
            args.sources, output_dir, args.val_split, args.seed
        )

    elif args.source:
        # Single source directory with train/val split
        print(f"\n{'='*70}")
        print(f"Organizing from: {args.source}")
        print(f"Validation split: {args.val_split*100:.0f}%")
        print(f"{'='*70}\n")

        train_count, valid_count = organize_from_single_source(
            Path(args.source), output_dir, args.val_split, args.seed
        )
        source_datasets = [Path(args.source).name]

    elif args.train_dir and args.valid_dir:
        # Existing train/valid directories
        print(f"\n{'='*70}")
        print(f"Organizing from existing splits:")
        print(f"  Train: {args.train_dir}")
        print(f"  Valid: {args.valid_dir}")
        print(f"{'='*70}\n")

        train_count, valid_count = organize_from_split_sources(
            Path(args.train_dir), Path(args.valid_dir), output_dir
        )
        source_datasets = [Path(args.train_dir).parent.name]

    else:
        print("\n⚠️  No source directories specified. Created empty folder structure.")
        print("   Use --source, --sources, or --train_dir/--valid_dir to populate with files.")

    # Create manifest
    create_manifest(
        output_dir, args.name, train_count, valid_count, source_datasets
    )

    print(f"\n{'='*70}")
    print(f"✅ Dataset prepared successfully!")
    print(f"{'='*70}")
    print(f"Location: {output_dir}")
    print(f"\nUse this path in your training config:")
    print(f'  "train_data": "{output_dir}/train/audio",')
    print(f'  "val_data": "{output_dir}/valid/audio"')


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Coordinator script to build FAISS-based speaker embedding index using 2 GPUs.

Extracts embeddings for all audio files and builds a FAISS index for fast similarity search.
"""

import subprocess
import sys
import json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '.')


def main():
    print("="*70)
    print("FAISS-based Speaker Embedding Index Builder")
    print("Using GPU 1 and GPU 2")
    print("="*70)

    dataset_root = Path('data/train_split')
    output_path = 'pretrained_models/speaker_faiss.index'

    # [1/2] List all audio files
    print("\n[1/3] Finding all audio files...")
    audio_files = list(dataset_root.glob("**/*.wav"))
    print(f"Found {len(audio_files)} audio files")

    # Split files between GPUs
    mid = len(audio_files) // 2

    gpu1_files = audio_files[:mid]
    gpu2_files = audio_files[mid:]

    print(f"\n[2/3] Extracting embeddings (parallel)...")
    print(f"  GPU 1: {len(gpu1_files)} files")
    print(f"  GPU 2: {len(gpu2_files)} files")

    # Save file lists
    gpu1_files_path = 'temp/files_1.json'
    gpu2_files_path = 'temp/files_2.json'

    Path('temp').mkdir(exist_ok=True)

    with open(gpu1_files_path, 'w') as f:
        json.dump([str(p) for p in gpu1_files], f)

    with open(gpu2_files_path, 'w') as f:
        json.dump([str(p) for p in gpu2_files], f)

    # Launch workers
    print("\nLaunching workers...")

    gpu1_cmd = [
        'uv', 'run', 'python', 'scripts/build_faiss_worker.py',
        '--files', gpu1_files_path,
        '--output', 'temp/embeddings_1.npy',
        '--gpu', '0'
    ]

    gpu2_cmd = [
        'uv', 'run', 'python', 'scripts/build_faiss_worker.py',
        '--files', gpu2_files_path,
        '--output', 'temp/embeddings_2.npy',
        '--gpu', '1'
    ]

    import os
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '1,2'

    proc1 = subprocess.Popen(gpu1_cmd, env=env)
    proc2 = subprocess.Popen(gpu2_cmd, env=env)

    # Wait for both
    print("Waiting for workers to complete...\n")
    proc1.wait()
    proc2.wait()

    print("\n✅ Both workers complete!")

    # Load embeddings
    print("\n[3/3] Building FAISS index...")

    embeddings_1 = np.load('temp/embeddings_1.npy')
    embeddings_2 = np.load('temp/embeddings_2.npy')

    with open('temp/embeddings_1_paths.json', 'r') as f:
        paths_1 = json.load(f)

    with open('temp/embeddings_2_paths.json', 'r') as f:
        paths_2 = json.load(f)

    # Combine
    all_embeddings = np.vstack([embeddings_1, embeddings_2])
    all_paths = paths_1 + paths_2

    print(f"Combined embeddings: {len(all_paths)} files")

    # Build FAISS index
    print("Building FAISS index...")

    import faiss

    embedding_dim = all_embeddings.shape[1]  # Should be 512

    # Use Inner Product (IP) since embeddings are L2-normalized
    # IP on normalized vectors = cosine similarity
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(all_embeddings.astype('float32'))

    # Save index
    print(f"Saving to {output_path}...")
    faiss.write_index(index, output_path)

    # Save file paths
    paths_file = 'pretrained_models/speaker_faiss_paths.json'
    with open(paths_file, 'w') as f:
        json.dump(all_paths, f)

    print(f"\n{'='*70}")
    print(f"✅ FAISS index saved to: {output_path}")
    print(f"  Total embeddings: {len(all_paths)}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  File paths saved to: {paths_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

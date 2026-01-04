#!/usr/bin/env python3
"""
Coordinator script to build speaker index using 2 GPUs in parallel.

Splits work and launches two worker processes.
"""

import subprocess
import sys
import json
from pathlib import Path
import re
from collections import defaultdict
import numpy as np
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, '.')

from snac import SNACWithSpeakerConditioning


def main():
    print("="*70)
    print("Parallel Speaker Index Builder")
    print("Using GPU 1 and GPU 2")
    print("="*70)

    dataset_root = Path('data/train_split')
    output_path = 'pretrained_models/speaker_index.json'
    n_clusters = 50

    # [1/3] Group audio files
    print("\n[1/3] Grouping audio files by speaker...")
    audio_files = list(dataset_root.glob("**/*.wav"))
    print(f"Found {len(audio_files)} audio files")

    speaker_to_files = defaultdict(list)

    for audio_path in audio_files:
        filename = audio_path.stem

        # Extract UUID
        uuid_match = re.search(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            filename,
            re.IGNORECASE
        )

        if uuid_match:
            speaker_id = uuid_match.group(0)
        else:
            parts = filename.split('_')
            speaker_id = parts[0] if len(parts) > 1 else filename

        speaker_to_files[speaker_id].append(audio_path)

    print(f"Found {len(speaker_to_files)} unique speakers")

    # Split speakers
    speaker_ids = list(speaker_to_files.keys())
    mid = len(speaker_ids) // 2

    gpu1_speakers = speaker_ids[:mid]
    gpu2_speakers = speaker_ids[mid:]

    print(f"\n[2/3] Extracting embeddings (parallel)...")
    print(f"  GPU 1: {len(gpu1_speakers)} speakers")
    print(f"  GPU 2: {len(gpu2_speakers)} speakers")

    # Save speaker lists
    gpu1_speakers_path = 'temp/speakers_1.json'
    gpu2_speakers_path = 'temp/speakers_2.json'
    speaker_files_path = 'temp/speaker_files.json'

    Path('temp').mkdir(exist_ok=True)

    with open(gpu1_speakers_path, 'w') as f:
        json.dump(gpu1_speakers, f)

    with open(gpu2_speakers_path, 'w') as f:
        json.dump(gpu2_speakers, f)

    with open(speaker_files_path, 'w') as f:
        json.dump({k: [str(p) for p in v] for k, v in speaker_to_files.items()}, f)

    # Launch workers
    print("\nLaunching workers...")

    gpu1_cmd = [
        'uv', 'run', 'python', 'scripts/build_index_worker.py',
        '--speakers', gpu1_speakers_path,
        '--files', speaker_files_path,
        '--output', 'temp/embeddings_1.json',
        '--gpu', '0'  # Will be GPU 1 due to CUDA_VISIBLE_DEVICES=1,2
    ]

    gpu2_cmd = [
        'uv', 'run', 'python', 'scripts/build_index_worker.py',
        '--speakers', gpu2_speakers_path,
        '--files', speaker_files_path,
        '--output', 'temp/embeddings_2.json',
        '--gpu', '1'  # Will be GPU 2 due to CUDA_VISIBLE_DEVICES=1,2
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
    print("\n[3/3] Combining results and clustering...")

    with open('temp/embeddings_1.json', 'r') as f:
        data1 = json.load(f)

    with open('temp/embeddings_2.json', 'r') as f:
        data2 = json.load(f)

    # Combine
    all_speaker_embeddings = {}

    for emb_dict in [data1['speaker_embeddings'], data2['speaker_embeddings']]:
        all_speaker_embeddings.update(emb_dict)

    print(f"Combined embeddings: {len(all_speaker_embeddings)} speakers")

    # Cluster
    speaker_ids = list(all_speaker_embeddings.keys())
    embeddings = torch.tensor([all_speaker_embeddings[sid] for sid in speaker_ids])

    print(f"Clustering into {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings.numpy())

    speaker_to_cluster = {}
    cluster_to_speakers = defaultdict(list)

    for speaker_id, cluster_id in zip(speaker_ids, cluster_labels):
        speaker_to_cluster[speaker_id] = int(cluster_id)
        cluster_to_speakers[int(cluster_id)].append(speaker_id)

    print(f"Clustering complete:")
    for cluster_id in sorted(cluster_to_speakers.keys())[:10]:
        n_speakers = len(cluster_to_speakers[cluster_id])
        print(f"  Cluster {cluster_id}: {n_speakers} speakers")

    # Save final index
    print(f"\nSaving to {output_path}...")

    final_data = {
        'speaker_embeddings': {
            k: (v if isinstance(v, list) else v.tolist())
            for k, v in all_speaker_embeddings.items()
        },
        'speaker_files': {
            k: [str(p) for p in v]
            for k, v in speaker_to_files.items()
        },
        'speaker_to_cluster': speaker_to_cluster,
        'cluster_to_speakers': {
            str(k): v for k, v in cluster_to_speakers.items()
        },
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(final_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ Speaker index saved to: {output_path}")
    print(f"  Total speakers: {len(all_speaker_embeddings)}")
    print(f"  Total clusters: {len(cluster_to_speakers)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

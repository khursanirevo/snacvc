#!/usr/bin/env python3
"""
Combine worker outputs and perform clustering.
"""

import json
import torch
from sklearn.cluster import KMeans
from collections import defaultdict
from pathlib import Path

def main():
    print("="*70)
    print("Combining embeddings and clustering")
    print("="*70)

    # Load embeddings from both workers
    print("\n[1/2] Loading embeddings...")
    with open('temp/embeddings_1.json', 'r') as f:
        data1 = json.load(f)

    with open('temp/embeddings_2.json', 'r') as f:
        data2 = json.load(f)

    # Combine
    all_speaker_embeddings = {}

    for emb_dict in [data1['speaker_embeddings'], data2['speaker_embeddings']]:
        all_speaker_embeddings.update(emb_dict)

    print(f"Combined embeddings: {len(all_speaker_embeddings)} speakers")

    # Load speaker files mapping
    with open('temp/speaker_files.json', 'r') as f:
        speaker_to_files = json.load(f)

    # Cluster
    print("\n[2/2] Clustering into 50 clusters...")
    speaker_ids = list(all_speaker_embeddings.keys())
    embeddings = torch.tensor([all_speaker_embeddings[sid] for sid in speaker_ids])

    n_clusters = 50
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
    output_path = 'pretrained_models/speaker_index.json'
    print(f"\nSaving to {output_path}...")

    final_data = {
        'speaker_embeddings': {
            k: (v if isinstance(v, list) else v.tolist())
            for k, v in all_speaker_embeddings.items()
        },
        'speaker_files': {
            k: [str(p) if not isinstance(p, str) else p for p in v]
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

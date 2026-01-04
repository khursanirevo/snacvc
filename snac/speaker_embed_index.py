#!/usr/bin/env python3
"""
Speaker Embedding Index for Semantic Hard Negative Mining.

Builds a speaker-level embedding index to enable true hard negative mining
by clustering similar speakers.

Key ideas:
1. Pre-compute speaker embeddings (one per speaker, not per sample)
2. Cluster speakers by similarity
3. Sample negatives from:
   - Same cluster (hard negatives: similar speakers)
   - Neighboring clusters (semi-hard)
   - Random clusters (easy)
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
from tqdm import tqdm

from snac import SNACWithSpeakerConditioning
from torch.utils.data import DataLoader, Dataset
import torchaudio


class SpeakerEmbeddingIndex:
    """
    Index of speaker embeddings for fast hard negative mining.

    Stores:
    - speaker_id -> embedding (mean of all samples)
    - speaker_id -> list of audio file paths
    - cluster_id -> list of speaker_ids
    """

    def __init__(self, model, device='cuda'):
        """
        Initialize the index.

        Args:
            model: SNAC model with speaker encoder
            device: torch device
        """
        self.model = model
        self.device = device

        # Storage
        self.speaker_embeddings = {}  # speaker_id -> embedding tensor
        self.speaker_files = defaultdict(list)  # speaker_id -> list of file paths
        self.speaker_to_cluster = {}  # speaker_id -> cluster_id
        self.cluster_to_speakers = defaultdict(list)  # cluster_id -> list of speaker_ids

        # Model for getting base model
        if hasattr(model, 'module'):
            self.model_base = model.module
        else:
            self.model_base = model

    @torch.no_grad()
    def extract_speaker_embedding(self, audio_path):
        """Extract speaker embedding from a single audio file."""
        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 24kHz
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            waveform = resampler(waveform)

        # Take first 2 seconds (or less if shorter)
        max_samples = 2 * 24000
        if waveform.shape[-1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[-1] < 24000:
            # Pad if too short
            padding = 24000 - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)

        # Extract embedding
        waveform = waveform.to(self.device)
        embedding = self.model_base.extract_speaker_embedding(waveform.unsqueeze(0))

        return embedding.squeeze(0).cpu()

    def build_from_dataset(self, dataset_root, max_samples_per_speaker=50,
                          num_workers=4):
        """
        Build speaker embedding index from dataset.

        Args:
            dataset_root: Path to dataset root
            max_samples_per_speaker: Max samples to average per speaker
            num_workers: Number of workers for data loading
        """
        dataset_root = Path(dataset_root)

        print(f"[SpeakerEmbeddingIndex] Building index from {dataset_root}")

        # Find all audio files grouped by speaker
        speaker_to_files = defaultdict(list)

        audio_files = list(dataset_root.glob("**/*.wav"))
        print(f"Found {len(audio_files)} audio files")

        for audio_path in tqdm(audio_files, desc="Grouping by speaker"):
            # Extract speaker ID from filename
            # Pattern: speaker_uuid-timestamp-range_segXXXX.wav
            # OR: simple_filename_segXXXX.wav
            filename = audio_path.stem  # Remove extension

            # Try to extract UUID (first UUID-like pattern)
            import re
            uuid_match = re.search(
                r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
                filename,
                re.IGNORECASE
            )

            if uuid_match:
                # Use UUID as speaker ID
                speaker_id = uuid_match.group(0)
            else:
                # Fallback: use part before first underscore
                parts = filename.split('_')
                speaker_id = parts[0] if len(parts) > 1 else filename

            speaker_to_files[speaker_id].append(audio_path)

        print(f"Found {len(speaker_to_files)} unique speakers")

        # Extract embeddings for each speaker
        print("Extracting speaker embeddings...")

        for speaker_id, file_paths in tqdm(speaker_to_files.items(), desc="Speakers"):
            # Limit samples per speaker
            if len(file_paths) > max_samples_per_speaker:
                file_paths = np.random.choice(file_paths, max_samples_per_speaker, replace=False)

            # Extract embeddings for all samples
            embeddings = []
            for audio_path in file_paths:
                try:
                    emb = self.extract_speaker_embedding(audio_path)
                    embeddings.append(emb)
                except Exception as e:
                    continue

            if len(embeddings) == 0:
                continue

            # Average embeddings to get speaker-level embedding
            speaker_emb = torch.stack(embeddings).mean(dim=0)

            # Normalize
            speaker_emb = speaker_emb / (torch.norm(speaker_emb) + 1e-8)

            # Store
            self.speaker_embeddings[speaker_id] = speaker_emb
            self.speaker_files[speaker_id] = [str(fp) for fp in file_paths]

        print(f"Built embeddings for {len(self.speaker_embeddings)} speakers")

    def build_clusters(self, n_clusters=50):
        """
        Cluster speakers by similarity.

        Uses K-means clustering on speaker embeddings.

        Args:
            n_clusters: Number of clusters
        """
        from sklearn.cluster import KMeans

        if len(self.speaker_embeddings) < n_clusters:
            n_clusters = max(2, len(self.speaker_embeddings) // 2)
            print(f"Adjusting clusters to {n_clusters} based on speaker count")

        # Get embedding matrix
        speaker_ids = list(self.speaker_embeddings.keys())
        embeddings = torch.stack([self.speaker_embeddings[sid] for sid in speaker_ids]).numpy()

        print(f"Clustering {len(speaker_ids)} speakers into {n_clusters} clusters...")

        # Fit K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Store mapping
        for speaker_id, cluster_id in zip(speaker_ids, cluster_labels):
            self.speaker_to_cluster[speaker_id] = int(cluster_id)
            self.cluster_to_speakers[int(cluster_id)].append(speaker_id)

        print(f"Clustering complete:")
        for cluster_id in range(n_clusters):
            n_speakers = len(self.cluster_to_speakers[cluster_id])
            print(f"  Cluster {cluster_id}: {n_speakers} speakers")

    def get_hard_negatives(self, anchor_speaker_id, num_negatives=6,
                          hard_ratio=0.5, semi_hard_ratio=0.3):
        """
        Get hard negative speakers for a given anchor speaker.

        Args:
            anchor_speaker_id: The anchor speaker ID
            num_negatives: Total number of negatives to return
            hard_ratio: Fraction from same cluster (hard)
            semi_hard_ratio: Fraction from neighboring clusters (semi-hard)

        Returns:
            - List of negative speaker IDs
        """
        if anchor_speaker_id not in self.speaker_to_cluster:
            # Random sampling if speaker not in index
            all_speakers = list(self.speaker_embeddings.keys())
            all_speakers.remove(anchor_speaker_id)
            return np.random.choice(all_speakers, min(num_negatives, len(all_speakers)), replace=False).tolist()

        anchor_cluster = self.speaker_to_cluster[anchor_speaker_id]

        # Calculate how many from each tier
        n_hard = int(num_negatives * hard_ratio)
        n_semi_hard = int(num_negatives * semi_hard_ratio)
        n_random = num_negatives - n_hard - n_semi_hard

        negatives = []

        # 1. Hard negatives: same cluster (similar speakers)
        same_cluster_speakers = self.cluster_to_speakers[anchor_cluster]
        same_cluster_speakers = [s for s in same_cluster_speakers if s != anchor_speaker_id]

        if len(same_cluster_speakers) > 0:
            if len(same_cluster_speakers) >= n_hard:
                selected = np.random.choice(same_cluster_speakers, n_hard, replace=False)
            else:
                selected = same_cluster_speakers  # Use all if not enough
            negatives.extend(selected)

        # 2. Semi-hard negatives: neighboring clusters
        # Find clusters with most similar centroids
        anchor_emb = self.speaker_embeddings[anchor_speaker_id]
        cluster_similarities = {}

        for cluster_id, speaker_ids in self.cluster_to_speakers.items():
            if cluster_id == anchor_cluster:
                continue

            # Compute mean embedding for cluster
            cluster_embs = [self.speaker_embeddings[sid] for sid in speaker_ids]
            cluster_mean = torch.stack(cluster_embs).mean(dim=0)

            # Similarity to anchor
            sim = torch.dot(anchor_emb, cluster_mean).item()
            cluster_similarities[cluster_id] = sim

        # Sort by similarity (most similar first)
        sorted_clusters = sorted(cluster_similarities.items(), key=lambda x: x[1], reverse=True)

        for cluster_id, _ in sorted_clusters[:n_semi_hard]:
            cluster_speakers = self.cluster_to_speakers[cluster_id]
            if len(cluster_speakers) > 0:
                speaker = np.random.choice(cluster_speakers)
                negatives.append(speaker)

        # 3. Random negatives: any cluster
        all_speakers = list(self.speaker_embeddings.keys())
        all_speakers = [s for s in all_speakers if s != anchor_speaker_id and s not in negatives]

        remaining = n_random - (len(negatives) - n_hard)
        if remaining > 0 and len(all_speakers) > 0:
            selected = np.random.choice(all_speakers, min(remaining, len(all_speakers)), replace=False)
            negatives.extend(selected)

        # If still not enough, add more random
        if len(negatives) < num_negatives:
            all_speakers = list(self.speaker_embeddings.keys())
            all_speakers = [s for s in all_speakers if s != anchor_speaker_id and s not in negatives]
            additional = np.random.choice(all_speakers, min(num_negatives - len(negatives), len(all_speakers)), replace=False)
            negatives.extend(additional)

        return negatives[:num_negatives]

    def get_random_samples(self, speaker_id, num_samples=1):
        """Get random audio file paths for a given speaker."""
        files = self.speaker_files.get(speaker_id, [])
        if len(files) == 0:
            return []

        if len(files) >= num_samples:
            return np.random.choice(files, num_samples, replace=False).tolist()
        else:
            return files

    def save(self, path):
        """Save index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert tensors to numpy for JSON serialization
        data = {
            'speaker_embeddings': {
                k: v.numpy().tolist() for k, v in self.speaker_embeddings.items()
            },
            'speaker_files': dict(self.speaker_files),
            'speaker_to_cluster': self.speaker_to_cluster,
            'cluster_to_speakers': {
                str(k): v for k, v in self.cluster_to_speakers.items()
            },
        }

        with open(path, 'w') as f:
            json.dump(data, f)

        print(f"Saved speaker index to {path}")

    @classmethod
    def load(cls, path, model, device='cuda'):
        """Load index from disk."""
        path = Path(path)

        with open(path, 'r') as f:
            data = json.load(f)

        # Create instance
        index = cls(model, device)

        # Load embeddings
        for speaker_id, emb_list in data['speaker_embeddings'].items():
            index.speaker_embeddings[speaker_id] = torch.tensor(emb_list)

        # Load files
        index.speaker_files = defaultdict(list, data['speaker_files'])

        # Load clusters
        index.speaker_to_cluster = data['speaker_to_cluster']
        index.cluster_to_speakers = {
            int(k): v for k, v in data['cluster_to_speakers'].items()
        }

        print(f"Loaded speaker index from {path}")
        print(f"  Speakers: {len(index.speaker_embeddings)}")
        print(f"  Clusters: {len(index.cluster_to_speakers)}")

        return index


def build_speaker_index(model, dataset_root, output_path, n_clusters=50,
                        device='cuda', max_samples_per_speaker=50):
    """
    Build and save speaker embedding index.

    Args:
        model: SNAC model
        dataset_root: Path to training data
        output_path: Where to save the index
        n_clusters: Number of clusters for hard negative mining
        device: torch device
        max_samples_per_speaker: Max samples to average per speaker

    Returns:
        - SpeakerEmbeddingIndex instance
    """
    # Create index
    index = SpeakerEmbeddingIndex(model, device)

    # Build from dataset
    index.build_from_dataset(
        dataset_root=dataset_root,
        max_samples_per_speaker=max_samples_per_speaker
    )

    # Build clusters
    index.build_clusters(n_clusters=n_clusters)

    # Save
    index.save(output_path)

    return index


if __name__ == "__main__":
    print("Speaker Embedding Index Builder")
    print("="*70)
    print("\nThis module provides:")
    print("1. SpeakerEmbeddingIndex: Build and query speaker embedding index")
    print("2. build_speaker_index(): Convenience function to build index")
    print("\nUsage:")
    print("  from snac.speaker_embed_index import build_speaker_index")
    print("  index = build_speaker_index(")
    print("      model=model,")
    print("      dataset_root='data/train_split',")
    print("      output_path='pretrained_models/speaker_index.json',")
    print("      n_clusters=50")
    print("  )")

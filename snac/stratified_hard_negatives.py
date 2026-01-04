#!/usr/bin/env python3
"""
Stratified Hard Negative Mining for Voice Conversion.

Instead of just using "similar speakers" as hard negatives, we use a
stratified sampling approach:

1. Easy negatives: Very different speakers (similarity < 0.3)
2. Medium negatives: Moderately different speakers (similarity 0.3-0.6)
3. Hard negatives: Similar speakers (similarity 0.6-0.85)

Benefits:
- More diverse training signal
- Covers full speaker space
- Prevents overfitting to specific speaker clusters
- Better generalization
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict


class StratifiedHardNegativeSampler:
    """
    Samples hard negatives from multiple difficulty tiers.

    Uses FAISS for fast search but samples from different similarity ranges.
    """

    def __init__(self, faiss_index, embedding_cache, config):
        """
        Initialize the sampler.

        Args:
            faiss_index: FaissSpeakerIndex with loaded index
            embedding_cache: Dict mapping file_path → embedding tensor
            config: Training config with sampling ratios
        """
        self.faiss_index = faiss_index
        self.embedding_cache = embedding_cache

        # Sampling configuration
        self.num_negatives = config.get('max_negatives', 6)

        # Difficulty tier ratios (should sum to 1.0)
        # Easy: very different speakers
        # Medium: moderately different
        # Hard: similar speakers
        self.ratio_easy = config.get('neg_ratio_easy', 0.3)
        self.ratio_medium = config.get('neg_ratio_medium', 0.4)
        self.ratio_hard = config.get('neg_ratio_hard', 0.3)

        # Similarity thresholds
        self.threshold_easy_medium = config.get('threshold_easy_medium', 0.3)
        self.threshold_medium_hard = config.get('threshold_medium_hard', 0.6)
        self.threshold_hard_same = config.get('same_speaker_threshold', 0.85)

        # Pre-compute similarity ranges for faster sampling
        self._build_similarity_bins()

    def _build_similarity_bins(self):
        """
        Pre-compute speaker similarity bins for stratified sampling.

        For each speaker, pre-compute which other speakers fall into each
        difficulty tier. This makes sampling much faster during training.
        """
        import faiss

        print("Building stratified similarity bins...")

        # Get all embeddings from cache (in FAISS order)
        all_embs = []
        valid_paths = []

        for path in self.faiss_index.file_paths:
            if path in self.embedding_cache:
                emb = self.embedding_cache[path].numpy()
                all_embs.append(emb)
                valid_paths.append(path)

        all_embs = np.vstack(all_embs).astype('float32')

        # Build FAISS index for fast similarity search (if not already searchable)
        # Note: faiss_index already has an index, but we need to access it
        index = self.faiss_index.index

        # Compute similarities for all pairs (expensive but one-time)
        # For each speaker, find neighbors and bin them by similarity
        self.speaker_bins = {}

        print(f"Computing similarity bins for {len(valid_paths)} speakers...")

        for i, path in enumerate(valid_paths):
            query_emb = all_embs[i:i+1]

            # Search for k neighbors (enough to fill all bins)
            k = min(1000, len(valid_paths))  # Limit for efficiency
            similarities, indices = index.search(query_emb, k=k)

            # Initialize bins
            easy_bin = []
            medium_bin = []
            hard_bin = []

            # Bin by similarity (skip self)
            for sim, idx in zip(similarities[0], indices[0]):
                if idx == i:  # Skip self
                    continue

                if sim < self.threshold_easy_medium:
                    easy_bin.append(idx)
                elif sim < self.threshold_medium_hard:
                    medium_bin.append(idx)
                elif sim < self.threshold_hard_same:
                    hard_bin.append(idx)

            self.speaker_bins[path] = {
                'easy': easy_bin,
                'medium': medium_bin,
                'hard': hard_bin,
            }

        # Print statistics
        easy_sizes = [len(bins['easy']) for bins in self.speaker_bins.values()]
        medium_sizes = [len(bins['medium']) for bins in self.speaker_bins.values()]
        hard_sizes = [len(bins['hard']) for bins in self.speaker_bins.values()]

        print(f"Similarity bins built:")
        print(f"  Easy: {np.mean(easy_sizes):.1f} speakers (avg)")
        print(f"  Medium: {np.mean(medium_sizes):.1f} speakers (avg)")
        print(f"  Hard: {np.mean(hard_sizes):.1f} speakers (avg)")

    def get_stratified_negatives(self, query_embedding: torch.Tensor, exclude_path: str = None) -> List[torch.Tensor]:
        """
        Sample negatives from all difficulty tiers.

        Args:
            query_embedding: Query speaker embedding (D,)
            exclude_path: Path to exclude (e.g., self)

        Returns:
            List of negative speaker embeddings
        """
        # Find the query speaker's bin
        # For simplicity, we'll use FAISS search and bin results dynamically
        # (Pre-computed bins would require mapping from embedding to file path)

        # Search for many candidates
        k = min(500, len(self.faiss_index.file_paths))
        query_np = query_embedding.detach().cpu().numpy().reshape(1, -1)
        query_np = query_np / (np.linalg.norm(query_np) + 1e-8).astype('float32')

        similarities, indices = self.faiss_index.index.search(query_np, k=k)

        # Bin by similarity
        easy_candidates = []
        medium_candidates = []
        hard_candidates = []

        for sim, idx in zip(similarities[0], indices[0]):
            if idx >= len(self.faiss_index.file_paths):
                continue

            path = self.faiss_index.file_paths[idx]

            # Exclude self
            if exclude_path and path == exclude_path:
                continue

            # Must be in cache
            if path not in self.embedding_cache:
                continue

            if sim < self.threshold_easy_medium:
                easy_candidates.append((path, sim))
            elif sim < self.threshold_medium_hard:
                medium_candidates.append((path, sim))
            elif sim < self.threshold_hard_same:
                hard_candidates.append((path, sim))

        # Calculate how many from each tier
        n_easy = int(self.num_negatives * self.ratio_easy)
        n_medium = int(self.num_negatives * self.ratio_medium)
        n_hard = self.num_negatives - n_easy - n_medium  # Remainder goes to hard

        # Sample from each tier
        selected = []

        # Easy
        if len(easy_candidates) > 0:
            sampled = np.random.choice(len(easy_candidates),
                                      min(n_easy, len(easy_candidates)),
                                      replace=False)
            selected.extend([easy_candidates[i][0] for i in sampled])

        # Medium
        if len(medium_candidates) > 0:
            sampled = np.random.choice(len(medium_candidates),
                                      min(n_medium, len(medium_candidates)),
                                      replace=False)
            selected.extend([medium_candidates[i][0] for i in sampled])

        # Hard
        if len(hard_candidates) > 0:
            sampled = np.random.choice(len(hard_candidates),
                                      min(n_hard, len(hard_candidates)),
                                      replace=False)
            selected.extend([hard_candidates[i][0] for i in sampled])

        # If we don't have enough, fill with random from remaining
        if len(selected) < self.num_negatives:
            all_candidates = easy_candidates + medium_candidates + hard_candidates
            remaining_paths = [c[0] for c in all_candidates if c[0] not in selected]

            if len(remaining_paths) > 0:
                needed = self.num_negatives - len(selected)
                sampled = np.random.choice(len(remaining_paths),
                                          min(needed, len(remaining_paths)),
                                          replace=False)
                selected.extend([remaining_paths[i] for i in sampled])

        # Return embeddings
        negative_embs = []
        for path in selected[:self.num_negatives]:
            emb = self.embedding_cache[path]
            negative_embs.append(emb)

        return negative_embs


def get_stratified_negatives_legacy(model, query_embedding: torch.Tensor,
                                    faiss_index, embedding_cache: torch.Tensor,
                                    config: dict) -> List[torch.Tensor]:
    """
    Legacy version: Stratified sampling without pre-computed bins.

    Simpler but slower during training. Useful for testing without pre-computation.
    """
    num_negatives = config.get('max_negatives', 6)

    # Search for candidates
    k = min(500, len(faiss_index.file_paths))
    query_np = query_embedding.detach().cpu().numpy().reshape(1, -1)
    query_np = query_np / (np.linalg.norm(query_np) + 1e-8).astype('float32')

    similarities, indices = faiss_index.index.search(query_np, k=k)

    # Tier thresholds
    threshold_easy_medium = config.get('threshold_easy_medium', 0.3)
    threshold_medium_hard = config.get('threshold_medium_hard', 0.6)
    threshold_hard_same = config.get('same_speaker_threshold', 0.85)

    # Ratios
    ratio_easy = config.get('neg_ratio_easy', 0.3)
    ratio_medium = config.get('neg_ratio_medium', 0.4)
    ratio_hard = config.get('neg_ratio_hard', 0.3)

    # Bin by similarity
    easy_candidates = []
    medium_candidates = []
    hard_candidates = []

    for sim, idx in zip(similarities[0], indices[0]):
        if idx >= len(faiss_index.file_paths):
            continue

        path = faiss_index.file_paths[idx]
        if path not in embedding_cache:
            continue

        if sim < threshold_easy_medium:
            easy_candidates.append((path, sim))
        elif sim < threshold_medium_hard:
            medium_candidates.append((path, sim))
        elif sim < threshold_hard_same:
            hard_candidates.append((path, sim))

    # Calculate allocations
    n_easy = int(num_negatives * ratio_easy)
    n_medium = int(num_negatives * ratio_medium)
    n_hard = num_negatives - n_easy - n_medium

    # Sample
    selected = []

    if len(easy_candidates) > 0:
        count = min(n_easy, len(easy_candidates))
        sampled = np.random.choice(len(easy_candidates), count, replace=False)
        selected.extend([easy_candidates[i][0] for i in sampled])

    if len(medium_candidates) > 0:
        count = min(n_medium, len(medium_candidates))
        sampled = np.random.choice(len(medium_candidates), count, replace=False)
        selected.extend([medium_candidates[i][0] for i in sampled])

    if len(hard_candidates) > 0:
        count = min(n_hard, len(hard_candidates))
        sampled = np.random.choice(len(hard_candidates), count, replace=False)
        selected.extend([hard_candidates[i][0] for i in sampled])

    # Fill remainder
    if len(selected) < num_negatives:
        all_candidates = easy_candidates + medium_candidates + hard_candidates
        remaining = [c[0] for c in all_candidates if c[0] not in selected]
        if remaining:
            needed = num_negatives - len(selected)
            sampled = np.random.choice(len(remaining), min(needed, len(remaining)), replace=False)
            selected.extend([remaining[i] for i in sampled])

    # Return embeddings
    negative_embs = []
    for path in selected[:num_negatives]:
        negative_embs.append(embedding_cache[path])

    return negative_embs

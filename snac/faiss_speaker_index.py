#!/usr/bin/env python3
"""
FAISS-based Speaker Embedding Index for Dynamic Hard Negative Mining.

Uses FAISS for fast similarity search to find hard negatives during training.

Key advantages:
1. No speaker labels needed - treats each audio file independently
2. Dynamic similarity search - finds hard negatives based on actual embedding similarity
3. Fast - FAISS provides millisecond-level search over 100K+ embeddings
4. Accurate - doesn't rely on potentially incorrect speaker groupings
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import faiss
import torchaudio

try:
    from snac import SNACWithSpeakerConditioning
except ImportError:
    SNACWithSpeakerConditioning = None


class FaissSpeakerIndex:
    """
    FAISS-based speaker embedding index for fast hard negative mining.

    Uses FAISS IndexFlatIP (inner product) for fast similarity search.
    """

    def __init__(self, model, device='cuda'):
        """
        Initialize the index.

        Args:
            model: SNAC model with speaker encoder (ERes2NetV2)
            device: torch device
        """
        self.model = model
        self.device = device

        # FAISS index (will be loaded or built)
        self.index = None
        self.embeddings = None  # numpy array [N, D]
        self.file_paths = []  # list of file paths

        # Get base model (handle DDP wrapper)
        if hasattr(model, 'module'):
            self.model_base = model.module
        else:
            self.model_base = model

    @torch.no_grad()
    def extract_speaker_embedding(self, audio_path):
        """
        Extract speaker embedding from a single audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            numpy array of shape [512] (normalized)
        """
        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 24kHz
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            waveform = resampler(waveform)

        # Take first 2 seconds (48000 samples)
        max_samples = 2 * 24000
        if waveform.shape[-1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[-1] < 24000:
            # Pad if too short
            padding = 24000 - waveform.shape[-1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)

        # Extract embedding using ERes2NetV2
        waveform = waveform.to(self.device)
        embedding = self.model_base.speaker_encoder(waveform.unsqueeze(0))

        # Normalize to unit length (important for inner product search)
        embedding = embedding / (torch.norm(embedding) + 1e-8)

        return embedding.squeeze(0).cpu().numpy()

    def build_from_audio_list(self, audio_files, batch_size=64, desc="Extracting embeddings"):
        """
        Build index from a list of audio files.

        Args:
            audio_files: List of audio file paths
            batch_size: Batch size for embedding extraction
            desc: Description for progress bar
        """
        print(f"[FaissSpeakerIndex] Building index from {len(audio_files)} files")

        embeddings_list = []
        valid_files = []

        # Extract embeddings
        for audio_path in tqdm(audio_files, desc=desc):
            try:
                emb = self.extract_speaker_embedding(audio_path)
                embeddings_list.append(emb)
                valid_files.append(str(audio_path))
            except Exception as e:
                # Skip failed files
                continue

        # Stack embeddings
        self.embeddings = np.vstack(embeddings_list).astype('float32')
        self.file_paths = valid_files

        print(f"Extracted {len(self.embeddings)} embeddings")

        # Build FAISS index
        print("Building FAISS index...")
        embedding_dim = self.embeddings.shape[1]

        # Use IndexFlatIP (inner product) - assumes normalized vectors
        self.index = faiss.IndexFlatIP(embedding_dim)

        # Add embeddings to index
        self.index.add(self.embeddings)

        print(f"FAISS index built: {self.index.ntotal} vectors, dim={embedding_dim}")

    def save(self, index_path, paths_path=None):
        """
        Save FAISS index and file paths to disk.

        Args:
            index_path: Where to save the FAISS index
            paths_path: Where to save the file paths (optional)
        """
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        print(f"Saved FAISS index to: {index_path}")

        # Save file paths
        if paths_path is None:
            paths_path = index_path.with_suffix('.json')

        with open(paths_path, 'w') as f:
            json.dump(self.file_paths, f)

        print(f"File paths saved to: {paths_path}")

    def load(self, index_path, paths_path=None):
        """
        Load FAISS index and file paths from disk.

        Args:
            index_path: Path to FAISS index file
            paths_path: Path to file paths JSON (optional)
        """
        index_path = Path(index_path)

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        print(f"Loaded FAISS index from: {index_path}")
        print(f"  Total embeddings: {self.index.ntotal}")

        # Load file paths
        if paths_path is None:
            paths_path = index_path.with_suffix('.json')

        with open(paths_path, 'r') as f:
            self.file_paths = json.load(f)

        print(f"Loaded {len(self.file_paths)} file paths")

    def search(self, query_embedding, k=10, exclude_indices=None):
        """
        Search for nearest neighbors.

        Args:
            query_embedding: Query embedding (numpy array or torch tensor)
            k: Number of neighbors to return
            exclude_indices: Indices to exclude from results (e.g., self)

        Returns:
            - distances: Similarity scores [k]
            - indices: Indices in file_paths [k]
            - file_paths: Actual file paths [k]
        """
        # Convert to numpy if needed
        if torch.is_tensor(query_embedding):
            query_embedding = query_embedding.cpu().numpy()

        # Ensure 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_embedding = query_embedding.astype('float32')

        # Search
        distances, indices = self.index.search(query_embedding, k)

        # Filter out excluded indices
        if exclude_indices is not None:
            mask = np.ones(k, dtype=bool)
            for i, idx in enumerate(indices[0]):
                if idx in exclude_indices:
                    mask[i] = False
            distances = distances[mask][:k]
            indices = indices[mask][:k]

        # Get file paths
        file_paths = [self.file_paths[idx] for idx in indices[0]]

        return distances[0], indices[0], file_paths

    def find_hard_negatives(self, query_embedding, num_negatives=6, exclude_file_path=None):
        """
        Find hard negatives for a query embedding.

        Args:
            query_embedding: Query embedding (numpy array or torch tensor)
            num_negatives: Number of hard negatives to return
            exclude_file_path: File path to exclude (e.g., the anchor itself)

        Returns:
            - List of hard negative file paths
        """
        # Get index to exclude
        exclude_indices = None
        if exclude_file_path is not None and exclude_file_path in self.file_paths:
            exclude_indices = [self.file_paths.index(exclude_file_path)]

        # Search with extra results to account for exclusions
        k = num_negatives + (1 if exclude_indices is not None else 0)

        distances, indices, file_paths = self.search(
            query_embedding, k=k, exclude_indices=exclude_indices
        )

        # Return top num_negatives
        return file_paths[:num_negatives]

    @torch.no_grad()
    def find_hard_negatives_from_audio(self, audio_path, num_negatives=6):
        """
        Find hard negatives for an audio file.

        Args:
            audio_path: Path to anchor audio file
            num_negatives: Number of hard negatives to return

        Returns:
            - List of hard negative file paths
        """
        # Extract query embedding
        query_emb = self.extract_speaker_embedding(audio_path)

        # Find hard negatives
        return self.find_hard_negatives(query_emb, num_negatives=num_negatives, exclude_file_path=audio_path)


def build_faiss_index(model, audio_files, output_path, device='cuda', batch_size=64):
    """
    Build and save FAISS speaker embedding index.

    Args:
        model: SNAC model with speaker encoder
        audio_files: List of audio file paths
        output_path: Where to save the FAISS index
        device: torch device
        batch_size: Batch size for embedding extraction

    Returns:
        - FaissSpeakerIndex instance
    """
    # Create index
    index = FaissSpeakerIndex(model, device)

    # Build from audio files
    index.build_from_audio_list(audio_files, batch_size=batch_size)

    # Save
    index.save(output_path)

    return index


if __name__ == "__main__":
    print("FAISS Speaker Embedding Index")
    print("=" * 70)
    print("\nThis module provides:")
    print("1. FaissSpeakerIndex: Build and query FAISS-based speaker index")
    print("2. build_faiss_index(): Convenience function to build index")
    print("\nUsage:")
    print("  from snac.faiss_speaker_index import build_faiss_index")
    print("  index = build_faiss_index(")
    print("      model=model,")
    print("      audio_files=list_of_wav_files,")
    print("      output_path='pretrained_models/speaker_faiss.index',")
    print("  )")
    print("\nThen use for hard negative mining:")
    print("  hard_negs = index.find_hard_negatives_from_audio('anchor.wav', num_negatives=6)")

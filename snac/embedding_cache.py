#!/usr/bin/env python3
"""
Optimized Embedding Cache with Memory-Mapping.

Features:
1. Instant loading - memory-mapped files, no copy overhead
2. Low memory footprint - lazy loading, only accessed pages are loaded
3. Optional compression - PCA to reduce 512 dims → 128 dims
4. Fast lookup - dictionary mapping file path → embedding index

Usage:
    from snac.embedding_cache import OptimizedEmbeddingCache

    cache = OptimizedEmbeddingCache.load('pretrained_models/embeddings_cache.npy')
    embedding = cache.get('path/to/audio.wav')
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, List
from sklearn.decomposition import PCA


class OptimizedEmbeddingCache:
    """
    Memory-mapped embedding cache for fast lookup and low memory overhead.

    Uses numpy memory-mapping for instant loading without copying data into RAM.
    """

    def __init__(self, embeddings: np.ndarray, file_paths: List[str]):
        """
        Initialize cache with embeddings and file paths.

        Args:
            embeddings: Embedding array (N, D) as numpy array
            file_paths: List of file paths corresponding to embeddings
        """
        self.embeddings = embeddings
        self.file_paths = file_paths
        self.embedding_dim = embeddings.shape[1]

        # Build lookup dict: file_path -> index
        self.path_to_idx = {path: idx for idx, path in enumerate(file_paths)}

        print(f"Loaded embedding cache:")
        print(f"  Embeddings: {len(embeddings)}")
        print(f"  Dimension: {self.embedding_dim}")
        print(f"  Memory: {embeddings.nbytes / 1024 / 1024:.1f} MB (memory-mapped)")

    @classmethod
    def load(cls, cache_path: str, mmap_mode: str = 'r'):
        """
        Load embedding cache from disk using memory-mapping.

        Args:
            cache_path: Path to .npy file (without extension)
            mmap_mode: Memory-mapping mode ('r' for read-only)

        Returns:
            OptimizedEmbeddingCache instance
        """
        cache_path = Path(cache_path)

        # Load embeddings (memory-mapped)
        npy_path = cache_path.with_suffix('.npy')
        print(f"Loading embeddings from: {npy_path}")

        embeddings = np.load(npy_path, mmap_mode=mmap_mode)

        # Load file paths
        paths_path = cache_path.with_suffix('.json')
        with open(paths_path, 'r') as f:
            file_paths = json.load(f)

        return cls(embeddings, file_paths)

    def get(self, file_path: str) -> Optional[torch.Tensor]:
        """
        Get embedding for a specific file path.

        Args:
            file_path: Audio file path

        Returns:
            Embedding tensor (D,) or None if not found
        """
        idx = self.path_to_idx.get(file_path)
        if idx is None:
            return None

        # Get embedding (this triggers disk page load on first access)
        emb = self.embeddings[idx]

        # Convert to torch tensor
        return torch.from_numpy(emb).float()

    def get_batch(self, file_paths: List[str]) -> torch.Tensor:
        """
        Get embeddings for multiple file paths at once.

        More efficient than calling get() multiple times.

        Args:
            file_paths: List of file paths

        Returns:
            Embeddings tensor (N, D)
        """
        indices = []
        valid_paths = []

        for path in file_paths:
            idx = self.path_to_idx.get(path)
            if idx is not None:
                indices.append(idx)
                valid_paths.append(path)

        if len(indices) == 0:
            return torch.zeros(len(file_paths), self.embedding_dim)

        # Get embeddings (triggers disk page load for new pages)
        embs = self.embeddings[indices]

        # Convert to torch tensor
        return torch.from_numpy(embs).float()

    def search_similar(self, query_embedding: torch.Tensor, k: int = 10,
                      exclude_paths: Optional[List[str]] = None) -> tuple:
        """
        Search for k most similar embeddings.

        Args:
            query_embedding: Query embedding (D,)
            k: Number of neighbors to return
            exclude_paths: Paths to exclude from results

        Returns:
            (similarities, indices, file_paths)
        """
        # Convert to numpy
        if torch.is_tensor(query_embedding):
            query_np = query_embedding.detach().cpu().numpy()
        else:
            query_np = query_embedding

        # Ensure 2D and normalized
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)
        query_np = query_np / (np.linalg.norm(query_np) + 1e-8)

        # Compute similarities
        # Dot product with normalized embeddings = cosine similarity
        similarities = np.dot(self.embeddings, query_np.T).flatten()

        # Get top-k
        k = min(k, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:k]

        # Filter excluded paths
        if exclude_paths:
            exclude_set = set(exclude_paths)
            filtered = []
            for idx in top_indices:
                if idx < len(self.file_paths):
                    path = self.file_paths[idx]
                    if path not in exclude_set:
                        filtered.append((similarities[idx], idx, path))
                if len(filtered) >= k:
                    break

            if filtered:
                similarities_result = [s[0] for s in filtered]
                indices_result = [s[1] for s in filtered]
                paths_result = [s[2] for s in filtered]
                return similarities_result, indices_result, paths_result

        # Return results
        similarities_result = similarities[top_indices]
        paths_result = [self.file_paths[idx] for idx in top_indices if idx < len(self.file_paths)]

        return similarities_result, top_indices.tolist(), paths_result

    def compress(self, n_components: int = 128, output_path: Optional[str] = None):
        """
        Compress embeddings using PCA to reduce memory footprint.

        Args:
            n_components: Target dimension (default 128)
            output_path: Optional path to save compressed cache

        Returns:
            New OptimizedEmbeddingCache with compressed embeddings
        """
        print(f"Compressing embeddings: {self.embedding_dim} → {n_components} dims")

        # Fit PCA
        pca = PCA(n_components=n_components)
        embeddings_compressed = pca.fit_transform(self.embeddings)

        # Save PCA transform
        if output_path:
            pca_path = Path(output_path).with_suffix('.pca.pkl')
            import pickle
            with open(pca_path, 'wb') as f:
                pickle.dump(pca, f)
            print(f"Saved PCA transform to: {pca_path}")

        print(f"Compression ratio: {self.embedding_dim / n_components:.2f}x")
        print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

        return OptimizedEmbeddingCache(embeddings_compressed, self.file_paths)


def build_cache_from_faiss(
    faiss_index_path: str,
    output_path: str
):
    """
    Build embedding cache from existing FAISS index.

    Useful if you already have FAISS index and want to create
    the optimized cache without re-extracting embeddings.

    Args:
        faiss_index_path: Path to FAISS index
        output_path: Where to save the cache
    """
    import json

    print(f"Building cache from FAISS index: {faiss_index_path}")

    # Load file paths from FAISS
    paths_path = Path(faiss_index_path).with_suffix('.json')
    with open(paths_path, 'r') as f:
        file_paths = json.load(f)

    # Load FAISS index
    import faiss
    index = faiss.read_index(str(Path(faiss_index_path)))
    print(f"FAISS index: {index.ntotal} vectors")

    # The FAISS index doesn't store the raw embeddings in a recoverable way
    # (IndexFlatIP only stores the vectors in an optimized structure)

    # So we need to either:
    # 1. Have stored the embeddings separately when building FAISS
    # 2. Re-extract them (but that defeats the purpose)

    print("Note: To use this function, you need to have saved the raw embeddings")
    print("when building the FAISS index. The FAISS index itself doesn't")
    print("store embeddings in a recoverable format.")

    print("\nRecommended: Use scripts/build_embedding_cache.py to build")
    print("the cache directly from audio files, then use that cache")
    print("for both FAISS and OptimizedEmbeddingCache.")


if __name__ == '__main__':
    print("Optimized Embedding Cache")
    print("=" * 70)
    print("\nThis module provides:")
    print("  - OptimizedEmbeddingCache: Memory-mapped embedding cache")
    print("  - Instant loading, low memory overhead")
    print("\nUsage:")
    print("  from snac.embedding_cache import OptimizedEmbeddingCache")
    print("  cache = OptimizedEmbeddingCache.load('pretrained_models/embeddings_cache.npy')")
    print("  emb = cache.get('path/to/audio.wav')")

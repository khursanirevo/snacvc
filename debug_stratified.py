#!/usr/bin/env python3
"""Debug stratified sampling performance."""

import time
import torch
import numpy as np
import faiss
import json
from pathlib import Path
from snac import SNACWithSpeakerConditioning
from snac.embedding_cache import OptimizedEmbeddingCache

def benchmark_stratified():
    print("Loading model...")
    model = SNACWithSpeakerConditioning.from_pretrained_base(
        repo_id="hubertsiuzdak/snac_24khz",
        speaker_emb_dim=512,
        speaker_encoder_type='eres2net',
        freeze_base=True,
    ).to('cuda:0')
    model.eval()

    print("Loading FAISS index...")
    index_path = Path("pretrained_models/speaker_faiss.index")
    faiss_idx = faiss.read_index(str(index_path))

    # Load file paths
    import json
    paths_path = index_path.with_suffix('.json')
    with open(paths_path) as f:
        file_paths = json.load(f)

    # Create a simple wrapper object
    class SimpleFAISS:
        def __init__(self, index, paths):
            self.index = index
            self.file_paths = paths

    faiss_index = SimpleFAISS(faiss_idx, file_paths)

    print("Loading embedding cache...")
    embedding_cache = OptimizedEmbeddingCache.load("pretrained_models/embeddings_cache.npy")

    # Load config
    with open("configs/phase4_gan_benchmark_stratified.json") as f:
        config = json.load(f)

    print("\n" + "="*70)
    print("BENCHMARK: Stratified Sampling")
    print("="*70)

    # Test with a random embedding
    query_emb = torch.randn(512).cuda()

    # Test 1: FAISS search only
    print("\n1. Testing FAISS search (k=500)...")
    k = min(500, len(faiss_index.file_paths))
    start = time.time()
    query_np = query_emb.detach().cpu().numpy().reshape(1, -1)
    query_np = query_np / (np.linalg.norm(query_np) + 1e-8).astype('float32')
    similarities, indices = faiss_index.index.search(query_np, k=k)
    faiss_time = time.time() - start
    print(f"   Time: {faiss_time*1000:.2f}ms")
    print(f"   Found {len(indices[0])} results")

    # Test 2: Binning logic
    print("\n2. Testing binning logic...")
    start = time.time()
    threshold_easy_medium = config.get('threshold_easy_medium', 0.3)
    threshold_medium_hard = config.get('threshold_medium_hard', 0.6)
    threshold_hard_same = config.get('same_speaker_threshold', 0.85)

    easy_candidates = []
    medium_candidates = []
    hard_candidates = []

    for sim, idx in zip(similarities[0], indices[0]):
        if idx >= len(faiss_index.file_paths):
            continue
        path = faiss_index.file_paths[idx]
        if path not in embedding_cache.path_to_idx:
            continue

        if sim < threshold_easy_medium:
            easy_candidates.append((path, sim))
        elif sim < threshold_medium_hard:
            medium_candidates.append((path, sim))
        elif sim < threshold_hard_same:
            hard_candidates.append((path, sim))

    binning_time = time.time() - start
    print(f"   Time: {binning_time*1000:.2f}ms")
    print(f"   Easy: {len(easy_candidates)}, Medium: {len(medium_candidates)}, Hard: {len(hard_candidates)}")

    # Test 3: Embedding retrieval
    print("\n3. Testing embedding retrieval...")
    start = time.time()
    negative_embs = []
    for path, _ in (easy_candidates[:2] + medium_candidates[:2] + hard_candidates[:2]):
        emb = embedding_cache.get(path)
        if emb is not None:
            negative_embs.append(emb)
    retrieval_time = time.time() - start
    print(f"   Time: {retrieval_time*1000:.2f}ms")
    print(f"   Retrieved {len(negative_embs)} embeddings")

    # Test 4: Full stratified function
    print("\n4. Testing full stratified function...")
    from snac.stratified_hard_negatives import get_stratified_negatives_legacy

    start = time.time()
    result = get_stratified_negatives_legacy(model, query_emb, faiss_index, embedding_cache, config)
    full_time = time.time() - start
    print(f"   Time: {full_time*1000:.2f}ms")
    print(f"   Returned {len(result)} embeddings")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"FAISS search:  {faiss_time*1000:.2f}ms")
    print(f"Binning:       {binning_time*1000:.2f}ms")
    print(f"Retrieval:     {retrieval_time*1000:.2f}ms")
    print(f"Full function: {full_time*1000:.2f}ms")
    print(f"\nPer-sample overhead: {full_time*1000:.2f}ms")
    print(f"Batch size 6 = {full_time*6*1000:.2f}ms per batch")

if __name__ == "__main__":
    benchmark_stratified()

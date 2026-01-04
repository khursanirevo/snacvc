#!/usr/bin/env python3
"""
Analyze speaker embeddings from trained model.

Tools:
- Embedding space visualization (PCA/t-SNE)
- Speaker clustering analysis
- Same-speaker vs different-speaker similarity distribution
- Embedding quality metrics
"""

import sys
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
import json

sys.path.insert(0, str(Path(__file__).dirname().resolve())

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from snac import SNACWithSpeakerConditioning


def extract_embeddings_from_dataloader(model, dataloader, device, max_samples=1000):
    """
    Extract speaker embeddings from dataloader.

    Returns:
        - embeddings: (N, 512) numpy array
        - speaker_ids: (N,) numpy array
        - file_paths: (N,) list of file paths
    """
    model.eval()

    all_embeddings = []
    all_speaker_ids = []
    all_file_paths = []

    sample_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= max_samples:
                break

            audio = batch['audio'].to(device)
            speaker_id = batch.get('speaker_id', [f"speaker_{batch_idx}"] * len(audio))
            file_path = batch.get('file_path', [f"sample_{batch_idx}_{i}" for i in range(len(audio))])

            # Get model_base (handle DDP)
            if hasattr(model, 'module'):
                model_base = model.module
            else:
                model_base = model

            # Extract embeddings
            audio_input = audio.unsqueeze(1) if audio.dim() == 2 else audio
            embeddings = model_base.extract_speaker_embedding(audio_input)

            all_embeddings.append(embeddings.cpu())
            all_speaker_ids.extend(speaker_id)
            all_file_paths.extend(file_path)

            sample_count += len(audio)

    # Concatenate
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    speaker_ids = np.array(all_speaker_ids)

    return embeddings, speaker_ids, all_file_paths


def compute_similarity_statistics(embeddings, speaker_ids):
    """
    Compute same-speaker and different-speaker similarity statistics.

    Returns:
        - same_speaker_sim: Array of similarities for same-speaker pairs
        - diff_speaker_sim: Array of similarities for different-speaker pairs
    """
    # Normalize embeddings
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    n = len(embeddings)
    same_speaker_sim = []
    diff_speaker_sim = []

    for i in range(n):
        for j in range(i+1, n):
            sim = np.dot(embeddings[i], embeddings[j])

            if speaker_ids[i] == speaker_ids[j]:
                same_speaker_sim.append(sim)
            else:
                diff_speaker_sim.append(sim)

    return np.array(same_speaker_sim), np.array(diff_speaker_sim)


def plot_similarity_distributions(same_speaker_sim, diff_speaker_sim, output_path):
    """Plot same-speaker vs different-speaker similarity distributions."""

    plt.figure(figsize=(10, 6))

    plt.hist(diff_speaker_sim, bins=50, alpha=0.5, label='Different Speaker', density=True)
    plt.hist(same_speaker_sim, bins=50, alpha=0.5, label='Same Speaker', density=True)

    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Speaker Similarity Distribution')
    plt.legend()
    plt.grid(alpha=0.3)

    # Add statistics
    same_mean = same_speaker_sim.mean()
    diff_mean = diff_speaker_sim.mean()

    plt.axvline(same_mean, color='blue', linestyle='--', alpha=0.7, label=f'Same Mean: {same_mean:.3f}')
    plt.axvline(diff_mean, color='orange', linestyle='--', alpha=0.7, label=f'Diff Mean: {diff_mean:.3f}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Similarity distribution plot saved to {output_path}")


def plot_embedding_space_2d(embeddings, speaker_ids, output_path, method='pca'):
    """
    Visualize embedding space in 2D using PCA or t-SNE.

    Args:
        embeddings: (N, D) numpy array
        speaker_ids: (N,) numpy array
        output_path: Where to save the plot
        method: 'pca' or 'tsne'
    """
    # Reduce dimensionality
    if method == 'pca':
        reducer = PCA(n_components=2)
        embeddings_2d = reducer.fit_transform(embeddings)
        print(f"PCA explained variance ratio: {reducer.explained_variance_ratio_}")
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Plot
    plt.figure(figsize=(12, 8))

    # Get unique speakers
    unique_speakers = np.unique(speaker_ids)

    # Color mapping
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(unique_speakers))))

    for idx, speaker in enumerate(unique_speakers):
        mask = speaker_ids == speaker
        label = speaker if idx < 10 else None  # Only label first 10 to avoid clutter
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                   c=[colors[idx % 20]], label=label, alpha=0.6, s=50)

    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(f'Speaker Embedding Space ({method.upper()})')

    if len(unique_speakers) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Embedding space plot saved to {output_path}")


def compute_clustering_metrics(embeddings, speaker_ids):
    """
    Compute clustering quality metrics.

    Returns:
        - metrics: Dict of clustering metrics
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    # Normalize embeddings
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    metrics = {}

    try:
        metrics['silhouette_score'] = silhouette_score(embeddings, speaker_ids)
        metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings, speaker_ids)
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, speaker_ids)
    except Exception as e:
        print(f"Warning: Could not compute clustering metrics: {e}")
        metrics['silhouette_score'] = None
        metrics['davies_bouldin_score'] = None
        metrics['calinski_harabasz_score'] = None

    return metrics


def analyze_embeddings(model, dataloader, device, output_dir,
                       max_samples=1000, visualize=True):
    """
    Complete embedding analysis pipeline.

    Args:
        model: Trained SNAC model
        dataloader: Data loader
        device: torch device
        output_dir: Where to save results
        max_samples: Maximum samples to analyze
        visualize: Whether to create visualizations

    Returns:
        - analysis_results: Dict of all analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("SPEAKER EMBEDDING ANALYSIS")
    print("="*70)

    # 1. Extract embeddings
    print("\n[1/5] Extracting embeddings...")
    embeddings, speaker_ids, file_paths = extract_embeddings_from_dataloader(
        model, dataloader, device, max_samples
    )
    print(f"Extracted {len(embeddings)} embeddings from {len(np.unique(speaker_ids))} speakers")

    # 2. Similarity statistics
    print("\n[2/5] Computing similarity statistics...")
    same_speaker_sim, diff_speaker_sim = compute_similarity_statistics(embeddings, speaker_ids)

    print(f"Same-speaker similarity: {same_speaker_sim.mean():.4f} ± {same_speaker_sim.std():.4f}")
    print(f"Different-speaker similarity: {diff_speaker_sim.mean():.4f} ± {diff_speaker_sim.std():.4f}")
    print(f"Separation (same - diff): {(same_speaker_sim.mean() - diff_speaker_sim.mean()):.4f}")

    # 3. Clustering metrics
    print("\n[3/5] Computing clustering metrics...")
    clustering_metrics = compute_clustering_metrics(embeddings, speaker_ids)
    for key, value in clustering_metrics.items():
        if value is not None:
            print(f"  {key}: {value:.4f}")

    # 4. Visualizations
    if visualize:
        print("\n[4/5] Creating visualizations...")

        # Similarity distributions
        plot_similarity_distributions(
            same_speaker_sim, diff_speaker_sim,
            output_dir / "similarity_distribution.png"
        )

        # PCA visualization
        plot_embedding_space_2d(
            embeddings, speaker_ids,
            output_dir / "embedding_space_pca.png",
            method='pca'
        )

        # t-SNE visualization (if enough samples)
        if len(embeddings) >= 50:
            plot_embedding_space_2d(
                embeddings, speaker_ids,
                output_dir / "embedding_space_tsne.png",
                method='tsne'
            )

    # 5. Save results
    print("\n[5/5] Saving results...")
    results = {
        'num_samples': len(embeddings),
        'num_speakers': len(np.unique(speaker_ids)),
        'same_speaker_similarity': {
            'mean': float(same_speaker_sim.mean()),
            'std': float(same_speaker_sim.std()),
            'min': float(same_speaker_sim.min()),
            'max': float(same_speaker_sim.max()),
        },
        'different_speaker_similarity': {
            'mean': float(diff_speaker_sim.mean()),
            'std': float(diff_speaker_sim.std()),
            'min': float(diff_speaker_sim.min()),
            'max': float(diff_speaker_sim.max()),
        },
        'separation': float(same_speaker_sim.mean() - diff_speaker_sim.mean()),
        'clustering_metrics': {
            k: float(v) if v is not None else None
            for k, v in clustering_metrics.items()
        }
    }

    with open(output_dir / "analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAnalysis complete! Results saved to {output_dir}")
    print("="*70)

    return results


if __name__ == "__main__":
    print("Speaker Embedding Analysis Tool")
    print("Usage: analyze_embeddings(model, dataloader, device, output_dir)")

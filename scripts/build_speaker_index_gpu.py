#!/usr/bin/env python3
"""
GPU-accelerated speaker embedding index builder.

Processes speakers in batches on GPU for much faster index building.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import json
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from snac import SNACWithSpeakerConditioning


def build_speaker_index_gpu(model, dataset_root, output_path, n_clusters=50,
                            device='cuda:1', max_samples_per_speaker=50,
                            batch_size=32):
    """
    Build speaker embedding index using GPU batch processing.

    Args:
        model: SNAC model
        dataset_root: Path to training data
        output_path: Where to save the index
        n_clusters: Number of clusters
        device: torch device
        max_samples_per_speaker: Max samples to average per speaker
        batch_size: Batch size for GPU processing

    Returns:
        - SpeakerEmbeddingIndex instance
    """
    from collections import defaultdict
    import numpy as np
    from sklearn.cluster import KMeans

    dataset_root = Path(dataset_root)
    import re

    print("="*70)
    print("Building Speaker Embedding Index (GPU Accelerated)")
    print("="*70)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Max samples per speaker: {max_samples_per_speaker}")
    print("="*70)

    # Get model_base
    if hasattr(model, 'module'):
        model_base = model.module
    else:
        model_base = model

    model.eval()

    # Find all audio files grouped by speaker
    print("\n[1/3] Grouping audio files by speaker...")
    speaker_to_files = defaultdict(list)

    audio_files = list(dataset_root.glob("**/*.wav"))
    print(f"Found {len(audio_files)} audio files")

    for audio_path in tqdm(audio_files, desc="Grouping by speaker"):
        filename = audio_path.stem

        # Extract UUID as speaker ID
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

    # Extract embeddings in batches
    print("\n[2/3] Extracting speaker embeddings (GPU batched)...")

    speaker_embeddings = {}
    speaker_files_final = {}

    # Process speakers in batches
    speaker_ids = list(speaker_to_files.keys())

    for start_idx in tqdm(range(0, len(speaker_ids), batch_size), desc="Speaker batches"):
        end_idx = min(start_idx + batch_size, len(speaker_ids))
        batch_speaker_ids = speaker_ids[start_idx:end_idx]

        # Prepare batch: collect audio files for each speaker
        batch_audio_files = []
        batch_file_to_speaker = []
        batch_speaker_indices = []

        for speaker_idx, speaker_id in enumerate(batch_speaker_ids):
            files = speaker_to_files[speaker_id]

            # Limit samples
            if len(files) > max_samples_per_speaker:
                files = np.random.choice(files, max_samples_per_speaker, replace=False).tolist()

            # Add to batch
            start_file_idx = len(batch_audio_files)
            for file_path in files:
                batch_audio_files.append(file_path)
                batch_file_to_speaker.append(speaker_idx)

            batch_speaker_indices.append((start_file_idx, len(batch_audio_files)))

        # Load all audio files in batch
        try:
            import torchaudio

            batch_waveforms = []
            batch_sr = None

            for audio_path in batch_audio_files:
                waveform, sr = torchaudio.load(audio_path)

                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # Resample to 24kHz
                if sr != 24000:
                    if batch_sr is None:
                        resampler = torchaudio.transforms.Resample(sr, 24000)
                    waveform = resampler(waveform)

                # Take first 2 seconds
                max_samples = 2 * 24000
                if waveform.shape[-1] > max_samples:
                    waveform = waveform[:, :max_samples]
                elif waveform.shape[-1] < 24000:
                    padding = 24000 - waveform.shape[-1]
                    waveform = torch.nn.functional.pad(waveform, (0, padding))

                # Normalize
                waveform = waveform / (waveform.abs().max() + 1e-8)

                batch_waveforms.append(waveform)

            # Stack into batch
            batch_waveforms = torch.stack(batch_waveforms).to(device)  # (N, 1, T)

            # Extract embeddings in batch
            with torch.no_grad():
                batch_embeddings = model_base.extract_speaker_embedding(batch_waveforms)  # (N, 512)

            # Normalize embeddings
            batch_embeddings = batch_embeddings / (torch.norm(batch_embeddings, dim=1, keepdim=True) + 1e-8)

            # Average embeddings per speaker
            batch_embeddings = batch_embeddings.cpu()

            for speaker_id, (start_idx, end_idx) in zip(batch_speaker_ids, batch_speaker_indices):
                speaker_embs = batch_embeddings[start_idx:end_idx]  # (num_files, 512)
                speaker_emb = speaker_embs.mean(dim=0)  # (512,)

                # Renormalize
                speaker_emb = speaker_emb / (torch.norm(speaker_emb) + 1e-8)

                speaker_embeddings[speaker_id] = speaker_emb

                # Store files
                files = speaker_to_files[speaker_id]
                if len(files) > max_samples_per_speaker:
                    files = files[:max_samples_per_speaker]
                speaker_files_final[speaker_id] = [str(f) for f in files]

        except Exception as e:
            print(f"Error processing batch {start_idx}-{end_idx}: {e}")
            continue

    print(f"Extracted embeddings for {len(speaker_embeddings)} speakers")

    # Cluster speakers
    print("\n[3/3] Clustering speakers...")

    if len(speaker_embeddings) < n_clusters:
        n_clusters = max(2, len(speaker_embeddings) // 2)
        print(f"Adjusting clusters to {n_clusters}")

    # Get embedding matrix
    speaker_ids = list(speaker_embeddings.keys())
    embeddings = torch.stack([speaker_embeddings[sid] for sid in speaker_ids]).numpy()

    # Fit K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # Store mapping
    speaker_to_cluster = {}
    cluster_to_speakers = defaultdict(list)

    for speaker_id, cluster_id in zip(speaker_ids, cluster_labels):
        speaker_to_cluster[speaker_id] = int(cluster_id)
        cluster_to_speakers[int(cluster_id)].append(speaker_id)

    print(f"Clustering complete:")
    for cluster_id in sorted(cluster_to_speakers.keys())[:10]:
        n_speakers = len(cluster_to_speakers[cluster_id])
        print(f"  Cluster {cluster_id}: {n_speakers} speakers")

    if len(cluster_to_speakers) > 10:
        print(f"  ... and {len(cluster_to_speakers) - 10} more clusters")

    # Create index object
    from snac.speaker_embed_index import SpeakerEmbeddingIndex

    index = SpeakerEmbeddingIndex(model, device)
    index.speaker_embeddings = speaker_embeddings
    index.speaker_files = speaker_files_final
    index.speaker_to_cluster = speaker_to_cluster
    index.cluster_to_speakers = dict(cluster_to_speakers)

    # Save
    index.save(output_path)

    return index


def main():
    parser = argparse.ArgumentParser(description="Build speaker embedding index (GPU accelerated)")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/phase4_gan/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--dataset-root", type=str,
                        default="data/train_split",
                        help="Path to training data")
    parser.add_argument("--output", type=str,
                        default="pretrained_models/speaker_index.json",
                        help="Output path for speaker index")
    parser.add_argument("--n-clusters", type=int, default=50,
                        help="Number of speaker clusters")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Max samples per speaker for averaging")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for GPU processing")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: cuda:0 or cuda if available)")

    args = parser.parse_args()

    # Load model
    print("\nLoading model...")
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU 0 (first available GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']

            if list(model_state.keys())[0].startswith('module.'):
                model_state = {k.replace('module.', ''): v for k, v in model_state.items()}

            if 'config' in checkpoint:
                config = checkpoint['config']
                model = SNACWithSpeakerConditioning(
                    sampling_rate=config.get('sampling_rate', 24000),
                    speaker_emb_dim=config.get('speaker_emb_dim', 512),
                    speaker_encoder_type=config.get('speaker_encoder_type', 'eres2net'),
                ).to(device)
            else:
                model = SNACWithSpeakerConditioning.from_pretrained_base(
                    repo_id="hubertsiuzdak/snac_24khz",
                    speaker_emb_dim=512,
                    speaker_encoder_type='eres2net',
                ).to(device)

            model.load_state_dict(model_state)
        else:
            raise ValueError("Invalid checkpoint format")

        model.eval()
        print(f"✅ Model loaded from {args.checkpoint}")

    except FileNotFoundError:
        print(f"⚠️  Checkpoint not found: {args.checkpoint}")
        print("Loading pretrained model from HuggingFace instead...")

        model = SNACWithSpeakerConditioning.from_pretrained_base(
            repo_id="hubertsiuzdak/snac_24khz",
            speaker_emb_dim=512,
            speaker_encoder_type='eres2net',
        ).to(device)
        model.eval()

        print("✅ Loaded pretrained model")

    # Build index
    build_speaker_index_gpu(
        model=model,
        dataset_root=args.dataset_root,
        output_path=args.output,
        n_clusters=args.n_clusters,
        device=args.device,
        max_samples_per_speaker=args.max_samples,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()

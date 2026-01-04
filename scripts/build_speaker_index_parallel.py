#!/usr/bin/env python3
"""
Parallel speaker index builder using 2 GPUs.

Splits speakers between GPU 1 and GPU 2 for 2x speedup.
"""

import sys
import torch
import torch.multiprocessing as mp
from pathlib import Path
from collections import defaultdict
import numpy as np
import re
import torchaudio
import json
from tqdm import tqdm

sys.path.insert(0, '.')

from snac import SNACWithSpeakerConditioning
from sklearn.cluster import KMeans


def extract_speaker_embeddings_worker(gpu_id, speaker_ids, speaker_to_files, output_queue):
    """Worker function to extract embeddings on one GPU."""
    try:
        device = torch.device(f'cuda:{gpu_id}')

        # Load model on this GPU
        model = SNACWithSpeakerConditioning.from_pretrained_base(
            repo_id='hubertsiuzdak/snac_24khz',
            speaker_emb_dim=512,
            speaker_encoder_type='eres2net',
        ).to(device)

        model.eval()

        if hasattr(model, 'module'):
            model_base = model.module
        else:
            model_base = model

        print(f"[GPU {gpu_id}] Processing {len(speaker_ids)} speakers...")

        # Process speakers
        speaker_embeddings = {}
        batch_size = 64

        for start_idx in range(0, len(speaker_ids), batch_size):
            end_idx = min(start_idx + batch_size, len(speaker_ids))
            batch_speakers = speaker_ids[start_idx:end_idx]

            # Load audios (1 sample per speaker for speed)
            all_waveforms = []

            for speaker_id in batch_speakers:
                files = speaker_to_files[speaker_id][:1]  # Just 1 file per speaker

                for audio_path in files:
                    try:
                        waveform, sr = torchaudio.load(audio_path)

                        # Convert to mono
                        if waveform.shape[0] > 1:
                            waveform = waveform.mean(dim=0, keepdim=True)

                        # Resample
                        if sr != 24000:
                            resampler = torchaudio.transforms.Resample(sr, 24000)
                            waveform = resampler(waveform)

                        # Truncate/pad to 2 seconds
                        max_samples = 2 * 24000
                        if waveform.shape[-1] > max_samples:
                            waveform = waveform[:, :max_samples]
                        elif waveform.shape[-1] < 24000:
                            waveform = torch.nn.functional.pad(waveform, (0, 24000 - waveform.shape[-1]))

                        # Normalize
                        waveform = waveform / (waveform.abs().max() + 1e-8)

                        all_waveforms.append(waveform)
                        break
                    except Exception as e:
                        print(f"[GPU {gpu_id}] Error loading {audio_path}: {e}")
                        continue

            if len(all_waveforms) == 0:
                continue

            # Extract embeddings in batch
            try:
                with torch.no_grad():
                    batch = torch.stack(all_waveforms).to(device)
                    embs = model_base.extract_speaker_embedding(batch)
                    embs = embs / (torch.norm(embs, dim=1, keepdim=True) + 1e-8)
                    embs = embs.cpu()

                # Store
                for i, speaker_id in enumerate(batch_speakers):
                    speaker_embeddings[speaker_id] = embs[i]

            except Exception as e:
                print(f"[GPU {gpu_id}] Error extracting batch: {e}")
                continue

        print(f"[GPU {gpu_id}] Extracted {len(speaker_embeddings)} embeddings")
        output_queue.put((gpu_id, speaker_embeddings))

    except Exception as e:
        print(f"[GPU {gpu_id}] Worker failed: {e}")
        import traceback
        traceback.print_exc()
        output_queue.put((gpu_id, {}))


def build_speaker_index_parallel(dataset_root='data/train_split',
                                 output_path='pretrained_models/speaker_index.json',
                                 n_clusters=50,
                                 gpu_ids=[1, 2]):
    """
    Build speaker index using 2 GPUs in parallel.
    """
    print("="*70)
    print(f"Parallel Speaker Index Builder (GPUs {gpu_ids})")
    print("="*70)

    dataset_root = Path(dataset_root)

    # Find and group audio files
    print("\n[1/3] Grouping audio files by speaker...")
    audio_files = list(dataset_root.glob("**/*.wav"))
    print(f"Found {len(audio_files)} audio files")

    speaker_to_files = defaultdict(list)

    for audio_path in audio_files:
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

    # Split speakers between GPUs
    speaker_ids = list(speaker_to_files.keys())
    mid = len(speaker_ids) // 2

    gpu1_speakers = speaker_ids[:mid]
    gpu2_speakers = speaker_ids[mid:]

    print(f"\n[2/3] Extracting embeddings (parallel)...")
    print(f"  GPU {gpu_ids[0]}: {len(gpu1_speakers)} speakers")
    print(f"  GPU {gpu_ids[1]}: {len(gpu2_speakers)} speakers")

    # Start workers
    output_queue = mp.Queue()

    ctx = mp.get_context('spawn')
    process1 = ctx.Process(
        target=extract_speaker_embeddings_worker,
        args=(gpu_ids[0], gpu1_speakers, speaker_to_files, output_queue)
    )
    process2 = ctx.Process(
        target=extract_speaker_embeddings_worker,
        args=(gpu_ids[1], gpu2_speakers, speaker_to_files, output_queue)
    )

    process1.start()
    process2.start()

    # Wait for results
    results = {}
    for _ in range(2):
        gpu_id, embeddings = output_queue.get()
        results[gpu_id] = embeddings
        print(f"  GPU {gpu_id} complete: {len(embeddings)} embeddings")

    process1.join()
    process2.join()

    # Combine results
    all_speaker_embeddings = {}
    for gpu_id, embeddings in results.items():
        all_speaker_embeddings.update(embeddings)

    print(f"\n✅ Total embeddings extracted: {len(all_speaker_embeddings)}")

    # Cluster speakers
    print(f"\n[3/3] Clustering speakers into {n_clusters} clusters...")

    speaker_ids = list(all_speaker_embeddings.keys())
    embeddings = torch.stack([all_speaker_embeddings[sid] for sid in speaker_ids]).numpy()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    speaker_to_cluster = {}
    cluster_to_speakers = defaultdict(list)

    for speaker_id, cluster_id in zip(speaker_ids, cluster_labels):
        speaker_to_cluster[speaker_id] = int(cluster_id)
        cluster_to_speakers[int(cluster_id)].append(speaker_id)

    print(f"Clustering complete:")
    for cluster_id in sorted(cluster_to_speakers.keys())[:10]:
        n_speakers = len(cluster_to_speakers[cluster_id])
        print(f"  Cluster {cluster_id}: {n_speakers} speakers")

    # Save
    print(f"\nSaving to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'speaker_embeddings': {
            k: v.numpy().tolist() for k, v in all_speaker_embeddings.items()
        },
        'speaker_files': {k: v for k, v in speaker_to_files.items()},
        'speaker_to_cluster': speaker_to_cluster,
        'cluster_to_speakers': {
            str(k): v for k, v in cluster_to_speakers.items()
        },
    }

    with open(output_path, 'w') as f:
        json.dump(data, f)

    print(f"\n{'='*70}")
    print(f"✅ Speaker index saved to: {output_path}")
    print(f"  Total speakers: {len(all_speaker_embeddings)}")
    print(f"  Total clusters: {len(cluster_to_speakers)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    build_speaker_index_parallel(
        dataset_root='data/train_split',
        output_path='pretrained_models/speaker_index.json',
        n_clusters=50,
        gpu_ids=[1, 2]
    )

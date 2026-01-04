#!/usr/bin/env python3
"""
Pre-compute and save speaker embeddings to disk.

This solves two problems:
1. Startup time: Load cached embeddings instantly vs 30+ min recomputing
2. Memory: Use memory-mapped files (lazy loading, no upfront cost)

Usage:
    uv run python scripts/build_embedding_cache.py
    uv run python scripts/build_embedding_cache.py --output pretrained_models/embeddings_cache.npy
"""

import sys
import torch
import numpy as np
import torchaudio
import argparse
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from snac import SNACWithSpeakerConditioning


def extract_embedding(model, audio_path, device='cuda', max_samples=48000):
    """Extract speaker embedding from a single audio file."""
    try:
        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 24kHz
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            waveform = resampler(waveform)

        # Take first 2 seconds
        if waveform.shape[-1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[-1] < 24000:
            waveform = torch.nn.functional.pad(waveform, (0, 24000 - waveform.shape[-1]))

        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)

        # Extract embedding
        waveform = waveform.to(device)
        with torch.no_grad():
            embedding = model.extract_speaker_embedding(waveform.unsqueeze(0))

        return embedding.squeeze(0).cpu()

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def build_embedding_cache(
    model,
    audio_files,
    output_path,
    device='cuda',
    batch_size=1,
    use_multiprocessing=False,
    num_workers=None
):
    """
    Build and save embedding cache.

    Args:
        model: SNAC model with speaker encoder
        audio_files: List of audio file paths
        output_path: Where to save the cache (.npy file)
        device: torch device
        batch_size: Batch size for processing (for GPU utilization)
        use_multiprocessing: Whether to use multiprocessing for faster processing
        num_workers: Number of workers for multiprocessing
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Building embedding cache for {len(audio_files)} files")
    print(f"Output: {output_path}")

    embeddings_list = []
    valid_paths = []

    if use_multiprocessing and num_workers > 1:
        # Multiprocessing mode (faster for many files)
        from multiprocessing import Pool
        from functools import partial

        print(f"Using multiprocessing with {num_workers} workers")

        # Process in parallel
        with Pool(num_workers) as pool:
            extract_fn = partial(extract_embedding, model, device=device)
            results = list(tqdm(
                pool.imap(extract_fn, audio_files),
                total=len(audio_files),
                desc="Extracting embeddings"
            ))

        for audio_path, emb in zip(audio_files, results):
            if emb is not None:
                embeddings_list.append(emb.numpy())
                valid_paths.append(str(audio_path))

    else:
        # Sequential mode (simpler, for debugging)
        for audio_path in tqdm(audio_files, desc="Extracting embeddings"):
            emb = extract_embedding(model, audio_path, device=device)
            if emb is not None:
                embeddings_list.append(emb.numpy())
                valid_paths.append(str(audio_path))

    # Stack embeddings
    if len(embeddings_list) == 0:
        raise ValueError("No embeddings were extracted successfully!")

    embeddings = np.vstack(embeddings_list).astype('float32')

    print(f"\nExtracted {len(embeddings)} embeddings")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Memory size: {embeddings.nbytes / 1024 / 1024:.1f} MB")

    # Save embeddings
    np_path = output_path.with_suffix('.npy')
    np.save(np_path, embeddings)
    print(f"Saved embeddings to: {np_path}")

    # Save file paths (corresponding to embeddings)
    import json
    paths_path = output_path.with_suffix('.json')
    with open(paths_path, 'w') as f:
        json.dump(valid_paths, f)
    print(f"Saved file paths to: {paths_path}")

    return embeddings, valid_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/train_split',
                       help='Path to training data directory')
    parser.add_argument('--output', type=str, default='pretrained_models/embeddings_cache.npy',
                       help='Where to save the embedding cache')
    parser.add_argument('--model', type=str, default='hubertsiuzdak/snac_24khz',
                       help='Pretrained model name or path')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of workers for multiprocessing')
    parser.add_argument('--extensions', nargs='+', default=['.wav', '.WAV'],
                       help='Audio file extensions to include')

    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model}")
    model = SNACWithSpeakerConditioning.from_pretrained_base(
        repo_id=args.model,
        speaker_emb_dim=512,
        speaker_encoder_type='eres2net',
        freeze_base=True,
    ).to(args.device)
    model.eval()

    # Find audio files
    data_path = Path(args.data)
    audio_files = []
    for ext in args.extensions:
        audio_files.extend(list(data_path.glob(f'*{ext}')))

    audio_files = [str(f) for f in audio_files]

    print(f"Found {len(audio_files)} audio files")

    if len(audio_files) == 0:
        print("No audio files found!")
        return

    # Build cache
    use_mp = args.workers > 1
    build_embedding_cache(
        model,
        audio_files,
        args.output,
        device=args.device,
        use_multiprocessing=use_mp,
        num_workers=args.workers
    )

    print("\n✅ Done!")
    print(f"\nTo load the cache:")
    print(f"  embeddings = np.load('{args.output}')")
    print(f"  paths = json.load(open('{args.output.with_suffix('.json')}'))")


if __name__ == '__main__':
    main()

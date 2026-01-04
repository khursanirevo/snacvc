#!/usr/bin/env python3
"""
Worker script to extract embeddings for a subset of audio files.

Usage:
    CUDA_VISIBLE_DEVICES=1,2 python build_faiss_worker.py --files file_list_1.json --output embs_1.npy --gpu 0
    CUDA_VISIBLE_DEVICES=1,2 python build_faiss_worker.py --files file_list_2.json --output embs_2.npy --gpu 1
"""

import sys
import torch
import argparse
from pathlib import Path
import torchaudio
import json
import numpy as np
from tqdm import tqdm

sys.path.insert(0, '.')

from snac import SNACWithSpeakerConditioning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", required=True, help="Path to file list JSON")
    parser.add_argument("--output", required=True, help="Output path for embeddings (.npy)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')

    # Load file list
    with open(args.files, 'r') as f:
        file_list = json.load(f)

    print(f"[GPU {args.gpu}] Loading model...")
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

    print(f"[GPU {args.gpu}] Extracting embeddings for {len(file_list)} files...")

    embeddings_list = []
    file_paths = []

    batch_size = 64

    for start_idx in tqdm(range(0, len(file_list), batch_size), desc=f"GPU {args.gpu}"):
        end_idx = min(start_idx + batch_size, len(file_list))
        batch_files = file_list[start_idx:end_idx]

        # Load audios
        all_waveforms = []

        for audio_path in batch_files:
            try:
                waveform, sr = torchaudio.load(str(audio_path))

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

            except Exception as e:
                print(f"[GPU {args.gpu}] Error loading {audio_path}: {e}")
                continue

        if len(all_waveforms) == 0:
            continue

        # Extract embeddings
        try:
            with torch.no_grad():
                batch = torch.stack(all_waveforms).to(device)
                embs = model_base.extract_speaker_embedding(batch)
                embs = embs / (torch.norm(embs, dim=1, keepdim=True) + 1e-8)
                embs = embs.cpu().numpy()

            # Store
            for i, audio_path in enumerate(batch_files):
                if i < len(embs):
                    embeddings_list.append(embs[i])
                    file_paths.append(str(audio_path))

        except Exception as e:
            print(f"[GPU {args.gpu}] Error extracting batch: {e}")
            continue

    # Save embeddings and file paths
    embeddings_array = np.array(embeddings_list)
    np.save(args.output, embeddings_array)

    # Save corresponding file paths
    paths_file = args.output.replace('.npy', '_paths.json')
    with open(paths_file, 'w') as f:
        json.dump(file_paths, f)

    print(f"[GPU {args.gpu}] ✅ Complete! Saved {len(embeddings_list)} embeddings to {args.output}")


if __name__ == "__main__":
    main()

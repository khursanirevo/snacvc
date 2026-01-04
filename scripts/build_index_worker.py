#!/usr/bin/env python3
"""
Worker script to extract embeddings for a subset of speakers.
Meant to be run in parallel on different GPUs.

Usage:
    CUDA_VISIBLE_DEVICES=1 python build_index_worker.py --speakers speakers_1.json --output embs_1.json
    CUDA_VISIBLE_DEVICES=2 python build_index_worker.py --speakers speakers_2.json --output embs_2.json
"""

import sys
import torch
import argparse
from pathlib import Path
from collections import defaultdict
import torchaudio
import json
from tqdm import tqdm

sys.path.insert(0, '.')

from snac import SNACWithSpeakerConditioning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speakers", required=True, help="Path to speaker IDs JSON")
    parser.add_argument("--files", required=True, help="Path to speaker files mapping JSON")
    parser.add_argument("--output", required=True, help="Output path for embeddings JSON")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')

    # Load speaker IDs
    with open(args.speakers, 'r') as f:
        speaker_ids = json.load(f)

    # Load file mapping
    with open(args.files, 'r') as f:
        speaker_to_files = json.load(f)

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

    print(f"[GPU {args.gpu}] Extracting embeddings for {len(speaker_ids)} speakers...")

    speaker_embeddings = {}
    batch_size = 64

    for start_idx in tqdm(range(0, len(speaker_ids), batch_size), desc=f"GPU {args.gpu}"):
        end_idx = min(start_idx + batch_size, len(speaker_ids))
        batch_speakers = speaker_ids[start_idx:end_idx]

        # Load audios
        all_waveforms = []

        for speaker_id in batch_speakers:
            files = speaker_to_files.get(speaker_id, [])

            if len(files) == 0:
                continue

            # Use first file
            audio_path = files[0]

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
                embs = embs.cpu()

            # Store
            for i, speaker_id in enumerate(batch_speakers):
                if i < len(embs):
                    speaker_embeddings[speaker_id] = embs[i].numpy().tolist()

        except Exception as e:
            print(f"[GPU {args.gpu}] Error extracting batch: {e}")
            continue

    # Save
    print(f"[GPU {args.gpu}] Saving {len(speaker_embeddings)} embeddings...")

    output_data = {
        'speaker_embeddings': speaker_embeddings,
        'gpu_id': args.gpu,
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f)

    print(f"[GPU {args.gpu}] ✅ Complete! Saved to {args.output}")


if __name__ == "__main__":
    main()

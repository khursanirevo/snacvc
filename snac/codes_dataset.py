"""
Code-level dataset for Phase 11 training with random segment extraction.

Instead of loading full audio and extracting segments, we:
1. Load pre-computed codes from parquet files
2. Randomly extract segments from the codes directly
3. Load corresponding 48kHz audio targets

This is much more efficient than loading full audio files!
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import numpy as np


class PrecomputedCodesDataset(Dataset):
    """
    Dataset that loads pre-computed quantized codes and performs random segment extraction.

    Key innovation: Random segment extraction happens at CODE level, not audio level!

    For 4 seconds at 24kHz with SNAC (hop=512):
    - Scale 0 (stride 4): ~47 codes
    - Scale 1 (stride 2): ~94 codes
    - Scale 2 (stride 1): ~188 codes
    """
    def __init__(
        self,
        codes_dir: str,
        audio_48khz_dir: str,
        segment_length_sec: float = 4.0,
        sampling_rate: int = 24000,
        hop_length: int = 512,
    ):
        self.codes_dir = Path(codes_dir)
        self.audio_48khz_dir = Path(audio_48khz_dir)
        self.segment_length_sec = segment_length_sec
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length

        # Load metadata
        metadata_file = self.codes_dir / "metadata.json"
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        self.vq_strides = self.metadata['vq_strides']  # [4, 2, 1]
        self.num_scales = len(self.vq_strides)

        # Calculate segment length in codes for each scale
        # For residual VQ, all scales must produce the SAME temporal dimension after upsampling
        # Scale 0: N codes → repeat by stride 4 → N*4
        # Scale 1: N*2 codes → repeat by stride 2 → N*4
        # Scale 2: N*4 codes → repeat by stride 1 → N*4
        # So we calculate based on scale 0, then multiply by strides
        segment_samples = int(segment_length_sec * sampling_rate)
        base_length = segment_samples // (hop_length * self.vq_strides[0])  # Scale 0

        # All scales produce same temporal dimension after upsampling
        self.segment_lengths = [
            base_length * (self.vq_strides[0] // self.vq_strides[i])  # Relative to scale 0
            for i in range(len(self.vq_strides))
        ]

        print(f"Segment length: {segment_length_sec}s = {segment_samples} samples")
        print(f"  Scale 0: {self.segment_lengths[0]} codes (stride {self.vq_strides[0]})")
        print(f"  Scale 1: {self.segment_lengths[1]} codes (stride {self.vq_strides[1]})")
        print(f"  Scale 2: {self.segment_lengths[2]} codes (stride {self.vq_strides[2]})")

        # Get all parquet files
        self.parquet_files = sorted(self.codes_dir.glob("codes_batch_*.parquet"))
        print(f"Found {len(self.parquet_files)} code files")

        # Scan all files to build index of valid segments
        print("Scanning files to build segment index...")
        self._build_index()

    def _build_index(self):
        """Build index of all files with their code lengths."""
        self.file_index = []  # List of (file_idx, num_valid_segments)

        for file_idx, parquet_file in enumerate(self.parquet_files):
            try:
                df = pd.read_parquet(parquet_file, columns=['shape_scale_0'])
                num_codes = df['shape_scale_0'].iloc[0][0]  # Length of scale 0 codes

                # Calculate how many valid segments we can extract
                # Each segment needs segment_lengths[0] codes from scale 0
                num_valid_segments = max(0, num_codes - self.segment_lengths[0] + 1)

                if num_valid_segments > 0:
                    self.file_index.append({
                        'file_idx': file_idx,
                        'parquet_file': parquet_file,
                        'num_codes': num_codes,
                        'num_segments': num_valid_segments,
                    })
            except Exception as e:
                print(f"Warning: Error reading {parquet_file}: {e}")
                continue

        # Calculate cumulative segments for global indexing
        self.cumulative_segments = [0]
        for info in self.file_index:
            self.cumulative_segments.append(self.cumulative_segments[-1] + info['num_segments'])

        self.total_segments = self.cumulative_segments[-1]
        print(f"Total segments: {self.total_segments:,}")

    def __len__(self):
        return self.total_segments

    def __getitem__(self, idx):
        """Get a random segment from codes."""
        # Find which file contains this segment
        file_idx = bisect.bisect_right(self.cumulative_segments, idx) - 1
        segment_idx_in_file = idx - self.cumulative_segments[file_idx]

        file_info = self.file_index[file_idx]

        # Load the parquet file
        df = pd.read_parquet(file_info['parquet_file'])

        # Select a random row from the file
        row_idx = torch.randint(0, len(df), (1,)).item()
        row = df.iloc[row_idx]

        # Get the maximum start position for each scale
        max_starts = [
            row[f'shape_scale_{i}'][0] - self.segment_lengths[i]
            for i in range(self.num_scales)
        ]
        max_start_scale_0 = max_starts[0]  # Use scale 0 as reference

        # Random start position (in scale 0 codes)
        if max_start_scale_0 <= 0:
            # File too short, return first segment available
            start_pos = 0
        else:
            start_pos = torch.randint(0, max_start_scale_0 + 1, (1,)).item()

        # Extract codes for each scale
        codes = []
        for scale in range(self.num_scales):
            # Start position for this scale (account for VQ stride)
            scale_start = start_pos * self.vq_strides[scale]

            # Extract segment from codes
            full_codes = row[f'codes_scale_{scale}']
            segment_codes = full_codes[scale_start:scale_start + self.segment_lengths[scale]]

            # Pad if needed
            if len(segment_codes) < self.segment_lengths[scale]:
                pad_length = self.segment_lengths[scale] - len(segment_codes)
                segment_codes = np.pad(segment_codes, (0, pad_length), mode='constant')

            codes.append(torch.tensor(segment_codes, dtype=torch.long))

        # Get corresponding 48kHz audio file
        audio_file_name = Path(row['file_path']).name
        audio_path = self.audio_48khz_dir / audio_file_name

        # Load 48kHz audio and extract corresponding segment
        # Convert start position (in 24kHz codes) to 48kHz audio samples
        # At 24kHz: start_pos * hop_length * stride[0] = position in 24kHz audio
        # At 48kHz: multiply by 2 = position in 48kHz audio
        start_sample_24k = start_pos * self.hop_length * self.vq_strides[0]
        start_sample_48k = start_sample_24k * 2  # Convert to 48kHz position
        segment_samples = int(self.segment_length_sec * 48000)  # 48kHz

        try:
            audio_48k, sr = torchaudio.load(str(audio_path))

            # Convert to mono if needed
            if audio_48k.shape[0] > 1:
                audio_48k = audio_48k.mean(dim=0, keepdim=True)

            # Resample if needed
            if sr != 48000:
                resampler = torchaudio.transforms.Resample(sr, 48000)
                audio_48k = resampler(audio_48k)

            # Extract segment
            if audio_48k.shape[-1] >= start_sample_48k + segment_samples:
                audio_segment = audio_48k[..., start_sample_48k:start_sample_48k + segment_samples]
            else:
                # File too short, pad
                audio_segment = audio_48k[..., start_sample_48k:]
                if audio_segment.shape[-1] < segment_samples:
                    pad_length = segment_samples - audio_segment.shape[-1]
                    audio_segment = F.pad(audio_segment, (0, pad_length), mode='constant')

        except Exception as e:
            # If audio loading fails, return zeros
            audio_segment = torch.zeros(1, segment_samples)

        return codes, audio_segment


def collate_fn(batch):
    """Collate function with padding."""
    codes_batch = [[] for _ in range(3)]  # 3 scales
    audio_batch = []

    max_audio_len = max(audio.shape[-1] for _, audio in batch)

    for codes, audio in batch:
        # Pad audio
        if audio.shape[-1] < max_audio_len:
            audio = F.pad(audio, (0, max_audio_len - audio.shape[-1]), mode='constant')
        audio_batch.append(audio)

        # Collect codes
        for scale in range(3):
            codes_batch[scale].append(codes[scale])

    # Stack
    audio_batch = torch.stack(audio_batch)  # (B, 1, T)
    codes_batch = [torch.stack(codes_batch[scale]) for scale in range(3)]

    return codes_batch, audio_batch


import bisect

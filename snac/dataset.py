"""
Optimized Audio Dataset for SNAC Training

Improvements over SimpleAudioDataset:
1. Efficient loading using torchaudio.info() and frame_offset
2. Skip too-short files instead of zero-padding
3. Cached resampler for better performance
4. Better error handling and validation
5. Reproducible random sampling
"""

import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset


class OptimizedAudioDataset(Dataset):
    """
    Optimized dataset for audio loading with efficient segment extraction.

    Key improvements:
    - Uses torchaudio.info() to get duration without loading full audio
    - Loads only the required segment using frame_offset and num_frames
    - Skips files that are too short instead of padding
    - Caches resampler to avoid recomputation
    - Sets random seed for reproducible sampling
    """

    def __init__(
        self,
        dataset_root,
        sampling_rate=24000,
        segment_length=1.0,  # seconds (changed from 4.0 to match SNAC paper)
        augment=True,
        extract_speaker_ids=False,
        seed=42,
        min_length_ratio=1.0,  # Skip files shorter than segment_length * ratio
    ):
        """
        Args:
            dataset_root: Path to audio files directory
            sampling_rate: Target sampling rate (default 24kHz for SNAC)
            segment_length: Length of audio segments in seconds
            augment: Apply gain augmentation
            extract_speaker_ids: Extract speaker IDs from filenames
            seed: Random seed for reproducibility
            min_length_ratio: Minimum file length as ratio of segment_length
                             (e.g., 1.0 = skip files shorter than segment_length)
        """
        self.dataset_root = Path(dataset_root)
        self.sampling_rate = sampling_rate
        self.segment_length = int(segment_length * sampling_rate)
        self.min_length = int(self.segment_length * min_length_ratio)
        self.augment = augment
        self.extract_speaker_ids = extract_speaker_ids

        # Set random seed for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)

        # Cached resampler (created on first use if needed)
        self._resampler = None
        self._original_sr = None

        # Collect all audio files
        audio_extensions = ['.wav', '.WAV', '.mp3', '.flac', '.ogg', '.m4a']
        self.samples = []

        for ext in audio_extensions:
            self.samples.extend(list(self.dataset_root.glob(f'*{ext}')))

        print(f"Found {len(self.samples)} audio files in {dataset_root}")

        if len(self.samples) == 0:
            raise ValueError(f"No audio files found in {dataset_root}")

        # Pre-filter files that are too short using torchaudio.info()
        self._filter_short_files()

        # Extract speaker IDs from filenames if requested
        self.speaker_to_idx = {}
        if extract_speaker_ids:
            self._extract_speaker_ids()

    def _filter_short_files(self):
        """Filter out files that are too short using torchaudio.info()."""
        valid_samples = []
        skipped = 0

        print(f"Filtering files shorter than {self.min_length / self.sampling_rate:.1f}s...")

        for audio_path in self.samples:
            try:
                info = torchaudio.info(str(audio_path))
                num_frames = info.num_frames

                if num_frames >= self.min_length:
                    valid_samples.append(audio_path)
                else:
                    skipped += 1
            except Exception as e:
                print(f"Warning: Could not read {audio_path.name}: {e}")
                skipped += 1

        self.samples = valid_samples
        print(f"  Kept {len(self.samples)} files, skipped {skipped} short files")

        if len(self.samples) == 0:
            raise ValueError(f"No audio files meet minimum length requirement ({self.min_length / self.sampling_rate:.1f}s)")

    def _extract_speaker_ids(self):
        """Extract speaker IDs from filenames."""
        speaker_names = set()
        for audio_path in self.samples:
            filename = audio_path.stem
            if '_' in filename:
                speaker_name = filename.split('_')[0]
            else:
                speaker_name = filename[:4]
            speaker_names.add(speaker_name)

        self.speaker_to_idx = {name: idx for idx, name in enumerate(sorted(speaker_names))}
        print(f"Extracted {len(self.speaker_to_idx)} unique speakers from filenames")

    def _get_resampler(self, original_sr):
        """Get cached resampler for the original sample rate."""
        if self._resampler is None or self._original_sr != original_sr:
            self._resampler = torchaudio.transforms.Resample(original_sr, self.sampling_rate)
            self._original_sr = original_sr
        return self._resampler

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path = self.samples[idx]

        try:
            # Get audio info WITHOUT loading the full audio
            info = torchaudio.info(str(audio_path))
            total_samples = info.num_frames
            sr = info.sample_rate

            # Calculate random start position
            max_start = total_samples - self.segment_length
            if max_start < 0:
                # This shouldn't happen after filtering, but just in case
                max_start = 0

            start = torch.randint(0, max_start + 1, (1,)).item()

            # Load ONLY the segment we need! (efficient loading)
            audio, sr = torchaudio.load(
                str(audio_path),
                frame_offset=start,
                num_frames=self.segment_length
            )

            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

            # Resample if necessary (using cached resampler)
            if sr != self.sampling_rate:
                resampler = self._get_resampler(sr)
                audio = resampler(audio)

            # Ensure correct length after resampling
            if audio.shape[-1] > self.segment_length:
                audio = audio[..., :self.segment_length]
            elif audio.shape[-1] < self.segment_length:
                # Small padding for rounding errors
                audio = F.pad(audio, (0, self.segment_length - audio.shape[-1]))

            # Augmentation
            if self.augment and torch.rand(1).item() > 0.5:
                gain = 10 ** (torch.randn(1).item() * 0.1)  # ±10dB
                audio = audio * gain

            result = {'audio': audio.squeeze(0)}  # (T,)

            # Add speaker ID if requested
            if self.extract_speaker_ids:
                filename = audio_path.stem
                if '_' in filename:
                    speaker_name = filename.split('_')[0]
                else:
                    speaker_name = filename[:4]

                speaker_id = self.speaker_to_idx.get(speaker_name, 0)
                result['speaker_id'] = speaker_id

            return result

        except Exception as e:
            print(f"Error loading {audio_path.name}: {e}")
            # Return next sample as fallback
            next_idx = (idx + 1) % len(self.samples)
            return self.__getitem__(next_idx)


# Backward compatibility alias
SimpleAudioDataset = OptimizedAudioDataset

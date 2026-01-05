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
    - Supports variable segment lengths (single float or list of floats)
    """

    def __init__(
        self,
        dataset_root,
        sampling_rate=24000,
        segment_length=1.0,  # seconds (can be float or list of floats)
        augment=True,
        extract_speaker_ids=False,
        seed=42,
        min_length_ratio=1.0,  # Skip files shorter than min(segment_length) * ratio
        filter_short=False,  # Whether to pre-filter short files (SLOW!)
    ):
        """
        Args:
            dataset_root: Path to audio files directory
            sampling_rate: Target sampling rate (default 24kHz for SNAC)
            segment_length: Length of audio segments in seconds.
                           Can be a single float (e.g., 2.0) or list of floats
                           (e.g., [1.0, 2.0, 3.0, 4.0]) for random length per batch
            augment: Apply gain augmentation
            extract_speaker_ids: Extract speaker IDs from filenames
            seed: Random seed for reproducibility
            min_length_ratio: Minimum file length as ratio of min(segment_length)
                             (e.g., 1.0 = skip files shorter than shortest segment)
            filter_short: Whether to pre-filter short files (default: False, much faster)
        """
        self.dataset_root = Path(dataset_root)
        self.sampling_rate = sampling_rate
        self.augment = augment
        self.extract_speaker_ids = extract_speaker_ids
        self.filter_short = filter_short

        # Handle segment_length as either single value or list
        if isinstance(segment_length, (list, tuple)):
            self.segment_lengths = segment_length
            self.min_segment_length = min(segment_length)
        else:
            self.segment_lengths = [segment_length]
            self.min_segment_length = segment_length

        self.min_length = int(self.min_segment_length * sampling_rate * min_length_ratio)

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

        # Pre-filter files that are too short (ONLY if requested)
        if self.filter_short:
            self._filter_short_files()
        else:
            print(f"Skipping pre-filtering (filter_short=False)")
            if len(self.segment_lengths) > 1:
                print(f"  Variable segment lengths: {self.segment_lengths}s (random per batch)")
            print(f"  Short files will be skipped during loading (much faster)")

        # Extract speaker IDs from filenames if requested
        self.speaker_to_idx = {}
        if extract_speaker_ids:
            self._extract_speaker_ids()

    def _filter_short_files(self):
        """Filter out files that are too short using torchaudio.info()."""
        valid_samples = []
        skipped = 0

        min_length_sec = self.min_length / self.sampling_rate
        print(f"Filtering files shorter than {min_length_sec:.1f}s (this may take a while)...")

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
            raise ValueError(f"No audio files meet minimum length requirement ({min_length_sec:.1f}s)")

        if len(self.segment_lengths) > 1:
            print(f"  Variable segment lengths: {self.segment_lengths}s (random per batch)")

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

    def load_audio(self, idx, segment_length_sec):
        """
        Load audio with a specific segment length.

        Args:
            idx: Sample index
            segment_length_sec: Segment length in seconds

        Returns:
            Dictionary with 'audio' key (and optionally 'speaker_id')
        """
        audio_path = self.samples[idx]
        segment_length = int(segment_length_sec * self.sampling_rate)

        try:
            # Get audio info WITHOUT loading the full audio
            info = torchaudio.info(str(audio_path))
            total_samples = info.num_frames
            sr = info.sample_rate

            # Calculate random start position
            max_start = total_samples - segment_length
            if max_start < 0:
                # File too short for this segment length
                return None

            start = torch.randint(0, max_start + 1, (1,)).item()

            # Load ONLY the segment we need! (efficient loading)
            audio, sr = torchaudio.load(
                str(audio_path),
                frame_offset=start,
                num_frames=segment_length
            )

            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

            # Resample if necessary (using cached resampler)
            if sr != self.sampling_rate:
                resampler = self._get_resampler(sr)
                audio = resampler(audio)

            # Ensure correct length after resampling
            if audio.shape[-1] > segment_length:
                audio = audio[..., :segment_length]
            elif audio.shape[-1] < segment_length:
                # Small padding for rounding errors only
                audio = F.pad(audio, (0, segment_length - audio.shape[-1]))

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
            return None

    def __getitem__(self, idx):
        """
        Get a sample. Returns index for lazy loading.

        For backward compatibility with single segment_length,
        returns loaded audio. For variable lengths, use
        variable_length_collate which will reload with random length.
        """
        # If only one segment length, load directly (efficient)
        if len(self.segment_lengths) == 1:
            return self.load_audio(idx, self.segment_lengths[0])
        else:
            # Return index for lazy loading by collate function
            return {'idx': idx}


# Backward compatibility alias
SimpleAudioDataset = OptimizedAudioDataset


def variable_length_collate(dataset):
    """
    Custom collate function that randomizes segment length per batch.

    All samples in a batch will have the SAME length (no padding needed),
    but each batch uses a different randomly chosen length.

    Args:
        dataset: OptimizedAudioDataset instance

    Returns:
        Collate function to use with DataLoader
    """
    def collate_fn(batch):
        # Handle both loaded audio (single segment) and indices (variable segments)
        if batch and 'idx' not in batch[0]:
            # Single segment length case: already loaded
            batch = [item for item in batch if item is not None]
            if len(batch) == 0:
                return {'audio': torch.empty(0)}

            result = {'audio': torch.stack([item['audio'] for item in batch])}
            if 'speaker_id' in batch[0]:
                result['speaker_id'] = torch.stack([item['speaker_id'] for item in batch])
            return result

        # Variable segment lengths case: reload with random length
        # Extract indices
        indices = [item['idx'] for item in batch if item is not None]

        if len(indices) == 0:
            return {'audio': torch.empty(0)}

        # Randomly choose segment length for this batch
        segment_length = random.choice(dataset.segment_lengths)

        # Load all samples with this segment length
        batch_data = []
        for idx in indices:
            sample = dataset.load_audio(idx, segment_length)
            if sample is not None:
                batch_data.append(sample)

        if len(batch_data) == 0:
            return {'audio': torch.empty(0)}

        # Collate into batch
        result = {}

        # Stack audio tensors (all same length, no padding!)
        try:
            result['audio'] = torch.stack([item['audio'] for item in batch_data])
        except RuntimeError as e:
            # Fallback: pad if lengths somehow differ
            print(f"Warning: Length mismatch in batch, padding: {e}")
            max_len = max(item['audio'].shape[0] for item in batch_data)
            padded_audios = []
            for item in batch_data:
                audio = item['audio']
                if audio.shape[0] < max_len:
                    audio = F.pad(audio, (0, max_len - audio.shape[0]))
                padded_audios.append(audio)
            result['audio'] = torch.stack(padded_audios)

        # Add speaker_id if present
        if 'speaker_id' in batch_data[0]:
            result['speaker_id'] = torch.stack([
                batch_data[i]['speaker_id'] for i in range(len(batch_data))
            ])

        return result

    return collate_fn


def curriculum_collate(dataset, segment_length):
    """
    Collate function for curriculum learning (fixed segment length per epoch).

    Args:
        dataset: OptimizedAudioDataset instance
        segment_length: Fixed segment length for this epoch

    Returns:
        Collate function to use with DataLoader
    """
    def collate_fn(batch):
        # Extract indices
        indices = [item['idx'] for item in batch if item is not None]

        if len(indices) == 0:
            return {'audio': torch.empty(0)}

        # Load all samples with the fixed segment length for this epoch
        batch_data = []
        for idx in indices:
            sample = dataset.load_audio(idx, segment_length)
            if sample is not None:
                batch_data.append(sample)

        if len(batch_data) == 0:
            return {'audio': torch.empty(0)}

        # Collate into batch
        result = {'audio': torch.stack([item['audio'] for item in batch_data])}

        if 'speaker_id' in batch_data[0]:
            result['speaker_id'] = torch.stack([
                batch_data[i]['speaker_id'] for i in range(len(batch_data))
            ])

        return result

    return collate_fn




#!/usr/bin/env python3
"""
Audio Augmentation for Synthetic Voice Conversion.

Creates pseudo-speaker pairs by applying pitch/formant shifting to audio.
This teaches the model that speaker embeddings should dominate over
acoustic patterns in codes.

Key insight:
- Pitch/formant-shifted audio contains "wrong" speaker acoustic patterns
- When decoded with original speaker embedding, should reconstruct original
- Forces model to trust embedding over codes for speaker identity
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T
import random
from typing import Tuple, Optional
import numpy as np
import scipy.signal as signal


class PitchShifter:
    """
    Pitch shifting for audio augmentation.

    Uses resampling-based pitch shifting for efficiency.
    Preserves duration while changing pitch.
    """

    # Class-level cache for resamplers (shared across all instances)
    _resampler_cache = {}

    def __init__(self, sample_rate: int = 24000):
        """
        Args:
            sample_rate: Audio sampling rate (default 24kHz for SNAC)
        """
        self.sample_rate = sample_rate

    def shift_semitones(self, audio: torch.Tensor, semitones: float) -> torch.Tensor:
        """
        Shift pitch by semitones using resampling.

        Args:
            audio: (B, 1, T) or (1, T) audio tensor
            semitones: Pitch shift in semitones (positive = higher, negative = lower)

        Returns:
            Pitch-shifted audio with same duration
        """
        if semitones == 0:
            return audio

        device = audio.device
        dtype = audio.dtype

        # Calculate resampling ratio for pitch shift
        # Pitch shift factor = 2^(semitones/12)
        pitch_factor = 2.0 ** (semitones / 12.0)

        # To preserve duration, we need to compensate the resampling
        # Resample up by pitch_factor, then down (or vice versa)
        if pitch_factor > 1.0:
            # Pitch up: resample up then down
            up_freq = int(self.sample_rate * pitch_factor)
            down_freq = self.sample_rate

            # Get or create resamplers (cached)
            cache_key_up = (self.sample_rate, up_freq, dtype)
            cache_key_down = (up_freq, down_freq, dtype)

            if cache_key_up not in self._resampler_cache:
                self._resampler_cache[cache_key_up] = T.Resample(
                    orig_freq=self.sample_rate,
                    new_freq=up_freq,
                    dtype=dtype
                )
            if cache_key_down not in self._resampler_cache:
                self._resampler_cache[cache_key_down] = T.Resample(
                    orig_freq=up_freq,
                    new_freq=down_freq,
                    dtype=dtype
                )

            resampler_up = self._resampler_cache[cache_key_up].to(device)
            resampler_down = self._resampler_cache[cache_key_down].to(device)
            audio_shifted = resampler_down(resampler_up(audio))
        else:
            # Pitch down: resample down then up
            down_freq = int(self.sample_rate * pitch_factor)
            up_freq = self.sample_rate

            # Get or create resamplers (cached)
            cache_key_down = (self.sample_rate, down_freq, dtype)
            cache_key_up = (down_freq, up_freq, dtype)

            if cache_key_down not in self._resampler_cache:
                self._resampler_cache[cache_key_down] = T.Resample(
                    orig_freq=self.sample_rate,
                    new_freq=down_freq,
                    dtype=dtype
                )
            if cache_key_up not in self._resampler_cache:
                self._resampler_cache[cache_key_up] = T.Resample(
                    orig_freq=down_freq,
                    new_freq=up_freq,
                    dtype=dtype
                )

            resampler_down = self._resampler_cache[cache_key_down].to(device)
            resampler_up = self._resampler_cache[cache_key_up].to(device)
            audio_shifted = resampler_up(resampler_down(audio))

        return audio_shifted

    def __call__(self, audio: torch.Tensor, semitones: float) -> torch.Tensor:
        return self.shift_semitones(audio, semitones)


class FormantShifter:
    """
    Formant shifting for audio augmentation.

    Modifies the spectral envelope to simulate different vocal tract lengths.
    This creates more natural speaker transformations (e.g., male ↔ female).
    """

    def __init__(self, sample_rate: int = 24000):
        """
        Args:
            sample_rate: Audio sampling rate (default 24kHz for SNAC)
        """
        self.sample_rate = sample_rate

    def shift_formants(self, audio: torch.Tensor, shift_factor: float) -> torch.Tensor:
        """
        Shift formants by modifying the spectral envelope.

        Args:
            audio: (B, 1, T) or (1, T) audio tensor
            shift_factor: Formant shift factor (positive = higher formants, negative = lower)
                           ±0.2 means ±20% formant frequency shift

        Returns:
            Formant-shifted audio with same duration
        """
        if shift_factor == 0:
            return audio

        device = audio.device
        dtype = audio.dtype

        # Convert to numpy for scipy processing
        if audio.dim() == 3:
            audio_np = audio[0, 0].cpu().numpy()
        else:
            audio_np = audio[0].cpu().numpy()

        # Compute spectrogram
        f, t, Zxx = signal.stft(audio_np, fs=self.sample_rate, nperseg=2048, noverlap=1536)

        # Find spectral envelope (peaks in spectrum)
        # Use median filtering as simple envelope estimation
        envelope = np.median(np.abs(Zxx), axis=0)
        envelope = np.convolve(envelope, np.ones(5)/5, mode='same')  # Smooth

        # Shift formant frequencies by resampling spectral envelope
        # Formant shift > 0: shift envelope up (shorter vocal tract, female-like)
        # Formant shift < 0: shift envelope down (longer vocal tract, male-like)
        formant_stretched = np.interp(
            np.linspace(0, 1, len(envelope)),
            np.linspace(0, 1, len(envelope)),
            envelope,
            kind='linear'
        )

        if shift_factor > 0:
            # Shift formants up (compress in frequency)
            new_indices = np.linspace(0, 1, len(envelope))
            stretched_indices = np.clip(new_indices * (1 - shift_factor), 0, 1)
            formant_stretched = np.interp(new_indices, stretched_indices, envelope, kind='linear')
        else:
            # Shift formants down (expand in frequency)
            new_indices = np.linspace(0, 1, len(envelope))
            stretched_indices = np.clip(new_indices * (1 + abs(shift_factor)), 0, 1)
            formant_stretched = np.interp(stretched_indices, new_indices, envelope, kind='linear')

        # Apply formant scaling to each frequency bin
        # Normalize and rescale
        envelope_normalized = envelope / (np.mean(envelope) + 1e-8)
        formant_normalized = formant_stretched / (np.mean(formant_stretched) + 1e-8)
        scaling_factor = formant_normalized / (envelope_normalized + 1e-8)

        # Apply scaling to spectrogram
        Zxx_modified = Zxx * scaling_factor[np.newaxis, :]

        # Inverse STFT
        _, audio_shifted_np = signal.istft(Zxx_modified, fs=self.sample_rate, nperseg=2048, noverlap=1536)

        # Handle length mismatch
        if len(audio_shifted_np) > len(audio_np):
            audio_shifted_np = audio_shifted_np[:len(audio_np)]
        elif len(audio_shifted_np) < len(audio_np):
            audio_shifted_np = np.pad(audio_shifted_np, (0, len(audio_np) - len(audio_shifted_np)))

        # Convert back to tensor
        audio_shifted = torch.from_numpy(audio_shifted_np).to(device).to(dtype)

        if audio.dim() == 3:
            audio_shifted = audio_shifted.unsqueeze(0).unsqueeze(0)

        return audio_shifted

    def __call__(self, audio: torch.Tensor, shift_factor: float) -> torch.Tensor:
        return self.shift_formants(audio, shift_factor)


def augment_audio_for_voice_conversion_advanced(
    audio: torch.Tensor,
    pitch_shift_range: list = [-2, -1, 1, 2],
    formant_shift_range: list = [-0.2, -0.1, 0.1, 0.2],
    probability: float = 0.5
) -> Tuple[torch.Tensor, bool, float, float]:
    """
    Apply random pitch and/or formant shifting for synthetic voice conversion.

    Args:
        audio: (B, 1, T) or (1, T) audio tensor
        pitch_shift_range: List of semitone values to randomly choose from
        formant_shift_range: List of formant shift factors (±0.2 = ±20%)
        probability: Probability of applying augmentation

    Returns:
        (augmented_audio, was_augmented, semitones_used, formant_shift_used)
        - If augmentation not applied, returns original audio, False, 0.0, 0.0
    """
    # Use persistent shifters (module-level singleton)
    if not hasattr(augment_audio_for_voice_conversion_advanced, '_pitch_shifter'):
        augment_audio_for_voice_conversion_advanced._pitch_shifter = PitchShifter(sample_rate=24000)
    if not hasattr(augment_audio_for_voice_conversion_advanced, '_formant_shifter'):
        augment_audio_for_voice_conversion_advanced._formant_shifter = FormantShifter(sample_rate=24000)

    # Randomly decide whether to augment
    if random.random() > probability:
        return audio, False, 0.0, 0.0

    # Apply pitch shift
    semitones = random.choice(pitch_shift_range)
    audio_aug = augment_audio_for_voice_conversion_advanced._pitch_shifter(audio, semitones)

    # Apply formant shift
    formant_shift = random.choice(formant_shift_range)
    audio_aug = augment_audio_for_voice_conversion_advanced._formant_shifter(audio_aug, formant_shift)

    return audio_aug, True, semitones, formant_shift


def augment_audio_for_voice_conversion(
    audio: torch.Tensor,
    pitch_shift_range: list = [-2, -1, 1, 2],
    probability: float = 0.5
) -> Tuple[torch.Tensor, bool, float]:
    """
    Apply random pitch shifting for synthetic voice conversion.

    Args:
        audio: (B, 1, T) or (1, T) audio tensor
        pitch_shift_range: List of semitone values to randomly choose from
        probability: Probability of applying augmentation

    Returns:
        (augmented_audio, was_augmented, semitones_used)
        - If augmentation not applied, returns original audio, False, 0.0
    """
    # Use persistent pitch shifter (module-level singleton)
    if not hasattr(augment_audio_for_voice_conversion, '_pitch_shifter'):
        augment_audio_for_voice_conversion._pitch_shifter = PitchShifter(sample_rate=24000)

    # Randomly decide whether to augment
    if random.random() > probability:
        return audio, False, 0.0

    # Random pitch shift amount
    semitones = random.choice(pitch_shift_range)

    # Apply pitch shift (uses cached resamplers)
    audio_aug = augment_audio_for_voice_conversion._pitch_shifter(audio, semitones)

    return audio_aug, True, semitones


class SyntheticVoiceConversionAugmentation(nn.Module):
    """
    Module for synthetic voice conversion augmentation.

    Can be used as a nn.Module for easier integration in training loops.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        pitch_shift_range: list = [-2, -1, 1, 2],
        probability: float = 0.5
    ):
        """
        Args:
            sample_rate: Audio sampling rate
            pitch_shift_range: List of semitone values
            probability: Probability of applying augmentation
        """
        super().__init__()
        self.pitch_shifter = PitchShifter(sample_rate)
        self.pitch_shift_range = pitch_shift_range
        self.probability = probability

    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, bool, float]:
        """
        Apply augmentation during training.

        Args:
            audio: (B, 1, T) audio tensor

        Returns:
            (augmented_audio, was_augmented, semitones_used)
        """
        if self.training and random.random() < self.probability:
            semitones = random.choice(self.pitch_shift_range)
            audio_aug = self.pitch_shifter(audio, semitones)
            return audio_aug, True, semitones
        else:
            return audio, False, 0.0

    def get_augmentation_info(self) -> dict:
        """Get info about augmentation settings."""
        return {
            'pitch_shift_range': self.pitch_shift_range,
            'probability': self.probability,
            'sample_rate': self.pitch_shifter.sample_rate
        }


if __name__ == '__main__':
    print("Audio Augmentation for Synthetic Voice Conversion")
    print("=" * 70)
    print("\nThis module provides:")
    print("  - PitchShifter: Shift audio pitch by semitones")
    print("  - augment_audio_for_voice_conversion: Random augmentation")
    print("  - SyntheticVoiceConversionAugmentation: nn.Module wrapper")
    print("\nUsage:")
    print("  from snac.audio_augmentation import augment_audio_for_voice_conversion")
    print("  ")
    print("  # Apply augmentation with 50% probability")
    print("  audio_aug, was_aug, semitones = augment_audio_for_voice_conversion(")
    print("      audio,")
    print("      pitch_shift_range=[-2, -1, 1, 2],")
    print("      probability=0.5")
    print("  )")

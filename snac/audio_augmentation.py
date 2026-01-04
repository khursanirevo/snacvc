#!/usr/bin/env python3
"""
Audio Augmentation for Synthetic Voice Conversion.

Creates pseudo-speaker pairs by applying pitch shifting to audio.
This teaches the model that speaker embeddings should dominate over
acoustic patterns in codes.

Key insight:
- Pitch-shifted audio contains "wrong" speaker acoustic patterns
- When decoded with original speaker embedding, should reconstruct original
- Forces model to trust embedding over codes for speaker identity
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T
import random
from typing import Tuple, Optional
import numpy as np


class PitchShifter:
    """
    Pitch shifting for audio augmentation.

    Uses resampling-based pitch shifting for efficiency.
    Preserves duration while changing pitch.
    """

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

        # Calculate resampling ratio for pitch shift
        # Pitch shift factor = 2^(semitones/12)
        pitch_factor = 2.0 ** (semitones / 12.0)

        # To preserve duration, we need to compensate the resampling
        # Resample up by pitch_factor, then down (or vice versa)
        if pitch_factor > 1.0:
            # Pitch up: resample up then down
            resampler_up = T.Resample(
                orig_freq=self.sample_rate,
                new_freq=int(self.sample_rate * pitch_factor),
                dtype=audio.dtype
            )
            resampler_down = T.Resample(
                orig_freq=int(self.sample_rate * pitch_factor),
                new_freq=self.sample_rate,
                dtype=audio.dtype
            )
            audio_shifted = resampler_down(resampler_up(audio))
        else:
            # Pitch down: resample down then up
            resampler_down = T.Resample(
                orig_freq=self.sample_rate,
                new_freq=int(self.sample_rate * pitch_factor),
                dtype=audio.dtype
            )
            resampler_up = T.Resample(
                orig_freq=int(self.sample_rate * pitch_factor),
                new_freq=self.sample_rate,
                dtype=audio.dtype
            )
            audio_shifted = resampler_up(resampler_down(audio))

        return audio_shifted

    def __call__(self, audio: torch.Tensor, semitones: float) -> torch.Tensor:
        return self.shift_semitones(audio, semitones)


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
    # Randomly decide whether to augment
    if random.random() > probability:
        return audio, False, 0.0

    # Random pitch shift amount
    semitones = random.choice(pitch_shift_range)

    # Apply pitch shift
    pitch_shifter = PitchShifter(sample_rate=24000)
    audio_aug = pitch_shifter(audio, semitones)

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

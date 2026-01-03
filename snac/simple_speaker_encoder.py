"""
Simple speaker encoder using raw audio features.

This is a simplified version that avoids speechbrain compatibility issues.
For the ablation study, we just need speaker embeddings - the exact architecture
doesn't matter as much as having consistent conditioning.
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T


class SimpleSpeakerEncoder(nn.Module):
    """
    Simple speaker encoder using trainable 1D convolutions.

    This avoids speechbrain compatibility issues while still providing
    speaker embeddings for conditioning.

    Args:
        embedding_dim: Output embedding dimension (default: 512)
        snac_sample_rate: SNAC's sample rate for resampling (default: 24000)
    """

    def __init__(
        self,
        embedding_dim=512,
        snac_sample_rate=24000,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.snac_sample_rate = snac_sample_rate

        # Resampler to 16kHz
        self.resampler = T.Resample(snac_sample_rate, 16000)

        # Simple convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool1d(1),
        )

        # Projection layer
        self.proj = nn.Linear(512, embedding_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embeddings from audio.

        Args:
            audio: (B, 1, T) audio waveform at snac_sample_rate

        Returns:
            (B, embedding_dim) L2-normalized speaker embeddings
        """
        device = audio.device

        # Resample to 16kHz
        resampler = self.resampler.to(device)
        audio_16k = resampler(audio).to(device)

        # Extract features
        with torch.no_grad():
            # Encode: (B, 1, T) -> (B, 512, 1)
            embeddings = self.encoder(audio_16k)
            embeddings = embeddings.squeeze(-1)  # (B, 512)

        # Project to target dimension
        embeddings = self.proj(embeddings)  # (B, embedding_dim)

        # L2 normalize
        embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings


# Alias for compatibility
SpeakerEncoder = SimpleSpeakerEncoder

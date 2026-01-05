#!/usr/bin/env python3
"""
Whisper-based Content Preservation Loss.

Uses OpenAI's Whisper model to ensure linguistic content is preserved
during voice conversion. This prevents the model from accidentally
changing words or phonemes while transforming speaker characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from typing import Optional


class WhisperContentLoss(nn.Module):
    """
    Content preservation loss using Whisper embeddings.

    Ensures that the linguistic content (phonemes, words) remains
    consistent after voice conversion by comparing Whisper embeddings.
    """

    def __init__(self, model_size: str = "large-v3", device: str = "cuda"):
        """
        Initialize Whisper model for content loss.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            device: Device to load model on
        """
        super().__init__()
        self.device = device
        self.model_size = model_size

        print(f"Loading Whisper {model_size} for content loss...")

        try:
            import whisper
            self.whisper = whisper.load_model(model_size, device=device)
            self.whisper.eval()

            # Freeze Whisper parameters (we only use it for feature extraction)
            for param in self.whisper.parameters():
                param.requires_grad = False

            print(f"✅ Whisper {model_size} loaded for content preservation")
        except ImportError:
            print("❌ Whisper not installed. Install with: pip install openai-whisper")
            raise ImportError("openai-whisper package required for content loss")

        # Get embedding dimension from Whisper model config
        # Whisper's encoder output dimension is model.dims (1280 for large-v3)
        self.embedding_dim = self.whisper.dims
        print(f"Whisper embedding dimension: {self.embedding_dim}")

    def extract_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract Whisper encoder embedding from audio.

        Args:
            audio: (B, 1, T) audio tensor at any sample rate

        Returns:
            (B, D) embeddings where D is Whisper's embedding dimension
        """
        import whisper

        # Convert to 16kHz (Whisper's native sample rate)
        if audio.shape[-1] < 16000:
            # Pad if too short
            audio = F.pad(audio, (0, 16000 - audio.shape[-1]))

        # Resample if needed (assuming 24kHz input)
        if audio.shape[-1] > 16000 or audio.shape[-1] % 16000 != 0:
            # Take first 30 seconds max (Whisper's limit)
            audio = audio[:, :, :min(30 * 24000, audio.shape[-1])]

            # Resample to 16kHz
            resampler = torchaudio.transforms.Resample(
                orig_freq=24000,
                new_freq=16000
            ).to(audio.device)
            audio_16k = resampler(audio.squeeze(1))
        else:
            audio_16k = audio.squeeze(1)

        B = audio_16k.shape[0]
        embeddings_list = []

        for i in range(B):
            # Pad or trim to 30 seconds (Whisper's max)
            audio_single = whisper.pad_or_trim(audio_16k[i])

            # Compute log mel spectrogram
            mel = whisper.log_mel_spectrogram(audio_single)

            # Extract encoder embeddings
            with torch.no_grad():
                encoder_output = self.whisper.encoder(mel.to(self.device))

            # Global average pooling over time dimension
            embedding = encoder_output.mean(dim=0)  # (D,)

            embeddings_list.append(embedding)

        return torch.stack(embeddings_list)  # (B, D)

    def forward(self, audio_original: torch.Tensor, audio_reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Compute content preservation loss between original and reconstructed audio.

        Args:
            audio_original: (B, 1, T) original audio
            audio_reconstructed: (B, 1, T) reconstructed audio

        Returns:
            Content loss (lower is better)
        """
        # Extract embeddings
        emb_original = self.extract_embedding(audio_original)
        emb_reconstructed = self.extract_embedding(audio_reconstructed)

        # Normalize embeddings
        emb_original = F.normalize(emb_original, dim=-1)
        emb_reconstructed = F.normalize(emb_reconstructed, dim=-1)

        # Cosine similarity loss (we want high similarity)
        similarity = F.cosine_similarity(emb_original, emb_reconstructed, dim=-1)

        # Convert to loss (1 - similarity)
        loss = (1.0 - similarity).mean()

        return loss


def content_loss(model: WhisperContentLoss, audio_original: torch.Tensor,
                audio_reconstructed: torch.Tensor, config: dict) -> torch.Tensor:
    """
    Compute content preservation loss.

    Args:
        model: WhisperContentLoss instance
        audio_original: Original audio
        audio_reconstructed: Reconstructed audio
        config: Training config

    Returns:
        Content loss value
    """
    weight = config.get('lambda_content', 0.2)
    loss = model(audio_original, audio_reconstructed)
    return weight * loss

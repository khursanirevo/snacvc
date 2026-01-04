"""
ERes2NetV2 speaker encoder from GPT-SoVITS ProPlus.

ERes2NetV2 is an enhanced Res2Net for speaker verification with
superior short-duration performance.

Reference: https://huggingface.co/spaces/lj1995/GPT-SoVITS-ProPlus
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T
import os
from .eres2net import ERes2NetV2


class ERes2NetSpeakerEncoder(nn.Module):
    """
    Speaker encoder using ERes2NetV2 from GPT-SoVITS.

    Args:
        embedding_dim: Output embedding dimension (default: 512)
        freeze: If True, keep ERes2NetV2 frozen (default: True)
        snac_sample_rate: SNAC's sample rate for resampling (default: 24000)
        checkpoint_path: Path to pretrained weights (default: auto-download)
        baseWidth: ERes2NetV2 baseWidth (24 for GPT-SoVITS)
        scale: ERes2NetV2 scale (4 for GPT-SoVITS)
        expansion: ERes2NetV2 expansion (4 for GPT-SoVITS)

    Input:
        audio: (B, 1, T) audio waveform at SNAC's sample rate

    Output:
        (B, embedding_dim) L2-normalized speaker embeddings
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        freeze: bool = True,
        snac_sample_rate: int = 24000,
        checkpoint_path: str = None,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.snac_sample_rate = snac_sample_rate

        # ERes2NetV2 configuration from GPT-SoVITS ProPlus
        # Model: pretrained_eres2netv2w24s4ep4.ckpt
        # IMPORTANT: These parameters MUST match the checkpoint exactly!
        self.model = ERes2NetV2(
            baseWidth=24,        # CRITICAL: Must be 24 to match checkpoint
            scale=4,             # CRITICAL: Must be 4 to match checkpoint
            expansion=4,         # CRITICAL: Must be 4 to match checkpoint
            num_blocks=[3, 4, 6, 3],
            m_channels=64,
            feat_dim=80,
            embedding_size=192,  # CRITICAL: Checkpoint outputs 192-dim
            pooling_func='TSTP',
        )

        # Load pretrained weights - MUST succeed or crash
        if checkpoint_path is None:
            checkpoint_path = "pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"ERes2NetV2 checkpoint not found at {checkpoint_path}\n"
                f"Download with: uv run python scripts/download_eres2net.py"
            )

        # Load checkpoint and CRASH if it doesn't match
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=True)

            if missing or unexpected:
                raise RuntimeError(
                    f"ERes2NetV2 checkpoint architecture mismatch!\n"
                    f"Missing keys: {len(missing)}\n"
                    f"Unexpected keys: {len(unexpected)}\n"
                    f"This means the model architecture doesn't match the checkpoint.\n"
                    f"CRITICAL: Cannot continue training without proper pretrained weights!"
                )

            print(f"✅ Successfully loaded ERes2NetV2 from {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(
                f"CRITICAL: Failed to load ERes2NetV2 checkpoint properly!\n"
                f"Error: {e}\n"
                f"Training CANNOT continue without pretrained speaker encoder!"
            ) from e

        # Resampler: SNAC sample rate -> 16kHz (ERes2NetV2 expects 16kHz)
        self.resampler = T.Resample(snac_sample_rate, 16000)

        # Mel spectrogram extractor (Kaldi fbank style: 80 mel bins)
        self.mel_extractor = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            hop_length=160,  # 10ms
            n_mels=80,
            f_min=0,
            f_max=8000,
            power=2.0,
        )

        # Projection: 192-dim (ERes2NetV2 output) -> embedding_dim (512)
        # ERes2NetV2 outputs 192-dim, we need to project to 512-dim
        self.proj = nn.Linear(192, embedding_dim)

        # Freeze ERes2NetV2 (we don't want to finetune the speaker encoder)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            self.freeze_encoder = True
        else:
            self.freeze_encoder = False

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalized speaker embeddings from audio.

        Args:
            audio: (B, 1, T) audio waveform at self.snac_sample_rate

        Returns:
            (B, embedding_dim) L2-normalized speaker embeddings
        """
        device = audio.device

        # Resample from SNAC sample rate to 16kHz
        resampler = self.resampler.to(device)
        audio_16k = resampler(audio)  # (B, 1, T_16k)

        # Extract mel spectrograms (Kaldi fbank style)
        mel_extractor = self.mel_extractor.to(device)

        # Remove channel dim for mel extraction: (B, 1, T) -> (B, T)
        audio_16k = audio_16k.squeeze(1)

        # Compute mel spectrograms: (B, T) -> (B, 80, T')
        mel_specs = mel_extractor(audio_16k)

        # Convert to log scale (Kaldi fbank uses log)
        mel_specs = torch.log(mel_specs + 1e-8)  # Add small value to avoid log(0)

        # Convert to (B, T', 80) format for ERes2NetV2
        mel_specs = mel_specs.transpose(1, 2)  # (B, 80, T') -> (B, T', 80)

        # Extract embeddings using ERes2NetV2 forward() method
        # forward() uses TSTP pooling + seg_1 layer to output 192-dim
        if self.freeze_encoder:
            with torch.no_grad():
                embeddings = self.model(mel_specs)  # (B, 192)
        else:
            embeddings = self.model(mel_specs)  # (B, 192)

        # Project from 192-dim to target embedding dimension (512)
        embeddings = self.proj(embeddings)  # (B, embedding_dim)

        # L2 normalize for stability and cosine similarity
        embeddings = nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings

    @torch.no_grad()
    def extract_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for extracting a single speaker embedding.

        Args:
            audio: (B, 1, T) audio waveform

        Returns:
            (B, embedding_dim) speaker embedding
        """
        return self.forward(audio)

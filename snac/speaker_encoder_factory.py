"""
Speaker encoder factory for configurable encoder selection.

Supports multiple pretrained speaker encoders:
- ERes2NetV2 (GPT-SoVITS) - default, best performance
- ECAPA-TDNN (SpeechBrain) - alternative, good performance

CRITICAL: All encoders MUST use pretrained weights. Training will crash if weights fail to load.
"""

from typing import Optional
import torch.nn as nn


class SpeakerEncoderFactory:
    """Factory for creating speaker encoder instances."""

    _encoders = {
        'eres2net': {
            'class': 'snac.eres2net_encoder.ERes2NetSpeakerEncoder',
            'description': 'ERes2NetV2 from GPT-SoVITS (pretrained, frozen, 34M params)',
            'default': True,
        },
        'ecapa': {
            'class': 'snac.speaker_encoder.SpeakerEncoder',
            'description': 'ECAPA-TDNN from SpeechBrain (pretrained, frozen, 20M params)',
            'default': False,
        },
    }

    @classmethod
    def create(
        cls,
        encoder_type: str = 'eres2net',
        embedding_dim: int = 512,
        snac_sample_rate: int = 24000,
        freeze: bool = True,
        **kwargs
    ) -> nn.Module:
        """
        Create a speaker encoder instance.

        Args:
            encoder_type: Type of encoder ('eres2net', 'ecapa')
            embedding_dim: Output embedding dimension
            snac_sample_rate: SNAC model's sample rate
            freeze: Whether to freeze encoder weights (always True for pretrained)
            **kwargs: Additional encoder-specific arguments

        Returns:
            Speaker encoder module

        Raises:
            ValueError: If encoder_type is not recognized
            RuntimeError: If pretrained weights fail to load

        CRITICAL: Training will CRASH if pretrained weights cannot be loaded!
        """
        encoder_type = encoder_type.lower()

        if encoder_type not in cls._encoders:
            available = ', '.join(cls._encoders.keys())
            raise ValueError(
                f"Unknown speaker encoder type: '{encoder_type}'. "
                f"Available types: {available}"
            )

        # Import and instantiate encoder
        if encoder_type == 'eres2net':
            from .eres2net_encoder import ERes2NetSpeakerEncoder
            return ERes2NetSpeakerEncoder(
                embedding_dim=embedding_dim,
                freeze=freeze,
                snac_sample_rate=snac_sample_rate,
                **kwargs
            )

        elif encoder_type == 'ecapa':
            from .speaker_encoder import SpeakerEncoder as ECAPAEncoder
            return ECAPAEncoder(
                embedding_dim=embedding_dim,
                freeze=freeze,
                snac_sample_rate=snac_sample_rate
            )

        else:
            raise ValueError(f"Unhandled encoder type: {encoder_type}")

    @classmethod
    def list_encoders(cls) -> dict:
        """Return information about available encoders."""
        return cls._encoders.copy()

    @classmethod
    def get_default_encoder(cls) -> str:
        """Return the default encoder type."""
        for name, info in cls._encoders.items():
            if info.get('default', False):
                return name
        return 'eres2net'  # Fallback

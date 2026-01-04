import json
import math
import os
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from .layers import Encoder, Decoder
from .vq import ResidualVectorQuantize


class SNAC(nn.Module):
    def __init__(
        self,
        sampling_rate=44100,
        encoder_dim=64,
        encoder_rates=[3, 3, 7, 7],
        latent_dim=None,
        decoder_dim=1536,
        decoder_rates=[7, 7, 3, 3],
        attn_window_size=32,
        codebook_size=4096,
        codebook_dim=8,
        vq_strides=[8, 4, 2, 1],
        noise=True,
        depthwise=True,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim
        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(
            encoder_dim,
            encoder_rates,
            depthwise=depthwise,
            attn_window_size=attn_window_size,
        )
        self.n_codebooks = len(vq_strides)
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.vq_strides = vq_strides
        self.attn_window_size = attn_window_size
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            vq_strides=vq_strides,
        )
        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            noise,
            depthwise=depthwise,
            attn_window_size=attn_window_size,
        )

    def preprocess(self, audio_data):
        length = audio_data.shape[-1]
        lcm = math.lcm(self.vq_strides[0], self.attn_window_size or 1)
        pad_to = self.hop_length * lcm
        right_pad = math.ceil(length / pad_to) * pad_to - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        return audio_data

    def forward(self, audio_data: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data)
        z = self.encoder(audio_data)
        z_q, codes = self.quantizer(z)
        audio_hat = self.decoder(z_q)
        return audio_hat[..., :length], codes

    def encode(self, audio_data: torch.Tensor) -> List[torch.Tensor]:
        audio_data = self.preprocess(audio_data)
        z = self.encoder(audio_data)
        _, codes = self.quantizer(z)
        return codes

    def decode(self, codes: List[torch.Tensor]) -> torch.Tensor:
        z_q = self.quantizer.from_codes(codes)
        audio_hat = self.decoder(z_q)
        return audio_hat

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        model = cls(**config)
        return model

    @classmethod
    def from_pretrained(cls, repo_id, **kwargs):
        from huggingface_hub import hf_hub_download

        if not os.path.isdir(repo_id):
            config_path = hf_hub_download(repo_id=repo_id, filename="config.json", **kwargs)
            model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", **kwargs)
            model = cls.from_config(config_path)
            state_dict = torch.load(model_path, map_location="cpu")
        else:
            model = cls.from_config(os.path.join(repo_id, "config.json"))
            state_dict = torch.load(os.path.join(repo_id, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model


class SNACWithSpeakerConditioning(nn.Module):
    """
    SNAC with speaker conditioning using FiLM layers.

    This model wraps a pretrained SNAC model and adds speaker conditioning
    to the decoder using FiLM (Feature-wise Linear Modulation) layers.
    The base SNAC model is frozen, and only the FiLM parameters are trained.

    Args:
        sampling_rate: Audio sample rate (default: from pretrained)
        encoder_dim: Encoder dimension
        encoder_rates: Encoder downsampling strides
        latent_dim: Latent dimension
        decoder_dim: Decoder dimension
        decoder_rates: Decoder upsampling strides
        attn_window_size: Attention window size
        codebook_size: VQ codebook size
        codebook_dim: VQ codebook dimension
        vq_strides: VQ strides for hierarchical quantization
        noise: Whether to use noise in decoder
        depthwise: Whether to use depthwise convolutions
        speaker_emb_dim: Dimension of speaker embedding (default: 512)
        freeze_base: Whether to freeze base SNAC model (default: True)
    """

    def __init__(
        self,
        sampling_rate=44100,
        encoder_dim=64,
        encoder_rates=[3, 3, 7, 7],
        latent_dim=None,
        decoder_dim=1536,
        decoder_rates=[7, 7, 3, 3],
        attn_window_size=32,
        codebook_size=4096,
        codebook_dim=8,
        vq_strides=[8, 4, 2, 1],
        noise=True,
        depthwise=True,
        speaker_emb_dim=512,
        speaker_encoder_type='ecapa',  # NEW: Configurable speaker encoder (ECAPA-TDNN default)
        freeze_base=True,
    ):
        super().__init__()

        # Initialize base SNAC model
        self.base_model = SNAC(
            sampling_rate=sampling_rate,
            encoder_dim=encoder_dim,
            encoder_rates=encoder_rates,
            latent_dim=latent_dim,
            decoder_dim=decoder_dim,
            decoder_rates=decoder_rates,
            attn_window_size=attn_window_size,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            vq_strides=vq_strides,
            noise=noise,
            depthwise=depthwise,
        )

        # Copy relevant attributes
        self.sampling_rate = sampling_rate
        self.latent_dim = self.base_model.latent_dim
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.hop_length = self.base_model.hop_length
        self.attn_window_size = attn_window_size
        self.vq_strides = vq_strides
        self.n_codebooks = len(vq_strides)
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        # Freeze base model parameters
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
            self.base_model.eval()

        # Speaker encoder (pretrained, frozen)
        # Uses factory pattern to support multiple encoder types
        from .speaker_encoder_factory import SpeakerEncoderFactory
        self.speaker_encoder = SpeakerEncoderFactory.create(
            encoder_type=speaker_encoder_type,
            embedding_dim=speaker_emb_dim,
            snac_sample_rate=sampling_rate,
            freeze=True  # Always freeze pretrained encoders
        )

        # Build conditioned decoder
        self._build_conditioned_decoder(speaker_emb_dim)

    def _build_conditioned_decoder(self, cond_dim):
        """Build decoder with FiLM conditioning."""
        from .layers import DecoderBlockWithFiLM, WNConv1d

        channels = self.decoder_dim
        rates = self.decoder_rates
        d_out = 1
        input_channel = self.latent_dim

        # Initial layers (same as original Decoder)
        layers = [
            WNConv1d(input_channel, input_channel, kernel_size=7, padding=3, groups=input_channel),
            WNConv1d(input_channel, channels, kernel_size=1),
        ]

        if self.attn_window_size is not None:
            from .attention import LocalMHA
            layers += [LocalMHA(dim=channels, window_size=self.attn_window_size)]

        self.initial_layers = nn.Sequential(*layers)

        # DecoderBlocks with FiLM conditioning
        self.decoder_blocks = nn.ModuleList()
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            groups = output_dim  # depthwise=True

            self.decoder_blocks.append(
                DecoderBlockWithFiLM(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    stride=stride,
                    noise=True,
                    groups=groups,
                    cond_dim=cond_dim
                )
            )

        # Final layers
        from .layers import Snake1d
        self.final_layers = nn.Sequential(
            Snake1d(channels // 2**len(rates)),
            WNConv1d(channels // 2**len(rates), d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def extract_speaker_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from audio.

        Args:
            audio: (B, 1, T) audio waveform

        Returns:
            (B, speaker_emb_dim) L2-normalized speaker embedding
        """
        return self.speaker_encoder(audio)

    def preprocess(self, audio_data: torch.Tensor) -> torch.Tensor:
        """Preprocess audio using base model's preprocessing."""
        return self.base_model.preprocess(audio_data)

    def encode(self, audio_data: torch.Tensor) -> List[torch.Tensor]:
        """
        Encode audio to hierarchical codes using frozen encoder.

        Args:
            audio_data: (B, 1, T) audio waveform

        Returns:
            List of 4 code tensors at different temporal resolutions
        """
        audio_data = self.preprocess(audio_data)
        z = self.base_model.encoder(audio_data)
        _, codes = self.base_model.quantizer(z)
        return codes

    def decode(
        self,
        codes: List[torch.Tensor],
        speaker_embedding: torch.Tensor = None,
        reference_audio: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Decode codes with optional speaker conditioning.

        Args:
            codes: List of 4 code tensors from encode()
            speaker_embedding: Optional (B, speaker_emb_dim) speaker embedding
            reference_audio: Optional (B, 1, T) reference audio for speaker extraction

        Returns:
            (B, 1, T) reconstructed audio waveform
        """
        # Reconstruct latent from codes using frozen VQ decoder
        z_q = self.base_model.quantizer.from_codes(codes)

        # Extract speaker embedding if reference audio provided
        if speaker_embedding is None and reference_audio is not None:
            speaker_embedding = self.extract_speaker_embedding(reference_audio)

        if speaker_embedding is None:
            # Zero conditioning (no speaker modification)
            B = z_q.shape[0]
            cond_dim = self.decoder_blocks[0].res1.film.gamma_fc.out_features
            speaker_embedding = torch.zeros(B, cond_dim, device=z_q.device, dtype=z_q.dtype)

        # Pass through conditioned decoder
        x = self.initial_layers(z_q)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, speaker_embedding)

        audio_hat = self.final_layers(x)
        return audio_hat

    def forward(
        self,
        audio_data: torch.Tensor,
        speaker_embedding: torch.Tensor = None,
        reference_audio: torch.Tensor = None,
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Full encode-decode pass with speaker conditioning.

        Args:
            audio_data: (B, 1, T) input audio waveform
            speaker_embedding: Optional (B, speaker_emb_dim) speaker embedding
            reference_audio: Optional (B, 1, T) reference audio for speaker extraction

        Returns:
            audio_hat: (B, 1, T) reconstructed audio waveform
            codes: List of 4 code tensors
        """
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data)

        # Encode using frozen encoder
        z = self.base_model.encoder(audio_data)
        z_q, codes = self.base_model.quantizer(z)

        # Extract speaker embedding
        if speaker_embedding is None and reference_audio is not None:
            speaker_embedding = self.extract_speaker_embedding(reference_audio)
        if speaker_embedding is None:
            speaker_embedding = self.extract_speaker_embedding(audio_data)

        # Decode with conditioning
        x = self.initial_layers(z_q)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, speaker_embedding)

        audio_hat = self.final_layers(x)
        return audio_hat[..., :length], codes

    @classmethod
    def from_pretrained_base(
        cls,
        repo_id: str,
        speaker_emb_dim: int = 512,
        speaker_encoder_type: str = 'ecapa',  # NEW
        freeze_base: bool = True,
        **kwargs
    ):
        """
        Load pretrained SNAC and add speaker conditioning.

        Args:
            repo_id: HuggingFace repo ID (e.g., "hubertsiuzdak/snac_24khz")
            speaker_emb_dim: Dimension of speaker embedding
            speaker_encoder_type: Type of speaker encoder ('eres2net', 'ecapa', 'simple')
            freeze_base: Whether to freeze base SNAC parameters
            **kwargs: Additional arguments for SNAC constructor

        Returns:
            SNACWithSpeakerConditioning instance with pretrained weights
        """
        # Load base SNAC model
        base_model = SNAC.from_pretrained(repo_id, **kwargs)

        # Create conditioned model with same config
        model = cls(
            sampling_rate=base_model.sampling_rate,
            encoder_dim=base_model.encoder_dim,
            encoder_rates=base_model.encoder_rates,
            latent_dim=base_model.latent_dim,
            decoder_dim=base_model.decoder_dim,
            decoder_rates=base_model.decoder_rates,
            attn_window_size=base_model.attn_window_size,
            codebook_size=base_model.codebook_size,
            codebook_dim=base_model.codebook_dim,
            vq_strides=base_model.vq_strides,
            speaker_emb_dim=speaker_emb_dim,
            speaker_encoder_type=speaker_encoder_type,  # NEW
            freeze_base=freeze_base,
        )

        # Copy pretrained weights to base model
        model.base_model.load_state_dict(base_model.state_dict())

        return model

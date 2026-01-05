"""
Adapter layers for conditioning the encoder latent BEFORE quantization.

This allows the model to learn speaker-conditioned codes directly,
rather than trying to override speaker information at the decoder.
"""

import torch
import torch.nn as nn
from typing import Optional


class FiLMAdapter(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) adapter for conditioning latent features.

    Applies: gamma(speaker_emb) * features + beta(speaker_emb)

    This modulates the encoder's latent representation BEFORE quantization,
    allowing different speakers to produce different codes.
    """

    def __init__(
        self,
        latent_dim: int,
        speaker_emb_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2
    ):
        """
        Args:
            latent_dim: Dimension of latent features from encoder
            speaker_emb_dim: Dimension of speaker embedding
            hidden_dim: Hidden dimension for FiML projection
            num_layers: Number of MLP layers for computing gamma/beta
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.speaker_emb_dim = speaker_emb_dim

        # Build MLP for computing scale (gamma) and shift (beta)
        layers = []
        input_dim = speaker_emb_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim

        # Final layer outputs both gamma and beta
        self.mlp = nn.Sequential(*layers)

        # Output projections
        self.gamma_proj = nn.Linear(hidden_dim, latent_dim)
        self.beta_proj = nn.Linear(hidden_dim, latent_dim)

        # Initialize to identity transformation (no modulation)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation to latent features.

        Args:
            x: (B, D, T) latent features from encoder
            speaker_emb: (B, speaker_emb_dim) speaker embedding

        Returns:
            (B, D, T) modulated latent features
        """
        B, D, T = x.shape

        # Compute modulation parameters from speaker embedding
        h = self.mlp(speaker_emb)  # (B, hidden_dim)
        gamma = self.gamma_proj(h)  # (B, latent_dim)
        beta = self.beta_proj(h)    # (B, latent_dim)

        # Reshape for broadcasting: (B, D, 1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        # Apply FiLM modulation
        x_modulated = gamma * x + beta

        return x_modulated

    def identity_regularization(self, x: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute identity regularization loss.

        Penalizes deviation from identity transformation:
            ||adapter(x, emb) - x||^2

        This encourages the adapter to stay close to identity unless
        there's a strong reason to modulate (e.g., voice conversion).

        Args:
            x: (B, D, T) latent features from encoder
            speaker_emb: (B, speaker_emb_dim) speaker embedding

        Returns:
            Scalar loss value
        """
        x_modulated = self.forward(x, speaker_emb)
        loss = torch.nn.functional.mse_loss(x_modulated, x)
        return loss


class AdapterWrapper(nn.Module):
    """
    Wrapper that applies adapter layers before quantization.

    Architecture:
        Encoder Output → Adapter(speaker_emb) → Modulated Latent → VQ → Codes
    """

    def __init__(
        self,
        base_model,
        adapter_type: str = "film",
        adapter_hidden_dim: int = 512,
        adapter_num_layers: int = 2
    ):
        """
        Args:
            base_model: SNAC base model (encoder + VQ + decoder)
            adapter_type: Type of adapter ('film')
            adapter_hidden_dim: Hidden dimension for adapter
            adapter_num_layers: Number of layers in adapter MLP
        """
        super().__init__()

        self.base_model = base_model
        self.adapter_type = adapter_type

        # Get latent dimension from encoder output
        # SNACWithSpeakerConditioning has latent_dim attribute
        latent_dim = base_model.latent_dim

        if adapter_type == "film":
            self.adapter = FiLMAdapter(
                latent_dim=latent_dim,
                speaker_emb_dim=base_model.speaker_emb_dim,
                hidden_dim=adapter_hidden_dim,
                num_layers=adapter_num_layers
            )
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

        # Expose key attributes for compatibility
        self.sampling_rate = base_model.sampling_rate
        self.latent_dim = base_model.latent_dim
        self.speaker_emb_dim = base_model.speaker_emb_dim
        self.n_codebooks = base_model.n_codebooks

    def forward(
        self,
        audio_data: torch.Tensor,
        speaker_embedding: torch.Tensor = None
    ) -> tuple[torch.Tensor, list]:
        """
        Forward pass with adapter conditioning BEFORE VQ.

        Args:
            audio_data: (B, 1, T) input audio
            speaker_embedding: (B, speaker_emb_dim) speaker embedding

        Returns:
            audio_hat: (B, 1, T) reconstructed audio
            codes: List of code tensors
        """
        length = audio_data.shape[-1]

        # Access underlying SNAC model (base_model.base_model is the actual SNAC)
        snac_model = self.base_model.base_model

        # Preprocess audio
        audio_data = snac_model.preprocess(audio_data)

        # Encode to latent
        z = snac_model.encoder(audio_data)

        # Apply adapter conditioning BEFORE VQ
        if speaker_embedding is not None:
            z = self.adapter(z, speaker_embedding)
        else:
            # No conditioning, use latent as-is
            pass

        # Quantize to codes
        z_q, codes = snac_model.quantizer(z)

        # Decode (no additional conditioning needed here)
        # Note: decoder is frozen, so it just processes the conditioned latent
        audio_hat = snac_model.decoder(z_q)

        # Trim to original length
        audio_hat = audio_hat[..., :length]

        return audio_hat, codes

    def encode(
        self,
        audio_data: torch.Tensor,
        speaker_embedding: torch.Tensor = None
    ) -> list:
        """
        Encode audio with speaker conditioning.

        Args:
            audio_data: (B, 1, T) input audio
            speaker_embedding: (B, speaker_emb_dim) speaker embedding

        Returns:
            List of code tensors
        """
        # Access underlying SNAC model
        snac_model = self.base_model.base_model

        # Preprocess
        audio_data = snac_model.preprocess(audio_data)

        # Encode
        z = snac_model.encoder(audio_data)

        # Apply adapter BEFORE VQ
        if speaker_embedding is not None:
            z = self.adapter(z, speaker_embedding)

        # Quantize to codes
        _, codes = snac_model.quantizer(z)

        return codes

    def decode(
        self,
        codes: list,
        speaker_embedding: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Decode codes (no conditioning needed at decoder).

        Args:
            codes: List of code tensors
            speaker_embedding: Not used (kept for API compatibility)

        Returns:
            (B, 1, T) decoded audio
        """
        # Access underlying SNAC model
        snac_model = self.base_model.base_model

        # Reconstruct latent from codes
        z_q = snac_model.quantizer.from_codes(codes)

        # Decode (decoder is frozen, processes conditioned latent)
        audio_hat = snac_model.decoder(z_q)

        return audio_hat

    def extract_speaker_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from audio.

        Args:
            audio: (B, 1, T) audio tensor

        Returns:
            (B, speaker_emb_dim) speaker embedding
        """
        return self.base_model.extract_speaker_embedding(audio)

    def adapter_identity_regularization(
        self,
        audio: torch.Tensor,
        speaker_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adapter identity regularization loss.

        Penalizes deviation from identity transformation at the latent level:
            ||adapter(encoder(audio), emb) - encoder(audio)||^2

        This forces the adapter to stay close to identity unless there's
        strong incentive to modulate (e.g., voice conversion).

        Args:
            audio: (B, 1, T) input audio
            speaker_embedding: (B, speaker_emb_dim) speaker embedding

        Returns:
            Scalar regularization loss
        """
        # Access underlying SNAC model
        snac_model = self.base_model.base_model

        # Preprocess
        audio_prep = snac_model.preprocess(audio)

        # Encode to latent
        z = snac_model.encoder(audio_prep)

        # Compute identity regularization via adapter
        reg_loss = self.adapter.identity_regularization(z, speaker_embedding)

        return reg_loss


class MultiStageAdapterWrapper(nn.Module):
    """
    Multi-stage adapter that applies FiLM modulation at multiple encoder levels.

    Architecture:
        Audio → Conv → Block1 → Adapter1 → Block2 → Adapter2 → Block3 → Adapter3
              → Block4 → Adapter4 → MHA → Adapter5 → VQ → Codes

    Each adapter modulates features at a different scale:
    - Adapter1 (128-dim): Low-level features (timbre, spectral)
    - Adapter2 (256-dim): Mid-level features (formants, patterns)
    - Adapter3 (512-dim): Speaker patterns emerge
    - Adapter4 (1024-dim): Speaker identity
    - Adapter5 (1024-dim): Global context refinement

    This progressive modulation is easier to learn and less disruptive than
    a single large modulation at the end.
    """

    def __init__(
        self,
        base_model,
        adapter_hidden_dim: int = 512,
        adapter_num_layers: int = 2,
        adaptive_init: bool = True
    ):
        """
        Args:
            base_model: SNAC base model (encoder + VQ + decoder)
            adapter_hidden_dim: Hidden dimension for adapter MLPs
            adapter_num_layers: Number of layers in each adapter MLP
            adaptive_init: Whether to use adaptive initialization (early adapters smaller)
        """
        super().__init__()

        self.base_model = base_model
        self.adaptive_init = adaptive_init

        # Get dimensions from base model
        self.speaker_emb_dim = base_model.speaker_emb_dim

        # Create adapters for each encoder stage
        # Stage dimensions based on encoder architecture from hubertsiuzdak/snac_24khz:
        # Conv: 48 → Block1: 96 → Block2: 192 → Block3: 384 → Block4: 768 → Conv: 768
        self.adapters = nn.ModuleList([
            FiLMAdapter(48, self.speaker_emb_dim, adapter_hidden_dim, adapter_num_layers),    # After initial conv
            FiLMAdapter(96, self.speaker_emb_dim, adapter_hidden_dim, adapter_num_layers),    # After Block 1
            FiLMAdapter(192, self.speaker_emb_dim, adapter_hidden_dim, adapter_num_layers),   # After Block 2
            FiLMAdapter(384, self.speaker_emb_dim, adapter_hidden_dim, adapter_num_layers),   # After Block 3
            FiLMAdapter(768, self.speaker_emb_dim, adapter_hidden_dim, adapter_num_layers),   # After Block 4
        ])

        if adaptive_init:
            self._apply_adaptive_initialization()

        # Expose key attributes for compatibility
        self.sampling_rate = base_model.sampling_rate
        self.latent_dim = base_model.latent_dim
        self.n_codebooks = base_model.n_codebooks

    def _apply_adaptive_initialization(self):
        """
        Initialize adapters with progressively increasing modulation strength.

        Early adapters: Smaller weights (gentler modulation, preserve content)
        Late adapters: Larger weights (stronger modulation, shift speaker identity)

        Updated scales: Increased to promote learning in Stage 1 (0.5→1.0 instead of 0.1→1.0)
        """
        # Progressive scaling factors (increased from original to promote learning)
        scales = [0.5, 0.7, 0.85, 1.0, 1.0]

        for i, adapter in enumerate(self.adapters):
            scale = scales[i]

            # Scale gamma weights (modulation strength)
            adapter.gamma_proj.weight.data *= scale
            adapter.gamma_proj.bias.data = adapter.gamma_proj.bias.data * scale + (1 - scale)

            # Scale beta weights (shift amount)
            adapter.beta_proj.weight.data *= scale

    def forward(
        self,
        audio_data: torch.Tensor,
        speaker_embedding: torch.Tensor = None
    ) -> tuple[torch.Tensor, list]:
        """
        Forward pass with multi-stage adapter conditioning.

        Manually runs encoder block by block, applying adapter modulation
        after each stage.

        Args:
            audio_data: (B, 1, T) input audio
            speaker_embedding: (B, speaker_emb_dim) speaker embedding

        Returns:
            audio_hat: (B, 1, T) reconstructed audio
            codes: List of code tensors
        """
        length = audio_data.shape[-1]

        # Access underlying SNAC model
        snac_model = self.base_model.base_model

        # Preprocess audio
        audio_data = snac_model.preprocess(audio_data)

        # Manually run encoder with modulations at each stage
        encoder_blocks = snac_model.encoder.block

        # Initial convolution
        x = encoder_blocks[0](audio_data)  # (B, 64, T)

        # Apply adapter after initial conv
        if speaker_embedding is not None:
            x = self.adapters[0](x, speaker_embedding)

        # Encoder Block 1 (stride 3): 64 → 128
        x = encoder_blocks[1](x)  # (B, 128, T/3)
        if speaker_embedding is not None:
            x = self.adapters[1](x, speaker_embedding)

        # Encoder Block 2 (stride 3): 128 → 256
        x = encoder_blocks[2](x)  # (B, 256, T/9)
        if speaker_embedding is not None:
            x = self.adapters[2](x, speaker_embedding)

        # Encoder Block 3 (stride 7): 256 → 512
        x = encoder_blocks[3](x)  # (B, 512, T/63)
        if speaker_embedding is not None:
            x = self.adapters[3](x, speaker_embedding)

        # Encoder Block 4 (stride 7): 512 → 1024
        x = encoder_blocks[4](x)  # (B, 1024, T/441)
        if speaker_embedding is not None:
            x = self.adapters[4](x, speaker_embedding)

        # Local Multi-Head Attention (no adapter after MHA, let it process modulated features)
        if len(encoder_blocks) > 5:
            x = encoder_blocks[5](x)  # (B, 1024, T/441)

        # Quantize to codes
        z_q, codes = snac_model.quantizer(x)

        # Decode
        audio_hat = snac_model.decoder(z_q)

        # Trim to original length
        audio_hat = audio_hat[..., :length]

        return audio_hat, codes

    def encode(
        self,
        audio_data: torch.Tensor,
        speaker_embedding: torch.Tensor = None
    ) -> list:
        """
        Encode audio with multi-stage speaker conditioning.

        Args:
            audio_data: (B, 1, T) input audio
            speaker_embedding: (B, speaker_emb_dim) speaker embedding

        Returns:
            List of code tensors
        """
        # Access underlying SNAC model
        snac_model = self.base_model.base_model

        # Preprocess
        audio_data = snac_model.preprocess(audio_data)

        # Manually run encoder with modulations
        encoder_blocks = snac_model.encoder.block

        x = encoder_blocks[0](audio_data)
        if speaker_embedding is not None:
            x = self.adapters[0](x, speaker_embedding)

        x = encoder_blocks[1](x)
        if speaker_embedding is not None:
            x = self.adapters[1](x, speaker_embedding)

        x = encoder_blocks[2](x)
        if speaker_embedding is not None:
            x = self.adapters[2](x, speaker_embedding)

        x = encoder_blocks[3](x)
        if speaker_embedding is not None:
            x = self.adapters[3](x, speaker_embedding)

        x = encoder_blocks[4](x)
        if speaker_embedding is not None:
            x = self.adapters[4](x, speaker_embedding)

        if len(encoder_blocks) > 5:
            x = encoder_blocks[5](x)

        # Quantize to codes
        _, codes = snac_model.quantizer(x)

        return codes

    def decode(
        self,
        codes: list,
        speaker_embedding: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Decode codes (no conditioning needed at decoder).

        Args:
            codes: List of code tensors
            speaker_embedding: Not used (kept for API compatibility)

        Returns:
            (B, 1, T) decoded audio
        """
        # Access underlying SNAC model
        snac_model = self.base_model.base_model

        # Reconstruct latent from codes
        z_q = snac_model.quantizer.from_codes(codes)

        # Decode
        audio_hat = snac_model.decoder(z_q)

        return audio_hat

    def extract_speaker_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract speaker embedding from audio.

        Args:
            audio: (B, 1, T) audio tensor

        Returns:
            (B, speaker_emb_dim) speaker embedding
        """
        return self.base_model.extract_speaker_embedding(audio)

    def adapter_identity_regularization(
        self,
        audio: torch.Tensor,
        speaker_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-stage adapter identity regularization.

        Sum of identity losses from all adapter stages.
        Penalizes deviation from identity at each stage.

        Args:
            audio: (B, 1, T) input audio
            speaker_embedding: (B, speaker_emb_dim) speaker embedding

        Returns:
            Scalar regularization loss (sum over all stages)
        """
        snac_model = self.base_model.base_model
        audio_prep = snac_model.preprocess(audio)

        # Get intermediate features from encoder
        encoder_blocks = snac_model.encoder.block

        # Collect features at each stage
        features = []
        x = encoder_blocks[0](audio_prep)
        features.append(x)

        x = encoder_blocks[1](x)
        features.append(x)

        x = encoder_blocks[2](x)
        features.append(x)

        x = encoder_blocks[3](x)
        features.append(x)

        x = encoder_blocks[4](x)
        features.append(x)

        # Compute identity loss for each adapter
        total_loss = 0.0
        for i, (feat, adapter) in enumerate(zip(features, self.adapters)):
            stage_loss = adapter.identity_regularization(feat, speaker_embedding)
            total_loss = total_loss + stage_loss

        return total_loss

    def get_num_adapters(self) -> int:
        """Return number of adapter stages."""
        return len(self.adapters)

    def get_adapter_params(self) -> int:
        """Return total number of trainable parameters in all adapters."""
        return sum(p.numel() for adapter in self.adapters for p in adapter.parameters())


def add_adapter_to_model(
    model,
    adapter_type: str = "film",
    adapter_hidden_dim: int = 512,
    adapter_num_layers: int = 2
):
    """
    Wrap a SNACWithSpeakerConditioning model with adapter layers.

    Args:
        model: SNACWithSpeakerConditioning instance
        adapter_type: Type of adapter
        adapter_hidden_dim: Hidden dimension
        adapter_num_layers: Number of adapter layers

    Returns:
        AdapterWrapper instance
    """
    return AdapterWrapper(
        base_model=model.base_model,
        adapter_type=adapter_type,
        adapter_hidden_dim=adapter_hidden_dim,
        adapter_num_layers=adapter_num_layers
    )


class HybridVCWrapper(nn.Module):
    """
    Hybrid Voice Conversion Wrapper: Encoder Adapter + Decoder FiLM
    
    Architecture:
        Audio → Encoder → Adapter1 → ... → Adapter5 → VQ → codes
                                                                ↓
        Decoder ← FiLM(speaker_emb) ← latent_from_codes
        ↓
        audio_hat (target speaker)
    
    Key insight: Two-stage conditioning
    1. Encoder adapters (before VQ): Shift codes toward target speaker
    2. Decoder FiLM (after VQ): Guide decoder to produce target speaker
    
    This combines the strengths of both approaches:
    - Codes already contain speaker info (from encoder adapters)
    - Decoder knows how to handle them (from FiLM guidance)
    """
    
    def __init__(
        self,
        base_model,
        encoder_adapter_hidden_dim: int = 512,
        encoder_adapter_num_layers: int = 2,
        decoder_film_hidden_dim: int = 512,
        decoder_film_num_layers: int = 2,
        adaptive_init: bool = True,
    ):
        super().__init__()
        
        self.base_model = base_model
        self.speaker_emb_dim = base_model.speaker_encoder.embedding_dim
        self.latent_dim = base_model.latent_dim
        
        # Multi-stage encoder adapters (reuse existing)
        from snac.adapters import MultiStageAdapterWrapper
        self.encoder_adapter = MultiStageAdapterWrapper(
            base_model=base_model,
            adapter_hidden_dim=encoder_adapter_hidden_dim,
            adapter_num_layers=encoder_adapter_num_layers,
            adaptive_init=adaptive_init,
        )
        
        # Decoder FiLM adapter
        self.decoder_film = FiLMAdapter(
            latent_dim=self.latent_dim,
            speaker_emb_dim=self.speaker_emb_dim,
            hidden_dim=decoder_film_hidden_dim,
            num_layers=decoder_film_num_layers,
        )
        
        # Expose attributes for compatibility
        self.sampling_rate = base_model.sampling_rate
        self.latent_dim = base_model.latent_dim
        self.n_codebooks = base_model.n_codebooks
        self.adapters = nn.ModuleList([self.decoder_film])
        
    def forward(
        self,
        audio_data: torch.Tensor,
        speaker_embedding: torch.Tensor = None
    ) -> tuple[torch.Tensor, list]:
        """
        Forward pass with hybrid conditioning.
        
        Args:
            audio_data: (B, 1, T) input audio
            speaker_embedding: (B, speaker_emb_dim) target speaker embedding
            
        Returns:
            audio_hat: (B, 1, T) reconstructed audio with target speaker
            codes: List of code tensors
        """
        if speaker_embedding is None:
            # No conditioning, use base model
            return self.base_model(audio_data)
        
        # Encode with speaker conditioning (encoder adapters shift codes)
        codes = self.encoder_adapter.encode(audio_data, speaker_embedding)
        
        # Decode with speaker conditioning (decoder FiLM guides decoding)
        audio_hat = self.decode_with_film(codes, speaker_embedding)
        
        # Trim to original length
        length = audio_data.shape[-1]
        audio_hat = audio_hat[..., :length]
        
        return audio_hat, codes
    
    def decode_with_film(
        self,
        codes: list,
        speaker_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode codes with FiLM conditioning.
        
        Args:
            codes: List of code tensors
            speaker_embedding: (B, speaker_emb_dim) speaker embedding
            
        Returns:
            (B, 1, T) decoded audio
        """
        snac_model = self.base_model.base_model
        
        # Reconstruct latent from codes
        z_q = snac_model.quantizer.from_codes(codes)  # (B, latent_dim, T)
        
        # Apply FiLM modulation to latent before decoder
        z_modulated = self.decoder_film(z_q, speaker_embedding)
        
        # Decode
        audio_hat = snac_model.decoder(z_modulated)
        
        return audio_hat
    
    def encode(
        self,
        audio_data: torch.Tensor,
        speaker_embedding: torch.Tensor = None
    ) -> list:
        """
        Encode audio with speaker conditioning.
        
        Args:
            audio_data: (B, 1, T) input audio
            speaker_embedding: (B, speaker_emb_dim) target speaker embedding
            
        Returns:
            List of code tensors
        """
        return self.encoder_adapter.encode(audio_data, speaker_embedding)
    
    def decode(
        self,
        codes: list,
        speaker_embedding: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Decode codes with optional speaker conditioning.
        
        Args:
            codes: List of code tensors
            speaker_embedding: (B, speaker_emb_dim) speaker embedding (optional)
            
        Returns:
            (B, 1, T) decoded audio
        """
        if speaker_embedding is None:
            # No conditioning, use base decoder
            snac_model = self.base_model.base_model
            z_q = snac_model.quantizer.from_codes(codes)
            audio_hat = snac_model.decoder(z_q)
            return audio_hat
        
        # Use FiLM-conditioned decoder
        return self.decode_with_film(codes, speaker_embedding)
    
    def extract_speaker_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from audio."""
        return self.base_model.extract_speaker_embedding(audio)
    
    def get_num_adapters(self) -> int:
        """Return total number of adapters (encoder + decoder)."""
        return self.encoder_adapter.get_num_adapters() + 1  # +1 for decoder FiLM
    
    def get_adapter_params(self) -> int:
        """Return total trainable parameters in all adapters."""
        encoder_params = self.encoder_adapter.get_adapter_params()
        decoder_params = sum(p.numel() for p in self.decoder_film.parameters())
        return encoder_params + decoder_params

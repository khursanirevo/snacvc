"""
Speaker encoder using ECAPA-TDNN from SpeechBrain.

Extracts fixed-dimensional speaker embeddings from audio waveforms.
These embeddings are used to condition the SNAC decoder for speaker manipulation.
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T


class SpeakerEncoder(nn.Module):
    """
    Speaker encoder using ECAPA-TDNN from SpeechBrain.

    ECAPA-TDNN is a state-of-the-art speaker verification model that extracts
    speaker embeddings from audio. We use it frozen (pretrained) to extract
    speaker characteristics for conditioning.

    Args:
        embedding_dim: Output embedding dimension (default: 512)
        freeze: If True, keep ECAPA-TDNN frozen (default: True)
        snac_sample_rate: SNAC's sample rate for resampling (default: 24000)

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
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.snac_sample_rate = snac_sample_rate

        # Import SpeechBrain here to avoid unnecessary dependency if not using speaker conditioning
        try:
            from speechbrain.inference.speaker import SpeakerRecognition

            # Load pretrained ECAPA-TDNN model
            # Note: Model is loaded on CPU, will be moved to GPU in first forward pass
            self.model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/speechbrain_ecapa",
            )

            # ECAPA-TDNN output dimension
            self.ecapa_dim = 192

        except ImportError:
            raise ImportError(
                "SpeechBrain is required for speaker encoding. "
                "Install with: pip install speechbrain>=0.5.16"
            )

        # Resampler: SNAC sample rate -> 16kHz (ECAPA-TDNN expects 16kHz)
        self.resampler = T.Resample(snac_sample_rate, 16000)

        # Projection layer to map from ECAPA dim to desired embedding dim
        self.proj = nn.Linear(self.ecapa_dim, embedding_dim)

        # Freeze ECAPA-TDNN (we don't want to finetune the speaker encoder)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            # Also freeze the embedding model specifically
            if hasattr(self.model, 'mods') and hasattr(self.model.mods, 'embedding_model'):
                self.model.mods.embedding_model.eval()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract L2-normalized speaker embeddings from audio.

        Args:
            audio: (B, 1, T) audio waveform at self.snac_sample_rate

        Returns:
            (B, embedding_dim) L2-normalized speaker embeddings
        """
        device = audio.device

        # Resample from SNAC sample rate to 16kHz for ECAPA-TDNN
        resampler = self.resampler.to(device)
        audio_16k = resampler(audio).to(device)

        # ECAPA-TDNN expects (B, T) format - remove channel dim
        # audio shape is (B, 1, T), we need (B, T)
        if audio_16k.dim() == 3:
            audio_16k = audio_16k.squeeze(1)  # (B, 1, T) -> (B, T)

        # Ensure audio is on the correct device
        audio_16k = audio_16k.to(device)

        # CRITICAL: Move ALL model components to device BEFORE forward pass
        # This includes mods, embedding_model, and all submodules
        self.model = self.model.to(device)
        if hasattr(self.model, 'mods'):
            self.model.mods = self.model.mods.to(device)
            if hasattr(self.model.mods, 'embedding_model'):
                self.model.mods.embedding_model = self.model.mods.embedding_model.to(device)

        # Extract embeddings using frozen ECAPA-TDNN
        with torch.no_grad():
            # Manually compute what encode_batch does, but ensure wav_lens is on correct device
            # wav_lens: relative lengths (1.0 = full length for all samples after padding)
            wav_lens = torch.ones(audio_16k.shape[0], device=device)

            # Get embeddings directly from embedding_model
            embeddings = self.model.mods.embedding_model(audio_16k, wav_lens=wav_lens)

        # Project to target embedding dimension
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

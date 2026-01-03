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

        # Extract embeddings using frozen ECAPA-TDNN
        with torch.no_grad():
            # SpeechBrain expects (B, T), remove channel dim and ensure on device
            audio_16k = audio_16k.squeeze(1).to(device)

            # Encode batch - returns (B, 1, 192)
            model_device = next(self.model.parameters()).device
            if model_device != device:
                self.model = self.model.to(device)
            embeddings = self.model.encode_batch(audio_16k)

            # Remove extra dimension if present
            if embeddings.dim() == 3:
                embeddings = embeddings.squeeze(1)  # (B, 192)

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

"""
Cross-Attention for Speaker Conditioning.

Replaces FiLM (Feature-wise Linear Modulation) with cross-attention for more
expressive speaker control. Cross-attention allows the model to selectively
use different parts of the speaker embedding for each feature dimension.
"""

import torch
import torch.nn as nn
import math
from einops import rearrange


class CrossAttentionSpeakerConditioning(nn.Module):
    """
    Cross-attention based speaker conditioning.

    Instead of simple affine transformation (FiLM), uses cross-attention:
        output = Attention(queries=features, keys=speaker_emb, values=speaker_emb)

    This allows the model to selectively attend to different speaker characteristics
    for each feature and time step, enabling more expressive speaker control.

    Args:
        num_features: Number of feature channels (C)
        speaker_emb_dim: Dimension of speaker embedding
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)

    Input:
        x: (B, C, T) input features
        speaker_emb: (B, speaker_emb_dim) speaker embedding

    Output:
        (B, C, T) conditioned features
    """

    def __init__(self, num_features: int, speaker_emb_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.speaker_emb_dim = speaker_emb_dim
        self.num_heads = num_heads
        self.head_dim = num_features // num_heads

        assert num_features % num_heads == 0, "num_features must be divisible by num_heads"

        # Project speaker embedding to keys and values
        self.key_proj = nn.Linear(speaker_emb_dim, num_features)
        self.value_proj = nn.Linear(speaker_emb_dim, num_features)

        # Layer norm for features (before projection to queries)
        self.norm = nn.LayerNorm(num_features)

        # Output projection
        self.out_proj = nn.Linear(num_features, num_features)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize for identity transformation at start
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention speaker conditioning.

        Args:
            x: (B, C, T) input features
            speaker_emb: (B, speaker_emb_dim) speaker embedding

        Returns:
            (B, C, T) conditioned features
        """
        B, C, T = x.shape

        # Normalize features
        x_norm = self.norm(x)  # (B, C, T)

        # Project speaker embedding to keys and values
        # (B, speaker_emb_dim) -> (B, C)
        K = self.key_proj(speaker_emb)  # (B, C)
        V = self.value_proj(speaker_emb)  # (B, C)

        # Reshape for multi-head attention
        # (B, C) -> (B, num_heads, head_dim)
        K = K.reshape(B, self.num_heads, self.head_dim)
        V = V.reshape(B, self.num_heads, self.head_dim)
        x_norm = x_norm.reshape(B, self.num_heads, self.head_dim, T)

        # Transpose for attention: (B, num_heads, T, head_dim)
        x_norm = x_norm.transpose(2, 3)  # (B, num_heads, head_dim, T)
        K = K.unsqueeze(-1)  # (B, num_heads, head_dim, 1)
        V = V.unsqueeze(-1)  # (B, num_heads, head_dim, 1)

        # Scaled dot-product attention
        # Q: (B, num_heads, head_dim, T)
        # K, V: (B, num_heads, head_dim, 1)
        scores = torch.matmul(x_norm.transpose(-2, -1), K) / math.sqrt(self.head_dim)
        # scores: (B, num_heads, T, 1)

        # Attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # (B, num_heads, T, 1) x (B, num_heads, head_dim, 1) -> (B, num_heads, T, head_dim)
        attended = torch.matmul(attn_weights, V.transpose(-2, -1))
        # attended: (B, num_heads, T, head_dim)

        # Reshape back: (B, num_heads, T, head_dim) -> (B, C, T)
        attended = attended.transpose(2, 3).reshape(B, C, T)

        # Residual connection (important for stability)
        attended = attended + x

        # Output projection
        out = self.out_proj(attended.transpose(1, 2)).transpose(1, 2)

        return out


class SpeakerConditioningLayer(nn.Module):
    """
    Wrapper that can use either FiLM or Cross-Attention.
    """

    def __init__(
        self,
        num_features: int,
        speaker_emb_dim: int,
        conditioning_type: str = 'film',  # 'film' or 'cross_attention'
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.conditioning_type = conditioning_type

        if conditioning_type == 'cross_attention':
            self.conditioning = CrossAttentionSpeakerConditioning(
                num_features=num_features,
                speaker_emb_dim=speaker_emb_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        else:  # 'film'
            from .film import FiLM
            self.conditioning = FiLM(
                num_features=num_features,
                cond_dim=speaker_emb_dim
            )

    def forward(self, x: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        return self.conditioning(x, speaker_emb)

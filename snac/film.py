"""
FiLM (Feature-wise Linear Modulation) layers for speaker conditioning.

FiLM applies: output = gamma * x + beta
where gamma and beta are generated from conditioning vectors (e.g., speaker embeddings).
"""

import torch
import torch.nn as nn


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation layer.

    Applies an affine transformation to features based on conditioning information:
        output = gamma * x + beta

    where gamma (scale) and beta (shift) are predicted from the conditioning vector.
    This allows the model to modulate feature representations based on speaker identity.

    Args:
        num_features: Number of feature channels (C)
        cond_dim: Dimension of conditioning vector

    Input:
        x: (B, C, T) input features
        cond: (B, cond_dim) conditioning vector (e.g., speaker embedding)

    Output:
        (B, C, T) modulated features
    """

    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.num_features = num_features
        self.cond_dim = cond_dim

        # Networks to predict scale (gamma) and shift (beta) from conditioning
        self.gamma_fc = nn.Linear(cond_dim, num_features)
        self.beta_fc = nn.Linear(cond_dim, num_features)

        # Initialize to identity transformation (gamma=1, beta=0)
        # This ensures the layer starts as a no-op and learns modulation gradually
        nn.init.zeros_(self.gamma_fc.weight)
        nn.init.zeros_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

        # Set initial gamma to 1 (identity scaling)
        self.gamma_fc.bias.data.fill_(1.0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation.

        Args:
            x: (B, C, T) input features
            cond: (B, cond_dim) conditioning vector

        Returns:
            (B, C, T) modulated features
        """
        # Predict gamma and beta from conditioning
        gamma = self.gamma_fc(cond)  # (B, C)
        beta = self.beta_fc(cond)    # (B, C)

        # Reshape for broadcasting over time dimension
        # (B, C) -> (B, C, 1)
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)

        # Apply affine transformation
        return gamma * x + beta

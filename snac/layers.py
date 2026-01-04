import math

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm

from .attention import LocalMHA
from .film import FiLM


class Encoder(nn.Module):
    def __init__(
        self,
        d_model=64,
        strides=[3, 3, 7, 7],
        depthwise=False,
        attn_window_size=32,
    ):
        super().__init__()
        layers = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            groups = d_model // 2 if depthwise else 1
            layers += [EncoderBlock(output_dim=d_model, stride=stride, groups=groups)]
        if attn_window_size is not None:
            layers += [LocalMHA(dim=d_model, window_size=attn_window_size)]
        groups = d_model if depthwise else 1
        layers += [
            WNConv1d(d_model, d_model, kernel_size=7, padding=3, groups=groups),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        noise=False,
        depthwise=False,
        attn_window_size=32,
        d_out=1,
    ):
        super().__init__()
        if depthwise:
            layers = [
                WNConv1d(input_channel, input_channel, kernel_size=7, padding=3, groups=input_channel),
                WNConv1d(input_channel, channels, kernel_size=1),
            ]
        else:
            layers = [WNConv1d(input_channel, channels, kernel_size=7, padding=3)]

        if attn_window_size is not None:
            layers += [LocalMHA(dim=channels, window_size=attn_window_size)]

        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            groups = output_dim if depthwise else 1
            layers.append(DecoderBlock(input_dim, output_dim, stride, noise, groups=groups))

        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


class ResidualUnit(nn.Module):
    def __init__(self, dim=16, dilation=1, kernel=7, groups=1):
        super().__init__()
        pad = ((kernel - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=kernel, dilation=dilation, padding=pad, groups=groups),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, output_dim=16, input_dim=None, stride=1, groups=1):
        super().__init__()
        input_dim = input_dim or output_dim // 2
        self.block = nn.Sequential(
            ResidualUnit(input_dim, dilation=1, groups=groups),
            ResidualUnit(input_dim, dilation=3, groups=groups),
            ResidualUnit(input_dim, dilation=9, groups=groups),
            Snake1d(input_dim),
            WNConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class NoiseBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = WNConv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, T = x.shape
        noise = torch.randn((B, 1, T), device=x.device, dtype=x.dtype)
        h = self.linear(x)
        n = noise * h
        x = x + n
        return x


class DecoderBlock(nn.Module):
    def __init__(self, input_dim=16, output_dim=8, stride=1, noise=False, groups=1):
        super().__init__()
        layers = [
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
        ]
        if noise:
            layers.append(NoiseBlock(output_dim))
        layers.extend(
            [
                ResidualUnit(output_dim, dilation=1, groups=groups),
                ResidualUnit(output_dim, dilation=3, groups=groups),
                ResidualUnit(output_dim, dilation=9, groups=groups),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualUnitWithFiLM(nn.Module):
    """
    ResidualUnit with FiLM conditioning applied before the block.

    This is identical to ResidualUnit but applies FiLM modulation based on
    a conditioning vector (e.g., speaker embedding) before processing.

    Args:
        dim: Number of channels
        dilation: Dilation factor for convolutions
        kernel: Kernel size
        groups: Number of groups for grouped convolutions
        cond_dim: Dimension of conditioning vector

    Input:
        x: (B, C, T) input features
        cond: (B, cond_dim) conditioning vector

    Output:
        (B, C, T) output features with residual connection
    """

    def __init__(self, dim=16, dilation=1, kernel=7, groups=1, cond_dim=512):
        super().__init__()
        self.film = FiLM(num_features=dim, cond_dim=cond_dim)

        pad = ((kernel - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=kernel, dilation=dilation, padding=pad, groups=groups),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x, cond):
        """
        Apply FiLM conditioning then residual block.

        Args:
            x: (B, C, T) input features
            cond: (B, cond_dim) conditioning vector

        Returns:
            (B, C, T) output features with residual connection
        """
        # Apply FiLM conditioning to input
        x_film = self.film(x, cond)

        # Apply residual block to conditioned features
        y = self.block(x_film)

        # Handle padding from strided convolutions
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]

        # Residual connection (use original x, not x_film)
        return x + y


class DecoderBlockWithFiLM(nn.Module):
    """
    DecoderBlock with FiLM conditioning before each ResidualUnit.

    This is identical to DecoderBlock but replaces ResidualUnits with
    ResidualUnitWithFiLM, allowing speaker conditioning at multiple scales.

    Args:
        input_dim: Input channel dimension
        output_dim: Output channel dimension
        stride: Upsampling stride for transposed convolution
        noise: Whether to add noise injection (for training)
        groups: Number of groups for grouped convolutions
        cond_dim: Dimension of conditioning vector (e.g., speaker embedding)

    Input:
        x: (B, input_dim, T) input features
        cond: (B, cond_dim) conditioning vector

    Output:
        (B, output_dim, T') upsampled and conditioned output features
    """

    def __init__(self, input_dim=16, output_dim=8, stride=1, noise=False, groups=1, cond_dim=512):
        super().__init__()

        # Upsampling layers
        layers = [
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
        ]

        # Optional noise injection for training stability
        if noise:
            layers.append(NoiseBlock(output_dim))

        self.upsample = nn.Sequential(*layers)

        # ResidualUnits with FiLM conditioning at different dilations
        # Each dilation provides a different receptive field
        self.res1 = ResidualUnitWithFiLM(output_dim, dilation=1, groups=groups, cond_dim=cond_dim)
        self.res2 = ResidualUnitWithFiLM(output_dim, dilation=3, groups=groups, cond_dim=cond_dim)
        self.res3 = ResidualUnitWithFiLM(output_dim, dilation=9, groups=groups, cond_dim=cond_dim)

    def forward(self, x, cond):
        """
        Upsample and apply conditioned residual units.

        Args:
            x: (B, input_dim, T) input features
            cond: (B, cond_dim) conditioning vector (speaker embedding)

        Returns:
            (B, output_dim, T') output features
        """
        # Upsample
        x = self.upsample(x)

        # Apply conditioned residual units
        x = self.res1(x, cond)
        x = self.res2(x, cond)
        x = self.res3(x, cond)

        return x


class ResidualUnitWithSpeakerConditioning(nn.Module):
    """
    ResidualUnit with flexible speaker conditioning (FiLM or Cross-Attention).

    This is a generalized version of ResidualUnitWithFiLM that supports
    multiple conditioning mechanisms through the SpeakerConditioningLayer wrapper.

    Args:
        dim: Number of channels
        dilation: Dilation factor for convolutions
        kernel: Kernel size
        groups: Number of groups for grouped convolutions
        cond_dim: Dimension of conditioning vector
        conditioning_type: Type of conditioning ('film' or 'cross_attention')
        num_heads: Number of attention heads (for cross-attention)

    Input:
        x: (B, C, T) input features
        cond: (B, cond_dim) conditioning vector

    Output:
        (B, C, T) output features with residual connection
    """

    def __init__(self, dim=16, dilation=1, kernel=7, groups=1, cond_dim=512,
                 conditioning_type='film', num_heads=8):
        super().__init__()
        from .cross_attention import SpeakerConditioningLayer

        self.conditioning = SpeakerConditioningLayer(
            num_features=dim,
            speaker_emb_dim=cond_dim,
            conditioning_type=conditioning_type,
            num_heads=num_heads
        )

        pad = ((kernel - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=kernel, dilation=dilation, padding=pad, groups=groups),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x, cond):
        """
        Apply speaker conditioning then residual block.

        Args:
            x: (B, C, T) input features
            cond: (B, cond_dim) conditioning vector

        Returns:
            (B, C, T) output features with residual connection
        """
        # Apply conditioning (FiLM or Cross-Attention)
        x_cond = self.conditioning(x, cond)

        # Apply residual block to conditioned features
        y = self.block(x_cond)

        # Handle padding from strided convolutions
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]

        # Residual connection (use original x, not x_cond)
        return x + y


class DecoderBlockWithSpeakerConditioning(nn.Module):
    """
    DecoderBlock with flexible speaker conditioning (FiLM or Cross-Attention).

    This is a generalized version of DecoderBlockWithFiLM that supports
    multiple conditioning mechanisms.

    Args:
        input_dim: Input channel dimension
        output_dim: Output channel dimension
        stride: Upsampling stride for transposed convolution
        noise: Whether to add noise injection (for training)
        groups: Number of groups for grouped convolutions
        cond_dim: Dimension of conditioning vector
        conditioning_type: Type of conditioning ('film' or 'cross_attention')
        num_heads: Number of attention heads (for cross-attention)

    Input:
        x: (B, input_dim, T) input features
        cond: (B, cond_dim) conditioning vector

    Output:
        (B, output_dim, T') upsampled and conditioned output features
    """

    def __init__(self, input_dim=16, output_dim=8, stride=1, noise=False, groups=1,
                 cond_dim=512, conditioning_type='film', num_heads=8):
        super().__init__()

        # Upsampling layers
        layers = [
            Snake1d(input_dim),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
        ]

        # Optional noise injection for training stability
        if noise:
            layers.append(NoiseBlock(output_dim))

        self.upsample = nn.Sequential(*layers)

        # ResidualUnits with speaker conditioning at different dilations
        self.res1 = ResidualUnitWithSpeakerConditioning(
            output_dim, dilation=1, groups=groups, cond_dim=cond_dim,
            conditioning_type=conditioning_type, num_heads=num_heads
        )
        self.res2 = ResidualUnitWithSpeakerConditioning(
            output_dim, dilation=3, groups=groups, cond_dim=cond_dim,
            conditioning_type=conditioning_type, num_heads=num_heads
        )
        self.res3 = ResidualUnitWithSpeakerConditioning(
            output_dim, dilation=9, groups=groups, cond_dim=cond_dim,
            conditioning_type=conditioning_type, num_heads=num_heads
        )

    def forward(self, x, cond):
        """
        Upsample and apply conditioned residual units.

        Args:
            x: (B, input_dim, T) input features
            cond: (B, cond_dim) conditioning vector (speaker embedding)

        Returns:
            (B, output_dim, T') output features
        """
        # Upsample
        x = self.upsample(x)

        # Apply conditioned residual units
        x = self.res1(x, cond)
        x = self.res2(x, cond)
        x = self.res3(x, cond)

        return x


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)

"""
Multi-Period Discriminator (MPD) and Multi-Resolution STFT Discriminator (MRD)
for GAN-based audio quality assessment.

Based on:
- MPD: HiFi-GAN, BigVGAN
- MRD: BigVGAN, UnivNet
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator (MPD) from HiFi-GAN/BigVGAN.

    Multiple sub-discriminators operating at different periods to capture
    periodic patterns in speech waveforms.

    Each sub-discriminator:
    1. Reshapes audio into 2D with period as second dimension
    2. Applies 2D convolutions to capture periodic patterns
    3. Outputs scalar real/fake prediction + feature maps
    """
    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        """
        Args:
            periods: List of periods for sub-discriminators (prime numbers work best)
        """
        super().__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(period) for period in periods
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple:
        """
        Args:
            y: Real audio (B, 1, T)
            y_hat: Generated audio (B, 1, T)

        Returns:
            y_d_rs: List of discriminator outputs for real audio
            y_d_gs: List of discriminator outputs for fake audio
            fmap_rs: List of feature maps from real audio
            fmap_gs: List of feature maps from fake audio
        """
        y_d_rs = []  # Real outputs
        y_d_gs = []  # Fake outputs
        fmap_rs = []  # Real feature maps
        fmap_gs = []  # Fake feature maps

        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(nn.Module):
    """Single-period discriminator sub-module.

    Operates on 2D representation of audio with shape (B, 1, T//period, period).
    This allows the discriminator to learn periodic patterns at the given period.
    """
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        """
        Args:
            period: Period for 2D reshaping
            kernel_size: Convolution kernel size (time dimension only)
            stride: Stride for downsampling (time dimension only)
        """
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(kernel_size//2, 0)),
            nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(kernel_size//2, 0)),
            nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(kernel_size//2, 0)),
            nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(kernel_size//2, 0)),
            nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(kernel_size//2, 0)),
        ])
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Audio waveform (B, 1, T)

        Returns:
            feat: Flattened discriminator output (B, -1)
            fmap: List of feature maps from each layer
        """
        fmap = []

        # Convert to 2D: (B, 1, T) -> (B, 1, T//period, period)
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = nn.functional.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = self.leaky_relu(x)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)  # (B, -1)

        return x, fmap


class MultiResolutionSTFTDiscriminator(nn.Module):
    """Multi-Resolution STFT Discriminator (MRD) from BigVGAN/UnivNet.

    Multiple discriminators operating on STFT spectrograms at different resolutions.
    Each discriminator processes both real and imaginary parts of the complex spectrogram.
    """
    def __init__(self, fft_sizes: List[int] = [1024, 2048, 4096]):
        """
        Args:
            fft_sizes: List of FFT sizes for different resolutions
        """
        super().__init__()
        self.discriminators = nn.ModuleList([
            STFTDiscriminator(fft_size) for fft_size in fft_sizes
        ])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple:
        """
        Args:
            y: Real audio (B, 1, T)
            y_hat: Generated audio (B, 1, T)

        Returns:
            y_d_rs: List of discriminator outputs for real audio
            y_d_gs: List of discriminator outputs for fake audio
            fmap_rs: List of feature maps from real audio
            fmap_gs: List of feature maps from fake audio
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class STFTDiscriminator(nn.Module):
    """Single-resolution STFT discriminator.

    Processes complex STFT spectrogram by splitting into real and imaginary parts.
    """
    def __init__(self, fft_size: int = 1024, hop_length: int = 256, win_length: int = 1024):
        """
        Args:
            fft_size: FFT size
            hop_length: Hop length for STFT
            win_length: Window length for STFT
        """
        super().__init__()
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length

        # Spectral convolution
        # Operates on 2 channels: real and imaginary parts
        self.convs = nn.ModuleList([
            nn.Conv2d(2, 32, kernel_size=(3, 9), padding=(1, 4)),
            nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(32, 32, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
        ])
        self.conv_post = nn.Conv2d(32, 1, (3, 3), padding=(1, 1))
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: Audio waveform (B, 1, T)

        Returns:
            feat: Flattened discriminator output (B, -1)
            fmap: List of feature maps from each layer
        """
        fmap = []
        x = x.squeeze(1)  # (B, T)

        # Compute STFT
        spec = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True
        )  # (B, freq, time, complex)

        # Split real and imaginary parts
        spec_real = spec.real
        spec_imag = spec.imag
        spec = torch.stack([spec_real, spec_imag], dim=1)  # (B, 2, freq, time)

        for conv in self.convs:
            spec = conv(spec)
            spec = self.leaky_relu(spec)
            fmap.append(spec)

        spec = self.conv_post(spec)
        fmap.append(spec)
        spec = torch.flatten(spec, 1, -1)

        return spec, fmap

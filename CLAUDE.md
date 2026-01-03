# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Management

**IMPORTANT: Always use `uv` for package management and running Python commands.**

```bash
# Install dependencies
uv pip install -r requirements.txt

# Run Python scripts
uv run python script.py

# Run specific module
uv run python -m module_name

# Install in development mode
uv pip install -e .
```

Never use `pip`, `python`, or `python3` directly - always prefix with `uv run` or use `uv pip`.

## Project Overview

SNAC (Multi-Scale Neural Audio Codec) is a neural audio codec that compresses audio into discrete codes at low bitrates. The key innovation is hierarchical tokenization where coarse tokens are sampled less frequently, covering broader time spans. This enables efficient audio compression and is particularly useful for language modeling approaches to audio generation.

## Installation and Development

### Installation
```bash
# Install in development mode
uv pip install -e .

# or from PyPI
uv pip install snac
```

### Dependencies
Core dependencies are managed via `requirements.txt`:
```bash
uv pip install -r requirements.txt
```
- `torch` - PyTorch framework
- `numpy` - Numerical operations
- `einops` - Tensor manipulation
- `huggingface_hub` - Loading pretrained models

### Version Management
The package version is defined in `snac/__init__.py` (currently `__version__ = "1.2.1"`). The `setup.py` automatically extracts this version.

## Architecture

### Core Components

The codebase consists of four main modules in the `snac/` directory:

1. **`snac.py`** - Main `SNAC` class that provides the primary API
2. **`layers.py`** - Encoder and Decoder neural network architectures
3. **`vq.py`** - Vector quantization modules (VectorQuantize, ResidualVectorQuantize)
4. **`attention.py`** - Local multi-head attention with rotary embeddings

### Key Architectural Concepts

**Hierarchical Multi-Scale Processing**: The encoder uses downsampling strides `[3, 3, 7, 7]` to create features at multiple time scales, while the decoder uses the reverse `[7, 7, 3, 3]`. This allows the model to capture both local details and global structure.

**Multi-Codebook Quantization**: The `ResidualVectorQuantize` uses multiple codebooks with different strides `[8, 4, 2, 1]`, each operating on residual information. This enables hierarchical representation at different temporal resolutions.

**Local Attention**: Local multi-head attention (`LocalMHA`) processes audio in windows to handle long sequences without quadratic memory complexity.

### Data Flow

```
Audio Input (B, 1, T)
    ↓
Encoder (multi-scale downsampling)
    ↓
Latent Representation (B, latent_dim, T')
    ↓
ResidualVectorQuantize (hierarchical quantization)
    ↓
Codes (list of token sequences at different time scales)
    ↓
Decoder (multi-scale upsampling)
    ↓
Audio Output (B, 1, T)
```

### The SNAC Class API

The main `SNAC` class in `snac/snac.py` provides three key methods:

- **`forward(audio_data)`**: Full encode-decode pass. Returns reconstructed audio and codes. Handles automatic padding via `preprocess()`.

- **`encode(audio_data)`**: Encode only. Returns a list of token sequences, each at a different temporal resolution. The list length equals `n_codebooks` (4 by default), with shapes `[B, N_codebook, T / stride]`.

- **`decode(codes)`**: Decode from hierarchical codes. Accepts the list output from `encode()` and reconstructs audio.

- **`from_pretrained(path)`**: Class method to load pretrained models from HuggingFace Hub or local paths.

### Audio Preprocessing

The `preprocess()` method pads audio to ensure proper alignment:
- Computes LCM of `vq_strides[0]` and `attn_window_size`
- Pads to multiples of `hop_length * lcm`
- This ensures all quantization and attention operations work correctly

### Model Configuration

Models are configured via constructor arguments (no config files at runtime):
- `sampling_rate`: Audio sample rate (24000, 32000, or 44100 Hz)
- `encoder_dim` / `decoder_dim`: Network dimensions
- `encoder_rates` / `decoder_rates`: Downsampling/upsampling strides
- `vq_strides`: Temporal strides for each codebook `[8, 4, 2, 1]`
- `codebook_size` / `codebook_dim`: VQ codebook parameters
- `attn_window_size`: Local attention window size (default: 32)
- `noise`: Whether to add noise during decoder training (default: True)

### Pretrained Models

Available models on HuggingFace:
- `hubertsiuzdak/snac_24khz` - 0.98 kbps, 19.8M params, optimized for speech
- `hubertsiuzdak/snac_32khz` - 1.9 kbps, 54.5M params, for music/sound effects
- `hubertsiuzdak/snac_44khz` - 2.6 kbps, 54.5M params, for music/sound effects

All models support single-channel (mono) audio only.

### Code Structure

The codebase is adapted from the Descript Audio Codec (DAC) with the key modification of hierarchical tokenization with different sampling rates per codebook.

**Straight-Through Estimators**: During training, quantization uses straight-through gradients (detach + pass-through) while maintaining discrete codes during inference.

### Usage Patterns

```python
import torch
from snac import SNAC

# Load model
model = SNAC.from_pretrained("hubertsiuzdak/snac_32khz").eval().cuda()

# Audio shape: (batch, channels, samples)
audio = torch.randn(1, 1, 32000).cuda()

# Encode only
codes = model.encode(audio)  # Returns list of 4 tensors with different temporal resolutions

# Decode only
audio_hat = model.decode(codes)

# Combined (autoencoder style)
audio_hat, codes = model(audio)
```

### Important Notes

- All models currently support **mono audio only** (shape: `(B, 1, T)`)
- The `codes` returned by `encode()` is a **list** of tensors, not a single tensor
- Each code in the list has a different temporal resolution (different time dimension)
- Audio is automatically padded to ensure proper alignment before encoding
- For language modeling applications, the hierarchical codes allow efficient modeling of long-form audio (e.g., ~3 minutes with 2048 context window at ~10 Hz for coarsest tokens)

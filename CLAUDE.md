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

**Current Focus**: Voice conversion through speaker-conditioned training, with multiple phases exploring different conditioning strategies.

## Training Phases

The project has evolved through multiple training phases, each exploring different speaker conditioning approaches:

- **Phase 3**: Contrastive learning with speaker embeddings
- **Phase 4**: FiLM conditioning at DECODER (after VQ)
- **Phase 5**: [Not currently used]
- **Phase 6**: Adapter conditioning at ENCODER (BEFORE VQ) ← **Current phase**

### Phase 6 Architecture (Recommended)

The key insight is that to get speaker-conditioned codes, the conditioning must happen **before quantization**:

```
Audio → Encoder → Latent → Adapter(speaker_emb) → Modulated Latent → VQ → Speaker-Conditioned Codes → Decoder
```

This way:
- Encoding `audio_A` with `emb_B` produces codes that decode to speaker B
- The adapter learns to modulate the encoder latent based on target speaker
- Only ~1M trainable parameters (adapter), base model frozen

### Why Phase 6 over Phase 4?

**Phase 4 (FiLM at Decoder)**:
```
Audio → Encoder → Latent → VQ → Codes (speaker-agnostic)
                                       ↓
                                 Decoder + FiLM(speaker_emb)
```
Problem: Codes already selected, FiLM tries to override speaker info too late

**Phase 6 (Adapter BEFORE VQ)**:
```
Audio → Encoder → Latent → Adapter(emb) → Modulated Latent → VQ → Conditioned Codes
```
Solution: Adapter shifts latent BEFORE code selection, codes already contain speaker info

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
- `faiss-cpu` - Fast similarity search for hard negative mining

### Version Management
The package version is defined in `snac/__init__.py` (currently `__version__ = "1.2.1"`). The `setup.py` automatically extracts this version.

## Architecture

### Core Components

The codebase consists of these main modules in the `snac/` directory:

**Base SNAC Model:**
1. **`snac.py`** - Main `SNAC` class for audio codec
2. **`layers.py`** - Encoder and Decoder architectures
3. **`vq.py`** - Vector quantization (VectorQuantize, ResidualVectorQuantize)
4. **`attention.py`** - Local multi-head attention with rotary embeddings

**Speaker Conditioning (Phase 4):**
5. **`speaker_encoder_factory.py`** - Factory for speaker encoders (ERes2NetV2, ECAPA, Simple)
6. **`discriminators.py`** - Multi-Period Discriminator (MPD) and Multi-Resolution STFT Discriminator (MRD)

**Speaker Conditioning (Phase 6):**
7. **`adapters.py`** - AdapterWrapper for conditioning BEFORE VQ
   - `FiLMAdapter`: Feature-wise Linear Modulation (gamma * latent + beta)
   - `AdapterWrapper`: Wraps SNAC, applies adapter to encoder latent before quantization

**Training Support:**
8. **`voice_conversion_loss.py`** - Voice conversion loss with separate identity/VC speaker losses
9. **`contrastive_loss.py`** - Contrastive learning loss
10. **`audio_augmentation.py`** - Pitch shifting and formant shifting for synthetic VC
11. **`faiss_speaker_index.py`** - FAISS-based hard negative mining
12. **`embedding_cache.py`** - Memory-mapped embedding cache for fast loading
13. **`stratified_hard_negatives.py`** - Stratified sampling (easy/medium/hard negatives)
14. **`codebook_adversarial_loss.py`** - Optional codebook purification (not recommended)

### Key Architectural Concepts

**Hierarchical Multi-Scale Processing**: Encoder downsampling `[3, 3, 7, 7]`, decoder upsampling `[7, 7, 3, 3]`.

**Multi-Codebook Quantization**: Multiple codebooks with strides `[8, 4, 2, 1]`, each operating on residual information.

**FiLM (Feature-wise Linear Modulation)**:
```python
# Modulates features: gamma(speaker_emb) * features + beta(speaker_emb)
# In Phase 6, applied BEFORE VQ to shift encoder latent
z_modulated = gamma * z + beta
```

**AdapterWrapper (Phase 6)**:
```python
# Wraps SNAC model
# Forward: audio → preprocess → encode → adapter(emb) → VQ → decode
class AdapterWrapper:
    def forward(self, audio, speaker_embedding=None):
        z = snac_model.encoder(audio)
        if speaker_embedding is not None:
            z = adapter(z, speaker_embedding)  # Modulate BEFORE VQ
        z_q, codes = snac_model.quantizer(z)
        audio_hat = snac_model.decoder(z_q)
        return audio_hat, codes
```

### Speaker Encoders

**ERes2NetV2** (recommended, from GPT-SoVITS ProPlus):
- 512-dimensional embeddings
- Pretrained on speaker verification
- Frozen during training

Factory pattern in `speaker_encoder_factory.py` supports multiple encoder types.

### Data Flow (Phase 6)

```
Audio Input (B, 1, T)
    ↓
Preprocess (padding)
    ↓
Encoder (multi-scale downsampling)
    ↓
Latent (B, latent_dim, T')
    ↓
Adapter(speaker_emb) → gamma * latent + beta  ← SPEAKER CONDITIONING HERE
    ↓
Modulated Latent (B, latent_dim, T')
    ↓
ResidualVectorQuantize (hierarchical quantization)
    ↓
Speaker-Conditioned Codes (list at different time scales)
    ↓
Decoder (multi-scale upsampling)
    ↓
Audio Output (B, 1, T)
```

### Training Scripts

**Phase 4** (FiLM at decoder):
- `train_phase4_gan.py` - Main training script
- `train_phase4_gan_ddp.py` - Multi-GPU DDP training
- `inference_phase4.py` - Test voice conversion quality

**Phase 6** (Adapter BEFORE VQ):
- `train_phase6_adapters.py` - Main training script (uses `AdapterWrapper`)
- Config: `configs/phase6_adapters.json`

### Configuration Files

Training configs use JSON format:
- `configs/phase4_gan_semantic_negatives.json` - Phase 4 configuration
- `configs/phase6_adapters.json` - Phase 6 configuration

Key config parameters:
```json
{
  "adapter_type": "film",
  "adapter_hidden_dim": 512,
  "adapter_num_layers": 2,

  "lambda_speaker_identity": 0.25,  // Lower weight for identity preservation
  "lambda_speaker_vc": 2.0,          // Higher weight for VC learning

  "use_faiss_hard_negatives": true,
  "use_synthetic_vc": true,
  "gan_weight": 1.0
}
```

## Loss Functions

### Voice Conversion Loss (Phase 6)

```
vc = lambda_recon * recon + lambda_speaker_identity * spk_id + lambda_speaker_vc * spk_vc
```

Where:
- `recon`: Reconstruction quality (L1 + multi-scale STFT)
- `spk_id`: Identity speaker loss (encode with own emb, should match)
- `spk_vc`: Voice conversion speaker loss (encode with target emb, should match target)

At initialization (adapter ≈ identity):
- `spk_id` ≈ 0.3-0.4 (base model preserves ~60% speaker)
- `spk_vc` ≈ 0.98 (adapter does nothing, wrong speaker)

Training goal: `spk_vc` decreases to approach `spk_id`.

### Other Losses

- **Adversarial loss** (`lambda_adv = 1.0`): GAN generator loss
- **Feature matching** (`lambda_fm = 2.0`): Matches intermediate features (highest weight!)
- **Synthetic VC** (`lambda_synthetic = 0.3`): Pitch-shifted audio with own embedding

## Important Training Notes

### Phase 6 Encoding

For Phase 6, ALWAYS encode WITH speaker embedding:
```python
# Correct:
codes = model.encode(audio, speaker_embedding=target_emb)
audio_out = model.decode(codes)

# Wrong (Phase 4 style):
codes = model.encode(audio)
audio_out = model.decode(codes, speaker_embedding=target_emb)
```

### Speaker Loss Monitoring

Watch these metrics during training:
- `spk_id`: Should stay ~0.3-0.4 (baseline, frozen model limitation)
- `spk_vc`: Should decrease from 0.98 → 0.4 (adapter learns)
- `spk_synth`: Pitch shift robustness (should be similar to spk_id)
- Gap `spk_vc - spk_id`: Should shrink from ~0.6 → <0.1

### Loss Weights Impact

Higher `lambda_speaker_vc` (currently 2.0):
- Prioritizes learning voice conversion over identity
- Accelerates `spk_vc` decrease
- May slightly increase `spk_id` (acceptable trade-off)

## Usage Patterns

### Basic SNAC Usage
```python
import torch
from snac import SNAC

# Load base model
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()

# Audio shape: (batch, channels, samples)
audio = torch.randn(1, 1, 24000).cuda()

# Encode
codes = model.encode(audio)  # List of 4 tensors

# Decode
audio_hat = model.decode(codes)
```

### Phase 6: Speaker-Conditioned Encoding
```python
from snac import SNACWithSpeakerConditioning
from snac.adapters import AdapterWrapper

# Load base model
base_model = SNACWithSpeakerConditioning.from_pretrained_base(
    "hubertsiuzdak/snac_24khz",
    speaker_encoder_type="eres2net",
    freeze_base=True
).cuda()

# Wrap with adapter
model = AdapterWrapper(
    base_model=base_model,
    adapter_type="film",
    adapter_hidden_dim=512,
    adapter_num_layers=2
).cuda()

# Encode with speaker conditioning
codes = model.encode(audio, speaker_embedding=target_speaker_emb)
audio_hat = model.decode(codes)

# Or full forward
audio_hat, codes = model(audio, speaker_embedding=target_speaker_emb)
```

### Training Phase 6
```bash
# Single GPU
uv run python train_phase6_adapters.py \
    --config configs/phase6_adapters.json \
    --device 3

# Multi-GPU (DDP)
uv run python train_phase6_adapters.py \
    --config configs/phase6_adapters.json \
    --ddp
```

### Monitoring Training
```bash
# Real-time logs
tail -f logs/phase6/training.log

# Check for speaker loss convergence
grep "Batch" logs/phase6/training.log | tail -20
```

## Important Notes

- All models support **mono audio only** (shape: `(B, 1, T)`)
- `codes` from `encode()` is a **list** of tensors with different temporal resolutions
- Audio is automatically padded to ensure proper alignment
- Phase 6: Only adapter parameters are trainable (~1M), base SNAC frozen (~89M)
- Speaker encoders are always frozen
- FAISS index and embedding cache should be pre-built for training (see `scripts/`)

## Pretrained Models

Available on HuggingFace:
- `hubertsiuzdak/snac_24khz` - 0.98 kbps, 19.8M params, for speech

Speaker encoder (manual download):
- ERes2NetV2: Place in `pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt`

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Current Session

**Date**: 2025-01-13

**No active training jobs** (check with `ps aux | grep -E "(train|finetune)" | grep -v grep`)

---

## Quick Start Guide

### Training (Phase 10: Decoder-Only Fine-tuning)

**REPRODUCIBLE TRAINING SCRIPTS - USE THESE:**

```bash
# Start training (reproducible, background mode)
./train_decoder_only.sh 0    # Run on GPU 0
./train_decoder_only.sh 3    # Run on GPU 3

# Check training status
./train_status.sh

# Monitor logs
tail -f logs/phase10_decoder_only/training.log
```

**Manual training (if scripts don't work):**

```bash
# Single GPU
uv run python finetune.py --config configs/phase10_revolab_all.json --device 0
```

**Full documentation:** `TRAINING_README.md`

### Inference

```bash
# Encode/decode audio
uv run python inference.py \
    --checkpoint checkpoints/phase10_revolab_all/best_model.pt \
    --input audio.wav \
    --output reconstructed.wav

# Generate from codes
uv run python generate.py \
    --checkpoint checkpoints/phase10_revolab_all/best_model.pt \
    --output generated.wav
```

### Tokenization (for BPE analysis)

```bash
# Convert audio to SNAC tokens
uv run python research/tokenization/audio_to_tokens.py \
    --input_dir /mnt/data/combine/train \
    --output_dir /mnt/data/tokens/train \
    --checkpoint checkpoints/phase10_revolab_all/best_model.pt \
    --segment_length 4.0 \
    --batch_size 200
```

---

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

## Long-Running Tasks (Training)

**IMPORTANT: Use the reproducible training scripts for Phase 10:**

```bash
./train_decoder_only.sh 0
```

This script already handles:
- Background execution with nohup
- PID file management
- Log file setup
- Pre-flight checks
- Monitoring commands output

**Manual training (only if scripts don't work):**

```bash
# ALWAYS run in background with nohup
nohup uv run python finetune.py --config config.json --device 0 > /tmp/training.log 2>&1 &
TRAIN_PID=$!

# Verify it started
sleep 3
ps -p $TRAIN_PID && echo "✓ Running" || echo "✗ Failed"

# Monitor logs
tail -f /tmp/training.log
```

## Project Overview

SNAC (Multi-Scale Neural Audio Codec) is a neural audio codec that compresses audio into discrete codes at low bitrates. The key innovation is hierarchical tokenization where coarse tokens are sampled less frequently, covering broader time spans.

**Current Phase**: Phase 10 - Simple fine-tuning on Levantine Arabic speech datasets (2.8M utterances). No voice conversion or adapters - just reconstruction quality improvement.

**Previous Phases** (archived):
- Phase 3-6: Voice conversion experiments with speaker conditioning (contrastive learning, FiLM, adapters)
- Phase 7-9: Various training strategies (GAN losses, curriculum learning, multi-stage)

**Tokenization Research**: Analyzing SNAC token patterns for BPE compression. See `research/tokenization/TOKENIZER_README.md` for details.

## Training Phases

**Phase 10** (current): Simple fine-tuning on Levantine Arabic
- Full model training (encoder + decoder)
- Curriculum learning: 1s → 2s → 3s → 4s segments
- Loss: L1 + multi-scale STFT
- Checkpoint: `checkpoints/phase10_revolab_all/best_model.pt`
- Results: 29% validation loss improvement

**Archived Phases** (not recommended for new work):
- Phase 3-6: Voice conversion experiments (speaker conditioning)
- Phase 4: FiLM conditioning at decoder (after VQ)
- Phase 6: Adapter conditioning at encoder (before VQ)

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

### Core SNAC Model (`snac/` directory)

**Main Components:**
- `snac.py` - Main `SNAC` class for audio codec
- `layers.py` - Encoder and Decoder architectures (SEANet blocks)
- `vq.py` - Vector quantization (ResidualVectorQuantize)
- `attention.py` - Local multi-head attention with rotary embeddings
- `dataset.py` - OptimizedAudioDataset for efficient training

**Hierarchical Multi-Scale Processing:**
- Encoder downsampling strides: `[3, 3, 7, 7]`
- Decoder upsampling strides: `[7, 7, 3, 3]`
- 4 codebook levels with temporal strides: `[8, 4, 2, 1]`
- Residual quantization (each codebook operates on previous residual)

**Data Flow (Phase 10 simple fine-tuning):**
```
Audio Input (B, 1, T) → Encoder → Latent (B, 512, T') → VQ → Codes (4 scales) → Decoder → Audio Output
```

**Voice Conditioning Components** (archived, not used in Phase 10):
- `adapters.py` - FiLM adapters for speaker conditioning
- `speaker_encoder_factory.py` - Speaker encoders (ERes2NetV2, ECAPA)
- `discriminators.py` - GAN discriminators (MPD, MRD)
- `voice_conversion_loss.py` - Speaker identity + VC losses
- `contrastive_loss.py` - Contrastive learning losses
- `audio_augmentation.py` - Pitch/formant shifting for synthetic VC
- `faiss_speaker_index.py` - FAISS-based hard negative mining
- `embedding_cache.py` - Memory-mapped embedding cache

### Configuration Files

**Phase 10** (current):
- `configs/phase10_revolab_all.json` - Full dataset fine-tuning with curriculum
- `configs/phase10_curriculum.json` - Curriculum learning configuration
- `configs/phase10_combined.json` - Combined Levantine datasets

**Key config parameters (Phase 10):**
```json
{
  "train_data": "/mnt/data/combine/train/audio",
  "val_data": "/mnt/data/combine/valid/audio",
  "batch_size": 48,
  "learning_rate": 15e-6,
  "l1_weight": 1.0,
  "stft_weight": 1.0,
  "n_ffts": [1024, 2048, 4096],
  "freeze_encoder": true,
  "freeze_vq": true,
  "curriculum": [
    {"epochs": [1, 2], "length": 1.0, "batch_multiplier": 2.0},
    {"epochs": [3, 4], "length": 2.0, "batch_multiplier": 1.0},
    {"epochs": [5, 6], "length": 3.0, "batch_multiplier": 0.6},
    {"epochs": [7, 10], "length": 4.0, "batch_multiplier": 0.45}
  ]
}
```

## Loss Functions

### Phase 10: Reconstruction Loss

```
loss = l1_weight * L1_loss + stft_weight * multi_scale_STFT_loss
```

- **L1 loss**: Time-domain reconstruction error (MAE)
- **Multi-scale STFT loss**: Frequency-domain reconstruction at FFT sizes `[1024, 2048, 4096]`

This is a simple reconstruction loss - no adversarial, speaker, or feature matching losses.

### Voice Conversion Loss (archived Phase 6)

```
vc = lambda_recon * recon + lambda_speaker_identity * spk_id + lambda_speaker_vc * spk_vc
```

Not used in Phase 10. See archived scripts if needed.

## Important Training Notes

### Curriculum Learning (Phase 10)

Training uses curriculum learning with progressively longer segments:
| Epochs | Segment | Batch Size | Purpose |
|--------|---------|------------|---------|
| 1-2 | 1.0s | 96 | Fast iterations, foundation |
| 3-4 | 2.0s | 48 | Medium context |
| 5-6 | 3.0s | 29 | Longer context |
| 7-10 | 4.0s | 22 | Full context, refinement |

This stabilizes training and improves final quality.

### Freezing Strategy

In Phase 10 config: `freeze_encoder: true, freeze_vq: true`
- Only decoder is trained (lightweight, ~15M params)
- Encoder and VQ remain at pretrained quality
- Faster training, less overfitting

To train full model, set both to `false`.

## Important Notes

- All models support **mono audio only** (shape: `(B, 1, T)`)
- `codes` from `encode()` is a **list** of 4 tensors with different temporal resolutions
- Audio is automatically padded to ensure proper alignment
- Package version in `snac/__init__.py`: currently `1.3.0`

## Pretrained Models

**HuggingFace:**
- `hubertsiuzdak/snac_24khz` - 0.98 kbps, 19.8M params, speech (24 kHz)
- `hubertsiuzdak/snac_32khz` - 1.9 kbps, 54.5M params, music (32 kHz)
- `hubertsiuzdak/snac_44khz` - 2.6 kbps, 54.5M params, music (44 kHz)

**Fine-tuned (this repo):**
- `checkpoints/phase10_revolab_all/best_model.pt` - Fine-tuned on Levantine Arabic (29% improvement)

## Dataset Preparation

Organize audio files as:
```
dataset/
├── train/
│   ├── file1.wav
│   └── ...
└── val/
    ├── file1.wav
    └── ...
```

Or use the script:
```bash
uv run python prepare_dataset_folder.py \
    --input_dir /path/to/audio \
    --train_ratio 0.95 \
    --output_dir /path/to/dataset
```

## Repository Structure

```
/mnt/data/work/snac/
├── finetune.py                  # Core training script (Phase 10)
├── inference.py                 # Inference script
├── generate.py                  # Generation script
├── prepare_dataset_folder.py    # Dataset preparation
├── train_decoder_only.sh        # Training launcher (reproducible)
├── train_status.sh              # Training status checker
├── TRAINING_README.md           # Training documentation
├── CLAUDE.md                    # This file
├── README.md                    # Project README
├── setup.py                     # Package setup
├── requirements.txt             # Dependencies
│
├── configs/                     # Training configurations
│   ├── phase10_revolab_all.json  # Main Phase 10 config (decoder-only)
│   ├── phase10_curriculum.json
│   └── phase10_combined.json
│
├── snac/                        # Main package
│   ├── snac.py                 # SNAC model
│   ├── layers.py               # Encoder/Decoder
│   ├── vq.py                   # Vector quantization
│   ├── dataset.py              # Dataset classes
│   └── ... (other modules)
│
├── checkpoints/                 # Trained models
│   └── phase10_revolab_all/
│       ├── best_model.pt
│       └── checkpoint_epoch*.pt
│
├── logs/                        # Training logs
│   └── phase10_decoder_only/
│       └── training.log
│
└── research/                    # Research experiments
    └── tokenization/           # BPE/tokenization research
        ├── audio_to_tokens.py
        ├── TOKENIZER_README.md
        └── BPE.md
```

## First Thing To Check In Any New Session

```bash
# Check if any training is still running
ps aux | grep -E "(finetune|train)" | grep -v grep

# Or use the status script
./train_status.sh

# Check GPU status
nvidia-smi

# Check recent logs (if training exists)
ls -lt logs/*/training.log 2>/dev/null | head -1
```

## Training Scripts Structure

**Root level scripts (for Phase 10 decoder-only training):**
- `train_decoder_only.sh` - Main training script (reproducible, background mode)
- `train_status.sh` - Check running training status
- `TRAINING_README.md` - Full training documentation
- `finetune.py` - Core training script (used by train_decoder_only.sh)
- `inference.py` - Inference script
- `generate.py` - Generation script
- `prepare_dataset_folder.py` - Dataset preparation

**To start training:**
```bash
./train_decoder_only.sh 0  # GPU 0
```

**To check status:**
```bash
./train_status.sh
```

**Training outputs:**
- Checkpoints: `checkpoints/phase10_revolab_all/`
- Logs: `logs/phase10_decoder_only/training.log`
- Background logs: `/tmp/phase10_decoder_only_gpu<N>.log`
- PID files: `/tmp/phase10_decoder_only_gpu<N>.pid`

## Tokenization / BPE Analysis

For analyzing SNAC token patterns and BPE compression potential, see `research/tokenization/TOKENIZER_README.md`.

Quick start:
```bash
# Convert audio to tokens
uv run python research/tokenization/audio_to_tokens.py \
    --input_dir /mnt/data/combine/train \
    --output_dir /mnt/data/tokens/train \
    --checkpoint checkpoints/phase10_revolab_all/best_model.pt \
    --segment_length 4.0

# Analyze existing tokens
uv run python research/tokenization/audio_to_tokens.py --analyze_only --output_dir /mnt/data/tokens/train
```

The script provides:
- Frequency distribution (Gini coefficient)
- Bigram patterns
- Transition sparsity
- Repetition analysis
- BPE recommendations

## GPU Access Control Monitor

**Location**: `/mnt/data/work/gpu_access_control/`

Protects GPU 0 by automatically killing processes from users other than `sani`.

**Quick Start**:
```bash
cd /mnt/data/work/gpu_access_control

# Test run
sudo ./manage.sh run

# Install as service (24/7 monitoring)
sudo ./manage.sh install && sudo ./manage.sh start

# Check status
sudo ./manage.sh status

# View logs
sudo ./manage.sh logs
```

**Configuration**: Edit `monitor.py` to change `ALLOWED_USER` and `GPU_ID`.

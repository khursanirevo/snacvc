# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Current Session

**Date**: 2026-01-16

**Phase 12 active**: Dual-head decoder (24kHz + 48kHz outputs)

**Status**: 🚀 **Training running** on GPU 3

**Training details**:
- **Location**: `checkpoints/phase12_dual_head/`
- **Logs**: `/tmp/phase12_dual_head_gpu3.log`
- **Dataset**: 3,447,109 training + 55,419 validation samples
- **Architecture**: Shared decoder → 24kHz head + 48kHz head
- **Strategy**: Warmup (epoch 1, 48kHz only) → Main (epochs 2-15, both heads)
- **VRAM**: 112GB / 144GB (GPU 3) - higher in main phase
- **Current Epoch**: 4/15 (resumed from checkpoint)

**Key Phase 12 features**:
- ✅ Dual-head decoder: 24kHz (pretrained SNAC) + 48kHz (new)
- ✅ Fair comparison with Phase 10 (same 24kHz metrics)
- ✅ Using pretrained SNAC decoder (matches pretrained SNAC encoder)
- ✅ Smart weight initialization from pretrained SNAC block 5
- ✅ Higher LRs (2x faster): warmup 2e-4, main 5e-5, 24kHz 1e-5
- ✅ Phase 10 style epoch ranges: 1-2, 3-4, 5-6, 7-15
- ✅ Weighted loss: 0.3 × 24kHz + 1.0 × 48kHz
- ✅ Checkpoint resumption support

**Recent fixes**:
- ✅ Fixed checkpoint saving bug (None optimizer handling)
- ✅ Added epoch_ranges parameter for curriculum learning
- ✅ Increased learning rates for faster training

**Phase 11 completed**:
- ✅ 48kHz decoder training with random slicing
- ✅ Cached codes: 3.4M training + 55k validation samples
- ✅ Checkpoint resumption, validation, time-based checkpointing

---

## CRITICAL BEHAVIOR RULES

**READ THESE FIRST. These rules prevent common mistakes.**

### 1. NEVER Work Around Technical Blockers

When you encounter an error implementing the user's exact instruction:
- ✅ **STOP immediately**
- ✅ **Show the user the exact error**
- ✅ **ASK how to proceed**
- ❌ **DO NOT implement your own workaround without permission**

**Example of WRONG behavior:**
- User says: "Use SIDON upsampling"
- You encounter: SIDON device conflict error
- You did: Switched to torchaudio resampling instead ❌
- **Correct behavior**: Report the error and ask how to fix SIDON

### 2. Follow Exact Instructions

- If user says "use X", don't switch to "Y" because it's easier or you think it's better
- Only change approach with **explicit permission** from the user
- Don't make assumptions about what the user "really wants"

### 3. Ask Before Implementing

When unsure about the approach:
- Ask clarifying questions
- Present options if there are multiple ways to solve the problem
- Wait for user direction before proceeding

---

## Quick Start Guide

### Training (Phase 12: Dual-Head Decoder - ⭐ CURRENT)

**CURRENT APPROACH: Dual-head decoder (24kHz + 48kHz) with pretrained SNAC decoder**

```bash
# Start Phase 12 training
./train_dual_head_48khz.sh 0    # GPU 0
./train_dual_head_48khz.sh 3    # GPU 3

# Monitor logs
tail -f /tmp/phase12_dual_head_gpu3.log

# Manual training
uv run python finetune_dual_head_48khz_cached.py \
    --device 0 \
    --batch_size 96 \
    --epochs 15 \
    --warmup_epochs 1 \
    --lr 2e-4 \
    --main_lr 5e-5 \
    --lr_24k_final_conv 1e-5 \
    --segment_schedule "1.0,2.0,3.0,4.0" \
    --batch_multiplier "2.0,1.0,0.6,0.45" \
    --epoch_ranges "1-2,3-4,5-6,7-15" \
    --output_dir checkpoints/phase12_dual_head

# Resume from checkpoint
uv run python finetune_dual_head_48khz_cached.py \
    --device 3 \
    --resume checkpoints/phase12_dual_head/checkpoint_epoch3.pt \
    --output_dir checkpoints/phase12_dual_head
```

**Architecture**: Shared decoder → 24kHz final conv + 48kHz upsampler + final conv

**Training phases**:
- **Warmup (epoch 1)**: Only 48kHz trains, decoder frozen, LR=2e-4
- **Main (epochs 2-15)**: Both train, decoder unfrozen
  - Shared decoder + 48kHz: LR=5e-5
  - 24kHz final conv: LR=1e-5 (5× smaller)

**Curriculum (Phase 10 style)**:
- Epochs 1-2: 1.0s segments, batch=192
- Epochs 3-4: 2.0s segments, batch=96
- Epochs 5-6: 3.0s segments, batch=57
- Epochs 7-15: 4.0s segments, batch=43

**Documentation**: `PHASE12_README.md`

---

### Training (Phase 11: 48kHz Decoder - ✅ COMPLETED)

**APPROACH: Cached codes + random slicing + Phase 10 base checkpoint**

```bash
# Test training on 5k rows (for debugging)
uv run python finetune_decoder_48khz_simple_cached.py \
    --device 0 \
    --batch_size 32 \
    --epochs 15 \
    --warmup_epochs 1 \
    --limit 5000 \
    --output_dir checkpoints/phase11_decoder_48khz_test

# Start full training
uv run python finetune_decoder_48khz_simple_cached.py \
    --device 0 \
    --batch_size 32 \
    --epochs 15 \
    --warmup_epochs 1 \
    --output_dir checkpoints/phase11_decoder_48khz_partial

# Resume from checkpoint (after crash/interruption)
uv run python finetune_decoder_48khz_simple_cached.py \
    --device 0 \
    --resume checkpoints/phase11_decoder_48khz_partial/checkpoint_epoch5.pt \
    --output_dir checkpoints/phase11_decoder_48khz_partial

# Resume from best model
uv run python finetune_decoder_48khz_simple_cached.py \
    --device 0 \
    --resume checkpoints/phase11_decoder_48khz_partial/best_model.pt \
    --output_dir checkpoints/phase11_decoder_48khz_partial

# Monitor logs
tail -f /tmp/phase11_partial_gpu0.log
```

**Checkpoint Resumption:**
- ✅ Restores model, optimizer, scheduler state
- ✅ Automatically determines segment length from resumed epoch
- ✅ Restores frozen/unfrozen decoder state based on phase
- ✅ Continues training from next epoch

**Bug Fixes:**
- ✅ Shape mismatch in random slicing (fixed by padding shorter tensors)
- ✅ All scales now have consistent sizes for batch stacking

**Alternative training approaches:**
```bash
# Smart init + warmup (original approach)
./train_decoder_48khz_warmup.sh 0

# Fast training with pre-computed codes
./train_decoder_48khz_workflow.sh 0
```

**Full documentation:** `PHASE11_OPTIONS.md`, `PHASE11_README.md`, `TODO.md`

### Training (Phase 10: 24kHz Decoder - COMPLETED)

**REPRODUCIBLE TRAINING SCRIPTS:**

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

**IMPORTANT: Use the reproducible training scripts:**

**Phase 11 (48kHz):**
```bash
./train_decoder_48khz_warmup.sh 0
```

**Phase 10 (24kHz):**
```bash
./train_decoder_only.sh 0
```

These scripts handle:
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

**Current Phases**:
- **Phase 12** (active): Dual-head decoder (24kHz + 48kHz outputs)
- **Phase 11** (completed): 48kHz decoder output with random slicing
- **Phase 10** (completed): Simple fine-tuning on Levantine Arabic speech datasets (2.8M utterances)

## Training Phases

**Phase 10** (completed): Simple fine-tuning on Levantine Arabic
- Full model training (encoder + decoder)
- Curriculum learning: 1s → 2s → 3s → 4s segments
- Loss: L1 + multi-scale STFT
- Checkpoint: `checkpoints/phase10_revolab_all/best_model.pt`
- Results: 29% validation loss improvement

**Phase 11** (completed): 48kHz decoder output with random slicing
- **Goal**: Modify SNAC decoder to output 48kHz audio (instead of 24kHz)
- **Strategy**: Random segment slicing + smart weight initialization + two-phase training
- **Architecture**: Phase 10 decoder → NEW 2x upsampler → Final conv → 48kHz
- **Random Slicing**: Each sample uses random start position (dynamic curriculum learning)
- **Critical Alignment**: Multi-scale formula ensures codes + audio extract from SAME temporal position
- **Checkpoint Resumption**: `--resume` flag restores model, optimizer, scheduler, epoch, phase, segment length
- Scripts: `finetune_decoder_48khz_simple_cached.py`, `cache_codes_multigpu.py`
- Docs: `PHASE11_OPTIONS.md`, `PHASE11_README.md`

**Phase 12** (current): Dual-head decoder (24kHz + 48kHz outputs)
- **Goal**: Train dual-head decoder for fair comparison with Phase 10
- **Architecture**: Shared decoder → 24kHz final conv + 48kHz upsampler → final conv
- **Decoder**: Uses pretrained SNAC decoder (matches pretrained SNAC encoder)
- **Strategy**:
  - Warmup (epoch 1): Train only 48kHz upsampler (decoder frozen, LR: 2e-4)
  - Main (epochs 2-15): Train both heads (decoder unfrozen, 24kHz LR: 1e-5, 48kHz LR: 5e-5)
- **Curriculum**: Phase 10 style epoch ranges (1-2: 1.0s, 3-4: 2.0s, 5-6: 3.0s, 7-15: 4.0s)
- **Loss weights**: 0.3 × 24kHz + 1.0 × 48kHz
- **Fair comparison**: 24kHz head uses pretrained SNAC (same as Phase 10 baseline)
- **Smart initialization**:
  - 24kHz head: From pretrained SNAC final conv
  - 48kHz upsampler: From pretrained block 5 (first 64 channels)
- **Higher LRs**: 2x faster than original (warmup 2e-4, main 5e-5)
- **Checkpoint resumption**: Full support with `--resume` flag
- Scripts: `finetune_dual_head_48khz_cached.py`, `train_dual_head_48khz.sh`
- Docs: `PHASE12_README.md`

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

**Data Flow (Phase 11: 48kHz decoder output):**
```
Audio Input (24kHz) → Encoder (frozen) → Latent → VQ (frozen) → Quantized Codes
                                                           ↓
                                    Old Decoder (frozen during warmup, up to block 3)
                                                           ↓
                                              96-channel features at 24kHz
                                                           ↓
                                          NEW 2x Upsampler
                                                           ↓
                                              96-channel features at 48kHz
                                                           ↓
                                          Snake1d + Final Conv (96 → 1)
                                                           ↓
                                                  Audio Output (48kHz)
```

**Phase 11 Architecture Details:**
- Old decoder upsampling rates: `[8, 8, 4, 2]` = 512x total (outputs 24kHz)
- New decoder adds an extra 2x upsampler = 1024x total (outputs 48kHz)
- Two-phase training:
  - **Warmup (epochs 1-3)**: Train only new upsampler + final conv (LR: 5e-5)
  - **Main (epochs 4-15)**: Unfreeze entire decoder (LR: 1e-5)

### Configuration Files

**Phase 10** (completed):
- `configs/phase10_revolab_all.json` - Full dataset fine-tuning with curriculum
- `configs/phase10_curriculum.json` - Curriculum learning configuration
- `configs/phase10_combined.json` - Combined Levantine datasets

**Phase 11** (current):
- `configs/phase11_decoder_48khz.json` - 48kHz decoder training config

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

**Key config parameters (Phase 11):**
```json
{
  "pretrained_model": "hubertsiuzdak/snac_24khz",
  "train_data": "/mnt/data/combine/train/audio",
  "val_data": "/mnt/data/combine/valid/audio",
  "num_epochs": 15,
  "warmup_epochs": 3,
  "learning_rate": 1e-5,
  "warmup_learning_rate": 5e-5,
  "batch_size": 32,
  "segment_length": 4.0,
  "l1_weight": 1.0,
  "stft_weight": 1.0,
  "n_ffts": [1024, 2048, 4096, 8192]
}
```

## Loss Functions

### Phase 10 & 11: Reconstruction Loss

```
loss = l1_weight * L1_loss + stft_weight * multi_scale_STFT_loss
```

- **L1 loss**: Time-domain reconstruction error (MAE)
- **Multi-scale STFT loss**: Frequency-domain reconstruction at FFT sizes `[1024, 2048, 4096]` (Phase 10) or `[1024, 2048, 4096, 8192]` (Phase 11)

This is a simple reconstruction loss - no adversarial, speaker, or feature matching losses.

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

In Phase 11: Encoder and VQ are frozen throughout training.

To train full model, set both to `false`.

### Phase 11: Cached Codes Optimization

For faster training, pre-compute encoder+VQ codes once:
- **Location**: `/mnt/data/codes_phase11/`
- **Train**: `/mnt/data/codes_phase11/train/` (parquet files)
- **Val**: `/mnt/data/codes_phase11/val/`
- **Caching script**: `cache_codes_multigpu.py` (uses all 4 GPUs in parallel)
- **Monitor**: `tail -f /tmp/caching_train_gpu{0,1,2,3}.log`

### Hierarchical Code Slicing with Segment Length Schedule

**IMPORTANT**: When using segment length schedule (curriculum learning), the codes are **NOT using full codes** - they are sliced according to the segment length.

**SNAC Hierarchical VQ Structure:**
- Encoder downsampling: `[3, 3, 7, 7]` = 441x total
- 24kHz / 441 = ~54.4 Hz latent frame rate
- VQ strides: `[8, 4, 2, 1]` for 4 codebook levels (only first 3 used in decoder)
  - Scale 0: stride 8 (coarsest, covers longest time span)
  - Scale 1: stride 4 (medium)
  - Scale 2: stride 2 (fine)
  - Scale 3: stride 1 (finest, not used in cached codes)

**Token Counts per Segment Length:**

| Segment | Latent Frames | Scale 0 (÷8) | Scale 1 (÷4) | Scale 2 (÷2) | Audio Samples |
|---------|--------------|--------------|--------------|--------------|---------------|
| 1.0s | 55 | 7 tokens | 14 tokens | 28 tokens | 48,000 |
| 2.0s | 109 | 14 tokens | 28 tokens | 55 tokens | 96,000 |
| 3.0s | 164 | 21 tokens | 41 tokens | 82 tokens | 144,000 |
| 4.0s | 218 | 28 tokens | 55 tokens | 109 tokens | 192,000 |

**Note**: Uses ceiling division (`math.ceil`) to ensure proper alignment when upsampled.

**Code Slicing Implementation:**
```python
import math

def _update_segment_info(self):
    """Calculate token counts for current segment length."""
    num_latent_frames = int(math.ceil(self.segment_length * self.latent_frame_rate))
    # Use ceiling division to ensure proper alignment when upsampled
    self.num_frames_scale0 = math.ceil(num_latent_frames / 8)
    self.num_frames_scale1 = math.ceil(num_latent_frames / 4)
    self.num_frames_scale2 = math.ceil(num_latent_frames / 2)
    self.num_audio_samples = int(self.segment_length * 48000)

# In __getitem__():
codes = [
    codes[0][:self.num_frames_scale0],  # Scale 0: coarsest (stride 8)
    codes[1][:self.num_frames_scale1],  # Scale 1: medium (stride 4)
    codes[2][:self.num_frames_scale2],  # Scale 2: fine (stride 2)
]
```

This ensures each scale is sliced independently according to its temporal resolution, preventing mismatched code-audio alignments during training with varying segment lengths.

### Random Segment Slicing (Phase 11 Enhancement)

**IMPORTANT**: Phase 11 uses **random segment slicing** during training for dynamic curriculum learning. Instead of always slicing from position 0, each sample uses a random start position.

**Multi-Scale Alignment Formula (CRITICAL):**

For random slicing to work correctly, all 3 code scales must extract from the **SAME temporal position**:

```python
# Calculate random start position (in scale 0 token space)
vq_strides = [8, 4, 2]  # hierarchical strides for scales 0, 1, 2

max_start_pos = min(
    max(0, full_len_scale0 - num_frames_scale0),
    max(0, full_len_scale1 - num_frames_scale1),
    max(0, full_len_scale2 - num_frames_scale2)
)
start_pos = random.randint(0, max_start_pos) if max_start_pos > 0 else 0

# CRITICAL: Multi-scale alignment formula
# Ensures all scales extract from SAME time position
scale_start = start_pos * (vq_strides[0] // vq_strides[scale])

codes = [
    codes[0][start_pos * (8 // 8) : start_pos * (8 // 8) + num_frames_scale0],  # Scale 0
    codes[1][start_pos * (8 // 4) : start_pos * (8 // 4) + num_frames_scale1],  # Scale 1: start_pos * 2
    codes[2][start_pos * (8 // 2) : start_pos * (8 // 2) + num_frames_scale2],  # Scale 2: start_pos * 4
]

# Audio slicing (aligned with same random position)
hop_length = 441
start_sample_48k = start_pos * hop_length * vq_strides[0] * 2  # 2x for 48kHz
audio = audio[:, start_sample_48k:start_sample_48k + num_audio_samples]
```

**Verification Example (start_pos = 5):**
```
Scale 0: token 5  → 5 * 8 * 441 = 17,640 samples @ 24kHz = 0.735s
Scale 1: token 10 → 10 * 4 * 441 = 17,640 samples @ 24kHz = 0.735s ✅
Scale 2: token 20 → 20 * 2 * 441 = 17,640 samples @ 24kHz = 0.735s ✅
Audio 48kHz: sample 35,280 = 0.735s ✅
```

**Benefits:**
- ✅ Each epoch sees different segments (better generalization)
- ✅ Same cached file produces multiple random slices
- ✅ More diverse training data without storing duplicates
- ✅ Critical alignment prevents garbled audio from misaligned codes

**Implementation**: `finetune_decoder_48khz_simple_cached.py` (CachedCodesDataset class)

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
- `checkpoints/phase11_decoder_48khz/best_model.pt` - 48kHz decoder (smart init + warmup)

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
├── finetune.py                       # Core training script (Phase 10)
├── finetune_decoder_48khz.py         # Phase 11: Standard 48kHz training (random init)
├── finetune_decoder_48khz_warmup.py  # Phase 11: Smart init + warmup (⭐ RECOMMENDED)
├── finetune_decoder_48khz_fast.py    # Phase 11: Fast training with pre-computed codes
├── finetune_decoder_48khz_cached_codes.py  # Phase 11: Training with cached codes
├── inference.py                      # Inference script
├── generate.py                       # Generation script
├── prepare_dataset_folder.py         # Dataset preparation
│
├── train_decoder_only.sh             # Phase 10: Training launcher
├── train_decoder_48khz.sh            # Phase 11: Standard training launcher
├── train_decoder_48khz_warmup.sh     # Phase 11: Smart init + warmup launcher (⭐ RECOMMENDED)
├── train_decoder_48khz_workflow.sh   # Phase 11: Fast workflow launcher
├── train_status.sh                   # Training status checker
│
├── TRAINING_README.md                # Phase 10 training docs
├── PHASE11_README.md                 # Phase 11 training docs
├── PHASE11_OPTIONS.md                # Phase 11: 3 training approaches comparison
├── CLAUDE.md                         # This file
├── README.md                         # Project README
├── setup.py                          # Package setup
├── requirements.txt                  # Dependencies
│
├── configs/                          # Training configurations
│   ├── phase10_revolab_all.json      # Phase 10 config (decoder-only)
│   ├── phase10_curriculum.json
│   ├── phase10_combined.json
│   └── phase11_decoder_48khz.json    # Phase 11 config (48kHz output)
│
├── snac/                             # Main package
│   ├── snac.py                       # SNAC model
│   ├── layers.py                     # Encoder/Decoder
│   ├── vq.py                         # Vector quantization
│   ├── dataset.py                    # Dataset classes
│   └── ... (other modules)
│
├── checkpoints/                      # Trained models
│   ├── phase10_revolab_all/          # Phase 10: 24kHz decoder
│   └── phase11_decoder_48khz/        # Phase 11: 48kHz decoder
│       ├── best_model.pt
│       └── checkpoint_epoch*.pt
│
└── logs/                             # Training logs
    ├── phase10_decoder_only/
    └── phase11_decoder_48khz/
```

## First Thing To Check In Any New Session

```bash
# Check if any training is still running
ps aux | grep -E "(finetune|train)" | grep -v grep

# Check if caching is running (Phase 11)
ps aux | grep cache_codes_multigpu | grep -v grep

# Or use the status script
./train_status.sh

# Check GPU status
nvidia-smi

# Check recent logs (if training exists)
ls -lt logs/*/training.log 2>/dev/null | head -1

# For Phase 11 specific check
ps aux | grep finetune_decoder_48khz

# Check Phase 11 caching progress (if running)
ls -lh /mnt/data/codes_phase11/train/*.parquet 2>/dev/null | wc -l
```

## Training Scripts Structure

**Root level scripts:**

**Phase 12 (Dual-head decoder):**
- `train_dual_head_48khz.sh` - ⭐ CURRENT: Dual-head training script
- `finetune_dual_head_48khz_cached.py` - Dual-head training with cached codes
- `PHASE12_README.md` - Full Phase 12 documentation

**Phase 11 (48kHz decoder):**
- `train_decoder_48khz_warmup.sh` - ⭐ Smart init + warmup (RECOMMENDED)
- `train_decoder_48khz_workflow.sh` - Fast training with pre-computed codes
- `train_decoder_48khz.sh` - Standard training (random init)
- `PHASE11_OPTIONS.md` - Comparison of 3 training approaches
- `PHASE11_README.md` - Full Phase 11 documentation
- `finetune_decoder_48khz_warmup.py` - Smart initialization training
- `finetune_decoder_48khz_fast.py` - Fast training with pre-computed codes
- `finetune_decoder_48khz_simple_cached.py` - Cached codes + Phase 10 init + OneCycleLR + segment schedule
- `cache_codes_multigpu.py` - Multi-GPU caching script (uses all 4 GPUs)
- `cache_codes_full.py` - Single-GPU caching (slow, deprecated)
- `cache_codes_full_batched.py` - Single-GPU batched caching (deprecated)
- `precompute_codes.py` - Pre-compute quantized codes (deprecated)

**Phase 10 (24kHz decoder):**
- `train_decoder_only.sh` - Main training script (reproducible, background mode)
- `train_status.sh` - Check running training status
- `TRAINING_README.md` - Full training documentation
- `finetune.py` - Core training script (used by train_decoder_only.sh)

**Common scripts:**
- `inference.py` - Inference script
- `generate.py` - Generation script
- `prepare_dataset_folder.py` - Dataset preparation

**To start Phase 12 training:**
```bash
./train_dual_head_48khz.sh 0  # GPU 0
```

**To start Phase 11 training:**
```bash
./train_decoder_48khz_warmup.sh 0  # GPU 0
```

**To start Phase 10 training:**
```bash
./train_decoder_only.sh 0  # GPU 0
```

**To check status:**
```bash
./train_status.sh
```

**Training outputs:**
- Phase 12: `checkpoints/phase12_dual_head/`, `/tmp/phase12_dual_head_gpu<N>.log`
- Phase 11: `checkpoints/phase11_decoder_48khz/`, `/tmp/phase11_partial_gpu<N>.log`
- Phase 10: `checkpoints/phase10_revolab_all/`, `logs/phase10_decoder_only/training.log`
- Background logs: `/tmp/phase*_gpu<N>.log`
- PID files: `/tmp/phase*_gpu<N>.pid`

**Phase 11 Cached Codes (for fast training):**
- Cached encoder+VQ codes location: `/mnt/data/codes_phase11/`
- Train codes: `/mnt/data/codes_phase11/train/` (parquet files: `codes_batch_gpu0_*.parquet`, etc.)
- Val codes: `/mnt/data/codes_phase11/val/`
- Multi-GPU caching script: `cache_codes_multigpu.py` (uses all 4 GPUs in parallel)
- Training script with cached codes: `finetune_decoder_48khz_cached_codes.py`
- Monitor caching: `tail -f /tmp/caching_train_gpu{0,1,2,3}.log`
- Check caching progress: `ls -lh /mnt/data/codes_phase11/train/*.parquet | wc -l`

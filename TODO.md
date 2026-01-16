# Phase 12 Training TODO - Dual-Head Decoder (24kHz + 48kHz)

## Goal
Train **dual-head decoder** that outputs **both 24kHz and 48kHz audio** from the same shared encoder+VQ backbone.

## Key Innovation
Unlike Phase 11 (48kHz-only), Phase 12 maintains the 24kHz output head from pretrained SNAC, enabling:
1. **Direct comparison**: 24kHz loss vs Phase 10's 24kHz loss (same metrics!)
2. **Pretrained baseline**: 24kHz head uses pretrained SNAC (same as Phase 10 baseline)
3. **Multi-resolution output**: Generate either 24kHz or 48kHz as needed
4. **Better training**: 24kHz loss stabilizes shared decoder for 48kHz learning

## Current Status

**🚀 TRAINING RUNNING ON GPU 3**

### Training Progress:
- **Current**: Epoch 4/15, resumed from checkpoint (epoch 3, batch 60476)
- **Location**: `checkpoints/phase12_dual_head/`
- **Logs**: `/tmp/phase12_dual_head_gpu3.log`
- **Dataset**: 3,447,109 training + 55,419 validation samples
- **VRAM**: 112GB / 144GB (GPU 3) - higher in main phase
- **Segment**: 2.0s (epochs 3-4)

### Recent Improvements (Jan 16, 2026):
- ✅ **Increased LRs** (2x faster): warmup 2e-4, main 5e-5, 24kHz 1e-5
- ✅ **Phase 10 style curriculum**: Epoch ranges (1-2: 1.0s, 3-4: 2.0s, 5-6: 3.0s, 7-15: 4.0s)
- ✅ **Checkpoint saving bug fix**: Handle None optimizer in warmup
- ✅ **Checkpoint resumption**: Full support with `--resume` flag
- ✅ **File organization**: Deprecated scripts moved to `archive/`

### Architecture
```
Shared Decoder (blocks 0-5) → 64ch @ 24kHz
                                ├→ 24kHz Final Conv → 24kHz output (pretrained SNAC)
                                └→ 2x Upsampler → 48kHz Final Conv → 48kHz output (new)
```

### Implemented Features:
- **✅ Dual-head decoder** - `DualHeadDecoder` class
  - Shared decoder (blocks 0-5) from pretrained SNAC (matches pretrained encoder)
  - 24kHz head: Single Conv1d (64 → 1), initializes from pretrained SNAC
  - 48kHz head: Snake1d → DecoderBlock(stride=2) → Conv1d (64 → 1)

- **✅ Smart weight initialization**
  - 24kHz head: From pretrained SNAC final conv
  - 48kHz upsampler: From pretrained block 5 (first 64 channels)

- **✅ Two-phase training with different LRs**
  - Warmup (epoch 1): Only 48kHz trains, decoder frozen, LR: 2e-4 (INCREASED)
  - Main (epochs 2-15): Both train, 24kHz LR: 1e-5, 48kHz LR: 5e-5 (INCREASED)

- **✅ Phase 10 style curriculum**
  - Epochs 1-2: 1.0s segments, batch=192
  - Epochs 3-4: 2.0s segments, batch=96
  - Epochs 5-6: 3.0s segments, batch=57
  - Epochs 7-15: 4.0s segments, batch=43

- **✅ Weighted dual loss**
  - Loss weights: 0.3 × 24kHz + 1.0 × 48kHz
  - 24kHz for monitoring in warmup, for training in main phase

- **✅ Checkpoint resumption**
  - Restores model, optimizer, scheduler, epoch, phase, segment length
  - Use `--resume checkpoints/phase12_dual_head/checkpoint_epoch3.pt`

### Implementation Tasks:
- [x] Create `DualHeadDecoder` class with two output heads
- [x] Load pretrained SNAC weights for 24kHz head
- [x] Implement smart weight initialization for 48kHz upsampler
- [x] Add two-phase training with different LRs per component
- [x] Implement weighted dual loss (24kHz + 48kHz)
- [x] Fix device placement for pretrained SNAC model
- [x] Add epoch_ranges parameter for Phase 10 style curriculum
- [x] Increase learning rates for faster training (2x)
- [x] Fix checkpoint saving bug (None optimizer handling)
- [x] Implement checkpoint resumption with full state restore
- [x] Start training on GPU 3
- [x] Resume from checkpoint epoch 3

### File Organization:
- **Root**: Core training scripts (finetune_dual_head_48khz_cached.py, finetune.py, etc.)
- **archive/**: Deprecated Phase 11 variants, old shell scripts
- **scripts/**: Precompute, caching, utility scripts
- **tests/**: Test scripts for cached codes, random slicing

---

# Phase 11 Training (COMPLETED ✅)

## Goal
Train 48kHz decoder with **random segment slicing** during training (dynamic curriculum learning)

## Status: COMPLETED
- **Location**: `checkpoints/phase11_decoder_48khz_full/`
- **Dataset**: 3.4M training + 55k validation samples (cached)
- **Features**: Validation, checkpointing, time-based checkpoints, best model saving

### Key Features Implemented:
- ✅ Random slicing with critical multi-scale alignment
- ✅ Smart weight initialization from pretrained SNAC
- ✅ Checkpoint resumption with `--resume` flag
- ✅ Validation function with 55,419 samples
- ✅ Advanced checkpointing (every 5000 iterations, every 30 min)
- ✅ Shape mismatch bug fix (padding shorter tensors)
- ✅ Testing mode with `--limit` option

---



## Known Issues (Non-Critical)

### Code Quality Issues
**These issues don't affect training but could be improved:**

1. **Duplicate `reconstruction_loss` function** (lines 239 and 414)
   - Function defined twice, second definition overwrites first
   - Impact: Code maintenance issue, no runtime effect
   - Fix: Remove first definition (lines 239-249)

2. **Dataset loaded twice at startup** (lines 481-486 and 575-580)
   - Same dataset created twice before and after resume logic
   - Impact: ~30-60 seconds wasted at startup
   - Fix: Move first dataset loading after resume logic

3. **No mixed precision training (AMP)**
   - Training uses full float32 precision
   - Impact: 1.5-2x slower training, higher memory usage
   - Fix: Add `torch.cuda.amp.autocast()` and `GradScaler`

4. **Checkpoint missing metadata**
   - Checkpoints don't save `limit`, `val_limit`, `segment_schedule`
   - Impact: Resumed training won't respect these if used
   - Fix: Add these fields to checkpoint dictionary

5. **No signal handling for crashes**
   - If process is killed (SIGTERM), no emergency checkpoint saved
   - Impact: Could lose progress since last checkpoint
   - Fix: Add signal handler for SIGTERM/SIGINT

**Current training status**: None of these issues are critical enough to interrupt the running full-scale training. Can be addressed in future runs.

## Architecture
```
Raw Audio (24kHz) → SNAC Encoder → Codes (frozen, pre-computed FULL)
                                                ↓
                         Random Segment Slicer (per epoch, per batch)
                                                ↓
                         Slice: 1s → 2s → 3s → 4s (schedule)
                                                ↓
                          Decoder (frozen epoch 1, unfrozen epoch 2+)
                                                ↓
                         Upsampler Layer (2x, smart weight init)
                                                ↓
                                  48kHz Audio Output
                                                ↓
                            Match with pre-generated 48kHz audio (same slice)
```

## Training Strategy

### Two-Phase Training

**Phase 1 (Warmup)**: Epoch 1
- Train ONLY new layers: upsampler + final conv
- Pretrained decoder: FROZEN
- Learning rate: 5e-5
- Goal: Adapt new layers without distilling pretrained features

**Phase 2 (Main)**: Epochs 2-15
- Unfreeze entire decoder
- OneCycleLR scheduler with peak LR = 1e-4
- Goal: Fine-tune entire model for optimal 48kHz output

### Segment Length Schedule
- Epochs 1-3: 1.0s segments (fast iterations, foundation)
- Epochs 4-6: 2.0s segments (medium context)
- Epochs 7-9: 3.0s segments (longer context)
- Epochs 10-15: 4.0s segments (full context, refinement)

## Multi-Scale Random Slicing

### VQ Strides and Token Counts

**SNAC 24kHz Parameters:**
- Encoder downsampling: [3, 3, 7, 7] = 441x total
- Latent frame rate: 24kHz / 441 ≈ 54.4 Hz
- VQ strides: [8, 4, 2, 1] for 4 codebook levels (only first 3 used)

**Token Counts per Segment Length:**

| Segment | Latent Frames | Scale 0 (÷8) | Scale 1 (÷4) | Scale 2 (÷2) | Audio Samples |
|---------|--------------|--------------|--------------|--------------|---------------|
| 1.0s | 54 | 6 tokens | 12 tokens | 24 tokens | 48,000 |
| 2.0s | 108 | 12 tokens | 24 tokens | 48 tokens | 96,000 |
| 3.0s | 162 | 18 tokens | 36 tokens | 72 tokens | 144,000 |
| 4.0s | 216 | 24 tokens | 48 tokens | 96 tokens | 192,000 |

**Note**: Uses floor division (rounding down) to ensure all scales align perfectly.

### Critical Alignment Formula

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

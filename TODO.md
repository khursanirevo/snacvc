# Phase 11 Training TODO - Random Segment Slicing

## Goal
Train 48kHz decoder with **random segment slicing** during training (dynamic curriculum learning)

## Current Status

**🎉 TESTING IN PROGRESS: 50k rows with validation + advanced checkpointing**

### Training Progress:
- **Current**: Testing on 50k rows with full features
- **Location**: `checkpoints/phase11_decoder_48khz_test/`
- **Logs**: `logs/phase11_decoder_48khz_test/training.log`
- **Features**: Validation, 5k-iteration checkpoints, time-based checkpointing

### Caching (COMPLETE ✅):
- **Training**: 2.8M files → 2857 parquet files in `/mnt/data/codes_phase11/train/`
- **Validation**: 55,419 files → 56 parquet files in `/mnt/data/codes_phase11/val/`
- **Status**: Complete! Ready for full-scale training

### Implemented Features:
- **✅ Checkpoint Resumption** - `--resume` flag
  - Restores model, optimizer, scheduler state
  - Automatically determines segment length from resumed epoch
  - Restores frozen/unfrozen decoder state based on phase

- **✅ Smart Weight Initialization** - Implemented in `finetune_decoder_48khz_simple_cached.py`
  - Fastai technique: Average pretrained weights (128→64 channels)
  - Source: Pretrained decoder `blocks[5]` (2x upsampler)
  - Better convergence than random initialization

- **✅ Random Slicing with Critical Alignment** - Implemented in `CachedCodesDataset`
  - Random start position calculation for each sample
  - Multi-scale alignment formula: `scale_start = start_pos * (vq_strides[0] // vq_strides[scale])`
  - Audio slicing aligned with same random position
  - **BUG FIX**: Padding shorter tensors to prevent shape mismatch in batches

- **✅ Validation Function** - Complete with 55,419 validation samples
  - Runs after each epoch
  - Computes L1 + STFT loss on validation set
  - Best model saved based on validation loss

- **✅ Advanced Checkpointing**:
  - Checkpoint every 5000 iterations: `checkpoint_epoch{N}_iter{M}.pt`
  - Time-based checkpointing every 30 min: `checkpoint_epoch{N}_time_{timestamp}.pt`
  - End-of-epoch checkpoints: `checkpoint_epoch{N}.pt`
  - Best model: `best_model.pt` (based on validation loss)

- **✅ Testing Mode** - `--limit` option
  - Test training on small subset (e.g., 50000 rows)
  - Verify bug fixes before committing to full-scale training

### Implementation Tasks:
- [x] Stop old caching processes
- [x] Remove old parquet files
- [x] Modify `cache_codes_multigpu.py` to store FULL codes (no slicing)
- [x] Implement smart weight initialization (fastai approach)
- [x] Start multi-GPU caching with full codes
- [x] Implement random slicing dataset class in `CachedCodesDataset`
- [x] Add critical multi-scale alignment for random start positions
- [x] Align audio slicing with codes (same random position)
- [x] Fix shape mismatch bug (padding shorter tensors)
- [x] Add `--limit` option for testing
- [x] Add validation function
- [x] Add checkpoint every 5000 iterations
- [x] Add time-based checkpointing (every 30 min)
- [x] Cache validation dataset (55,419 files)
- [ ] Complete 50k row test run
- [ ] Full-scale training (all 2.8M rows + validation)

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

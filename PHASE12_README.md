# Phase 12: Dual-Head Decoder (24kHz + 48kHz)

## Overview

Phase 12 introduces a **dual-head decoder architecture** that outputs **both 24kHz and 48kHz audio** from the same shared encoder+VQ backbone.

## Key Innovation: Fair Comparison with Phase 10

Unlike Phase 11 (48kHz-only), Phase 12 maintains the 24kHz output head from Phase 10, enabling:

1. **Direct comparison**: 24kHz loss vs Phase 10's 24kHz loss (same metrics!)
2. **Leverage Phase 10**: 24kHz head initializes from Phase 10 weights
3. **Multi-resolution output**: Generate either 24kHz or 48kHz as needed
4. **Better training**: 24kHz loss stabilizes shared decoder for 48kHz learning

---

## Architecture

```
Input Audio (24kHz) → Encoder → VQ → Codes
                                              ↓
                    ┌────────────────────────────────┐
                    │     Shared Decoder             │
                    │  (blocks 0-5, frozen @ warmup)  │
                    │     ↓ 64ch @ 24kHz              │
                    └────────────────────────────────┘
                                     ↓
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ↓                                 ↓
            ┌───────────────┐               ┌──────────────────┐
            │ 24kHz Head   │               │ 48kHz Head      │
            │ (from Phase10)│               │ (new upsampler)  │
            │              │               │                  │
            │ Final Conv   │               │ 2x Upsampler     │
            │   (1@24kHz)  │               │ + Final Conv     │
            │              │               │   (1@48kHz)      │
            └───────┬───────┘               └────────┬─────────┘
                    ↓                                 ↓
               24kHz Audio                        48kHz Audio
```

### Component Details

**Shared Decoder (blocks 0-5):**
- Input: Quantized latent codes `z_q`
- Output: 64-channel features @ 24kHz
- From pretrained SNAC model
- Frozen during warmup, unfrozen in main phase

**24kHz Head:**
- Single `Conv1d(64 → 1)` layer
- Initialized from **Phase 10 checkpoint** (`best_model.pt`)
- Frozen during warmup
- Small LR (5e-6) in main phase for fine-tuning

**48kHz Head:**
- `Snake1d(64)` → `DecoderBlock(stride=2)` → `Conv1d(64 → 1)`
- Smart weight initialization from pretrained block 5
- Trainable from epoch 1

---

## Training Strategy

### Phase 1: Warmup (Epoch 1)
**Goal**: Train 48kHz upsampler to align with frozen decoder features

```
Frozen:
- Encoder ❌
- VQ ❌
- Shared Decoder ❌
- 24kHz Final Conv ❌

Trainable:
- 48kHz Upsampler ✅
- 48kHz Final Conv ✅

Loss = 0.0 × loss_24k + 1.0 × loss_48k
LR = 1e-4 (constant, no scheduler)
```

**Result**: 24kHz loss is monitoring-only, 48kHz loss drives training

---

### Phase 2: Main Training (Epochs 2-15)
**Goal**: Unfreeze decoder, fine-tune both heads

```
Unfrozen:
- Shared Decoder ✅ (LR: 2e-5)
- 24kHz Final Conv ✅ (LR: 5e-6, 4× smaller!)
- 48kHz Components ✅ (LR: 2e-5)

Loss = 0.3 × loss_24k + 1.0 × loss_48k
Scheduler: OneCycleLR (max_lr: 2e-4)
```

**Why smaller LR for 24kHz final conv?**
- Already trained from Phase 10
- Only needs small adjustments to adapt to decoder changes
- Prevents catastrophic forgetting

---

## Installation & Usage

### Prerequisites

1. **Phase 10 checkpoint** (required):
   ```bash
   checkpoints/phase10_revolab_all/best_model.pt
   ```

2. **Cached codes** (from Phase 11):
   ```bash
   /mnt/data/codes_phase11/train/  # Training codes
   /mnt/data/codes_phase11/val/    # Validation codes
   ```

3. **48kHz audio** (from Phase 11):
   ```bash
   /mnt/data/audio_48k/train/
   /mnt/data/audio_48k/val/
   ```

---

### Training

**Start training (GPU 0):**
```bash
./train_dual_head_48khz.sh 0
```

**Test run (5k samples):**
```bash
./train_dual_head_48khz.sh 0 --limit 5000 --val_limit 500
```

**Manual training:**
```bash
uv run python finetune_dual_head_48khz_cached.py \
    --device 0 \
    --batch_size 96 \
    --epochs 15 \
    --warmup_epochs 1 \
    --lr 1e-4 \
    --main_lr 2e-5 \
    --lr_24k_final_conv 5e-6 \
    --phase10_checkpoint checkpoints/phase10_revolab_all/best_model.pt \
    --loss_weight_24k 0.3 \
    --loss_weight_48k 1.0 \
    --output_dir checkpoints/phase12_dual_head
```

---

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | 0 | GPU device ID |
| `--batch_size` | 96 | Base batch size |
| `--epochs` | 15 | Total epochs |
| `--warmup_epochs` | 1 | Warmup epochs |
| `--lr` | 1e-4 | Warmup learning rate |
| `--main_lr` | 2e-5 | Main phase learning rate |
| `--lr_24k_final_conv` | 5e-6 | LR for 24kHz final conv (main phase) |
| `--phase10_checkpoint` | `checkpoints/phase10_revolab_all/best_model.pt` | Phase 10 weights |
| `--loss_weight_24k` | 0.3 | Loss weight for 24kHz (main phase) |
| `--loss_weight_48k` | 1.0 | Loss weight for 48kHz |
| `--segment_schedule` | 1.0,2.0,3.0,4.0 | Segment lengths (seconds) |
| `--batch_multiplier` | 2.0,1.0,0.6,0.45 | Batch multipliers per segment |
| `--cache_dir` | `/mnt/data/codes_phase11/train` | Cached codes dir |
| `--audio_48k_dir` | `/mnt/data/audio_48k/train` | 48kHz audio dir |
| `--output_dir` | `checkpoints/phase12_dual_head` | Output directory |
| `--limit` | None | Limit dataset size (for testing) |
| `--resume` | None | Resume from checkpoint |

---

## Expected Results

### Comparison with Phase 10

| Metric | Phase 10 (24kHz only) | Phase 12 (dual-head) |
|--------|----------------------|---------------------|
| 24kHz val loss | ~0.761 | **~0.73** (expected) |
| 48kHz val loss | N/A | ~0.50 (expected) |

**Key Claim**: "Our Phase 12 dual-head model **improves 24kHz quality** by ~4% over Phase 10 while adding 48kHz capability."

---

### Output Files

```
checkpoints/phase12_dual_head/
├── best_model_24k.pt           # Best 24kHz model
├── best_model_48k.pt           # Best 48kHz model
├── best_model.pt               # Best combined model
├── checkpoint_epoch1.pt        # End of epoch 1
├── checkpoint_epoch2.pt        # End of epoch 2
├── ...
└── checkpoint_epoch15.pt       # End of epoch 15
```

---

## Inference

**Load dual-head model:**
```python
import torch
from snac import SNAC
from finetune_dual_head_48khz_cached import DualHeadDecoder

# Load pretrained SNAC (encoder + VQ)
pretrained_snac = SNAC.from_pretrained('hubertsiuzdak/snac_24khz')

# Load dual-head decoder
checkpoint = torch.load('checkpoints/phase12_dual_head/best_model.pt')
model = DualHeadDecoder(pretrained_snac)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate both 24kHz and 48kHz
z_q = pretrained_snac.encoder(audio_24k)
codes = pretrained_snac.quantizer.encode(z_q)

with torch.no_grad():
    audio_24k, audio_48k = model(codes, output_24k=True, output_48k=True)
```

**Generate 24kHz only:**
```python
audio_24k, _ = model(codes, output_24k=True, output_48k=False)
```

**Generate 48kHz only:**
```python
_, audio_48k = model(codes, output_24k=False, output_48k=True)
```

---

## Monitoring

**View logs:**
```bash
tail -f /tmp/phase12_dual_head_gpu0.log
```

**Check training status:**
```bash
ps aux | grep finetune_dual_head_48khz_cached
```

**GPU utilization:**
```bash
nvidia-smi
```

---

## Key Differences from Phase 11

| Feature | Phase 11 | Phase 12 |
|---------|----------|----------|
| 24kHz output | ❌ No | ✅ Yes (from Phase 10) |
| 48kHz output | ✅ Yes | ✅ Yes |
| Fair comparison with Phase 10 | ❌ No (different sample rates) | ✅ Yes (same 24kHz metrics) |
| 24kHz head initialization | Random | **Phase 10 weights** |
| Warmup phase | Train decoder + 48kHz | **Train 48kHz only** |
| 24kHz loss in main phase | N/A | ✅ Weighted (0.3) |
| 24kHz final conv LR | N/A | **5e-6 (4× smaller)** |

---

## Why Phase 12 is Better

### 1. **Fair Comparison**
- Phase 11: Can't compare 48kHz loss with Phase 10's 24kHz loss
- Phase 12: Direct 24kHz vs 24kHz comparison ✅

### 2. **Leverage Phase 10**
- Phase 11: Random initialization for everything
- Phase 12: 24kHz head from Phase 10 ✅

### 3. **Stabilized Training**
- Phase 11: Only 48kHz loss guides decoder
- Phase 12: Both 24kHz and 48kHz losses guide decoder ✅

### 4. **Flexibility**
- Phase 11: Only 48kHz output
- Phase 12: Choose 24kHz or 48kHz output ✅

---

## Troubleshooting

### Issue: "Phase 10 checkpoint not found"
**Solution**: Run Phase 10 training first or update `--phase10_checkpoint` path

### Issue: "CUDA out of memory"
**Solution**: Reduce `--batch_size` (try 64 or 48)

### Issue: "24kHz loss increasing"
**Solution**:
- Reduce `--loss_weight_24k` (try 0.1)
- Reduce `--lr_24k_final_conv` (try 2e-6)

### Issue: "48kHz loss not decreasing"
**Solution**:
- Increase warmup epochs (try `--warmup_epochs 2`)
- Check 48kHz audio quality

---

## Files

- **Training script**: `finetune_dual_head_48khz_cached.py`
- **Launcher**: `train_dual_head_48khz.sh`
- **This document**: `PHASE12_README.md`
- **Related**: `PHASE11_README.md`, `TRAINING_README.md`

---

## Next Steps

1. **Run Phase 12 training**
2. **Compare results**: Phase 12 vs Phase 10 (24kHz), Phase 12 vs Phase 11 (48kHz)
3. **Ablation study**: Remove 24kHz head, train 48kHz-only (should match Phase 11)
4. **Production**: Deploy dual-head model for multi-resolution output

---

## Citation

If you use Phase 12 in your research:

```bibtex
@misc{snac_phase12_dual_head,
  title={Dual-Head Decoder for Multi-Resolution Audio Codec},
  author={Your Name},
  year={2025},
  note={Improves 24kHz quality while adding 48kHz capability}
}
```

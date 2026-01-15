# Phase 11 Training Options Summary

We now have **3 different training approaches** for Phase 11 (48kHz decoder):

---

## Option 1: Smart Initialization + Warmup ⭐ **RECOMMENDED**

**Best convergence** - Copy existing weights + gradual training

```bash
./train_decoder_48khz_warmup.sh 0
```

**How it works:**
1. **Smart Initialization**: Copy the existing 2x upsampler layer weights to the new layer (instead of random init)
2. **Warmup Phase** (epochs 1-3): Train only the new layer with higher LR (5e-5)
3. **Main Phase** (epochs 4-15): Unfreeze entire decoder, train with normal LR (1e-5)

**Benefits:**
- ✅ No random initialization shock
- ✅ Stable training from epoch 1
- ✅ Best final quality
- ✅ Faster convergence

**Files:**
- `finetune_decoder_48khz_warmup.py`
- `train_decoder_48khz_warmup.sh`
- `configs/phase11_decoder_48khz.json` (has `warmup_epochs`)

---

## Option 2: Fast Training with Pre-computed Data ⚡ **FASTEST**

**3-5x faster** - Pre-compute both codes AND 48kHz audio

```bash
./train_decoder_48khz_fast.sh 0
```

**How it works:**
1. **Pre-computation** (one-time, ~2-4 hours):
   - Generate 48kHz audio using SIDON upsampler
   - Generate quantized codes using SNAC encoder+VQ
2. **Training**: Load both pre-computed, train decoder only

**Benefits:**
- ⚡ **3-5x faster** (no encoder/VQ/SIDON forward passes)
- 💾 **Less memory** (don't load encoder/VQ/SIDON)
- 📈 **Higher batch size** possible
- 💿 **Smaller checkpoints** (decoder only)
- 🔄 **Reproducible** (fixed precomputed data)

**Best for:**
- Large datasets
- Multiple training runs (data computed once)
- Limited GPU memory
- Production deployment

**Directory Structure:**
```
/mnt/data/combine/train/audio           (input: 24kHz)
/mnt/data/combine_48khz/train/audio    (precomputed: 48kHz)
/mnt/data/codes_phase11/train          (precomputed: codes)
```

**Files:**
- `precompute_48khz_audio.py` - Generate 48kHz audio
- `precompute_codes.py` - Generate quantized codes
- `finetune_decoder_48khz_fast.py` - Training with precomputed data
- `train_decoder_48khz_fast.sh` - Full workflow launcher

---

## Option 3: Standard Training

**Simple** - Basic decoder training with random init

```bash
./train_decoder_48khz.sh 0
```

**How it works:**
- Randomly initialize new upsampling layer
- Train entire decoder from scratch

**Drawbacks:**
- ❌ Random initialization (slow convergence)
- ❌ Less stable training
- ❌ Lower final quality

**Best for:**
- Quick experiments
- Baseline comparison

**Files:**
- `finetune_decoder_48khz.py`
- `train_decoder_48khz.sh`

---

## Comparison

| Method | Speed | Convergence | Quality | Complexity |
|--------|-------|-------------|---------|------------|
| **Warmup** | Baseline | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Low |
| **Fast (pre-computed)** | ⭐⭐⭐⭐⭐ 3-5x | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Medium |
| **Standard** | Baseline | ⭐⭐ | ⭐⭐ | Low |

---

## Recommendation

**For fastest training with best quality: Option 2 (Fast)** - pre-compute everything once, train 3-5x faster

**For simplicity: Option 1 (Warmup)** - no precomputation needed, good convergence

---

## All Training Scripts

| Script | Description |
|--------|-------------|
| `train_decoder_48khz_fast.sh` | ⚡ Fastest: Pre-compute + train (3-5x speedup) |
| `train_decoder_48khz_warmup.sh` | Smart init + warmup (simple, no precompute) |
| `train_decoder_48khz.sh` | Standard training (baseline) |

**Utility Scripts:**
| Script | Description |
|--------|-------------|
| `precompute_48khz_audio.py` | Generate 48kHz audio using SIDON |
| `precompute_codes.py` | Generate quantized codes using SNAC |

---

## Training Flow Diagram

### Option 1: Warmup
```
Pretrained SNAC 24kHz
       ↓
Copy 2x upsampler → New 2x layer (smart init)
       ↓
┌─────────────────────────────────────┐
│ Phase 1: Warmup (epochs 1-3)       │
│  - Train: new layer only           │
│  - LR: 5e-5                        │
│  - Freeze: other decoder layers    │
└─────────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│ Phase 2: Main (epochs 4-15)        │
│  - Train: entire decoder           │
│  - LR: 1e-5                        │
│  - Unfreeze: all decoder params    │
└─────────────────────────────────────┘
       ↓
48kHz output
```

### Option 2: Fast (Pre-computed)
```
Step 1 (one-time):
  24kHz Audio → SIDON → 48kHz Audio → Save to disk
  24kHz Audio → Encoder → VQ → Codes → Save to disk

Step 2 (every epoch):
  Load Codes → Decoder → 48kHz prediction
  Load 48kHz target → Compute loss → Backprop
```

---

## Config Parameters

Key parameters in `configs/phase11_decoder_48khz.json`:

```json
{
  "num_epochs": 15,              // Total training epochs
  "warmup_epochs": 3,             // Warmup phase duration (Option 1)
  "learning_rate": 1e-5,          // Main training LR
  "warmup_learning_rate": 5e-5,   // Warmup phase LR
  "batch_size": 32,               // Adjust based on GPU memory
  "segment_length": 4.0,           // Audio segment length (seconds)
  "l1_weight": 1.0,
  "stft_weight": 1.0,
  "n_ffts": [1024, 2048, 4096, 8192]
}
```

---

## Expected Training Time

| Phase | Epochs | Time (H200) | Time (V100/A100) |
|-------|--------|-------------|-------------------|
| Warmup | 3 | ~3-4 hours | ~8-10 hours |
| Main | 12 | ~12-15 hours | ~30-40 hours |
| **Total** | **15** | **~15-19 hours** | **~38-50 hours** |

---

## Monitoring

```bash
# View logs
tail -f logs/phase11_decoder_48khz/training.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check if running
ps aux | grep finetune_decoder_48khz
```

---

## Files Created

All Phase 11 files:
- `finetune_decoder_48khz.py` - Standard training (random init)
- `finetune_decoder_48khz_warmup.py` - Smart init + warmup ⭐
- `finetune_decoder_48khz_fast.py` - Pre-computed codes training
- `precompute_codes.py` - Code pre-computation utility
- `train_decoder_48khz.sh` - Standard launcher
- `train_decoder_48khz_warmup.sh` - Warmup launcher ⭐
- `train_decoder_48khz_workflow.sh` - Fast workflow launcher
- `configs/phase11_decoder_48khz.json` - Training config
- `PHASE11_README.md` - Full documentation

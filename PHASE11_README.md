# Phase 11: SNAC Decoder 48kHz Output

Train SNAC decoder to output **48kHz audio** while keeping encoder and VQ frozen (pretrained).

## 🚀 Fast Training (Recommended)

Since encoder and VQ are frozen, we can **pre-compute quantized codes** once and reuse them:

```bash
# One-step workflow (pre-compute + train)
./train_decoder_48khz_workflow.sh 0

# Or manual two-step process:
# Step 1: Pre-compute codes (~1-2 hours)
uv run python precompute_codes.py \
    --pretrained_model hubertsiuzdak/snac_24khz \
    --data_dir /mnt/data/combine/train/audio \
    --output_dir /mnt/data/codes_phase11/train \
    --segment_length 4.0 \
    --batch_size 32 \
    --device 0

# Step 2: Train with pre-computed codes (faster!)
uv run python finetune_decoder_48khz_fast.py \
    --config configs/phase11_decoder_48khz.json \
    --codes_dir /mnt/data/codes_phase11 \
    --device 0
```

**Benefits of Fast Training:**
- ⚡ **2-3x faster** - No encoder/VQ forward pass
- 💾 **Less memory** - Don't load encoder/VQ (~4M params saved)
- 📈 **Higher batch size** - More GPU memory available
- 💿 **Smaller checkpoints** - Save decoder only

## Overview

**Goal**: Modify SNAC decoder to output 48kHz instead of 24kHz without retraining encoder/VQ.

**Architecture**:
```
Input (24kHz) → Encoder (frozen) → VQ (frozen) → Decoder (trainable) → Output (48kHz)
                  ↓                        ↓
              Pretrained              Pretrained
              SNAC 24kHz              SNAC 24kHz
```

**Training Target**: SIDON upsampler (sarulab-speech/sidon-v0.1) generates 48kHz reference audio.

## Key Changes

### Decoder Architecture

**Original (24kHz output)**:
```
Decoder rates: [8, 8, 4, 2] = 512x upsampling
Input: 24kHz → Latent → Output: 24kHz
```

**New (48kHz output)**:
```
Decoder rates: [8, 8, 4, 2, 2] = 1024x upsampling
Input: 24kHz → Latent → Output: 48kHz
```

Added extra `2x` upsampling layer for 2x output sample rate.

## Training Strategy

1. **Freeze**: Encoder and VQ (from pretrained SNAC 24kHz)
2. **Replace**: Decoder with new 48kHz decoder
3. **Train**: Only decoder parameters (~15M params)
4. **Target**: SIDON upsampler's 48kHz output

## Usage

### Start Training

```bash
# Run on GPU 0
./train_decoder_48khz.sh 0

# Run on GPU 3
./train_decoder_48khz.sh 3
```

### Manual Training

```bash
uv run python finetune_decoder_48khz.py \
    --config configs/phase11_decoder_48khz.json \
    --device 0
```

### Resume from Checkpoint

```bash
uv run python finetune_decoder_48khz.py \
    --config configs/phase11_decoder_48khz.json \
    --device 0 \
    --resume checkpoints/phase11_decoder_48khz/checkpoint_epoch5.pt
```

## Monitoring

```bash
# View logs
tail -f logs/phase11_decoder_48khz/training.log

# Or background log
tail -f /tmp/phase11_decoder_48khz_gpu0.log

# Check GPU
watch -n 1 nvidia-smi

# Check if running
ps aux | grep finetune_decoder_48khz
```

## Configuration

Edit `configs/phase11_decoder_48khz.json`:

```json
{
  "pretrained_model": "hubertsiuzdak/snac_24khz",
  "train_data": "/mnt/data/combine/train/audio",
  "val_data": "/mnt/data/combine/valid/audio",
  "batch_size": 32,
  "num_epochs": 15,
  "learning_rate": 10e-6,
  "segment_length": 4.0,
  "l1_weight": 1.0,
  "stft_weight": 1.0,
  "n_ffts": [1024, 2048, 4096, 8192]
}
```

## Inference with 48kHz Decoder

After training, use the 48kHz decoder:

```python
import torch
from snac import SNAC

# Load checkpoint
checkpoint = torch.load("checkpoints/phase11_decoder_48khz/best_model.pt")

# Create model with 48kHz decoder
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
model.load_state_dict(checkpoint['model'])
model.eval()

# Encode (24kHz input)
audio_24k = torch.randn(1, 1, 24000)
codes = model.encode(audio_24k)

# Decode (48kHz output)
audio_48k = model.decode(codes)  # Outputs 48kHz!
```

## Training Flow

```
1. Load 24kHz audio
2. Forward through frozen encoder → latent codes
3. Forward through frozen VQ → quantized codes
4. Forward through trainable decoder → 48kHz prediction
5. Generate 48kHz target using SIDON upsampler
6. Compute loss (L1 + Multi-scale STFT)
7. Backpropagate to decoder only
8. Repeat
```

## Dependencies

Install SIDON upsampler dependencies:

```bash
pip install torchaudio transformers huggingface_hub
```

## Expected Results

- **Input**: 24kHz audio
- **Output**: 48kHz audio (2x upsampling)
- **Quality**: Should match SIDON upsampler quality after training
- **Training time**: ~15-20 epochs (~18-24 hours on single GPU)

## Troubleshooting

### SIDON download fails
```bash
# Pre-download SIDON models
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('sarulab-speech/sidon-v0.1', filename='feature_extractor_cuda.pt')
hf_hub_download('sarulab-speech/sidon-v0.1', filename='decoder_cuda.pt')
"
```

### CUDA out of memory
Reduce batch size in config:
```json
{
  "batch_size": 16,  // was 32
  "eval_batch_size": 24  // was 48
}
```

### Slow SIDON processing
SIDON processes in chunks of 60 seconds. For faster training, consider:
1. Pre-compute SIDON targets offline
2. Use shorter segment_length (e.g., 2.0 instead of 4.0)

## Files

- `finetune_decoder_48khz.py` - Main training script
- `configs/phase11_decoder_48khz.json` - Training configuration
- `train_decoder_48khz.sh` - Training launcher script
- `checkpoints/phase11_decoder_48khz/` - Trained models

# SNAC Architecture: Detailed Data Flow

This document provides a comprehensive explanation of the SNAC (Multi-Scale Neural Audio Codec) architecture, showing the complete data transformation from input audio to output audio.

## Overview

SNAC is a neural audio codec that compresses audio into discrete codes at low bitrates (~0.98 kbps for speech). The key innovation is **hierarchical multi-scale tokenization**, where coarse tokens are sampled less frequently and cover broader time spans.

```
Audio → Encoder → Latent → Vector Quantization → Codes → Decoder → Audio
```

## Model Parameters (Pretrained `hubertsiuzdak/snac_24khz`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `sampling_rate` | 24000 Hz | Audio sample rate |
| `encoder_dim` | 64 | Initial encoder dimension |
| `encoder_rates` | [3, 3, 7, 7] | Downsampling strides |
| `latent_dim` | 1024 | Latent representation dimension |
| `decoder_dim` | 1536 | Initial decoder dimension |
| `decoder_rates` | [7, 7, 3, 3] | Upsampling strides |
| `attn_window_size` | 32 | Local attention window size |
| `codebook_size` | 4096 | Number of codebook entries |
| `codebook_dim` | 8 | Quantized code dimension |
| `vq_strides` | [8, 4, 2, 1] | Hierarchical quantization strides |

---

## Complete Data Flow: Audio to Audio

### Input Stage

```
Input: audio (B, 1, T)
  B = batch size
  1 = mono channel
  T = time steps (samples at 24kHz)
```

**Example:** For a 10-second audio clip at 24kHz:
- Shape: `(1, 1, 240000)` (batch=1, mono, 10s × 24kHz = 240k samples)

---

## Stage 1: Preprocessing

### Padding for Alignment

The audio is padded to ensure proper alignment for both the encoder hop length and attention window.

```python
# Compute total downsampling (hop length)
hop_length = prod(encoder_rates) = 3 × 3 × 7 × 7 = 441

# Compute least common multiple for alignment
lcm = lcm(vq_strides[0], attn_window_size) = lcm(8, 32) = 32

# Pad to multiple of hop_length × lcm
pad_to = hop_length × lcm = 441 × 32 = 14112

# Right padding
padded_length = ceil(T / pad_to) × pad_to
```

**Example:**
- Input: `(1, 1, 240000)` (240k samples)
- Pad to: 14112 → 240000 already aligned, no padding needed
- Output: `(1, 1, 240000)`

**After Stage 1:**
- Shape: `(B, 1, T_padded)`
- Data: Padded audio waveform

---

## Stage 2: Encoder (Multi-Scale Downsampling)

The encoder progressively downsamples the audio while increasing the channel dimension, extracting features at multiple temporal resolutions.

### Architecture Overview

```
Audio → Conv1d → [EncoderBlock × 4] → LocalMHA → Conv1d → Latent
```

### Step 2.1: Initial Convolution

```python
WNConv1d(in_channels=1, out_channels=64, kernel_size=7, padding=3)
```

- **Input:** `(B, 1, T_padded)`
- **Operation:** 1D convolution with weight normalization
- **Output:** `(B, 64, T_padded)`

**Example:**
- Input: `(1, 1, 240000)`
- Output: `(1, 64, 240000)`

---

### Step 2.2: Encoder Block 1 (Stride 3)

Each `EncoderBlock` consists of:
- 3 ResidualUnits (dilations: 1, 3, 9)
- Snake1d activation
- Strided convolution (downsampling)

```python
ResidualUnit(dim=64, dilation=1, groups=1)
ResidualUnit(dim=64, dilation=3, groups=1)
ResidualUnit(dim=64, dilation=9, groups=1)
Snake1d(64)
WNConv1d(64, 128, kernel_size=2*stride, stride=3)  # Downsampling!
```

**Dimensions:**
- **Input:** `(B, 64, T)`
- **After residual units:** `(B, 64, T)` (same temporal length)
- **After stride-3 conv:** `(B, 128, T/3)` (downsampled 3×, channels doubled)

**Example:**
- Input: `(1, 64, 240000)`
- Output: `(1, 128, 80000)` (240000 / 3)

---

### Step 2.3: Encoder Block 2 (Stride 3)

```python
ResidualUnit(dim=128, dilation=1, groups=1)
ResidualUnit(dim=128, dilation=3, groups=1)
ResidualUnit(dim=128, dilation=9, groups=1)
Snake1d(128)
WNConv1d(128, 256, kernel_size=6, stride=3)  # Downsampling!
```

**Dimensions:**
- **Input:** `(B, 128, T/3)`
- **Output:** `(B, 256, T/9)` (downsampled 9× total)

**Example:**
- Input: `(1, 128, 80000)`
- Output: `(1, 256, 26667)` (80000 / 3)

---

### Step 2.4: Encoder Block 3 (Stride 7)

```python
ResidualUnit(dim=256, dilation=1, groups=1)
ResidualUnit(dim=256, dilation=3, groups=1)
ResidualUnit(dim=256, dilation=9, groups=1)
Snake1d(256)
WNConv1d(256, 512, kernel_size=14, stride=7)  # Downsampling!
```

**Dimensions:**
- **Input:** `(B, 256, T/9)`
- **Output:** `(B, 512, T/63)` (downsampled 63× total)

**Example:**
- Input: `(1, 256, 26667)`
- Output: `(1, 512, 3810)` (26667 / 7 ≈ 3810)

---

### Step 2.5: Encoder Block 4 (Stride 7)

```python
ResidualUnit(dim=512, dilation=1, groups=1)
ResidualUnit(dim=512, dilation=3, groups=1)
ResidualUnit(dim=512, dilation=9, groups=1)
Snake1d(512)
WNConv1d(512, 1024, kernel_size=14, stride=7)  # Downsampling!
```

**Dimensions:**
- **Input:** `(B, 512, T/63)`
- **Output:** `(B, 1024, T/441)` (downsampled 441× total)

**Example:**
- Input: `(1, 512, 3810)`
- Output: `(1, 1024, 544)` (3810 / 7 ≈ 544)

---

### Step 2.6: Local Multi-Head Attention (LocalMHA)

After downsampling, a local attention mechanism captures long-range dependencies within windows.

```python
LocalMHA(dim=1024, window_size=32, dim_head=64)
```

**Operation:**
1. **Layer normalization:** `(B, 1024, T/441)` → `(B, T/441, 1024)`
2. **Split into windows:** `(B, T/441, 1024)` → `(B, num_windows, 32, 1024)`
3. **QKV projection:** `Linear(1024 → 3072)` → split into Q, K, V
   - Each: `(B, num_windows, 32, 16_heads, 64)` (1024 / 64 = 16 heads)
4. **Rotary position encoding:** Applied to Q and K
5. **Scaled dot-product attention:** Within each window
6. **Output projection:** `Linear(1024 → 1024)`
7. **Residual connection:** Add input back

**Dimensions:**
- **Input:** `(B, 1024, T/441)`
- **Output:** `(B, 1024, T/441)` (same shape)

**Example:**
- Input: `(1, 1024, 544)`
- Number of windows: 544 / 32 = 17 windows
- Output: `(1, 1024, 544)`

---

### Step 2.7: Final Convolution

```python
WNConv1d(1024, 1024, kernel_size=7, padding=3, groups=1)
```

**Dimensions:**
- **Input:** `(B, 1024, T/441)`
- **Output:** `(B, 1024, T/441)`

**Example:**
- Input: `(1, 1024, 544)`
- Output: `(1, 1024, 544)`

---

**After Stage 2 (Encoder):**
- **Shape:** `(B, 1024, T/441)`
- **Data:** Compressed latent representation
- **Compression ratio:** 441× in time, 1024× in channels (net: 441/1024 ≈ 0.43× original size)

**Example:**
- Input audio: `(1, 1, 240000)` = 240k samples
- Latent: `(1, 1024, 544)` = 557k values (but at lower temporal resolution)

---

## Stage 3: Hierarchical Vector Quantization

The latent is quantized using **Residual Vector Quantization (RVQ)** with hierarchical strides. This is the key innovation of SNAC.

### Architecture Overview

```
Latent → Quantizer 1 (stride 8) → Residual 1
       → Quantizer 2 (stride 4) → Residual 2
       → Quantizer 3 (stride 2) → Residual 3
       → Quantizer 4 (stride 1) → Final quantized latent
```

### Quantization Process (for each codebook)

Each `VectorQuantize` layer performs:

1. **Optional striding:** Downsample if stride > 1
2. **Projection to codebook dimension:** `Conv1d(latent_dim → 8)`
3. **Find nearest codebook entry:** L2-normalized, then compute distance
4. **Straight-through estimator:** Use quantized value in forward, gradients in backward

```python
# For each quantizer i:
z_residual = z - sum(z_q_0, ..., z_q_{i-1})

# Optional striding
if stride > 1:
    z_pooled = avg_pool1d(z_residual, stride)
else:
    z_pooled = z_residual

# Project to codebook dimension
z_e = Conv1d(1024 → 8)(z_pooled)  # (B, 8, T_strided)

# L2 normalize
z_e_norm = F.normalize(z_e, dim=1)
codebook_norm = F.normalize(codebook, dim=1)

# Find nearest codebook entry
dist = ||z_e_norm||² - 2(z_e_norm @ codebook_norm.T) + ||codebook_norm||²
indices = argmax(-dist)  # (B, T_strided)

# Decode to quantized latent
z_q = codebook[indices]  # (B, 8, T_strided)

# Straight-through estimator
z_q_st = z_e + (z_q - z_e).detach()

# Project back to latent dimension
z_q_final = Conv1d(8 → 1024)(z_q_st)  # (B, 1024, T_strided)

# Upsample if strided
if stride > 1:
    z_q_final = repeat_interleave(z_q_final, stride, dim=-1)  # (B, 1024, T)
```

---

### Codebook 1: Coarsest Scale (Stride 8)

**Purpose:** Capture broad spectral/temporal patterns at lowest temporal resolution

```python
VectorQuantize(input_dim=1024, codebook_size=4096, codebook_dim=8, stride=8)
```

**Step 3.1.1: Strided Average Pooling**
- **Input:** `(B, 1024, T/441)`
- **Operation:** `avg_pool1d(kernel_size=8, stride=8)`
- **Output:** `(B, 1024, T/3528)` (downsampled 8× more)

**Example:**
- Input: `(1, 1024, 544)`
- Output: `(1, 1024, 68)` (544 / 8)

**Step 3.1.2: Project to Codebook Dimension**
- **Input:** `(B, 1024, T/3528)`
- **Operation:** `Conv1d(1024 → 8, kernel_size=1)`
- **Output:** `(B, 8, T/3528)`

**Example:**
- Input: `(1, 1024, 68)`
- Output: `(1, 8, 68)`

**Step 3.1.3: Quantize**
- **Input:** `(B, 8, T/3528)`
- **Codebook:** `(4096, 8)`
- **Operation:** Find nearest codebook entry for each timestep
- **Output indices:** `(B, T/3528)` with values in `[0, 4095]`
- **Output quantized:** `(B, 8, T/3528)`

**Example:**
- Input: `(1, 8, 68)`
- Indices: `(1, 68)` with values like `[42, 157, 2930, ..., 1024]`
- Quantized: `(1, 8, 68)` = codebook embeddings at those indices

**Step 3.1.4: Project Back & Upsample**
- **Input:** `(B, 8, T/3528)`
- **Operation:**
  - `Conv1d(8 → 1024, kernel_size=1)`
  - `repeat_interleave(stride=8)`
- **Output:** `(B, 1024, T/441)`

**Example:**
- Input: `(1, 8, 68)`
- Output: `(1, 1024, 544)` (68 × 8 = 544)

**Step 3.1.5: Accumulate**
```python
z_q = z_q + z_q_1  # Start accumulating
residual = z - z_q_1  # For next codebook
```

---

### Codebook 2: Medium-Coarse Scale (Stride 4)

**Purpose:** Capture mid-level details at medium temporal resolution

```python
VectorQuantize(input_dim=1024, codebook_size=4096, codebook_dim=8, stride=4)
```

**Step 3.2.1: Strided Average Pooling (on residual)**
- **Input:** `(B, 1024, T/441)`
- **Operation:** `avg_pool1d(kernel_size=4, stride=4)`
- **Output:** `(B, 1024, T/1764)` (downsampled 4×)

**Example:**
- Input: `(1, 1024, 544)`
- Output: `(1, 1024, 136)` (544 / 4)

**Step 3.2.2-3.2.5:** Same as Codebook 1 (project → quantize → project back → upsample)

**Example:**
- Input: `(1, 1024, 136)`
- After projection: `(1, 8, 136)`
- Indices: `(1, 136)` with values in `[0, 4095]`
- After upsample: `(1, 1024, 544)` (136 × 4 = 544)

**Step 3.2.6: Accumulate**
```python
z_q = z_q + z_q_2  # Accumulate
residual = residual - z_q_2  # Update residual
```

---

### Codebook 3: Medium-Fine Scale (Stride 2)

**Purpose:** Capture finer details at higher temporal resolution

```python
VectorQuantize(input_dim=1024, codebook_size=4096, codebook_dim=8, stride=2)
```

**Step 3.3.1: Strided Average Pooling**
- **Input:** `(B, 1024, T/441)`
- **Operation:** `avg_pool1d(kernel_size=2, stride=2)`
- **Output:** `(B, 1024, T/882)` (downsampled 2×)

**Example:**
- Input: `(1, 1024, 544)`
- Output: `(1, 1024, 272)` (544 / 2)

**Step 3.3.2-3.3.5:** Same as before

**Example:**
- Input: `(1, 1024, 272)`
- After projection: `(1, 8, 272)`
- Indices: `(1, 272)` with values in `[0, 4095]`
- After upsample: `(1, 1024, 544)` (272 × 2 = 544)

**Step 3.3.6: Accumulate**
```python
z_q = z_q + z_q_3
residual = residual - z_q_3
```

---

### Codebook 4: Finest Scale (Stride 1)

**Purpose:** Capture fine-grained details at full latent resolution

```python
VectorQuantize(input_dim=1024, codebook_size=4096, codebook_dim=8, stride=1)
```

**Step 3.4.1: No Striding**
- **Input:** `(B, 1024, T/441)`
- **Output:** `(B, 1024, T/441)` (no change)

**Example:**
- Input: `(1, 1024, 544)`
- Output: `(1, 1024, 544)`

**Step 3.4.2-3.4.5:** Same as before (no upsampling needed)

**Example:**
- Input: `(1, 1024, 544)`
- After projection: `(1, 8, 544)`
- Indices: `(1, 544)` with values in `[0, 4095]`
- After projection back: `(1, 1024, 544)`

**Step 3.4.6: Final Accumulation**
```python
z_q = z_q + z_q_4  # Final quantized latent
```

---

**After Stage 3 (Vector Quantization):**

**Output 1: Quantized Latent**
- **Shape:** `(B, 1024, T/441)`
- **Data:** Reconstructed latent from quantized codes

**Output 2: Hierarchical Codes**
- **Codebook 1 (coarsest):** `(B, T/3528)` = 68 tokens
- **Codebook 2:** `(B, T/1764)` = 136 tokens
- **Codebook 3:** `(B, T/882)` = 272 tokens
- **Codebook 4 (finest):** `(B, T/441)` = 544 tokens

**Example (10s audio):**
```
codes = [
    torch.Size([1, 68]),   # Coarse: 1 token every ~353 samples (14.7 ms)
    torch.Size([1, 136]),  # Medium: 1 token every ~176 samples (7.3 ms)
    torch.Size([1, 272]),  # Fine: 1 token every ~88 samples (3.7 ms)
    torch.Size([1, 544]),  # Finest: 1 token every ~44 samples (1.8 ms)
]
```

**Total tokens:** 68 + 136 + 272 + 544 = **1020 tokens** for 10 seconds

**Bitrate:**
- Each token: log₂(4096) = 12 bits
- Total bits: 1020 × 12 = 12,240 bits
- Bitrate: 12,240 bits / 10 seconds = **1,224 bps ≈ 1.2 kbps**

**Note:** The paper reports 0.98 kbps, likely due to entropy coding or different audio content.

---

## Stage 4: Decoder (Multi-Scale Upsampling)

The decoder reconstructs the audio from the quantized latent, progressively upsampling while decreasing the channel dimension.

### Architecture Overview

```
Quantized Latent → Conv1d → LocalMHA → [DecoderBlock × 4] → Snake1d → Conv1d → Tanh → Audio
```

### Step 4.1: Initial Convolution

```python
WNConv1d(in_channels=1024, out_channels=1536, kernel_size=7, padding=3)
```

- **Input:** `(B, 1024, T/441)`
- **Operation:** Expand channels for decoder
- **Output:** `(B, 1536, T/441)`

**Example:**
- Input: `(1, 1024, 544)`
- Output: `(1, 1536, 544)`

---

### Step 4.2: Local Multi-Head Attention (LocalMHA)

```python
LocalMHA(dim=1536, window_size=32, dim_head=64)
```

**Operation:** Same as encoder attention, but with 1536 channels (24 heads)

**Dimensions:**
- **Input:** `(B, 1536, T/441)`
- **Output:** `(B, 1536, T/441)`

**Example:**
- Input: `(1, 1536, 544)`
- Output: `(1, 1536, 544)`

---

### Step 4.3: Decoder Block 1 (Stride 7)

Each `DecoderBlock` consists of:
- Snake1d activation
- Transposed convolution (upsampling)
- Optional noise injection
- 2 ResidualUnits

```python
Snake1d(1536)
WNConvTranspose1d(1536, 768, kernel_size=14, stride=7)  # Upsampling!
NoiseBlock(768)  # Optional
ResidualUnit(768, dilation=1)
ResidualUnit(768, dilation=3)
```

**Dimensions:**
- **Input:** `(B, 1536, T/441)`
- **After transposed conv:** `(B, 768, T/63)` (upsampled 7×, channels halved)
- **After residual units:** `(B, 768, T/63)`

**Example:**
- Input: `(1, 1536, 544)`
- Output: `(1, 768, 3810)` (544 × 7 ≈ 3810)

---

### Step 4.4: Decoder Block 2 (Stride 7)

```python
Snake1d(768)
WNConvTranspose1d(768, 384, kernel_size=14, stride=7)  # Upsampling!
NoiseBlock(384)
ResidualUnit(384, dilation=1)
ResidualUnit(384, dilation=3)
```

**Dimensions:**
- **Input:** `(B, 768, T/63)`
- **Output:** `(B, 384, T/9)` (upsampled 49× total)

**Example:**
- Input: `(1, 768, 3810)`
- Output: `(1, 384, 26670)` (3810 × 7 ≈ 26670)

---

### Step 4.5: Decoder Block 3 (Stride 3)

```python
Snake1d(384)
WNConvTranspose1d(384, 192, kernel_size=6, stride=3)  # Upsampling!
NoiseBlock(192)
ResidualUnit(192, dilation=1)
ResidualUnit(192, dilation=3)
```

**Dimensions:**
- **Input:** `(B, 384, T/9)`
- **Output:** `(B, 192, T/3)` (upsampled 147× total)

**Example:**
- Input: `(1, 384, 26670)`
- Output: `(1, 192, 80010)` (26670 × 3 ≈ 80010)

---

### Step 4.6: Decoder Block 4 (Stride 3)

```python
Snake1d(192)
WNConvTranspose1d(192, 96, kernel_size=6, stride=3)  # Upsampling!
NoiseBlock(96)
ResidualUnit(96, dilation=1)
ResidualUnit(96, dilation=3)
```

**Dimensions:**
- **Input:** `(B, 192, T/3)`
- **Output:** `(B, 96, T)` (upsampled 441× total = original resolution!)

**Example:**
- Input: `(1, 192, 80010)`
- Output: `(1, 96, 240030)` (80010 × 3 ≈ 240030)

---

### Step 4.7: Final Activation & Projection

```python
Snake1d(96)
WNConv1d(96, 1, kernel_size=7, padding=3)
nn.Tanh()
```

**Dimensions:**
- **Input:** `(B, 96, T)`
- **After Snake1d:** `(B, 96, T)`
- **After Conv1d:** `(B, 1, T)`
- **After Tanh:** `(B, 1, T)` (values in [-1, 1])

**Example:**
- Input: `(1, 96, 240030)`
- Output: `(1, 1, 240030)`

---

### Step 4.8: Trim to Original Length

The decoder output is trimmed to match the original input length (before padding).

```python
audio_hat = audio_hat[..., :original_length]
```

**Example:**
- Decoder output: `(1, 1, 240030)` (30 samples extra from rounding)
- Trimmed: `(1, 1, 240000)` (matches input)

---

**After Stage 4 (Decoder):**
- **Shape:** `(B, 1, T_original)`
- **Data:** Reconstructed audio waveform in range [-1, 1]

**Example:**
- Input audio: `(1, 1, 240000)` (10 seconds)
- Output audio: `(1, 1, 240000)` (10 seconds reconstructed)

---

## Summary: Complete Data Flow

```
Input:  (B, 1, 240000)  # 10s audio at 24kHz
  ↓
[Preprocessing: Pad to alignment]
  ↓ (B, 1, 240000)
  ↓
[Encoder]
  ├─ Conv1d(1 → 64, k=7)               → (B, 64, 240000)
  ├─ EncoderBlock(stride=3)             → (B, 128, 80000)
  ├─ EncoderBlock(stride=3)             → (B, 256, 26667)
  ├─ EncoderBlock(stride=7)             → (B, 512, 3810)
  ├─ EncoderBlock(stride=7)             → (B, 1024, 544)
  ├─ LocalMHA(window=32)                → (B, 1024, 544)
  └─ Conv1d(1024 → 1024, k=7)           → (B, 1024, 544)
  ↓
[Hierarchical Vector Quantization]
  ├─ Codebook 1 (stride=8)  → indices[0] = (B, 68),   z_q_1
  ├─ Codebook 2 (stride=4)  → indices[1] = (B, 136),  z_q_2
  ├─ Codebook 3 (stride=2)  → indices[2] = (B, 272),  z_q_3
  └─ Codebook 4 (stride=1)  → indices[3] = (B, 544),  z_q_4
  ↓ (B, 1024, 544)  # Quantized latent
  ↓
[Decoder]
  ├─ Conv1d(1024 → 1536, k=7)           → (B, 1536, 544)
  ├─ LocalMHA(window=32)                → (B, 1536, 544)
  ├─ DecoderBlock(stride=7)             → (B, 768, 3810)
  ├─ DecoderBlock(stride=7)             → (B, 384, 26670)
  ├─ DecoderBlock(stride=3)             → (B, 192, 80010)
  ├─ DecoderBlock(stride=3)             → (B, 96, 240030)
  ├─ Snake1d + Conv1d(96 → 1, k=7)      → (B, 1, 240030)
  └─ Tanh()                             → (B, 1, 240030)
  ↓
[Trim to original length]
  ↓
Output: (B, 1, 240000)  # 10s reconstructed audio
```

---

## Key Architectural Innovations

### 1. Hierarchical Multi-Scale Tokenization

Different codebooks operate at different temporal resolutions:
- **Coarse codebook** (stride 8): Captures broad patterns, 1 token per ~14.7 ms
- **Medium codebooks** (stride 4, 2): Capture mid-level details
- **Fine codebook** (stride 1): Captures fine details, 1 token per ~1.8 ms

This allows efficient compression where fine details are only encoded where needed.

### 2. Residual Vector Quantization (RVQ)

Each codebook quantizes the **residual** from previous codebooks:
```
z_q_0 = 0
z_q_1 = VQ_1(z)
z_q_2 = VQ_2(z - z_q_1)
z_q_3 = VQ_3(z - z_q_1 - z_q_2)
z_q_4 = VQ_4(z - z_q_1 - z_q_2 - z_q_3)
z_q_final = z_q_1 + z_q_2 + z_q_3 + z_q_4
```

This allows progressive refinement: each codebook adds details not captured by previous ones.

### 3. Local Multi-Head Attention

Captures long-range dependencies within local windows (32 timesteps):
- **Encoder:** At latent resolution (544 timesteps → 17 windows of 32)
- **Decoder:** Same windowing at each upsampling stage

Uses **rotary position encodings** for temporal awareness.

### 4. Snake Activation

`Snake1d(x) = x + (1/sin(α)) * sin(α * x)`

Learnable periodic activation that's effective for audio.

### 5. Weight Normalization

All convolutions use weight normalization (instead of batch norm) for stable training.

---

## Memory and Computation

### Parameter Count

- **Encoder:** ~10M parameters
- **Decoder:** ~9.8M parameters
- **Quantizer:** ~1M parameters (4 codebooks × 4096 × 8 dims)
- **Total:** ~20M parameters

### Memory Footprint (10s audio batch)

- **Input audio:** 240k samples × 2 bytes (float16) = 480 KB
- **Latent:** 544 timesteps × 1024 dims × 4 bytes = 2.2 MB
- **Codes:** 1020 tokens × 4 bytes (int32) = 4 KB
- **Model:** 20M parameters × 4 bytes = 80 MB

### Compression Ratio

- **Original:** 240k samples × 2 bytes = 480 KB
- **Compressed:** 1020 tokens × 2 bytes (log2 quantized) = 2 KB
- **Ratio:** 240:1 compression

---

## Usage Examples

### Encoding

```python
from snac import SNAC

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
audio = torch.randn(1, 1, 24000)  # 1 second at 24kHz

codes = model.encode(audio)
# codes[0]: (1, 7)    - Coarse tokens
# codes[1]: (1, 14)   - Medium tokens
# codes[2]: (1, 27)   - Fine tokens
# codes[3]: (1, 54)   - Finest tokens
```

### Decoding

```python
audio_hat = model.decode(codes)
# audio_hat: (1, 1, 24000) - Reconstructed audio
```

### Full Forward

```python
audio_hat, codes = model(audio)
# Returns both reconstructed audio and codes
```

---

## References

- Paper: [SNAC: Multi-Scale Neural Audio Codec](https://arxiv.org/abs/2402.10509) (if available)
- Pretrained model: `hubertsiuzdak/snac_24khz` on HuggingFace
- Inspired by: SoundStream, EnCodec, VQ-VAE

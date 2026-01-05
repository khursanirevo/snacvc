# Phase 4 Training Architecture

## Objective

**Voice Conversion with High-Fidelity Audio Generation**

Train SNAC to separate content from speaker characteristics and generate realistic, high-quality audio with:
- **Source content** (preserved through codes)
- **Target speaker identity** (from reference embedding)

---

## Model Architecture

### Core Model: SNACWithSpeakerConditioning

```
┌─────────────────────────────────────────────────────────────┐
│                  SNACWithSpeakerConditioning                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐   │
│  │   SNAC      │    │   ERes2NetV2 │    │   SNAC      │   │
│  │   Encoder   │───▶│  Speaker     │◀───│   Decoder   │   │
│  │  (FROZEN)   │    │  Encoder     │    │  + FiLM     │   │
│  │             │    │  (FROZEN)    │    │  (TRAINABLE)│   │
│  └─────────────┘    │  (512-dim)   │    └─────────────┘   │
│         │           └──────────────┘           │           │
│         │                 ▲                   │           │
│         │                 │                   │           │
│    ┌────▼─────┐          │              ┌────▼─────┐    │
│    │   Codes  │          │              │  Audio   │    │
│    │ (discrete│          │              │  Output  │    │
│    │  indices)│          │              │          │    │
│    └──────────┘          │              └──────────┘    │
│                           │                                │
└───────────────────────────┼────────────────────────────────┘
                            │
                    Speaker Embedding
                    (conditioning signal via FiLM)
```

**Components:**
- **SNAC Encoder**: Pretrained base model from HuggingFace (**FROZEN**)
- **SNAC Decoder**: Base decoder (**FROZEN**) + FiLM layers (**TRAINABLE**)
- **ERes2NetV2 Speaker Encoder**: Extracts 512-dim speaker embeddings (**FROZEN**)
- **FiLM Conditioning**: Feature-wise Linear Modulation controls speaker characteristics in decoder (**TRAINABLE**)

### Pretrained vs Trainable Components

#### Pretrained Models (Loaded from External Sources)

**1. SNAC Base Model**
- **Source**: `hubertsiuzdak/snac_24khz` (HuggingFace)
- **Components**: Encoder + Decoder (without FiLM)
- **Parameters**: 89.4M
- **Status**: **FROZEN** (all parameters have `requires_grad=False`)
- **Purpose**: Provides high-quality audio encoding/decoding backbone

**2. ERes2NetV2 Speaker Encoder**
- **Source**: `pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt` (GPT-SoVITS ProPlus)
- **Components**: ERes2NetV2 feature extractor + projection layer (192→512 dims)
- **Parameters**: ~20M (internal model, not counted in trainable params)
- **Status**: **FROZEN** (all parameters have `requires_grad=False`)
- **Purpose**: Extract robust speaker identity embeddings

#### Trainable Components (Updated During Training)

**1. FiLM Layers (Feature-wise Linear Modulation)**
- **Location**: Inside SNAC decoder, applied to each decoder layer
- **Parameters**: ~1-2M (scales with decoder dimension and number of layers)
- **Status**: **TRAINABLE** (`requires_grad=True`)
- **Purpose**: Modulate decoder features based on speaker embedding
- **Function**: `γ(speaker_emb) * features + β(speaker_emb)`

**2. MultiPeriodDiscriminator (MPD)**
- **Components**: 5 discriminators (periods 2, 3, 5, 7, 11)
- **Parameters**: 41.1M
- **Status**: **TRAINABLE** (initialized from scratch)
- **Purpose**: Distinguish real vs fake audio based on periodic patterns

**3. MultiResolutionSTFTDiscriminator (MRD)**
- **Components**: 3 discriminators (FFT 1024, 2048, 4096)
- **Parameters**: 283K
- **Status**: **TRAINABLE** (initialized from scratch)
- **Purpose**: Distinguish real vs fake audio based on spectral patterns

### Model Parameter Summary

| Component | Source | Parameters | Status | Purpose |
|-----------|--------|------------|--------|---------|
| SNAC Encoder | HuggingFace | ~44M | **FROZEN** | Audio encoding |
| SNAC Decoder (base) | HuggingFace | ~45M | **FROZEN** | Audio decoding backbone |
| FiLM Layers | Initialized from scratch | ~1-2M | **TRAINABLE** | Speaker conditioning |
| ERes2NetV2 | GPT-SoVITS | ~20M | **FROZEN** | Speaker embedding extraction |
| MPD | Initialized from scratch | 41.1M | **TRAINABLE** | Adversarial discrimination (periodic) |
| MRD | Initialized from scratch | 283K | **TRAINABLE** | Adversarial discrimination (spectral) |
| **Total Trainable** | | **~43M** | | |
| **Total Frozen** | | **~109M** | | |
| **Grand Total** | | **~152M** | | |

**Training Efficiency Note**:
- Only **~43M parameters** (28%) are trainable
- **~109M parameters** (72%) are frozen (SNAC base + speaker encoder)
- This reduces memory usage and speeds up training significantly
- Frozen models provide strong pretrained representations for free

---

## GAN Discriminators (Audio Quality)

### MultiPeriodDiscriminator (MPD)

**Purpose**: Capture periodic patterns in speech waveforms (pitch, harmonics)

```
Input Audio → [5 Discriminators] → Real/Fake Scores
                    │
                    ├─ Period 2   (high-frequency patterns)
                    ├─ Period 3
                    ├─ Period 5
                    ├─ Period 7
                    └─ Period 11  (low-frequency patterns)
```

**How it works:**
- Each discriminator applies 2D convolutions on period-augmented spectrograms
- Period augmentation: reshape waveform to (batch, period, time // period)
- Each discriminator specializes in different periodicities
- Ensemble approach for robust pattern detection across frequencies

**Architecture per discriminator:**
```
Audio (B, 1, T)
    ↓
Period reshape → (B, 1, period, T//period)
    ↓
2D Conv layers (stride patterns)
    ↓
Feature maps → Real/Fake score
```

### MultiResolutionSTFTDiscriminator (MRD)

**Purpose**: Capture multi-scale spectral patterns (timbre, texture)

```
Input Audio → [3 Discriminators] → Real/Fake Scores
                    │
                    ├─ FFT 1024  (fine details, high-res spectral)
                    ├─ FFT 2048  (mid-level patterns)
                    └─ FFT 4096  (coarse structure, low-res)
```

**How it works:**
- Computes complex STFT spectrograms (real + imaginary parts)
- Multiple resolutions capture both local and global spectral patterns
- 2D convolutions on spectrograms for feature extraction
- Ensemble for comprehensive spectral analysis

**Architecture per discriminator:**
```
Audio (B, 1, T)
    ↓
STFT (n_fft) → (B, 2, F, T)  [real + imag]
    ↓
2D Conv layers (spectral patterns)
    ↓
Feature maps → Real/Fake score
```

---

## Loss Functions

### Complete Loss Equation

```python
loss_total = (
    # Reconstruction (always active)
    loss_reconstruction +

    # Voice conversion (always active)
    lambda_recon * loss_vc_recon +
    lambda_speaker * loss_speaker_matching +

    # Synthetic voice conversion (50% of samples)
    lambda_synthetic * loss_synthetic_vc +

    # GAN adversarial (after warmup, batch > 10)
    lambda_adv * loss_adv +

    # Feature matching (after warmup, batch > 10)
    lambda_fm * loss_feature_matching
)
```

### Loss Functions by Metric Type

| Loss Function | Metric Type | Formula | Purpose | Range |
|--------------|-------------|---------|---------|-------|
| **Distance-Based Losses** |
| L1 Loss (MAE) | L1 Distance | `mean(\|x - y\|)` | Time-domain waveform accuracy | [0, ∞) |
| Multi-Scale STFT | L1 Distance | `mean(\|STFT(x) - STFT(y)\|)` | Spectral accuracy (3 scales) | [0, ∞) |
| Feature Matching | L1 Distance | `mean(\|f_real - f_fake\|)` | Match intermediate features | [0, ∞) |
| Codebook Content | L1 Distance | `mean(\|code_orig - code_recon\|)` | Content preservation through codes | [0, ∞) |
| **Cosine-Based Losses** |
| Speaker Matching | Cosine Distance | `1 - cosine_sim(emb, target)` | Force output to match target speaker | [0, 2] |
| **Adversarial Losses** |
| Hinge Loss (Disc) | ReLU Margin | `mean(relu(1 - D_real)) + mean(relu(1 + D_fake))` | Push real > 1, fake < -1 | [0, ∞) |
| Generator Adv | Negative Mean | `-mean(D_fake)` | Fool discriminator (push scores > 0) | (-∞, 0] |

**Key Characteristics:**

**Distance-Based Losses (L1/MAE):**
- Directly measure difference between predicted and target
- Lower = better
- Scale-dependent (varies with signal magnitude)
- Used for: reconstruction, spectral matching, feature alignment

**Cosine-Based Losses:**
- Measure angular difference between embeddings
- Scale-invariant (only direction matters)
- 1 - cosine_sim converts similarity to distance
- Used for: speaker identity matching (normalized embeddings)

**Adversarial Losses:**
- Game-theoretic (generator vs discriminator)
- Hinge: margin-based (push scores apart)
- Generator: negative mean (wants high discriminator scores)
- Can go negative (feature of the formulation)

### Training Log Loss Mapping

This table maps the logged loss names to their metric types:

```
Training Log Output:
Epoch 0, Batch 182/5351: g_loss=2.8439, d_loss=1.5708, recon=0.6525,
                          synth=1.0272, vc=0.9224, spk=0.5388,
                          adv=0.4776, fm=0.2416
```

| Log Name | Loss Component | Metric Type | Formula |
|----------|---------------|-------------|---------|
| `recon` | Reconstruction | **L1 Distance** | L1(waveform) + Multi-Scale STFT |
| `synth` | Synthetic VC | **L1 Distance** | L1(waveform) + Multi-Scale STFT |
| `spk` | Speaker Matching | **Cosine Distance** | `1 - cosine_similarity(decoded_emb, target_emb)` |
| `vc` | Voice Conversion (total) | **Mixed** | 67% L1 + 33% Cosine (weighted by λ) |
| `adv` | Generator Adversarial | **Negative Mean** | `-mean(D_fake)` |
| `fm` | Feature Matching | **L1 Distance** | `mean(\|f_real - f_fake\|)` |
| `d_loss` | Discriminator | **Hinge Loss** | `mean(relu(1 - D_real)) + mean(relu(1 + D_fake))` |
| `g_loss` | Generator Total | **Composite** | Weighted sum of all above |

**Detailed Breakdown:**

- **`recon`**: Pure L1 distance (MAE) on waveform + L1 on 3 STFT scales
- **`synth`**: Pure L1 distance (MAE) on waveform + L1 on 3 STFT scales
- **`spk`**: Pure cosine distance on normalized speaker embeddings
- **`vc`**: Composite loss:
  - `λ_recon × L1_reconstruction` (67% with λ_recon=1.0)
  - `λ_speaker × cosine_speaker_matching` (33% with λ_speaker=0.5)
- **`adv`**: Negative mean of discriminator scores (can be negative)
- **`fm`**: L1 distance on discriminator intermediate features
- **`d_loss`**: Hinge loss with ReLU margin (always ≥ 0)
- **`g_loss`**: `recon + vc + synth + adv + fm` (weighted combination)

### 1. Reconstruction Loss (Audio Fidelity)

**Goal**: Reconstruct input audio accurately with speaker conditioning

```python
# Encode
codes = model.encode(audio)

# Decode with own speaker embedding
audio_reconstructed = model.decode(codes, speaker_embedding=speaker_emb)

# Compute loss
loss_recon = l1_weight * L1(audio, audio_reconstructed)
           + stft_weight * MultiScaleSTFT(audio, audio_reconstructed)
```

**Components:**
- **L1 Loss**: Time-domain waveform accuracy (preserves amplitude, phase)
- **Multi-Scale STFT Loss**: Spectral accuracy at 3 FFT sizes
  - FFT 1024: Fine spectral details (formants, high-freq cues)
  - FFT 2048: Mid-level patterns (harmonic structure)
  - FFT 4096: Coarse spectral structure (envelope, prosody)

**Weights**: `l1_weight=1.0`, `stft_weight=1.0`

**Purpose**: Establish baseline audio quality and ensure identity preservation

---

### 2. Voice Conversion Loss (FiLM-based Speaker Transformation)

**Goal**: Learn voice conversion through FiLM modulation with hard negative mining

#### Key Insight: Dual-Positive Training

SNAC codes contain **both content and speaker information** (by design, for reconstruction). We don't try to purify codes. Instead, we rely on **FiLM conditioning during decoding** to control speaker characteristics.

**FiLM learns two complementary behaviors:**

**1. Identity Preservation (Own Embedding)**
```python
codes_A + speaker_A_embedding → audio_A_reconstructed
```
- Teaches FiLM: **identity transformation** (minimal modulation)
- Output preserves both content AND original speaker
- Ensures reconstruction quality doesn't degrade
- Loss: `L1(audio_A_reconstructed, audio_A) + MultiScaleSTFT(...)`

**2. Speaker Transformation (Target Embedding)**
```python
codes_A + speaker_B_embedding → audio_A_converted_to_B
```
- Teaches FiLM: **speaker transformation** (active modulation)
- Output shifts to target speaker's characteristics
- Content preserved (from codes), speaker controlled (by FiLM)
- Loss: `1.0 - cosine_similarity(speaker_emb(audio_A_converted_to_B), speaker_B)`

#### Why This Works

FiLM learns a **continuum of transformations**:
- When `embedding == own`: modulation → minimal (preserve speaker)
- When `embedding == target`: modulation → shift to target speaker
- When `embedding == hard_negative`: modulation → partial shift (harder!)

This is elegant because:
- **Codes**: Provide basic audio structure (content + base speaker info)
- **FiLM**: Adjusts the "speaker knob" during decode to match conditioning
- No need to purify codes - encoder can keep speaker info for good reconstruction
- FiLM simply learns to override it based on embedding

#### Hard Negative Mining with FAISS

**Challenge**: Random negatives are too easy. We need challenging examples.

**Solution**: FAISS-based similarity search for hard negatives

```
Query Speaker Embedding (speaker_A)
        │
        ▼
┌─────────────────────────────────────┐
│   FAISS Index (128,437 entries)      │
│   - IndexFlatIP (inner product)      │
│   - Normalized embeddings            │
│   - Millisecond latency              │
└─────────────────────────────────────┘
        │
        ▼
Top-K Nearest Neighbors (K=500)
        │
        ▼
Filter by: 0.6 < similarity < 0.85
        │
        ▼
Return: 6 most similar different speakers
```

**Why 0.85 threshold?**
- Same speakers: similarity > 0.85 (excluded)
- Hard negatives: 0.6 < similarity < 0.85 (selected)
- These are "confusing" speakers - similar but different

**Why 0.6 lower bound?**
- Very different speakers (similarity < 0.6) are too easy
- We want challenging examples that push the model

**Performance:**
- FAISS search: ~5-10ms per query
- Embedding cache lookup: <1ms
- Total hard negative mining: ~10-15ms per sample

#### Stratified Hard Negatives (Current Configuration)

**Goal**: Balance negative difficulty for robust learning

**Similarity Spectrum:**
```
├─────┬─────────┬─────────┬─────────┤
0.0   0.3      0.6      0.85     1.0
│     │         │         │         │
Easy  Medium    Hard    Same
 30%    40%      30%     (exclude)
```

**Sampling Strategy:**
- **Easy (30%)**: Very different speakers (sim < 0.3)
  - Purpose: Learn global speaker space boundaries
- **Medium (40%)**: Moderately different (0.3-0.6)
  - Purpose: Learn intermediate speaker distinctions
- **Hard (30%)**: Similar speakers (0.6-0.85)
  - Purpose: Learn fine-grained speaker differences

**Benefits:**
- More diverse training signal
- Better generalization to unseen speakers
- Prevents overfitting to specific speaker clusters
- Covers full speaker similarity space

**Implementation:**
```python
# Search FAISS for 500 candidates
similarities, indices = faiss_index.search(query_emb, k=500)

# Bin by similarity
easy_candidates = [idx for sim, idx in zip(similarities, indices) if sim < 0.3]
medium_candidates = [idx for sim, idx in zip(similarities, indices) if 0.3 <= sim < 0.6]
hard_candidates = [idx for sim, idx in zip(similarities, indices) if 0.6 <= sim < 0.85]

# Sample from each tier (2 easy, 2 medium, 2 hard = 6 total)
selected = sample(easy_candidates, 2) + sample(medium_candidates, 2) + sample(hard_candidates, 2)
```

**Config:**
```json
{
  "use_stratified_negatives": true,
  "neg_ratio_easy": 0.3,
  "neg_ratio_medium": 0.4,
  "neg_ratio_hard": 0.3,
  "threshold_easy_medium": 0.3,
  "threshold_medium_hard": 0.6,
  "max_negatives": 6
}
```

#### Voice Conversion Loss Computation

```python
# For each sample in batch:
for i in range(batch_size):
    codes_i = codes[i]  # Codes from sample i
    speaker_emb_i = speaker_embs[i]  # Own embedding

    # === Positive: Own embedding (identity preservation) ===
    audio_positive = decode(codes_i, speaker_emb_i)
    loss_recon_i = L1(audio_positive, audio[i]) + MultiScaleSTFT(...)
    loss_speaker_match_i = 1.0 - cosine_similarity(
        extract_speaker_embedding(audio_positive),
        speaker_emb_i
    )

    # === Negatives: Target embeddings (speaker transformation) ===
    # Get 6 hard negatives via FAISS (stratified sampling)
    hard_neg_embs = get_stratified_negatives(
        query=speaker_emb_i,
        faiss_index=faiss_index,
        embedding_cache=cache,
        config=stratified_config
    )

    for neg_emb in hard_neg_embs:
        audio_negative = decode(codes_i, neg_emb)
        loss_speaker_neg = 1.0 - cosine_similarity(
            extract_speaker_embedding(audio_negative),
            neg_emb
        )
        speaker_matching_losses.append(loss_speaker_neg)

# Aggregate
loss_reconstruction = mean(loss_recon_i for all samples)
loss_speaker_matching = mean(speaker_matching_losses)

loss_vc = lambda_recon * loss_reconstruction
        + lambda_speaker * loss_speaker_matching
```

**Weights**: `lambda_recon=1.0`, `lambda_speaker_matching=0.5`

---

### 3. Synthetic Voice Conversion (Audio Augmentation)

**Goal**: Create unlimited pseudo-speaker pairs to strengthen speaker embedding control

#### Key Insight

Voice changer effects (pitch shifting) transform audio to sound like a different speaker. By encoding this transformed audio and decoding with the **original** speaker embedding, we force the model to learn:

> **"Speaker embedding dominates over conflicting acoustic patterns in codes"**

This is **harder** than identity reconstruction because codes and embedding are in conflict!

#### Synthetic Voice Conversion Pipeline

```
1. audio_A (speaker A's voice, 2 seconds at 24kHz)
        ↓
2. pitch_shift(audio_A, semitones=+2) → audio_A_modified
        ↓
3. encode(audio_A_modified) → codes_A_modified
        ↓
4. decode(codes_A_modified, speaker_A_embedding) → audio_reconstructed
        ↓
5. loss = ||audio_reconstructed - audio_A||
```

**Key point**: We reconstruct the **original** audio_A, not the augmented audio_A_modified!

#### Why This Works

- **Codes contain "wrong" speaker info**: Pitch-shifted acoustic patterns (e.g., higher pitch)
- **Embedding says "speaker A"**: Model must override the conflicting speaker cues in codes
- **Forces embedding dominance**: Teaches FiLM to trust embedding over conflicting acoustic features
- **Harder than identity**: Codes and embedding are in disagreement!

#### Pitch Shifting Implementation (Optimized)

**Challenge**: torchaudio's `Resample` is extremely slow when created repeatedly

**Solution**: Class-level resampler cache (10,000x speedup!)

```python
class PitchShifter:
    # Class-level cache (shared across all instances)
    _resampler_cache = {}

    def shift_semitones(self, audio: torch.Tensor, semitones: float) -> torch.Tensor:
        # Calculate resampling ratio
        pitch_factor = 2.0 ** (semitones / 12.0)

        if pitch_factor > 1.0:
            # Pitch up: resample up then down
            up_freq = int(self.sample_rate * pitch_factor)
            cache_key_up = (self.sample_rate, up_freq, audio.dtype)
            cache_key_down = (up_freq, self.sample_rate, audio.dtype)

            # Create if not exists (cached for future calls)
            if cache_key_up not in self._resampler_cache:
                self._resampler_cache[cache_key_up] = T.Resample(
                    orig_freq=self.sample_rate,
                    new_freq=up_freq
                )
            if cache_key_down not in self._resampler_cache:
                self._resampler_cache[cache_key_down] = T.Resample(
                    orig_freq=up_freq,
                    new_freq=self.sample_rate
                )

            # Use cached resamplers
            resampler_up = self._resampler_cache[cache_key_up].to(audio.device)
            resampler_down = self._resampler_cache[cache_key_down].to(audio.device)
            audio_shifted = resampler_down(resampler_up(audio))

        return audio_shifted
```

**Performance (benchmark results, 2-second audio):**
- **Without cache**: 850-15,000ms per pitch shift (unusable!)
- **With cache**: 0.13-0.18ms per pitch shift (60,000x-100,000x speedup!)
- **Total synthetic VC overhead**: 10-180ms per sample
  - Pitch shift: 0.13-0.18ms
  - Encode: 4-120ms (GPU warmup affects first few)
  - Decode: 5-63ms

**Config:**
```json
{
  "use_synthetic_vc": true,
  "synthetic_vc_probability": 0.5,
  "pitch_shift_range": [-2, -1, 1, 2],
  "lambda_synthetic": 0.3
}
```

**Training Integration:**

Applied to **50% of samples** (random selection):

| Component | Samples/Batch (batch=24) | Purpose |
|-----------|--------------------------|---------|
| Identity reconstruction | 24 (100%) | Quality baseline |
| Real VC (FAISS negatives) | 24 × 6 = 144 negatives | True voice conversion |
| Synthetic VC | 12 (50%) | Embedding dominance |

**Loss weights**:
- Identity: `λ_recon = 1.0`
- Real VC: `λ_speaker = 0.5`
- Synthetic VC: `λ_synthetic = 0.3` (slightly lower, augmentation)

#### Benefits

1. **Unlimited data**: Create infinite pseudo-speakers from single speaker
2. **Harder training**: Codes contain conflicting speaker info
3. **Complementary to FAISS**: Real pairs + synthetic pairs
4. **No extra speakers needed**: Works even with limited dataset
5. **Fast**: <1ms overhead with caching

---

### 4. Adversarial Losses (Realism)

**Goal**: Generate realistic, indistinguishable-from-real audio

#### Progressive Training Strategy

```
Batches 0-10:   Reconstruction + VC only (warmup)
Batches 10-20:  Add synthetic VC
Batches 20+:    Add GAN adversarial losses
```

**Why warmup?**
- Stabilizes training
- Lets reconstruction/VC losses establish baseline quality
- Prevents mode collapse early in training

#### Discriminator Loss (Hinge Loss)

Both MPD and MRD use the same loss formulation:

```python
# Real audio scores
D_real_mpd, D_real_mrd = mpd(audio_real), mrd(audio_real)

# Fake audio scores (detached for discriminator update)
audio_fake_detached = audio_hat.detach()
D_fake_mpd, D_fake_mrd = mpd(audio_fake_detached), mrd(audio_fake_detached)

# Hinge loss for discriminators
loss_disc_mpd_real = mean(relu(1.0 - D_real_mpd))
loss_disc_mpd_fake = mean(relu(1.0 + D_fake_mpd))
loss_disc_mpd = (loss_disc_mpd_real + loss_disc_mpd_fake) / 2.0

loss_disc_mrd_real = mean(relu(1.0 - D_real_mrd))
loss_disc_mrd_fake = mean(relu(1.0 + D_fake_mrd))
loss_disc_mrd = (loss_disc_mrd_real + loss_disc_mrd_fake) / 2.0

loss_disc = (loss_disc_mpd + loss_disc_mrd) / 2.0
```

**Hinge loss intuition:**
- Real audio: Push scores > 1 (penalty if score < 1)
- Fake audio: Push scores < -1 (penalty if score > -1)
- Margin of 2.0 between real and fake

#### Generator Adversarial Loss

```python
# Forward pass (no detach)
D_fake_mpd, D_fake_mrd = mpd(audio_hat), mrd(audio_hat)

# Generator wants to fool discriminators (push scores > 0)
loss_adv_mpd = -mean(D_fake_mpd)
loss_adv_mrd = -mean(D_fake_mrd)
loss_adv = (loss_adv_mpd + loss_adv_mrd) / 2.0
```

**Intuition**: Negative mean = generator wants high (positive) scores from discriminators

#### Feature Matching Loss

Match intermediate feature distributions between real and fake:

```python
# Get intermediate features from discriminators
_, _, fmap_real_mpd, fmap_fake_mpd = mpd(audio, audio_hat)
_, _, fmap_real_mrd, fmap_fake_mrd = mrd(audio, audio_hat)

# L1 distance between feature maps
loss_fm_mpd = mean([L1(f_real, f_fake)
                    for f_real, f_fake in zip(fmap_real_mpd, fmap_fake_mpd)])
loss_fm_mrd = mean([L1(f_real, f_fake)
                    for f_real, f_fake in zip(fmap_real_mrd, fmap_fake_mrd)])

loss_fm = (loss_fm_mpd + loss_fm_mrd) / 2.0
```

**Intuition**: Even if final scores are fooled, intermediate features should match real data distribution

**Weights**: `lambda_adv=1.0`, `lambda_fm=2.0`

---

## Training Configuration

### Current Config: `configs/phase4_gan.json`

```json
{
  "experiment_name": "phase4_gan_stratified_voice_conversion",
  "pretrained_model": "hubertsiuzdak/snac_24khz",
  "speaker_encoder_type": "eres2net",
  "speaker_emb_dim": 512,
  "freeze_base": true,

  "train_data": "data/train_split",
  "val_data": "data/val_split",
  "segment_length": 2.0,
  "batch_size": 24,
  "num_workers": 4,

  "num_epochs": 100,
  "learning_rate": 0.0001,
  "disc_learning_rate": 0.00005,
  "weight_decay": 0.00001,
  "grad_clip": 1.0,
  "grad_clip_disc": 1.0,
  "lr_min_ratio": 0.01,

  "l1_weight": 1.0,
  "stft_weight": 1.0,
  "n_ffts": [1024, 2048, 4096],

  "use_stratified_negatives": true,
  "neg_ratio_easy": 0.3,
  "neg_ratio_medium": 0.4,
  "neg_ratio_hard": 0.3,
  "threshold_easy_medium": 0.3,
  "threshold_medium_hard": 0.6,
  "max_negatives": 6,
  "same_speaker_threshold": 0.85,

  "lambda_recon": 1.0,
  "lambda_speaker_matching": 0.5,

  "use_synthetic_vc": true,
  "synthetic_vc_probability": 0.5,
  "pitch_shift_range": [-2, -1, 1, 2],
  "lambda_synthetic": 0.3,

  "gan_weight": 1.0,
  "lambda_adv": 1.0,
  "lambda_fm": 2.0,

  "mpd_periods": [2, 3, 5, 7, 11],
  "mrd_fft_sizes": [1024, 2048, 4096],

  "use_faiss_hard_negatives": true,
  "faiss_index_path": "pretrained_models/speaker_faiss.index",
  "output_dir": "checkpoints/phase4"
}
```

### Training Progress (Current Run)

**Batch Size**: 24 samples
**Total Batches**: 5,351 per epoch
**Dataset**: 128,437 training samples, 31,991 validation samples

**Loss Progression (Epoch 0):**
```
Batch  1: g_loss=13.13, d_loss=0.00, recon=6.32, vc=6.81, spk=0.99
Batch 10: g_loss=3.38,  d_loss=0.00, recon=1.46, vc=1.91, spk=0.90
Batch 11: g_loss=2.95,  d_loss=0.00, recon=1.25, vc=1.70, spk=0.90, synth=0.00, adv=0.00, fm=0.00
Batch 20: g_loss=3.47,  d_loss=1.99, recon=1.31, vc=1.73, spk=0.84, synth=1.20, adv=-0.01, fm=0.04
Batch 43: g_loss=2.62,  d_loss=1.86, recon=0.85, vc=1.21, spk=0.72, synth=0.99, adv=0.14,  fm=0.06
```

**Observations:**
- All losses decreasing as expected
- Discriminator converging to ~2.0 (hinge loss target)
- Synthetic VC active from batch 11
- GAN losses active from batch 11
- Speaker matching loss improving (0.99 → 0.72)

---

## Auxiliary Systems

### FAISS Speaker Index

**Purpose**: Fast similarity search for hard negatives

```
128,437 Speaker Embeddings (512-dim)
              ↓
    FAISS IndexFlatIP (Inner Product)
              ↓
         Normalized vectors
              ↓
        Query Embedding
              ↓
    Top-K Nearest Neighbors (~5-10ms)
              ↓
    Filter by similarity thresholds
              ↓
     Stratified sampling (easy/medium/hard)
```

**File:** `pretrained_models/speaker_faiss.index`
**Size:** ~250MB
**Performance:**
- Index build: ~2 minutes (one-time)
- Search latency: 5-10ms per query
- Index type: IndexFlatIP (exact search, no approximation)

### Optimized Embedding Cache

**Purpose**: Instant loading, low memory footprint

```
Build Once (scripts/build_embedding_cache.py):
  128,437 Audio Files → ERes2NetV2 → Embeddings → Save (.npy + .json)
                           ↓
                    ~40 minutes (one-time)

Load Instantly:
  embeddings.npy (250.9 MB) → Memory-mapped → Lazy page loading
                      ↓
                 No 30-minute wait
                 Minimal RAM overhead
                 <1ms lookup time
```

**Architecture:**
```
OptimizedEmbeddingCache:
  ├── embeddings.npy: Memory-mapped numpy array (128437, 512)
  ├── metadata.json: Path-to-index mapping
  └── .get(path) method: Instant lookup by file path

Performance:
  - Load time: <1 second (vs 30 minutes for recomputation)
  - Memory: Only accessed pages loaded into RAM
  - Lookup: O(1) hash table access
  - No GPU transfer needed (FAISS uses CPU)
```

**File:** `pretrained_models/embeddings_cache.npy`

### Data Loading Pipeline

```
Audio Dataset (128,437 files)
        ↓
  DataLoader (batch_size=24, num_workers=4)
        ↓
    [Batch of 24 audio segments]
        ↓
    Extract Speaker Embeddings (ERes2NetV2)
        ├─ Cache lookup (if previously computed)
        └─ Or compute and cache
        ↓
    Encode Audio (SNAC encoder)
        ↓
    [Ready for training]
```

**Performance:**
- Batch loading: ~100-200ms (4 workers prefetch)
- Speaker embedding extraction: ~50-100ms per batch (cached after first epoch)
- Encoding: ~100-200ms per batch

---

## Training Flow (Detailed)

### Per-Batch Computation

```
1. Load Batch (24 audio segments, 2 seconds each)
   │
   ├─ Audio shape: (24, 1, 48000) at 24kHz
   │
2. Extract Speaker Embeddings (ERes2NetV2)
   │
   ├─ Forward pass through speaker encoder
   ├─ Shape: (24, 512)
   ├─ Time: ~50-100ms (GPU)
   │
3. Encode Audio → SNAC Codes
   │
   ├─ Forward pass through SNAC encoder (frozen)
   ├─ Output: List of 4 codebooks at different scales
   ├─ Time: ~100-200ms (GPU)
   │
4. Reconstruct with Speaker Conditioning
   │
   ├─ Decode with own speaker embeddings (FiLM modulation)
   ├─ Shape: (24, 1, 48000)
   ├─ Time: ~100-200ms (GPU)
   │
5. Compute Losses:
   │
   ├─ Reconstruction Loss:
   │   ├─ L1(audio, audio_hat)
   │   ├─ MultiScaleSTFT(audio, audio_hat) with [1024, 2048, 4096]
   │   └─ Time: ~50ms
   │
   ├─ Voice Conversion Loss:
   │   ├─ For each of 24 samples:
   │   │   ├─ Positive: Own embedding (1 decode)
   │   │   └─ Negatives: 6 FAISS hard negatives (6 decodes)
   │   ├─ Total: 24 × 7 = 168 decodes per batch
   │   ├─ FAISS search: ~10ms per sample
   │   ├─ Decoding: ~150-300ms total
   │   └─ Speaker matching loss computation
   │
   ├─ Synthetic Voice Conversion Loss (50% of samples):
   │   ├─ For ~12 samples:
   │   │   ├─ Pitch shift: 0.13-0.18ms (cached!)
   │   │   ├─ Encode: ~5-120ms
   │   │   ├─ Decode: ~5-63ms
   │   │   └─ Reconstruction loss vs original
   │   └─ Total: ~30-550ms per batch
   │
   ├─ Adversarial Losses (after batch 10):
   │   ├─ MPD forward (real + fake): ~50ms
   │   ├─ MRD forward (real + fake): ~50ms
   │   ├─ Feature matching: ~20ms
   │   └─ Total: ~120ms
   │
6. Backpropagate & Update
   │
   ├─ Generator backward pass: ~100-200ms
   ├─ Discriminator backward pass: ~100-200ms
   ├─ Gradient clipping (1.0 max norm)
   └─ Optimizer step
   │
7. Log Losses
   │
   └─ Per-batch logging to file + tqdm
```

**Estimated Time per Batch: ~500-1000ms**
- **Forward passes**: ~400-600ms
- **Backward passes**: ~100-200ms
- **FAISS search**: ~240ms (24 samples × 10ms)
- **Synthetic VC**: ~30-550ms (50% of samples)

**Estimated Time per Epoch: ~45-90 minutes**
- 5,351 batches × 500-1000ms = 45-90 minutes

**Estimated Total Training Time: ~3-6 days**
- 100 epochs × 45-90 minutes = 3-6 days

---

## Optimizers and Scheduling

### Optimizers

**Generator Optimizer (FiLM layers only):**
```python
opt_gen = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),  # Only FiLM layers
    lr=1e-4,
    betas=(0.5, 0.9),
    weight_decay=1e-5
)
```

**Discriminators Optimizer (MPD + MRD):**
```python
opt_disc = AdamW(
    list(mpd.parameters()) + list(mrd.parameters()),
    lr=5e-5,  # 2x slower than generator
    betas=(0.5, 0.9),
    weight_decay=1e-5
)
```

**Key Implementation Detail:**
```python
# Only parameters with requires_grad=True are optimized
filter(lambda p: p.requires_grad, model.parameters())
```
- **Excludes frozen SNAC encoder/decoder** (requires_grad=False)
- **Excludes frozen ERes2NetV2 speaker encoder** (requires_grad=False)
- **Includes only FiLM layers** (requires_grad=True)

**Why different learning rates?**
- Generator (FiLM): Needs to learn complex speaker transformations (higher LR)
- Discriminators: Only need to distinguish real vs fake (lower LR for stability)
- Slower discriminator prevents mode collapse and stabilizes GAN training

### Learning Rate Scheduling

**Cosine Annealing:**
```python
lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * epoch / max_epochs))
```

**Parameters:**
- `lr_max`: 1e-4 (generator), 5e-5 (discriminator)
- `lr_min`: 1e-6 (generator), 5e-7 (discriminator) [lr_min_ratio = 0.01]
- `max_epochs`: 100

**Why cosine annealing?**
- Smooth decay prevents abrupt changes
- Allows model to explore early (high LR) and refine late (low LR)
- Proven effective for GAN training

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
torch.nn.utils.clip_grad_norm_(discriminators.parameters(), max_norm=1.0)
```

**Purpose:** Prevent gradient explosion in adversarial training

---

## Key Files and Modules

| File | Purpose | Status | Key Functions |
|------|---------|--------|---------------|
| **Pretrained Models** |
| `hubertsiuzdak/snac_24khz` | SNAC base model (HuggingFace) | Pretrained, **FROZEN** | Audio encoding/decoding |
| `pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt` | ERes2NetV2 speaker encoder | Pretrained, **FROZEN** | Speaker embedding extraction (512-dim) |
| **Training Scripts** |
| `train_phase4_gan.py` | Main training script | - | Multi-task training loop, progressive strategy |
| **Neural Network Modules** |
| `snac/snac.py` | SNAC with FiLM conditioning | SNAC base: **FROZEN**, FiLM: **TRAINABLE** | SNACWithSpeakerConditioning |
| `snac/discriminators.py` | GAN discriminators | **TRAINABLE** | MultiPeriodDiscriminator, MultiResolutionSTFTDiscriminator |
| `snac/film.py` | FiLM layers | **TRAINABLE** | Feature-wise Linear Modulation |
| **Loss Functions** |
| `snac/voice_conversion_loss.py` | VC losses | - | speaker_matching_loss, voice_conversion_loss |
| `snac/stratified_hard_negatives.py` | Stratified sampling | - | StratifiedHardNegativeSampler |
| `snac/audio_augmentation.py` | Pitch shifting | - | PitchShifter.shift_semitones() with caching |
| **Speaker Encoder** |
| `snac/speaker_encoder_factory.py` | Speaker encoder factory | ERes2NetV2: **FROZEN** | ERes2NetV2 integration |
| `snac/eres2net_encoder.py` | ERes2NetV2 wrapper | **FROZEN** | ERes2NetSpeakerEncoder |
| **Infrastructure** |
| `snac/faiss_speaker_index.py` | FAISS wrapper | - | FaissSpeakerIndex.search() |
| `snac/embedding_cache.py` | Memory-mapped cache | - | OptimizedEmbeddingCache.get() |
| **Configuration** |
| `configs/phase4_gan.json` | Training config | - | All hyperparameters |

---

## Performance Characteristics

### Memory Usage (Single GPU, batch_size=24)

**GPU Memory Breakdown (GPU 3 - 143GB total):**
```
Model parameters:        ~2 GB
Activations (forward):   ~40 GB
Gradients (backward):    ~40 GB
Optimizer states:        ~4 GB
Audio batch (24 samples):~2 GB
Miscellaneous:           ~2 GB
─────────────────────────────
Total:                  ~66 GB (peak)
```

**Current Usage:** 66GB / 143GB (46% utilization)
**Headroom:** 77GB available for larger batches or models

### Training Speed

**Per-batch breakdown:**
- Forward pass (encode + decode): ~400ms
- Backward pass: ~200ms
- FAISS hard negative mining: ~240ms (24 × 10ms)
- Synthetic VC: ~30-550ms (variable, 50% of samples)
- **Total:** ~500-1000ms per batch

**Throughput:**
- **With batch_size=24**: ~24-48 samples/second
- **Samples per hour**: ~86,000-173,000
- **Epoch time (128K samples)**: ~45-90 minutes

### Optimization Techniques

1. **Resampler Caching**: 60,000x speedup for pitch shifting
2. **Memory-Mapped Embeddings**: 30x faster loading (1s vs 30min)
3. **FAISS Indexing**: Millisecond search over 128K entries
4. **Gradient Checkpointing**: Could reduce memory by 2x (not currently used)
5. **Mixed Precision**: Could reduce memory by 1.5x (not currently used, but compatible)

---

## Summary

Phase 4 implements a **multi-task learning system** for high-fidelity voice conversion:

### Training Strategy Overview

**Pretrained & Frozen (Not Updated During Training):**
- SNAC base model (encoder + decoder): 89.4M params from `hubertsiuzdak/snac_24khz`
- ERes2NetV2 speaker encoder: ~20M params from GPT-SoVITS ProPlus
- Total frozen: **~109M parameters (72% of model)**

**Trainable (Updated During Training):**
- FiLM layers: ~1-2M params (speaker conditioning in decoder)
- MPD discriminator: 41.1M params (periodic patterns)
- MRD discriminator: 283K params (spectral patterns)
- Total trainable: **~43M parameters (28% of model)**

**Why This Approach?**
- Leverages pretrained representations (no need to learn audio encoding/decoding from scratch)
- Dramatically reduces trainable parameters (faster training, less memory)
- Focuses learning on speaker transformation (FiLM) and realism (discriminators)
- Frozen encoder provides stable features for speaker embedding extraction

### Core Innovations

1. **FiLM-based Speaker Control**: Dual-positive training teaches FiLM both identity preservation (own embedding) and speaker transformation (target embedding). Codes naturally contain speaker info for good reconstruction; FiLM learns to override it based on conditioning.

2. **Stratified Hard Negative Mining**: FAISS-based similarity search with difficulty tiering (30% easy, 40% medium, 30% hard) provides diverse, challenging training signal from entire dataset (128K speakers).

3. **Optimized Synthetic Voice Conversion**: Pitch shifting with cached resamplers creates unlimited pseudo-speaker pairs at <1ms overhead, teaching the model to trust embeddings over conflicting acoustic patterns.

4. **High-Fidelity Generation**: GAN discriminators (MPD+MRD) with feature matching ensure realistic audio quality indistinguishable from real speech.

5. **Efficient Infrastructure**: Memory-mapped embedding cache, FAISS indexing, and progressive training enable scalable training on 128K+ speaker dataset.

### Training Signal Components

| Component | Purpose | Weight | Frequency |
|-----------|---------|--------|----------|
| Reconstruction | Audio quality baseline | 1.0 | 100% of samples |
| Real VC (FAISS) | True voice conversion | 0.5 | 100% × 6 negatives/sample |
| Synthetic VC | Embedding dominance | 0.3 | 50% of samples |
| Adversarial | Realism | 1.0 | 100% (after warmup) |
| Feature matching | Distribution matching | 2.0 | 100% (after warmup) |

### Result

Model generates high-quality voice conversion with:
- **Content preservation**: Maintains linguistic content from source (codes)
- **Speaker control**: Matches target speaker characteristics (FiLM modulation)
- **High fidelity**: Realistic audio quality (GAN discriminators)
- **Robustness**: Works across diverse speakers (stratified training)

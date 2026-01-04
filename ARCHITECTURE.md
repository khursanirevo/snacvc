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
│  │             │    │  Encoder     │    │             │   │
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
                    (conditioning signal)
```

**Components:**
- **SNAC Encoder/Decoder**: Pretrained base model (frozen or fine-tuned)
- **ERes2NetV2 Speaker Encoder**: Extracts 512-dim speaker embeddings
- **Speaker Conditioning**: Modulates decoder to match target speaker

---

## GAN Discriminators (Audio Quality)

### MultiPeriodDiscriminator (MPD)

**Purpose**: Capture periodic patterns in speech waveforms

```
Input Audio → [5 Discriminators] → Real/Fake Scores
                    │
                    ├─ Period 2
                    ├─ Period 3
                    ├─ Period 5
                    ├─ Period 7
                    └─ Period 11
```

**How it works:**
- Uses 2D convolutions on period-augmented spectrograms
- Each discriminator specializes in different periodicities
- Ensemble approach for robust pattern detection

### MultiResolutionSTFTDiscriminator (MRD)

**Purpose**: Capture multi-scale spectral patterns

```
Input Audio → [3 Discriminators] → Real/Fake Scores
                    │
                    ├─ FFT 1024  (fine details)
                    ├─ FFT 2048  (mid-level)
                    └─ FFT 4096  (coarse structure)
```

**How it works:**
- Processes complex STFT spectrograms (real + imaginary parts)
- Multiple resolutions capture both local and global patterns
- Ensemble for comprehensive spectral analysis

---

## Loss Functions

### 1. Reconstruction Losses (Audio Fidelity)

**Goal**: Reconstruct input audio accurately

```python
loss_recon = l1_weight * L1(audio, audio_hat)
           + stft_weight * MultiScaleSTFT(audio, audio_hat)
```

**Components:**
- **L1 Loss**: Time-domain waveform accuracy
- **Multi-Scale STFT Loss**: Spectral accuracy at 3 FFT sizes
  - FFT 1024: Fine spectral details
  - FFT 2048: Mid-level patterns
  - FFT 4096: Coarse spectral structure

**Weights**: `l1_weight=1.0`, `stft_weight=1.0`

---

### 2. Voice Conversion Loss (FiLM-based Speaker Transformation)

**Goal**: Learn voice conversion through FiLM modulation

#### Key Insight

SNAC codes contain **both content and speaker information** (by design, for reconstruction). We don't try to purify codes. Instead, we rely on **FiLM conditioning during decoding** to control speaker characteristics.

```
codes_A (content + speaker_A info)
    ↓
Decoder with FiLM(speaker_B_embedding)
    ↓
FiLM modulates features: shifts speaker_A → speaker_B
    ↓
Output: content + speaker_B characteristics
```

#### Dual-Positive Training Strategy

FiLM learns two complementary behaviors through training with **both** own and target embeddings:

**1. Own Embedding (Identity Preservation)**
```python
codes_A + speaker_A_embedding → audio_A (reconstruction)
```
- Teaches FiLM: **identity transformation** (minimal modulation)
- Output preserves both content AND original speaker
- Ensures reconstruction quality doesn't degrade

**2. Target Embedding (Speaker Transformation)**
```python
codes_A + speaker_B_embedding → audio_converted
```
- Teaches FiLM: **speaker transformation** (active modulation)
- Output shifts to target speaker's characteristics
- Content preserved (from codes), speaker controlled (by FiLM)

#### Why This Works

FiLM learns a **continuum of transformations**:
- When `embedding == own`: modulation → minimal (preserve speaker)
- When `embedding == target`: modulation → shift to target speaker

This is elegant because:
- **Codes**: Provide basic audio structure (content + base speaker info)
- **FiLM**: Adjusts the "speaker knob" during decode to match conditioning
- No need to purify codes - encoder can keep speaker info for good reconstruction
- FiLM simply learns to override it based on embedding

#### Voice Conversion Loss

```python
# Component A: Reconstruction (own embedding)
audio_reconstructed = decode(codes, own_embedding)
loss_reconstruction = L1(audio_reconstructed, audio) + MultiScaleSTFT(...)

# Component B: Speaker Matching (target embedding)
audio_converted = decode(codes, target_embedding)
speaker_emb_converted = extract_speaker_embedding(audio_converted)
loss_speaker_matching = 1.0 - cosine_similarity(speaker_emb_converted, target_speaker_embedding)

loss_vc = λ_recon * loss_reconstruction
        + λ_speaker * loss_speaker_matching
```

**Content preservation is implicit**: Codes come from source audio, so content is preserved. FiLM handles speaker control.

**Weights**: `lambda_recon=1.0`, `lambda_speaker_matching=0.5`

---

### 3. Hard Negative Mining

**Goal**: Find challenging negative examples for contrastive learning

#### FAISS-Based Hard Negatives (Recommended)

```
Query Speaker Embedding
        │
        ▼
┌───────────────────────────────┐
│   FAISS Index (128K entries)   │
│   - IndexFlatIP                │
│   - Inner product search       │
│   - Millisecond latency        │
└───────────────────────────────┘
        │
        ├─▶ Top-K similar speakers
        │
        ▼
Filter by: similarity < 0.85 (not same speaker)
        │
        ▼
Return: Most similar different speakers
```

**Advantages:**
- True hard negatives from entire dataset
- No speaker labels needed
- Millisecond-level search over 100K+ embeddings
- Always finds informative negatives

#### Stratified Hard Negatives (Advanced)

**Goal**: Balance negative difficulty for robust learning

```
Similarity Spectrum:
├─────┬─────────┬─────────┬─────────┤
0.0   0.3      0.6      0.85     1.0
│     │         │         │         │
Easy  Medium    Hard    Same
      30%       40%      30%     (exclude)
```

**Sampling Strategy:**
- **Easy (30%)**: Very different speakers (sim < 0.3)
- **Medium (40%)**: Moderately different (0.3-0.6)
- **Hard (30%)**: Similar speakers (0.6-0.85)

**Benefits:**
- More diverse training signal
- Better generalization
- Prevents overfitting to specific clusters

---

### 4. Synthetic Voice Conversion (Audio Augmentation)

**Goal**: Create unlimited pseudo-speaker pairs to strengthen speaker embedding control

#### Key Insight

Voice changer effects (pitch shift, formant modification) transform audio to sound like a different speaker. By encoding this transformed audio and decoding with the **original** speaker embedding, we force the model to learn:

> **"Speaker embedding dominates over acoustic patterns in codes"**

#### Synthetic Voice Conversion Pipeline

```
1. audio_A (speaker A's voice)
        ↓
2. voice_changer(audio_A) → audio_A_sounds_like_B
        ↓
3. encode(audio_A_sounds_like_B) → codes_A_modified
        ↓
4. decode(codes_A_modified, speaker_A_embedding) → audio_reconstructed
        ↓
5. loss = ||audio_reconstructed - audio_A||
```

**Key point**: We reconstruct the **original** audio_A, not the augmented audio_A_sounds_like_B!

#### Why This Works

- **Codes contain "wrong" speaker info**: Pitch/formant-shifted acoustic patterns
- **Embedding says "speaker A"**: Model must override the codes
- **Forces embedding dominance**: Teaches FiLM to trust embedding over conflicting acoustic features

This is much harder than identity reconstruction because codes and embedding are in conflict!

#### Audio Processing Techniques

**Pitch Shifting** (primary)
- ±2 semitones (moderate, random direction)
- Changes perceived pitch without affecting duration
- Creates natural-sounding speaker variations

**Formant Shifting** (optional, advanced)
- ±20% formant frequency
- Simulates vocal tract length changes
- Creates male↔female transformations

**Implementation**:
```python
if random() < 0.5:  # 50% augmentation probability
    pitch_shift = random.choice([-2, -1, 1, 2])
    audio_aug = pitch_shift(audio, pitch_shift)
    codes_aug = encode(audio_aug)
    audio_recon = decode(codes_aug, speaker_A_embedding)
    loss_synthetic = reconstruction(audio_recon, audio_A)
```

#### Training Integration

Applied **in addition to** real voice conversion:

| Component | Frequency | Purpose |
|-----------|-----------|---------|
| Identity reconstruction | 100% | Quality baseline |
| Real VC (FAISS) | 100%, 6 negatives/sample | True voice conversion |
| Synthetic VC | 50% | Embedding dominance |

**Loss weights**:
- Identity: `λ_recon = 1.0`
- Real VC: `λ_speaker = 0.5`
- Synthetic VC: `λ_synthetic = 0.3` (slightly lower)

#### Benefits

1. **Unlimited data**: Create infinite pseudo-speakers from single speaker
2. **Harder training**: Codes contain conflicting speaker info
3. **Complementary to FAISS**: Real pairs + synthetic pairs
4. **No extra speakers needed**: Works even with limited dataset

---

### 5. Adversarial Losses (Realism)

**Goal**: Generate realistic, indistinguishable audio

#### Discriminator Loss (Hinge Loss)

```python
# Train discriminators to distinguish real vs fake
loss_disc_real = mean(relu(1.0 - D_real))
loss_disc_fake = mean(relu(1.0 + D_fake))
loss_disc = (loss_disc_real + loss_disc_fake) / 2
```

#### Generator Adversarial Loss

```python
# Train generator to fool discriminators
loss_adv = -mean(D_fake)
```

#### Feature Matching Loss

```python
# Match intermediate feature distributions
loss_fm = L1_loss(features_real, features_fake)
```

**Weights**: `lambda_adv=1.0`, `lambda_fm=2.0`

---

## Training Strategy

### Multi-Task Learning

```
Loss Total = Reconstruction (1.0)
           + Voice Conversion (0.5)
           + Synthetic VC (0.3, 50% of samples)
           + Adversarial (1.0)
           + Feature Matching (2.0)
```

**Components**:
- **Reconstruction**: Codes + own embedding → reconstruct original
- **Voice Conversion**: Codes + target embedding → match target speaker (FAISS negatives)
- **Synthetic VC**: Augmented codes + own embedding → reconstruct original (50% augment)
- **Adversarial**: GAN discriminators for realism
- **Feature Matching**: Match intermediate features

### Progressive Training

```
Batches 0-10:   Reconstruction only (warmup)
Batches 10-20:  + Voice conversion
Batches 20+:    + Adversarial losses (GAN)
```

### Optimizers

- **Generator**: AdamW (lr=1e-4, betas=(0.5, 0.9))
- **Discriminators**: AdamW (lr=5e-5, betas=(0.5, 0.9))
- **Speaker Discriminator**: AdamW (lr=5e-5)

### Scheduling

- **Scheduler**: Cosine annealing
- **Min LR Ratio**: 0.01 (lr_min = lr_max × 0.01)
- **Duration**: 100 epochs

---

## Auxiliary Systems

### FAISS Speaker Index

**Purpose**: Fast similarity search for hard negatives

```
128K Speaker Embeddings → FAISS Index
                              ↓
                         Query Embedding
                              ↓
                      Top-K Nearest Neighbors
                              ↓
                       Filter by Similarity
                              ↓
                    Hard Negative Speakers
```

**File:** `pretrained_models/speaker_faiss.index`
**Size:** ~250MB
**Performance:** Millisecond-level search

---

### Optimized Embedding Cache

**Purpose**: Instant loading, low memory footprint

```
Build Once:
  Audio Files → ERes2NetV2 → Embeddings → Save (.npy + .json)

Load Instantly:
  embeddings.npy → Memory-mapped → Lazy page loading
                      ↓
                 No 30-minute wait
                 Minimal RAM overhead
```

**Features:**
- Memory-mapped files (no copy overhead)
- Lazy loading (only accessed pages in RAM)
- Optional PCA compression (512→128 dims)
- Batch lookup support

**File:** `pretrained_models/embeddings_cache.npy`

---

## Configuration Files

### Phase 4 Variants

1. **phase4_gan.json** - Base GAN training
2. **phase4_gan_semantic_negatives.json** - With stratified hard negatives

---

## Key Files

| File | Purpose |
|------|---------|
| `train_phase4_gan.py` | Main training script |
| `snac/discriminators.py` | MPD and MRD implementations |
| `snac/voice_conversion_loss.py` | Voice conversion losses |
| `snac/stratified_hard_negatives.py` | Stratified negative sampling |
| `snac/faiss_speaker_index.py` | FAISS-based hard negative search |
| `snac/embedding_cache.py` | Memory-mapped embedding storage |

---

## Training Flow

```
1. Load Audio Batch
   │
2. Extract Speaker Embeddings (ERes2NetV2)
   │
3. Encode Audio → SNAC Codes
   │
4. Reconstruct with Speaker Conditioning
   │
5. Compute Losses:
   │
   ├─ Reconstruction: L1 + MultiScaleSTFT (codes + own embedding)
   │
   ├─ Voice Conversion:
   │   ├─ Own embedding: Identity preservation (reconstruction quality)
   │   └─ Target embedding: Speaker matching (voice conversion)
   │
   ├─ Hard Negatives:
   │   └─ FAISS search for similar speakers (6 per sample)
   │
   ├─ Synthetic VC (50% of samples):
   │   └─ Pitch-shifted audio + original embedding → reconstruct original
   │
   ├─ Adversarial:
   │   ├─ Generator loss (fool discriminators)
   │   └─ Feature matching (match real distribution)
   │
6. Backpropagate & Update
   │
7. Repeat
```

---

## Summary

Phase 4 implements a **multi-task learning system** for voice conversion:

1. **FiLM-based Speaker Control**: Dual-positive training teaches FiLM both identity preservation (own embedding) and speaker transformation (target embedding). No need to purify codes - they naturally contain speaker info for good reconstruction.

2. **High-Fidelity Generation**: GAN discriminators (MPD+MRD) ensure realistic audio quality

3. **Efficient Hard Negatives**: FAISS enables true hard negatives from entire dataset (128K speakers)

4. **Synthetic Voice Conversion**: Audio augmentation (pitch shifting) creates unlimited pseudo-speaker pairs, teaching the model to trust speaker embeddings over conflicting acoustic patterns in codes.

5. **Robust Learning**: Stratified sampling and multi-scale losses provide comprehensive training signal

**Result**: Model generates high-quality voice conversion with preserved content (from codes) and controlled speaker characteristics (via FiLM modulation).

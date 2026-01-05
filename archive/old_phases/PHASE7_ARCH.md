# Phase 7 Architecture: Reconstruction Consistency Loss

## Overview

Phase 7 builds on Phase 6's adapter-before-VQ architecture by adding a **reconstruction consistency loss** that regularizes the adapter's behavior. The key innovation is comparing conditioned vs unconditioned reconstructions to prevent the adapter from degrading audio quality while learning voice conversion.

## Core Architecture

### Phase 6 Foundation (Adapter BEFORE VQ)

```
Audio → Encoder → Latent z
                       ↓
                Adapter(speaker_emb)
                       ↓
                z_modulated = γ*z + β
                       ↓
                      VQ
                       ↓
         Speaker-Conditioned Codes
                       ↓
                    Decoder
                       ↓
                 Audio Output
```

**Key insight from Phase 6:** By modulating the encoder latent BEFORE quantization, the adapter produces speaker-conditioned codes that decode to the target speaker.

### Phase 7 Addition: Consistency Regularization

Phase 7 adds a parallel **unconditioned reconstruction** path:

```
                        ┌─→ With Emb → Adapter → VQ → audio_hat (conditioned)
Audio → Encoder → z ─┤
                        └─→ No Emb → (bypass) → VQ → audio_hat_uncond (baseline)
                                                        ↓
                                            Consistency Loss: L1+L2(audio_hat, audio_hat_uncond)
```

**IMPORTANT:** All speaker conditioning happens during ENCODE (before VQ). Decode() does NOT use speaker_embedding:
- `encode(audio, emb=X)` → adapter modulates latent → conditioned codes
- `decode(codes)` → decoder processes codes → audio (speaker already in codes!)
- The `speaker_embedding` parameter in `decode()` is IGNORED and has been removed from all Phase 7 calls

## Implementation Details

### 1. Dual Reconstruction Paths

**File:** `train_phase7.py` lines 435-459

```python
# Path 1: Conditioned reconstruction (with speaker embedding)
audio_hat, _ = model_base(audio, speaker_embedding=speaker_embs)

# Path 2: Unconditioned reconstruction (baseline, no speaker embedding)
audio_hat_uncond, _ = model_base(audio, speaker_embedding=None)

# Consistency loss
loss_l1 = torch.mean(torch.abs(audio_hat - audio_hat_uncond))
loss_recon_consistency = loss_l1
if config.get('use_l2_consistency', False):
    loss_l2 = torch.mean((audio_hat - audio_hat_uncond) ** 2)
    loss_recon_consistency = loss_l1 + loss_l2
```

**What happens when `speaker_embedding=None`:**

File: `snac/adapters.py` lines 169-174

```python
# Apply adapter conditioning BEFORE VQ
if speaker_embedding is not None:
    z = self.adapter(z, speaker_embedding)
else:
    # No conditioning, use latent as-is
    pass
```

When `speaker_embedding=None`:
- Adapter is **bypassed** (no FiLM modulation)
- Latent `z` passes through unchanged to VQ
- This is the **baseline SNAC reconstruction** (frozen model)

### 2. Synthetic VC (Pitch-Shifted Audio)

**IMPORTANT:** Synthetic VC was BUGGY in initial Phase 7 implementation. **FIXED:**

```python
# WRONG (OLD):
audio_aug = pitch_shift(audio)
codes_aug = encode(audio_aug)  # No speaker conditioning
audio_recon = decode(codes_aug, speaker_embedding=original_emb)
audio_recon_uncond = decode(codes_aug, speaker_embedding=None)
# Problem: decode() IGNORES speaker_embedding! Both calls are identical.

# CORRECT (NEW):
speaker_emb_original = speaker_embs[i]  # Extract BEFORE pitch shift
audio_aug = pitch_shift(audio)
codes_aug = encode(audio_aug, speaker_embedding=speaker_emb_original)  # Adapter modulates!
audio_recon = decode(codes_aug)  # Already conditioned during encode
# Adapter receives mismatched pair: (pitch-shifted audio, normal pitch embedding)
# Adapter learns: "restore original speaker from pitch-shifted codes"
```

**Why the fix matters:**
- OLD: Extracted embedding FROM pitch-shifted audio → embedding matches audio → adapter has nothing to do
- NEW: Uses ORIGINAL embedding → mismatched pair → adapter learns to restore speaker

### 3. Adapter Behavior

**File:** `snac/adapters.py` lines 67-92

FiLM (Feature-wise Linear Modulation):

```python
def forward(self, x, speaker_emb):
    h = self.mlp(speaker_emb)
    gamma = self.gamma_proj(h)  # (B, latent_dim)
    beta = self.beta_proj(h)    # (B, latent_dim)

    gamma = gamma.unsqueeze(-1)  # (B, latent_dim, 1)
    beta = beta.unsqueeze(-1)

    return gamma * x + beta
```

**Initialization (lines 61-65):**
```python
nn.init.zeros_(self.gamma_proj.weight)  # gamma starts at 0
nn.init.ones_(self.gamma_proj.bias)     # gamma = 1 (identity scale)
nn.init.zeros_(self.beta_proj.weight)   # beta starts at 0
nn.init.zeros_(self.beta_proj.bias)     # beta = 0 (no shift)
```

At initialization: `gamma ≈ 1, beta ≈ 0`, so `gamma * x + beta ≈ x` (identity transform)

## Loss Function Breakdown

### Complete Loss Equation

**File:** `train_phase7.py` lines 602-610

```python
lambda_synthetic = config.get('lambda_synthetic', 0.3)
lambda_recon_consistency = config.get('lambda_recon_consistency', 0.0)

loss_gen = (
    loss_recon +                              # Main reconstruction quality
    vc_losses['total'] +                      # Voice conversion (recon + speaker)
    lambda_recon_consistency * loss_recon_consistency +  # NEW: Consistency
    lambda_synthetic * synthetic_vc_loss +    # Synthetic VC
    lambda_adv * loss_adv +                   # GAN adversarial
    lambda_fm * loss_fm +                     # Feature matching
    loss_codebook_adv_enc                    # Codebook adversarial (optional)
)
```

---

## Detailed Loss Component Explanations

### 1. loss_recon (Main Reconstruction Loss)

**What it computes:**
```python
audio_hat = model(audio, speaker_embedding=speaker_embs)  # Identity reconstruction
loss_recon = reconstruction_loss(audio, audio_hat, config)
```

**reconstruction_loss breakdown** (from config):
```python
l1_weight = 1.0
stft_weight = 1.0
n_ffts = [1024, 2048, 4096]

loss = l1_weight * L1(audio, audio_hat) + stft_weight * multi_scale_STFT(audio, audio_hat)
```

**Purpose:**
- Ensures the model can reconstruct the original audio quality
- L1 loss preserves waveform similarity
- Multi-scale STFT loss preserves spectral characteristics
- **Applied to identity cases only** (using own speaker embeddings)

**Why it's needed:**
- Base requirement: model must produce high-quality audio
- Without this, adapter could degrade quality while learning speaker modulation
- L1 + STFT combination gives both time-domain and frequency-domain matching

**Expected values:**
- Initialization: ~0.3-0.4 (frozen SNAC model already good)
- During training: should stay < 0.4
- If > 0.5: adapter is degrading quality

**Code location:** `train_phase7.py` lines 444-445

**Weight:** `lambda_recon = 1.0`

---

### 2. loss_recon_consistency (Reconstruction Consistency Loss)

**What it computes:**
```python
# Conditioned reconstruction (with adapter)
audio_hat = model(audio, speaker_embedding=speaker_embs)

# Unconditioned reconstruction (adapter bypassed)
audio_hat_uncond = model(audio, speaker_embedding=None)

# Consistency: penalize difference
loss_l1 = torch.mean(torch.abs(audio_hat - audio_hat_uncond))
loss_recon_consistency = loss_l1

if config.get('use_l2_consistency', False):
    loss_l2 = torch.mean((audio_hat - audio_hat_uncond) ** 2)
    loss_recon_consistency = loss_l1 + loss_l2
```

**Purpose:**
- Regularizes the adapter to stay close to baseline SNAC reconstruction
- Forces adapter to make **minimal but effective** changes
- Prevents quality degradation from over-modulation

**Why it's needed:**
- Adapter could learn large (gamma, beta) values that shift latent dramatically
- This would change speaker characteristics BUT degrade quality
- Consistency loss ensures changes are subtle and controlled
- **Applied to identity cases only** (synthetic VC also uses this)

**Key insight:**
- `speaker_embedding=None` → adapter is bypassed (see `snac/adapters.py` lines 170-174)
- `audio_hat_uncond` is the FROZEN SNAC baseline (no speaker conditioning)
- By comparing to baseline, we prevent adapter from deviating too far

**Expected values:**
- **Target: < 1e-3** (very tight constraint for identity)
- Initialization: 0.001-0.002 (adapter ≈ identity)
- During training: ideally stays < 0.01
- If > 0.05: adapter making large changes (may degrade quality)

**Applied to:**
- Main identity reconstruction (lines 447-459) ✓
- **NOT applied to synthetic VC** (removed - was incorrect)
- **NOT applied to VC cases** (hard negatives)

**Code location:** `train_phase7.py` lines 447-459 only

**Weight:** `lambda_recon_consistency = 0.5`

**IMPORTANT:** Synthetic VC consistency was removed because:
- OLD code compared `decode(codes, emb=X)` vs `decode(codes, emb=None)`
- But decode() IGNORES speaker_embedding - both calls are identical!
- NEW code uses ORIGINAL embedding + encodes pitch-shifted audio
- Adapter actively restores original speaker - no consistency penalty needed

---

### 3. vc_losses (Voice Conversion Loss)

**What it computes:**
```python
# Called from train_phase7.py line 468-472
vc_losses = voice_conversion_loss(
    model_base, audio, codes, speaker_embs, config,
    faiss_index=faiss_index,
    embedding_cache=embedding_cache
)

# Returns dict with:
# {
#   'total': combined loss,
#   'reconstruction': identity reconstruction loss,
#   'speaker_matching': combined speaker loss,
#   'speaker_matching_identity': identity speaker loss,
#   'speaker_matching_vc': voice conversion speaker loss
# }
```

**voice_conversion_loss breakdown** (`snac/voice_conversion_loss.py`):

For each audio sample `i` in batch:

#### 3a. Identity Case (Own Embedding)
```python
speaker_emb_positive = speaker_embs[i:i+1]  # Own embedding

# Encode with own embedding (adapter modulates)
codes_positive = model.encode(audio[i:i+1], speaker_embedding=speaker_emb_positive)

# Decode
audio_positive = model.decode(codes_positive)

# 1. Reconstruction: should match original audio
loss_recon_i = reconstruction_loss(audio[i:i+1], audio_positive, config)

# 2. Speaker matching: should match own speaker
loss_speaker_identity_i = speaker_matching_loss(
    model, audio_positive, speaker_emb_positive, config
)

# Speaker matching loss:
#   decoded_emb = extract_speaker_embedding(audio_positive)
#   similarity = cosine_similarity(decoded_emb, speaker_emb_positive)
#   loss = (1 - similarity).mean()
```

#### 3b. Voice Conversion Case (Hard Negative Embeddings)
```python
# Get hard negatives from FAISS (different speakers)
hard_neg_embs = faiss_index.search(speaker_embs[i], k=num_negatives)

for speaker_emb_negative in hard_neg_embs:
    # Encode with TARGET speaker embedding
    codes_negative = model.encode(audio[i:i+1], speaker_embedding=speaker_emb_negative)

    # Decode
    audio_negative = model.decode(codes_negative)

    # Speaker matching: should match TARGET (different) speaker
    loss_speaker_vc_i = speaker_matching_loss(
        model, audio_negative, speaker_emb_negative, config
    )
```

#### 3c. Combined Loss
```python
lambda_recon = 1.0
lambda_speaker_identity = 0.25  # Lower weight for identity
lambda_speaker_vc = 0.7         # Higher weight for VC

loss_identity = lambda_recon * loss_recon + lambda_speaker_identity * loss_speaker_identity
loss_vc = lambda_speaker_vc * loss_speaker_vc

total = (loss_identity + loss_vc) / batch_size
```

**Purpose:**
- **Identity**: Teaches adapter that `encode(audio, own_emb)` → reconstruct original
- **VC**: Teaches adapter that `encode(audio_A, emb_B)` → codes that decode to speaker B
- Uses hard negatives (similar speakers) for difficult contrastive learning

**Why it's needed:**
- Main `loss_recon` only covers one identity reconstruction per batch
- This provides per-sample identity + multiple VC pairs per sample
- Hard negatives from FAISS ensure learning difficult cases
- Separates identity preservation from voice conversion learning

**Expected values:**
- `speaker_matching_identity` (spk_id): 0.3-0.4 (baseline, frozen model limitation)
- `speaker_matching_vc` (spk_vc):
  - Init: 0.98-1.0 (adapter does nothing, wrong speaker)
  - Target: 0.4-0.5 (adapter learns voice conversion)
- Gap (spk_vc - spk_id): Should shrink from 0.6 → <0.1

**Code location:**
- Call: `train_phase7.py` lines 468-472
- Implementation: `snac/voice_conversion_loss.py` lines 90-280

**Weights:**
- `lambda_recon = 1.0` (within vc_loss)
- `lambda_speaker_identity = 0.25`
- `lambda_speaker_vc = 0.7`

---

### 4. synthetic_vc_loss (Synthetic Voice Conversion Loss)

**What it computes:**
```python
# For each sample in batch (with probability synthetic_vc_probability = 0.5)
if random.random() < 0.5:
    # 1. Get ORIGINAL speaker embedding (extracted BEFORE augmentation)
    speaker_emb_i = speaker_embs[i:i+1]

    # 2. Pitch shift the audio (create pseudo-speaker)
    audio_aug = pitch_shift(audio[i:i+1], semitones=random.choice([-2, -1, 1, 2]))

    # 3. Encode pitch-shifted audio WITH ORIGINAL speaker embedding
    # Adapter receives: mismatched pair (pitch-shifted audio + normal pitch embedding)
    # Adapter learns: "restore original speaker from pitch-shifted audio"
    codes_aug = model.encode(audio_aug, speaker_embedding=speaker_emb_i)

    # 4. Decode (decoder ignores speaker_embedding, but we pass it for API)
    audio_synthetic_recon = model.decode(codes_aug, speaker_embedding=speaker_emb_i)

    # 5. Reconstruction loss: should match ORIGINAL audio (not augmented)
    loss_synth_recon = reconstruction_loss(audio[i:i+1], audio_synthetic_recon, config)

    # 6. Speaker matching: decoded audio should match ORIGINAL speaker
    loss_synth_speaker = speaker_matching_loss(
        model, audio_synthetic_recon, speaker_emb_i, config
    )

    loss_synth = loss_synth_recon
```

**Purpose:**
- Teaches adapter to separate **pitch** from **speaker characteristics**
- Forces adapter to restore original speaker from pitch-shifted audio
- Makes adapter robust to acoustic variations

**Why it's needed:**
- Real VC: speakers have different pitch ranges
- Synthetic VC: explicitly trains pitch-invariance
- **Critical fix:** Uses ORIGINAL speaker embedding (extracted before pitch shift)
- Without this fix: embedding would be extracted FROM pitch-shifted audio (mismatch!)

**Key insight:**
- **OLD (WRONG):** `encode(audio_pitch_shifted)` → no adapter → codes encode pitch-shifted characteristics
- **NEW (CORRECT):** `encode(audio_pitch_shifted, emb=original)` → adapter modulates toward original speaker
- Adapter receives **mismatched pair**: (high pitch audio, normal pitch embedding)
- Adapter learns: "ignore pitch in audio, use embedding to restore speaker"

**Expected values:**
- `synthetic_vc_loss`: 0.3-0.6 (more difficult than regular reconstruction)
- `spk_synth` (synthetic speaker loss): similar to spk_id (~0.3-0.4)
- If spk_synth >> spk_id: adapter not robust to pitch variations

**Code location:** `train_phase7.py` lines 519-551

**Weight:** `lambda_synthetic = 0.3`

---

### 5. loss_adv (GAN Adversarial Loss)

**What it computes:**
```python
# Multi-Period Discriminator (MPD)
y_d_rs_mpd, y_d_gs_mpd, _, _ = mpd(audio, audio_hat)
loss_disc_mpd = discriminator_loss(y_d_rs_mpd, y_d_gs_mpd)

# Multi-Resolution STFT Discriminator (MRD)
y_d_rs_mrd, y_d_gs_mrd, _, _ = mrd(audio, audio_hat)
loss_disc_mrd = discriminator_loss(y_d_rs_mrd, y_d_gs_mrd)

# For generator:
# Real audio should be classified as real (score → 1)
# Fake audio (audio_hat) should be classified as real (score → 1)
loss_adv = -(adversarial_loss(audio_hat))  # Generator wants to fool discriminator

# For discriminator:
# Real audio → score 1, Fake audio → score 0
loss_disc = 0.5 * (loss_disc_mpd + loss_disc_mrd)
```

**Purpose:**
- Makes reconstructions more realistic
- Discriminator learns to distinguish real vs generated audio
- Generator learns to fool discriminator → higher perceptual quality

**Why it's needed:**
- L1 + STFT losses ensure waveform/spectral matching
- But they don't capture high-level perceptual quality
- GAN adversarial loss adds realism and reduces artifacts
- Standard in speech synthesis/voice conversion

**Expected values:**
- `loss_adv`: -0.03 to -0.1 (negative = generator fooling discriminator)
- `loss_disc`: ~2.0 (discriminator learning)
- If adv > 0: discriminator too strong, generator not learning

**Warmup:** Skipped for first 10 batches to stabilize training

**Code location:**
- Generator: `train_phase7.py` lines 579-596
- Discriminator: `train_phase7.py` lines 598-617

**Weight:** `lambda_adv = 1.0`

---

### 6. loss_fm (Feature Matching Loss)

**What it computes:**
```python
# MPD forward (returns feature maps)
y_d_rs_mpd, y_d_gs_mpd, fmap_rs_mpd, fmap_gs_mpd = mpd(audio, audio_hat)

# MRD forward (returns feature maps)
y_d_rs_mrd, y_d_gs_mrd, fmap_rs_mrd, fmap_gs_mrd = mrd(audio, audio_hat)

# Feature matching: L1 distance between real and fake feature maps
loss_fm_mpd = sum(L1(fmap_rs, fmap_gs) for fmap_rs, fmap_gs in zip(fmap_rs_mpd, fmap_gs_mpd))
loss_fm_mrd = sum(L1(fmap_rs, fmap_gs) for fmap_rs, fmap_gs in zip(fmap_rs_mrd, fmap_gs_mrd))

loss_fm = loss_fm_mpd + loss_fm_mrd
```

**Purpose:**
- Matches intermediate features of discriminator
- Ensures generated audio has similar statistics to real audio
- **Most important GAN loss** (highest weight)

**Why it's needed:**
- Adversarial loss alone can cause mode collapse or artifacts
- Feature matching provides gradient at multiple layers
- More stable training than adversarial alone
- Proven effective in speech synthesis (HiFi-GAN, etc.)

**Key insight:**
- Discriminator features capture: periodicity, frequency patterns, timbre
- Matching these features ensures realistic audio characteristics
- Higher weight (2.0) because it's crucial for quality

**Expected values:**
- `loss_fm`: 0.01-0.03 (relatively low, features matching well)
- If > 0.1: discriminator features diverging, quality issues

**Code location:** `train_phase7.py` lines 566-577 (MPD/MRD forward)

**Weight:** `lambda_fm = 2.0` (highest weight!)

---

### 7. loss_codebook_adv_enc (Codebook Adversarial Loss)

**What it computes:**
```python
# Optional: Train speaker discriminator to predict speaker from codes
# Then train encoder to FOOL this discriminator (gradient reversal)

# Forward through speaker discriminator
adv_losses = adversarial_codebook_loss_v2(
    codes=codes,
    speaker_discriminator=speaker_disc,
    speaker_ids=speaker_ids,
    mode='encoder'  # Train encoder to fool discriminator
)

loss_codebook_adv_enc = adv_losses['encoder_loss']
```

**Purpose:**
- Optional: Prevents speaker information leakage into codes
- Makes codes content-only, speaker information in embedding
- Not always used (use_codebook_adversarial flag)

**Why it's needed (if used):**
- In theory: adapter handles all speaker conditioning
- In practice: some speaker info might leak into codes
- Adversarial loss purifies codebook to be speaker-invariant
- Currently **optional** and not main focus

**Expected values:**
- Often 0.0 (disabled by default)
- If enabled: 0.05-0.15

**Code location:** `train_phase7.py` lines 570-600

**Weight:** `lambda_codebook_adv = 0.1`

---

## Loss Weights Summary (configs/phase7.json)

```json
{
  "lambda_recon": 1.0,                    // Main reconstruction (quality)
  "lambda_speaker_identity": 0.25,        // Identity preservation (lower)
  "lambda_speaker_vc": 0.7,               // Voice conversion (higher than identity)
  "lambda_recon_consistency": 0.5,        // Regularize adapter (keep < 1e-3)
  "lambda_synthetic": 0.3,                // Synthetic VC (pitch invariance)
  "lambda_adv": 1.0,                      // GAN adversarial (realism)
  "lambda_fm": 2.0                        // Feature matching (highest!)
}
```

**Weight rationale:**
1. **lambda_fm = 2.0** (highest) - Feature matching is most important for quality
2. **lambda_recon = 1.0** - Base reconstruction quality
3. **lambda_adv = 1.0** - GAN adversarial for realism
4. **lambda_recon_consistency = 0.5** - Regularize adapter (identity only)
5. **lambda_speaker_vc = 0.7** - Voice conversion learning
6. **lambda_speaker_identity = 0.25** - Identity preservation (lower, let VC dominate)
7. **lambda_synthetic = 0.3** - Synthetic VC (pitch invariance)

---

## Why Consistency Loss Makes Sense

### The Problem It Solves

**Phase 6 issue:** Adapter could learn to make large modifications to the latent space, causing:
1. Quality degradation (adapter over-modulates)
2. Instability (large deviations from pretrained model)
3. Poor generalization (learns speaker-specific quirks instead of general characteristics)

**Phase 7 solution:** Penalize deviation from the baseline reconstruction:
- Forces adapter to make **minimal but effective** changes
- Preserves the pretrained model's quality
- Regularizes the transformation strength

### Interpretation

**Low `recon_con` (< 0.01):** Adapter makes small, efficient changes
- Good! Adapter is learning to modulate elegantly

**Increasing `recon_con`:** Adapter is deviating more from baseline
- Could be OK if voice conversion is improving
- Monitor with other metrics (spk_vc, recon quality)

**Very high `recon_con` (> 0.1):** Adapter is making drastic changes
- Warning sign: potential quality degradation
- Consider increasing `lambda_recon_consistency` weight

## Training Dynamics

### Expected Behavior

**Initialization (step 0):**
- `recon_con ≈ 0.001-0.002` (adapter ≈ identity, gamma≈1, beta≈0)
- `spk_vc ≈ 1.0` (no voice conversion yet)
- `recon ≈ 0.3-0.4` (baseline quality)

**Early training (steps 0-1000):**
- `recon_con` increases to 0.01-0.02 (adapter learns to modulate)
- `spk_vc` decreases to 0.8-0.9 (voice conversion improving)
- `recon` stays ~0.3 (quality maintained)

**Mid training (steps 1000-5000):**
- `recon_con` stabilizes or grows slowly
- `spk_vc` continues decreasing toward 0.4-0.5
- `recon` may increase slightly if adapter over-modulates

**Convergence:**
- Balance between `recon_con` (consistency) and `spk_vc` (conversion)
- Ideal: `recon_con` < 0.05, `spk_vc` < 0.5, `recon` < 0.4

## Monitoring Metrics

**File:** `train_phase7.py` lines 679-685

Progress bar shows:
```
g_loss, d_loss, recon, recon_con, vc, spk_id, spk_vc, spk_synth, spk, adv, fm
```

**Key metrics for Phase 7:**
- `recon_con`: Consistency loss (should stay < 0.05 ideally)
- `spk_vc`: Voice conversion speaker loss (should decrease to < 0.5)
- `recon`: Reconstruction quality (should stay < 0.4)
- `spk_id`: Identity preservation (baseline ~0.3-0.4)

## Comparison with Phase 6

| Aspect | Phase 6 | Phase 7 |
|--------|---------|---------|
| Architecture | Adapter before VQ | Same |
| Main loss | recon + vc | recon + vc + consistency |
| VC weight | lambda_speaker_vc = 2.0 | lambda_speaker_vc = 0.7 |
| Consistency | None | lambda_recon_consistency = 0.5 |
| Synthetic VC | Basic recon + speaker | Adds consistency loss |
| Regularization | Implicit (via losses) | Explicit (consistency) |

## Configuration File

**File:** `configs/phase7.json`

Key settings:
```json
{
  "experiment_name": "phase7_recon_consistency",
  "adapter_type": "film",
  "adapter_hidden_dim": 512,
  "adapter_num_layers": 2,
  "lambda_recon_consistency": 0.5,
  "use_l2_consistency": true,
  "lambda_adapter_identity": 0.1,  // NOT YET IMPLEMENTED
  "output_dir": "checkpoints/phase7"
}
```

## Notes

1. **`lambda_adapter_identity`** is defined in config but NOT YET USED in code
   - Would add direct penalty on (gamma-1)^2 + beta^2
   - This would be a MORE DIRECT regularization of adapter parameters
   - Currently commented out pending implementation

2. **Decode ignores speaker_embedding:**
   - `decode(codes, speaker_embedding=None)` is identical to `decode(codes)`
   - All speaker conditioning happens during ENCODING (before VQ)
   - This is by design in Phase 6/7 architecture

3. **Consistency vs Voice Conversion trade-off:**
   - Higher `lambda_recon_consistency` → better quality, slower VC learning
   - Lower `lambda_recon_consistency` → faster VC learning, risk of quality loss
   - Current: 0.5 (balanced)

4. **Synthetic VC is crucial:**
   - Teaches adapter to separate pitch from speaker characteristics
   - Pitch-shifted codes + original embedding = restored speaker
   - Consistency loss ensures smooth restoration

## Future Improvements

1. **Implement `lambda_adapter_identity`:** Direct penalty on FiLM parameters
2. **Dynamic weight scheduling:** Start high, decrease over time
3. **Per-layer consistency:** Track which adapter layers deviate most
4. **Content-aware consistency:** Allow more deviation in formants, less in linguistic content

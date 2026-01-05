# Chatterbox Voice Conversion Analysis & Application to SNAC

## Chatterbox's VC Approach

### Architecture Flow
```
Source Audio (content)
    ↓
S3 Tokenizer (16kHz) → Discrete Tokens
    ↓
Flow Matching Decoder (CONDITIONED on target speaker)
    ├─ Speech Tokens (from source)
    ├─ Speaker Embedding (from target voice) ← CAMPPlus x-vector
    ├─ Reference Mel (from target voice)
    └─ CFG (Classifier-Free Guidance)
    ↓
HiFi-GAN Vocoder → Converted Audio
```

### Key Components

1. **S3 Tokenizer**: Encodes audio to discrete tokens (speaker-agnostic)
2. **Speaker Encoder**: CAMPPlus (x-vector) extracts 512-dim speaker embeddings
3. **Flow Matching**: Generative model conditioned on:
   - `spks`: Speaker embedding (80-dim projected from 512-dim)
   - `prompt_token`: Reference speech tokens from target
   - `prompt_feat`: Reference mel features from target
4. **Conditioning Location**: **Decoder/generator ONLY** (not during encoding)

### Critical Insight
**Chatterbox does NOT modify the tokenizer/encoder.** The speech tokens remain speaker-agnostic. All speaker conditioning happens in the decoder/generator phase.

---

## Applying to SNAC for Voice Conversion

### Current SNAC Architecture
```
Audio → Encoder → Latent → VQ → Codes → Decoder → Audio
```

### Two Approaches for VC

### Approach 1: Decoder Conditioning (Chatterbox-style) ✅ RECOMMENDED

**Flow:**
```
Source Audio → SNAC.encode() → Codes (content, speaker-agnostic)
    ↓
Target Audio → Speaker Encoder → Embedding (timbre)
    ↓
SNAC.decode(codes, speaker_embedding=target_emb) → Converted Audio
```

**Implementation:**
- Modify SNAC **Decoder** to accept speaker embedding
- Add FiLM (Feature-wise Linear Modulation) to decoder layers
- Keep encoder frozen and unmodified
- Codes remain speaker-agnostic (reversible)

**Pros:**
- ✅ Codes remain reversible and speaker-agnostic
- ✅ Can change speaker after encoding
- ✅ Same codes can be decoded with different speakers
- ✅ Matches Chatterbox's proven approach

**Cons:**
- ❌ Decoder needs to learn speaker transfer from scratch
- ❌ May have difficulty if decoder was trained without conditioning

### Approach 2: Encoder Conditioning (Phase 6 - what we tried)

**Flow:**
```
Source Audio → SNAC.encode(audio, speaker_emb=target) → Speaker-Conditioned Codes
    ↓
SNAC.decode(codes) → Converted Audio
```

**Implementation:**
- Add Adapter/FiLM BEFORE VQ (modulates encoder latent)
- Codes themselves contain speaker information
- Decoder unchanged

**Pros:**
- ✅ Speaker info baked into codes
- ✅ Can standardize on one codebook

**Cons:**
- ❌ Codes are NOT speaker-agnostic (can't reconstruct original)
- ❌ Can't change speaker after encoding
- ❌ Lossy for original speaker

---

## Recommended Implementation Plan

### Phase 10: SNAC Voice Conversion (Decoder Conditioning)

**Architecture:**
```python
class SNACWithSpeakerDecoder(nn.Module):
    def __init__(self, base_snac, speaker_emb_dim=512):
        self.snac = base_snac  # Freeze encoder and VQ
        self.speaker_encoder = CAMPPlus()  # Or ERes2NetV2
        self.decoder_film = FiLMAdapter(
            decoder_channels=[512, 512, 512, 512],
            speaker_emb_dim=512
        )

    def encode(self, audio):
        # Speaker-agnostic encoding
        return self.snac.encode(audio)

    def decode(self, codes, speaker_embedding=None):
        # Apply FiLM conditioning to decoder
        return self.snac.decode(codes, speaker_embedding)

    def voice_convert(self, source_audio, target_speaker_audio):
        # Encode source (content)
        codes = self.encode(source_audio)

        # Extract target speaker embedding
        target_emb = self.speaker_encoder(target_speaker_audio)

        # Decode with target speaker
        return self.decode(codes, target_emb)
```

**Training Strategy:**
1. **Freeze**: Encoder, VQ/Quantizer
2. **Trainable**: Decoder FiLM adapters, Speaker encoder (optional)
3. **Loss**:
   - Reconstruction: `decode(encode(A), spk=B) ≈ A_with_B_voice`
   - Speaker loss: Ensure speaker characteristics transfer
   - Content preservation: Maintain linguistic content

**Data Augmentation:**
- Synthetic VC: Pitch-shifted audio with own embedding (identity)
- Cross-speaker: Real A→B pairs
- Hard negatives: FAISS-based mining (from Phase 6)

---

## Key Differences from Phase 6

| Aspect | Phase 6 (Encoder Conditioning) | Phase 10 (Decoder Conditioning) |
|--------|-------------------------------|----------------------------------|
| Conditioning location | Before VQ | In decoder |
| Code properties | Speaker-conditioned | Speaker-agnostic |
| Reversibility | ❌ Cannot reconstruct original | ✅ Can reconstruct original |
| Speaker flexibility | Fixed at encode time | Flexible at decode time |
| Chatterbox alignment | ❌ Different approach | ✅ Matches Chatterbox |

---

## Next Steps

1. **Implement FiLM decoder adapters** for SNAC
   - Add FiLM layers to each decoder block
   - Project 512-dim speaker emb → (gamma, beta) for each layer

2. **Prepare VC training data**
   - Source audio A + target speaker B pairs
   - Use same dataset as Phase 6
   - FAISS indexing for hard negatives

3. **Training script**: `train_phase10_vc_decoder.py`
   - Load fine-tuned SNAC from Phase 9
   - Add decoder FiLM conditioning
   - Train with voice conversion loss

4. **Inference script**: `inference_phase10_vc.py`
   - `voice_convert(source_audio, target_speaker_audio)`
   - Test on held-out speakers

---

## References from Chatterbox

**Speaker Encoder:**
- `chatterbox/src/chatterbox/models/s3gen/xvector.py` - CAMPPlus implementation
- Line 57 in `s3gen.py`: `self.speaker_encoder = CAMPPlus()`
- Line 152: `ref_x_vector = self.speaker_encoder.inference(ref_wav_16)`

**Conditioning in Decoder:**
- Line 96 in `flow_matching.py`: `spk_emb_dim=80`
- Line 184: `pred = self.estimator(y, mask, mu, t.squeeze(), spks, cond)`
- Line 42: `spks` parameter passed to forward

**Flow Matching:**
- `CausalConditionalCFM` - generative model
- Uses Classifier-Free Guidance (CFG) for better quality

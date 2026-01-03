# SNAC Speaker Conditioning - Ablation Study

## Overview

This ablation study compares three approaches for training speaker-conditioned SNAC using FiLM layers:

### Phase 1: Reconstruction Only (Baseline)
**Config**: `configs/phase1_reconstruction_only.json`
- **Loss**: L1 + multi-scale STFT
- **Batch size**: 8
- **Speaker loss**: None (weight = 0.0)
- **GPU**: 3
- **Status**: ✅ Running

**Expected behavior**: Model learns to reconstruct audio but speaker conditioning is weak since it's never explicitly trained.

### Phase 2: Reconstruction + Self Speaker Loss
**Config**: `configs/phase2_with_speaker_loss.json`
- **Loss**: L1 + multi-scale STFT + 0.1 × speaker_consistency_loss
- **Batch size**: 8
- **Speaker loss**: Cosine similarity between original and reconstructed speaker embeddings
- **GPU**: 1
- **Status**: ✅ Running

**Issue**: Speaker loss is tiny (1e-5) because self-reconstruction trivially preserves speaker identity.

### Phase 3: Reconstruction + Contrastive Speaker Loss (NEW!)
**Config**: `configs/phase3_contrastive.json`
- **Loss**: L1 + multi-scale STFT + 0.5 × contrastive_speaker_loss
- **Batch size**: 16 (doubled!)
- **Speaker loss**: Margin-based contrastive loss with negative sampling
- **GPU**: TBD (suggested: 2 or 0)
- **Status**: 🔄 Ready to start

**Key innovation**: For each audio in a batch of 16:
- ✅ Positive: Decode with correct speaker embedding → should reconstruct well
- ❌ Negatives: Decode with 15 wrong speaker embeddings → should reconstruct poorly
- Loss: `max(0, loss_negative - loss_positive + margin)`

This explicitly teaches the model that speaker embeddings matter!

---

## Why Negative Sampling Matters

### Phase 2 Problem
```python
# Current approach (Phase 2)
audio → encode → decode(same_speaker_emb) → reconstruct
loss = recon_loss + 0.1 × (1 - cosine_similarity(orig_emb, recon_emb))
```

**Issue**: The model encodes and decodes the SAME audio, so speaker similarity is trivial. The loss is tiny because the model naturally preserves some speaker characteristics even without explicit conditioning.

### Phase 3 Solution
```python
# New approach (Phase 3)
batch = [audio_1, audio_2, ..., audio_16]
speaker_embs = [emb_1, emb_2, ..., emb_16]

For each audio_i:
  codes = encode(audio_i)

  # Positive: correct speaker
  audio_positive = decode(codes, emb_i)
  loss_positive = recon_loss(audio_i, audio_positive)

  # Negatives: wrong speakers
  for j ≠ i:
    audio_negative = decode(codes, emb_j)
    loss_negative = recon_loss(audio_i, audio_negative)

  # Contrastive loss: make negative loss >> positive loss
  loss_contrastive = max(0, loss_negative - loss_positive + 0.1)
```

**Result**: The model learns that decoding with the WRONG speaker embedding produces BAD reconstructions. This is the signal we need!

---

## Training Commands

### Phase 1 (Running)
```bash
CUDA_VISIBLE_DEVICES=3 uv run python train.py --config configs/phase1_reconstruction_only.json 2>&1 | tee phase1_training.log
```

### Phase 2 (Running)
```bash
CUDA_VISIBLE_DEVICES=1 uv run python train.py --config configs/phase2_with_speaker_loss.json 2>&1 | tee phase2_training.log
```

### Phase 3 (NEW - Ready to start)
```bash
# Run on GPU 2 (currently 83% utilized, should be fine)
CUDA_VISIBLE_DEVICES=2 uv run python train_contrastive.py --config configs/phase3_contrastive.json 2>&1 | tee phase3_training.log
```

---

## Comparison Metrics

After training completes, we'll compare:

1. **Reconstruction Quality**
   - L1 distance
   - Multi-scale spectral loss
   - PESQ/STOI (if time permits)

2. **Speaker Similarity** (quantitative)
   - Cosine similarity between original and reconstructed embeddings
   - Higher = better speaker preservation

3. **Speaker Manipulation** (qualitative)
   - Encode content from speaker A
   - Decode with speaker embedding from B
   - Listen: Does it preserve content with speaker B's voice?

4. **Training Dynamics**
   - Convergence speed
   - Loss curves
   - Stability

---

## Expected Results

| Phase | Recon Loss | Speaker Preservation | Speaker Manipulation |
|-------|------------|---------------------|---------------------|
| 1     | Best       | Weak                | Weak                |
| 2     | Good       | Medium              | Medium              |
| 3     | Good       | Strong              | Strong              |

**Hypothesis**: Phase 3 will show the best speaker manipulation because:
- Larger batch (16 vs 8) = more diverse negatives per batch
- Contrastive loss explicitly teaches speaker conditioning
- Negative sampling provides strong learning signal

---

## Notes

- All phases use same dataset (79,479 train, 19,964 val)
- All phases freeze base SNAC, only train FiLM parameters
- Phase 3 is 2-3x slower per batch due to negative sampling (16 reconstructions per audio vs 1)
- GPU memory is plentiful (12GB used out of 144GB), so batch_size=32 is also possible

To increase batch size to 32:
```json
"batch_size": 32,
"max_negatives": 15  # Keep this reasonable
```

This would give even better negative sampling but slower training.

# Diagnostics and Ablation Tools Guide

## Overview

Comprehensive diagnostic tools and ablation study framework for Phase 4 GAN training.

## Diagnostic Tools

### 1. Training Monitor (`scripts/diagnostics/monitor_training.py`)

**Purpose**: Real-time training health monitoring

**Features**:
- Gradient norm tracking (detect exploding/vanishing gradients)
- Parameter norm tracking
- GAN health checks (mode collapse, discriminator dominance)
- Speaker similarity metrics
- Loss trend analysis

**Usage in training**:
```python
from scripts.diagnostics.monitor_training import generate_diagnostics_report

# Run every N batches
metrics, status = generate_diagnostics_report(
    model=model, mpd=mpd, mrd=mrd,
    batch=batch, device=device,
    losses={'gen': g_loss, 'disc': d_loss, ...},
    prev_losses=prev_losses,
    step=current_step,
    output_dir="diagnostics/"
)

# Check status
if status == 'critical':
    print("⚠️ Training health issues detected!")
```

**Health indicators**:
- ✅ **Healthy**: All metrics normal
- ⚠️ **Warning**: Minor issues (e.g., slight imbalance)
- ❌ **Critical**: Severe issues (mode collapse, extreme imbalance)

### 2. Embedding Analyzer (`scripts/diagnostics/analyze_embeddings.py`)

**Purpose**: Analyze speaker embedding quality

**Features**:
- Same-speaker vs different-speaker similarity distribution
- Clustering metrics (silhouette score, Davies-Bouldin index)
- 2D visualization (PCA/t-SNE)
- Embedding space analysis

**Usage**:
```python
from scripts.diagnostics.analyze_embeddings import analyze_embeddings

results = analyze_embeddings(
    model=model,
    dataloader=val_loader,
    device=device,
    output_dir="embeddings_analysis/",
    max_samples=1000,
    visualize=True
)

# Key metrics:
print(f"Same-speaker similarity: {results['same_speaker_similarity']['mean']:.4f}")
print(f"Different-speaker similarity: {results['different_speaker_similarity']['mean']:.4f}")
print(f"Separation: {results['separation']:.4f}")  # Should be > 0.3
```

**Expected results**:
- Same-speaker similarity: > 0.7
- Different-speaker similarity: < 0.3
- Separation: > 0.4 (good), > 0.6 (excellent)

### 3. Sample Generator (`scripts/diagnostics/generate_samples.py`)

**Purpose**: Generate audio samples for manual inspection

**Features**:
- Reconstruction quality (input vs output)
- Voice conversion (content A + voice B)
- Multi-speaker synthesis (same content, different voices)
- Long-form generation (temporal coherence)

**Usage**:
```python
from scripts.diagnostics.generate_samples import generate_evaluation_samples

# Generate evaluation samples
generate_evaluation_samples(
    model=model,
    dataset=val_dataset,
    device=device,
    output_dir="samples/epoch_10/",
    num_samples=10
)

# Each sample contains:
# - original.wav: Input audio
# - reconstruction.wav: Model output
# - converted_to_<speaker>.wav: Voice conversion examples
# - metadata.json: Sample information
```

**Manual inspection checklist**:
- [ ] Reconstruction fidelity (artifacts, distortion)
- [ ] Voice conversion quality (speaker preserved, content preserved)
- [ ] Long-form coherence (no jitter, consistent speaker)
- [ ] Naturalness (robotic vs human-like)

## Ablation Studies

### Ablation Runner (`scripts/ablations/run_ablations.py`)

**Purpose**: Systematic ablation studies

**Available ablations**:

| Ablation | Description | Purpose |
|----------|-------------|---------|
| `baseline` | Full model | Reference |
| `no_contrastive` | Remove contrastive loss | Test speaker conditioning importance |
| `no_gan` | Remove GAN loss | Test adversarial training importance |
| `low_contrastive` | Contrastive weight 0.1 | Test sensitivity to weight |
| `high_contrastive` | Contrastive weight 1.0 | Test sensitivity to weight |
| `no_hard_negative` | Random negatives | Test hard negative mining |
| `more_negatives` | 12 negatives instead of 6 | Test negative count |
| `segment_1s` | 1-second segments | Test segment length |
| `segment_4s` | 4-second segments | Test segment length |
| `batch_4` | Batch size 4 | Test batch size |
| `batch_16` | Batch size 16 | Test batch size |

**Usage**:

```bash
# Run specific ablations (10 epochs each for quick testing)
uv run python scripts/ablations/run_ablations.py \
    --ablations baseline no_contrastive no_gan \
    --epochs 10 \
    --gpu-ids 1 3

# Run all ablations
uv run python scripts/ablations/run_ablations.py \
    --ablations baseline no_contrastive no_gan low_contrastive high_contrastive \
    --epochs 10 \
    --ddp

# Compare existing results (no training)
uv run python scripts/ablations/run_ablations.py --compare-only
```

**Output**:
- `ablations/phase4/config_<name>.json`: Ablation configs
- `ablations/phase4/logs_<name>.log`: Training logs
- `ablations/phase4/checkpoints_<name>/`: Model checkpoints
- `ablations/phase4/ablation_comparison.png`: Comparison plots

**Interpreting results**:
- Lower loss = better
- Compare to baseline to see component importance
- If `no_contrastive` ≈ `baseline`: Contrastive loss not helping
- If `no_gan` » `baseline`: GAN loss crucial

## Integration with Training

### Quick Integration

Add diagnostics to your training:

```bash
# Run training with diagnostics
CUDA_VISIBLE_DEVICES=1,2 uv run python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29500 \
    train_phase4_gan.py \
    --config configs/phase4_gan.json \
    --ddp \
    --run-diagnostics \
    --diagnostics-interval 100 \
    --sample-interval 5
```

This will:
- Run diagnostics every 100 batches
- Generate samples every 5 epochs
- Save all results to `diagnostics/` folder

### Programmatic Usage

See `scripts/diagnostics/integrate_diagnostics.py` for code examples.

## Workflow Recommendations

### During Training

1. **Every 100 batches**: Run diagnostics
   - Check GAN health
   - Monitor gradient norms
   - Track speaker similarity

2. **Every epoch**: Save metrics
   - Plot loss curves
   - Check for anomalies

3. **Every 5 epochs**: Generate samples
   - Manual inspection
   - Voice conversion tests

### After Training

1. **Run embedding analysis**
   ```bash
   uv run python scripts/diagnostics/analyze_embeddings.py
   ```

2. **Generate evaluation samples**
   ```bash
   uv run python scripts/diagnostics/generate_samples.py
   ```

3. **Compare ablations** (if run)
   ```bash
   uv run python scripts/ablations/run_ablations.py --compare-only
   ```

## Troubleshooting

### Diagnostics show "critical" health

**Possible causes**:
- Discriminator too strong → reduce learning rate
- Mode collapse → increase discriminator gradient penalty
- Loss imbalance → adjust loss weights

**Solutions**:
```python
# Reduce discriminator learning rate
config['disc_learning_rate'] = 0.00001  # was 0.00005

# Increase gradient penalty
config['lambda_gp'] = 10.0  # if using gradient penalty

# Adjust loss weights
config['lambda_adv'] = 0.5  # was 1.0
```

### Speaker similarity too low (< 0.5)

**Possible causes**:
- Speaker encoder not capturing speaker info
- Model not using speaker embeddings properly
- FiLM conditioning too weak

**Solutions**:
- Increase contrastive loss weight
- Unfreeze speaker encoder (see upcoming fixes)
- Try cross-attention conditioning

### Ablation crashes with OOM

**Solutions**:
- Reduce batch size in config
- Use gradient accumulation
- Use single GPU instead of DDP

## Next Steps

After running diagnostics and ablations, proceed to fixes:

1. **Critical Issue #4**: Fix hard negative mining (batch size too small)
2. **Critical Issue #1**: Unfreeze/fine-tune speaker encoder
3. **Issue #7**: Add automated training stability monitoring
4. **Issue #6**: Implement comprehensive evaluation metrics

See implementation guides in upcoming fixes.

#!/usr/bin/env python3
"""
Integration helper: Add speaker index support to training.

This shows how to modify train_phase4_gan.py to use semantic hard negative mining.
"""

INTEGRATION_GUIDE = """
# ============================================================================
# INTEGRATION: Semantic Hard Negative Mining with Speaker Index
# ============================================================================

## Step 1: Add imports to train_phase4_gan.py

Add near the top of the file:
```python
from snac.speaker_embed_index import SpeakerEmbeddingIndex
from snac.contrastive_loss import ContrastiveSpeakerLoss
```

## Step 2: Add CLI argument

In the argument parser section (around line 720):
```python
parser.add_argument('--speaker-index', type=str, default=None,
                   help='Path to speaker embedding index for semantic hard negative mining')
parser.add_argument('--build-speaker-index', action='store_true',
                   help='Build speaker index before training')
```

## Step 3: Load speaker index in main()

After loading the model (around line 560):
```python
# Load speaker index if provided
speaker_index = None
if args.speaker_index and Path(args.speaker_index).exists():
    print(f"Loading speaker index from {args.speaker_index}")
    speaker_index = SpeakerEmbeddingIndex.load(
        args.speaker_index,
        model_base,
        device=device
    )
    print(f"  Loaded {len(speaker_index.speaker_embeddings)} speakers")
    print(f"  Loaded {len(speaker_index.cluster_to_speakers)} clusters")

# Or build new index
elif args.build_speaker_index:
    print("Building speaker index...")
    from scripts.build_speaker_index import build_speaker_index

    index_path = "pretrained_models/speaker_index.json"
    speaker_index = build_speaker_index(
        model=model_base,
        dataset_root=config['train_data'],
        output_path=index_path,
        n_clusters=50,
        device=device
    )
```

## Step 4: Initialize contrastive loss with speaker index

Before training loop (around line 650):
```python
# Initialize contrastive loss function
contrastive_loss_fn = ContrastiveSpeakerLoss(
    margin=config.get('contrastive_margin', 0.1),
    temperature=config.get('contrastive_temperature', 0.1),
    speaker_index=speaker_index
)
```

## Step 5: Use new loss function in training

Replace the contrastive loss call (around line 335):
```python
# OLD:
loss_contrast = contrastive_speaker_loss(model_base, audio, codes, speaker_embs, config)

# NEW:
if use_contrastive:
    loss_contrast, contrast_metrics = contrastive_loss_fn(speaker_embs, batch, config)
else:
    loss_contrast = torch.tensor(0.0, device=device)
    contrast_metrics = {}
```

## Step 6: Log contrastive metrics

After computing losses (around line 370):
```python
# Log contrastive metrics
if len(contrast_metrics) > 0:
    for key, value in contrast_metrics.items():
        if isinstance(value, float):
            # Track in pbar
            pbar.set_postfix({
                **{f'contrast_{key}': value for key, value in contrast_metrics.items()}
            })
```

## Complete Example

See the updated contrastive loss section below for full implementation.
"""

# ============================================================================
# UPDATED CONTRASTIVE LOSS FUNCTION
# ============================================================================

UPDATED_CONTRASTIVE_LOSS = """
def contrastive_speaker_loss_with_index(model_base, audio, codes, speaker_embs,
                                       config, speaker_index=None, batch=None):
    \"\"\"
    Improved contrastive speaker loss with semantic hard negative mining.

    If speaker_index is provided, uses semantic clustering for negatives.
    Otherwise, falls back to similarity-based sampling.

    Args:
        model_base: SNAC model (unwrapped from DDP)
        audio: (B, 1, T) audio tensor
        codes: Encoded codes (from model.encode())
        speaker_embs: (B, D) speaker embeddings
        config: Training config
        speaker_index: Optional SpeakerEmbeddingIndex for semantic sampling
        batch: Batch dict (for getting speaker IDs)

    Returns:
        - loss: Contrastive loss scalar
        - metrics: Dict of metrics
    \"\"\"
    B = audio.shape[0]
    device = speaker_embs.device

    # If speaker index available and we have speaker IDs, use semantic sampling
    if speaker_index is not None and batch is not None:
        return _contrastive_with_semantic_negatives(
            model_base, audio, codes, speaker_embs, config, speaker_index, batch
        )
    else:
        # Fallback to original method
        return contrastive_speaker_loss(model_base, audio, codes, speaker_embs, config)


def _contrastive_with_semantic_negatives(model_base, audio, codes, speaker_embs,
                                         config, speaker_index, batch):
    \"\"\"
    Contrastive loss with semantic hard negatives from speaker index.

    This is MUCH more efficient than reconstruction-based contrastive loss:
    - No need to decode for each negative
    - Uses pre-computed speaker clusters
    - Directly optimizes embedding space
    \"\"\"
    from snac.contrastive_loss import ContrastiveSpeakerLoss

    # Initialize loss function with speaker index
    loss_fn = ContrastiveSpeakerLoss(
        margin=config.get('contrastive_margin', 0.1),
        speaker_index=speaker_index
    )

    # Compute loss directly on embeddings (no reconstruction needed!)
    loss, metrics = loss_fn(speaker_embs, batch, config)

    return loss, metrics
"""

# ============================================================================
# CONFIG UPDATES
# ============================================================================

CONFIG_UPDATES = """
# Add these to your config file (configs/phase4_gan.json):

{
  // Enable semantic hard negative mining
  "speaker_index_path": "pretrained_models/speaker_index.json",

  // Contrastive loss settings
  "contrastive_weight": 0.5,
  "contrastive_margin": 0.1,
  "contrastive_temperature": 0.1,
  "max_negatives": 6,

  // Or disable to use baseline
  "use_semantic_negatives": true
}
"""

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

USAGE_EXAMPLES = """
# 1. Build speaker index first (one-time setup)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/build_speaker_index.py \\
    --checkpoint checkpoints/phase4_gan/best_model.pt \\
    --dataset-root data/train_split \\
    --output pretrained_models/speaker_index.json \\
    --n-clusters 50

# 2. Train with semantic hard negative mining
CUDA_VISIBLE_DEVICES=1,2 uv run python -m torch.distributed.run \\
    --nproc_per_node=2 \\
    --master_port=29500 \\
    train_phase4_gan.py \\
    --config configs/phase4_gan.json \\
    --ddp \\
    --speaker-index pretrained_models/speaker_index.json

# 3. Or build index and train in one go
CUDA_VISIBLE_DEVICES=1,2 uv run python -m torch.distributed.run \\
    --nproc_per_node=2 \\
    --master_port=29500 \\
    train_phase4_gan.py \\
    --config configs/phase4_gan.json \\
    --ddp \\
    --build-speaker-index
"""

if __name__ == "__main__":
    print("="*70)
    print("Semantic Hard Negative Mining Integration Guide")
    print("="*70)
    print("\nSee INTEGRATION_GUIDE above for step-by-step instructions.")
    print("\nQuick start:")
    print("1. Build speaker index:")
    print("   uv run python scripts/build_speaker_index.py")
    print("\n2. Train with index:")
    print("   uv run python train_phase4_gan.py --speaker-index pretrained_models/speaker_index.json")
    print("="*70)

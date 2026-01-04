#!/usr/bin/env python3
"""
Build speaker embedding index for semantic hard negative mining.

This script:
1. Loads the pretrained SNAC model
2. Extracts speaker embeddings from all training data
3. Clusters speakers by similarity
4. Saves the index for use during training

Usage:
    uv run python scripts/build_speaker_index.py
"""

import sys
import torch
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from snac import SNACWithSpeakerConditioning
from snac.speaker_embed_index import build_speaker_index


def main():
    parser = argparse.ArgumentParser(description="Build speaker embedding index")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/phase4_gan/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--dataset-root", type=str,
                        default="data/train_split",
                        help="Path to training data")
    parser.add_argument("--output", type=str,
                        default="pretrained_models/speaker_index.json",
                        help="Output path for speaker index")
    parser.add_argument("--n-clusters", type=int, default=50,
                        help="Number of speaker clusters")
    parser.add_argument("--max-samples", type=int, default=50,
                        help="Max samples per speaker for averaging")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")

    args = parser.parse_args()

    print("="*70)
    print("Building Speaker Embedding Index")
    print("="*70)
    print(f"Model: {args.checkpoint}")
    print(f"Dataset: {args.dataset_root}")
    print(f"Output: {args.output}")
    print(f"Clusters: {args.n_clusters}")
    print("="*70)

    # Load model
    print("\n[1/3] Loading model...")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

        # Check if it's a DDP checkpoint
        if list(checkpoint['model_state_dict'].keys())[0].startswith('module.'):
            # Remove 'module.' prefix
            model_state = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        else:
            model_state = checkpoint['model_state_dict']

        # Get model config
        if 'config' in checkpoint:
            config = checkpoint['config']
            model = SNACWithSpeakerConditioning(
                sampling_rate=config.get('sampling_rate', 24000),
                encoder_dim=config.get('encoder_dim', 64),
                encoder_rates=config.get('encoder_rates', [3, 3, 7, 7]),
                decoder_dim=config.get('decoder_dim', 1536),
                decoder_rates=config.get('decoder_rates', [7, 7, 3, 3]),
                attn_window_size=config.get('attn_window_size', 32),
                codebook_size=config.get('codebook_size', 4096),
                codebook_dim=config.get('codebook_dim', 8),
                vq_strides=config.get('vq_strides', [8, 4, 2, 1]),
                speaker_emb_dim=config.get('speaker_emb_dim', 512),
                speaker_encoder_type=config.get('speaker_encoder_type', 'eres2net'),
            ).to(device)
        else:
            # Load with default config
            model = SNACWithSpeakerConditioning.from_pretrained_base(
                repo_id="hubertsiuzdak/snac_24khz",
                speaker_emb_dim=512,
                speaker_encoder_type='eres2net',
            ).to(device)

        model.load_state_dict(model_state)
        model.eval()

        print(f"✅ Model loaded from {args.checkpoint}")

    except FileNotFoundError:
        print(f"⚠️  Checkpoint not found: {args.checkpoint}")
        print("Loading pretrained model from HuggingFace instead...")

        model = SNACWithSpeakerConditioning.from_pretrained_base(
            repo_id="hubertsiuzdak/snac_24khz",
            speaker_emb_dim=512,
            speaker_encoder_type='eres2net',
        ).to(device)
        model.eval()

        print("✅ Loaded pretrained model")

    # Build index
    print("\n[2/3] Building speaker index...")
    print("This may take a while...")

    index = build_speaker_index(
        model=model,
        dataset_root=args.dataset_root,
        output_path=args.output,
        n_clusters=args.n_clusters,
        device=device,
        max_samples_per_speaker=args.max_samples
    )

    # Summary
    print("\n[3/3] Summary:")
    print(f"  Total speakers: {len(index.speaker_embeddings)}")
    print(f"  Total clusters: {len(index.cluster_to_speakers)}")

    # Show cluster distribution
    print("\n  Cluster distribution:")
    for cluster_id in sorted(index.cluster_to_speakers.keys())[:10]:
        n_speakers = len(index.cluster_to_speakers[cluster_id])
        print(f"    Cluster {cluster_id}: {n_speakers} speakers")

    if len(index.cluster_to_speakers) > 10:
        print(f"    ... and {len(index.cluster_to_speakers) - 10} more clusters")

    print("\n" + "="*70)
    print(f"✅ Speaker index saved to: {args.output}")
    print("="*70)

    # Show usage
    print("\nTo use this index in training:")
    print("  1. Add to config:")
    print('     "speaker_index_path": "pretrained_models/speaker_index.json"')
    print("  2. Training script will automatically load it")
    print("\nOr load manually:")
    print("  from snac.speaker_embed_index import SpeakerEmbeddingIndex")
    print("  index = SpeakerEmbeddingIndex.load(")
    print('      "pretrained_models/speaker_index.json",')
    print("      model=model")
    print("  )")


if __name__ == "__main__":
    main()

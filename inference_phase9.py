#!/usr/bin/env python3
"""
Phase 9 Inference: Evaluate reconstruction quality on custom data.

Tests L1 and STFT reconstruction losses on audio files from specified directories.
Phase 9 is simple SNAC fine-tuning (no speaker conditioning, no adapters).

Compares original SNAC model vs fine-tuned checkpoint.
"""

import json
import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from snac import SNAC


def reconstruction_loss(audio, audio_hat, n_ffts=[1024, 2048, 4096]):
    """Calculate L1 and multi-scale STFT loss."""
    # L1 loss
    loss_l1 = F.l1_loss(audio_hat, audio)

    # Multi-scale STFT loss
    loss_stft = 0.0
    for n_fft in n_ffts:
        audio_stft = torch.stft(audio.squeeze(1), n_fft=n_fft, return_complex=True)
        audio_hat_stft = torch.stft(audio_hat.squeeze(1), n_fft=n_fft, return_complex=True)
        loss_stft += F.l1_loss(audio_hat_stft.abs(), audio_stft.abs())

    loss_stft = loss_stft / len(n_ffts)

    # Combined loss
    loss = loss_l1 + loss_stft

    return loss, loss_l1, loss_stft


def load_model(checkpoint_path, device):
    """Load Phase 9 fine-tuned model."""
    print(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})

    # Load pretrained model
    model = SNAC.from_pretrained(config.get('pretrained_model', 'hubertsiuzdak/snac_24khz')).to(device)

    # Load fine-tuned weights
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print(f"✅ Checkpoint loaded (epoch {checkpoint.get('epoch', '?')})")
    return model, config


def evaluate_directory(model, directory, device, segment_length=2.0, batch_size=64,
                        num_samples=None, seed=42):
    """Evaluate reconstruction loss on audio files using batching with random sampling."""
    directory = Path(directory)

    # Find all audio files
    audio_extensions = ['.wav', '.WAV', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(directory.glob(f'*{ext}')))

    if len(audio_files) == 0:
        print(f"⚠️  No audio files found in {directory}")
        return None

    # Random sampling if num_samples is specified
    if num_samples is not None and num_samples < len(audio_files):
        random.seed(seed)
        audio_files = random.sample(audio_files, num_samples)
        print(f"\n📁 Randomly sampled {num_samples} files from {len(audio_files)} total in {directory}")
    else:
        print(f"\n📁 Evaluating {len(audio_files)} files from {directory}")

    # Collect all segments first
    print("Loading and segmenting audio files...")
    segments = []
    segment_samples = int(segment_length * 24000)
    min_samples = 12000  # 0.5 seconds minimum

    for audio_path in tqdm(audio_files, desc="Loading files"):
        try:
            # Load audio
            audio, sr = torchaudio.load(str(audio_path))

            # Convert to mono if stereo
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

            # Resample to 24kHz if needed
            if sr != 24000:
                resampler = torchaudio.transforms.Resample(sr, 24000)
                audio = resampler(audio)

            # Extract segments
            total_samples = audio.shape[-1]

            for start_idx in range(0, total_samples, segment_samples):
                end_idx = min(start_idx + segment_samples, total_samples)
                segment = audio[:, start_idx:end_idx]

                # Skip if too short (less than 0.5 seconds)
                if segment.shape[-1] < min_samples:
                    continue

                # Pad to segment length if needed
                if segment.shape[-1] < segment_samples:
                    segment = F.pad(segment, (0, segment_samples - segment.shape[-1]))

                segments.append(segment)

        except Exception as e:
            print(f"⚠️  Error loading {audio_path.name}: {e}")
            continue

    if len(segments) == 0:
        print(f"⚠️  No valid segments from {directory}")
        return None

    print(f"Processing {len(segments)} segments in batches of {batch_size}...")

    # Process in batches
    model.eval()
    total_loss = 0.0
    total_loss_l1 = 0.0
    total_loss_stft = 0.0
    num_segments = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(segments), batch_size), desc="Batch inference"):
            batch_segments = segments[i:i + batch_size]

            # Stack into batch (all segments are same length)
            batch = torch.stack(batch_segments)  # (B, 1, T)

            # Move to device
            batch = batch.to(device)

            # Forward pass
            audio_hat, codes = model(batch)

            # Calculate loss
            loss, loss_l1, loss_stft = reconstruction_loss(batch, audio_hat)

            total_loss += loss.item() * batch.size(0)
            total_loss_l1 += loss_l1.item() * batch.size(0)
            total_loss_stft += loss_stft.item() * batch.size(0)
            num_segments += batch.size(0)

    if num_segments == 0:
        print(f"⚠️  No valid segments processed from {directory}")
        return None

    # Calculate averages
    metrics = {
        'loss': total_loss / num_segments,
        'l1': total_loss_l1 / num_segments,
        'stft': total_loss_stft / num_segments,
        'num_segments': num_segments,
        'num_files': len(audio_files),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 9 Inference - Compare Base vs Fine-tuned")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/phase9_conservative/best_model.pt",
                        help="Path to fine-tuned checkpoint file")
    parser.add_argument("--data_dirs", type=str, nargs='+', required=True,
                        help="Directories containing audio files to evaluate")
    parser.add_argument("--segment_length", type=float, default=2.0,
                        help="Segment length in seconds for evaluation")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference (larger = faster GPU utilization)")
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="Number of random files to sample (reproducible)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--device", type=str, default="cuda:3",
                        help="Device to use")
    parser.add_argument("--n_ffts", type=int, nargs='+', default=[1024, 2048, 4096],
                        help="STFT window sizes for loss calculation")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")
    print(f"Number of samples per directory: {args.num_samples}")

    # Load fine-tuned model
    print("\n" + "="*70)
    print("Loading Fine-tuned Model")
    print("="*70)
    model_ft, config = load_model(args.checkpoint, device)
    n_ffts = config.get('n_ffts', args.n_ffts)
    print(f"Using n_ffts={n_ffts} for STFT loss")

    # Load base pretrained model for comparison
    print("\n" + "="*70)
    print("Loading Base Pretrained Model")
    print("="*70)
    model_name = config.get('pretrained_model', 'hubertsiuzdak/snac_24khz')
    print(f"Loading from: {model_name}")
    model_base = SNAC.from_pretrained(model_name).to(device)
    model_base.eval()
    print("✅ Base model loaded")

    # Evaluate both models on each directory
    all_results = {}

    for data_dir in args.data_dirs:
        dir_name = Path(data_dir).name
        print("\n" + "="*70)
        print(f"Evaluating: {dir_name}")
        print("="*70)

        # Evaluate base model
        print("\n🔵 BASE MODEL (pretrained)")
        metrics_base = evaluate_directory(
            model_base, data_dir, device,
            segment_length=args.segment_length,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            seed=args.seed
        )

        # Evaluate fine-tuned model
        print("\n🟢 FINE-TUNED MODEL")
        metrics_ft = evaluate_directory(
            model_ft, data_dir, device,
            segment_length=args.segment_length,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            seed=args.seed
        )

        if metrics_base and metrics_ft:
            # Calculate improvement
            improvement = {
                'loss': metrics_base['loss'] - metrics_ft['loss'],
                'l1': metrics_base['l1'] - metrics_ft['l1'],
                'stft': metrics_base['stft'] - metrics_ft['stft'],
            }

            all_results[dir_name] = {
                'base': metrics_base,
                'finetuned': metrics_ft,
                'improvement': improvement
            }

    # Print comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    for dir_name, results in all_results.items():
        base = results['base']
        ft = results['finetuned']
        imp = results['improvement']

        print(f"\n📊 {dir_name}:")
        print(f"  Samples: {base['num_files']} files, {base['num_segments']} segments")
        print(f"\n  📦 Loss (lower is better):")
        print(f"     Base:       {base['loss']:.4f}")
        print(f"     Fine-tuned: {ft['loss']:.4f}")
        print(f"     Δ:          {imp['loss']:+.4f} ({imp['loss']/base['loss']*100:+.1f}%)")
        print(f"\n  📏 L1 Loss:")
        print(f"     Base:       {base['l1']:.4f}")
        print(f"     Fine-tuned: {ft['l1']:.4f}")
        print(f"     Δ:          {imp['l1']:+.4f} ({imp['l1']/base['l1']*100:+.1f}%)")
        print(f"\n  🌊 STFT Loss:")
        print(f"     Base:       {base['stft']:.4f}")
        print(f"     Fine-tuned: {ft['stft']:.4f}")
        print(f"     Δ:          {imp['stft']:+.4f} ({imp['stft']/base['stft']*100:+.1f}%)")

    # Overall averages
    if all_results:
        print("\n" + "-"*70)
        print("OVERALL AVERAGES (across all directories):")

        avg_base_loss = sum(r['base']['loss'] for r in all_results.values()) / len(all_results)
        avg_ft_loss = sum(r['finetuned']['loss'] for r in all_results.values()) / len(all_results)
        avg_imp_loss = (avg_base_loss - avg_ft_loss) / avg_base_loss * 100

        avg_base_l1 = sum(r['base']['l1'] for r in all_results.values()) / len(all_results)
        avg_ft_l1 = sum(r['finetuned']['l1'] for r in all_results.values()) / len(all_results)
        avg_imp_l1 = (avg_base_l1 - avg_ft_l1) / avg_base_l1 * 100

        avg_base_stft = sum(r['base']['stft'] for r in all_results.values()) / len(all_results)
        avg_ft_stft = sum(r['finetuned']['stft'] for r in all_results.values()) / len(all_results)
        avg_imp_stft = (avg_base_stft - avg_ft_stft) / avg_base_stft * 100

        print(f"\n  📦 Loss:")
        print(f"     Base:       {avg_base_loss:.4f}")
        print(f"     Fine-tuned: {avg_ft_loss:.4f}")
        print(f"     Δ:          {avg_imp_loss:+.1f}%")
        print(f"\n  📏 L1:")
        print(f"     Base:       {avg_base_l1:.4f}")
        print(f"     Fine-tuned: {avg_ft_l1:.4f}")
        print(f"     Δ:          {avg_imp_l1:+.1f}%")
        print(f"\n  🌊 STFT:")
        print(f"     Base:       {avg_base_stft:.4f}")
        print(f"     Fine-tuned: {avg_ft_stft:.4f}")
        print(f"     Δ:          {avg_imp_stft:+.1f}%")

    print("="*70 + "\n")

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ Results saved to: {output_path}")


if __name__ == "__main__":
    main()

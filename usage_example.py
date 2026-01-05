#!/usr/bin/env python3
"""
Simple script to use Phase 9 fine-tuned SNAC model for audio reconstruction.

Usage:
    # Reconstruct single file with fine-tuned model
    python usage_example.py --input input.wav --output output.wav

    # Compare base vs fine-tuned
    python usage_example.py --input input.wav --compare

    # Batch process directory
    python usage_example.py --input_dir audio_files/ --output_dir reconstructed/
"""

import argparse
from pathlib import Path

import torch
import torchaudio

from snac import SNAC


def load_model(checkpoint_path=None, device="cuda"):
    """Load SNAC model (fine-tuned or base)."""
    if checkpoint_path and Path(checkpoint_path).exists():
        # Load fine-tuned checkpoint
        print(f"Loading fine-tuned checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get('config', {})

        model = SNAC.from_pretrained(
            config.get('pretrained_model', 'hubertsiuzdak/snac_24khz')
        ).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        print(f"✅ Loaded fine-tuned model (epoch {checkpoint.get('epoch', '?')})")
    else:
        # Load base pretrained
        print("Loading base pretrained SNAC model")
        model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
        model.eval()
        print("✅ Loaded base model")

    return model


def load_audio(audio_path, target_sr=24000):
    """Load and preprocess audio."""
    audio, sr = torchaudio.load(str(audio_path))

    # Convert to mono
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)

    return audio, sr


def reconstruct_audio(model, audio, device):
    """Reconstruct audio using SNAC model."""
    # Add batch dimension: (1, 1, T)
    if audio.dim() == 2:
        audio = audio.unsqueeze(0)

    audio = audio.to(device)

    with torch.no_grad():
        audio_recon, codes = model(audio)

    return audio_recon.squeeze(0).cpu()


def main():
    parser = argparse.ArgumentParser(description="Reconstruct audio with SNAC")
    parser.add_argument("--input", type=str, help="Input audio file")
    parser.add_argument("--output", type=str, help="Output audio file")
    parser.add_argument("--input_dir", type=str, help="Input directory")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/phase9_conservative/best_model.pt",
                        help="Path to fine-tuned checkpoint (empty for base model)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare base vs fine-tuned (generates two outputs)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    if args.compare:
        # Compare both models
        print("="*70)
        print("COMPARISON MODE: Base SNAC vs Fine-tuned")
        print("="*70 + "\n")

        model_base = load_model(None, device)  # Load base
        model_ft = load_model(args.checkpoint, device)  # Load fine-tuned

        audio, orig_sr = load_audio(args.input)

        print(f"Reconstructing with base model...")
        recon_base = reconstruct_audio(model_base, audio, device)

        print(f"Reconstructing with fine-tuned model...")
        recon_ft = reconstruct_audio(model_ft, audio, device)

        # Save both
        input_path = Path(args.input)
        base_output = input_path.stem + "_base_snac.wav"
        ft_output = input_path.stem + "_finetuned_snac.wav"

        torchaudio.save(base_output, recon_base, 24000)
        print(f"✅ Saved: {base_output}")

        torchaudio.save(ft_output, recon_ft, 24000)
        print(f"✅ Saved: {ft_output}")

    elif args.input_dir:
        # Batch process directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model = load_model(args.checkpoint, device)

        input_dir = Path(args.input_dir)
        audio_files = list(input_dir.glob("*.wav")) + list(input_dir.glob("*.mp3"))

        print(f"Found {len(audio_files)} files")
        print(f"Processing...\n")

        for audio_path in audio_files:
            try:
                print(f"  {audio_path.name}...")
                audio, _ = load_audio(audio_path)
                recon = reconstruct_audio(model, audio, device)

                output_path = output_dir / audio_path.name
                torchaudio.save(str(output_path), recon, 24000)
                print(f"  ✅ Saved to: {output_path}")
            except Exception as e:
                print(f"  ⚠️  Error: {e}")

    else:
        # Single file reconstruction
        model = load_model(args.checkpoint, device)

        audio, orig_sr = load_audio(args.input)
        print(f"Input: {args.input}")
        print(f"  Original sample rate: {orig_sr} Hz")
        print(f"  Duration: {audio.shape[-1] / 24000:.2f} seconds")
        print(f"  Shape: {audio.shape}")

        print(f"\nReconstructing...")
        recon = reconstruct_audio(model, audio, device)

        output = args.output or Path(args.input).stem + "_reconstructed.wav"
        torchaudio.save(output, recon, 24000)
        print(f"✅ Saved to: {output}")


if __name__ == "__main__":
    main()

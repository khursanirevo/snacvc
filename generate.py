#!/usr/bin/env python3
"""
Generate audio samples comparing original vs base SNAC vs fine-tuned SNAC.

Creates side-by-side audio files for listening comparison.
"""

import json
import random
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

from snac import SNAC


def load_model(model_name_or_path, device):
    """Load SNAC model from pretrained or checkpoint."""
    if Path(model_name_or_path).exists():
        # Load checkpoint
        print(f"Loading checkpoint from {model_name_or_path}")
        checkpoint = torch.load(model_name_or_path, map_location=device)
        config = checkpoint.get('config', {})

        model = SNAC.from_pretrained(config.get('pretrained_model', 'hubertsiuzdak/snac_24khz')).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        print(f"✅ Checkpoint loaded (epoch {checkpoint.get('epoch', '?')})")
        return model
    else:
        # Load pretrained
        print(f"Loading pretrained model: {model_name_or_path}")
        model = SNAC.from_pretrained(model_name_or_path).to(device)
        model.eval()
        print("✅ Pretrained model loaded")
        return model


def reconstruct_audio(model, audio_path, device, segment_length=2.0):
    """Reconstruct audio using model, handling long files by segmenting."""
    # Load audio
    audio, sr = torchaudio.load(str(audio_path))

    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Resample to 24kHz if needed
    if sr != 24000:
        resampler = torchaudio.transforms.Resample(sr, 24000)
        audio = resampler(audio)

    # Limit to first segment_length seconds for comparison
    max_samples = int(segment_length * 24000)
    audio = audio[:, :max_samples]

    # Add batch dimension
    audio = audio.unsqueeze(0)  # (1, 1, T)

    # Move to device
    audio = audio.to(device)

    # Reconstruct
    with torch.no_grad():
        audio_hat, codes = model(audio)

    return audio.squeeze(0).cpu(), audio.squeeze(0).cpu()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate audio reconstructions for comparison")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/phase10_curriculum/best_model.pt",
                        help="Path to fine-tuned checkpoint")
    parser.add_argument("--data_dirs", type=str, nargs='+', required=True,
                        help="Directories containing audio files")
    parser.add_argument("--output_dir", type=str, default="comparison_samples",
                        help="Output directory for audio files")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples per directory")
    parser.add_argument("--segment_length", type=float, default=2.0,
                        help="Segment length in seconds")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda:3",
                        help="Device to use")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")

    # Set random seed
    random.seed(args.seed)

    # Load models
    print("\n" + "="*70)
    print("Loading Models")
    print("="*70)

    # Load fine-tuned model first to get the base model name
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get('config', {})
    base_model_name = config.get('pretrained_model', 'hubertsiuzdak/snac_24khz')

    model_base = SNAC.from_pretrained(base_model_name).to(device)
    model_base.eval()
    print("✅ Base model loaded")

    model_ft = SNAC.from_pretrained(base_model_name).to(device)
    model_ft.load_state_dict(checkpoint['model'])
    model_ft.eval()
    print(f"✅ Fine-tuned model loaded (epoch {checkpoint.get('epoch', '?')})")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each directory
    for data_dir in args.data_dirs:
        data_dir = Path(data_dir)
        dir_name = data_dir.name

        print("\n" + "="*70)
        print(f"Processing: {dir_name}")
        print("="*70)

        # Find audio files
        audio_extensions = ['.wav', '.WAV', '.mp3', '.flac', '.ogg', '.m4a']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(list(data_dir.glob(f'*{ext}')))

        if len(audio_files) == 0:
            print(f"⚠️  No audio files found in {data_dir}")
            continue

        # Random sampling
        if args.num_samples < len(audio_files):
            selected_files = random.sample(audio_files, args.num_samples)
        else:
            selected_files = audio_files

        print(f"Selected {len(selected_files)} files")

        # Create subdir for this dataset
        dataset_output = output_dir / dir_name
        dataset_output.mkdir(parents=True, exist_ok=True)

        # Process each file
        for i, audio_path in enumerate(tqdm(selected_files, desc="Generating")):
            try:
                # Load original
                audio_orig, sr = torchaudio.load(str(audio_path))
                if audio_orig.shape[0] > 1:
                    audio_orig = audio_orig.mean(dim=0, keepdim=True)
                if sr != 24000:
                    resampler = torchaudio.transforms.Resample(sr, 24000)
                    audio_orig = resampler(audio_orig)

                # Trim to segment_length
                max_samples = int(args.segment_length * 24000)
                audio_orig = audio_orig[:, :max_samples]

                # Save original
                orig_path = dataset_output / f"{i:03d}_original.wav"
                torchaudio.save(str(orig_path), audio_orig, 24000)

                # Reconstruct with base model
                audio_base, _ = reconstruct_audio(model_base, audio_path, device, args.segment_length)
                base_path = dataset_output / f"{i:03d}_base_snac.wav"
                torchaudio.save(str(base_path), audio_base, 24000)

                # Reconstruct with fine-tuned model
                audio_ft, _ = reconstruct_audio(model_ft, audio_path, device, args.segment_length)
                ft_path = dataset_output / f"{i:03d}_finetuned_snac.wav"
                torchaudio.save(str(ft_path), audio_ft, 24000)

            except Exception as e:
                print(f"⚠️  Error processing {audio_path.name}: {e}")
                continue

    print("\n" + "="*70)
    print(f"✅ Generated samples saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

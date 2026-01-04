"""
Inference script for speaker manipulation with SNAC.

This script allows you to:
1. Encode content audio from one source
2. Decode it with speaker characteristics from another source
3. Save the output audio

Example usage:
    python inference.py \
        --checkpoint checkpoints/best.pt \
        --content content.wav \
        --speaker target_speaker.wav \
        --output output.wav
"""

import argparse
from pathlib import Path

import torch
import torchaudio
from snac import SNACWithSpeakerConditioning


def load_audio(audio_path, target_sample_rate):
    """Load and preprocess audio file."""
    # Load audio
    audio, sr = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Resample if necessary
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)

    return audio  # (1, T)


def manipulate_speaker(
    model,
    content_audio_path,
    reference_speaker_path,
    output_path,
    device='cuda',
):
    """
    Manipulate speaker characteristics while preserving content.

    Args:
        model: SNACWithSpeakerConditioning instance
        content_audio_path: Path to audio with desired content
        reference_speaker_path: Path to audio with desired speaker
        output_path: Path to save output audio
        device: Device to run inference on
    """
    model.eval()

    print(f"Loading content audio from: {content_audio_path}")
    content_audio = load_audio(content_audio_path, model.sampling_rate)
    content_audio = content_audio.to(device)
    print(f"  Content audio shape: {content_audio.shape}")

    print(f"Loading reference speaker audio from: {reference_speaker_path}")
    ref_audio = load_audio(reference_speaker_path, model.sampling_rate)
    ref_audio = ref_audio.to(device)
    print(f"  Reference audio shape: {ref_audio.shape}")

    print("Encoding content...")
    with torch.no_grad():
        # Encode content
        codes = model.encode(content_audio)
        print(f"  Encoded to {len(codes)} codebooks")
        for i, code in enumerate(codes):
            print(f"    Codebook {i}: {code.shape}")

    print("Decoding with reference speaker characteristics...")
    with torch.no_grad():
        # Decode with reference speaker
        audio_output = model.decode(codes, reference_audio=ref_audio)

    print(f"Output audio shape: {audio_output.shape}")

    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_output = audio_output.squeeze(0).cpu()
    torchaudio.save(str(output_path), audio_output, model.sampling_rate)

    print(f"Saved speaker-manipulated audio to: {output_path}")


def reconstruct_speaker(
    model,
    audio_path,
    output_path,
    device='cuda',
):
    """
    Reconstruct audio with speaker conditioning (for testing).

    This encodes and decodes the same audio, which should give high quality
    reconstruction if the model is trained properly.

    Args:
        model: SNACWithSpeakerConditioning instance
        audio_path: Path to input audio
        output_path: Path to save output audio
        device: Device to run inference on
    """
    model.eval()

    print(f"Loading audio from: {audio_path}")
    audio = load_audio(audio_path, model.sampling_rate)
    audio = audio.to(device)
    print(f"  Audio shape: {audio.shape}")

    print("Reconstructing with speaker conditioning...")
    with torch.no_grad():
        # Encode and decode with same speaker
        audio_output, _ = model(audio, reference_audio=audio)

    print(f"Output audio shape: {audio_output.shape}")

    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_output = audio_output.squeeze(0).cpu()
    torchaudio.save(str(output_path), audio_output, model.sampling_rate)

    print(f"Saved reconstructed audio to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Speaker manipulation with SNAC')

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--pretrained', type=str, default='hubertsiuzdak/snac_24khz',
                        help='Pretrained SNAC model (used if checkpoint only contains FiLM weights)')

    # Mode selection
    parser.add_argument('--mode', type=str, choices=['manipulate', 'reconstruct'], default='manipulate',
                        help='Operation mode: manipulate (change speaker) or reconstruct (same speaker)')

    # Audio arguments
    parser.add_argument('--content', type=str,
                        help='Path to content audio (for --mode manipulate)')
    parser.add_argument('--speaker', type=str,
                        help='Path to reference speaker audio (for --mode manipulate)')
    parser.add_argument('--input', type=str,
                        help='Path to input audio (for --mode reconstruct)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save output audio')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint}")

    # First create model from pretrained
    model = SNACWithSpeakerConditioning.from_pretrained_base(
        repo_id=args.pretrained,
        speaker_emb_dim=512,
        freeze_base=True,
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Model loaded successfully")
    print(f"  Sampling rate: {model.sampling_rate} Hz")
    print(f"  Speaker embedding dim: {model.speaker_encoder.embedding_dim}")

    # Run based on mode
    if args.mode == 'manipulate':
        if not args.content or not args.speaker:
            parser.error("--content and --speaker are required for --mode manipulate")

        manipulate_speaker(
            model=model,
            content_audio_path=args.content,
            reference_speaker_path=args.speaker,
            output_path=args.output,
            device=device,
        )
    elif args.mode == 'reconstruct':
        if not args.input:
            parser.error("--input is required for --mode reconstruct")

        reconstruct_speaker(
            model=model,
            audio_path=args.input,
            output_path=args.output,
            device=device,
        )

    print("Done!")


if __name__ == '__main__':
    main()

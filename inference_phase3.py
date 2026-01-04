"""
Inference script for Phase 3 SNAC with contrastive speaker conditioning.

Phase 3 models are trained with contrastive loss, which makes speaker conditioning
more effective by teaching the model to distinguish between different speakers.

This script allows you to:
1. Encode content audio from one source
2. Decode it with speaker characteristics from another source (with better speaker disentanglement)
3. Save the output audio
4. Test speaker similarity metrics

Example usage:
    python inference_phase3.py \
        --checkpoint checkpoints/phase3_contrastive/best.pt \
        --content content.wav \
        --speaker target_speaker.wav \
        --output output.wav

    # Batch processing for multiple speakers
    python inference_phase3.py \
        --checkpoint checkpoints/phase3_contrastive/best.pt \
        --content content.wav \
        --speaker_dir speakers/ \
        --output_dir results/

    # Speaker similarity test
    python inference_phase3.py \
        --checkpoint checkpoints/phase3_contrastive/best.pt \
        --mode similarity \
        --audio1 speaker1.wav \
        --audio2 speaker2.wav
"""

import argparse
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from snac import SNACWithSpeakerConditioning


def load_audio(audio_path, target_sample_rate):
    """Load and preprocess audio file."""
    audio, sr = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Resample if necessary
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)

    return audio  # (1, T)


def compute_speaker_similarity(model, audio1_path, audio2_path, device='cuda'):
    """
    Compute cosine similarity between speaker embeddings of two audio files.

    This is useful for:
    - Verifying speaker identity
    - Finding similar speakers
    - Checking speaker disentanglement quality

    Args:
        model: SNACWithSpeakerConditioning instance
        audio1_path: Path to first audio file
        audio2_path: Path to second audio file
        device: Device to run inference on

    Returns:
        float: Cosine similarity score (-1 to 1, higher = more similar)
    """
    model.eval()

    print(f"Computing speaker similarity...")
    print(f"  Audio 1: {audio1_path}")
    print(f"  Audio 2: {audio2_path}")

    # Load audios
    audio1 = load_audio(audio1_path, model.sampling_rate).to(device)
    audio2 = load_audio(audio2_path, model.sampling_rate).to(device)

    # Extract speaker embeddings
    with torch.no_grad():
        emb1 = model.extract_speaker_embedding(audio1)  # (1, emb_dim)
        emb2 = model.extract_speaker_embedding(audio2)  # (1, emb_dim)

    # Compute cosine similarity
    similarity = F.cosine_similarity(emb1, emb2, dim=-1).item()

    print(f"  Speaker similarity: {similarity:.4f}")
    print(f"  Interpretation:")
    if similarity > 0.85:
        print(f"    -> Very high similarity: Likely the SAME speaker")
    elif similarity > 0.7:
        print(f"    -> High similarity: Very similar speakers")
    elif similarity > 0.5:
        print(f"    -> Moderate similarity: Some characteristics in common")
    else:
        print(f"    -> Low similarity: Different speakers")

    return similarity


def manipulate_speaker(
    model,
    content_audio_path,
    reference_speaker_path,
    output_path,
    device='cuda',
    return_similarity=False,
):
    """
    Manipulate speaker characteristics while preserving content.
    Uses contrastive-trained model for better speaker disentanglement.

    Args:
        model: SNACWithSpeakerConditioning instance
        content_audio_path: Path to audio with desired content
        reference_speaker_path: Path to audio with desired speaker
        output_path: Path to save output audio
        device: Device to run inference on
        return_similarity: If True, also compute speaker similarities

    Returns:
        dict: Optional similarity scores if return_similarity=True
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

    # Compute speaker similarities (useful for contrastive models)
    similarities = {}
    if return_similarity:
        with torch.no_grad():
            emb_content = model.extract_speaker_embedding(content_audio)
            emb_ref = model.extract_speaker_embedding(ref_audio)
            similarities['content_ref_similarity'] = F.cosine_similarity(
                emb_content, emb_ref, dim=-1
            ).item()
            print(f"  Content-Reference similarity: {similarities['content_ref_similarity']:.4f}")

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

    # Verify speaker characteristics (optional)
    if return_similarity:
        with torch.no_grad():
            emb_output = model.extract_speaker_embedding(audio_output)
            similarities['output_ref_similarity'] = F.cosine_similarity(
                emb_output, emb_ref, dim=-1
            ).item()
            similarities['output_content_similarity'] = F.cosine_similarity(
                emb_output, emb_content, dim=-1
            ).item()
            print(f"\nSpeaker similarity metrics:")
            print(f"  Output vs Reference: {similarities['output_ref_similarity']:.4f} (higher = better speaker transfer)")
            print(f"  Output vs Content: {similarities['output_content_similarity']:.4f} (lower = better speaker change)")

    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio_output = audio_output.squeeze(0).cpu()
    torchaudio.save(str(output_path), audio_output, model.sampling_rate)

    print(f"Saved speaker-manipulated audio to: {output_path}")

    return similarities if return_similarity else None


def batch_speaker_manipulation(
    model,
    content_audio_path,
    speaker_dir,
    output_dir,
    device='cuda',
):
    """
    Apply multiple speaker characteristics to the same content.
    Useful for testing speaker disentanglement quality.

    Args:
        model: SNACWithSpeakerConditioning instance
        content_audio_path: Path to audio with desired content
        speaker_dir: Directory containing speaker reference audio files
        output_dir: Directory to save output audio files
        device: Device to run inference on
    """
    model.eval()

    speaker_dir = Path(speaker_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files in speaker directory
    audio_extensions = ['.wav', '.WAV', '.mp3', '.flac', '.ogg', '.m4a']
    speaker_files = []
    for ext in audio_extensions:
        speaker_files.extend(list(speaker_dir.glob(f'*{ext}')))

    if len(speaker_files) == 0:
        print(f"No audio files found in {speaker_dir}")
        return

    print(f"Found {len(speaker_files)} speaker files")
    print(f"Processing content: {content_audio_path}\n")

    # Load content audio once
    content_audio = load_audio(content_audio_path, model.sampling_rate).to(device)

    # Extract content codes once
    with torch.no_grad():
        codes = model.encode(content_audio)
        print(f"Content encoded to {len(codes)} codebooks\n")

    # Process each speaker
    results = []
    for speaker_file in tqdm(speaker_files, desc="Processing speakers"):
        speaker_name = speaker_file.stem
        output_path = output_dir / f"{content_audio_path.stem}_to_{speaker_name}{speaker_file.suffix}"

        # Load speaker reference
        ref_audio = load_audio(str(speaker_file), model.sampling_rate).to(device)

        # Decode with speaker
        with torch.no_grad():
            audio_output = model.decode(codes, reference_audio=ref_audio)

        # Compute similarities
        with torch.no_grad():
            emb_content = model.extract_speaker_embedding(content_audio)
            emb_ref = model.extract_speaker_embedding(ref_audio)
            emb_output = model.extract_speaker_embedding(audio_output)

            sim_content_ref = F.cosine_similarity(emb_content, emb_ref, dim=-1).item()
            sim_output_ref = F.cosine_similarity(emb_output, emb_ref, dim=-1).item()

        # Save output
        audio_output = audio_output.squeeze(0).cpu()
        torchaudio.save(str(output_path), audio_output, model.sampling_rate)

        results.append({
            'speaker': speaker_name,
            'output': str(output_path),
            'content_ref_sim': sim_content_ref,
            'output_ref_sim': sim_output_ref,
        })

    # Print summary
    print("\n" + "="*80)
    print("BATCH SPEAKER MANIPULATION RESULTS")
    print("="*80)
    print(f"{'Speaker':<30} {'Output File':<40} {'Ref Sim':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['speaker']:<30} {Path(r['output']).name:<40} {r['output_ref_sim']:.4f}")
    print("="*80)

    print(f"\nAll outputs saved to: {output_dir}")


def reconstruct_speaker(
    model,
    audio_path,
    output_path,
    device='cuda',
):
    """
    Reconstruct audio with speaker conditioning (for testing).
    This encodes and decodes the same audio.

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
    parser = argparse.ArgumentParser(
        description='Speaker manipulation with Phase 3 contrastive-trained SNAC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic speaker transfer
  python inference_phase3.py \\
      --checkpoint checkpoints/phase3_contrastive/best.pt \\
      --content content.wav \\
      --speaker target_speaker.wav \\
      --output output.wav

  # With similarity metrics
  python inference_phase3.py \\
      --checkpoint checkpoints/phase3_contrastive/best.pt \\
      --content content.wav \\
      --speaker target_speaker.wav \\
      --output output.wav \\
      --return_similarity

  # Batch process multiple speakers
  python inference_phase3.py \\
      --checkpoint checkpoints/phase3_contrastive/best.pt \\
      --content content.wav \\
      --speaker_dir speakers/ \\
      --output_dir results/

  # Compute speaker similarity
  python inference_phase3.py \\
      --checkpoint checkpoints/phase3_contrastive/best.pt \\
      --mode similarity \\
      --audio1 speaker1.wav \\
      --audio2 speaker2.wav

  # Reconstruction test
  python inference_phase3.py \\
      --checkpoint checkpoints/phase3_contrastive/best.pt \\
      --mode reconstruct \\
      --input input.wav \\
      --output recon.wav
        """
    )

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--pretrained', type=str, default='hubertsiuzdak/snac_24khz',
                        help='Pretrained SNAC model (used if checkpoint only contains FiLM weights)')

    # Mode selection
    parser.add_argument('--mode', type=str, choices=['manipulate', 'reconstruct', 'similarity', 'batch'],
                        default='manipulate',
                        help='Operation mode: manipulate (change speaker), reconstruct (same speaker), '
                             'similarity (compute speaker similarity), batch (process multiple speakers)')

    # Audio arguments for manipulate mode
    parser.add_argument('--content', type=str,
                        help='Path to content audio (for --mode manipulate/batch)')
    parser.add_argument('--speaker', type=str,
                        help='Path to reference speaker audio (for --mode manipulate)')
    parser.add_argument('--speaker_dir', type=str,
                        help='Directory with speaker audio files (for --mode batch)')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory for batch mode (for --mode batch)')

    # Audio arguments for reconstruct mode
    parser.add_argument('--input', type=str,
                        help='Path to input audio (for --mode reconstruct)')

    # Audio arguments for similarity mode
    parser.add_argument('--audio1', type=str,
                        help='Path to first audio (for --mode similarity)')
    parser.add_argument('--audio2', type=str,
                        help='Path to second audio (for --mode similarity)')

    # Common arguments
    parser.add_argument('--output', type=str,
                        help='Path to save output audio (for --mode manipulate/reconstruct)')
    parser.add_argument('--return_similarity', action='store_true',
                        help='Return speaker similarity metrics (for --mode manipulate)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    print(f"Loading model from checkpoint: {args.checkpoint}")

    # Load checkpoint first to detect encoder type
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Auto-detect speaker encoder type from checkpoint keys
    speaker_encoder_type = 'ecapa'  # default
    state_dict_keys = list(checkpoint['model_state_dict'].keys())

    if any('speaker_encoder.model.mods' in k for k in state_dict_keys):
        speaker_encoder_type = 'eres2net'
        print("Detected speaker encoder: ERes2NetV2")
    elif any('speaker_encoder.encoder.0.' in k for k in state_dict_keys):
        speaker_encoder_type = 'simple'
        print("Detected speaker encoder: Simple (trainable CNN)")
    elif any('speaker_encoder.model.mods.embedding_model' in k for k in state_dict_keys):
        speaker_encoder_type = 'ecapa'
        print("Detected speaker encoder: ECAPA-TDNN (SpeechBrain)")
    else:
        print(f"Could not auto-detect encoder type, using default: {speaker_encoder_type}")

    # First create model from pretrained with detected encoder type
    model = SNACWithSpeakerConditioning.from_pretrained_base(
        repo_id=args.pretrained,
        speaker_emb_dim=512,
        speaker_encoder_type=speaker_encoder_type,
        freeze_base=True,
    )

    # Load checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Model loaded successfully")
    print(f"  Sampling rate: {model.sampling_rate} Hz")
    print(f"  Speaker embedding dim: {model.speaker_encoder.embedding_dim}")

    # Print training config if available
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"\nTraining config:")
        print(f"  Phase: 3 (Contrastive)")
        print(f"  Contrastive weight: {config.get('contrastive_weight', 'N/A')}")
        print(f"  Contrastive margin: {config.get('contrastive_margin', 'N/A')}")
        print(f"  Max negatives: {config.get('max_negatives', 'N/A')}")
        print(f"  Best val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    print()

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
            return_similarity=args.return_similarity,
        )

    elif args.mode == 'batch':
        if not args.content or not args.speaker_dir or not args.output_dir:
            parser.error("--content, --speaker_dir, and --output_dir are required for --mode batch")

        batch_speaker_manipulation(
            model=model,
            content_audio_path=args.content,
            speaker_dir=args.speaker_dir,
            output_dir=args.output_dir,
            device=device,
        )

    elif args.mode == 'similarity':
        if not args.audio1 or not args.audio2:
            parser.error("--audio1 and --audio2 are required for --mode similarity")

        compute_speaker_similarity(
            model=model,
            audio1_path=args.audio1,
            audio2_path=args.audio2,
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

    print("\nDone!")


if __name__ == '__main__':
    main()

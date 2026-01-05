#!/usr/bin/env python3
"""
Phase 6 Inference: Test voice conversion with adapter BEFORE VQ.

Tests:
1. Identity reconstruction (own embedding)
2. Synthetic VC (pitch-shifted audio + own embedding)
3. Voice conversion (different target embedding)

Phase 6: Adapter modulates encoder latent BEFORE quantization
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
import argparse

from snac import SNACWithSpeakerConditioning
from snac.adapters import AdapterWrapper
from snac.audio_augmentation import PitchShifter


def load_checkpoint(checkpoint_path, device):
    """Load trained model checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint
    config = checkpoint.get('config', {})
    speaker_encoder_type = config.get('speaker_encoder_type', 'eres2net')

    # Initialize base model
    base_model = SNACWithSpeakerConditioning.from_pretrained_base(
        "hubertsiuzdak/snac_24khz",
        speaker_encoder_type=speaker_encoder_type
    ).to(device)

    # Wrap with AdapterWrapper (Phase 6)
    model = AdapterWrapper(
        base_model=base_model,
        adapter_type=config.get('adapter_type', 'film'),
        adapter_hidden_dim=config.get('adapter_hidden_dim', 512),
        adapter_num_layers=config.get('adapter_num_layers', 2),
    ).to(device)

    # Load trained weights (handle DDP if present)
    model_state = checkpoint['model_state_dict']
    if list(model_state.keys())[0].startswith('module.'):
        # Remove 'module.' prefix for non-DDP loading
        model_state = {k.replace('module.', ''): v for k, v in model_state.items()}

    model.load_state_dict(model_state)
    model.eval()

    print(f"✅ Checkpoint loaded (epoch {checkpoint.get('epoch', '?')}, step {checkpoint.get('step', '?')})")
    return model, config


def extract_embedding(model, audio, device):
    """Extract speaker embedding from audio."""
    with torch.no_grad():
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
        audio = audio.to(device)
        emb = model.extract_speaker_embedding(audio)
    return emb


def pitch_shift_audio(audio, semitones, sample_rate=24000):
    """Apply pitch shifting to audio."""
    pitch_shifter = PitchShifter(sample_rate=sample_rate)
    with torch.no_grad():
        audio_shifted = pitch_shifter(audio, semitones)
    return audio_shifted



def main():
    parser = argparse.ArgumentParser(description="Phase 6 Voice Conversion Inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase6/step_500.pt",
                        help="Path to checkpoint file")
    parser.add_argument("--audio", type=str, required=True,
                        help="Path to source audio file")
    parser.add_argument("--target", type=str, default=None,
                        help="Path to target speaker audio (for voice conversion)")
    parser.add_argument("--output_dir", type=str, default="inference_output",
                        help="Output directory for generated audio")
    parser.add_argument("--device", type=str, default="cuda:3",
                        help="Device to use")
    parser.add_argument("--pitch_shift", type=int, default=2,
                        help="Semitones to shift for synthetic VC test")

    args = parser.parse_args()

    device_str = args.device if args.device.startswith("cuda") else f"cuda:{args.device}"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, config = load_checkpoint(args.checkpoint, device)

    # Load source audio
    print(f"\n{'='*70}")
    print(f"Source audio: {args.audio}")
    audio_source, sr = torchaudio.load(args.audio)

    # Convert to mono if needed
    if audio_source.shape[0] > 1:
        audio_source = audio_source.mean(dim=0, keepdim=True)

    # Resample to 24kHz if needed
    if sr != 24000:
        resampler = torchaudio.transforms.Resample(sr, 24000)
        audio_source = resampler(audio_source)

    # Trim to 10 seconds
    max_length = 10 * 24000
    audio_source = audio_source[:, :max_length]

    # Ensure shape is (B, C, T) = (1, 1, T)
    if audio_source.dim() == 2:
        audio_source = audio_source.unsqueeze(1)  # (B, T) -> (B, 1, T)

    audio_source = audio_source.to(device)

    print(f"Audio shape: {audio_source.shape}")
    print(f"Duration: {audio_source.shape[-1] / 24000:.2f} seconds")

    # Extract source speaker embedding
    print(f"\n{'='*70}")
    print("Extracting source speaker embedding...")
    source_emb = extract_embedding(model, audio_source, device)
    print(f"Source embedding shape: {source_emb.shape}")

    # Test 1: Identity Reconstruction
    print(f"\n{'='*70}")
    print("Test 1: Identity Reconstruction (own embedding)")
    print("-" * 70)

    with torch.no_grad():
        audio_recon, codes = model(audio_source, speaker_embedding=source_emb)

    output_path = output_dir / "1_identity_reconstruction.wav"
    torchaudio.save(str(output_path), audio_recon.squeeze(0).cpu(), 24000)
    print(f"✅ Saved: {output_path}")

    # Test 2: Synthetic VC (pitch-shifted audio + own embedding)
    print(f"\n{'='*70}")
    print(f"Test 2: Synthetic Voice Conversion (pitch shift ±{args.pitch_shift} semitones)")
    print("-" * 70)

    # Pitch up
    audio_pitch_up = pitch_shift_audio(audio_source, args.pitch_shift)
    with torch.no_grad():
        audio_recon_pitch_up, _ = model(audio_pitch_up, speaker_embedding=source_emb)
        audio_recon_pitch_up = audio_recon_pitch_up[:, :, :audio_source.shape[-1]]

    output_path = output_dir / f"2_synthetic_vc_pitch_up_{args.pitch_shift}.wav"
    torchaudio.save(str(output_path), audio_recon_pitch_up.squeeze(0).cpu(), 24000)
    print(f"✅ Pitch up saved: {output_path}")

    # Pitch down
    audio_pitch_down = pitch_shift_audio(audio_source, -args.pitch_shift)
    with torch.no_grad():
        audio_recon_pitch_down, _ = model(audio_pitch_down, speaker_embedding=source_emb)
        audio_recon_pitch_down = audio_recon_pitch_down[:, :, :audio_source.shape[-1]]

    output_path = output_dir / f"2_synthetic_vc_pitch_down_{args.pitch_shift}.wav"
    torchaudio.save(str(output_path), audio_recon_pitch_down.squeeze(0).cpu(), 24000)
    print(f"✅ Pitch down saved: {output_path}")

    # Test 3: Voice Conversion (different target embedding)
    if args.target:
        print(f"\n{'='*70}")
        print(f"Test 3: Voice Conversion (target speaker)")
        print(f"Target audio: {args.target}")
        print("-" * 70)

        # Load target audio
        audio_target, sr_target = torchaudio.load(args.target)

        # Convert to mono
        if audio_target.shape[0] > 1:
            audio_target = audio_target.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr_target != 24000:
            resampler = torchaudio.transforms.Resample(sr_target, 24000)
            audio_target = resampler(audio_target)

        audio_target = audio_target[:, :max_length]

        # Ensure shape is (B, C, T)
        if audio_target.dim() == 2:
            audio_target = audio_target.unsqueeze(1)

        audio_target = audio_target.to(device)

        # Extract target speaker embedding
        print("Extracting target speaker embedding...")
        target_emb = extract_embedding(model, audio_target, device)
        print(f"Target embedding shape: {target_emb.shape}")

        # Compute speaker similarity
        similarity = torch.nn.functional.cosine_similarity(
            source_emb.cpu(), target_emb.cpu(), dim=-1
        ).item()
        print(f"Speaker similarity: {similarity:.4f}")

        # Perform voice conversion
        with torch.no_grad():
            audio_vc, _ = model(audio_source, speaker_embedding=target_emb)

        output_path = output_dir / "3_voice_conversion.wav"
        torchaudio.save(str(output_path), audio_vc.squeeze(0).cpu(), 24000)
        print(f"✅ Saved: {output_path}")

        # Also save target reference
        output_path = output_dir / "3_target_reference.wav"
        torchaudio.save(str(output_path), audio_target.squeeze(0).cpu(), 24000)
        print(f"✅ Target reference saved: {output_path}")
    else:
        print(f"\n{'='*70}")
        print("Test 3: Skipped (no --target provided)")
        print("Use --target <audio_file> to test voice conversion with different speaker")

    # Save source reference
    output_path = output_dir / "0_source_original.wav"
    torchaudio.save(str(output_path), audio_source.squeeze(0).cpu(), 24000)
    print(f"\n✅ Source reference saved: {output_path}")

    print(f"\n{'='*70}")
    print("Inference complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

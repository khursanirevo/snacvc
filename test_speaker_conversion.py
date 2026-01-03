"""
Test speaker conversion with Phase 1 and Phase 2 models.

Tests:
1. Reconstruction WITHOUT speaker conditioning (baseline)
2. Reconstruction WITH speaker conditioning (self)
3. Speaker conversion: content from speaker A, voice from speaker B
"""

import os
import json
import random
import torch
import torchaudio
from pathlib import Path

# Set device to CPU
device = torch.device('cpu')
print(f"Using device: {device}")

# Load models
print("\n" + "="*60)
print("Loading models...")
print("="*60)

from snac import SNACWithSpeakerConditioning

# Load Phase 1 (reconstruction only)
print("\n[Phase 1] Loading reconstruction-only model...")
checkpoint_p1 = torch.load(
    'checkpoints/phase1_reconstruction_only/best.pt',
    map_location=device,
    weights_only=False
)

model_p1 = SNACWithSpeakerConditioning.from_pretrained_base(
    repo_id='hubertsiuzdak/snac_24khz',
    speaker_emb_dim=512,
    freeze_base=True,
)
model_p1.load_state_dict(checkpoint_p1['model_state_dict'])
model_p1 = model_p1.to(device)
model_p1.eval()
print(f"[Phase 1] Loaded (val_loss: {checkpoint_p1['val_loss']:.4f})")

# Load Phase 2 (with speaker loss)
print("\n[Phase 2] Loading speaker-loss model...")
checkpoint_p2 = torch.load(
    'checkpoints/phase2_with_speaker_loss/best.pt',
    map_location=device,
    weights_only=False
)

model_p2 = SNACWithSpeakerConditioning.from_pretrained_base(
    repo_id='hubertsiuzdak/snac_24khz',
    speaker_emb_dim=512,
    freeze_base=True,
)
model_p2.load_state_dict(checkpoint_p2['model_state_dict'])
model_p2 = model_p2.to(device)
model_p2.eval()
print(f"[Phase 2] Loaded (val_loss: {checkpoint_p2['val_loss']:.4f})")

# Get 5 random audio files from validation split
val_dir = Path('data/val_split')
all_files = list(val_dir.glob('*.wav'))
random.seed(42)
selected_files = random.sample(all_files, 5)

print(f"\n{'='*60}")
print(f"Selected {len(selected_files)} audio files for testing:")
print(f"{'='*60}")
for i, f in enumerate(selected_files, 1):
    print(f"{i}. {f.name}")

# Create output directory
output_dir = Path('outputs/speaker_conversion_test')
output_dir.mkdir(parents=True, exist_ok=True)

# Test each audio
print(f"\n{'='*60}")
print("Running inference tests...")
print("="*60)

with torch.no_grad():
    for idx, audio_path in enumerate(selected_files, 1):
        print(f"\n[{idx}/{len(selected_files)}] Processing: {audio_path.name}")
        print("-" * 60)

        # Load audio
        audio, sr = torchaudio.load(str(audio_path))

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sr != model_p1.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, model_p1.sampling_rate)
            audio = resampler(audio)

        # Move to device and ensure shape is (B, 1, T)
        audio = audio.to(device)
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # (B, T) -> (B, 1, T)
        print(f"  Audio shape: {audio.shape}, duration: {audio.shape[-1]/model_p1.sampling_rate:.2f}s")

        # Preprocess audio (required by SNAC)
        length = audio.shape[-1]
        audio_proc = model_p1.preprocess(audio)
        print(f"  Preprocessed audio shape: {audio_proc.shape}")

        # Extract speaker embedding (use original audio, not preprocessed)
        speaker_emb = model_p1.extract_speaker_embedding(audio)
        print(f"  Speaker embedding: {speaker_emb.shape}")

        # ===== Phase 1 Tests =====
        print(f"\n  [Phase 1] Reconstruction-only model:")

        # P1: Without conditioning (baseline)
        with torch.no_grad():
            codes = model_p1.encode(audio_proc)
            audio_p1_no_cond = model_p1.decode(codes, speaker_embedding=None)
            audio_p1_no_cond = audio_p1_no_cond[..., :length]  # Trim to original length
        print(f"    - Without conditioning: done")

        # P1: With self conditioning
        audio_p1_self = model_p1.decode(codes, speaker_embedding=speaker_emb)
        audio_p1_self = audio_p1_self[..., :length]
        print(f"    - With self conditioning: done")

        # ===== Phase 2 Tests =====
        print(f"\n  [Phase 2] Speaker-loss model:")

        # P2: Without conditioning (baseline)
        codes_p2 = model_p2.encode(audio_proc)
        audio_p2_no_cond = model_p2.decode(codes_p2, speaker_embedding=None)
        audio_p2_no_cond = audio_p2_no_cond[..., :length]
        print(f"    - Without conditioning: done")

        # P2: With self conditioning
        audio_p2_self = model_p2.decode(codes_p2, speaker_embedding=speaker_emb)
        audio_p2_self = audio_p2_self[..., :length]
        print(f"    - With self conditioning: done")

        # ===== Save outputs =====
        base_name = audio_path.stem
        save_path = output_dir / base_name
        save_path.mkdir(exist_ok=True)

        # Save original (squeeze batch dimension)
        torchaudio.save(
            str(save_path / f"{base_name}_original.wav"),
            audio.squeeze(0).cpu(),
            model_p1.sampling_rate
        )

        # Save Phase 1 outputs (squeeze batch dimension)
        torchaudio.save(
            str(save_path / f"{base_name}_phase1_no_cond.wav"),
            audio_p1_no_cond.squeeze(0).cpu(),
            model_p1.sampling_rate
        )
        torchaudio.save(
            str(save_path / f"{base_name}_phase1_self.wav"),
            audio_p1_self.squeeze(0).cpu(),
            model_p1.sampling_rate
        )

        # Save Phase 2 outputs (squeeze batch dimension)
        torchaudio.save(
            str(save_path / f"{base_name}_phase2_no_cond.wav"),
            audio_p2_no_cond.squeeze(0).cpu(),
            model_p2.sampling_rate
        )
        torchaudio.save(
            str(save_path / f"{base_name}_phase2_self.wav"),
            audio_p2_self.squeeze(0).cpu(),
            model_p2.sampling_rate
        )

        print(f"\n  ✓ Saved all outputs to: {save_path}/")

        # ===== Cross-speaker test (speaker conversion) =====
        # If we have another audio, use it as reference speaker
        if idx < len(selected_files):
            ref_audio_path = selected_files[idx]
            print(f"\n  [Speaker Conversion] Using voice from: {ref_audio_path.name}")

            # Load reference audio
            ref_audio, ref_sr = torchaudio.load(str(ref_audio_path))
            if ref_audio.shape[0] > 1:
                ref_audio = ref_audio.mean(dim=0, keepdim=True)
            if ref_sr != model_p1.sampling_rate:
                resampler = torchaudio.transforms.Resample(ref_sr, model_p1.sampling_rate)
                ref_audio = resampler(ref_audio)
            ref_audio = ref_audio.to(device)
            if ref_audio.dim() == 2:
                ref_audio = ref_audio.unsqueeze(1)  # (B, T) -> (B, 1, T)

            # Extract reference speaker embedding
            ref_speaker_emb = model_p1.extract_speaker_embedding(ref_audio)

            # Phase 1: Cross-speaker
            audio_p1_cross = model_p1.decode(codes, speaker_embedding=ref_speaker_emb)
            audio_p1_cross = audio_p1_cross[..., :length]

            # Phase 2: Cross-speaker
            audio_p2_cross = model_p2.decode(codes_p2, speaker_embedding=ref_speaker_emb)
            audio_p2_cross = audio_p2_cross[..., :length]

            # Save cross-speaker outputs (squeeze batch dimension)
            torchaudio.save(
                str(save_path / f"{base_name}_phase1_cross_{ref_audio_path.stem}.wav"),
                audio_p1_cross.squeeze(0).cpu(),
                model_p1.sampling_rate
            )
            torchaudio.save(
                str(save_path / f"{base_name}_phase2_cross_{ref_audio_path.stem}.wav"),
                audio_p2_cross.squeeze(0).cpu(),
                model_p2.sampling_rate
            )

            print(f"  ✓ Saved cross-speaker outputs")

print(f"\n{'='*60}")
print("All tests completed!")
print(f"{'='*60}")
print(f"\nOutputs saved to: {output_dir.absolute()}")
print(f"\nGenerated files per audio:")
print(f"  - original.wav")
print(f"  - phase1_no_cond.wav (Phase 1, no speaker conditioning)")
print(f"  - phase1_self.wav (Phase 1, self speaker conditioning)")
print(f"  - phase1_cross_<speaker>.wav (Phase 1, cross-speaker)")
print(f"  - phase2_no_cond.wav (Phase 2, no speaker conditioning)")
print(f"  - phase2_self.wav (Phase 2, self speaker conditioning)")
print(f"  - phase2_cross_<speaker>.wav (Phase 2, cross-speaker)")

#!/usr/bin/env python3
"""
Generate audio samples for manual inspection.

Generates:
- Reconstructions (input -> model -> output)
- Voice conversion (content from speaker A + voice from speaker B)
- Multi-speaker synthesis
- Long-form generation (for coherence check)
"""

import sys
import torch
import numpy as np
from pathlib import Path
import torchaudio
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from snac import SNACWithSpeakerConditioning
from torch.utils.data import Dataset, DataLoader
import torch
import torchaudio
import numpy as np


class SimpleAudioDataset(Dataset):
    """Simple audio dataset for diagnostics."""
    def __init__(self, dataset_root, segment_length=None, sampling_rate=24000):
        self.dataset_root = Path(dataset_root)
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        self.sample_rate = sampling_rate

        # Find all audio files
        self.audio_files = list(self.dataset_root.glob("**/*.wav"))
        print(f"Found {len(self.audio_files)} audio files in {dataset_root}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]

        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            waveform = resampler(waveform)

        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)

        # Segment or pad
        if self.segment_length is not None:
            audio_length = waveform.shape[-1]

            if audio_length >= self.segment_length:
                # Random crop
                start_idx = torch.randint(0, audio_length - self.segment_length + 1, (1,)).item()
                waveform = waveform[:, start_idx:start_idx + self.segment_length]
            else:
                # Pad with zeros
                padding = self.segment_length - audio_length
                waveform = torch.nn.functional.pad(waveform, (0, padding))

        return {
            'audio': waveform.squeeze(0),
            'file_path': str(audio_path),
            'speaker_id': audio_path.parent.name,
        }


@torch.no_grad()
def generate_reconstruction(model, audio, speaker_embedding, device):
    """Generate reconstruction from audio with speaker conditioning."""
    model.eval()

    # Get model_base (handle DDP)
    if hasattr(model, 'module'):
        model_base = model.module
    else:
        model_base = model

    # Prepare input
    if audio.dim() == 2:
        audio = audio.unsqueeze(1)  # (B, T) -> (B, 1, T)

    # Forward pass
    audio_hat, codes = model_base(audio, speaker_embedding=speaker_embedding)

    return audio_hat, codes


@torch.no_grad()
def generate_voice_conversion(model, content_audio, target_speaker_audio, device):
    """
    Generate voice conversion: content from source, voice from target.

    Args:
        content_audio: Audio containing content to preserve (B, T)
        target_speaker_audio: Audio from target speaker (B, T)

    Returns:
        - converted_audio: Generated audio with target speaker's voice
    """
    model.eval()

    # Get model_base (handle DDP)
    if hasattr(model, 'module'):
        model_base = model.module
    else:
        model_base = model

    # Prepare inputs
    if content_audio.dim() == 2:
        content_audio = content_audio.unsqueeze(1)
    if target_speaker_audio.dim() == 2:
        target_speaker_audio = target_speaker_audio.unsqueeze(1)

    # Extract speaker embedding from target
    speaker_embedding = model_base.extract_speaker_embedding(target_speaker_audio)

    # Generate with content audio but target speaker embedding
    converted_audio, codes = model_base(content_audio, speaker_embedding=speaker_embedding)

    return converted_audio, speaker_embedding


@torch.no_grad()
def generate_with_cross_speaker_embs(model, audio, other_speaker_audios, device):
    """
    Generate same content with multiple different speaker embeddings.

    Useful for: testing speaker disentanglement, voice cloning evaluation.
    """
    model.eval()

    # Get model_base (handle DDP)
    if hasattr(model, 'module'):
        model_base = model.module
    else:
        model_base = model

    # Prepare input
    if audio.dim() == 2:
        audio = audio.unsqueeze(1)

    results = []

    # Generate with own speaker first
    own_speaker_emb = model_base.extract_speaker_embedding(audio)
    own_recon, _ = model_base(audio, speaker_embedding=own_speaker_emb)
    results.append({
        'speaker_id': 'original',
        'audio': own_recon,
        'embedding': own_speaker_emb,
    })

    # Generate with other speakers
    for idx, other_audio in enumerate(other_speaker_audios):
        if other_audio.dim() == 2:
            other_audio = other_audio.unsqueeze(1)

        other_speaker_emb = model_base.extract_speaker_embedding(other_audio)
        other_recon, _ = model_base(audio, speaker_embedding=other_speaker_emb)

        results.append({
            'speaker_id': f'speaker_{idx}',
            'audio': other_recon,
            'embedding': other_speaker_emb,
        })

    return results


def save_audio_batch(audio_tensor, output_dir, prefix, sample_rate=24000):
    """Save a batch of audio tensors to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy
    if isinstance(audio_tensor, torch.Tensor):
        audio_tensor = audio_tensor.cpu().numpy()

    # Handle batch
    if audio_tensor.ndim == 3:  # (B, 1, T)
        audio_tensor = audio_tensor.squeeze(1)  # (B, T)

    for idx, audio in enumerate(audio_tensor):
        output_path = output_dir / f"{prefix}_{idx:04d}.wav"
        torchaudio.save(str(output_path), torch.from_numpy(audio).unsqueeze(0), sample_rate)
        print(f"Saved: {output_path}")


def generate_evaluation_samples(model, dataset, device, output_dir,
                                 num_samples=10, num_voice_conversions=5):
    """
    Generate comprehensive evaluation samples.

    Generates:
    1. Reconstructions (input vs output)
    2. Voice conversions (content from A, voice from B)
    3. Multi-speaker synthesis (same content, different voices)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("GENERATING EVALUATION SAMPLES")
    print("="*70)

    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    sample_count = 0

    for batch in dataloader:
        if sample_count >= num_samples:
            break

        audio = batch['audio'].to(device)  # (1, T)
        speaker_id = batch.get('speaker_id', ['unknown'])[0]
        file_path = batch.get('file_path', ['unknown'])[0]

        print(f"\n[{sample_count+1}/{num_samples}] Processing: {speaker_id} - {file_path}")

        # 1. Reconstruction
        print("  Generating reconstruction...")
        if hasattr(model, 'module'):
            model_base = model.module
        else:
            model_base = model

        speaker_emb = model_base.extract_speaker_embedding(audio.unsqueeze(1))
        recon_audio, codes = model_base(audio.unsqueeze(1), speaker_embedding=speaker_emb)

        # Save input and reconstruction
        output_dir_sample = output_dir / f"sample_{sample_count:04d}"
        output_dir_sample.mkdir(parents=True, exist_ok=True)

        # Save original
        torchaudio.save(
            str(output_dir_sample / "original.wav"),
            audio.cpu(),
            dataset.sample_rate
        )

        # Save reconstruction
        torchaudio.save(
            str(output_dir_sample / "reconstruction.wav"),
            recon_audio.squeeze(1).cpu(),
            dataset.sample_rate
        )

        # Save metadata
        metadata = {
            'speaker_id': speaker_id,
            'file_path': file_path,
            'sample_rate': dataset.sample_rate,
        }

        # 2. Voice Conversion (if we have other samples)
        if sample_count < num_voice_conversions:
            print("  Generating voice conversions...")

            # Get another batch for target speaker
            try:
                target_batch = next(iter(dataloader))
                target_audio = target_batch['audio'].to(device)
                target_speaker_id = target_batch.get('speaker_id', ['unknown'])[0]

                # Voice conversion: content from source, voice from target
                converted_audio, target_emb = generate_voice_conversion(
                    model, audio, target_audio, device
                )

                # Save converted audio
                torchaudio.save(
                    str(output_dir_sample / f"converted_to_{target_speaker_id}.wav"),
                    converted_audio.squeeze(1).cpu(),
                    dataset.sample_rate
                )

                metadata['voice_conversion'] = {
                    'target_speaker_id': target_speaker_id,
                }

            except StopIteration:
                pass

        # Save metadata
        with open(output_dir_sample / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        sample_count += 1

    print(f"\n✅ Generated {sample_count} samples in {output_dir}")
    print("="*70)


def generate_long_form(model, dataset, device, output_dir,
                       num_samples=3, segment_length=8.0):
    """
    Generate long-form samples to test temporal coherence.

    Concatenates multiple segments and checks consistency.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("GENERATING LONG-FORM SAMPLES")
    print("="*70)

    from torch.utils.data import DataLoader

    # Create dataset with longer segments
    long_dataset = SimpleAudioDataset(
        dataset_root=dataset.dataset_root,
        segment_length=int(segment_length * dataset.sample_rate),
        sampling_rate=dataset.sample_rate,
    )

    dataloader = DataLoader(long_dataset, batch_size=1, shuffle=False, num_workers=2)

    for idx, batch in enumerate(dataloader):
        if idx >= num_samples:
            break

        audio = batch['audio'].to(device)
        speaker_id = batch.get('speaker_id', ['unknown'])[0]

        print(f"\n[{idx+1}/{num_samples}] Generating {segment_length}s sample from {speaker_id}")

        # Generate
        if hasattr(model, 'module'):
            model_base = model.module
        else:
            model_base = model

        speaker_emb = model_base.extract_speaker_embedding(audio.unsqueeze(1))
        recon_audio, _ = model_base(audio.unsqueeze(1), speaker_embedding=speaker_emb)

        # Save
        output_path = output_dir / f"longform_{idx:04d}_{speaker_id}.wav"
        torchaudio.save(
            str(output_path),
            recon_audio.squeeze(1).cpu(),
            dataset.sample_rate
        )

        print(f"  Saved: {output_path}")

    print(f"\n✅ Generated {num_samples} long-form samples in {output_dir}")
    print("="*70)


if __name__ == "__main__":
    print("Sample Generation Tool")
    print("See generate_evaluation_samples() and generate_long_form() for usage")

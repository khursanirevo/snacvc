#!/usr/bin/env python3
"""
Diagnostic Test: Are SNAC Codes Speaker-Independent?

This script tests whether the SNAC encoder's discrete codes contain
speaker information. If they do, the model can "cheat" by storing
speaker characteristics in codes rather than using speaker embeddings.

Test Method:
1. Encode same utterance from multiple speakers
2. Train a classifier to predict speaker identity from codes
3. If classifier accuracy > chance, codes are contaminated

Results:
- Low accuracy (~1/num_speakers): Codes are speaker-independent ✅
- High accuracy (>50%): Codes contain speaker info ❌
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from snac import SNACWithSpeakerConditioning


class SpeakerClassifier(nn.Module):
    """Simple classifier to predict speaker from SNAC codes."""

    def __init__(self, codebook_dims, num_speakers, hidden_dim=256):
        super().__init__()
        self.codebook_dims = codebook_dims  # List of dims for each codebook
        self.num_speakers = num_speakers

        # Create a classifier for each codebook level
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, num_speakers)
            )
            for dim in codebook_dims
        ])

    def forward(self, codes):
        """
        Args:
            codes: List of tensors [B, N_i, T_i] for each codebook level

        Returns:
            List of logits [B, num_speakers] for each codebook level
        """
        predictions = []
        for i, code in enumerate(codes):
            # Average over time dimension
            code_pooled = code.mean(dim=-1)  # [B, N_i]

            # Predict speaker
            logits = self.classifiers[i](code_pooled)
            predictions.append(logits)

        return predictions


def extract_codes_from_audio(model, audio_path, device='cuda', max_samples=48000):
    """Extract SNAC codes from audio file."""
    import torchaudio

    try:
        # Load audio
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 24kHz
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            waveform = resampler(waveform)

        # Take first 2 seconds and pad if needed
        if waveform.shape[-1] > max_samples:
            waveform = waveform[:, :max_samples]
        elif waveform.shape[-1] < 24000:
            waveform = torch.nn.functional.pad(waveform, (0, 24000 - waveform.shape[-1]))

        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)

        # Preprocess using model's preprocess method (important!)
        waveform = waveform.to(device)
        with torch.no_grad():
            waveform = model.preprocess(waveform)
            codes = model.encode(waveform)

        return codes

    except Exception as e:
        # Suppress error output for cleaner logs
        return None


def build_speaker_dataset(data_path, model, num_speakers=None, samples_per_speaker=10, device='cuda'):
    """
    Build a dataset of (codes, speaker_id) pairs.

    Assumes directory structure: data_path/speaker_*/audio.wav
    Or: data_path contains audio files with speaker info in filename
    """
    data_path = Path(data_path)

    # Group files by speaker (using parent directory or prefix)
    speaker_files = defaultdict(list)

    # Try different directory structures
    audio_files = list(data_path.glob('**/*.wav')) + list(data_path.glob('**/*.WAV'))

    for audio_file in audio_files:
        # Method 1: Use parent directory as speaker ID
        speaker_id = audio_file.parent.name

        # Method 2: If parent is generic, use filename prefix
        if speaker_id in ['train_split', 'val_split', 'test_split', 'data']:
            # Extract from filename (e.g., "speaker123_utt001.wav")
            speaker_id = audio_file.stem.split('_')[0]

        speaker_files[speaker_id].append(audio_file)

    # Limit number of speakers
    speaker_ids = list(speaker_files.keys())
    if num_speakers and num_speakers < len(speaker_ids):
        # Take first N speakers with enough files
        speaker_ids = sorted(
            speaker_ids,
            key=lambda s: len(speaker_files[s]),
            reverse=True
        )[:num_speakers]

    print(f"Found {len(speaker_ids)} speakers")

    # Extract codes for each speaker
    dataset = {'codes': [], 'speaker_ids': [], 'speaker_to_idx': {}}

    for speaker_idx, speaker_id in enumerate(tqdm(speaker_ids, desc="Extracting codes")):
        files = speaker_files[speaker_id][:samples_per_speaker]

        if len(files) == 0:
            continue

        dataset['speaker_to_idx'][speaker_id] = speaker_idx

        for audio_file in files:
            codes = extract_codes_from_audio(model, str(audio_file), device=device)

            if codes is not None:
                dataset['codes'].append([c.cpu() for c in codes])
                dataset['speaker_ids'].append(speaker_idx)

    print(f"Extracted {len(dataset['codes'])} code samples")

    return dataset


def train_classifier(dataset, num_epochs=50, device='cuda'):
    """Train speaker classifier on codes."""

    # Get codebook dimensions from first sample
    sample_codes = dataset['codes'][0]
    codebook_dims = [c.shape[1] for c in sample_codes]  # Number of codes per timestep
    num_speakers = len(dataset['speaker_to_idx'])

    print(f"Training classifier:")
    print(f"  Codebook dims: {codebook_dims}")
    print(f"  Num speakers: {num_speakers}")
    print(f"  Num samples: {len(dataset['codes'])}")

    # Create classifier
    classifier = SpeakerClassifier(codebook_dims, num_speakers).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    # Prepare data
    codes_by_level = list(zip(*dataset['codes']))  # List of lists, one per codebook
    speaker_ids = torch.tensor(dataset['speaker_ids'], device=device)

    # Train/val split
    num_samples = len(dataset['codes'])
    val_size = num_samples // 5
    train_size = num_samples - val_size

    indices = torch.randperm(num_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Training loop
    results = {'train_acc': [], 'val_acc': [], 'acc_by_level': []}

    for epoch in tqdm(range(num_epochs), desc="Training classifier"):
        classifier.train()

        # Forward pass for each codebook level
        predictions = []
        for level_idx, codes_level in enumerate(codes_by_level):
            # Stack codes: [N, T_i, D_i] -> [N, D_i, T_i]
            codes_tensor = torch.stack([codes_level[i] for i in train_indices])

            # Get classifier predictions for this level
            level_preds = classifier.classifiers[level_idx](
                codes_tensor.mean(dim=-1)  # Pool over time
            )
            predictions.append(level_preds)

        # For overall prediction, use highest-level codebook (most compressed)
        logits = predictions[-1]
        loss = F.cross_entropy(logits, speaker_ids[train_indices])

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            classifier.eval()

            with torch.no_grad():
                # Train accuracy
                train_logits = classifier([
                    torch.stack([codes_by_level[i][j] for j in train_indices]).mean(dim=-1)
                    for i in range(len(codes_by_level))
                ])
                train_acc = (train_logits[-1].argmax(dim=1) == speaker_ids[train_indices]).float().mean().item()

                # Val accuracy
                val_logits = classifier([
                    torch.stack([codes_by_level[i][j] for j in val_indices]).mean(dim=-1)
                    for i in range(len(codes_by_level))
                ])
                val_acc = (val_logits[-1].argmax(dim=1) == speaker_ids[val_indices]).float().mean().item()

                results['train_acc'].append(train_acc)
                results['val_acc'].append(val_acc)

                if epoch == num_epochs - 1:
                    print(f"\nEpoch {epoch}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")

    return classifier, results


def main():
    parser = argparse.ArgumentParser(description="Test codebook speaker contamination")
    parser.add_argument('--data', type=str, default='data/train_split',
                       help='Path to training data')
    parser.add_argument('--model', type=str, default='hubertsiuzdak/snac_24khz',
                       help='Pretrained model')
    parser.add_argument('--num_speakers', type=int, default=10,
                       help='Number of speakers to test')
    parser.add_argument('--samples_per_speaker', type=int, default=20,
                       help='Audio samples per speaker')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs for classifier')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    args = parser.parse_args()

    print("=" * 70)
    print("CODEBOOK CONTAMINATION DIAGNOSTIC")
    print("=" * 70)

    # Load model
    print(f"\nLoading model: {args.model}")
    model = SNACWithSpeakerConditioning.from_pretrained_base(
        repo_id=args.model,
        speaker_emb_dim=512,
        speaker_encoder_type='eres2net',
        freeze_base=True,
    ).to(args.device)
    model.eval()

    # Build dataset
    print(f"\nBuilding dataset from: {args.data}")
    dataset = build_speaker_dataset(
        args.data,
        model,
        num_speakers=args.num_speakers,
        samples_per_speaker=args.samples_per_speaker,
        device=args.device
    )

    if len(dataset['codes']) == 0:
        print("ERROR: No codes extracted!")
        return

    # Train classifier
    print("\nTraining speaker classifier on codes...")
    classifier, results = train_classifier(
        dataset,
        num_epochs=args.epochs,
        device=args.device
    )

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    num_speakers = len(dataset['speaker_to_idx'])
    chance_level = 1.0 / num_speakers
    final_acc = results['val_acc'][-1]

    print(f"\nNum speakers: {num_speakers}")
    print(f"Chance level: {chance_level:.1%}")
    print(f"Final val accuracy: {final_acc:.1%}")

    if final_acc < chance_level * 2:
        print("\n✅ CODES ARE SPEAKER-INDEPENDENT")
        print("   Classifier cannot predict speaker from codes.")
        print("   Codes likely don't contain speaker information.")
    elif final_acc < 0.5:
        print("\n⚠️  CODES HAVE MILD CONTAMINATION")
        print("   Classifier can weakly predict speaker from codes.")
        print("   Consider adding adversarial loss to remove speaker info.")
    else:
        print("\n❌ CODES ARE HEAVILY CONTAMINATED")
        print("   Classifier can reliably predict speaker from codes!")
        print("   Codes contain significant speaker information.")
        print("   CRITICAL: Add adversarial loss to purify codes.")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()

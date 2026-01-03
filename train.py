"""
Training script for SNAC with speaker conditioning.

This script trains the FiLM layers while keeping the base SNAC model frozen.
It uses a multi-speaker dataset to learn speaker-aware audio reconstruction.
"""

import os
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchaudio
from tqdm import tqdm


class SimpleAudioDataset(Dataset):
    """
    Simple dataset for audio files without speaker labels.

    The model learns speaker characteristics from audio embeddings.
    No pre-labeled speakers needed!

    Expected directory structure:
    dataset_root/
        *.wav (flat list of audio files)
    """

    def __init__(
        self,
        dataset_root,
        sampling_rate=24000,
        segment_length=2.0,  # seconds
        augment=True,
    ):
        self.dataset_root = Path(dataset_root)
        self.sampling_rate = sampling_rate
        self.segment_length = int(segment_length * sampling_rate)
        self.augment = augment

        # Collect all audio files (flat structure, no speaker folders)
        audio_extensions = ['.wav', '.WAV', '.mp3', '.flac', '.ogg', '.m4a']
        self.samples = []

        for ext in audio_extensions:
            self.samples.extend(list(self.dataset_root.glob(f'*{ext}')))

        print(f"Found {len(self.samples)} audio files in {dataset_root}")

        if len(self.samples) == 0:
            raise ValueError(f"No audio files found in {dataset_root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path = self.samples[idx]

        # Load audio
        audio, sr = torchaudio.load(str(audio_path))

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            audio = resampler(audio)

        # Segment or pad
        if audio.shape[-1] < self.segment_length:
            # Pad with zeros
            audio = F.pad(audio, (0, self.segment_length - audio.shape[-1]))
        elif audio.shape[-1] > self.segment_length:
            # Random crop
            start = torch.randint(0, audio.shape[-1] - self.segment_length, (1,)).item()
            audio = audio[..., start:start + self.segment_length]

        # Augmentation (optional)
        if self.augment and torch.rand(1).item() > 0.5:
            # Gain augmentation
            gain = 10 ** (torch.randn(1).item() * 0.1)  # ±10dB
            audio = audio * gain

        return {
            'audio': audio.squeeze(0),  # (T,)
        }


class SpeakerDataset(Dataset):
    """
    Dataset for speaker conditioning training.

    Expected directory structure:
    dataset_root/
        speaker_0001/
            audio_0001.wav
            audio_0002.wav
            ...
        speaker_0002/
            audio_0001.wav
            ...
    """

    def __init__(
        self,
        dataset_root,
        sampling_rate=24000,
        segment_length=2.0,  # seconds
        augment=True,
    ):
        self.dataset_root = Path(dataset_root)
        self.sampling_rate = sampling_rate
        self.segment_length = int(segment_length * sampling_rate)
        self.augment = augment

        # Collect audio files with speaker labels
        self.samples = []
        self.speakers = []

        for speaker_dir in sorted(self.dataset_root.iterdir()):
            if speaker_dir.is_dir():
                speaker_id = len(self.speakers)
                self.speakers.append(speaker_dir.name)

                for audio_file in speaker_dir.glob("*.wav"):
                    self.samples.append({
                        'path': audio_file,
                        'speaker_id': speaker_id
                    })

        print(f"Found {len(self.samples)} samples from {len(self.speakers)} speakers")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio
        audio, sr = torchaudio.load(sample['path'])

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
            audio = resampler(audio)

        # Segment or pad
        if audio.shape[-1] < self.segment_length:
            # Pad with zeros
            audio = F.pad(audio, (0, self.segment_length - audio.shape[-1]))
        else:
            # Random crop
            start = torch.randint(0, audio.shape[-1] - self.segment_length, (1,)).item()
            audio = audio[..., start:start + self.segment_length]

        # Augmentation (optional)
        if self.augment and torch.rand(1).item() > 0.5:
            # Gain augmentation
            gain = 10 ** (torch.randn(1).item() * 0.1)  # ±10dB
            audio = audio * gain

        return {
            'audio': audio.squeeze(0),  # (T,)
            'speaker_id': sample['speaker_id'],
        }


def multiscale_spectral_loss(audio_original, audio_reconstructed, n_ffts=[1024, 2048, 4096]):
    """
    Multi-scale STFT magnitude loss.

    Args:
        audio_original: (B, 1, T) ground truth
        audio_reconstructed: (B, 1, T) reconstruction
        n_ffts: List of FFT sizes for different scales

    Returns:
        scalar loss
    """
    loss = 0
    for n_fft in n_ffts:
        hop_length = n_fft // 4

        # Compute STFT magnitude spectrograms
        stft_orig = torch.stft(
            audio_original.squeeze(1),
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True
        ).abs()

        stft_recon = torch.stft(
            audio_reconstructed.squeeze(1),
            n_fft=n_fft,
            hop_length=hop_length,
            return_complex=True
        ).abs()

        # Spectral magnitude loss
        loss += F.l1_loss(stft_recon, stft_orig)

    return loss / len(n_ffts)


def reconstruction_loss(audio_original, audio_reconstructed, config):
    """
    Combined reconstruction loss with L1 and multi-scale spectral loss.

    Args:
        audio_original: (B, 1, T) ground truth
        audio_reconstructed: (B, 1, T) reconstruction
        config: dict with loss parameters

    Returns:
        scalar loss
    """
    # L1 loss
    loss_l1 = F.l1_loss(audio_reconstructed, audio_original)

    # Multi-scale STFT loss
    loss_stft = multiscale_spectral_loss(
        audio_original,
        audio_reconstructed,
        n_ffts=config.get('n_ffts', [1024, 2048, 4096])
    )

    # Combined
    loss = config.get('l1_weight', 1.0) * loss_l1 + config.get('stft_weight', 1.0) * loss_stft

    return loss


def speaker_consistency_loss(model, audio_original, audio_reconstructed):
    """
    Speaker consistency loss to ensure reconstructed audio preserves speaker identity.

    Uses cosine similarity between speaker embeddings of original and reconstructed audio.

    Args:
        model: SNACWithSpeakerConditioning instance
        audio_original: (B, 1, T) ground truth
        audio_reconstructed: (B, 1, T) reconstruction

    Returns:
        scalar loss (lower is better, we want similarity = 1)
    """
    # Extract speaker embeddings
    speaker_emb_orig = model.extract_speaker_embedding(audio_original)
    speaker_emb_recon = model.extract_speaker_embedding(audio_reconstructed)

    # Cosine similarity (1 = identical, 0 = orthogonal, -1 = opposite)
    # We want similarity to be 1, so loss = 1 - cosine_similarity
    similarity = F.cosine_similarity(speaker_emb_orig, speaker_emb_recon, dim=-1)
    loss = (1.0 - similarity).mean()

    return loss


def train_epoch(model, dataloader, optimizer, device, config):
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_loss_recon = 0
    total_loss_speaker = 0
    num_batches = 0

    use_speaker_loss = config.get('speaker_loss_weight', 0) > 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        audio = batch['audio'].to(device)  # (B, T)
        audio = audio.unsqueeze(1)  # (B, 1, T)

        # Forward pass
        optimizer.zero_grad()

        # Full encode-decode with speaker conditioning
        audio_hat, _ = model(audio, reference_audio=audio)

        # Reconstruction loss
        loss_recon = reconstruction_loss(audio, audio_hat, config)
        total_loss_recon += loss_recon.item()

        # Speaker consistency loss (optional)
        if use_speaker_loss:
            loss_speaker = speaker_consistency_loss(model, audio, audio_hat)
            total_loss_speaker += loss_speaker.item()
            loss = loss_recon + config['speaker_loss_weight'] * loss_speaker
        else:
            loss = loss_recon

        # Backward pass
        loss.backward()

        # Gradient clipping
        if config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Progress bar with all loss components
        postfix = {'loss': loss.item(), 'recon': loss_recon.item()}
        if use_speaker_loss:
            postfix['speaker'] = loss_speaker.item()
        pbar.set_postfix(postfix)

    # Return average losses
    avg_loss = total_loss / num_batches
    avg_recon = total_loss_recon / num_batches
    avg_speaker = total_loss_speaker / num_batches if use_speaker_loss else 0

    return avg_loss, avg_recon, avg_speaker


@torch.no_grad()
def evaluate(model, dataloader, device, config):
    """Evaluate on validation set."""
    model.eval()

    total_loss = 0
    total_loss_recon = 0
    total_loss_speaker = 0
    num_batches = 0

    use_speaker_loss = config.get('speaker_loss_weight', 0) > 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        audio = batch['audio'].to(device)
        audio = audio.unsqueeze(1)

        # Forward pass
        audio_hat, _ = model(audio, reference_audio=audio)

        # Reconstruction loss
        loss_recon = reconstruction_loss(audio, audio_hat, config)
        total_loss_recon += loss_recon.item()

        # Speaker consistency loss (optional)
        if use_speaker_loss:
            loss_speaker = speaker_consistency_loss(model, audio, audio_hat)
            total_loss_speaker += loss_speaker.item()
            loss = loss_recon + config['speaker_loss_weight'] * loss_speaker
        else:
            loss = loss_recon

        total_loss += loss.item()
        num_batches += 1

    # Return average losses
    avg_loss = total_loss / num_batches
    avg_recon = total_loss_recon / num_batches
    avg_speaker = total_loss_speaker / num_batches if use_speaker_loss else 0

    return avg_loss, avg_recon, avg_speaker


def main(config, resume_path=None, device_id=0):
    """Main training function."""
    # Set device
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # IMPORTANT: Set default CUDA device BEFORE loading model
    # This ensures from_pretrained() loads to the correct GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)

    # Create model from pretrained SNAC
    print(f"Loading pretrained SNAC from {config['pretrained_model']}")
    from snac import SNACWithSpeakerConditioning
    model = SNACWithSpeakerConditioning.from_pretrained_base(
        repo_id=config['pretrained_model'],
        speaker_emb_dim=config['speaker_emb_dim'],
        freeze_base=config['freeze_base'],
    )
    model = model.to(device)

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_path:
        print(f"Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint['epoch']} (val_loss: {checkpoint['val_loss']:.4f})")
    else:
        print(f"Loaded pretrained model from {config['pretrained_model']}")

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    # Create datasets (no speaker labels needed!)
    train_dataset = SimpleAudioDataset(
        dataset_root=config['train_data'],
        sampling_rate=model.sampling_rate,
        segment_length=config['segment_length'],
        augment=True,
    )

    val_dataset = SimpleAudioDataset(
        dataset_root=config['val_data'],
        sampling_rate=model.sampling_rate,
        segment_length=config['segment_length'],
        augment=False,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
    )

    # Optimizer (only for FiLM parameters)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        betas=(0.9, 0.999),
        weight_decay=config.get('weight_decay', 1e-5),
    )

    # Resume optimizer state if checkpoint exists
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Resumed optimizer state")

    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=config['learning_rate'] * config.get('lr_min_ratio', 0.01),
    )

    # Resume scheduler state if checkpoint exists
    if resume_path:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # Step scheduler to match the resumed epoch
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed scheduler state (current lr: {optimizer.param_groups[0]['lr']:.2e})")

    # Training loop
    best_val_loss = float('inf')
    patience = config.get('early_stopping_patience', 15)
    epochs_without_improvement = 0

    # Load best_val_loss from checkpoint if resuming
    if resume_path:
        checkpoint = torch.load(resume_path, map_location=device)
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        # Note: We don't restore epochs_without_improvement to avoid early stopping immediately

    for epoch in range(start_epoch, config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Train
        train_loss, train_recon, train_speaker = train_epoch(model, train_loader, optimizer, device, config)
        print(f"Train loss: {train_loss:.4f} (recon: {train_recon:.4f}")
        if config.get('speaker_loss_weight', 0) > 0:
            print(f", speaker: {train_speaker:.4f})")
        else:
            print(")")

        # Evaluate
        val_loss, val_recon, val_speaker = evaluate(model, val_loader, device, config)
        print(f"Val loss: {val_loss:.4f} (recon: {val_recon:.4f}")
        if config.get('speaker_loss_weight', 0) > 0:
            print(f", speaker: {val_speaker:.4f})")
        else:
            print(")")

        # Step scheduler
        scheduler.step()

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'train_recon': train_recon,
            'train_speaker': train_speaker,
            'val_loss': val_loss,
            'val_recon': val_recon,
            'val_speaker': val_speaker,
            'config': config,
        }

        # Create output directory
        os.makedirs(config['output_dir'], exist_ok=True)

        # Save latest
        torch.save(checkpoint, os.path.join(config['output_dir'], 'latest.pt'))

        # Save best
        if is_best:
            torch.save(checkpoint, os.path.join(config['output_dir'], 'best.pt'))

        # Save periodic
        if (epoch + 1) % config.get('save_every', 10) == 0:
            torch.save(checkpoint, os.path.join(config['output_dir'], f'epoch_{epoch+1}.pt'))

        print(f"Saved checkpoint (best_val_loss: {best_val_loss:.4f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SNAC with speaker conditioning')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    args = parser.parse_args()

    if args.config:
        # Load config from file
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default config
        config = {
            # Model
            'pretrained_model': 'hubertsiuzdak/snac_24khz',
            'speaker_emb_dim': 512,
            'freeze_base': True,

            # Data
            'train_data': 'data/train',
            'val_data': 'data/val',
            'segment_length': 2.0,
            'batch_size': 8,
            'num_workers': 4,

            # Training
            'num_epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'grad_clip': 1.0,
            'lr_min_ratio': 0.01,
            'save_every': 10,
            'output_dir': 'checkpoints/speaker_conditioned_snac',

            # Loss
            'l1_weight': 1.0,
            'stft_weight': 1.0,
            'speaker_loss_weight': 0.0,  # Set to 0 for Phase 1, 0.1 for Phase 2
            'n_ffts': [1024, 2048, 4096],
        }

    main(config, resume_path=args.resume, device_id=args.device)

"""
Phase 4: SNAC training with GAN discriminators (MPD + MRD)
Based on Phase 3 contrastive training, adds adversarial losses for realistic audio generation.
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

from snac import SNACWithSpeakerConditioning
from snac.discriminators import MultiPeriodDiscriminator, MultiResolutionSTFTDiscriminator


# ============== Dataset (reused from Phase 3) ==============

class SimpleAudioDataset(Dataset):
    """Simple dataset for audio files without speaker labels."""

    def __init__(
        self,
        dataset_root,
        sampling_rate=24000,
        segment_length=4.0,  # seconds
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


# ============== Loss Functions (reused from Phase 3) ==============

def multiscale_spectral_loss(audio_original, audio_reconstructed, n_ffts=[1024, 2048, 4096]):
    """Multi-scale STFT magnitude loss."""
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
    """Combined reconstruction loss with L1 and multi-scale spectral loss."""
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


def contrastive_speaker_loss(model, audio, codes, speaker_embs, config):
    """
    Contrastive speaker loss with hard negative mining.

    Improvements:
    - Hard negative mining: selects negatives with similarity 0.5-0.85 (semi-hard)
    - Falls back to semi-hard: 0.3-0.85 if not enough hard negatives
    - Random sampling within tiers for diversity
    """
    B = audio.shape[0]
    device = audio.device
    original_lengths = [audio[i].shape[-1] for i in range(B)]

    recon_losses = []
    num_negatives = min(B - 1, config.get('max_negatives', 8))
    similarity_matrix = F.cosine_similarity(speaker_embs.unsqueeze(1), speaker_embs.unsqueeze(0), dim=-1)
    same_speaker_threshold = config.get('same_speaker_threshold', 0.85)

    for i in range(B):
        codes_i = [c[i:i+1] for c in codes]

        # Positive
        speaker_emb_positive = speaker_embs[i:i+1]
        audio_positive = model.decode(codes_i, speaker_embedding=speaker_emb_positive)
        audio_positive = audio_positive[..., :original_lengths[i]]
        loss_positive = reconstruction_loss(audio[i:i+1], audio_positive, config)
        recon_losses.append(loss_positive)

        # Hard negative mining: prioritize semi-hard negatives
        # Tier 1: Hard negatives (similarity 0.5-0.85) - most informative
        hard_mask = (similarity_matrix[i] >= 0.5) & (similarity_matrix[i] < same_speaker_threshold) & (torch.arange(B, device=device) != i)
        hard_indices = torch.where(hard_mask)[0]

        # Tier 2: Semi-hard negatives (similarity 0.3-0.85)
        semi_hard_mask = (similarity_matrix[i] >= 0.3) & (similarity_matrix[i] < same_speaker_threshold) & (torch.arange(B, device=device) != i)
        semi_hard_indices = torch.where(semi_hard_mask)[0]

        # Tier 3: All valid negatives (similarity < 0.85)
        all_mask = (similarity_matrix[i] < same_speaker_threshold) & (torch.arange(B, device=device) != i)
        all_indices = torch.where(all_mask)[0]

        # Select negatives with priority
        selected_indices = []
        if len(hard_indices) > 0:
            # Prefer hard negatives - randomly sample from them
            perm = torch.randperm(len(hard_indices), device=device)
            selected_from_hard = hard_indices[perm][:num_negatives]
            selected_indices.append(selected_from_hard)

        remaining = num_negatives - len(selected_indices)
        if remaining > 0 and len(semi_hard_indices) > len(hard_indices):
            # Fill remaining with semi-hard (excluding already selected)
            semi_hard_remaining = semi_hard_indices[~torch.isin(semi_hard_indices, torch.cat(selected_indices) if len(selected_indices) > 0 else torch.tensor([], device=device))]
            if len(semi_hard_remaining) > 0:
                perm = torch.randperm(len(semi_hard_remaining), device=device)
                selected_indices.append(semi_hard_remaining[perm][:remaining])
                remaining = num_negatives - sum(len(idx) for idx in selected_indices)

        if remaining > 0:
            # Final fallback: all valid negatives
            all_remaining = all_indices[~torch.isin(all_indices, torch.cat(selected_indices) if len(selected_indices) > 0 else torch.tensor([], device=device))]
            if len(all_remaining) > 0:
                perm = torch.randperm(len(all_remaining), device=device)
                selected_indices.append(all_remaining[perm][:remaining])

        if len(selected_indices) > 0:
            selected_indices = torch.cat(selected_indices)[:num_negatives]
        else:
            # Shouldn't happen, but safety fallback
            selected_indices = torch.tensor([j for j in range(B) if j != i], device=device)[:num_negatives]

        # Decode with selected negative speaker embeddings
        for j in selected_indices:
            speaker_emb_negative = speaker_embs[j:j+1]
            audio_negative = model.decode(codes_i, speaker_embedding=speaker_emb_negative)
            audio_negative = audio_negative[..., :original_lengths[i]]
            loss_negative = reconstruction_loss(audio[i:i+1], audio_negative, config)
            recon_losses.append(loss_negative)

    # Margin-based contrastive loss
    negative_losses = []
    idx = 0

    for i in range(B):
        if idx >= len(recon_losses):
            break
        pos_loss = recon_losses[idx]
        idx += 1

        neg_losses_for_sample = []
        for _ in range(num_negatives):
            if idx >= len(recon_losses):
                break
            neg_losses_for_sample.append(recon_losses[idx])
            idx += 1

        if neg_losses_for_sample:
            neg_losses_tensor = torch.stack(neg_losses_for_sample)
            margin = config.get('contrastive_margin', 0.1)
            margin_loss = F.relu(neg_losses_tensor - pos_loss + margin).mean()
            negative_losses.append(margin_loss)

    if negative_losses:
        contrastive_loss = torch.stack(negative_losses).mean()
    else:
        contrastive_loss = torch.tensor(0.0, device=device)

    return contrastive_loss


# ============== GAN Loss Functions ==============

def discriminator_loss(disc_real, disc_fake):
    """Hinge loss for discriminators."""
    loss_real = 0
    loss_fake = 0
    for d_real, d_fake in zip(disc_real, disc_fake):
        loss_real += torch.mean(F.relu(1.0 - d_real))
        loss_fake += torch.mean(F.relu(1.0 + d_fake))
    return (loss_real + loss_fake) / len(disc_real)


def generator_loss(disc_fake):
    """Hinge loss for generator."""
    loss = 0
    for d_fake in disc_fake:
        loss += -torch.mean(d_fake)
    return loss / len(disc_fake)


def feature_matching_loss(fmap_real, fmap_fake):
    """L1 feature matching loss between real and fake feature maps."""
    loss = 0
    count = 0
    for maps_real, maps_fake in zip(fmap_real, fmap_fake):
        for map_real, map_fake in zip(maps_real, maps_fake):
            loss += F.l1_loss(map_fake, map_real)
            count += 1
    return loss / count if count > 0 else torch.tensor(0.0)


# ============== Training Loop ==============

def train_epoch(model, mpd, mrd, dataloader, opt_gen, opt_disc, device, config,
                output_dir=None, epoch=0, start_step=0, scheduler_gen=None, scheduler_disc=None):
    """Train for one epoch with GAN + contrastive loss."""
    model.train()
    mpd.train()
    mrd.train()

    total_loss_gen = 0
    total_loss_disc = 0
    total_loss_recon = 0
    total_loss_contrast = 0
    total_loss_adv = 0
    total_loss_fm = 0
    num_batches = 0

    use_contrastive = config.get('contrastive_weight', 0) > 0
    use_gan = config.get('gan_weight', 0) > 0
    save_every_steps = config.get('save_every_steps', 4000)

    # Loss weights
    lambda_adv = config.get('lambda_adv', 1.0)
    lambda_fm = config.get('lambda_fm', 2.0)
    lambda_contrast = config.get('contrastive_weight', 0.5)

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        audio = batch['audio'].to(device)  # (B, T)
        audio = audio.unsqueeze(1)  # (B, 1, T)
        B = audio.shape[0]

        # ===== Generator Forward =====
        opt_gen.zero_grad()

        # Extract speaker embeddings
        speaker_embs = model.extract_speaker_embedding(audio)
        codes = model.encode(audio)

        # Reconstruction with speaker conditioning
        # Note: model.forward() does preprocessing internally and trims to original length
        audio_hat, _ = model(audio, speaker_embedding=speaker_embs)

        # Reconstruction loss
        loss_recon = reconstruction_loss(audio, audio_hat, config)

        # Contrastive speaker loss (from Phase 3)
        loss_contrast = torch.tensor(0.0, device=device)
        if use_contrastive:
            loss_contrast = contrastive_speaker_loss(model, audio, codes, speaker_embs, config)

        # Adversarial losses (skip for first few batches to stabilize)
        loss_adv = torch.tensor(0.0, device=device)
        loss_fm = torch.tensor(0.0, device=device)

        if use_gan and num_batches > 10:  # Warmup period
            # MPD forward - ONE call with (real, fake)
            y_d_rs_mpd, y_d_gs_mpd, fmap_rs_mpd, fmap_gs_mpd = mpd(audio, audio_hat)

            # MRD forward - ONE call with (real, fake)
            y_d_rs_mrd, y_d_gs_mrd, fmap_rs_mrd, fmap_gs_mrd = mrd(audio, audio_hat)

            # Generator adversarial loss
            loss_adv_mpd = generator_loss(y_d_gs_mpd)
            loss_adv_mrd = generator_loss(y_d_gs_mrd)
            loss_adv = (loss_adv_mpd + loss_adv_mrd) / 2.0

            # Feature matching loss
            loss_fm_mpd = feature_matching_loss(fmap_rs_mpd, fmap_gs_mpd)
            loss_fm_mrd = feature_matching_loss(fmap_rs_mrd, fmap_gs_mrd)
            loss_fm = (loss_fm_mpd + loss_fm_mrd) / 2.0

        # Total generator loss
        loss_gen = loss_recon + lambda_contrast * loss_contrast + \
                   lambda_adv * loss_adv + lambda_fm * loss_fm

        loss_gen.backward()
        if config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        opt_gen.step()

        # ===== Discriminator Forward =====
        loss_disc = torch.tensor(0.0, device=device)
        if use_gan and num_batches > 10:
            opt_disc.zero_grad()

            # MPD forward
            y_d_rs_mpd, y_d_gs_mpd, _, _ = mpd(audio, audio_hat.detach())
            loss_disc_mpd = discriminator_loss(y_d_rs_mpd, y_d_gs_mpd)

            # MRD forward
            y_d_rs_mrd, y_d_gs_mrd, _, _ = mrd(audio, audio_hat.detach())
            loss_disc_mrd = discriminator_loss(y_d_rs_mrd, y_d_gs_mrd)

            loss_disc = (loss_disc_mpd + loss_disc_mrd) / 2.0
            loss_disc.backward()

            if config.get('grad_clip_disc', 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(list(mpd.parameters()) + list(mrd.parameters()),
                                                 config['grad_clip_disc'])
            opt_disc.step()

        # Update metrics
        total_loss_gen += loss_gen.item()
        total_loss_disc += loss_disc.item()
        total_loss_recon += loss_recon.item()
        total_loss_contrast += loss_contrast.item()
        total_loss_adv += loss_adv.item()
        total_loss_fm += loss_fm.item()
        num_batches += 1

        # Progress bar
        postfix = {
            'g_loss': loss_gen.item(),
            'd_loss': loss_disc.item(),
            'recon': loss_recon.item()
        }
        if use_contrastive:
            postfix['contrast'] = loss_contrast.item()
        if use_gan and num_batches > 10:
            postfix['adv'] = loss_adv.item()
            postfix['fm'] = loss_fm.item()
        pbar.set_postfix(postfix)

        # Step-based checkpointing
        current_step = start_step + num_batches
        if output_dir and save_every_steps > 0 and current_step % save_every_steps == 0:
            checkpoint = {
                'epoch': epoch,
                'step': current_step,
                'model_state_dict': model.state_dict(),
                'mpd_state_dict': mpd.state_dict(),
                'mrd_state_dict': mrd.state_dict(),
                'opt_gen_state_dict': opt_gen.state_dict(),
                'opt_disc_state_dict': opt_disc.state_dict(),
                'scheduler_gen_state_dict': scheduler_gen.state_dict() if scheduler_gen else None,
                'scheduler_disc_state_dict': scheduler_disc.state_dict() if scheduler_disc else None,
                'config': config,
            }
            checkpoint_path = output_dir / f'step_{current_step}.pt'
            torch.save(checkpoint, checkpoint_path)
            pbar.write(f"Saved checkpoint at step {current_step}")

    # Return average losses and total steps
    return {
        'gen': total_loss_gen / num_batches,
        'disc': total_loss_disc / num_batches,
        'recon': total_loss_recon / num_batches,
        'contrast': total_loss_contrast / num_batches,
        'adv': total_loss_adv / num_batches,
        'fm': total_loss_fm / num_batches,
        'total_steps': start_step + num_batches,
    }


@torch.no_grad()
def evaluate(model, mpd, mrd, dataloader, device, config):
    """Evaluate on validation set."""
    model.eval()
    mpd.eval()
    mrd.eval()

    total_loss_gen = 0
    total_loss_recon = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        audio = batch['audio'].to(device)
        audio = audio.unsqueeze(1)

        # Extract speaker embeddings
        speaker_embs = model.extract_speaker_embedding(audio)

        # Forward pass (no GAN loss on validation for speed)
        audio_hat, _ = model(model.preprocess(audio), speaker_embedding=speaker_embs)

        # Reconstruction loss only
        loss_recon = reconstruction_loss(audio, audio_hat, config)
        total_loss_recon += loss_recon.item()
        total_loss_gen += loss_recon.item()  # Use recon loss for checkpointing
        num_batches += 1

    # Return average losses
    return {
        'gen': total_loss_gen / num_batches,
        'disc': 0.0,  # Not computed on validation
        'recon': total_loss_recon / num_batches,
        'contrast': 0.0,
        'adv': 0.0,
        'fm': 0.0,
    }


def main(config, resume_path=None, device_id=0):
    """Main training function."""
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)

    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    model = SNACWithSpeakerConditioning.from_pretrained_base(
        repo_id=config['pretrained_model'],
        speaker_emb_dim=config['speaker_emb_dim'],
        speaker_encoder_type=config.get('speaker_encoder_type', 'ecapa'),
        freeze_base=config['freeze_base'],
    ).to(device)

    # Create discriminators
    mpd = MultiPeriodDiscriminator(periods=config.get('mpd_periods', [2, 3, 5, 7, 11])).to(device)
    mrd = MultiResolutionSTFTDiscriminator(fft_sizes=config.get('mrd_fft_sizes', [1024, 2048, 4096])).to(device)

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"MPD: {sum(p.numel() for p in mpd.parameters()):,} parameters")
    print(f"MRD: {sum(p.numel() for p in mrd.parameters()):,} parameters")

    # Resume from checkpoint if specified
    start_epoch = 0
    start_step = 0
    if resume_path and Path(resume_path).exists():
        print(f"Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        mpd.load_state_dict(checkpoint['mpd_state_dict'])
        mrd.load_state_dict(checkpoint['mrd_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        start_step = checkpoint.get('step', 0)
        print(f"Resumed from epoch {checkpoint.get('epoch', 0)}, step {start_step}")

    # Create datasets
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

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=config['num_workers'],
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=config['num_workers'],
                            pin_memory=True)

    # Optimizers
    opt_gen = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        betas=(0.5, 0.9),  # GAN standard
        weight_decay=config.get('weight_decay', 1e-5),
    )

    opt_disc = AdamW(
        list(mpd.parameters()) + list(mrd.parameters()),
        lr=config['disc_learning_rate'],
        betas=(0.5, 0.9),
    )

    # Schedulers
    scheduler_gen = CosineAnnealingLR(
        opt_gen, T_max=config['num_epochs'],
        eta_min=config['learning_rate'] * config.get('lr_min_ratio', 0.01),
    )
    scheduler_disc = CosineAnnealingLR(
        opt_disc, T_max=config['num_epochs'],
        eta_min=config['disc_learning_rate'] * config.get('lr_min_ratio', 0.01),
    )

    # Resume optimizers if checkpoint exists
    if resume_path and Path(resume_path).exists():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
        opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
        scheduler_gen.load_state_dict(checkpoint['scheduler_gen_state_dict'])
        scheduler_disc.load_state_dict(checkpoint['scheduler_disc_state_dict'])
        for _ in range(start_epoch):
            scheduler_gen.step()
            scheduler_disc.step()

    # Training loop
    best_val_loss = float('inf')
    patience = config.get('early_stopping_patience', 10)
    epochs_without_improvement = 0

    for epoch in range(start_epoch, config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print(f"Learning rate: {opt_gen.param_groups[0]['lr']:.2e}")

        # Train
        train_metrics = train_epoch(model, mpd, mrd, train_loader, opt_gen, opt_disc, device, config,
                                    output_dir=output_dir, epoch=epoch, start_step=start_step,
                                    scheduler_gen=scheduler_gen, scheduler_disc=scheduler_disc)
        start_step = train_metrics['total_steps']  # Update for next epoch
        print(f"Train - G: {train_metrics['gen']:.4f}, D: {train_metrics['disc']:.4f}, "
              f"Recon: {train_metrics['recon']:.4f}, Contrast: {train_metrics['contrast']:.4f}, "
              f"Adv: {train_metrics['adv']:.4f}, FM: {train_metrics['fm']:.4f}")

        # Validate
        val_metrics = evaluate(model, mpd, mrd, val_loader, device, config)
        print(f"Val   - G: {val_metrics['gen']:.4f}, D: {val_metrics['disc']:.4f}, "
              f"Recon: {val_metrics['recon']:.4f}")

        # Step schedulers
        scheduler_gen.step()
        scheduler_disc.step()

        # Checkpoint
        is_best = val_metrics['gen'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['gen']
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        checkpoint = {
            'epoch': epoch,
            'step': start_step,
            'model_state_dict': model.state_dict(),
            'mpd_state_dict': mpd.state_dict(),
            'mrd_state_dict': mrd.state_dict(),
            'opt_gen_state_dict': opt_gen.state_dict(),
            'opt_disc_state_dict': opt_disc.state_dict(),
            'scheduler_gen_state_dict': scheduler_gen.state_dict(),
            'scheduler_disc_state_dict': scheduler_disc.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'config': config,
        }

        torch.save(checkpoint, output_dir / 'latest.pt')
        if is_best:
            torch.save(checkpoint, output_dir / 'best.pt')

        if (epoch + 1) % config.get('save_every', 10) == 0:
            torch.save(checkpoint, output_dir / f'epoch_{epoch+1}.pt')

        print(f"Saved checkpoint (best_gen_loss: {best_val_loss:.4f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/phase4_gan.json')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config, args.resume, args.device)

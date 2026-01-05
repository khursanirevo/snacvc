"""
Phase 8: Hybrid Voice Conversion - Encoder Adapter + Decoder FiLM

Architecture:
    Audio → Encoder → Adapter1→...→Adapter5 → VQ → codes
                                                   ↓
    Decoder ← FiLM(speaker_emb) ← latent
    ↓
    audio_hat (target speaker voice)

Why this works:
1. Encoder adapters (before VQ): Shift codes toward target speaker
2. Decoder FiLM (after VQ): Guide decoder to produce target speaker
3. Both work together for voice conversion!
"""

import os
import json
import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from snac import SNACWithSpeakerConditioning
from snac.discriminators import MultiPeriodDiscriminator, MultiResolutionSTFTDiscriminator
from snac.adapters import HybridVCWrapper

# Import utilities
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_phase8_multistage import SimpleAudioDataset, reconstruction_loss, \
    generator_loss, discriminator_loss, feature_matching_loss


# ============== Loss Functions ==============

def voice_conversion_loss_hybrid(model, audio, target_embs, config):
    """
    Voice conversion loss for hybrid model.

    Audio from speaker A + embedding B → should sound like speaker B
    """
    speaker_enc = model.base_model.speaker_encoder

    # Encode with target speaker embedding
    audio_hat, _ = model(audio, speaker_embedding=target_embs)

    # Extract speaker embedding from reconstructed audio
    speaker_embs_recon = speaker_enc(audio_hat.squeeze(1))

    # Cosine similarity
    similarity = F.cosine_similarity(target_embs, speaker_embs_recon, dim=-1)

    # Loss: 1 - similarity (lower is better)
    loss = (1.0 - similarity).mean()

    return loss


# ============== Training ==============

def train_epoch(model, mpd, mrd, dataloader, opt_gen, opt_disc, device, config,
                output_dir=None, epoch=0):

    model.eval()  # Keep base model frozen
    model_base = model

    lambda_speaker_matching = config.get('lambda_speaker_matching', 2.0)
    lambda_recon_consistency = config.get('lambda_recon_consistency', 0.1)
    lambda_synthetic = config.get('lambda_synthetic', 0.1)
    lambda_adv = config.get('lambda_adv', 1.0)
    lambda_fm = config.get('lambda_fm', 2.0)

    use_gan = config.get('gan_weight', 1.0) > 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} | Hybrid VC (Enc+Dec)")

    total_loss_gen = 0.0
    total_loss_disc = 0.0
    total_loss_recon = 0.0
    total_loss_spk_match = 0.0
    total_loss_spk_id = 0.0  # Same-speaker baseline
    total_loss_recon_con = 0.0
    total_loss_synth = 0.0
    total_loss_adv = 0.0
    total_loss_fm = 0.0
    num_batches = 0

    for batch in pbar:
        audio = batch['audio'].to(device)
        audio = audio.unsqueeze(1)
        B = audio.shape[0]

        # ===== Generator Forward =====
        opt_gen.zero_grad()

        # Extract speaker embeddings
        speaker_embs = model_base.base_model.extract_speaker_embedding(audio)

        # For VC: Shuffle to get different speakers
        perm = torch.randperm(B, device=device)
        target_embs = speaker_embs[perm]

        # Reconstruction with target speaker (hybrid: enc adapter + dec film)
        audio_hat, _ = model_base(audio, speaker_embedding=target_embs)

        # Reconstruction with SAME speaker (baseline)
        audio_hat_same, _ = model_base(audio, speaker_embedding=speaker_embs)

        # Reconstruction WITHOUT conditioning (for consistency)
        audio_hat_uncond, _ = model_base(audio, speaker_embedding=None)

        # Reconstruction loss (content preservation)
        loss_recon = reconstruction_loss(audio, audio_hat, config)

        # Voice conversion loss (different speaker)
        loss_spk_vc = voice_conversion_loss_hybrid(
            model_base, audio, target_embs, config
        )

        # Same-speaker identity loss (baseline - should be lower)
        loss_spk_id = voice_conversion_loss_hybrid(
            model_base, audio, speaker_embs, config
        )

        # Combined speaker matching loss
        loss_spk_match = loss_spk_vc

        # Reconstruction consistency (don't deviate too much from base)
        loss_recon_con = F.l1_loss(audio_hat, audio_hat_uncond)

        # Synthetic VC (pitch shift augmentation)
        loss_synth = torch.tensor(0.0, device=device)
        use_synthetic = config.get('use_synthetic_vc', False)
        if use_synthetic and num_batches > 10:
            from snac.audio_augmentation import augment_audio_for_voice_conversion_advanced

            synth_prob = config.get('synthetic_vc_probability', 0.3)
            synth_losses = []

            for i in range(audio.shape[0]):
                if random.random() < synth_prob:
                    audio_i = audio[i:i+1]
                    speaker_emb_i = speaker_embs[i:i+1]

                    # Apply pitch shift
                    audio_aug, was_aug, semitones, formant_shift = \
                        augment_audio_for_voice_conversion_advanced(
                            audio_i,
                            pitch_shift_range=config.get('pitch_shift_range', [-2, -1, 1, 2]),
                            formant_shift_range=config.get('formant_shift_range', []),
                            probability=1.0
                        )

                    if was_aug:
                        # Encode augmented audio with ORIGINAL speaker embedding
                        audio_recon, _ = model_base(audio_aug, speaker_embedding=speaker_emb_i)

                        # Should match original audio (hybrid model restores speaker)
                        loss_i = reconstruction_loss(audio[i:i+1], audio_recon, config)
                        synth_losses.append(loss_i)

            if synth_losses:
                loss_synth = torch.stack(synth_losses).mean()

        # Adversarial losses
        loss_adv = torch.tensor(0.0, device=device)
        loss_fm = torch.tensor(0.0, device=device)

        if use_gan and num_batches > 10:
            y_d_rs_mpd, y_d_gs_mpd, fmap_rs_mpd, fmap_gs_mpd = mpd(audio, audio_hat)
            y_d_rs_mrd, y_d_gs_mrd, fmap_rs_mrd, fmap_gs_mrd = mrd(audio, audio_hat)

            loss_adv_mpd = generator_loss(y_d_gs_mpd)
            loss_adv_mrd = generator_loss(y_d_gs_mrd)
            loss_adv = (loss_adv_mpd + loss_adv_mrd) / 2.0

            loss_fm_mpd = feature_matching_loss(fmap_rs_mpd, fmap_gs_mpd)
            loss_fm_mrd = feature_matching_loss(fmap_rs_mrd, fmap_gs_mrd)
            loss_fm = (loss_fm_mpd + loss_fm_mrd) / 2.0

        # Total generator loss
        loss_gen = (loss_recon +
                    lambda_speaker_matching * loss_spk_match +
                    lambda_recon_consistency * loss_recon_con +
                    lambda_synthetic * loss_synth +
                    lambda_adv * loss_adv + lambda_fm * loss_fm)

        loss_gen.backward()

        # Gradient clipping
        if config.get('grad_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

        opt_gen.step()

        # ===== Discriminator =====
        loss_disc = torch.tensor(0.0, device=device)
        if use_gan and num_batches > 10:
            opt_disc.zero_grad()

            y_d_rs_mpd, y_d_gs_mpd, _, _ = mpd(audio, audio_hat.detach())
            loss_disc_mpd = discriminator_loss(y_d_rs_mpd, y_d_gs_mpd)

            y_d_rs_mrd, y_d_gs_mrd, _, _ = mrd(audio, audio_hat.detach())
            loss_disc_mrd = discriminator_loss(y_d_rs_mrd, y_d_gs_mrd)

            loss_disc = (loss_disc_mpd + loss_disc_mrd) / 2.0
            loss_disc.backward()

            if config.get('grad_clip_disc', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(mpd.parameters()) + list(mrd.parameters()),
                    config['grad_clip_disc']
                )
            opt_disc.step()

        # Update metrics
        total_loss_gen += loss_gen.item()
        total_loss_disc += loss_disc.item()
        total_loss_recon += loss_recon.item()
        total_loss_spk_match += loss_spk_match.item()
        total_loss_spk_id += loss_spk_id.item()
        total_loss_recon_con += loss_recon_con.item()
        total_loss_synth += loss_synth.item()
        total_loss_adv += loss_adv.item()
        total_loss_fm += loss_fm.item()
        num_batches += 1

        # Progress - show VC and ID losses separately
        postfix = {
            'g_loss': loss_gen.item(),
            'd_loss': loss_disc.item(),
            'recon': loss_recon.item(),
            'vc': loss_spk_vc.item(),  # Voice conversion (diff speaker)
            'id': loss_spk_id.item(),  # Identity (same speaker)
            'recon_con': loss_recon_con.item(),
            'synth': loss_synth.item(),
            'adv': loss_adv.item(),
            'fm': loss_fm.item(),
        }
        pbar.set_postfix(postfix)

    # Return metrics
    return {
        'generator': total_loss_gen / num_batches,
        'discriminator': total_loss_disc / num_batches,
        'reconstruction': total_loss_recon / num_batches,
        'speaker_matching': total_loss_spk_match / num_batches,
        'speaker_identity': total_loss_spk_id / num_batches,  # Same-speaker baseline
        'recon_consistency': total_loss_recon_con / num_batches,
        'synthetic': total_loss_synth / num_batches,
        'adversarial': total_loss_adv / num_batches,
        'feature_matching': total_loss_fm / num_batches,
    }


def validate(model, mpd, mrd, dataloader, device, config):
    """Validation loop."""
    model.eval()
    model_base = model

    total_loss_recon = 0.0
    total_loss_spk_match = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            audio = batch['audio'].to(device).unsqueeze(1)
            speaker_embs = model_base.base_model.extract_speaker_embedding(audio)

            # Shuffle for VC
            B = audio.shape[0]
            perm = torch.randperm(B, device=device)
            target_embs = speaker_embs[perm]

            audio_hat, _ = model_base(audio, speaker_embedding=target_embs)

            loss_recon = reconstruction_loss(audio, audio_hat, config)
            loss_spk_match = voice_conversion_loss_hybrid(model_base, audio, target_embs, config)

            total_loss_recon += loss_recon.item()
            total_loss_spk_match += loss_spk_match.item()
            num_batches += 1

    return {
        'val_reconstruction': total_loss_recon / num_batches,
        'val_speaker_matching': total_loss_spk_match / num_batches,
    }


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description="Train Phase 8: Hybrid VC (Enc+Dec)")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output dir
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base model
    print("\nLoading base model...")
    base_model = SNACWithSpeakerConditioning.from_pretrained_base(
        config['pretrained_model'],
        speaker_encoder_type=config['speaker_encoder_type'],
        freeze_base=True,  # Keep everything frozen
    ).to(device)

    # Wrap with hybrid VC wrapper
    print("\nInitializing Hybrid VC model...")
    print("  - Multi-stage encoder adapters (before VQ)")
    print("  - Decoder FiLM adapter (after VQ)")
    model = HybridVCWrapper(
        base_model=base_model,
        encoder_adapter_hidden_dim=config.get('encoder_adapter_hidden_dim', 512),
        encoder_adapter_num_layers=config.get('encoder_adapter_num_layers', 2),
        decoder_film_hidden_dim=config.get('decoder_film_hidden_dim', 512),
        decoder_film_num_layers=config.get('decoder_film_num_layers', 2),
        adaptive_init=config.get('adaptive_init', True),
    ).to(device)

    print(f"\nEncoder adapter stages: {model.encoder_adapter.get_num_adapters()}")
    print(f"Total adapters (enc+dec): {model.get_num_adapters()}")
    print(f"Total adapter parameters: {model.get_adapter_params():,}")

    # Breakdown
    enc_params = model.encoder_adapter.get_adapter_params()
    dec_params = sum(p.numel() for p in model.decoder_film.parameters())
    print(f"  Encoder adapters: {enc_params:,} params")
    print(f"  Decoder FiLM: {dec_params:,} params")

    # Discriminators
    print("\nInitializing discriminators...")
    mpd = MultiPeriodDiscriminator(periods=config.get('mpd_periods', [2, 3, 5, 7, 11])).to(device)
    mrd = MultiResolutionSTFTDiscriminator(fft_sizes=config.get('mrd_fft_sizes', [1024, 2048, 4096])).to(device)

    # Optimizers (only adapter parameters)
    trainable_params = []
    for adapter in model.encoder_adapter.adapters:
        trainable_params.extend(list(adapter.parameters()))
    trainable_params.extend(list(model.decoder_film.parameters()))

    print(f"Training {len(trainable_params)} parameter groups")

    opt_gen = AdamW(
        trainable_params,
        lr=config['learning_rate'],
        betas=(0.8, 0.99),
        weight_decay=config['weight_decay']
    )
    opt_disc = AdamW(list(mpd.parameters()) + list(mrd.parameters()),
                     lr=config['disc_learning_rate'], betas=(0.8, 0.99))

    # Schedulers
    scheduler_gen = CosineAnnealingLR(
        opt_gen, T_max=config['num_epochs'],
        eta_min=config['learning_rate'] * config.get('lr_min_ratio', 0.01)
    )
    scheduler_disc = CosineAnnealingLR(
        opt_disc, T_max=config['num_epochs'],
        eta_min=config['disc_learning_rate'] * config.get('lr_min_ratio', 0.01)
    )

    # Datasets
    print("\nLoading datasets...")
    train_dataset = SimpleAudioDataset(
        config['train_data'],
        sampling_rate=24000,
        segment_length=config['segment_length'],
        augment=True,
    )
    val_dataset = SimpleAudioDataset(
        config['val_data'],
        sampling_rate=24000,
        segment_length=config['segment_length'],
        augment=False,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        opt_gen.load_state_dict(checkpoint['opt_gen'])
        opt_disc.load_state_dict(checkpoint['opt_disc'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    print("\n" + "="*70)
    print("Training: Hybrid Voice Conversion")
    print("  Architecture: Encoder Adapter(s) + Decoder FiLM")
    print("  Encoder: FROZEN")
    print("  VQ: FROZEN")
    print("  Decoder: FROZEN (but conditioned via FiLM)")
    print("  Encoder Adapters: TRAINABLE (shift codes)")
    print("  Decoder FiLM: TRAINABLE (guide decoder)")
    print("="*70 + "\n")

    for epoch in range(start_epoch, config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print(f"Learning rate: {scheduler_gen.get_last_lr()[0]:.2e}")

        # Train
        train_metrics = train_epoch(
            model, mpd, mrd, train_loader, opt_gen, opt_disc, device, config,
            output_dir=output_dir, epoch=epoch
        )

        # Validate
        val_metrics = validate(model, mpd, mrd, val_loader, device, config)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train recon: {train_metrics['reconstruction']:.4f}")
        print(f"  Train VC (diff spk): {train_metrics['speaker_matching']:.4f}")
        print(f"  Train ID (same spk): {train_metrics['speaker_identity']:.4f}")
        print(f"  Val recon: {val_metrics['val_reconstruction']:.4f}")
        print(f"  Val spk_match: {val_metrics['val_speaker_matching']:.4f}")

        # Step schedulers
        scheduler_gen.step()
        scheduler_disc.step()

        # Save checkpoint
        val_loss = val_metrics['val_reconstruction']
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if (epoch + 1) % config.get('save_every', 5) == 0 or is_best:
            checkpoint_path = output_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'opt_gen': opt_gen.state_dict(),
                'opt_disc': opt_disc.state_dict(),
                'best_val_loss': best_val_loss,
                'config': config,
            }, checkpoint_path)
            print(f"Saved: {checkpoint_path}")

            if is_best:
                best_path = output_dir / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'config': config,
                }, best_path)
                print(f"Saved best model: {best_path}")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

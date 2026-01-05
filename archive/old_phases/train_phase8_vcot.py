"""
Phase 8: Voice Conversion with DECODER FINE-TUNING

Key change from previous attempts:
- Encoder: FROZEN (preserves content extraction)
- VQ: FROZEN (stable codebook)
- Decoder: UNFROZEN ← This is the key!
- Adapter: TRAINED

Why this works:
- Adapter shifts encoder latent before VQ
- Decoder learns to decode the shifted codes
- No more "decoder doesn't know how to handle shifted codes" problem
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
from snac.faiss_speaker_index import FaissSpeakerIndex
from snac.embedding_cache import OptimizedEmbeddingCache
from snac.audio_augmentation import augment_audio_for_voice_conversion_advanced
from snac.adapters import MultiStageAdapterWrapper

# Import from Phase 8
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_phase8_multistage import SimpleAudioDataset, reconstruction_loss, \
    generator_loss, discriminator_loss, feature_matching_loss


# ============== Loss Functions ==============

def voice_conversion_loss_simple(model, audio, speaker_embs, config):
    """
    Simple voice conversion loss:
    - Encode with target speaker embedding
    - Should match target speaker
    """
    speaker_enc = model.base_model.speaker_encoder

    # Encode with target speaker embedding
    audio_hat, _ = model(audio, speaker_embedding=speaker_embs)

    # Extract speaker embedding from reconstructed audio
    speaker_embs_recon = speaker_enc(audio_hat.squeeze(1))

    # Cosine similarity
    similarity = F.cosine_similarity(speaker_embs, speaker_embs_recon, dim=-1)

    # Loss: 1 - similarity (lower is better)
    loss = (1.0 - similarity).mean()

    return loss


# ============== Training ==============

def train_epoch(model, mpd, mrd, dataloader, opt_gen, opt_disc, device, config,
                output_dir=None, epoch=0, faiss_index=None, embedding_cache=None):

    model.eval()  # Keep base model in eval mode, but decoder will have grad enabled
    model_base = model

    lambda_speaker_matching = config.get('lambda_speaker_matching', 2.0)
    lambda_recon_consistency = config.get('lambda_recon_consistency', 0.1)
    lambda_synthetic = config.get('lambda_synthetic', 0.1)
    lambda_adv = config.get('lambda_adv', 1.0)
    lambda_fm = config.get('lambda_fm', 2.0)

    use_gan = config.get('gan_weight', 1.0) > 0
    use_contrastive = config.get('contrastive_weight', 0.0) > 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} | Voice Conversion (Decoder Fine-tuning)")

    total_loss_gen = 0.0
    total_loss_disc = 0.0
    total_loss_recon = 0.0
    total_loss_spk_match = 0.0
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

        # For VC: Use target speaker embeddings (from FAISS negatives or shuffle)
        if use_contrastive and embedding_cache is not None:
            # Get hard negatives from FAISS
            import numpy as np
            speaker_ids = batch.get('speaker_id', None)

            # For each sample, get a different speaker
            target_embs = speaker_embs.clone()
            for i in range(B):
                # Simple: shuffle to get different speaker
                other_idx = (i + random.randint(1, B-1)) % B
                target_embs[i] = speaker_embs[other_idx]
        else:
            # Simple: shuffle within batch
            target_embs = speaker_embs.clone()
            perm = torch.randperm(B, device=device)
            target_embs = target_embs[perm]

        # Reconstruction with target speaker embedding (VC)
        audio_hat, _ = model_base(audio, speaker_embedding=target_embs)

        # Reconstruction WITHOUT speaker conditioning (for consistency)
        audio_hat_uncond, _ = model_base(audio, speaker_embedding=None)

        # Reconstruction loss (content preservation)
        loss_recon = reconstruction_loss(audio, audio_hat, config)

        # Speaker matching loss (should sound like target speaker)
        loss_spk_match = voice_conversion_loss_simple(
            model_base, audio, target_embs, config
        )

        # Reconstruction consistency (don't deviate too much from base)
        loss_recon_con = F.l1_loss(audio_hat, audio_hat_uncond)

        # Synthetic VC (pitch shift augmentation)
        loss_synth = torch.tensor(0.0, device=device)
        use_synthetic = config.get('use_synthetic_vc', False)
        if use_synthetic and num_batches > 10:
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

                        # Should match original audio (adapter restores speaker)
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
        total_loss_recon_con += loss_recon_con.item()
        total_loss_synth += loss_synth.item()
        total_loss_adv += loss_adv.item()
        total_loss_fm += loss_fm.item()
        num_batches += 1

        # Progress
        postfix = {
            'g_loss': loss_gen.item(),
            'd_loss': loss_disc.item(),
            'recon': loss_recon.item(),
            'spk_match': loss_spk_match.item(),
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
            loss_spk_match = voice_conversion_loss_simple(model_base, audio, target_embs, config)

            total_loss_recon += loss_recon.item()
            total_loss_spk_match += loss_spk_match.item()
            num_batches += 1

    return {
        'val_reconstruction': total_loss_recon / num_batches,
        'val_speaker_matching': total_loss_spk_match / num_batches,
    }


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description="Train Phase 8: Voice Conversion with Decoder Fine-tuning")
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
        freeze_base=config.get('freeze_base', False),  # Don't freeze yet
    ).to(device)

    # Freeze encoder and VQ, but NOT decoder
    print("\nFreezing encoder and VQ...")
    for param in base_model.base_model.encoder.parameters():
        param.requires_grad = False
    for param in base_model.base_model.quantizer.parameters():
        param.requires_grad = False

    # Keep decoder trainable!
    print("Decoder is TRAINABLE (will learn to decode shifted codes)")

    # Keep speaker encoder frozen
    for param in base_model.speaker_encoder.parameters():
        param.requires_grad = False

    # Wrap with multi-stage adapter
    print("\nInitializing multi-stage adapter...")
    model = MultiStageAdapterWrapper(
        base_model=base_model,
        adapter_hidden_dim=config['adapter_hidden_dim'],
        adapter_num_layers=config['adapter_num_layers'],
        adaptive_init=config.get('adaptive_init', True),
    ).to(device)

    print(f"Adapter stages: {model.get_num_adapters()}")
    print(f"Adapter parameters: {model.get_adapter_params():,}")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    decoder_params = sum(p.numel() for p in model.base_model.base_model.decoder.parameters() if p.requires_grad)
    print(f"Trainable decoder parameters: {decoder_params:,}")
    print(f"Total trainable parameters: {trainable_params:,}")

    # Discriminators
    print("\nInitializing discriminators...")
    mpd = MultiPeriodDiscriminator(periods=config.get('mpd_periods', [2, 3, 5, 7, 11])).to(device)
    mrd = MultiResolutionSTFTDiscriminator(fft_sizes=config.get('mrd_fft_sizes', [1024, 2048, 4096])).to(device)

    # Optimizers (adapter + decoder)
    opt_gen = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.get('learning_rate', 5e-5),
        betas=(0.8, 0.99),
        weight_decay=config['weight_decay']
    )
    opt_disc = AdamW(list(mpd.parameters()) + list(mrd.parameters()),
                     lr=config['disc_learning_rate'], betas=(0.8, 0.99))

    # Schedulers
    scheduler_gen = CosineAnnealingLR(
        opt_gen, T_max=config['num_epochs'],
        eta_min=config.get('learning_rate', 5e-5) * config.get('lr_min_ratio', 0.01)
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

    # Load FAISS and embedding cache
    faiss_index = None
    embedding_cache = None

    if config.get('use_faiss_hard_negatives', False):
        faiss_path = Path(config.get('faiss_index_path', 'pretrained_models/speaker_faiss.index'))
        if faiss_path.exists():
            print(f"\nLoading FAISS index from {faiss_path}...")
            faiss_index = FaissSpeakerIndex.load(str(faiss_path))
            print(f"FAISS index loaded: {faiss_index.index.ntotal} embeddings")

        cache_path = Path('pretrained_models/embeddings_cache.npy')
        if cache_path.exists():
            print(f"Loading embedding cache from {cache_path}...")
            embedding_cache = OptimizedEmbeddingCache.load(str(cache_path))
            print(f"Embedding cache loaded: {len(embedding_cache)} embeddings")

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
    print("Training: Voice Conversion with Decoder Fine-tuning")
    print("  Encoder: FROZEN")
    print("  VQ: FROZEN")
    print("  Decoder: TRAINABLE ← Key difference!")
    print("  Adapter: TRAINABLE")
    print("="*70 + "\n")

    for epoch in range(start_epoch, config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print(f"Learning rate: {scheduler_gen.get_last_lr()[0]:.2e}")

        # Train
        train_metrics = train_epoch(
            model, mpd, mrd, train_loader, opt_gen, opt_disc, device, config,
            output_dir=output_dir, epoch=epoch,
            faiss_index=faiss_index,
            embedding_cache=embedding_cache
        )

        # Validate
        val_metrics = validate(model, mpd, mrd, val_loader, device, config)

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train recon: {train_metrics['reconstruction']:.4f}")
        print(f"  Train spk_match: {train_metrics['speaker_matching']:.4f}")
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

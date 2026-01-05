"""
Phase 6: SNAC training with ADAPTER LAYERS BEFORE QUANTIZATION
Based on Phase 4, but conditions encoder latent BEFORE VQ for speaker-conditioned codes.

Supports DDP (DistributedDataParallel) for multi-GPU training.
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchaudio
from tqdm import tqdm

from snac import SNACWithSpeakerConditioning
from snac.discriminators import MultiPeriodDiscriminator, MultiResolutionSTFTDiscriminator
from snac.faiss_speaker_index import FaissSpeakerIndex
from snac.embedding_cache import OptimizedEmbeddingCache
from snac.audio_augmentation import augment_audio_for_voice_conversion, augment_audio_for_voice_conversion_advanced
from snac.adapters import AdapterWrapper
from snac.voice_conversion_loss import voice_conversion_loss
# Codebook adversarial loss is optional and not recommended
try:
    from snac.codebook_adversarial_loss import SpeakerDiscriminator, adversarial_codebook_loss_v2, GradientReversal
    HAS_CODEBOOK_ADV = True
except ImportError:
    HAS_CODEBOOK_ADV = False


# ============== Dataset (reused from Phase 3) ==============

class SimpleAudioDataset(Dataset):
    """Simple dataset for audio files with optional speaker labels."""

    def __init__(
        self,
        dataset_root,
        sampling_rate=24000,
        segment_length=4.0,  # seconds
        augment=True,
        extract_speaker_ids=False,
    ):
        self.dataset_root = Path(dataset_root)
        self.sampling_rate = sampling_rate
        self.segment_length = int(segment_length * sampling_rate)
        self.augment = augment
        self.extract_speaker_ids = extract_speaker_ids

        # Collect all audio files (flat structure, no speaker folders)
        audio_extensions = ['.wav', '.WAV', '.mp3', '.flac', '.ogg', '.m4a']
        self.samples = []

        for ext in audio_extensions:
            self.samples.extend(list(self.dataset_root.glob(f'*{ext}')))

        print(f"Found {len(self.samples)} audio files in {dataset_root}")

        if len(self.samples) == 0:
            raise ValueError(f"No audio files found in {dataset_root}")

        # Extract speaker IDs from filenames if requested
        # For adversarial codebook purification
        self.speaker_to_idx = {}
        if extract_speaker_ids:
            # Build speaker mapping from filenames
            # Assumes format: speakerXXX_uttYYY.wav or similar
            speaker_names = set()
            for audio_path in self.samples:
                # Extract speaker name from filename (before first underscore or number)
                filename = audio_path.stem
                # Try different patterns
                if '_' in filename:
                    speaker_name = filename.split('_')[0]
                else:
                    # Use first 4 characters as speaker ID
                    speaker_name = filename[:4]
                speaker_names.add(speaker_name)

            # Create mapping
            self.speaker_to_idx = {name: idx for idx, name in enumerate(sorted(speaker_names))}
            print(f"Extracted {len(self.speaker_to_idx)} unique speakers from filenames")

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

        result = {'audio': audio.squeeze(0)}  # (T,)

        # Add speaker ID if requested
        if self.extract_speaker_ids:
            filename = audio_path.stem
            if '_' in filename:
                speaker_name = filename.split('_')[0]
            else:
                speaker_name = filename[:4]

            speaker_id = self.speaker_to_idx.get(speaker_name, 0)
            result['speaker_id'] = speaker_id

        return result


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


def contrastive_speaker_loss(model, audio, codes, speaker_embs, config, faiss_index=None, embedding_cache=None):
    """
    Contrastive speaker loss with FAISS-based hard negative mining for voice conversion.

    Goal: Model should learn to separate content (audio codes) from speaker identity (embedding).
    - When decoding codes from speaker A with embedding from speaker B, we should get
      audio with A's content but B's voice characteristics.

    Two modes:
    1. FAISS mode: Finds hard negatives from entire dataset using cached embeddings
    2. Fallback mode: Uses in-batch negatives with similarity-based selection

    FAISS mode advantages:
    - True hard negatives from entire 128K dataset (not just batch)
    - Hard negatives = speakers that sound similar to anchor (challenging to distinguish)
    - If model can distinguish these, it has learned good speaker disentanglement
    - No reliance on potentially incorrect speaker labels
    """
    B = audio.shape[0]
    device = audio.device
    original_lengths = [audio[i].shape[-1] for i in range(B)]

    recon_losses = []
    num_negatives = config.get('max_negatives', 6)

    use_faiss = faiss_index is not None and embedding_cache is not None

    for i in range(B):
        codes_i = [c[i:i+1] for c in codes]

        # Positive: decode with own speaker embedding (should reconstruct well)
        speaker_emb_positive = speaker_embs[i:i+1]
        audio_positive = model.decode(codes_i, speaker_embedding=speaker_emb_positive)
        audio_positive = audio_positive[..., :original_lengths[i]]
        loss_positive = reconstruction_loss(audio[i:i+1], audio_positive, config)
        recon_losses.append(loss_positive)

        # Negatives: decode with other speaker embeddings (should reconstruct poorly)
        # This teaches model: content comes from codes, voice comes from speaker embedding
        if use_faiss:
            # Use FAISS to find hard negatives - speakers with similar embeddings to anchor
            query_emb = speaker_embs[i].cpu().numpy()

            # Get indices of hard negatives (similarity search)
            k = num_negatives + 1  # +1 to account for self being returned
            distances, indices, _ = faiss_index.search(query_emb, k=k)

            # Filter out self and use cached embeddings
            hard_neg_embs = []
            for idx in indices:
                if idx >= 0 and idx < len(faiss_index.file_paths):
                    neg_path = faiss_index.file_paths[idx]
                    if neg_path in embedding_cache:
                        # Use cached embedding directly (no need to reload audio)
                        emb = embedding_cache[neg_path]
                        hard_neg_embs.append(emb)

                        if len(hard_neg_embs) >= num_negatives:
                            break

            # Decode with hard negative speaker embeddings
            for emb in hard_neg_embs[:num_negatives]:
                speaker_emb_negative = emb.unsqueeze(0).to(device)
                audio_negative = model.decode(codes_i, speaker_embedding=speaker_emb_negative)
                audio_negative = audio_negative[..., :original_lengths[i]]
                loss_negative = reconstruction_loss(audio[i:i+1], audio_negative, config)
                recon_losses.append(loss_negative)

        # Fallback: in-batch hard negative mining (same as before)
        if len(recon_losses) < 1 + num_negatives:
            # Compute similarity matrix
            similarity_matrix = F.cosine_similarity(speaker_embs.unsqueeze(1), speaker_embs.unsqueeze(0), dim=-1)
            same_speaker_threshold = config.get('same_speaker_threshold', 0.85)

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
                output_dir=None, epoch=0, start_step=0, scheduler_gen=None, scheduler_disc=None,
                use_ddp=False, faiss_index=None, embedding_cache=None, speaker_disc=None,
                opt_speaker_disc=None):
    """Train for one epoch with GAN + contrastive loss."""
    model.train()
    mpd.train()
    mrd.train()
    if speaker_disc is not None:
        speaker_disc.train()

    total_loss_gen = 0
    total_loss_disc = 0
    total_loss_recon = 0
    total_loss_contrast = 0
    total_loss_speaker_match = 0
    total_loss_speaker_match_identity = 0
    total_loss_speaker_match_vc = 0
    total_loss_speaker_match_synth = 0
    total_loss_synthetic_vc = 0
    total_loss_adv = 0
    total_loss_fm = 0
    total_loss_codebook_adv = 0
    num_batches = 0

    use_contrastive = config.get('contrastive_weight', 0) > 0
    use_gan = config.get('gan_weight', 0) > 0
    use_codebook_adv = speaker_disc is not None and config.get('use_codebook_adversarial', False)
    save_every_steps = config.get('save_every_steps', 4000)

    # Loss weights
    lambda_adv = config.get('lambda_adv', 1.0)
    lambda_fm = config.get('lambda_fm', 2.0)
    lambda_contrast = config.get('contrastive_weight', 0.5)
    lambda_codebook_adv = config.get('lambda_codebook_adv', 0.1)

    pbar = tqdm(dataloader, desc="Training", disable=not use_ddp or dist.get_rank() == 0)
    for batch in pbar:
        audio = batch['audio'].to(device)  # (B, T)
        audio = audio.unsqueeze(1)  # (B, 1, T)
        B = audio.shape[0]

        # Get speaker IDs if available (for adversarial codebook loss)
        speaker_ids = batch.get('speaker_id', None)
        if speaker_ids is not None:
            speaker_ids = speaker_ids.to(device)

        # ===== Generator Forward =====
        opt_gen.zero_grad()

        # Extract speaker embeddings
        # When using DDP, access underlying model via .module attribute
        model_base = model.module if use_ddp else model
        speaker_embs = model_base.base_model.extract_speaker_embedding(audio)
        codes = model_base.encode(audio)

        # Reconstruction with speaker conditioning
        # Note: model.forward() does preprocessing internally and trims to original length
        audio_hat, _ = model_base(audio, speaker_embedding=speaker_embs)

        # Reconstruction loss (main forward pass)
        loss_recon = reconstruction_loss(audio, audio_hat, config)

        # Voice conversion loss
        # This includes: reconstruction + speaker_matching
        # Content preservation is implicit (codes from source audio)
        vc_losses = {'total': torch.tensor(0.0, device=device),
                    'reconstruction': torch.tensor(0.0, device=device),
                    'speaker_matching': torch.tensor(0.0, device=device)}
        if use_contrastive:
            vc_losses = voice_conversion_loss(
                model_base, audio, codes, speaker_embs, config,
                faiss_index=faiss_index,
                embedding_cache=embedding_cache
            )

        # Synthetic voice conversion loss (audio augmentation)
        # Apply pitch shifting to create pseudo-speaker pairs
        # Teaches model to trust embedding over acoustic patterns in codes
        synthetic_vc_loss = torch.tensor(0.0, device=device)
        synthetic_speaker_losses = []  # Track speaker matching loss for synthetic VC
        use_synthetic_vc = config.get('use_synthetic_vc', False)
        if use_synthetic_vc and num_batches > 10:  # Warmup period
            import time as time_module
            synthetic_time = 0

            # Apply augmentation to random subset
            synthetic_aug_prob = config.get('synthetic_vc_probability', 0.5)
            pitch_shift_range = config.get('pitch_shift_range', [-2, -1, 1, 2])

            # Process each sample in batch
            synthetic_losses = []
            for i in range(audio.shape[0]):
                if random.random() < synthetic_aug_prob:
                    t0 = time_module.time()
                    # Apply pitch shift and formant shift
                    audio_i = audio[i:i+1]  # (1, 1, T)
                    t1 = time_module.time()

                    # Check if formant shifting is enabled
                    use_formant = config.get('use_formant_shift', False)
                    formant_range = config.get('formant_shift_range', [])

                    if use_formant and formant_range:
                        # Advanced augmentation with formant shift
                        audio_aug, was_aug, semitones, formant_shift = augment_audio_for_voice_conversion_advanced(
                            audio_i,
                            pitch_shift_range=pitch_shift_range,
                            formant_shift_range=formant_range,
                            probability=1.0  # Always augment in this loop
                        )
                    else:
                        # Regular pitch shift only
                        audio_aug, was_aug, semitones = augment_audio_for_voice_conversion(
                            audio_i,
                            pitch_shift_range=pitch_shift_range,
                            probability=1.0  # Always augment in this loop
                        )
                        formant_shift = 0.0
                    t2 = time_module.time()

                    if was_aug:
                        # Encode augmented audio
                        codes_aug = model_base.encode(audio_aug)
                        t3 = time_module.time()

                        # Reconstruct with ORIGINAL speaker embedding
                        # This teaches: embedding dominates codes for speaker identity
                        speaker_emb_i = speaker_embs[i:i+1]

                        # Get length for trimming
                        original_length = audio[i].shape[-1]

                        # Decode with original embedding
                        audio_synthetic_recon = model_base.decode(codes_aug, speaker_embedding=speaker_emb_i)
                        audio_synthetic_recon = audio_synthetic_recon[..., :original_length]
                        t4 = time_module.time()

                        # Reconstruction loss: should match ORIGINAL audio (not augmented)
                        loss_synth = reconstruction_loss(audio[i:i+1], audio_synthetic_recon, config)
                        t5 = time_module.time()

                        # Speaker matching loss: decoded audio should match ORIGINAL speaker embedding
                        # This teaches: even with pitch-shifted codes, original embedding restores speaker
                        from snac.voice_conversion_loss import speaker_matching_loss
                        loss_synth_spk = speaker_matching_loss(
                            model_base.base_model, audio_synthetic_recon, speaker_emb_i, config
                        )
                        synthetic_speaker_losses.append(loss_synth_spk)

                        synthetic_losses.append(loss_synth)

                        pitch_time = (t1 - t0) * 1000
                        aug_time = (t2 - t1) * 1000
                        enc_time = (t3 - t2) * 1000
                        dec_time = (t4 - t3) * 1000
                        loss_time = (t5 - t4) * 1000
                        total = (t5 - t0) * 1000

                        synthetic_time += total

            if synthetic_losses:
                synthetic_vc_loss = torch.stack(synthetic_losses).mean()

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

        # Codebook adversarial loss (remove speaker info from codes)
        loss_codebook_adv_enc = torch.tensor(0.0, device=device)
        loss_codebook_adv_disc = torch.tensor(0.0, device=device)

        if use_codebook_adv and num_batches > 20 and speaker_ids is not None:
            # Apply gradient reversal to codes for encoder loss
            # This makes encoder produce codes that fool the speaker discriminator
            speaker_disc_base = speaker_disc.module if use_ddp else speaker_disc

            # Compute adversarial loss
            adv_losses = adversarial_codebook_loss_v2(
                codes=codes,
                speaker_discriminator=speaker_disc_base,
                speaker_ids=speaker_ids,
                lambda_adv=lambda_codebook_adv,
                grad_rev_lambda=config.get('grad_rev_lambda', 1.0),
                mode='encoder'  # Train encoder to fool discriminator
            )
            loss_codebook_adv_enc = adv_losses['encoder_loss']

        # Total generator loss
        # Note: vc_losses['total'] already includes its own weighted reconstruction
        # We add the main reconstruction loss separately to ensure good quality
        lambda_synthetic = config.get('lambda_synthetic', 0.3)
        loss_gen = (loss_recon + vc_losses['total'] +
                    lambda_synthetic * synthetic_vc_loss +
                    lambda_adv * loss_adv + lambda_fm * loss_fm +
                    loss_codebook_adv_enc)

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

        # ===== Speaker Discriminator Forward (Codebook Adversarial) =====
        if use_codebook_adv and num_batches > 20 and speaker_ids is not None and opt_speaker_disc is not None:
            opt_speaker_disc.zero_grad()

            speaker_disc_base = speaker_disc.module if use_ddp else speaker_disc

            # Train speaker discriminator to predict speaker from codes
            adv_losses = adversarial_codebook_loss_v2(
                codes=codes,
                speaker_discriminator=speaker_disc_base,
                speaker_ids=speaker_ids,
                lambda_adv=lambda_codebook_adv,
                grad_rev_lambda=config.get('grad_rev_lambda', 1.0),
                mode='disc'  # Train discriminator
            )
            loss_codebook_adv_disc = adv_losses['disc_loss']

            loss_codebook_adv_disc.backward()

            if config.get('grad_clip_speaker_disc', 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(speaker_disc.parameters(), config['grad_clip_speaker_disc'])
            opt_speaker_disc.step()

        # Update metrics
        total_loss_gen += loss_gen.item()
        total_loss_disc += loss_disc.item()
        total_loss_recon += loss_recon.item()
        total_loss_contrast += vc_losses['total'].item()
        total_loss_speaker_match += vc_losses['speaker_matching'].item()
        total_loss_speaker_match_identity += vc_losses['speaker_matching_identity'].item()
        total_loss_speaker_match_vc += vc_losses['speaker_matching_vc'].item()
        # Track synthetic speaker loss separately
        synthetic_speaker_loss_avg = torch.stack(synthetic_speaker_losses).mean().item() if synthetic_speaker_losses else 0.0
        total_loss_speaker_match_synth += synthetic_speaker_loss_avg
        total_loss_synthetic_vc += synthetic_vc_loss.item()
        total_loss_adv += loss_adv.item()
        total_loss_fm += loss_fm.item()
        total_loss_codebook_adv += loss_codebook_adv_enc.item()
        num_batches += 1

        # Progress bar
        postfix = {
            'g_loss': loss_gen.item(),
            'd_loss': loss_disc.item(),
            'recon': loss_recon.item()
        }
        if use_contrastive:
            postfix['synth'] = synthetic_vc_loss.item()
            postfix['vc'] = vc_losses['total'].item()
            # Show separate speaker losses
            postfix['spk_id'] = vc_losses['speaker_matching_identity'].item()
            postfix['spk_vc'] = vc_losses['speaker_matching_vc'].item()
            postfix['spk_synth'] = synthetic_speaker_loss_avg
            postfix['spk'] = vc_losses['speaker_matching'].item()
        if use_gan and num_batches > 10:
            postfix['adv'] = loss_adv.item()
            postfix['fm'] = loss_fm.item()
        if use_codebook_adv and num_batches > 20:
            postfix['cbk'] = loss_codebook_adv_enc.item()
        pbar.set_postfix(postfix)

        # Log to file (every batch)
        pbar.write(f"Epoch {epoch}, Batch {num_batches}/{len(dataloader)}: " + ", ".join([f"{k}={v:.4f}" for k, v in postfix.items()]))
        sys.stdout.flush()  # Force flush to avoid buffering with nohup

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
            if speaker_disc is not None:
                checkpoint['speaker_disc_state_dict'] = speaker_disc.state_dict()
                if opt_speaker_disc is not None:
                    checkpoint['opt_speaker_disc_state_dict'] = opt_speaker_disc.state_dict()
            checkpoint_path = output_dir / f'step_{current_step}.pt'
            torch.save(checkpoint, checkpoint_path)
            pbar.write(f"Saved checkpoint at step {current_step}")
            sys.stdout.flush()  # Force flush to avoid buffering with nohup

    # Return average losses and total steps
    return {
        'gen': total_loss_gen / num_batches,
        'disc': total_loss_disc / num_batches,
        'recon': total_loss_recon / num_batches,
        'contrast': total_loss_contrast / num_batches,
        'speaker_match': total_loss_speaker_match / num_batches,
        'speaker_match_identity': total_loss_speaker_match_identity / num_batches,
        'speaker_match_vc': total_loss_speaker_match_vc / num_batches,
        'speaker_match_synth': total_loss_speaker_match_synth / num_batches,
        'synthetic_vc': total_loss_synthetic_vc / num_batches,
        'adv': total_loss_adv / num_batches,
        'fm': total_loss_fm / num_batches,
        'codebook_adv': total_loss_codebook_adv / num_batches,
        'total_steps': start_step + num_batches,
    }


@torch.no_grad()
def evaluate(model, mpd, mrd, dataloader, device, config, use_ddp=False):
    """Evaluate on validation set."""
    model.eval()
    mpd.eval()
    mrd.eval()

    # When using DDP, access underlying model via .module attribute
    model_base = model.module if use_ddp else model

    total_loss_gen = 0
    total_loss_recon = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating", disable=use_ddp and dist.get_rank() != 0):
        audio = batch['audio'].to(device)
        audio = audio.unsqueeze(1)

        # Extract speaker embeddings
        speaker_embs = model_base.base_model.extract_speaker_embedding(audio)

        # Forward pass (no GAN loss on validation for speed)
        audio_hat, _ = model_base(model_base.preprocess(audio), speaker_embedding=speaker_embs)

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


def setup_ddp():
    """Initialize DDP environment."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_ddp():
    """Cleanup DDP environment."""
    dist.destroy_process_group()


def main(config, resume_path=None, device_id=0, use_ddp=False):
    """Main training function with optional DDP support."""
    if use_ddp:
        local_rank, rank, world_size = setup_ddp()
        device = torch.device(f'cuda:{local_rank}')
        is_main = rank == 0
    else:
        device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
        local_rank = device_id
        rank = 0
        world_size = 1
        is_main = True

    if is_main:
        print(f"Using device: {device}")
        if use_ddp:
            print(f"DDP: rank={rank}, world_size={world_size}")

    # Create output directory (only on main process)
    output_dir = Path(config['output_dir'])
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create base model (frozen)
    base_model = SNACWithSpeakerConditioning.from_pretrained_base(
        repo_id=config['pretrained_model'],
        speaker_emb_dim=config['speaker_emb_dim'],
        speaker_encoder_type=config.get('speaker_encoder_type', 'eres2net'),
        freeze_base=config['freeze_base'],
    ).to(device)

    # Wrap with AdapterWrapper for conditioning BEFORE VQ
    model = AdapterWrapper(
        base_model=base_model,
        adapter_type=config.get('adapter_type', 'film'),
        adapter_hidden_dim=config.get('adapter_hidden_dim', 512),
        adapter_num_layers=config.get('adapter_num_layers', 2),
    ).to(device)

    # Create discriminators
    mpd = MultiPeriodDiscriminator(periods=config.get('mpd_periods', [2, 3, 5, 7, 11])).to(device)
    mrd = MultiResolutionSTFTDiscriminator(fft_sizes=config.get('mrd_fft_sizes', [1024, 2048, 4096])).to(device)

    # Create speaker discriminator for codebook adversarial training
    # (Will be initialized after we know the number of speakers from dataset)
    speaker_disc = None

    if is_main:
        adapter_params = sum(p.numel() for p in model.adapter.parameters())
        base_params = sum(p.numel() for p in model.base_model.parameters())
        total_params = adapter_params + base_params
        print(f"Adapter parameters: {adapter_params:,}")
        print(f"Base model (frozen): {base_params:,} parameters")
        print(f"Model total: {total_params:,} parameters")
        print(f"MPD: {sum(p.numel() for p in mpd.parameters()):,} parameters")
        print(f"MRD: {sum(p.numel() for p in mrd.parameters()):,} parameters")

    # Wrap with DDP if enabled
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        mpd = DDP(mpd, device_ids=[local_rank], output_device=local_rank)
        mrd = DDP(mrd, device_ids=[local_rank], output_device=local_rank)
        if speaker_disc is not None:
            speaker_disc = DDP(speaker_disc, device_ids=[local_rank], output_device=local_rank)

    # Resume from checkpoint if specified
    start_epoch = 0
    start_step = 0
    if resume_path and Path(resume_path).exists():
        if is_main:
            print(f"Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

        # Remove 'module.' prefix from DDP checkpoints if present
        model_state = checkpoint['model_state_dict']
        mpd_state = checkpoint['mpd_state_dict']
        mrd_state = checkpoint['mrd_state_dict']

        if use_ddp:
            # If loading from non-DDP checkpoint, add 'module.' prefix
            if not list(model_state.keys())[0].startswith('module.'):
                model_state = {f'module.{k}': v for k, v in model_state.items()}
                mpd_state = {f'module.{k}': v for k, v in mpd_state.items()}
                mrd_state = {f'module.{k}': v for k, v in mrd_state.items()}
        else:
            # If loading from DDP checkpoint, remove 'module.' prefix
            if list(model_state.keys())[0].startswith('module.'):
                model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
                mpd_state = {k.replace('module.', ''): v for k, v in mpd_state.items()}
                mrd_state = {k.replace('module.', ''): v for k, v in mrd_state.items()}

        model.load_state_dict(model_state)
        mpd.load_state_dict(mpd_state)
        mrd.load_state_dict(mrd_state)
        start_epoch = checkpoint.get('epoch', 0) + 1
        start_step = checkpoint.get('step', 0)
        if is_main:
            print(f"Resumed from epoch {checkpoint.get('epoch', 0)}, step {start_step}")

    # Check if using adversarial codebook purification
    use_codebook_adv = config.get('use_codebook_adversarial', False) and HAS_CODEBOOK_ADV
    if use_codebook_adv and not HAS_CODEBOOK_ADV:
        print("⚠ Codebook adversarial loss requested but module not available. Disabling.")

    # Create datasets
    train_dataset = SimpleAudioDataset(
        dataset_root=config['train_data'],
        sampling_rate=model.sampling_rate if not use_ddp else model.module.sampling_rate,
        segment_length=config['segment_length'],
        augment=True,
        extract_speaker_ids=use_codebook_adv,  # Extract speaker IDs if using adversarial loss
    )
    val_dataset = SimpleAudioDataset(
        dataset_root=config['val_data'],
        sampling_rate=model.sampling_rate if not use_ddp else model.module.sampling_rate,
        segment_length=config['segment_length'],
        augment=False,
        extract_speaker_ids=use_codebook_adv,
    )

    # Get number of speakers for discriminator
    num_speakers = 1
    if use_codebook_adv and len(train_dataset.speaker_to_idx) > 0:
        num_speakers = len(train_dataset.speaker_to_idx)
        if is_main:
            print(f"Adversarial codebook purification: {num_speakers} speakers")

        # Create speaker discriminator
        # SNAC has 4 codebooks with different dimensions, we'll use the codebook sizes
        # For now, assume standard SNAC codebook dims (will be updated based on actual codes)
        codebook_dims = config.get('codebook_dims', [512, 512, 512, 512])
        speaker_disc = SpeakerDiscriminator(
            codebook_dims=codebook_dims,
            num_speakers=num_speakers,
            hidden_dim=config.get('speaker_disc_hidden_dim', 256)
        ).to(device)

        if is_main:
            print(f"Speaker Discriminator: {sum(p.numel() for p in speaker_disc.parameters()):,} parameters")

    # Create samplers for DDP
    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        shuffle=shuffle_train if not use_ddp else None,
        num_workers=config['num_workers'],
        pin_memory=config.get('pin_memory', True),
        prefetch_factor=config.get('prefetch_factor', 2) if config['num_workers'] > 0 else None,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=config.get('pin_memory', True),
        prefetch_factor=config.get('prefetch_factor', 2) if config['num_workers'] > 0 else None
    )

    # Optimizers (only train adapter parameters, base model is frozen)
    trainable_params = model.adapter.parameters()
    opt_gen = AdamW(
        trainable_params,
        lr=config['learning_rate'],
        betas=(0.5, 0.9),  # GAN standard
        weight_decay=config.get('weight_decay', 1e-5),
    )

    opt_disc = AdamW(
        list(mpd.parameters()) + list(mrd.parameters()),
        lr=config['disc_learning_rate'],
        betas=(0.5, 0.9),
    )

    # Speaker discriminator optimizer (for codebook adversarial training)
    opt_speaker_disc = None
    if speaker_disc is not None:
        opt_speaker_disc = AdamW(
            speaker_disc.parameters(),
            lr=config.get('speaker_disc_lr', config['disc_learning_rate']),
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
    scheduler_speaker_disc = None
    if opt_speaker_disc is not None:
        scheduler_speaker_disc = CosineAnnealingLR(
            opt_speaker_disc, T_max=config['num_epochs'],
            eta_min=config.get('speaker_disc_lr', config['disc_learning_rate']) * config.get('lr_min_ratio', 0.01),
        )

    # Resume optimizers if checkpoint exists
    if resume_path and Path(resume_path).exists():
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
        opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
        scheduler_gen.load_state_dict(checkpoint['scheduler_gen_state_dict'])
        scheduler_disc.load_state_dict(checkpoint['scheduler_disc_state_dict'])
        if opt_speaker_disc is not None and 'opt_speaker_disc_state_dict' in checkpoint:
            opt_speaker_disc.load_state_dict(checkpoint['opt_speaker_disc_state_dict'])
        if scheduler_speaker_disc is not None and 'scheduler_speaker_disc_state_dict' in checkpoint:
            scheduler_speaker_disc.load_state_dict(checkpoint['scheduler_speaker_disc_state_dict'])
        for _ in range(start_epoch):
            scheduler_gen.step()
            scheduler_disc.step()
            if scheduler_speaker_disc is not None:
                scheduler_speaker_disc.step()

    # Load FAISS index and optimized embedding cache
    faiss_index = None
    embedding_cache = None

    if config.get('use_faiss_hard_negatives', False) and is_main:
        faiss_index_path = config.get('faiss_index_path', 'pretrained_models/speaker_faiss.index')
        embedding_cache_path = config.get('embedding_cache_path', 'pretrained_models/embeddings_cache.npy')

        # Try to load optimized cache first (instant)
        if Path(embedding_cache_path).exists() or Path(embedding_cache_path).with_suffix('.npy').exists():
            print(f"\n✅ Loading optimized embedding cache from {embedding_cache_path}...")
            embedding_cache = OptimizedEmbeddingCache.load(embedding_cache_path)
            print("✅ Embedding cache loaded (instant, memory-mapped)")

        # Then load FAISS index
        if Path(faiss_index_path).exists():
            print(f"\nLoading FAISS index from {faiss_index_path}...")
            # Pass base_model which has speaker_encoder
            model_for_faiss = model.module if use_ddp else model
            model_for_faiss = model_for_faiss.base_model  # AdapterWrapper's base_model
            faiss_index = FaissSpeakerIndex(model_for_faiss, device)
            faiss_index.load(faiss_index_path)
            print("✅ FAISS index loaded")

        else:
            print(f"\n⚠ FAISS index not found at {faiss_index_path}")
            print(f"To enable FAISS hard negatives:")
            print(f"  1. Run: uv run python scripts/build_embedding_cache.py")
            print(f"  2. Ensure faiss_index_path points to valid index")

        if embedding_cache is None:
            print(f"\n⚠ Optimized embedding cache not found at {embedding_cache_path}")
            print(f"Building optimized cache will take 30+ minutes...")
            print(f"To build cache once:")
            print(f"  uv run python scripts/build_embedding_cache.py --output {embedding_cache_path}")
            print(f"\nFalling back to in-batch hard negatives (less effective)")
            faiss_index = None  # Disable FAISS without cache


    # Training loop
    best_val_loss = float('inf')
    patience = config.get('early_stopping_patience', 10)
    epochs_without_improvement = 0

    for epoch in range(start_epoch, config['num_epochs']):
        if is_main:
            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
            print(f"Learning rate: {opt_gen.param_groups[0]['lr']:.2e}")

        # Set epoch for DistributedSampler
        if use_ddp:
            train_sampler.set_epoch(epoch)

        # Train
        train_metrics = train_epoch(model, mpd, mrd, train_loader, opt_gen, opt_disc, device, config,
                                    output_dir=output_dir, epoch=epoch, start_step=start_step,
                                    scheduler_gen=scheduler_gen, scheduler_disc=scheduler_disc,
                                    use_ddp=use_ddp, faiss_index=faiss_index, embedding_cache=embedding_cache,
                                    speaker_disc=speaker_disc, opt_speaker_disc=opt_speaker_disc)
        start_step = train_metrics['total_steps']  # Update for next epoch
        if is_main:
            train_str = (f"Train - G: {train_metrics['gen']:.4f}, D: {train_metrics['disc']:.4f}, "
                        f"Recon: {train_metrics['recon']:.4f}")
            if train_metrics['contrast'] > 0:
                train_str += (f", VC: {train_metrics['contrast']:.4f}, "
                            f"SpkMatch: {train_metrics['speaker_match']:.4f}")
            if train_metrics['synthetic_vc'] > 0:
                train_str += f", SynthVC: {train_metrics['synthetic_vc']:.4f}"
            if train_metrics['adv'] > 0:
                train_str += f", Adv: {train_metrics['adv']:.4f}, FM: {train_metrics['fm']:.4f}"
            if train_metrics['codebook_adv'] > 0:
                train_str += f", CbkAdv: {train_metrics['codebook_adv']:.4f}"
            print(train_str)

        # Validate
        val_metrics = evaluate(model, mpd, mrd, val_loader, device, config, use_ddp=use_ddp)
        if is_main:
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
            if is_main:
                print(f"Early stopping at epoch {epoch + 1}")
            break

        # Save checkpoint only on main process
        if is_main:
            checkpoint = {
                'epoch': epoch,
                'step': start_step,
                'model_state_dict': model.module.state_dict() if use_ddp else model.state_dict(),
                'mpd_state_dict': mpd.module.state_dict() if use_ddp else mpd.state_dict(),
                'mrd_state_dict': mrd.module.state_dict() if use_ddp else mrd.state_dict(),
                'opt_gen_state_dict': opt_gen.state_dict(),
                'opt_disc_state_dict': opt_disc.state_dict(),
                'scheduler_gen_state_dict': scheduler_gen.state_dict(),
                'scheduler_disc_state_dict': scheduler_disc.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': config,
            }
            if speaker_disc is not None:
                checkpoint['speaker_disc_state_dict'] = speaker_disc.module.state_dict() if use_ddp else speaker_disc.state_dict()
            if opt_speaker_disc is not None:
                checkpoint['opt_speaker_disc_state_dict'] = opt_speaker_disc.state_dict()
            if scheduler_speaker_disc is not None:
                checkpoint['scheduler_speaker_disc_state_dict'] = scheduler_speaker_disc.state_dict()

            torch.save(checkpoint, output_dir / 'latest.pt')
            if is_best:
                torch.save(checkpoint, output_dir / 'best.pt')

            if (epoch + 1) % config.get('save_every', 10) == 0:
                torch.save(checkpoint, output_dir / f'epoch_{epoch+1}.pt')

            print(f"Saved checkpoint (best_gen_loss: {best_val_loss:.4f})")

    # Cleanup DDP
    if use_ddp:
        cleanup_ddp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/phase4_gan.json')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--ddp', action='store_true', help='Enable DDP (DistributedDataParallel)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config, args.resume, args.device, use_ddp=args.ddp)

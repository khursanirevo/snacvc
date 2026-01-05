#!/usr/bin/env python3
"""
Voice Conversion Loss Module.

Implements training objectives for voice conversion:
- Reconstruction Loss: Codes + own embedding should reconstruct original audio
- Speaker Matching Loss: Output should match target speaker's characteristics

Key insight:
- Codes contain both content AND speaker information (by SNAC design)
- FiLM conditioning transforms features during decode: speaker_A → speaker_B
- Content is preserved because codes come from source audio
- Speaker is controlled by the embedding via FiLM modulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def speaker_matching_loss(model, audio_decoded, target_speaker_emb, config):
    """
    Speaker Matching Loss (Component A).

    Forces the decoded audio to have the target speaker's characteristics.

    Args:
        model: SNAC model with speaker encoder
        audio_decoded: Decoded audio (B, 1, T)
        target_speaker_emb: Target speaker embedding (B, D)
        config: Training config

    Returns:
        Loss value (lower is better)
    """
    # Extract speaker embedding from decoded audio
    decoded_speaker_emb = model.extract_speaker_embedding(audio_decoded)

    # Normalize both embeddings
    decoded_speaker_emb = F.normalize(decoded_speaker_emb, dim=-1)
    target_speaker_emb = F.normalize(target_speaker_emb, dim=-1)

    # Cosine similarity loss (we want high similarity = low loss)
    similarity = F.cosine_similarity(decoded_speaker_emb, target_speaker_emb, dim=-1)

    # Convert to loss (1 - similarity)
    loss = (1.0 - similarity).mean()

    return loss


def content_preservation_loss(model, audio_decoded, original_codes, config):
    """
    Content Preservation Loss (Component B).

    Ensures the content (codes) is preserved through the decode-recode cycle.

    Key idea:
    - Encode the decoded audio back to codes
    - Compare to original codes
    - If content is preserved, codes should be similar

    Args:
        model: SNAC model
        audio_decoded: Decoded audio (B, 1, T)
        original_codes: Original SNAC codes (list of tensors)
        config: Training config

    Returns:
        Loss value (lower is better)
    """
    # Re-encode the decoded audio
    recon_codes = model.encode(audio_decoded)

    # Compute loss for each codebook level
    losses = []
    for i, (code_orig, code_recon) in enumerate(zip(original_codes, recon_codes)):
        # Codes are discrete (Long), convert to float for L1 loss
        # This encourages the re-encoded codes to match original codes
        loss = F.l1_loss(code_recon.float(), code_orig.detach().float())

        # Optional: Weight different codebook levels differently
        # Coarser levels (lower i) capture more content, finer levels (higher i) capture details
        weight = config.get(f'codebook_weight_{i}', 1.0)
        losses.append(weight * loss)

    return sum(losses) / len(losses) if losses else torch.tensor(0.0, device=audio_decoded.device)


def voice_conversion_loss(model, audio, codes, speaker_embs, config, faiss_index=None, embedding_cache=None):
    """
    Voice Conversion Loss.

    Two components:
    1. Reconstruction: Own codes + own embedding = should reconstruct well
    2. Speaker matching: Codes + target embedding = should match target speaker

    This teaches the model:
    - FiLM learns identity transformation (own embedding)
    - FiLM learns speaker transformation (target embedding)
    - Content preservation is implicit (codes come from source audio)

    Args:
        model: SNAC model with speaker encoder
        audio: Original audio (B, 1, T)
        codes: Encodes SNAC codes (list of tensors)
        speaker_embs: Speaker embeddings (B, D)
        config: Training config
        faiss_index: Optional FAISS index for hard negatives
        embedding_cache: Cached embeddings for FAISS

    Returns:
        Dict of loss values
    """
    B = audio.shape[0]
    device = audio.device
    model_base = model.module if hasattr(model, 'module') else model

    original_lengths = [audio[i].shape[-1] for i in range(B)]
    num_negatives = config.get('max_negatives', 6)

    # Extract individual loss components (separated by type)
    reconstruction_losses = []
    speaker_matching_identity_losses = []  # Own embedding
    speaker_matching_vc_losses = []  # Hard negatives

    # Loss weights
    lambda_recon = config.get('lambda_recon', 1.0)
    lambda_speaker = config.get('lambda_speaker_matching', 0.5)

    # Use FAISS for hard negatives?
    use_faiss = faiss_index is not None and embedding_cache is not None
    use_stratified_sampling = config.get('use_stratified_negatives', False)

    for i in range(B):
        # ===== Positive: Encode with own embedding, decode =====
        speaker_emb_positive = speaker_embs[i:i+1]
        # For Phase 6: Encode WITH target speaker embedding to get conditioned codes
        audio_i = audio[i:i+1]

        # ===== Identity: Compare conditioned vs unconditioned =====
        # This measures: Does the adapter preserve identity relative to base model?

        # Conditioned reconstruction (with own embedding)
        codes_cond = model_base.encode(audio_i, speaker_embedding=speaker_emb_positive)
        audio_cond = model_base.decode(codes_cond)
        audio_cond = audio_cond[..., :original_lengths[i]]

        # Unconditioned reconstruction (no speaker embedding)
        codes_uncond = model_base.encode(audio_i, speaker_embedding=None)
        audio_uncond = model_base.decode(codes_uncond)
        audio_uncond = audio_uncond[..., :original_lengths[i]]

        # 1. Reconstruction loss (conditioned should match original)
        loss_recon = reconstruction_loss(audio_i, audio_cond, config)
        reconstruction_losses.append(loss_recon)

        # 2. Identity loss: conditioned vs unconditioned (should match!)
        # Both from same audio, so speaker characteristics should be similar
        emb_cond = model_base.extract_speaker_embedding(audio_cond)
        emb_uncond = model_base.extract_speaker_embedding(audio_uncond)

        # Normalize and compute cosine similarity
        emb_cond = F.normalize(emb_cond, dim=-1)
        emb_uncond = F.normalize(emb_uncond, dim=-1)
        similarity = F.cosine_similarity(emb_cond, emb_uncond, dim=-1)

        # Loss = 1 - similarity (lower is better)
        loss_speaker_match_identity = (1.0 - similarity).mean()
        speaker_matching_identity_losses.append(loss_speaker_match_identity)

        # Keep audio_positive for VC negatives
        audio_positive = audio_cond

        # ===== Negatives: Own codes + other embeddings =====
        if use_faiss:
            if use_stratified_sampling:
                # STRATIFIED SAMPLING: Sample from easy/medium/hard tiers
                from snac.stratified_hard_negatives import get_stratified_negatives_legacy

                hard_neg_embs = get_stratified_negatives_legacy(
                    model_base, speaker_embs[i], faiss_index, embedding_cache, config
                )
            else:
                # ORIGINAL: Just use most similar speakers
                query_emb = speaker_embs[i].detach().cpu().numpy()
                k = num_negatives + 1
                distances, indices, _ = faiss_index.search(query_emb, k=k)

                hard_neg_embs = []
                for idx in indices:
                    if idx >= 0 and idx < len(faiss_index.file_paths):
                        neg_path = faiss_index.file_paths[idx]
                        emb = embedding_cache.get(neg_path)
                        if emb is not None:
                            hard_neg_embs.append(emb)
                            if len(hard_neg_embs) >= num_negatives:
                                break

            # Process negatives
            for emb in hard_neg_embs[:num_negatives]:
                speaker_emb_negative = emb.unsqueeze(0).to(device)
                # For Phase 6: Encode audio with TARGET speaker embedding
                # This teaches: encode(audio_A, emb_B) → codes that decode to speaker B
                codes_negative = model_base.encode(audio_i, speaker_embedding=speaker_emb_negative)
                audio_negative = model_base.decode(codes_negative)
                audio_negative = audio_negative[..., :original_lengths[i]]

                # Speaker matching for VC: Should match target (negative) speaker
                loss_speaker_neg = speaker_matching_loss(
                    model_base, audio_negative, speaker_emb_negative, config
                )
                speaker_matching_vc_losses.append(loss_speaker_neg)

        # Fallback: In-batch negatives if FAISS not available or insufficient
        if len(speaker_matching_vc_losses) < num_negatives:
            # Compute similarity matrix
            similarity_matrix = F.cosine_similarity(
                speaker_embs.unsqueeze(1), speaker_embs.unsqueeze(0), dim=-1
            )
            same_speaker_threshold = config.get('same_speaker_threshold', 0.85)

            # Find hard negatives (similar but not same speaker)
            hard_mask = (similarity_matrix[i] >= 0.5) & (similarity_matrix[i] < same_speaker_threshold) & (torch.arange(B, device=device) != i)
            hard_indices = torch.where(hard_mask)[0]

            # Semi-hard negatives
            semi_hard_mask = (similarity_matrix[i] >= 0.3) & (similarity_matrix[i] < same_speaker_threshold) & (torch.arange(B, device=device) != i)
            semi_hard_indices = torch.where(semi_hard_mask)[0]

            # Select negatives
            selected_indices = []
            if len(hard_indices) > 0:
                perm = torch.randperm(len(hard_indices), device=device)
                selected_indices.append(hard_indices[perm][:num_negatives])

            remaining = num_negatives - len(selected_indices)
            if remaining > 0 and len(semi_hard_indices) > 0:
                semi_hard_remaining = semi_hard_indices[~torch.isin(semi_hard_indices, torch.cat(selected_indices) if selected_indices else torch.tensor([], device=device))]
                if len(semi_hard_remaining) > 0:
                    perm = torch.randperm(len(semi_hard_remaining), device=device)
                    selected_indices.append(semi_hard_remaining[perm][:remaining])

            if len(selected_indices) > 0:
                selected_indices = torch.cat(selected_indices)[:num_negatives]
            else:
                selected_indices = torch.tensor([j for j in range(B) if j != i], device=device)[:num_negatives]

            # Process in-batch negatives
            for j in selected_indices:
                speaker_emb_negative = speaker_embs[j:j+1]
                # For Phase 6: Encode audio with TARGET speaker embedding
                codes_negative = model_base.encode(audio_i, speaker_embedding=speaker_emb_negative)
                audio_negative = model_base.decode(codes_negative)
                audio_negative = audio_negative[..., :original_lengths[i]]

                # Speaker matching loss for VC
                loss_speaker_neg = speaker_matching_loss(
                    model_base, audio_negative, speaker_emb_negative, config
                )
                speaker_matching_vc_losses.append(loss_speaker_neg)

    # Aggregate losses
    loss_recon = torch.stack(reconstruction_losses).mean() if reconstruction_losses else torch.tensor(0.0, device=device)
    loss_speaker_identity = torch.stack(speaker_matching_identity_losses).mean() if speaker_matching_identity_losses else torch.tensor(0.0, device=device)
    loss_speaker_vc = torch.stack(speaker_matching_vc_losses).mean() if speaker_matching_vc_losses else torch.tensor(0.0, device=device)

    # Separate weights for identity vs VC speaker losses
    lambda_speaker_identity = config.get('lambda_speaker_identity', lambda_speaker * 0.5)  # Lower weight for identity
    lambda_speaker_vc = config.get('lambda_speaker_vc', lambda_speaker * 2.0)  # Higher weight for VC learning

    # Combined loss
    total_loss = (
        lambda_recon * loss_recon +
        lambda_speaker_identity * loss_speaker_identity +
        lambda_speaker_vc * loss_speaker_vc
    )

    # Compute weighted average for logging
    loss_speaker_combined = (
        lambda_speaker_identity * loss_speaker_identity +
        lambda_speaker_vc * loss_speaker_vc
    ) / (lambda_speaker_identity + lambda_speaker_vc)

    return {
        'total': total_loss,
        'reconstruction': loss_recon,
        'speaker_matching': loss_speaker_combined,
        'speaker_matching_identity': loss_speaker_identity,
        'speaker_matching_vc': loss_speaker_vc,
    }


def reconstruction_loss(audio_original, audio_reconstructed, config):
    """Reconstruction loss (L1 + multi-scale STFT)."""
    # L1 loss
    loss_l1 = F.l1_loss(audio_reconstructed, audio_original)

    # Multi-scale STFT loss
    loss_stft = 0.0
    n_ffts = config.get('n_ffts', [1024, 2048, 4096])

    for n_fft in n_ffts:
        hop_length = n_fft // 4

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

        loss_stft += F.l1_loss(stft_recon, stft_orig)

    loss_stft = loss_stft / len(n_ffts)

    # Combined
    loss = config.get('l1_weight', 1.0) * loss_l1 + config.get('stft_weight', 1.0) * loss_stft

    return loss

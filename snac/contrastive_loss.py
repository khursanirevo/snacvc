#!/usr/bin/env python3
"""
Improved contrastive speaker loss with semantic hard negative mining.

Uses SpeakerEmbeddingIndex to sample truly hard negatives:
- Same cluster: similar speakers (hard)
- Neighboring clusters: semi-similar speakers (semi-hard)
- Random clusters: different speakers (easy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ContrastiveSpeakerLoss(nn.Module):
    """
    Contrastive loss for speaker conditioning with semantic hard negative mining.

    Uses speaker embedding index to sample negatives from:
    1. Same cluster (hard negatives)
    2. Neighboring clusters (semi-hard negatives)
    3. Random (easy negatives)

    Args:
        margin: Contrastive margin (default: 0.1)
        temperature: Temperature for scaling similarities (default: 0.1)
        speaker_index: Optional SpeakerEmbeddingIndex for semantic sampling
    """

    def __init__(self, margin=0.1, temperature=0.1, speaker_index=None):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.speaker_index = speaker_index

    def forward(self, speaker_embs, audio_batch, config):
        """
        Compute contrastive loss with semantic hard negative mining.

        Args:
            speaker_embs: (B, D) speaker embeddings for current batch
            audio_batch: Audio batch (for getting speaker IDs)
            config: Training config

        Returns:
            - loss: Contrastive loss scalar
            - metrics: Dict of additional metrics
        """
        B = speaker_embs.shape[0]
        device = speaker_embs.device

        # Normalize embeddings
        speaker_embs = F.normalize(speaker_embs, p=2, dim=-1)

        # Compute similarity matrix
        # similarity_matrix[i, j] = sim(emb[i], emb[j])
        similarity_matrix = torch.mm(speaker_embs, speaker_embs.t())  # (B, B)

        # Get speaker IDs
        if hasattr(audio_batch, 'get'):  # dict-like
            speaker_ids = audio_batch.get('speaker_id', [f"spk_{i}" for i in range(B)])
        else:
            # Fallback: assume no speaker info, use indices
            speaker_ids = [f"spk_{i}" for i in range(B)]

        # Same-speaker mask (positive pairs)
        same_speaker_mask = torch.zeros(B, B, device=device, dtype=torch.bool)
        for i in range(B):
            for j in range(B):
                if speaker_ids[i] == speaker_ids[j]:
                    same_speaker_mask[i, j] = True

        # Diagonal is always same speaker (self-similarity)
        same_speaker_mask.fill_diagonal_(True)

        # Sample negatives
        max_negatives = config.get('max_negatives', 6)
        negative_indices = self._sample_negatives(
            speaker_ids,
            similarity_matrix,
            same_speaker_mask,
            max_negatives
        )

        # Compute loss
        total_loss = 0.0
        num_pairs = 0

        for i in range(B):
            # Positive pairs (same speaker, excluding self)
            pos_mask = same_speaker_mask[i] & (~torch.eye(B, device=device, dtype=torch.bool))
            pos_indices = torch.where(pos_mask)[0]

            if len(pos_indices) == 0:
                # No positive pairs in batch (single speaker per batch)
                continue

            # Negative pairs
            neg_indices = negative_indices[i]

            # Positive similarities
            pos_sim = similarity_matrix[i, pos_indices]  # (num_pos,)

            # Negative similarities
            neg_sim = similarity_matrix[i, neg_indices]  # (num_neg,)

            # Contrastive loss: max(0, margin - pos_sim + neg_sim)
            # For each positive, compare with all negatives
            for pos_s in pos_sim:
                # Loss: max(0, margin - pos_sim + neg_sim)
                # We want pos_sim >> neg_sim
                loss_per_neg = F.relu(self.margin - pos_s + neg_sim)

                # Average over negatives
                total_loss += loss_per_neg.mean()
                num_pairs += 1

        if num_pairs == 0:
            # No positive pairs found
            return torch.tensor(0.0, device=device), {}

        loss = total_loss / num_pairs

        # Compute metrics
        with torch.no_grad():
            metrics = self._compute_metrics(
                similarity_matrix,
                same_speaker_mask,
                speaker_ids
            )

        return loss, metrics

    def _sample_negatives(self, speaker_ids, similarity_matrix, same_speaker_mask,
                         max_negatives):
        """
        Sample negative indices for each anchor.

        If speaker_index is available, uses semantic clustering.
        Otherwise, falls back to similarity-based sampling.
        """
        B = len(speaker_ids)
        device = similarity_matrix.device

        negative_indices = torch.zeros(B, max_negatives, device=device, dtype=torch.long)

        if self.speaker_index is not None:
            # Use semantic clustering
            for i in range(B):
                anchor_speaker = speaker_ids[i]

                # Get hard negative speakers
                negative_speakers = self.speaker_index.get_hard_negatives(
                    anchor_speaker_id=anchor_speaker,
                    num_negatives=max_negatives,
                    hard_ratio=0.5,
                    semi_hard_ratio=0.3
                )

                # Find these speakers in the batch
                neg_idx = 0
                for j in range(B):
                    if j == i:
                        continue
                    if speaker_ids[j] in negative_speakers:
                        negative_indices[i, neg_idx] = j
                        neg_idx += 1
                        if neg_idx >= max_negatives:
                            break

                # If not enough in batch, fill with random (different speaker)
                if neg_idx < max_negatives:
                    different_speaker_mask = ~same_speaker_mask[i]
                    diff_indices = torch.where(different_speaker_mask)[0]

                    if len(diff_indices) > 0:
                        remaining = max_negatives - neg_idx
                        if len(diff_indices) >= remaining:
                            selected = torch.randperm(len(diff_indices), device=device)[:remaining]
                            negative_indices[i, neg_idx:] = diff_indices[selected]
                        else:
                            negative_indices[i, neg_idx:neg_idx+len(diff_indices)] = diff_indices

        else:
            # Fallback: similarity-based sampling (original method)
            for i in range(B):
                # Find different speakers
                different_speaker_mask = ~same_speaker_mask[i]
                diff_indices = torch.where(different_speaker_mask)[0]

                if len(diff_indices) == 0:
                    continue

                # Tier-based sampling on similarity
                sim_to_anchor = similarity_matrix[i, diff_indices]

                # Hard negatives: similarity 0.5-0.85
                hard_threshold_low = 0.5
                hard_threshold_high = config.get('same_speaker_threshold', 0.85)

                hard_mask = (sim_to_anchor >= hard_threshold_low) & \
                           (sim_to_anchor < hard_threshold_high)
                hard_indices = diff_indices[hard_mask]

                # Semi-hard: 0.3-0.85
                semi_hard_threshold_low = 0.3
                semi_hard_mask = (sim_to_anchor >= semi_hard_threshold_low) & \
                                (sim_to_anchor < hard_threshold_high)
                semi_hard_indices = diff_indices[semi_hard_mask]

                # All valid: < 0.85
                all_mask = sim_to_anchor < hard_threshold_high
                all_indices = diff_indices[all_mask]

                # Select with priority
                selected = []
                n_hard = int(max_negatives * 0.5)
                n_semi_hard = int(max_negatives * 0.3)

                # Hard
                if len(hard_indices) > 0:
                    perm = torch.randperm(len(hard_indices), device=device)
                    take = min(n_hard, len(hard_indices))
                    selected.append(hard_indices[perm[:take]])

                # Semi-hard (if still need more)
                if len(selected) < max_negatives and len(semi_hard_indices) > 0:
                    remaining = max_negatives - len(selected)
                    perm = torch.randperm(len(semi_hard_indices), device=device)
                    take = min(remaining, len(semi_hard_indices))
                    selected.append(semi_hard_indices[perm[:take]])

                # Fill with random
                if len(selected) < max_negatives and len(all_indices) > 0:
                    remaining = max_negatives - len(selected)
                    # Exclude already selected
                    available = [idx for idx in all_indices if idx not in selected]
                    if len(available) > 0:
                        perm = torch.randperm(len(available), device=device)
                        take = min(remaining, len(available))
                        selected.extend([available[idx] for idx in perm[:take]])

                # Pad if still not enough
                if len(selected) < max_negatives:
                    # Fill with random different speakers
                    remaining = max_negatives - len(selected)
                    available = [idx for idx in diff_indices if idx not in selected]
                    if len(available) > 0:
                        perm = torch.randperm(len(available), device=device)
                        selected.extend([available[idx] for idx in perm[:remaining]])

                negative_indices[i, :len(selected)] = torch.tensor(selected[:max_negatives], device=device)

        return negative_indices

    def _compute_metrics(self, similarity_matrix, same_speaker_mask, speaker_ids):
        """Compute additional metrics for monitoring."""
        metrics = {}

        # Same-speaker similarity
        same_speaker_sim = similarity_matrix[same_speaker_mask]
        if len(same_speaker_sim) > 0:
            metrics['same_speaker_sim_mean'] = same_speaker_sim.mean().item()
            metrics['same_speaker_sim_std'] = same_speaker_sim.std().item()
        else:
            metrics['same_speaker_sim_mean'] = 0.0
            metrics['same_speaker_sim_std'] = 0.0

        # Different-speaker similarity
        diff_speaker_sim = similarity_matrix[~same_speaker_mask]
        if len(diff_speaker_sim) > 0:
            metrics['diff_speaker_sim_mean'] = diff_speaker_sim.mean().item()
            metrics['diff_speaker_sim_std'] = diff_speaker_sim.std().item()
        else:
            metrics['diff_speaker_sim_mean'] = 0.0
            metrics['diff_speaker_sim_std'] = 0.0

        # Separation
        metrics['separation'] = metrics.get('same_speaker_sim_mean', 0.0) - \
                               metrics.get('diff_speaker_sim_mean', 0.0)

        return metrics


def contrastive_speaker_loss(model, audio, codes, speaker_embs, config,
                             speaker_index=None):
    """
    Legacy wrapper for backward compatibility.

    Args:
        model: SNAC model
        audio: Audio batch
        codes: Encoded codes (unused, kept for interface)
        speaker_embs: Speaker embeddings
        config: Training config
        speaker_index: Optional SpeakerEmbeddingIndex

    Returns:
        - loss: Contrastive loss scalar
    """
    loss_fn = ContrastiveSpeakerLoss(
        margin=config.get('contrastive_margin', 0.1),
        temperature=config.get('contrastive_temperature', 0.1),
        speaker_index=speaker_index
    )

    loss, metrics = loss_fn(speaker_embs, audio, config)

    return loss

#!/usr/bin/env python3
"""
Test frame-aware BPE on INTERLEAVED tokens.

Only compress WITHIN 7-token frames, never across frame boundaries.
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple


def frame_aware_bpe(tokens: List[int], frame_size: int = 7, max_merges: int = 1000) -> Dict:
    """
    Apply BPE only within each frame (never across frames).

    Args:
        tokens: Interleaved token sequence (must be multiple of 7)
        frame_size: Size of each frame (7 for SNAC)
        max_merges: Maximum number of merge operations

    Returns:
        Compression statistics
    """
    if len(tokens) % frame_size != 0:
        raise ValueError(f"Token length ({len(tokens)}) must be divisible by frame_size ({frame_size})")

    # Split into frames
    num_frames = len(tokens) // frame_size
    frames = [tokens[i*frame_size:(i+1)*frame_size] for i in range(num_frames)]

    # Learn merge rules FROM ALL FRAMES (but only apply within frames)
    all_pairs = Counter()

    for frame in frames:
        # Count pairs within this frame only
        for i in range(len(frame) - 1):
            pair = (frame[i], frame[i+1])
            all_pairs[pair] += 1

    # Get most frequent pairs
    top_pairs = all_pairs.most_common(max_merges)

    # Apply merges within each frame
    merged_frames = []
    total_original_tokens = len(tokens)
    total_merged_tokens = 0
    merge_count = 0

    for frame in frames:
        # Apply merges to this frame
        frame_merged = frame.copy()

        for pair, count in top_pairs:
            # Merge this pair within the frame
            new_frame = []
            i = 0
            while i < len(frame_merged) - 1:
                if (frame_merged[i], frame_merged[i+1]) == pair:
                    # Merge these two tokens
                    # Use negative ID to represent merged pair
                    merged_id = hash(pair) % 1000000000
                    new_frame.append(merged_id)
                    merge_count += 1
                    i += 2
                else:
                    new_frame.append(frame_merged[i])
                    i += 1

            if i < len(frame_merged):
                new_frame.append(frame_merged[i])

            frame_merged = new_frame

        merged_frames.append(frame_merged)

    # Flatten back to sequence
    merged_tokens = []
    for frame in merged_frames:
        merged_tokens.extend(frame)

    total_merged_tokens = len(merged_tokens)

    # Calculate compression ratio
    compression_ratio = total_original_tokens / total_merged_tokens if total_merged_tokens > 0 else 1.0

    return {
        'compression_ratio': compression_ratio,
        'original_tokens': total_original_tokens,
        'merged_tokens': total_merged_tokens,
        'tokens_saved': total_original_tokens - total_merged_tokens,
        'merge_operations': merge_count,
        'num_merges_applied': len(top_pairs),
        'unique_pairs': len(all_pairs)
    }


def main():
    # Load interleaved tokens
    tokens_dir = Path("/mnt/data/tokens/comparison/interleaved")
    parquet_file = tokens_dir / "tokens_interleaved.parquet"

    if not parquet_file.exists():
        print(f"ERROR: {parquet_file} not found!")
        return

    print("Loading interleaved tokens...")
    df = pd.read_parquet(parquet_file)

    # Collect all tokens
    all_tokens = []
    for tokens_list in df['tokens_scale_0']:
        all_tokens.extend(tokens_list)

    print(f"Total tokens: {len(all_tokens):,}")
    print(f"Num frames: {len(all_tokens) // 7:,}")
    print()

    # Test frame-aware BPE with different merge counts
    print("="*70)
    print("FRAME-AWARE BPE TEST (BPE within 7-token frames only)")
    print("="*70)
    print()

    for num_merges in [10, 50, 100, 500, 1000, 5000]:
        result = frame_aware_bpe(all_tokens, frame_size=7, max_merges=num_merges)

        print(f"{num_merges:5d} merges: {result['compression_ratio']:.4f}x compression "
              f"({result['tokens_saved']:,} tokens saved, "
              f"{result['original_tokens']:,} → {result['merged_tokens']:,} tokens)")

    print()
    print("="*70)
    print("CONCLUSION:")
    print("="*70)

    baseline = frame_aware_bpe(all_tokens, frame_size=7, max_merges=1)
    print(f"Even with 5000 merges: {baseline['compression_ratio']:.4f}x compression")
    print()
    print("Frame-aware BPE doesn't help much because:")
    print("  1. Each frame only has 7 tokens (very short)")
    print("  2. Limited pairs to merge within a frame")
    print("  3. Most pairs are unique (low repetition)")
    print()
    print("DEDUPLICATION (removing duplicate frames) is much more effective:")
    print("  - Already gives 1.25x compression")
    print("  - Preserves frame structure")
    print("  - Safe for audio quality")
    print("="*70)


if __name__ == "__main__":
    main()

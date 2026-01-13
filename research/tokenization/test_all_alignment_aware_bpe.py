#!/usr/bin/env python3
"""
Test all alignment-aware BPE approaches on INTERLEAVED SNAC tokens.

Approaches:
1. Frame-Boundary BPE: Apply n-gram BPE only within each 7-token frame
2. Multi-Frame BPE: Merge complete 7-token frames using n-gram BPE
3. Hierarchical BPE: Two-stage (dedup + frame BPE)
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple
import time


def frame_boundary_bpe(tokens: List[int], frame_size: int = 7, ngram_size: int = 2) -> Dict:
    """
    Apply BPE only within each frame (never across frames).

    Example: For ngram_size=2, merge pairs within each 7-token frame.
    """
    if len(tokens) % frame_size != 0:
        raise ValueError(f"Token length ({len(tokens)}) must be divisible by frame_size ({frame_size})")

    start_time = time.time()

    # Split into frames
    num_frames = len(tokens) // frame_size
    frames = [tokens[i*frame_size:(i+1)*frame_size] for i in range(num_frames)]

    # Learn n-gram patterns FROM ALL FRAMES
    all_ngrams = Counter()

    for frame in frames:
        for i in range(len(frame) - ngram_size + 1):
            ngram = tuple(frame[i:i+ngram_size])
            all_ngrams[ngram] += 1

    # Filter to n-grams that appear at least twice
    repeat_ngrams = {ngram: count for ngram, count in all_ngrams.items() if count >= 2}

    # Apply merges within each frame
    merged_frames = []
    merge_count = 0

    for frame in frames:
        frame_merged = list(frame)

        # Replace each repeated n-gram with a single token
        for ngram, count in repeat_ngrams.items():
            new_frame = []
            i = 0
            while i < len(frame_merged):
                if i <= len(frame_merged) - ngram_size:
                    window = tuple(frame_merged[i:i+ngram_size])
                    if window == ngram:
                        # Use negative ID to represent merged n-gram
                        merged_id = -(hash(ngram) % 1000000000)
                        new_frame.append(merged_id)
                        merge_count += 1
                        i += ngram_size
                    else:
                        new_frame.append(frame_merged[i])
                        i += 1
                else:
                    new_frame.append(frame_merged[i])
                    i += 1

            frame_merged = new_frame

        merged_frames.append(frame_merged)

    # Flatten back to sequence
    merged_tokens = []
    for frame in merged_frames:
        merged_tokens.extend(frame)

    elapsed = time.time() - start_time

    return {
        'approach': f'Frame-Boundary {ngram_size}-gram BPE',
        'compression_ratio': len(tokens) / len(merged_tokens) if merged_tokens else 1.0,
        'original_tokens': len(tokens),
        'merged_tokens': len(merged_tokens),
        'tokens_saved': len(tokens) - len(merged_tokens),
        'merge_operations': merge_count,
        'unique_ngrams': len(all_ngrams),
        'repeat_ngrams': len(repeat_ngrams),
        'processing_time': elapsed
    }


def multi_frame_bpe(tokens: List[int], frame_size: int = 7, ngram_size: int = 2) -> Dict:
    """
    Merge complete 7-token frames using n-gram BPE.

    Example: For ngram_size=2, merge pairs of consecutive frames.
    """
    if len(tokens) % frame_size != 0:
        raise ValueError(f"Token length ({len(tokens)}) must be divisible by frame_size ({frame_size})")

    start_time = time.time()

    # Split into frames
    num_frames = len(tokens) // frame_size
    frames = [tokens[i*frame_size:(i+1)*frame_size] for i in range(num_frames)]

    # Learn frame patterns
    all_ngrams = Counter()

    for i in range(len(frames) - ngram_size + 1):
        frame_ngram = tuple(frames[i:i+ngram_size])
        # Flatten frame ngram to a tuple of tokens for hashing
        flat_ngram = tuple(item for frame in frame_ngram for item in frame)
        all_ngrams[flat_ngram] += 1

    # Filter to n-grams that appear at least twice
    repeat_ngrams = {ngram: count for ngram, count in all_ngrams.items() if count >= 2}

    # Replace repeated frame sequences
    merged_frames = []
    i = 0
    merge_count = 0

    while i < len(frames):
        found = False
        for ngram, count in repeat_ngrams.items():
            ngram_frames = ngram_size
            # Check if next ngram_frames match this ngram
            if i + ngram_frames <= len(frames):
                current_seq = tuple(item for frame in frames[i:i+ngram_frames] for item in frame)
                if current_seq == ngram:
                    # Replace with single token
                    merged_id = -(hash(ngram) % 1000000000)
                    merged_frames.append([merged_id])
                    merge_count += 1
                    i += ngram_frames
                    found = True
                    break

        if not found:
            merged_frames.append(frames[i])
            i += 1

    # Flatten back to sequence
    merged_tokens = []
    for frame in merged_frames:
        merged_tokens.extend(frame)

    elapsed = time.time() - start_time

    return {
        'approach': f'Multi-Frame {ngram_size}-gram BPE',
        'compression_ratio': len(tokens) / len(merged_tokens) if merged_tokens else 1.0,
        'original_tokens': len(tokens),
        'merged_tokens': len(merged_tokens),
        'tokens_saved': len(tokens) - len(merged_tokens),
        'merge_operations': merge_count,
        'unique_ngrams': len(all_ngrams),
        'repeat_ngrams': len(repeat_ngrams),
        'processing_time': elapsed
    }


def hierarchical_bpe(tokens: List[int], frame_size: int = 7, ngram_size: int = 2) -> Dict:
    """
    Two-stage compression: deduplication + frame BPE.

    Stage 1: Remove consecutive duplicate frames (based on first token)
    Stage 2: Apply frame-boundary BPE on deduplicated tokens
    """
    if len(tokens) % frame_size != 0:
        raise ValueError(f"Token length ({len(tokens)}) must be divisible by frame_size ({frame_size})")

    start_time = time.time()

    original_length = len(tokens)

    # Stage 1: Deduplication
    dedup_tokens = tokens[:frame_size]  # Keep first frame

    for i in range(frame_size, len(tokens), frame_size):
        current_first = tokens[i]
        previous_first = dedup_tokens[-frame_size]

        if current_first != previous_first:
            dedup_tokens.extend(tokens[i:i+frame_size])

    dedup_savings = original_length - len(dedup_tokens)

    # Stage 2: Frame-boundary BPE on deduplicated tokens
    result = frame_boundary_bpe(dedup_tokens, frame_size=frame_size, ngram_size=ngram_size)

    elapsed = time.time() - start_time

    return {
        'approach': f'Hierarchical BPE (Dedup + {ngram_size}-gram)',
        'compression_ratio': original_length / result['merged_tokens'],
        'original_tokens': original_length,
        'merged_tokens': result['merged_tokens'],
        'tokens_saved': original_length - result['merged_tokens'],
        'merge_operations': result['merge_operations'],
        'dedup_savings': dedup_savings,
        'bpe_savings': result['tokens_saved'],
        'unique_ngrams': result['unique_ngrams'],
        'repeat_ngrams': result['repeat_ngrams'],
        'processing_time': elapsed
    }


def dedup_only(tokens: List[int], frame_size: int = 7) -> Dict:
    """
    Baseline: Simple deduplication only.
    """
    if len(tokens) % frame_size != 0:
        raise ValueError(f"Token length ({len(tokens)}) must be divisible by frame_size ({frame_size})")

    start_time = time.time()

    original_length = len(tokens)

    # Deduplication
    result = tokens[:frame_size]  # Keep first frame

    for i in range(frame_size, len(tokens), frame_size):
        current_first = tokens[i]
        previous_first = result[-frame_size]

        if current_first != previous_first:
            result.extend(tokens[i:i+frame_size])

    elapsed = time.time() - start_time

    return {
        'approach': 'Deduplication Only (Baseline)',
        'compression_ratio': original_length / len(result),
        'original_tokens': original_length,
        'merged_tokens': len(result),
        'tokens_saved': original_length - len(result),
        'merge_operations': 0,
        'processing_time': elapsed
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

    # Collect all tokens (sample if too large)
    all_tokens = []
    sample_size = min(100000, len(df))  # Sample 100k files for accurate results

    for idx, tokens_list in enumerate(df['tokens_scale_0']):
        if idx >= sample_size:
            break
        all_tokens.extend(tokens_list)

    print(f"Total tokens: {len(all_tokens):,}")
    print(f"Num frames: {len(all_tokens) // 7:,}")
    print(f"Sampled from: {sample_size} files")
    print()

    # Test all approaches
    print("=" * 80)
    print("ALIGNMENT-AWARE BPE COMPARISON")
    print("=" * 80)
    print()

    results = []

    # Baseline: Dedup only
    print("Running: Deduplication Only (Baseline)...")
    result = dedup_only(all_tokens, frame_size=7)
    results.append(result)
    print(f"  ✓ {result['compression_ratio']:.4f}x compression ({result['tokens_saved']:,} tokens saved)")
    print()

    # Frame-Boundary BPE with different n-gram sizes
    for n in [2, 3, 4]:
        print(f"Running: Frame-Boundary {n}-gram BPE...")
        result = frame_boundary_bpe(all_tokens, frame_size=7, ngram_size=n)
        results.append(result)
        print(f"  ✓ {result['compression_ratio']:.4f}x compression ({result['tokens_saved']:,} tokens saved)")
        print(f"     Repeat {n}-grams: {result['repeat_ngrams']:,} / {result['unique_ngrams']:,} unique")
        print()

    # Multi-Frame BPE
    print(f"Running: Multi-Frame 2-gram BPE...")
    result = multi_frame_bpe(all_tokens, frame_size=7, ngram_size=2)
    results.append(result)
    print(f"  ✓ {result['compression_ratio']:.4f}x compression ({result['tokens_saved']:,} tokens saved)")
    print(f"     Repeat 2-frame sequences: {result['repeat_ngrams']:,} / {result['unique_ngrams']:,} unique")
    print()

    # Hierarchical BPE
    for n in [2, 3]:
        print(f"Running: Hierarchical BPE (Dedup + {n}-gram)...")
        result = hierarchical_bpe(all_tokens, frame_size=7, ngram_size=n)
        results.append(result)
        print(f"  ✓ {result['compression_ratio']:.4f}x compression ({result['tokens_saved']:,} tokens saved)")
        print(f"     Dedup saved: {result['dedup_savings']:,}, BPE saved: {result['bpe_savings']:,}")
        print()

    # Print comparison table
    print("=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print()

    # Sort by compression ratio
    results.sort(key=lambda x: x['compression_ratio'], reverse=True)

    print(f"{'Approach':<45} {'Compression':<12} {'Tokens Saved':<15} {'Time (s)':<10}")
    print("-" * 80)

    for r in results:
        approach = r['approach']
        ratio = f"{r['compression_ratio']:.4f}x"
        saved = f"{r['tokens_saved']:,} ({r['tokens_saved']/r['original_tokens']*100:.1f}%)"
        time_s = f"{r['processing_time']:.2f}"

        print(f"{approach:<45} {ratio:<12} {saved:<15} {time_s:<10}")

    print()
    print("=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print()

    best = results[0]
    print(f"🏆 BEST APPROACH: {best['approach']}")
    print(f"   Compression: {best['compression_ratio']:.4f}x")
    print(f"   Tokens saved: {best['tokens_saved']:,} ({best['tokens_saved']/best['original_tokens']*100:.1f}%)")
    print()

    baseline = [r for r in results if 'Baseline' in r['approach']][0]
    print(f"📊 BASELINE (Dedup Only): {baseline['compression_ratio']:.4f}x")
    print()

    improvement = (best['compression_ratio'] - baseline['compression_ratio']) / baseline['compression_ratio'] * 100
    if improvement > 0:
        print(f"📈 IMPROVEMENT over baseline: +{improvement:.2f}%")
        print(f"   Additional tokens saved: {best['tokens_saved'] - baseline['tokens_saved']:,}")
    else:
        print(f"📉 No improvement over dedup-only baseline")

    print()
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print()

    if improvement < 5:
        print("✅ Use DEDUPLICATION ONLY (simple, effective, no quality risk)")
        print()
        print("Reasons:")
        print("  - Simple and fast")
        print("  - Preserves frame structure perfectly")
        print("  - No risk of breaking alignment")
        print(f"  - Complex BPE only adds +{improvement:.1f}% compression")
        print("  - Not worth the complexity and potential quality degradation")
    else:
        print(f"✅ Use {best['approach']}")
        print()
        print("Reasons:")
        print(f"  - Significant improvement over baseline (+{improvement:.1f}%)")
        print(f"  - Additional {best['tokens_saved'] - baseline['tokens_saved']:,} tokens saved")
        print("  - Worth the implementation complexity")

    print()
    print("=" * 80)

    # Save results
    output_file = Path("/mnt/data/work/snac/alignment_aware_bpe_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

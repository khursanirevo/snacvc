#!/usr/bin/env python3
"""
Compare BPE compression effectiveness on RAW vs INTERLEAVED token formats.

Tests which format benefits more from BPE merging with different n-gram sizes.
"""

import json
import argparse
import logging
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd

from snac.dataset import OptimizedAudioDataset


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def compress_tokens_worker(args: Tuple[List[int], int, int]) -> Tuple[int, int, Dict]:
    """Worker function for parallel BPE compression.

    Args:
        args: (tokens, n, worker_id) tuple

    Returns:
        (n, worker_id, compression_stats)
    """
    tokens, n, worker_id = args
    compressor = NgramBPECompressor()
    compressed, stats = compressor.compress_ngrams(tokens, n)
    return (n, worker_id, stats)


class BPECompressor:
    """Simple BPE implementation for comparison."""

    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.merge_rules = []

    def learn_vocab(self, tokens: List[int], num_merges: int) -> List[Tuple[int, int]]:
        """Learn BPE merge rules from token sequence."""
        # Count all adjacent pairs
        vocab = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            vocab[pair] = vocab.get(pair, 0) + 1

        # Learn merge rules
        merge_rules = []
        for _ in range(num_merges):
            if not vocab:
                break

            # Get most frequent pair
            best_pair = max(vocab.items(), key=lambda x: x[1])[0]
            merge_rules.append(best_pair)

            # Rebuild vocab with merged pair
            vocab = {}
            i = 0
            while i < len(tokens) - 1:
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    i += 2
                else:
                    pair = (tokens[i], tokens[i + 1]) if i < len(tokens) - 1 else None
                    if pair:
                        vocab[pair] = vocab.get(pair, 0) + 1
                    i += 1

        return merge_rules

    def apply_bpe(self, tokens: List[int], merge_rules: List[Tuple[int, int]]) -> List[int]:
        """Apply BPE merge rules to compress tokens."""
        tokens = tokens.copy()

        for pair in merge_rules:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    # Merge this pair into a single token (use negative IDs for merged)
                    merged_id = hash(pair) % 1000000000  # Unique ID for merged pair
                    new_tokens.append(merged_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens


class NgramBPECompressor:
    """N-gram based BPE compressor for testing."""

    def compress_ngrams(self, tokens: List[int], n: int) -> Tuple[List[int], Dict]:
        """
        Compress tokens by replacing repeated n-grams with single tokens.

        Args:
            tokens: Input token sequence
            n: N-gram size to compress

        Returns:
            Compressed tokens and compression statistics
        """
        if len(tokens) < n:
            return tokens, {'compression_ratio': 1.0, 'replacements': 0}

        # Count all n-grams
        ngram_counts = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngram_counts[ngram] += 1

        if not ngram_counts:
            return tokens, {'compression_ratio': 1.0, 'replacements': 0}

        # Find most common n-grams
        most_common = ngram_counts.most_common()

        # Replace top n-grams with single tokens
        compressed = tokens.copy()
        replacements = 0
        tokens_saved = 0

        for ngram, count in most_common:
            if count < 2:  # Skip if only appears once
                break

            # Create replacement token ID
            replacement_id = -(hash(ngram) % 1000000000)

            # Replace all occurrences
            new_compressed = []
            i = 0
            while i < len(compressed):
                if i <= len(compressed) - n:
                    window = tuple(compressed[i:i+n])
                    if window == ngram:
                        new_compressed.append(replacement_id)
                        replacements += 1
                        tokens_saved += (n - 1)
                        i += n
                    else:
                        new_compressed.append(compressed[i])
                        i += 1
                else:
                    new_compressed.append(compressed[i])
                    i += 1

            compressed = new_compressed

        compression_ratio = len(tokens) / len(compressed) if compressed else 1.0

        return compressed, {
            'compression_ratio': compression_ratio,
            'replacements': replacements,
            'original_length': len(tokens),
            'compressed_length': len(compressed),
            'tokens_saved': tokens_saved,
            'unique_ngrams': len(ngram_counts),
            'most_frequent_count': most_common[0][1] if most_common else 0
        }


class BPEComparator:
    """Compare BPE effectiveness on RAW vs INTERLEAVED formats."""

    def __init__(self, tokens_dir: str):
        self.tokens_dir = Path(tokens_dir)
        self.logger = logging.getLogger(__name__)

    def load_tokens(self) -> Dict[str, Any]:
        """Load both RAW and INTERLEAVED tokens."""
        data = {'raw': None, 'interleaved': None}

        # Load RAW format
        raw_file = self.tokens_dir / "raw" / "tokens_raw.parquet"
        if raw_file.exists():
            df = pd.read_parquet(raw_file)
            data['raw'] = df
            self.logger.info(f"Loaded RAW format: {len(df)} files")

        # Load INTERLEAVED format
        int_file = self.tokens_dir / "interleaved" / "tokens_interleaved.parquet"
        if int_file.exists():
            df = pd.read_parquet(int_file)
            data['interleaved'] = df
            self.logger.info(f"Loaded INTERLEAVED format: {len(df)} files")

        return data

    def extract_token_sequences(self, df: pd.DataFrame, format_type: str) -> List[List[int]]:
        """Extract token sequences from dataframe."""
        sequences = []

        if format_type == 'raw':
            # RAW format: separate scales (can be 3 or 4)
            scale_cols = [col for col in df.columns if col.startswith('tokens_scale_')]
            num_scales = len(scale_cols)

            for _, row in df.iterrows():
                for scale_idx in range(num_scales):
                    col = f'tokens_scale_{scale_idx}'
                    if col in df.columns:
                        val = row[col]
                        # Check if it's a list/tuple with content
                        if hasattr(val, '__iter__') and not isinstance(val, str):
                            val_list = list(val)
                            if len(val_list) > 0:
                                sequences.append(val_list)

        elif format_type == 'interleaved':
            # INTERLEAVED format: single sequence
            for _, row in df.iterrows():
                if 'tokens_scale_0' in df.columns:
                    val = row['tokens_scale_0']
                    if hasattr(val, '__iter__') and not isinstance(val, str):
                        val_list = list(val)
                        if len(val_list) > 0:
                            sequences.append(val_list)

        return sequences

    def test_ngram_bpe(self, sequences: List[List[int]], max_n: int = 7) -> Dict[int, Dict]:
        """Test BPE compression with different n-gram sizes."""
        compressor = NgramBPECompressor()
        results = {}

        # Sample sequences to speed up analysis
        sample_size = min(1000, len(sequences))
        sampled = sequences[:sample_size]

        # Flatten all tokens for analysis
        all_tokens = []
        for seq in sampled:
            all_tokens.extend(seq)

        self.logger.info(f"Testing BPE on {len(all_tokens):,} tokens from {len(sampled)} sequences")

        for n in range(1, max_n + 1):
            compressed, stats = compressor.compress_ngrams(all_tokens, n)
            results[n] = stats
            self.logger.info(
                f"  {n}-gram BPE: {stats['compression_ratio']:.2f}x compression, "
                f"{stats['replacements']} replacements"
            )

        return results

    def compare(self, max_n: int = 7):
        """Compare BPE effectiveness on both formats."""
        self.logger.info("="*70)
        self.logger.info("BPE COMPARISON: RAW vs INTERLEAVED")
        self.logger.info("="*70)

        # Load data
        data = self.load_tokens()

        if data['raw'] is None and data['interleaved'] is None:
            self.logger.error("No token data found!")
            return

        results = {}
        max_workers = mp.cpu_count()

        # Test RAW format
        if data['raw'] is not None:
            self.logger.info(f"\n{'='*70}")
            self.logger.info("RAW FORMAT (separate scales)")
            self.logger.info('='*70)

            raw_sequences = self.extract_token_sequences(data['raw'], 'raw')

            # Determine number of scales from dataframe
            scale_cols = [col for col in data['raw'].columns if col.startswith('tokens_scale_')]
            num_scales = len(scale_cols)

            self.logger.info(f"Total sequences: {len(raw_sequences)}")
            self.logger.info(f"Number of scales: {num_scales}")
            self.logger.info(f"Using {max_workers} parallel workers")

            # Test each scale separately
            for scale_idx in range(num_scales):
                scale_seqs = [s for i, s in enumerate(raw_sequences) if i % num_scales == scale_idx]
                all_tokens = []
                for seq in scale_seqs:
                    all_tokens.extend(seq)

                self.logger.info(f"\nScale {scale_idx} ({len(all_tokens):,} tokens):")

                # Prepare tasks for parallel processing
                tasks = [(all_tokens, n, scale_idx) for n in range(1, max_n + 1)]

                # Process in parallel
                scale_results = {}
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(compress_tokens_worker, task): task for task in tasks}

                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"  Scale {scale_idx} BPE"):
                        n, worker_id, stats = future.result()
                        scale_results[n] = stats
                        self.logger.info(
                            f"    {n}-gram: {stats['compression_ratio']:.2f}x compression, "
                            f"{stats['replacements']:,} replacements"
                        )

                results[f'raw_scale_{scale_idx}'] = scale_results

        # Test INTERLEAVED format
        if data['interleaved'] is not None:
            self.logger.info(f"\n{'='*70}")
            self.logger.info("INTERLEAVED FORMAT (7-token frames + dedup)")
            self.logger.info('='*70)

            int_sequences = self.extract_token_sequences(data['interleaved'], 'interleaved')
            all_tokens = []
            for seq in int_sequences:
                all_tokens.extend(seq)

            self.logger.info(f"Total tokens: {len(all_tokens):,}")

            # Prepare tasks for parallel processing
            tasks = [(all_tokens, n, 999) for n in range(1, max_n + 1)]

            # Process in parallel
            results['interleaved'] = {}
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(compress_tokens_worker, task): task for task in tasks}

                for future in tqdm(as_completed(futures), total=len(futures), desc="  Interleaved BPE"):
                    n, worker_id, stats = future.result()
                    results['interleaved'][n] = stats
                    self.logger.info(
                        f"    {n}-gram: {stats['compression_ratio']:.2f}x compression, "
                        f"{stats['replacements']:,} replacements"
                    )

        # Summary comparison
        self.logger.info(f"\n{'='*70}")
        self.logger.info("SUMMARY")
        self.logger.info('='*70)

        if 'interleaved' in results:
            self.logger.info("\nINTERLEAVED format compression ratios:")
            for n in range(1, max_n + 1):
                stats = results['interleaved'][n]
                self.logger.info(f"  {n}-gram: {stats['compression_ratio']:.2f}x")

        if 'raw_scale_0' in results:
            self.logger.info("\nRAW format compression ratios (Scale 0 - coarsest):")
            for n in range(1, max_n + 1):
                stats = results['raw_scale_0'][n]
                self.logger.info(f"  {n}-gram: {stats['compression_ratio']:.2f}x")

        # Save results
        report_file = self.tokens_dir / "bpe_comparison_report.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"\nReport saved to: {report_file}")
        self.logger.info("="*70)

        return results


def main():
    parser = argparse.ArgumentParser(description="Compare BPE effectiveness on token formats")
    parser.add_argument("--tokens_dir", type=str, required=True, help="Directory with comparison data")
    parser.add_argument("--max_n", type=int, default=7, help="Maximum n-gram size to test")

    args = parser.parse_args()

    global logger
    logger = setup_logging()

    logger.info(f"Tokens directory: {args.tokens_dir}")
    logger.info(f"Testing n-grams: 1-{args.max_n}")

    comparator = BPEComparator(args.tokens_dir)
    comparator.compare(max_n=args.max_n)


if __name__ == "__main__":
    main()

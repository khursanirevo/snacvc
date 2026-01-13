#!/usr/bin/env python3
"""
Compare RAW vs ORPHEUS-INTERLEAVED token formats.

Processes a sample of audio files and saves both formats for comparison.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from snac import SNAC

# Import functions from audio_to_tokens.py
from audio_to_tokens import (
    interleave_snac_codes,
    remove_duplicate_frames
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class TokenFormatComparator:
    """Compare RAW vs INTERLEAVED token formats."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        token_offset: int = 128266
    ):
        if device.isdigit():
            device = f"cuda:{device}"
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.token_offset = token_offset
        self.sampling_rate = 24000

        # Load model
        logger.info(f"Loading SNAC model from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        logger.info("Model loaded successfully")

    def tokenize_audio(self, waveform: torch.Tensor) -> List[torch.Tensor]:
        """Convert audio to SNAC codes."""
        with torch.no_grad():
            codes = self.model.encode(waveform)
        return codes

    def process_file(self, audio_path: str) -> Dict[str, Any]:
        """Process a single audio file and extract both formats."""
        try:
            import torchaudio
            waveform, sr = torchaudio.load(audio_path)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if needed
            if sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
                waveform = resampler(waveform)

            # Add batch dimension and move to device
            waveform = waveform.unsqueeze(0).to(self.device)

            # Tokenize
            codes = self.tokenize_audio(waveform)

            # Format 1: RAW (separate scales)
            raw_tokens = [c.squeeze(0).cpu().numpy() for c in codes]
            raw_total_tokens = sum(t.size for t in raw_tokens)

            # Format 2: INTERLEAVED (Orpheus-style)
            try:
                interleaved = interleave_snac_codes(codes, offset=self.token_offset)
                deduplicated = remove_duplicate_frames(interleaved)

                interleaved_tokens = np.array(interleaved)
                deduplicated_tokens = np.array(deduplicated)

                compression_ratio = len(interleaved) / len(deduplicated)
            except (ValueError, IndexError) as e:
                # Audio too short for interleaving
                interleaved_tokens = None
                deduplicated_tokens = None
                compression_ratio = None

            return {
                'file_path': audio_path,
                'duration': waveform.shape[-1] / self.sampling_rate,
                'raw_tokens': raw_tokens,
                'raw_total': raw_total_tokens,
                'interleaved': interleaved_tokens,
                'deduplicated': deduplicated_tokens,
                'interleaved_len': len(interleaved) if interleaved_tokens is not None else 0,
                'deduplicated_len': len(deduplicated) if deduplicated_tokens is not None else 0,
                'compression_ratio': compression_ratio,
                'success': True
            }

        except Exception as e:
            logger.warning(f"Error processing {audio_path}: {e}")
            return {
                'file_path': audio_path,
                'success': False,
                'error': str(e)
            }

    def compare(
        self,
        audio_dir: str,
        output_dir: str,
        max_files: int = 1000,
        batch_size: int = 32
    ):
        """Compare both formats on a sample of audio files."""
        import subprocess

        output_path = Path(output_dir)
        raw_path = output_path / "raw"
        interleaved_path = output_path / "interleaved"
        raw_path.mkdir(parents=True, exist_ok=True)
        interleaved_path.mkdir(parents=True, exist_ok=True)

        # Find audio files
        logger.info(f"Searching for audio files in {audio_dir}...")
        audio_path = Path(audio_dir)

        result = subprocess.run(
            ['find', str(audio_path), '-type', 'f', '-iname', '*.wav'],
            capture_output=True,
            text=True,
            check=False
        )
        audio_files = [f.strip() for f in result.stdout.split('\n') if f.strip()]

        if max_files:
            audio_files = audio_files[:max_files]

        logger.info(f"Found {len(audio_files)} audio files")

        # Process files
        results = []
        raw_data = []
        interleaved_data = []

        for audio_file in tqdm(audio_files, desc="Processing"):
            result = self.process_file(audio_file)

            if result.get('success', False):
                results.append(result)

                # Save RAW format
                if result['raw_tokens']:
                    row_raw = {
                        'file_path': result['file_path'],
                        'duration': result['duration'],
                    }
                    for scale_idx, tokens in enumerate(result['raw_tokens']):
                        row_raw[f'tokens_scale_{scale_idx}'] = tokens.tolist()
                        row_raw[f'n_tokens_scale_{scale_idx}'] = tokens.size
                    raw_data.append(row_raw)

                # Save INTERLEAVED format
                if result['deduplicated'] is not None:
                    row_int = {
                        'file_path': result['file_path'],
                        'duration': result['duration'],
                        'tokens_scale_0': result['deduplicated'].tolist(),
                        'n_tokens_scale_0': len(result['deduplicated'])
                    }
                    interleaved_data.append(row_int)

        # Save parquet files
        if raw_data:
            df_raw = pd.DataFrame(raw_data)
            raw_file = raw_path / "tokens_raw.parquet"
            df_raw.to_parquet(raw_file, index=False)
            logger.info(f"Saved RAW format: {raw_file}")

        if interleaved_data:
            df_int = pd.DataFrame(interleaved_data)
            int_file = interleaved_path / "tokens_interleaved.parquet"
            df_int.to_parquet(int_file, index=False)
            logger.info(f"Saved INTERLEAVED format: {int_file}")

        # Analyze and compare
        self.analyze_comparison(results, output_path)

    def analyze_comparison(self, results: List[Dict], output_path: Path):
        """Generate comparison report."""
        logger.info("\n" + "="*70)
        logger.info("FORMAT COMPARISON REPORT")
        logger.info("="*70)

        successful = [r for r in results if r.get('success', False)]
        failed = len(results) - len(successful)

        logger.info(f"\nFiles processed: {len(successful)}")
        logger.info(f"Files failed (too short for interleaved): {failed}")

        if not successful:
            return

        # Calculate statistics
        raw_tokens_per_sec = []
        interleaved_tokens_per_sec = []
        compression_ratios = []

        for r in successful:
            duration = r['duration']
            if duration > 0:
                raw_tokens_per_sec.append(r['raw_total'] / duration)

                if r['deduplicated_len'] > 0:
                    interleaved_tokens_per_sec.append(r['deduplicated_len'] / duration)

                if r['compression_ratio'] is not None:
                    compression_ratios.append(r['compression_ratio'])

        # RAW format stats
        logger.info(f"\n{'='*70}")
        logger.info("RAW FORMAT (Separate Scales)")
        logger.info('='*70)
        logger.info(f"Total tokens across all files: {sum(r['raw_total'] for r in successful):,}")
        logger.info(f"Tokens per second: {np.mean(raw_tokens_per_sec):.2f} ± {np.std(raw_tokens_per_sec):.2f}")
        logger.info(f"Min tokens/sec: {np.min(raw_tokens_per_sec):.2f}")
        logger.info(f"Max tokens/sec: {np.max(raw_tokens_per_sec):.2f}")

        # INTERLEAVED format stats
        if interleaved_tokens_per_sec:
            logger.info(f"\n{'='*70}")
            logger.info("INTERLEAVED FORMAT (Orpheus-style + Dedup)")
            logger.info('='*70)
            logger.info(f"Total tokens across all files: {sum(r['deduplicated_len'] for r in successful if r['deduplicated_len'] > 0):,}")
            logger.info(f"Tokens per second: {np.mean(interleaved_tokens_per_sec):.2f} ± {np.std(interleaved_tokens_per_sec):.2f}")
            logger.info(f"Min tokens/sec: {np.min(interleaved_tokens_per_sec):.2f}")
            logger.info(f"Max tokens/sec: {np.max(interleaved_tokens_per_sec):.2f}")

            # Compression comparison
            avg_raw = np.mean(raw_tokens_per_sec)
            avg_int = np.mean(interleaved_tokens_per_sec)
            reduction = (1 - avg_int / avg_raw) * 100

            logger.info(f"\n{'='*70}")
            logger.info("COMPARISON")
            logger.info('='*70)
            logger.info(f"RAW: {avg_raw:.2f} tokens/sec")
            logger.info(f"INTERLEAVED+DEDUP: {avg_int:.2f} tokens/sec")
            logger.info(f"Reduction: {reduction:.1f}%")
            logger.info(f"Compression ratio (dedup): {np.mean(compression_ratios):.2f}x")

        # Save report
        report = {
            'files_processed': len(successful),
            'files_failed': failed,
            'raw_format': {
                'avg_tokens_per_sec': float(np.mean(raw_tokens_per_sec)) if raw_tokens_per_sec else 0,
                'std_tokens_per_sec': float(np.std(raw_tokens_per_sec)) if raw_tokens_per_sec else 0,
            },
            'interleaved_format': {
                'avg_tokens_per_sec': float(np.mean(interleaved_tokens_per_sec)) if interleaved_tokens_per_sec else 0,
                'std_tokens_per_sec': float(np.std(interleaved_tokens_per_sec)) if interleaved_tokens_per_sec else 0,
                'avg_compression_ratio': float(np.mean(compression_ratios)) if compression_ratios else 0,
            } if interleaved_tokens_per_sec else None
        }

        report_file = output_path / "comparison_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"\nReport saved to: {report_file}")
        logger.info("="*70)


def main():
    parser = argparse.ArgumentParser(description="Compare RAW vs INTERLEAVED token formats")
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory with audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase10_revolab_all/best_model.pt")
    parser.add_argument("--max_files", type=int, default=1000, help="Max files to process")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--token_offset", type=int, default=128266)

    args = parser.parse_args()

    global logger
    logger = setup_logging()

    logger.info("Token Format Comparison Tool")
    logger.info(f"Audio directory: {args.audio_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Max files: {args.max_files}")

    comparator = TokenFormatComparator(
        checkpoint_path=args.checkpoint,
        device=args.device,
        token_offset=args.token_offset
    )

    comparator.compare(
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        max_files=args.max_files
    )


if __name__ == "__main__":
    main()

"""
Conditional 48kHz audio preparation using SIDON upsampler (working version).
Based on the working sidon_recon.py approach.

- If already 48kHz: copy as-is
- If not 48kHz: use SIDON upsampling with cuda models

Usage:
    # Test mode (10 non-48kHz files)
    python precompute_48khz_sidon.py --test_mode

    # Full processing
    python precompute_48khz_sidon.py \
        --input_dir /mnt/data/combine/train/audio \
        --output_dir /mnt/data/combine/train/audio_48khz
"""

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torchaudio
import transformers
from huggingface_hub import hf_hub_download
import soundfile as sf


def setup_logging(log_file):
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class SidonUpsampler:
    """SIDON upsampler using the working cuda.pt approach from sidon_recon.py"""

    def __init__(self, device='cuda'):
        self.device = device
        logger = logging.getLogger(__name__)

        logger.info("Loading SIDON upsampler (cuda versions)...")

        # Download CUDA models (not CPU versions!)
        fe_path = hf_hub_download("sarulab-speech/sidon-v0.1", filename="feature_extractor_cuda.pt")
        decoder_path = hf_hub_download("sarulab-speech/sidon-v0.1", filename="decoder_cuda.pt")

        # Load models directly to CUDA
        self.fe = torch.jit.load(fe_path, map_location='cuda').to('cuda')
        self.decoder = torch.jit.load(decoder_path, map_location='cuda').to('cuda')

        # Preprocessor
        self.preprocessor = transformers.SeamlessM4TFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0"
        )

        logger.info(f"✓ SIDON loaded on {device}")

    @torch.inference_mode()
    def upsample(self, audio_file):
        """
        Upsample audio to 48kHz using SIDON.

        Args:
            audio_file: Path to audio file

        Returns:
            (48000, audio_48k) tuple where audio_48k is numpy array
        """
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_file)

        # Normalize
        waveform = 0.9 * (waveform / np.abs(waveform).max())

        # Calculate target length
        target_n_samples = int(48_000 / sample_rate * waveform.shape[-1])

        # Ensure tensor format
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform, dtype=torch.float32)

        # Convert to mono if stereo
        if waveform.ndim > 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)

        # Add batch dimension
        waveform = waveform.view(1, -1)

        # Highpass filter
        wav = torchaudio.functional.highpass_biquad(waveform, sample_rate=sample_rate, cutoff_freq=50)

        # Resample to 16kHz for SIDON
        wav_16k = torchaudio.functional.resample(wav, orig_freq=sample_rate, new_freq=16_000)

        # Pad for SIDON processing
        wav_16k = torch.nn.functional.pad(wav_16k, (0, 24000))

        # Process chunks
        restoreds = []
        feature_cache = None

        for chunk in wav_16k.view(-1).split(16000 * 60):
            inputs = self.preprocessor(
                torch.nn.functional.pad(chunk, (40, 40)), sampling_rate=16_000, return_tensors="pt"
            ).to('cuda')

            feature = self.fe(inputs["input_features"].to("cuda"))["last_hidden_state"]

            if feature_cache is not None:
                feature = torch.cat([feature_cache, feature], dim=1)
                restored_wav = self.decoder(feature.transpose(1, 2))
                restored_wav = restored_wav[:, :, 4800:]
            else:
                restored_wav = self.decoder(feature.transpose(1, 2))
                restored_wav = restored_wav[:, :, 50 * 3:]

            feature_cache = feature[:, -5:, :]
            restoreds.append(restored_wav.cpu())

        restored_wav = torch.cat(restoreds, dim=-1)

        # Trim to target length
        restored_wav = restored_wav[:target_n_samples]

        # Convert to int16 PCM format (like sidon_recon.py)
        audio_48k = (restored_wav.view(-1, 1).numpy() * 32767).astype(np.int16).T

        return 48_000, audio_48k


class AudioDataset:
    """Audio dataset with conditional 48kHz preparation."""

    def __init__(self, audio_dir, test_mode=False, test_samples=10, start_idx=None, end_idx=None):
        self.audio_dir = Path(audio_dir)
        self.test_mode = test_mode

        # Get all audio files
        all_files = sorted(list(self.audio_dir.glob("*.wav")) +
                           list(self.audio_dir.glob("*.flac")) +
                           list(self.audio_dir.glob("*.mp3")))

        if len(all_files) == 0:
            raise ValueError(f"No audio files found in {audio_dir}")

        # Apply file range for parallel processing
        self.start_offset = start_idx if start_idx is not None else 0
        self.end_offset = end_idx if end_idx is not None else len(all_files)
        all_files = all_files[self.start_offset:self.end_offset]

        if test_mode:
            # Find files that are NOT 48kHz for testing
            non_48khz_files = []
            logger = logging.getLogger(__name__)
            for f in tqdm(all_files, desc="Finding non-48kHz files"):
                try:
                    _, sr = torchaudio.load(str(f))
                    if sr != 48000:
                        non_48khz_files.append(f)
                        if len(non_48khz_files) >= test_samples:
                            break
                except:
                    pass

            self.files = non_48khz_files
            logger.info(f"TEST MODE: Processing {len(self.files)} non-48kHz files (files {self.start_offset}-{self.end_offset})")
        else:
            self.files = all_files
            logger = logging.getLogger(__name__)
            logger.info(f"Processing files {self.start_offset} to {self.end_offset} ({len(self.files)} files)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.files[idx]


def main():
    parser = argparse.ArgumentParser(description="Conditional 48kHz audio preparation with SIDON")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory with audio")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for 48kHz audio")
    parser.add_argument("--test_mode", action='store_true',
                        help="Test mode: process only 10 non-48kHz files")
    parser.add_argument("--test_samples", type=int, default=10,
                        help="Number of non-48kHz files for test mode")
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Start file index (for parallel processing)")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="End file index (for parallel processing)")
    args = parser.parse_args()

    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "precompute_48khz_sidon.log"
    logger = setup_logging(log_file)

    logger.info("="*70)
    logger.info("Conditional 48kHz Audio Preparation with SIDON (working version)")
    logger.info("="*70)
    logger.info(f"Input directory:  {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Test mode: {args.test_mode}")
    logger.info(f"Device: cuda (SIDON requires CUDA)")
    logger.info("="*70)

    # Initialize SIDON upsampler
    sidon_upsampler = SidonUpsampler(device='cuda')

    # Create dataset
    logger.info(f"\nLoading dataset from {args.input_dir}...")
    dataset = AudioDataset(args.input_dir, test_mode=args.test_mode, test_samples=args.test_samples,
                          start_idx=args.start_idx, end_idx=args.end_idx)
    logger.info(f"✓ Dataset loaded: {len(dataset)} files")

    # Process files
    logger.info("\nProcessing audio files...")

    total_saved = 0      # SIDON upsampling
    total_copied = 0     # Direct copy (already 48kHz)
    total_skipped = 0    # Already existed
    total_errors = 0

    for audio_path in tqdm(dataset, desc="Processing"):
        output_path = output_dir / audio_path.name

        # Skip if file already exists
        if output_path.exists():
            total_skipped += 1
            continue

        try:
            # Check sample rate
            waveform, sr = torchaudio.load(str(audio_path))

            if sr == 48000:
                # Already 48kHz: copy directly (save as WAV)
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)

                # Change output path to .wav
                output_path_wav = output_path.with_suffix('.wav')
                torchaudio.save(str(output_path_wav), waveform, 48000, bits_per_sample=16)
                total_copied += 1
            else:
                # Use SIDON upsampling (save as WAV)
                sr_output, audio_48k = sidon_upsampler.upsample(str(audio_path))
                # Change output path to .wav
                output_path_wav = output_path.with_suffix('.wav')
                sf.write(str(output_path_wav), audio_48k.T, samplerate=sr_output, subtype='PCM_16', format='WAV')
                total_saved += 1

        except Exception as e:
            logger.error(f"Error processing {audio_path.name}: {e}")
            total_errors += 1
            continue

        if (total_saved + total_copied) % 100 == 0:
            logger.info(f"Progress - SIDON upsampled: {total_saved}, Copied (48kHz): {total_copied}, "
                       f"Skipped: {total_skipped}, Errors: {total_errors}")

    # Save metadata
    metadata = {
        'input_dir': args.input_dir,
        'output_dir': args.output_dir,
        'device': 'cuda',
        'test_mode': args.test_mode,
        'start_idx': args.start_idx,
        'end_idx': args.end_idx,
        'sidon_upsampled': total_saved,
        'copied_48khz': total_copied,
        'skipped': total_skipped,
        'errors': total_errors,
        'method': 'conditional_copy_or_sidon_upsample_cuda'
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("\n" + "="*70)
    logger.info("Processing Complete!")
    logger.info(f"SIDON upsampled: {total_saved}")
    logger.info(f"Copied (already 48kHz): {total_copied}")
    logger.info(f"Skipped (already existed): {total_skipped}")
    logger.info(f"Errors: {total_errors}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Metadata saved: {metadata_file}")
    logger.info("="*70)

    # Count output files
    output_files = list(output_dir.glob("*.wav")) + list(output_dir.glob("*.flac")) + list(output_dir.glob("*.mp3"))
    logger.info(f"\nVerification: {len(output_files)} audio files in output directory")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Audio to Token Converter for SNAC

Converts audio files to SNAC discrete tokens and saves them as Parquet files.
Designed for analyzing token patterns for potential BPE merging.

Usage:
    python audio_to_tokens.py --input_dir /path/to/audio --output_dir /path/to/tokens
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from queue import Queue
from threading import Thread, Event

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pyarrow import parquet as pq
from pyarrow import schema as pa

from snac import SNAC
from snac.dataset import OptimizedAudioDataset


class AudioPreloader:
    """
    Preloads audio files in a background thread to overlap I/O with GPU processing.

    Uses a producer-consumer pattern:
    - Producer thread: Loads audio files from disk
    - Main thread: Processes preloaded audio on GPU
    """

    def __init__(self, sampling_rate: int = 24000, queue_size: int = 128):
        self.sampling_rate = sampling_rate
        self.queue_size = queue_size
        self.queue: Queue = Queue(maxsize=queue_size)  # Buffer up to queue_size audio files
        self.stop_event = Event()
        self.producer_thread: Optional[Thread] = None
        self.audio_files: List[str] = []
        self.current_idx: int = 0

    def _load_audio(self, audio_path: str) -> Optional[Tuple[torch.Tensor, str]]:
        """Load and preprocess a single audio file."""
        try:
            import torchaudio
            waveform, sr = torchaudio.load(audio_path)

            # Resample if needed
            if sr != self.sampling_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)

            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            return waveform, audio_path
        except Exception as e:
            logger.warning(f"Error loading {audio_path}: {e}")
            return None

    def _producer_loop(self):
        """Producer thread: loads audio files into the queue."""
        while not self.stop_event.is_set() and self.current_idx < len(self.audio_files):
            audio_path = self.audio_files[self.current_idx]
            self.current_idx += 1

            result = self._load_audio(audio_path)
            if result is not None:
                # Block until queue has space (backpressure)
                self.queue.put(result)

        # Signal end of data
        self.queue.put(None)  # Sentinel value

    def start(self, audio_files: List[str], start_idx: int = 0):
        """Start the preloader thread with a list of audio files."""
        self.audio_files = audio_files
        self.current_idx = start_idx
        self.stop_event.clear()
        self.queue = Queue(maxsize=self.queue_size)  # Reset queue

        self.producer_thread = Thread(target=self._producer_loop, daemon=True)
        self.producer_thread.start()
        logger.info(f"Audio preloader started (buffering up to {self.queue_size} files)")

    def get_next_batch(self, batch_size: int) -> List[Tuple[torch.Tensor, str]]:
        """Get next batch of preloaded audio files."""
        batch = []
        for _ in range(batch_size):
            # This will block if queue is empty, waiting for producer
            item = self.queue.get()
            if item is None:  # Sentinel value - end of data
                # Put it back for other callers
                self.queue.put(None)
                break
            batch.append(item)
        return batch

    def stop(self):
        """Signal the producer thread to stop."""
        self.stop_event.set()
        if self.producer_thread and self.producer_thread.is_alive():
            self.producer_thread.join(timeout=5)

    @property
    def progress(self) -> int:
        """Return current progress (number of files loaded)."""
        return self.current_idx


def interleave_snac_codes(codes: List[torch.Tensor], offset: int = 128266) -> List[int]:
    """
    Interleave SNAC codes into 7-token frames for language modeling.

    This is the same pattern used in Orpheus training where:
    - SNAC scales are interleaved into groups of 7 tokens
    - Each scale has a different offset to prevent overlap

    Pattern per frame (at time position i):
    1. codes[0][0][i] + offset              (scale 0, coarsest)
    2. codes[1][0][2*i] + offset + 4096     (scale 1, first)
    3. codes[2][0][4*i] + offset + 8192     (scale 2, first)
    4. codes[2][0][4*i+1] + offset + 12288  (scale 2, second)
    5. codes[1][0][2*i+1] + offset + 16384  (scale 1, second)
    6. codes[2][0][4*i+2] + offset + 20480  (scale 2, third)
    7. codes[2][0][4*i+3] + offset + 24592  (scale 2, fourth)

    Args:
        codes: List of code tensors from SNAC encode() (typically 3-4 scales)
        offset: Base offset to add to all codes (default: 128266 for Orpheus)

    Returns:
        List of interleaved codes (length = 7 * num_frames)

    Raises:
        ValueError: If audio is too short for interleaving pattern
    """
    all_codes = []

    # Handle different SNAC configurations (3 or 4 scales)
    if len(codes) == 4:
        scale_0, scale_1, scale_2, scale_3 = codes
    elif len(codes) == 3:
        scale_0, scale_1, scale_2 = codes
        scale_3 = None  # Not used in Orpheus pattern
    else:
        raise ValueError(f"Expected 3 or 4 code scales, got {len(codes)}")

    # Number of frames is determined by scale 0 (coarsest)
    num_frames = scale_0.shape[1]

    # Validate that all scales have enough frames for the interleaving pattern
    # Scale 1 needs at least 2*num_frames, Scale 2 needs at least 4*num_frames
    if scale_1.shape[1] < 2 * num_frames:
        raise ValueError(
            f"Audio too short: scale 1 has {scale_1.shape[1]} frames, "
            f"need at least {2 * num_frames} for {num_frames} output frames"
        )
    if scale_2.shape[1] < 4 * num_frames:
        raise ValueError(
            f"Audio too short: scale 2 has {scale_2.shape[1]} frames, "
            f"need at least {4 * num_frames} for {num_frames} output frames"
        )

    for i in range(num_frames):
        all_codes.append(int(scale_0[0][i].item()) + offset)                      # Frame pos 0
        all_codes.append(int(scale_1[0][2*i].item()) + offset + 4096)             # Frame pos 1
        all_codes.append(int(scale_2[0][4*i].item()) + offset + 8192)             # Frame pos 2
        all_codes.append(int(scale_2[0][4*i+1].item()) + offset + 12288)          # Frame pos 3
        all_codes.append(int(scale_1[0][2*i+1].item()) + offset + 16384)          # Frame pos 4
        all_codes.append(int(scale_2[0][4*i+2].item()) + offset + 20480)          # Frame pos 5
        all_codes.append(int(scale_2[0][4*i+3].item()) + offset + 24592)          # Frame pos 6

    return all_codes


def remove_duplicate_frames(codes_list: List[int]) -> List[int]:
    """
    Remove consecutive duplicate frames from interleaved SNAC codes.

    This is the same deduplication used in Orpheus training:
    - Codes are organized in frames of 7 tokens
    - A frame is kept only if its first token differs from the previous frame's first token
    - First frame is always kept

    Args:
        codes_list: List of interleaved codes (length must be divisible by 7)

    Returns:
        Deduplicated list of codes

    Raises:
        ValueError: If input length is not divisible by 7
    """
    if len(codes_list) % 7 != 0:
        raise ValueError(f"Input list length ({len(codes_list)}) must be divisible by 7")

    result = codes_list[:7]  # Keep first frame

    for i in range(7, len(codes_list), 7):
        current_first = codes_list[i]
        previous_first = result[-7]

        # Only add frame if first token is different
        if current_first != previous_first:
            result.extend(codes_list[i:i+7])

    return result


def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


class AudioTokenizer:
    """Convert audio to SNAC tokens and save as Parquet."""

    def __init__(
        self,
        model_path: str = "hubertsiuzdak/snac_24khz",
        checkpoint_path: str = None,
        segment_length: float = 4.0,
        sampling_rate: int = 24000,
        device: str = "cuda",
        interleaved: bool = False,
        remove_duplicates: bool = False,
        token_offset: int = 128266
    ):
        # Handle device string (allow "0" -> "cuda:0")
        if device.isdigit():
            device = f"cuda:{device}"
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.segment_length = segment_length
        self.sampling_rate = sampling_rate
        self.segment_samples = int(segment_length * sampling_rate)

        # Token processing options
        self.interleaved = interleaved
        self.remove_duplicates = remove_duplicates
        self.token_offset = token_offset

        # Load model
        logger.info(f"Loading SNAC model from {model_path}...")
        if checkpoint_path:
            # Load from checkpoint
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model = SNAC.from_pretrained(model_path).to(self.device)
            model.load_state_dict(checkpoint['model'])
        else:
            # Load pretrained
            model = SNAC.from_pretrained(model_path).to(self.device)

        model.eval()
        self.model = model

        # Log processing mode
        if interleaved:
            logger.info(f"Token mode: INTERLEAVED (7-token frames)")
            if remove_duplicates:
                logger.info(f"  + Duplicate frame removal ENABLED")
            else:
                logger.info(f"  + Duplicate frame removal DISABLED")
        else:
            logger.info(f"Token mode: RAW (separate scales)")

        # Get codebook sizes for each level
        # SNAC has hierarchical VQ with different codebook sizes
        logger.info("Model loaded successfully")
        logger.info(f"Device: {self.device}")

    def tokenize(self, audio: torch.Tensor) -> List[torch.Tensor]:
        """
        Convert audio to SNAC codes.

        Args:
            audio: (batch, channels, samples) tensor

        Returns:
            List of code tensors at different scales (hierarchical)
            Each has shape (batch, codes, time_scale)
        """
        with torch.no_grad():
            codes = self.model.encode(audio)
        return codes

    def process_file(self, audio_path: str, full_audio: bool = False) -> Dict[str, Any]:
        """
        Process a single audio file and extract tokens.

        Args:
            audio_path: Path to audio file
            full_audio: If True, process entire audio without truncation

        Returns:
            Dictionary with:
                - file_path: Original audio file path
                - duration: Audio duration in seconds
                - tokens: List of token arrays (one per scale)
                - shape: Shape of each token array
                - metadata: Additional info
        """
        try:
            # Load audio
            import torchaudio
            waveform, sr = torchaudio.load(audio_path)

            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if needed
            if sr != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
                waveform = resampler(waveform)

            samples = waveform.shape[1]

            if full_audio:
                # Use entire audio without truncation
                # Just ensure minimum length for SNAC
                min_samples = 24000  # 1 second minimum
                if samples < min_samples:
                    # Pad if too short
                    padding = min_samples - samples
                    waveform = F.pad(waveform, (0, padding))
            else:
                # Pad/truncate to segment length
                if samples < self.segment_samples:
                    padding = self.segment_samples - samples
                    waveform = F.pad(waveform, (0, padding))
                elif samples > self.segment_samples:
                    # Truncate
                    waveform = waveform[:, :self.segment_samples]

            # Add batch dimension
            waveform = waveform.unsqueeze(0).to(self.device)

            # Tokenize
            codes = self.tokenize(waveform)

            # Convert to numpy for storage
            tokens_np = [c.squeeze(0).cpu().numpy() for c in codes]

            return {
                'file_path': audio_path,
                'duration': samples / self.sampling_rate,
                'tokens': tokens_np,  # List of arrays at different scales
                'shapes': [t.shape for t in tokens_np],
                'n_tokens': [t.size for t in tokens_np],
                'success': True
            }

        except Exception as e:
            logger.warning(f"Error processing {audio_path}: {e}")
            return {
                'file_path': audio_path,
                'success': False,
                'error': str(e)
            }

    def get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration quickly without loading full audio."""
        try:
            import torchaudio
            # Use backend_info to get duration without loading full audio
            info = torchaudio.backend.soundfile_backend.info(audio_path)
            return info.num_frames / info.sample_rate
        except Exception as e:
            # Fallback: load just the first chunk
            try:
                import torchaudio
                waveform, sr = torchaudio.load(audio_path, num_frames=1)
                return 0.0  # Duration unknown, will sort to beginning
            except:
                logger.warning(f"Error getting duration for {audio_path}: {e}")
                return 0.0

    def process_batch(
        self,
        audio_files: List[str],
        full_audio: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of audio files in parallel.

        Args:
            audio_files: List of audio file paths (similar lengths)
            full_audio: Process full audio without truncation

        Returns:
            List of results with tokens and metadata
        """
        import torchaudio

        results = []
        max_length = 0

        # First pass: load all audio and find max length
        waveforms = []
        for audio_path in audio_files:
            try:
                waveform, sr = torchaudio.load(audio_path)
                if sr != self.sampling_rate:
                    waveform = torchaudio.functional.resample(waveform, sr, self.sampling_rate)

                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                max_length = max(max_length, waveform.shape[1])
                waveforms.append(waveform)
            except Exception as e:
                logger.warning(f"Error loading {audio_path}: {e}")
                waveforms.append(None)

        # Second pass: pad and batch process
        batch_waveforms = []
        valid_indices = []

        for i, waveform in enumerate(waveforms):
            if waveform is None:
                results.append({
                    'file_path': audio_files[i],
                    'success': False,
                    'error': 'Failed to load'
                })
                continue

            # Pad to max length
            if waveform.shape[1] < max_length:
                waveform = F.pad(waveform, (0, max_length - waveform.shape[1]))

            batch_waveforms.append(waveform)
            valid_indices.append(i)

        if not batch_waveforms:
            return results

        # Stack into batch [B, 1, T] - keep channel dimension!
        batch_waveforms = torch.stack(batch_waveforms, dim=0).to(self.device)  # [B, 1, T]

        # Tokenize batch
        try:
            with torch.no_grad():
                codes_batch = self.tokenize(batch_waveforms)

            # Process results
            for batch_idx, file_idx in enumerate(valid_indices):
                codes = [c[batch_idx] for c in codes_batch]

                try:
                    # Apply interleaving and deduplication if enabled
                    if self.interleaved:
                        # Interleave into 7-token frames (Orpheus-style)
                        interleaved_codes = interleave_snac_codes(codes, offset=self.token_offset)

                        if self.remove_duplicates:
                            # Remove consecutive duplicate frames
                            interleaved_codes = remove_duplicate_frames(interleaved_codes)

                        # Store as interleaved codes
                        tokens_list = [np.array(interleaved_codes)]
                        shapes = [(len(interleaved_codes),)]  # 1D array
                        n_tokens = [len(interleaved_codes)]
                    else:
                        # Store as separate scales (raw mode)
                        tokens_np = [c.cpu().numpy() for c in codes]
                        tokens_list = tokens_np
                        shapes = [t.shape for t in tokens_np]
                        n_tokens = [t.size for t in tokens_np]

                    results.append({
                        'file_path': audio_files[file_idx],
                        'duration': waveforms[batch_idx].shape[1] / self.sampling_rate,
                        'tokens': tokens_list,
                        'shapes': shapes,
                        'n_tokens': n_tokens,
                        'success': True
                    })
                except (ValueError, IndexError) as e:
                    # Audio too short for interleaving pattern
                    results.append({
                        'file_path': audio_files[file_idx],
                        'success': False,
                        'error': f'Audio too short: {str(e)}'
                    })
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            for file_idx in valid_indices:
                results.append({
                    'file_path': audio_files[file_idx],
                    'success': False,
                    'error': str(e)
                })

        return results

    def process_preloaded_batch(
        self,
        waveforms_and_paths: List[Tuple[torch.Tensor, str]],
        full_audio: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of already-loaded audio waveforms on GPU.

        This is used with AudioPreloader to overlap I/O with GPU processing.

        Args:
            waveforms_and_paths: List of (waveform, file_path) tuples
            full_audio: Process full audio without truncation

        Returns:
            List of results with tokens and metadata
        """
        if not waveforms_and_paths:
            return []

        results = []
        waveforms_list = []
        audio_paths = []

        # Separate waveforms and paths, find max length
        max_length = 0
        for waveform, audio_path in waveforms_and_paths:
            max_length = max(max_length, waveform.shape[1])
            waveforms_list.append(waveform)
            audio_paths.append(audio_path)

        # Pad all waveforms to max length
        batch_waveforms = []
        for waveform in waveforms_list:
            if waveform.shape[1] < max_length:
                waveform = F.pad(waveform, (0, max_length - waveform.shape[1]))
            batch_waveforms.append(waveform)

        # Stack into batch [B, 1, T] and move to GPU
        batch_waveforms = torch.stack(batch_waveforms, dim=0).to(self.device)

        # Tokenize batch
        try:
            with torch.no_grad():
                codes_batch = self.tokenize(batch_waveforms)

            # Process results
            for batch_idx, audio_path in enumerate(audio_paths):
                codes = [c[batch_idx] for c in codes_batch]

                try:
                    # Apply interleaving and deduplication if enabled
                    if self.interleaved:
                        # Interleave into 7-token frames (Orpheus-style)
                        interleaved_codes = interleave_snac_codes(codes, offset=self.token_offset)

                        if self.remove_duplicates:
                            # Remove consecutive duplicate frames
                            interleaved_codes = remove_duplicate_frames(interleaved_codes)

                        # Store as interleaved codes
                        tokens_list = [np.array(interleaved_codes)]
                        shapes = [(len(interleaved_codes),)]  # 1D array
                        n_tokens = [len(interleaved_codes)]
                    else:
                        # Store as separate scales (raw mode)
                        tokens_np = [c.cpu().numpy() for c in codes]
                        tokens_list = tokens_np
                        shapes = [t.shape for t in tokens_np]
                        n_tokens = [t.size for t in tokens_np]

                    results.append({
                        'file_path': audio_path,
                        'duration': waveforms_list[batch_idx].shape[1] / self.sampling_rate,
                        'tokens': tokens_list,
                        'shapes': shapes,
                        'n_tokens': n_tokens,
                        'success': True
                    })
                except (ValueError, IndexError) as e:
                    # Audio too short for interleaving pattern
                    results.append({
                        'file_path': audio_path,
                        'success': False,
                        'error': f'Audio too short: {str(e)}'
                    })
        except Exception as e:
            logger.error(f"Error processing preloaded batch: {e}")
            for audio_path in audio_paths:
                results.append({
                    'file_path': audio_path,
                    'success': False,
                    'error': str(e)
                })

        return results

    def process_dataset(
        self,
        audio_dir: str,
        output_dir: str,
        batch_size: int = 5000,
        gpu_batch_size: int = 16,
        max_files: int = None,
        file_pattern: str = "*.wav",
        full_audio: bool = False,
        start_idx: int = 0,
        end_idx: int = None
    ) -> Dict[str, Any]:
        """
        Process entire dataset with GPU batching for efficiency.

        Args:
            audio_dir: Directory containing audio files
            output_dir: Where to save parquet files
            batch_size: Files per parquet batch
            gpu_batch_size: Files to process in parallel on GPU
            max_files: Maximum number of files to process (None = all)
            file_pattern: Glob pattern for audio files
            full_audio: Process entire audio without truncation
            start_idx: Starting file index (for splitting work across GPUs)
            end_idx: Ending file index (for splitting work across GPUs)

        Returns:
            Statistics about tokenized data
        """
        import subprocess

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all audio files
        logger.info(f"Searching for audio files in {audio_dir}...")
        audio_path = Path(audio_dir)

        try:
            result = subprocess.run(
                ['find', str(audio_path), '-type', 'f', '-iname', '*.wav'],
                capture_output=True,
                text=True,
                check=False
            )
            audio_files = [f.strip() for f in result.stdout.split('\n') if f.strip()]
        except Exception as e:
            logger.warning(f"find command failed: {e}, falling back to glob")
            audio_files = [str(f) for f in audio_path.glob("**/*.wav")]

        if max_files:
            audio_files = audio_files[:max_files]

        # Apply file range for splitting work across GPUs
        total_files = len(audio_files)
        if end_idx is None:
            end_idx = total_files

        # Validate indices
        if start_idx >= total_files:
            logger.error(f"start_idx ({start_idx}) >= total files ({total_files})")
            return {'total_files': 0, 'error': 'start_idx out of range'}

        if start_idx > 0 or end_idx < total_files:
            audio_files = audio_files[start_idx:end_idx]
            logger.info(f"Processing file range: [{start_idx}:{end_idx}] of {total_files} total files")

        logger.info(f"Found {len(audio_files)} audio files to process")
        logger.info(f"GPU batch size: {gpu_batch_size} (parallel processing)")
        if full_audio:
            logger.info("Processing FULL AUDIO (no truncation)")

        # Process in GPU batches
        all_data = []
        # Use dict for scale stats to handle both interleaved (1 scale) and raw (4 scales) modes
        scale_stats = {}
        batch_count = 0
        files_processed = 0
        total_duration = 0.0

        # Start audio preloader (background thread for I/O)
        preloader = AudioPreloader(sampling_rate=self.sampling_rate, queue_size=15000)
        preloader.start(audio_files, start_idx=0)  # Already sliced to range

        # Process in GPU batches using preloaded audio
        # Use a different loop condition since we're pulling from a queue
        batch_num = 0
        total_batches = (len(audio_files) + gpu_batch_size - 1) // gpu_batch_size

        with tqdm(total=total_batches, desc="Tokenizing (with I/O overlap)") as pbar:
            while True:
                # Get next batch of preloaded audio
                waveforms_and_paths = preloader.get_next_batch(gpu_batch_size)

                if not waveforms_and_paths:
                    # No more data
                    break

                # Process preloaded batch on GPU
                batch_results = self.process_preloaded_batch(waveforms_and_paths, full_audio=full_audio)

                for result in batch_results:
                    if result.get('success', False):
                        tokens_list = result['tokens']

                        # Create row for parquet
                        row = {
                            'file_path': result['file_path'],
                            'duration': result['duration'],
                        }

                        # Add tokens for each scale
                        for scale_idx, tokens in enumerate(tokens_list):
                            if scale_idx not in scale_stats:
                                scale_stats[scale_idx] = {'values': [], 'total': 0}

                            row[f'tokens_scale_{scale_idx}'] = tokens.flatten().tolist()
                            row[f'n_tokens_scale_{scale_idx}'] = tokens.size
                            row[f'shape_scale_{scale_idx}'] = str(tokens.shape)

                            # Collect statistics
                            scale_stats[scale_idx]['values'].extend(tokens.flatten().tolist())
                            scale_stats[scale_idx]['total'] += tokens.size

                        all_data.append(row)
                        files_processed += 1
                        total_duration += result['duration']

                    # Write to parquet every batch_size files
                    if len(all_data) >= batch_size:
                        df = pd.DataFrame(all_data)
                        parquet_file = output_path / f"tokens_batch_{batch_count:04d}.parquet"
                        df.to_parquet(parquet_file, index=False)
                        logger.info(f"  Saved batch {batch_count}: {parquet_file} ({len(df)} files, {files_processed} total)")
                        all_data = []
                        batch_count += 1

                # Update progress bar
                pbar.update(1)

        # Stop preloader thread
        preloader.stop()

        # Save remaining files to parquet
        if all_data:
            df = pd.DataFrame(all_data)
            parquet_file = output_path / f"tokens_batch_{batch_count:04d}.parquet"
            df.to_parquet(parquet_file, index=False)
            logger.info(f"  Saved final batch {batch_count}: {parquet_file} ({len(df)} files, {files_processed} total)")

        # Save metadata
        if files_processed > 0:
            metadata = {
                'total_files': files_processed,
                'total_duration': total_duration,
                'segment_length': self.segment_length,
                'sampling_rate': self.sampling_rate,
                'scales': len(scale_stats),
                'interleaved': self.interleaved,
                'remove_duplicates': self.remove_duplicates,
                'token_offset': self.token_offset if self.interleaved else None,
                'scale_stats': {}
            }

            for scale_idx, stats in scale_stats.items():
                values = np.array(stats['values'])
                metadata['scale_stats'][f'scale_{scale_idx}'] = {
                    'n_tokens': stats['total'],
                    'min': int(values.min()) if len(values) > 0 else 0,
                    'max': int(values.max()) if len(values) > 0 else 0,
                    'unique': int(len(set(values))),
                    'most_common': {}
                }

                # Get top 20 most common tokens at this scale
                unique, counts = np.unique(values, return_counts=True)
                top_indices = np.argsort(counts)[-20:][::-1]
                for idx in top_indices:
                    token = int(unique[idx])
                    count = int(counts[idx])
                    metadata['scale_stats'][f'scale_{scale_idx}']['most_common'][token] = count

            # Save metadata
            metadata_file = output_path / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"\nMetadata saved to: {metadata_file}")

            # Print summary
            logger.info("\n" + "="*70)
            logger.info("Tokenization Summary:")
            logger.info("="*70)
            logger.info(f"Total files processed: {metadata['total_files']}")
            logger.info(f"Total duration: {metadata['total_duration']/3600:.2f} hours")
            logger.info(f"Segment length: {metadata['segment_length']}s")
            logger.info(f"Number of scales: {metadata['scales']}")

            if metadata.get('interleaved', False):
                logger.info(f"Token mode: INTERLEAVED (7-token frames)")
                if metadata.get('remove_duplicates', False):
                    logger.info(f"  Duplicate frame removal: ENABLED")
                else:
                    logger.info(f"  Duplicate frame removal: DISABLED")
                logger.info(f"  Token offset: {metadata.get('token_offset', 128266)}")
            else:
                logger.info(f"Token mode: RAW (separate scales)")

            for scale_idx in range(metadata['scales']):
                stats = metadata['scale_stats'][f'scale_{scale_idx}']
                scale_name = "Interleaved" if metadata.get('interleaved', False) else f"Scale {scale_idx}"
                logger.info(f"\n{scale_name}:")
                logger.info(f"  Total tokens: {stats['n_tokens']:,}")
                logger.info(f"  Unique tokens: {stats['unique']:,}")
                logger.info(f"  Min token: {stats['min']}")
                logger.info(f"  Max token: {stats['max']}")

                logger.info(f"  Top 10 most common tokens:")
                top_10 = list(stats['most_common'].items())[:10]
                for token, count in top_10:
                    logger.info(f"    Token {token:6d}: {count:10d} occurrences ({count/stats['n_tokens']*100:.4f}%)")

            logger.info("="*70)

            return metadata

        else:
            logger.error("No files were successfully processed!")
            return {}


def analyze_ngrams(tokens: np.array, max_n: int = 7, top_k: int = 20) -> dict:
    """
    Analyze n-gram patterns from 1 to max_n.

    Args:
        tokens: 1D array of tokens
        max_n: Maximum n-gram length to analyze
        top_k: Number of top n-grams to show

    Returns:
        Dictionary with n-gram statistics
    """
    from collections import Counter

    ngram_stats = {}

    for n in range(1, max_n + 1):
        # Extract n-grams
        ngrams = []
        for tokens_list in [tokens]:  # Process as single sequence
            if len(tokens_list) >= n:
                for i in range(len(tokens_list) - n + 1):
                    ngrams.append(tuple(tokens_list[i:i+n]))

        # Count n-grams
        ngram_counter = Counter(ngrams)
        total_ngrams = len(ngrams)
        unique_ngrams = len(ngram_counter)

        # Calculate coverage statistics
        sorted_ngrams = ngram_counter.most_common()
        top_k_counts = [count for _, count in sorted_ngrams[:top_k]]
        top_k_coverage = sum(top_k_counts) / total_ngrams * 100 if total_ngrams > 0 else 0

        # Calculate possible vs actual
        possible_ngrams = len(set(tokens)) ** n

        ngram_stats[n] = {
            'total': total_ngrams,
            'unique': unique_ngrams,
            'possible': possible_ngrams,
            'diversity': unique_ngrams / possible_ngrams * 100 if possible_ngrams > 0 else 0,
            'top_k_coverage': top_k_coverage,
            'top_patterns': sorted_ngrams[:top_k]
        }

    return ngram_stats


def analyze_token_patterns(tokens_dir: str):
    """
    Analyze token patterns for BPE potential.

    Args:
        tokens_dir: Directory containing parquet token files
    """
    logger.info(f"Analyzing token patterns in {tokens_dir}...")

    tokens_path = Path(tokens_dir)
    parquet_files = sorted(tokens_path.glob("*.parquet"))

    if not parquet_files:
        logger.error(f"No parquet files found in {tokens_dir}")
        return

    logger.info(f"Found {len(parquet_files)} parquet files")

    # Load all data
    all_data = []
    for pq_file in tqdm(parquet_files, desc="Loading parquet files"):
        df = pd.read_parquet(pq_file)
        all_data.append(df)

    full_df = pd.concat(all_data, ignore_index=True)

    logger.info(f"Loaded {len(full_df)} tokenized audio segments")

    # Analyze patterns
    logger.info("\n" + "="*70)
    logger.info("BPE Potential Analysis (N-gram: 1-7)")
    logger.info("="*70)

    for scale_idx in range(4):
        token_col = f'tokens_scale_{scale_idx}'
        if token_col not in full_df.columns:
            continue

        # Get all tokens at this scale
        all_tokens = []
        for tokens_list in full_df[token_col]:
            all_tokens.extend(tokens_list)

        all_tokens = np.array(all_tokens)
        n_tokens = len(all_tokens)

        logger.info(f"\n{'='*70}")
        logger.info(f"SCALE {scale_idx} Analysis ({n_tokens:,} total tokens)")
        logger.info('='*70)

        # Run n-gram analysis for 1-7
        ngram_stats = analyze_ngrams(all_tokens, max_n=7, top_k=10)

        for n, stats in ngram_stats.items():
            logger.info(f"\n{n}-gram Patterns:")
            logger.info(f"  Total {n}-grams: {stats['total']:,}")
            logger.info(f"  Unique {n}-grams: {stats['unique']:,}")
            logger.info(f"  Possible {n}-grams: {stats['possible']:,}")
            logger.info(f"  Diversity: {stats['diversity']:.4f}%")
            logger.info(f"  Top-10 coverage: {stats['top_k_coverage']:.4f}%")

            if stats['top_patterns']:
                logger.info(f"  Top 10 {n}-grams:")
                for i, (pattern, count) in enumerate(stats['top_patterns'], 1):
                    pattern_str = str(pattern)
                    if len(pattern_str) > 60:
                        pattern_str = pattern_str[:57] + "..."
                    logger.info(f"    {i:2d}. {pattern_str:60s} : {count:8d}x ({count/stats['total']*100:.6f}%)")

        # BPE Recommendation based on n-grams
        logger.info(f"\nBPE Recommendation for Scale {scale_idx}:")
        logger.info("-" * 70)

        # Check if longer n-grams show more concentration
        coverage_1 = ngram_stats[1]['top_k_coverage']
        coverage_7 = ngram_stats[7]['top_k_coverage']
        diversity_1 = ngram_stats[1]['diversity']
        diversity_7 = ngram_stats[7]['diversity']

        logger.info(f"1-gram top-10 coverage: {coverage_1:.2f}%")
        logger.info(f"7-gram top-10 coverage: {coverage_7:.2f}%")
        logger.info(f"1-gram diversity: {diversity_1:.4f}%")
        logger.info(f"7-gram diversity: {diversity_7:.4f}%")

        if coverage_7 > coverage_1 * 2:
            logger.info(f"  ✓ Longer patterns show MORE concentration")
            logger.info(f"  → N-gram merging (up to 7 tokens) could be very effective")
        elif coverage_7 > coverage_1 * 1.2:
            logger.info(f"  ○ Longer patterns show moderately more concentration")
            logger.info(f"  → N-gram merging could help somewhat")
        else:
            logger.info(f"  ✗ Longer patterns don't show better concentration")
            logger.info(f"  → N-gram merging may not help")

        if diversity_7 < 0.1:
            logger.info(f"  ✓ Very low 7-gram diversity → Strong patterns")
        elif diversity_7 < 1.0:
            logger.info(f"  ○ Low 7-gram diversity → Some patterns")
        else:
            logger.info(f"  ✗ High 7-gram diversity → Weak patterns")

    logger.info("="*70)


def main():
    parser = argparse.ArgumentParser(description="Convert audio to SNAC tokens for BPE analysis")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with audio files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for parquet files")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to fine-tuned checkpoint (optional)")
    parser.add_argument("--model", type=str, default="hubertsiuzdak/snac_24khz", help="Pretrained model path")
    parser.add_argument("--segment_length", type=float, default=4.0, help="Audio segment length in seconds")
    parser.add_argument("--batch_size", type=int, default=5000, help="Files per parquet batch")
    parser.add_argument("--gpu_batch_size", type=int, default=32, help="Files to process in parallel on GPU")
    parser.add_argument("--max_files", type=int, default=None, help="Max files to process (for testing)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--full_audio", action="store_true", help="Process full audio without truncation")
    parser.add_argument("--analyze_only", action="store_true", help="Only analyze existing tokens, don't create new ones")
    parser.add_argument("--interleaved", action="store_true", help="Interleave codes into 7-token frames (Orpheus-style)")
    parser.add_argument("--remove_duplicates", action="store_true", help="Remove consecutive duplicate frames (requires --interleaved)")
    parser.add_argument("--token_offset", type=int, default=128266, help="Base token offset for interleaved mode")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting file index (for splitting work across GPUs)")
    parser.add_argument("--end_idx", type=int, default=None, help="Ending file index (for splitting work across GPUs)")

    args = parser.parse_args()

    # Validate arguments
    if args.remove_duplicates and not args.interleaved:
        parser.error("--remove_duplicates requires --interleaved")

    global logger
    logger = setup_logging()

    if args.analyze_only:
        # Just analyze existing tokens
        analyze_token_patterns(args.output_dir)
    else:
        # Create tokens
        logger.info("Audio to Token Converter for SNAC")
        logger.info(f"Input directory: {args.input_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"GPU batch size: {args.gpu_batch_size} (parallel processing)")

        tokenizer = AudioTokenizer(
            model_path=args.model,
            checkpoint_path=args.checkpoint,
            segment_length=args.segment_length,
            device=args.device,
            interleaved=args.interleaved,
            remove_duplicates=args.remove_duplicates,
            token_offset=args.token_offset
        )

        metadata = tokenizer.process_dataset(
            audio_dir=args.input_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            gpu_batch_size=args.gpu_batch_size,
            max_files=args.max_files,
            full_audio=args.full_audio,
            start_idx=args.start_idx,
            end_idx=args.end_idx
        )

        # After creating tokens, analyze them
        if metadata.get('total_files', 0) > 0:
            logger.info("\n" + "="*70)
            analyze_token_patterns(args.output_dir)


if __name__ == "__main__":
    main()

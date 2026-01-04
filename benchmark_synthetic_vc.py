#!/usr/bin/env python3
"""Benchmark synthetic voice conversion operations in isolation."""
print("starting benchmark_synthetic_vc.py...")
import torch
import time
from snac import SNACWithSpeakerConditioning
from snac.audio_augmentation import augment_audio_for_voice_conversion

print("Loading model...")
model = SNACWithSpeakerConditioning.from_pretrained_base(
    repo_id="hubertsiuzdak/snac_24khz",
    speaker_emb_dim=512,
    speaker_encoder_type='eres2net',
    freeze_base=True,
).to('cuda:0')
model.eval()

# Test audio: 2 seconds at 24kHz
duration = 2.0
sample_rate = 24000
num_samples = int(duration * sample_rate)
audio = torch.randn(1, 1, num_samples).to('cuda:0')
speaker_emb = torch.randn(1, 512).to('cuda:0')

pitch_shift_range = [-2, -1, 1, 2]

print("\n" + "="*70)
print(f"SYNTHETIC VC BENCHMARK - Audio: {duration}s ({num_samples} samples)")
print("="*70)

# Test each semitone value
for semitones in pitch_shift_range:
    print(f"\nSemitone: {semitones:+d}")
    print("-" * 70)

    # Warmup
    for _ in range(3):
        audio_aug, _, _ = augment_audio_for_voice_conversion(audio, pitch_shift_range=[semitones], probability=1.0)

    # Benchmark pitch shifting
    times = []
    for _ in range(10):
        start = time.perf_counter()
        audio_aug, was_aug, _ = augment_audio_for_voice_conversion(audio, pitch_shift_range=[semitones], probability=1.0)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    pitch_shift_time = sum(times) / len(times)
    print(f"  Pitch shift: {pitch_shift_time:.2f}ms avg")

    # Benchmark encode
    times = []
    for _ in range(10):
        start = time.perf_counter()
        codes = model.encode(audio_aug)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    encode_time = sum(times) / len(times)
    print(f"  Encode:      {encode_time:.2f}ms avg")

    # Benchmark decode
    times = []
    for _ in range(10):
        start = time.perf_counter()
        audio_recon = model.decode(codes, speaker_embedding=speaker_emb)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    decode_time = sum(times) / len(times)
    print(f"  Decode:      {decode_time:.2f}ms avg")

    total = pitch_shift_time + encode_time + decode_time
    print(f"  TOTAL:       {total:.2f}ms per sample")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("Overhead per synthetic VC sample:")
print("  - Pitch shift: ~5-15ms")
print("  - Encode:        ~100-200ms")
print("  - Decode:        ~100-200ms")
print("  - Total:         ~200-400ms per sample")
print(f"\nFor batch of 6 with 50% probability:")
print(f"  Expected samples: 3")
print(f"  Expected overhead: 600-1200ms per batch")

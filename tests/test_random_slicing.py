#!/usr/bin/env python3
"""Test random slicing implementation with cached codes."""

import torch
import sys
sys.path.insert(0, '/mnt/data/work/snac')

from finetune_decoder_48khz_simple_cached import CachedCodesDataset

print("Testing random slicing with cached codes...")
print("=" * 60)

# Create dataset
dataset = CachedCodesDataset(
    "/mnt/data/codes_phase11/train",
    "/mnt/data/combine/train/audio_48khz",
    segment_length=4.0,
)

print(f"Dataset size: {len(dataset)} files")
print(f"Segment length: 4.0s")
print(f"Expected shapes: S0=28, S1=55, S2=109, Audio=192000")
print("=" * 60)

# Test 10 samples
print("\nTesting 10 random samples:")
print("-" * 60)

for i in range(10):
    try:
        result = dataset[i]
        if result is None:
            print(f"Sample {i}: Skipped (file too short or error)")
            continue

        codes = result['codes']
        audio = result['audio']

        # Verify shapes
        print(f"\nSample {i}:")
        print(f"  Codes[0]: {codes[0].shape} (expected: [28])")
        print(f"  Codes[1]: {codes[1].shape} (expected: [55])")
        print(f"  Codes[2]: {codes[2].shape} (expected: [109])")
        print(f"  Audio: {audio.shape} (expected: [192000])")

        # Verify alignment
        # All scales should represent same time range
        # Check that ratios match expected vq_strides [8, 4, 2]
        if codes[0].shape[0] == 28 and codes[1].shape[0] == 55 and codes[2].shape[0] == 109:
            print(f"  ✅ Shapes correct!")
        else:
            print(f"  ❌ Shape mismatch!")

        if audio.shape[0] == 192000:
            print(f"  ✅ Audio length correct!")
        else:
            print(f"  ❌ Audio length mismatch!")

    except Exception as e:
        print(f"Sample {i}: ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete!")

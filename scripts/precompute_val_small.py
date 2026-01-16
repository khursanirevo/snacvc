"""
Quick script to cache validation codes for small subset
"""
import os
import json
import torch
from pathlib import Path
from tqdm import tqdm
import pandas as pd

os.environ['HF_HOME'] = '/mnt/data/work/snac/.hf_cache'
os.environ['HF_HUB_CACHE'] = '/mnt/data/work/snac/.hf_cache/hub'

from snac import SNAC
from snac.dataset import OptimizedAudioDataset
from finetune_decoder_48khz import SIDONUpsampler

device = torch.device('cuda:0')

print("Loading models...")
sidon = SIDONUpsampler(device)
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
for param in model.encoder.parameters():
    param.requires_grad = False
for param in model.quantizer.parameters():
    param.requires_grad = False

print("Loading validation dataset...")
dataset = OptimizedAudioDataset(
    "/mnt/data/combine/valid/audio",
    sampling_rate=24000,
    segment_length=4.0,
    augment=False,
)

print(f"\nCaching codes for first 1000 validation files...")
cache_dir = Path("/mnt/data/codes_phase11_small/val")
cache_dir.mkdir(parents=True, exist_ok=True)

all_data = []
batch_size = 1000
num_files = 1000

for idx in tqdm(range(min(num_files, len(dataset))), desc="Caching"):
    try:
        audio = dataset[idx]['audio']
        audio_tensor = audio.unsqueeze(0).unsqueeze(0).to(device)

        if audio_tensor.shape[-1] < 24000:
            continue

        # Generate 48kHz target
        sample_rate, audio_48k = sidon(audio_tensor, sample_rate=24000)
        if audio_48k is None or audio_48k.numel() == 0:
            continue

        # Encode
        with torch.no_grad():
            z = model.encoder(audio_tensor)
            _, codes = model.quantizer(z)

        all_data.append({
            'file_path': dataset.samples[idx].name,
            'codes_scale_0': codes[0].cpu().numpy().tolist(),
            'codes_scale_1': codes[1].cpu().numpy().tolist(),
            'codes_scale_2': codes[2].cpu().numpy().tolist(),
            'shape_scale_0': (codes[0].shape[0],),
            'shape_scale_1': (codes[1].shape[0],),
            'shape_scale_2': (codes[2].shape[0],),
        })

    except Exception as e:
        continue

# Save
if all_data:
    df = pd.DataFrame(all_data)
    df.to_parquet(cache_dir / f"codes_batch_0.parquet", index=False)
    print(f"✓ Saved batch 0 ({len(all_data)} files)")
else:
    print("✗ No valid files found!")

# Metadata
metadata = {
    'total_samples': len(all_data),
    'num_codebooks': 3,
    'codebook_size': 4096,
    'vq_strides': [4, 2, 1],
    'hop_length': 512,
}
with open(cache_dir / "metadata.json", 'w') as f:
    json.dump(metadata, f)

print(f"\n✅ Done! Cached {len(all_data)} validation files")

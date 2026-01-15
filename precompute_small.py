"""
Quick script to cache codes for small subset (10K files)
Then train fast with cached codes
"""
import os
import json
import torch
import torch.nn.functional as F
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

print("Loading dataset...")
dataset = OptimizedAudioDataset(
    "/mnt/data/combine/train/audio",
    sampling_rate=24000,
    segment_length=4.0,
    augment=False,
)

print(f"\nCaching codes for first 10000 files...")
cache_dir = Path("/mnt/data/codes_phase11_small/train")
cache_dir.mkdir(parents=True, exist_ok=True)

all_data = []
batch_size = 1000
num_files = 10000

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

        # Save batch
        if len(all_data) >= batch_size:
            df = pd.DataFrame(all_data)
            batch_num = len(list(cache_dir.glob("*.parquet")))
            df.to_parquet(cache_dir / f"codes_batch_{batch_num}.parquet", index=False)
            print(f"✓ Saved batch {batch_num} ({len(all_data)} files)")
            all_data = []

    except Exception as e:
        continue

# Save remaining
if all_data:
    df = pd.DataFrame(all_data)
    batch_num = len(list(cache_dir.glob("*.parquet")))
    df.to_parquet(cache_dir / f"codes_batch_{batch_num}.parquet", index=False)
    print(f"✓ Saved final batch {batch_num} ({len(all_data)} files)")

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

print(f"\n✅ Done! Cached codes to {cache_dir}")
print(f"Total batches: {len(list(cache_dir.glob('*.parquet')))}")

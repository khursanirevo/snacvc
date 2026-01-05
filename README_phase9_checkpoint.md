# Phase 9 Fine-tuned SNAC Model

This is a fine-tuned version of the SNAC audio codec, trained on Levantine Arabic data for improved reconstruction quality.

## Installation

```bash
# Install dependencies
pip install torch torchaudio
pip install snac  # or: pip install git+https://github.com/hubertsiuzdak/snac.git
```

## Quick Start

### 1. Basic Usage (Reconstruct Audio)

```python
import torch
from snac import SNAC

# Load fine-tuned model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)

# Load fine-tuned weights
checkpoint = torch.load("checkpoints/phase9_conservative/best_model.pt", map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()

# Load audio
import torchaudio
audio, sr = torchaudio.load("input.wav")
audio = audio.mean(dim=0, keepdim=True) if audio.shape[0] > 1 else audio  # Convert to mono
if sr != 24000:
    audio = torchaudio.transforms.Resample(sr, 24000)(audio)

# Add batch/channel dimensions: (1, 1, T)
audio = audio.unsqueeze(0).to(device)

# Reconstruct
with torch.no_grad():
    audio_reconstructed, codes = model(audio)

# Save output
torchaudio.save("output.wav", audio_reconstructed.squeeze(0).cpu(), 24000)
```

### 2. Compare: Base SNAC vs Fine-tuned

```python
import torch
import torchaudio
from snac import SNAC

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load base pretrained model
model_base = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
model_base.eval()

# Load fine-tuned model
checkpoint = torch.load("checkpoints/phase9_conservative/best_model.pt", map_location=device)
model_finetuned = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
model_finetuned.load_state_dict(checkpoint['model'])
model_finetuned.eval()

# Load audio
audio, sr = torchaudio.load("input.wav")
audio = audio.mean(dim=0, keepdim=True) if audio.shape[0] > 1 else audio
if sr != 24000:
    audio = torchaudio.transforms.Resample(sr, 24000)(audio)
audio = audio.unsqueeze(0).to(device)  # (1, 1, T)

# Reconstruct with both
with torch.no_grad():
    recon_base, _ = model_base(audio)
    recon_finetuned, _ = model_finetuned(audio)

# Save both outputs
torchaudio.save("output_base.wav", recon_base.squeeze(0).cpu(), 24000)
torchaudio.save("output_finetuned.wav", recon_finetuned.squeeze(0).cpu(), 24000)
```

## Model Details

- **Base Model**: `hubertsiuzdak/snac_24khz` (0.98 kbps neural audio codec)
- **Fine-tuning**: Phase 9 - Conservative approach (only decoder trained)
- **Training Data**: Levantine Arabic audio (~100k files)
- **Improvement**: ~11% better reconstruction (L1 + STFT loss)
- **Sample Rate**: 24kHz mono
- **Segment Length**: 2.0 seconds recommended

## Training Configuration

```python
# From configs/phase9_conservative.json
{
    "pretrained_model": "hubertsiuzdak/snac_24khz",
    "learning_rate": 5e-6,
    "freeze_encoder": true,
    "freeze_vq": true,
    "num_epochs": 10,
    "segment_length": 2.0,
    "batch_size": 32
}
```

## Evaluation Results

On 5k random samples vs base SNAC:

| Dataset | Loss (base) | Loss (fine-tuned) | Improvement |
|---------|-------------|-------------------|-------------|
| Levantine Arabic | 0.4305 | 0.3833 | **-11.0%** |
| General (ohmybahasa) | 0.7384 | 0.6599 | **-10.6%** |

## File Structure

```
checkpoints/phase9_conservative/
├── best_model.pt              # Best validation loss checkpoint (epoch 9)
├── checkpoint_epoch9.pt       # Final epoch checkpoint
├── baseline_metrics.json      # Pre-training baseline
└── config.json                # Training configuration
```

## Notes

- Model preserves original SNAC capabilities while improving reconstruction on training data
- Only decoder weights were modified (encoder and VQ frozen)
- Works with standard SNAC API - no code changes needed
- Input must be 24kHz mono audio

## Citation

If you use this model, please cite the original SNAC paper:

```bibtex
@article{siuzdak2024snac,
  title={SNAC: Multi-Scale Neural Audio Codec},
  author={Siuzdak, Hubert},
  year={2024}
}
```

## License

This fine-tuned model inherits the license from the original SNAC model.

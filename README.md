# SNAC Fine-tuning 🍿

Multi-**S**cale **N**ural **A**udio **C**odec (SNAC) fine-tuned on Levantine Arabic speech datasets.

## Quick Start

### 1. Install dependencies

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

### 2. Prepare your dataset

Organize your audio files as follows:

```
dataset/
├── train/
│   ├── file1.wav
│   ├── file2.wav
│   └── ...
└── val/
    ├── file1.wav
    ├── file2.wav
    └── ...
```

Or use the included dataset preparation script:

```bash
python prepare_dataset_folder.py \
    --input_dir /path/to/your/audio \
    --train_ratio 0.95 \
    --output_dir /path/to/dataset
```

### 3. Fine-tune SNAC

```bash
# Single GPU training
python finetune.py --config configs/phase10_revolab_all.json --device 0

# Multi-GPU training (DDP)
python finetune.py --config configs/phase10_revolab_all.json --ddp
```

### 4. Generate audio

```bash
# Encode/decode audio
python inference.py \
    --checkpoint checkpoints/phase10_revolab_all/best_model.pt \
    --input audio.wav \
    --output reconstructed.wav

# Generate from codes
python generate.py \
    --checkpoint checkpoints/phase10_revolab_all/best_model.pt \
    --output generated.wav
```

## Training Configuration

### Curriculum Learning

The training uses curriculum learning with progressively longer audio segments:

| Epochs | Segment Length | Batch Size | Description |
|--------|---------------|------------|-------------|
| 1-2 | 1.0s | 96 | Fast iterations, foundation |
| 3-4 | 2.0s | 48 | Medium context |
| 5-6 | 3.0s | 28 | Longer context |
| 7-10 | 4.0s | 21 | Full context, final refinement |

### Configuration Files

- `configs/phase10_revolab_all.json` - Full dataset with curriculum learning
- `configs/phase10_curriculum.json` - Curriculum learning configuration
- `configs/phase10_combined.json` - Combined Levantine datasets

Edit config files to adjust:
- Learning rate
- Batch size
- Segment lengths
- Curriculum schedule
- Loss weights (L1, STFT)

## Results

**Fine-tuned on 2.8M Levantine Arabic utterances:**

- Baseline val_loss: 0.3119
- **Best val_loss: 0.2212 (+29.06% improvement)**
- Final val_loss: 0.2214
- Training time: ~12 hours (single GPU)

**Checkpoints:**
- `checkpoints/phase10_revolab_all/best_model.pt` - Best validation model
- `checkpoints/phase10_revolab_all/checkpoint_epoch*.pt` - Epoch checkpoints
- `checkpoints/phase10_revolab_all/baseline_metrics.json` - Per-stage baselines

## Architecture

SNAC encodes audio into hierarchical discrete codes:

```
Audio Input → Encoder → Multi-scale Latent → VQ → Hierarchical Codes
                                              ↓
                                         Decoder → Audio Output
```

**Key Features:**
- Multi-scale tokenization with different temporal resolutions
- Coarse tokens sampled less frequently (~10 Hz for 24 kHz audio)
- Efficient compression at low bitrates (0.98 kbps for speech)
- Suitable for language modeling approaches to audio generation

## Loss Functions

The model is trained with combined reconstruction loss:

```
loss = l1_weight * L1_loss + stft_weight * multi_scale_STFT_loss
```

- **L1 loss**: Time-domain reconstruction error
- **Multi-scale STFT loss**: Frequency-domain reconstruction at multiple FFT sizes [1024, 2048, 4096]

## Pretrained Models

| Model | Bitrate | Sample Rate | Params | Use Case |
|-------|---------|-------------|--------|----------|
| [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz) | 0.98 kbps | 24 kHz | 19.8M | 🗣️ Speech |
| [hubertsiuzdak/snac_32khz](https://huggingface.co/hubertsiuzdak/snac_32khz) | 1.9 kbps | 32 kHz | 54.5M | 🎸 Music |
| [hubertsiuzdak/snac_44khz](https://huggingface.co/hubertsiuzdak/snac_44khz) | 2.6 kbps | 44 kHz | 54.5M | 🎸 Music |

## Requirements

See `requirements.txt` for full dependencies:

```bash
torch>=2.0.0
numpy
einops
huggingface_hub
tqdm
```

## Citation

If you use SNAC or this fine-tuning code, please cite:

```bibtex
@inproceedings{siuzdak2024snac,
  title={SNAC: Multi-Scale Neural Audio Codec},
  author={Siuzdak, Hubert and Gr{\"o}tschla, Florian and Lanzend{\"o}rfer, Luca A},
  booktitle={Audio Imagination: NeurIPS 2024 Workshop AI-Driven Speech, Music, and Sound Generation},
  year={2024}
}
```

## License

See LICENSE file for details.

## Acknowledgements

- Base SNAC implementation by [Hubert Siuzdak](https://github.com/hubertsiuzdak/snac)
- Module definitions adapted from [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec)

# Dataset Setup Guide

## Required Dataset Structure

The training script expects this structure:

```
data/
├── train/
│   ├── speaker_0001/
│   │   ├── audio_001.wav
│   │   ├── audio_002.wav
│   │   └── ...
│   ├── speaker_0002/
│   │   ├── audio_001.wav
│   │   └── ...
│   └── ...
└── val/
    ├── speaker_0001/
    │   └── audio_001.wav
    └── ...
```

## Options

### Option 1: LibriSpeech (Recommended)
**Free, multi-speaker English speech dataset**

```bash
# Download LibriSpeech (clean subset)
mkdir -p data/librispeech
cd data/librispeech

# Download train-clean-100 (~6GB)
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xvf train-clean-100.tar.gz

# Download dev-clean (~300MB)
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xvf dev-clean.tar.gz

# Create symbolic links for training script
cd ../..
ln -s librispeech/LibriSpeech/train-clean-100 data/train
ln -s librispeech/LibriSpeech/dev-clean data/val
```

**LibriSpeech structure**: Each speaker folder already exists!

### Option 2: VCTK
**Free, 109 speakers, 44kHz audio**

```bash
# Download VCTK
mkdir -p data/vctk
cd data/vctk

# Download from https://queens-dbalt.eecs.queensu.ca/VCTK/VCTK-Corpus-0.92.zip
unzip VCTK-Corpus-0.92.zip

cd ../..
ln -s vctk/VCTK-Corpus-0.92/wav48 data/train
# Use last speaker for val
mkdir -p data/val/p280
cp vctk/VCTK-Corpus-0.92/wav48/p280/* data/val/p280/
```

### Option 3: Custom Dataset
**Your own audio files**

```bash
# Organize your audio files
data/
├── train/
│   ├── speaker_001/
│   │   ├── *.wav files
│   ├── speaker_002/
│   │   └── *.wav files
└── val/
    └── (same structure)

# Or use this script to auto-organize:
# python organize_data.py --source /path/to/audio --output data/train
```

## Quick Start Commands

### Using LibriSpeech (Recommended)
```bash
cd /mnt/data/work/snac

# 1. Create data directories
mkdir -p data/train data/val

# 2. Download and prepare LibriSpeech
bash scripts/download_librispeech.sh

# 3. Verify dataset
uv run python -c "
from pathlib import Path
train = Path('data/train')
speakers = len([d for d in train.iterdir() if d.is_dir()])
files = len(list(train.glob('*/*.wav')))
print(f'Train: {speakers} speakers, {files} audio files')
"
```

### Using Your Own Data
```bash
# 1. Organize audio files by speaker
mkdir -p data/train/speaker_{001..999}
# Copy your wav files into speaker folders

# 2. Create validation split (20% of speakers)
python scripts/split_train_val.py
```

## Minimum Requirements

- **Speakers**: At least 10 different speakers
- **Audio per speaker**: At least 10 recordings
- **Duration**: 2-5 seconds per file (script segments longer files)
- **Format**: WAV files, mono, any sample rate (auto-resampled)
- **Total size**: Minimum 1GB, recommended 5GB+

## Verify Before Training

```bash
# Check dataset structure
uv run python -c "
from pathlib import Path

train = Path('data/train')
val = Path('data/val')

print('Train set:')
for speaker in sorted(train.iterdir())[:5]:
    if speaker.is_dir():
        files = list(speaker.glob('*.wav'))
        print(f'  {speaker.name}: {len(files)} files')

print(f'\nTotal train speakers: {len([d for d in train.iterdir() if d.is_dir()])}')
print(f'Total val speakers: {len([d for d in val.iterdir() if d.is_dir()])}')
"
```

## Data Augmentation (Optional)

The training script includes basic augmentation:
- Random gain (±10dB)
- Enabled by default in `SpeakerDataset`

## Sample Command for Quick Testing

Want to test with minimal data first?
```bash
# Create tiny test dataset
mkdir -p data/train/speaker_{001..003}
for i in {001..003}; do
  # Copy 10 wav files per speaker
  ls your_audio_source/*.wav | head -10 | xargs -I {} cp {} data/train/speaker_$i/
done

# Use val as subset of train
mkdir -p data/val/speaker_001
ls data/train/speaker_001/*.wav | head -5 | xargs -I {} cp {} data/val/speaker_001/

# Run quick test
uv run python train.py --config configs/phase1_reconstruction_only.json
```

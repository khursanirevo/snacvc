#!/bin/bash
# Split all audio files into 4-second segments

echo "=== Splitting Training Data ==="
.venv/bin/python split_audio.py \
    --input data/train \
    --output data/train_split \
    --segment-length 4.0 \
    --sampling-rate 24000

echo ""
echo "=== Splitting Validation Data ==="
.venv/bin/python split_audio.py \
    --input data/val \
    --output data/val_split \
    --segment-length 4.0 \
    --sampling-rate 24000

echo ""
echo "=== Done! ==="
echo "Update your config files to use:"
echo "  train_data: data/train_split"
echo "  val_data: data/val_split"

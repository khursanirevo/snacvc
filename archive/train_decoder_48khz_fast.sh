#!/bin/bash
#
# Phase 11: SNAC Decoder 48kHz - Complete Precompute + Train Workflow
# =====================================================================
#
# This script automates the full training pipeline:
# 1. Precompute 48kHz audio using SIDON (one-time, if not exists)
# 2. Precompute quantized codes using SNAC encoder+VQ (one-time, if not exists)
# 3. Train decoder with precomputed data (fast!)
#
# Directory Structure:
#   /mnt/data/combine/train/audio           (input: 24kHz audio)
#   /mnt/data/combine/train/audio_48khz     (output: 48kHz audio)
#   /mnt/data/codes_phase11/train          (output: quantized codes)
#
# Benefits:
#   - 3-5x faster training (no encoder/VQ/SIDON during training)
#   - Less memory usage
#   - Higher batch size possible
#   - Reproducible (fixed precomputed data)
#
# Usage:
#   ./train_decoder_48khz_fast.sh [GPU_ID]
#

set -e

GPU_ID=${1:-0}
CONFIG_FILE="configs/phase11_decoder_48khz.json"

# Data directories
INPUT_24KHZ_DIR="/mnt/data/combine"
OUTPUT_48KHZ_DIR="/mnt/data/combine_48khz"
CODES_DIR="/mnt/data/codes_phase11"

# Precompute script parameters
BATCH_SIZE=32
SEGMENT_LENGTH=4.0

echo "=========================================="
echo "Phase 11: 48kHz Decoder Training (Fast)"
echo "=========================================="
echo ""
echo "GPU: ${GPU_ID}"
echo "Config: ${CONFIG_FILE}"
echo ""

# Check if 48kHz audio is precomputed
TRAIN_48KHZ="${OUTPUT_48KHZ_DIR}/train/audio"
VAL_48KHZ="${OUTPUT_48KHZ_DIR}/valid/audio"

if [ ! -d "$TRAIN_48KHZ" ] || [ -z "$(ls -A $TRAIN_48KHZ 2>/dev/null)" ]; then
    echo "=========================================="
    echo "Step 1: Precomputing 48kHz Audio"
    echo "=========================================="
    echo ""
    echo "Input:  ${INPUT_24KHZ_DIR}/train/audio (24kHz)"
    echo "Output: ${TRAIN_48KHZ} (48kHz)"
    echo ""
    echo "This will take 1-2 hours depending on dataset size..."
    echo ""

    # Precompute train set
    echo "Processing training set..."
    uv run python precompute_48khz_audio.py \
        --input_dir "${INPUT_24KHZ_DIR}/train/audio" \
        --output_dir "${TRAIN_48KHZ}" \
        --batch_size $BATCH_SIZE \
        --device $GPU_ID

    echo ""
    echo "Processing validation set..."
    uv run python precompute_48khz_audio.py \
        --input_dir "${INPUT_24KHZ_DIR}/valid/audio" \
        --output_dir "${VAL_48KHZ}" \
        --batch_size $BATCH_SIZE \
        --device $GPU_ID

    echo ""
    echo "✅ 48kHz audio precomputation complete!"
    echo ""
else
    echo "✅ 48kHz audio already precomputed, skipping..."
    echo "  Found: $TRAIN_48KHZ"
    echo ""
fi

# Check if quantized codes are precomputed
TRAIN_CODES="${CODES_DIR}/train"
VAL_CODES="${CODES_DIR}/val"

if [ ! -d "$TRAIN_CODES" ] || [ -z "$(ls -A $TRAIN_CODES 2>/dev/null)" ]; then
    echo "=========================================="
    echo "Step 2: Precomputing Quantized Codes"
    echo "=========================================="
    echo ""
    echo "Input:  ${INPUT_24KHZ_DIR}/train/audio (24kHz)"
    echo "Output: ${TRAIN_CODES} (quantized codes)"
    echo ""
    echo "This will take 1-2 hours depending on dataset size..."
    echo ""

    # Precompute train set
    echo "Processing training set..."
    uv run python precompute_codes.py \
        --pretrained_model hubertsiuzdak/snac_24khz \
        --data_dir "${INPUT_24KHZ_DIR}/train/audio" \
        --output_dir "${TRAIN_CODES}" \
        --segment_length $SEGMENT_LENGTH \
        --batch_size $BATCH_SIZE \
        --device $GPU_ID

    echo ""
    echo "Processing validation set..."
    uv run python precompute_codes.py \
        --pretrained_model hubertsiuzdak/snac_24khz \
        --data_dir "${INPUT_24KHZ_DIR}/valid/audio" \
        --output_dir "${VAL_CODES}" \
        --segment_length $SEGMENT_LENGTH \
        --batch_size $BATCH_SIZE \
        --device $GPU_ID

    echo ""
    echo "✅ Quantized codes precomputation complete!"
    echo ""
else
    echo "✅ Quantized codes already precomputed, skipping..."
    echo "  Found: $TRAIN_CODES"
    echo ""
fi

# Start training with precomputed data
echo "=========================================="
echo "Step 3: Training with Precomputed Data"
echo "=========================================="
echo ""
echo "Using:"
echo "  - Precomputed codes: ${CODES_DIR}"
echo "  - Precomputed 48kHz audio: ${OUTPUT_48KHZ_DIR}"
echo "  - Config: ${CONFIG_FILE}"
echo ""
echo "Expected speedup: 3-5x faster than on-the-fly computation"
echo ""

LOG_FILE="/tmp/phase11_fast_gpu${GPU_ID}.log"
PID_FILE="/tmp/phase11_fast_gpu${GPU_ID}.pid"

echo "Starting training in background..."
echo "  Command: uv run python finetune_decoder_48khz_fast.py --config $CONFIG_FILE --codes_dir $CODES_DIR --audio_48khz_dir $OUTPUT_48KHZ_DIR --device $GPU_ID"
echo ""

nohup uv run python finetune_decoder_48khz_fast.py \
    --config "$CONFIG_FILE" \
    --codes_dir "$CODES_DIR" \
    --audio_48khz_dir "$OUTPUT_48KHZ_DIR" \
    --device "$GPU_ID" \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > "$PID_FILE"

sleep 5

if ps -p $TRAIN_PID > /dev/null; then
    echo "✅ Training started successfully!"
    echo ""
    echo "=========================================="
    echo "Monitoring Commands"
    echo "=========================================="
    echo ""
    echo "Monitor logs:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "Check if running:"
    echo "  ps -p $TRAIN_PID && echo '✓ Running' || echo '✗ Stopped'"
    echo ""
    echo "Stop training:"
    echo "  kill $TRAIN_PID"
    echo ""
    echo "=========================================="
    echo "Training Output"
    echo "=========================================="
    echo ""
    echo "Checkpoints: checkpoints/phase11_decoder_48khz/"
    echo "Logs: logs/phase11_decoder_48khz/training.log"
    echo ""
else
    echo "❌ Training failed to start!"
    echo "  Check: cat $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi

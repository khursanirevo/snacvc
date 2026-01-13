#!/bin/bash
#
# Phase 10: Decoder-Only Fine-tuning Script
# ==========================================
# This script runs the Phase 10 fine-tuning with ONLY the decoder trainable.
# Encoder and VQ (quantizer) are frozen.
#
# Usage:
#   ./train_decoder_only.sh [GPU_ID]
#
# Example:
#   ./train_decoder_only.sh 0    # Run on GPU 0
#   ./train_decoder_only.sh 3    # Run on GPU 3
#
# To monitor training:
#   tail -f logs/phase10_decoder_only/training.log
#
# To check if running:
#   ps aux | grep finetune.py
#

set -e  # Exit on error

# ============================================
# Configuration
# ============================================

# GPU device (default: 0, or pass as argument)
GPU_ID=${1:-0}

# Config file (already has freeze_encoder=true, freeze_vq=true)
CONFIG_FILE="configs/phase10_revolab_all.json"

# Log file
LOG_FILE="/tmp/phase10_decoder_only_gpu${GPU_ID}.log"
PID_FILE="/tmp/phase10_decoder_only_gpu${GPU_ID}.pid"

# Experiment name (for tracking)
EXPERIMENT_NAME="phase10_decoder_only"

# ============================================
# Pre-flight checks
# ============================================

echo "=========================================="
echo "Phase 10: Decoder-Only Fine-tuning"
echo "=========================================="
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if data paths exist
TRAIN_DATA=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['train_data'])")
VAL_DATA=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['val_data'])")

if [ ! -d "$TRAIN_DATA" ]; then
    echo "❌ Error: Training data not found: $TRAIN_DATA"
    exit 1
fi

if [ ! -d "$VAL_DATA" ]; then
    echo "❌ Error: Validation data not found: $VAL_DATA"
    exit 1
fi

# Check GPU availability
echo "Checking GPU ${GPU_ID}..."
if ! nvidia-smi --id=$GPU_ID --query-gpu=name --format=csv,noheader > /dev/null 2>&1; then
    echo "❌ Error: GPU ${GPU_ID} not found"
    echo "   Available GPUs:"
    nvidia-smi --query-gpu=index,name --format=csv,noheader
    exit 1
fi

# Check if another training is already running on this GPU
if ps aux | grep -v grep | grep "finetune.py.*--device ${GPU_ID}" > /dev/null; then
    echo "❌ Error: Training already running on GPU ${GPU_ID}"
    echo "   Run 'ps aux | grep finetune.py' to see running jobs"
    exit 1
fi

# ============================================
# Print experiment info
# ============================================

echo "✅ Pre-flight checks passed!"
echo ""
echo "Experiment Configuration:"
echo "  Config:      $CONFIG_FILE"
echo "  GPU:         ${GPU_ID}"
echo "  Train data:  $TRAIN_DATA"
echo "  Val data:    $VAL_DATA"
echo "  Log file:    $LOG_FILE"
echo ""
echo "Training Strategy:"
echo "  Encoder:     FROZEN ❄️"
echo "  VQ:          FROZEN ❄️"
echo "  Decoder:     TRAINABLE 🔥"
echo ""
echo "Curriculum Learning:"
echo "  Epochs 1-2:  1.0s segments, batch_size=96"
echo "  Epochs 3-4:  2.0s segments, batch_size=48"
echo "  Epochs 5-6:  3.0s segments, batch_size=29"
echo "  Epochs 7-10: 4.0s segments, batch_size=22"
echo ""

# ============================================
# Run training in background
# ============================================

echo "Starting training in background..."
echo "  Command: uv run python finetune.py --config $CONFIG_FILE --device $GPU_ID"
echo ""

# Run in background with nohup
nohup uv run python finetune.py \
    --config "$CONFIG_FILE" \
    --device "$GPU_ID" \
    > "$LOG_FILE" 2>&1 &

# Save PID
TRAIN_PID=$!
echo $TRAIN_PID > "$PID_FILE"

# Wait a moment and check if it started successfully
sleep 5

if ps -p $TRAIN_PID > /dev/null; then
    echo "✅ Training started successfully!"
    echo ""
    echo "=========================================="
    echo "Monitoring Commands"
    echo "=========================================="
    echo ""
    echo "Monitor training logs:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "Check if still running:"
    echo "  ps -p $TRAIN_PID && echo '✓ Running' || echo '✗ Stopped'"
    echo ""
    echo "Check GPU usage:"
    echo "  watch -n 1 nvidia-smi"
    echo ""
    echo "Stop training:"
    echo "  kill $TRAIN_PID"
    echo "  # or: kill \$(cat $PID_FILE)"
    echo ""
    echo "=========================================="
    echo "Training Output Location"
    echo "=========================================="
    echo ""
    echo "Checkpoints will be saved to:"
    OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['output_dir'])")
    echo "  $OUTPUT_DIR/"
    echo ""
    echo "Training logs will be saved to:"
    echo "  logs/${EXPERIMENT_NAME}/training.log"
    echo ""
    echo "=========================================="
else
    echo "❌ Training failed to start! Check log:"
    echo "  cat $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi

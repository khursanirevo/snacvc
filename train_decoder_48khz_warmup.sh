#!/bin/bash
#
# Phase 11: SNAC Decoder 48kHz with Smart Initialization + Warmup
# =============================================================
#
# Two-phase training strategy:
# 1. Smart Initialization: Copy existing 2x upsampler weights to new layer
# 2. Warmup Phase: Train only the new layer (epochs 1-3)
# 3. Main Phase: Train entire decoder (epochs 4-15)
#
# This approach:
# - Avoids random initialization
# - Gradual adaptation prevents instability
# - Better convergence than from scratch
#
# Usage:
#   ./train_decoder_48khz_warmup.sh [GPU_ID]
#

set -e

GPU_ID=${1:-0}
SCRIPT="finetune_decoder_48khz_warmup.py"
CONFIG_FILE="configs/phase11_decoder_48khz.json"
LOG_FILE="/tmp/phase11_warmup_gpu${GPU_ID}.log"
PID_FILE="/tmp/phase11_warmup_gpu${GPU_ID}.pid"

echo "=========================================="
echo "Phase 11: 48kHz Decoder with Warmup"
echo "=========================================="
echo ""

# Pre-flight checks
if [ ! -f "$SCRIPT" ]; then
    echo "❌ Error: Script not found: $SCRIPT"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config not found: $CONFIG_FILE"
    exit 1
fi

# Check GPU
echo "Checking GPU ${GPU_ID}..."
if ! nvidia-smi --id=$GPU_ID --query-gpu=name --format=csv,noheader > /dev/null 2>&1; then
    echo "❌ Error: GPU ${GPU_ID} not found"
    nvidia-smi --query-gpu=index,name --format=csv,noheader
    exit 1
fi

echo "✅ GPU ${GPU_ID} available"
echo ""

# Print training strategy
echo "Training Strategy:"
echo "  Input:      24kHz audio"
echo "  Encoder:    FROZEN ❄️ (pretrained)"
echo "  VQ:         FROZEN ❄️ (pretrained)"
echo "  Decoder:    SMART INITIALIZATION + 2-PHASE TRAINING"
echo ""
echo "Phase 1 (Warmup, epochs 1-3):"
echo "  - Copy existing 2x upsampler weights to new layer"
echo "  - Train only the new layer"
echo "  - Higher learning rate (5e-5)"
echo "  - Freeze all other decoder layers"
echo ""
echo "Phase 2 (Main, epochs 4-15):"
echo "  - Unfreeze entire decoder"
echo "  - Normal learning rate (1e-5)"
echo "  - Train all decoder parameters"
echo ""
echo "Expected results:"
echo "  - Better convergence than random init"
echo "  - Stable training from epoch 1"
echo "  - Final output: 48kHz audio"
echo ""

# Start training
echo "Starting training in background..."
echo "  Command: uv run python $SCRIPT --config $CONFIG_FILE --device $GPU_ID"
echo ""

nohup uv run python "$SCRIPT" \
    --config "$CONFIG_FILE" \
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

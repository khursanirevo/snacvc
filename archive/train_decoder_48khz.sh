#!/bin/bash
#
# Phase 11: SNAC Decoder 48kHz Output Training Script
# ====================================================
# Train SNAC decoder to output 48kHz audio while keeping encoder and VQ frozen.
#
# Architecture:
#   Input (24kHz) → Encoder (frozen) → VQ (frozen) → Decoder (trainable) → Output (48kHz)
#
# Target: SIDON upsampler's 48kHz output
#
# Usage:
#   ./train_decoder_48khz.sh [GPU_ID]
#
# Example:
#   ./train_decoder_48khz.sh 0    # Run on GPU 0
#   ./train_decoder_48khz.sh 3    # Run on GPU 3
#

set -e

# ============================================
# Configuration
# ============================================

GPU_ID=${1:-0}
CONFIG_FILE="configs/phase11_decoder_48khz.json"
LOG_FILE="/tmp/phase11_decoder_48khz_gpu${GPU_ID}.log"
PID_FILE="/tmp/phase11_decoder_48khz_gpu${GPU_ID}.pid"

# ============================================
# Pre-flight checks
# ============================================

echo "=========================================="
echo "Phase 11: SNAC Decoder 48kHz Training"
echo "=========================================="
echo ""

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check SIDON dependencies
echo "Checking dependencies..."
if ! python3 -c "import torchaudio, transformers, huggingface_hub" 2>/dev/null; then
    echo "❌ Error: Missing dependencies for SIDON upsampler"
    echo "   Install with: pip install torchaudio transformers huggingface_hub"
    exit 1
fi
echo "✓ SIDON dependencies OK"

# Check GPU
echo ""
echo "Checking GPU ${GPU_ID}..."
if ! nvidia-smi --id=$GPU_ID --query-gpu=name --format=csv,noheader > /dev/null 2>&1; then
    echo "❌ Error: GPU ${GPU_ID} not found"
    nvidia-smi --query-gpu=index,name --format=csv,noheader
    exit 1
fi

# Check for running training
if ps aux | grep -v grep | grep "finetune_decoder_48khz.py.*--device ${GPU_ID}" > /dev/null; then
    echo "❌ Error: Training already running on GPU ${GPU_ID}"
    echo "   Run 'ps aux | grep finetune_decoder_48khz' to check"
    exit 1
fi

# ============================================
# Print experiment info
# ============================================

echo "✅ Pre-flight checks passed!"
echo ""
echo "Experiment Configuration:"
echo "  Config:     $CONFIG_FILE"
echo "  GPU:        ${GPU_ID}"
echo "  Log file:   $LOG_FILE"
echo ""
echo "Training Strategy:"
echo "  Input:      24kHz audio"
echo "  Encoder:    FROZEN ❄️ (pretrained SNAC)"
echo "  VQ:         FROZEN ❄️ (pretrained SNAC)"
echo "  Decoder:    TRAINABLE 🔥 (new 48kHz decoder)"
echo "  Output:     48kHz audio"
echo "  Target:     SIDON upsampler (48kHz)"
echo ""
echo "Decoder Architecture:"
echo "  Original:   [8, 8, 4, 2] = 512x upsampling (24kHz → 24kHz)"
echo "  New:        [8, 8, 4, 2, 2] = 1024x upsampling (24kHz → 48kHz)"
echo ""

# ============================================
# Run training
# ============================================

echo "Starting training in background..."
echo "  Command: uv run python finetune_decoder_48khz.py --config $CONFIG_FILE --device $GPU_ID"
echo ""

nohup uv run python finetune_decoder_48khz.py \
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
    echo "Monitor training logs:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "Check if running:"
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
    echo "Training Output"
    echo "=========================================="
    echo ""
    echo "Checkpoints will be saved to:"
    echo "  checkpoints/phase11_decoder_48khz/"
    echo ""
    echo "Training logs:"
    echo "  logs/phase11_decoder_48khz/training.log"
    echo ""
else
    echo "❌ Training failed to start! Check log:"
    echo "  cat $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi

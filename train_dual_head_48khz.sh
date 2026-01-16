#!/bin/bash
# Phase 12: Dual-Head Decoder Training (24kHz + 48kHz)
# This script trains a dual-head decoder that outputs both 24kHz and 48kHz audio
# The 24kHz head is initialized from Phase 10 weights
# The 48kHz head is newly added with smart weight initialization

set -e

# Check if GPU ID is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <gpu_id> [additional_args...]"
    echo "Example: $0 0"
    echo "Example: $0 0 --limit 5000"
    exit 1
fi

GPU_ID=$1
shift  # Remove GPU ID from args, pass rest to training script

# Configuration
OUTPUT_DIR="checkpoints/phase12_dual_head"
LOG_FILE="/tmp/phase12_dual_head_gpu${GPU_ID}.log"
PID_FILE="/tmp/phase12_dual_head_gpu${GPU_ID}.pid"

# Phase 10 checkpoint for 24kHz head initialization
PHASE10_CKPT="checkpoints/phase10_revolab_all/best_model.pt"

# Check if Phase 10 checkpoint exists
if [ ! -f "$PHASE10_CKPT" ]; then
    echo "❌ Error: Phase 10 checkpoint not found: $PHASE10_CKPT"
    echo "Please run Phase 10 training first or update PHASE10_CKPT path"
    exit 1
fi

echo "======================================================================"
echo "PHASE 12: DUAL-HEAD DECODER TRAINING"
echo "======================================================================"
echo "GPU: $GPU_ID"
echo "Output: $OUTPUT_DIR"
echo "Phase 10 checkpoint: $PHASE10_CKPT"
echo "Log: $LOG_FILE"
echo "======================================================================"

# Check if training is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "⚠ Training already running on GPU $GPU_ID (PID: $OLD_PID)"
        echo "   Kill it first with: kill $OLD_PID"
        exit 1
    else
        echo "ℹ Stale PID file found, cleaning up..."
        rm -f "$PID_FILE"
    fi
fi

# Check if GPU is available
if ! nvidia-smi -i $GPU_ID > /dev/null 2>&1; then
    echo "❌ Error: GPU $GPU_ID not found"
    exit 1
fi

# Pre-flight checks
echo "Running pre-flight checks..."

# Check cached codes
if [ ! -d "/mnt/data/codes_phase11/train" ]; then
    echo "❌ Error: Cached codes not found at /mnt/data/codes_phase11/train"
    echo "   Run cache_codes_multigpu.py first"
    exit 1
fi

# Check 48kHz audio
if [ ! -d "/mnt/data/combine/train/audio_48khz" ]; then
    echo "❌ Error: 48kHz audio not found at /mnt/data/combine/train/audio_48khz"
    exit 1
fi

echo "✓ All checks passed"

# Kill any existing training on this GPU
pkill -f "finetune_dual_head_48khz_cached.py" || true
sleep 2

# Start training in background
echo ""
echo "Starting training..."

nohup uv run python finetune_dual_head_48khz_cached.py \
    --device $GPU_ID \
    --batch_size 96 \
    --epochs 15 \
    --warmup_epochs 1 \
    --lr 2e-4 \
    --main_lr 5e-5 \
    --lr_24k_final_conv 1e-5 \
    --segment_schedule "1.0,2.0,3.0,4.0" \
    --batch_multiplier "2.0,1.0,0.6,0.45" \
    --epoch_ranges "1-2,3-4,5-6,7-15" \
    --cache_dir /mnt/data/codes_phase11/train \
    --audio_48k_dir /mnt/data/combine/train/audio_48khz \
    --val_cache_dir /mnt/data/codes_phase11/val \
    --val_audio_48k_dir /mnt/data/combine/valid/audio_48khz \
    --output_dir "$OUTPUT_DIR" \
    --phase10_checkpoint "$PHASE10_CKPT" \
    --loss_weight_24k 0.3 \
    --loss_weight_48k 1.0 \
    $@ \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!

# Save PID
echo $TRAIN_PID > "$PID_FILE"

# Verify it started
sleep 3
if ps -p $TRAIN_PID > /dev/null; then
    echo "✓ Training started successfully (PID: $TRAIN_PID)"
    echo ""
    echo "Monitor logs:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "Check status:"
    echo "  ps -p $TRAIN_PID"
    echo ""
    echo "Stop training:"
    echo "  kill $TRAIN_PID"
    echo ""
else
    echo "❌ Training failed to start. Check log: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi

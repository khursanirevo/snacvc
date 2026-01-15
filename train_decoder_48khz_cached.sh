#!/bin/bash
# Phase 11: 48kHz Decoder Training with Code Caching
#
# This script uses finetune_decoder_48khz_cached.py which:
# 1. First run: Pre-computes all codes and 48kHz targets (slow, ~8-12 hours)
# 2. Subsequent runs: Uses cached codes (fast, ~2-3 hours per epoch)
#
# Usage:
#   ./train_decoder_48khz_cached.sh <GPU_ID>

set -e

GPU_ID=${1:-0}
CONFIG="configs/phase11_decoder_48khz.json"
CACHE_DIR="/mnt/data/codes_phase11_full"
LOG_FILE="/tmp/phase11_cached_training.log"
PID_FILE="/tmp/phase11_cached_gpu${GPU_ID}.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}Phase 11: 48kHz Decoder Training${NC}"
echo -e "${GREEN}  with Code Caching${NC}"
echo -e "${GREEN}======================================${NC}"

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo -e "${YELLOW}Training already running on GPU $GPU_ID (PID: $PID)${NC}"
        echo "Check logs: tail -f $LOG_FILE"
        exit 0
    else
        echo -e "${YELLOW}Removing stale PID file${NC}"
        rm "$PID_FILE"
    fi
fi

# Check cache exists
TRAIN_CACHE="$CACHE_DIR/train/metadata.json"
if [ -f "$TRAIN_CACHE" ]; then
    echo -e "${GREEN}✓ Code cache found at $CACHE_DIR${NC}"
    echo "  Using cached codes (fast training!)"
else
    echo -e "${YELLOW}✗ No cache found at $CACHE_DIR${NC}"
    echo "  First run will pre-compute codes (this takes 8-12 hours)"
    echo ""
    read -p "Continue with pre-computation? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled"
        exit 1
    fi
fi

# Set HF cache
export HF_HOME=/mnt/data/work/snac/.hf_cache
export HF_HUB_CACHE=/mnt/data/work/snac/.hf_cache/hub

# Check GPU status
echo ""
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv | grep "^$GPU_ID," || {
    echo -e "${RED}Error: GPU $GPU_ID not found${NC}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
    exit 1
}

# Start training
echo ""
echo -e "${GREEN}Starting training on GPU $GPU_ID...${NC}"
echo "Config: $CONFIG"
echo "Logs: $LOG_FILE"
echo "PID: $PID_FILE"
echo ""

nohup uv run python finetune_decoder_48khz_cached.py \
    --config "$CONFIG" \
    --cache_dir "$CACHE_DIR" \
    --device "$GPU_ID" \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > "$PID_FILE"

# Verify it started
sleep 5
if ps -p $TRAIN_PID > /dev/null; then
    echo -e "${GREEN}✓ Training started successfully!${NC}"
    echo ""
    echo "Monitor training:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "Check status:"
    echo "  ./train_status.sh"
    echo ""
    echo "Stop training:"
    echo "  kill $TRAIN_PID"
    echo "  # or: pkill -f finetune_decoder_48khz_cached"
else
    echo -e "${RED}✗ Training failed to start${NC}"
    echo "Check logs: cat $LOG_FILE"
    rm "$PID_FILE"
    exit 1
fi

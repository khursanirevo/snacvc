#!/bin/bash

# Learning Rate Scaling Ablation Study for Phase 9 Curriculum Learning
# Tests hypothesis: larger batch sizes need proportionally larger learning rates
#
# Experiments:
# 1. 1x batch with 1x LR (baseline): 5e-6
# 2. 2x batch with 2x LR (linear): 10e-6
# 3. 3x batch with 3x LR (linear): 15e-6
# 4. 2x batch with sqrt(2) LR (sqrt scaling): 7.07e-6
# 5. 3x batch with sqrt(3) LR (sqrt scaling): 8.66e-6

set -e  # Exit on error

# Create log directories
mkdir -p logs/phase9_ablation

echo "========================================="
echo "Phase 9: Learning Rate Scaling Ablation"
echo "========================================="
echo ""

# Experiment 1: 1x batch with 1x LR (baseline)
echo "[1/5] Starting: 1x batch × 1x LR (5e-6)"
echo "Config: configs/phase9_curriculum_1x_lr1x.json"
nohup uv run python train_phase9_finetune.py \
    --config configs/phase9_curriculum_1x_lr1x.json \
    --device 0 \
    > logs/phase9_ablation/1x_batch_1x_lr.log 2>&1 &
PID1=$!
echo "Started on GPU 0, PID: $PID1"
echo ""

# Experiment 2: 2x batch with 2x LR (linear scaling)
echo "[2/5] Starting: 2x batch × 2x LR (10e-6)"
echo "Config: configs/phase9_curriculum_2x_lr2x.json"
nohup uv run python train_phase9_finetune.py \
    --config configs/phase9_curriculum_2x_lr2x.json \
    --device 1 \
    > logs/phase9_ablation/2x_batch_2x_lr.log 2>&1 &
PID2=$!
echo "Started on GPU 1, PID: $PID2"
echo ""

# Experiment 3: 3x batch with 3x LR (linear scaling)
echo "[3/5] Starting: 3x batch × 3x LR (15e-6)"
echo "Config: configs/phase9_curriculum_3x_lr3x.json"
nohup uv run python train_phase9_finetune.py \
    --config configs/phase9_curriculum_3x_lr3x.json \
    --device 2 \
    > logs/phase9_ablation/3x_batch_3x_lr.log 2>&1 &
PID3=$!
echo "Started on GPU 2, PID: $PID3"
echo ""

# Experiment 4: 2x batch with sqrt(2) LR (sqrt scaling)
echo "[4/5] Starting: 2x batch × sqrt(2) LR (7.07e-6)"
echo "Config: configs/phase9_curriculum_2x_lr_sqrt.json"
nohup uv run python train_phase9_finetune.py \
    --config configs/phase9_curriculum_2x_lr_sqrt.json \
    --device 3 \
    > logs/phase9_ablation/2x_batch_lr_sqrt.log 2>&1 &
PID4=$!
echo "Started on GPU 3, PID: $PID4"
echo ""

# Experiment 5: 3x batch with sqrt(3) LR (sqrt scaling)
echo "[5/5] Starting: 3x batch × sqrt(3) LR (8.66e-6)"
echo "Config: configs/phase9_curriculum_3x_lr_sqrt.json"
nohup uv run python train_phase9_finetune.py \
    --config configs/phase9_curriculum_3x_lr_sqrt.json \
    --device 4 \
    > logs/phase9_ablation/3x_batch_lr_sqrt.log 2>&1 &
PID5=$!
echo "Started on GPU 4, PID: $PID5"
echo ""

# Summary
echo "========================================="
echo "All experiments started!"
echo "========================================="
echo ""
echo "Running experiments:"
echo "  GPU 0 (PID $PID1): 1x batch × 1x LR"
echo "  GPU 1 (PID $PID2): 2x batch × 2x LR (linear)"
echo "  GPU 2 (PID $PID3): 3x batch × 3x LR (linear)"
echo "  GPU 3 (PID $PID4): 2x batch × sqrt(2) LR (sqrt)"
echo "  GPU 4 (PID $PID5): 3x batch × sqrt(3) LR (sqrt)"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/phase9_ablation/1x_batch_1x_lr.log"
echo "  tail -f logs/phase9_ablation/2x_batch_2x_lr.log"
echo "  tail -f logs/phase9_ablation/3x_batch_3x_lr.log"
echo "  tail -f logs/phase9_ablation/2x_batch_lr_sqrt.log"
echo "  tail -f logs/phase9_ablation/3x_batch_lr_sqrt.log"
echo ""
echo "Check GPU usage:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "Kill all experiments:"
echo "  kill $PID1 $PID2 $PID3 $PID4 $PID5"

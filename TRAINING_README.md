# Training Scripts for Phase 10

Quick scripts for reproducible decoder-only fine-tuning.

## Scripts

### `train_decoder_only.sh` - Start Training

Train only the decoder (encoder and VQ frozen):

```bash
# Run on GPU 0 (default)
./train_decoder_only.sh

# Run on specific GPU
./train_decoder_only.sh 3
```

**What it does:**
- Pre-flight checks (data paths, GPU availability, etc.)
- Runs training in background with nohup
- Saves PID and log file
- Shows monitoring commands

**Output:**
- Checkpoints: `checkpoints/phase10_revolab_all/`
- Logs: `logs/phase10_decoder_only/training.log`
- Background log: `/tmp/phase10_decoder_only_gpu<N>.log`

**Configuration:**
Uses `configs/phase10_revolab_all.json` which has:
- `freeze_encoder: true` - Encoder frozen
- `freeze_vq: true` - VQ frozen
- Only decoder is trainable (~15M params)

### `train_status.sh` - Check Status

Check running training jobs:

```bash
./train_status.sh
```

**Shows:**
- Running training processes
- PID files and status
- Log file info
- GPU status
- Latest log tail

## Manual Commands

### Start training manually:
```bash
# Background with logging
nohup uv run python finetune.py \
    --config configs/phase10_revolab_all.json \
    --device 0 \
    > /tmp/training.log 2>&1 &

# Save PID
echo $! > /tmp/training.pid
```

### Monitor training:
```bash
# Live logs
tail -f logs/phase10_decoder_only/training.log

# Or background log
tail -f /tmp/phase10_decoder_only_gpu0.log

# Check GPU usage
watch -n 1 nvidia-smi

# Check if running
ps aux | grep finetune.py
```

### Stop training:
```bash
# Using PID
kill $(cat /tmp/phase10_decoder_only_gpu0.pid)

# Or find process
ps aux | grep finetune.py
kill <PID>

# Or kill all finetune processes
pkill -f finetune.py
```

## Resume from Checkpoint

To resume training from a checkpoint:

```bash
uv run python finetune.py \
    --config configs/phase10_revolab_all.json \
    --device 0 \
    --resume checkpoints/phase10_revolab_all/checkpoint_epoch5.pt
```

## Curriculum Learning

Training uses 4 stages with progressively longer segments:

| Epochs | Segment | Batch Size | Description |
|--------|---------|------------|-------------|
| 1-2 | 1.0s | 96 | Fast iterations, foundation |
| 3-4 | 2.0s | 48 | Medium context |
| 5-6 | 3.0s | 29 | Longer context |
| 7-10 | 4.0s | 22 | Full context, refinement |

Total training time: ~12 hours (single GPU, H200)

## Expected Results

From previous run:
- Baseline val_loss: 0.3119
- Best val_loss: 0.2212
- Improvement: **+29.06%**

## Troubleshooting

### Training failed to start
```bash
# Check log
cat /tmp/phase10_decoder_only_gpu0.log

# Check GPU availability
nvidia-smi

# Check data paths
ls /mnt/data/combine/train/audio | head -5
ls /mnt/data/combine/valid/audio | head -5
```

### GPU out of memory
Reduce batch size in config:
```json
{
  "batch_size": 32,  // was 48
  "eval_batch_size": 48  // was 64
}
```

### Training stopped unexpectedly
Check for errors in log:
```bash
tail -100 /tmp/phase10_decoder_only_gpu0.log | grep -i error
```

### Resume from last checkpoint
```bash
uv run python finetune.py \
    --config configs/phase10_revolab_all.json \
    --device 0 \
    --resume checkpoints/phase10_revolab_all/best_model.pt
```

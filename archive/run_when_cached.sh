#!/bin/bash
# Monitor caching and auto-start training when done

CACHE_DIR="/mnt/data/codes_phase11_small/train"
TARGET_FILES=5  # Actual files cached (4478 valid files / 1000 per batch)
LOG="/tmp/phase11_small_training.log"

echo "Monitoring caching..."
while true; do
    parquet_count=$(ls $CACHE_DIR/*.parquet 2>/dev/null | wc -l)
    echo "$(date): Found $parquet_count parquet files"

    if [ $parquet_count -ge $TARGET_FILES ]; then
        echo "✓ Caching complete! Starting training..."

        export HF_HOME=/mnt/data/work/snac/.hf_cache
        export HF_HUB_CACHE=/mnt/data/work/snac/.hf_cache/hub

        nohup uv run python finetune_decoder_48khz_cached.py \
            --config configs/phase11_decoder_48khz.json \
            --cache_dir /mnt/data/codes_phase11_small \
            --device 0 > $LOG 2>&1 &

        TRAIN_PID=$!
        echo "✓ Training started with PID: $TRAIN_PID"
        echo $TRAIN_PID > /tmp/phase11_small_training.pid

        # Wait and check if training is working
        sleep 30
        if ps -p $TRAIN_PID > /dev/null; then
            echo "✓ Training is running!"
            echo "Monitor: tail -f $LOG"
            break
        else
            echo "✗ Training failed to start"
            cat $LOG
            break
        fi
    fi

    sleep 30
done

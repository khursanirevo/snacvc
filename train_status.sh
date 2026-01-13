#!/bin/bash
#
# Training Status Helper Script
# =============================
# Helper script to check status of running training jobs

echo "=========================================="
echo "Phase 10 Training Status"
echo "=========================================="
echo ""

# Check for running finetune processes
RUNNING=$(ps aux | grep -v grep | grep "finetune.py" || true)

if [ -z "$RUNNING" ]; then
    echo "❌ No training processes running"
    echo ""
else
    echo "✅ Training processes found:"
    echo ""
    echo "$RUNNING"
    echo ""
fi

# Check for PID files
echo "PID Files:"
for pidfile in /tmp/phase10_*.pid; do
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if ps -p $pid > /dev/null 2>&1; then
            echo "  ✓ $pidfile: PID $pid (running)"
        else
            echo "  ✗ $pidfile: PID $pid (NOT running)"
        fi
    fi
done
echo ""

# Check log files
echo "Log Files:"
for logfile in /tmp/phase10_*.log; do
    if [ -f "$logfile" ]; then
        size=$(du -h "$logfile" | cut -f1)
        mtime=$(stat -c %y "$logfile" | cut -d'.' -f1)
        echo "  📝 $logfile"
        echo "     Size: $size, Last modified: $mtime"
    fi
done
echo ""

# GPU status
echo "=========================================="
echo "GPU Status"
echo "=========================================="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
echo ""

# Latest log tail
if [ -n "$RUNNING" ]; then
    echo "=========================================="
    echo "Latest Training Log"
    echo "=========================================="

    # Find the most recently modified log file
    latest_log=$(ls -t /tmp/phase10_*.log 2>/dev/null | head -1)

    if [ -n "$latest_log" ]; then
        echo "Showing last 20 lines from: $latest_log"
        echo ""
        tail -20 "$latest_log"
    fi
fi

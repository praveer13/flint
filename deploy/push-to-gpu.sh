#!/bin/bash
# Push flint binary + Python files to a GPU machine and run end-to-end test.
#
# Usage:
#   ./deploy/push-to-gpu.sh <ssh-host> [model]
#
# Example:
#   ./deploy/push-to-gpu.sh root@209.38.x.x
#   ./deploy/push-to-gpu.sh root@209.38.x.x meta-llama/Llama-3.1-8B-Instruct
#
# Prerequisites:
#   - RunPod instance running with SSH access
#   - flint binary built: zig build -Doptimize=ReleaseFast

set -euo pipefail

HOST="${1:?Usage: $0 <ssh-host> [model]}"
MODEL="${2:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
REMOTE_DIR="/root/flint"
BINARY="zig-out/bin/flint"

if [ ! -f "$BINARY" ]; then
    echo "Building flint..."
    zig build -Doptimize=ReleaseFast
fi

echo "==> Pushing to $HOST..."

# Create remote directory and push files
ssh "$HOST" "mkdir -p $REMOTE_DIR/python"
scp "$BINARY" "$HOST:$REMOTE_DIR/flint"
scp -r python/flint_shm python/flint_worker.py python/mock_worker.py python/requirements.txt \
    "$HOST:$REMOTE_DIR/python/"

# Push the remote setup + test script
scp deploy/gpu-setup.sh deploy/gpu-test.sh "$HOST:$REMOTE_DIR/"

echo "==> Setting up Python environment on remote..."
ssh "$HOST" "cd $REMOTE_DIR && bash gpu-setup.sh"

echo "==> Running end-to-end test with model: $MODEL..."
ssh "$HOST" "cd $REMOTE_DIR && bash gpu-test.sh '$MODEL'"

echo "==> Done. To SSH in manually: ssh $HOST"
echo "    cd $REMOTE_DIR && ./flint 8080"

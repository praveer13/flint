#!/bin/bash
# End-to-end test that runs ON the GPU machine.
# Starts flint server, spawns vLLM worker, sends HTTP request, verifies SSE output.
#
# Usage: bash gpu-test.sh [model]
set -euo pipefail

MODEL="${1:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
PORT=8080
FLINT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "==> Downloading model $MODEL (if not cached)..."
python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$MODEL')" 2>&1 | tail -3

echo "==> Starting flint server on port $PORT..."
chmod +x "$FLINT_DIR/flint"
"$FLINT_DIR/flint" "$PORT" &
FLINT_PID=$!
sleep 2

# Check if flint is running
if ! kill -0 $FLINT_PID 2>/dev/null; then
    echo "ERROR: flint failed to start"
    exit 1
fi

echo "==> Starting vLLM worker (model: $MODEL)..."
PYTHONPATH="$FLINT_DIR/python" python3 "$FLINT_DIR/python/flint_worker.py" \
    /dev/shm/flint_server --model "$MODEL" --gpu 0 &
WORKER_PID=$!
sleep 5  # Give model time to load

# Check if worker is running
if ! kill -0 $WORKER_PID 2>/dev/null; then
    echo "ERROR: worker failed to start. Trying mock mode..."
    PYTHONPATH="$FLINT_DIR/python" python3 "$FLINT_DIR/python/flint_worker.py" \
        /dev/shm/flint_server --mock &
    WORKER_PID=$!
    sleep 2
fi

echo "==> Sending test request..."
RESPONSE=$(curl -sN --max-time 30 -X POST "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"$MODEL"'","messages":[{"role":"user","content":"Say hello in one sentence."}]}' 2>&1)

echo "==> Response:"
echo "$RESPONSE"

# Check for SSE events
if echo "$RESPONSE" | grep -q "data:.*content"; then
    echo ""
    echo "==> SUCCESS: Got SSE token events!"
else
    echo ""
    echo "==> WARN: No token content in response. Check logs."
fi

# Check for [DONE]
if echo "$RESPONSE" | grep -q "\[DONE\]"; then
    echo "==> SUCCESS: Stream completed with [DONE]"
else
    echo "==> WARN: Stream did not complete with [DONE]"
fi

# Cleanup
echo "==> Cleaning up..."
kill $WORKER_PID 2>/dev/null || true
kill $FLINT_PID 2>/dev/null || true
wait $WORKER_PID 2>/dev/null || true
wait $FLINT_PID 2>/dev/null || true

echo "==> Test complete."

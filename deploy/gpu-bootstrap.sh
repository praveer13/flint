#!/usr/bin/env bash
# Run this ON the RunPod GPU pod after cloning the repo.
# Does everything: installs deps, builds (if needed), starts server + worker, tests.
#
# Usage (on the pod):
#   git clone https://github.com/praveer13/flint.git && cd flint && bash deploy/gpu-bootstrap.sh
set -e

echo "==> GPU check..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "==> Installing uv (fast Python package manager)..."
curl -LsSf https://astral.sh/uv/install.sh | sh 2>&1 | tail -3
export PATH="$HOME/.local/bin:$PATH"

echo "==> Installing Python deps via uv (keeping existing CUDA torch)..."
uv pip install --system numpy 'vllm>=0.6.0,<0.7.0' --no-deps 2>&1 | tail -5
# Install vllm's deps except torch (already has CUDA version from RunPod image)
uv pip install --system msgspec cloudpickle sentencepiece protobuf py-cpuinfo 'transformers>=4.36.0' 'tokenizers>=0.15.0' 'numpy>=1.24.0' partial-json openai tiktoken lm-format-enforcer outlines typing_extensions pillow prometheus-client 'fastapi' 'uvicorn[standard]' pydantic aiohttp 2>&1 | tail -5

echo "==> Verifying..."
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')"

# Use pre-built binary if available, otherwise note it's needed
if [ -f zig-out/bin/flint ]; then
    echo "==> Using pre-built flint binary"
    chmod +x zig-out/bin/flint
    FLINT=zig-out/bin/flint
else
    echo "ERROR: No pre-built binary found at zig-out/bin/flint"
    echo "Build locally and push: zig build -Doptimize=ReleaseFast && git add zig-out && git push"
    exit 1
fi

echo "==> Starting flint server on port 8080..."
$FLINT 8080 &>/tmp/flint.log &
FLINT_PID=$!
sleep 3

if ! kill -0 $FLINT_PID 2>/dev/null; then
    echo "ERROR: flint failed to start"
    cat /tmp/flint.log
    exit 1
fi
echo "  flint running (pid $FLINT_PID)"

echo "==> Test 1: Mock worker (quick sanity check)..."
PYTHONPATH=python python3 python/mock_worker.py /dev/shm/flint_server &>/tmp/mock.log &
MOCK_PID=$!
sleep 3

echo "  Sending request..."
curl -sN --max-time 10 -X POST http://localhost:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"test","messages":[{"role":"user","content":"hello"}]}'
echo ""

echo "  Stopping mock worker..."
kill $MOCK_PID 2>/dev/null; wait $MOCK_PID 2>/dev/null || true
sleep 2

echo ""
echo "==> Test 2: Real vLLM worker (TinyLlama-1.1B)..."
PYTHONPATH=python python3 python/flint_worker.py /dev/shm/flint_server \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --gpu 0 &>/tmp/vllm.log &
VLLM_PID=$!
echo "  Loading model (this takes 30-60s)..."
sleep 50

if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "  Worker died. Logs:"
    tail -20 /tmp/vllm.log
    echo ""
    echo "  Falling back to --mock mode..."
    PYTHONPATH=python python3 python/flint_worker.py /dev/shm/flint_server --mock &>/tmp/vllm.log &
    VLLM_PID=$!
    sleep 3
fi

echo "  Sending request..."
curl -sN --max-time 30 -X POST http://localhost:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"TinyLlama","messages":[{"role":"user","content":"Say hello in one sentence."}]}'
echo ""

echo ""
echo "==> Logs:"
echo "--- flint ---"
tail -5 /tmp/flint.log
echo "--- vllm worker ---"
tail -10 /tmp/vllm.log
echo ""
echo "==> Cleanup: kill $FLINT_PID $VLLM_PID"
echo "==> Or keep running and test manually:"
echo "    curl -sN -X POST http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}'"

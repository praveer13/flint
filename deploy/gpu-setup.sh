#!/bin/bash
# Setup script that runs ON the GPU machine.
# Installs Python deps (vLLM, numpy) into a venv.
set -euo pipefail

echo "==> Installing Python dependencies..."

# Use system python (RunPod images have python3 + pip)
pip install --quiet numpy vllm==0.6.6.post1 2>&1 | tail -5

echo "==> Verifying GPU..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo "==> Verifying vLLM..."
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')"

echo "==> Setup complete."

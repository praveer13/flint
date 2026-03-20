#!/usr/bin/env bash
# Deploy flint to RunPod GPU pod.
#
# Usage:
#   1. Build locally: zig build -Doptimize=ReleaseFast
#   2. Push to GitHub: git push
#   3. SSH into RunPod: ssh -i ~/.ssh/id_ed25519 POD_HOST
#   4. On the pod, paste: curl -sL https://raw.githubusercontent.com/praveer13/flint/main/deploy/gpu-bootstrap.sh | bash
#
# Or just paste this into the RunPod SSH session:
#
#   git clone https://github.com/praveer13/flint.git /root/flint && bash /root/flint/deploy/gpu-bootstrap.sh

echo "This script is documentation. See the commands above."
echo ""
echo "Quick steps:"
echo "  1. zig build -Doptimize=ReleaseFast && git add -A && git push"
echo "  2. ssh -i ~/.ssh/id_ed25519 wt6sbb714nyn2s-644118fd@ssh.runpod.io"
echo "  3. Paste: git clone https://github.com/praveer13/flint.git && cd flint && bash deploy/gpu-bootstrap.sh"

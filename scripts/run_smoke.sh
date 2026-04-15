#!/bin/bash
# Quick smoke test — runs 5 steps on CPU with tiny config
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

echo "=== Hivemind Smoke Test ==="
echo "Running 5 training steps on CPU..."
python train.py --config-name pilot_smoke

echo ""
echo "=== Running unit tests ==="
python -m pytest tests/ -v --tb=short

echo ""
echo "=== Smoke test passed ==="

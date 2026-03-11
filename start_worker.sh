#!/bin/bash
set -e

echo "Starting TransNetV2 Worker (Mac MPS GPU)..."

cd "$(dirname "$0")"

source .venv/bin/activate

export USE_GPU=true

exec python main.py

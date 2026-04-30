#!/bin/bash
# Simple launch script (skips build)

set -e

# Configuration
DATASET="/home/mohithr/pw/storage/archaia_dataset_hf_v3"
IMAGES_ROOT="/home/urmid/archaia/"
PORT=7881
SHARE="--share"

cd "$(dirname "$0")/.."

# Fix for PermissionError in shared /tmp/gradio
export GRADIO_TEMP_DIR="$(pwd)/.gradio_tmp"
mkdir -p "$GRADIO_TEMP_DIR"

export PYTHONPATH=$PYTHONPATH:.
export HF_HOME=$(pwd)/.hf_cache
mkdir -p "$HF_HOME"

echo "--- ArchAIaGPT: Launch Mode ---"

python app.py \
    --port ${PORT} \
    ${SHARE}

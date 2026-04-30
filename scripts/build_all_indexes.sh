#!/bin/bash
# Builds FAISS indexes for all supported models.

set -e

# Configuration
DATASET="/home/mohithr/pw/storage/archaia_dataset_hf_v3"
IMAGES_ROOT="/home/urmid/archaia/"
INDEX_DIR="indexes"
## Which models to build. Edit this list or override by exporting MODELS (space-separated)
## Example: export MODELS=("clip" "gemma") && ./scripts/build_all_indexes.sh
MODELS=("clip" "gemma" "qwen3-vl")

declare -A MODEL_IDS=(
    [clip]="openai/clip-vit-base-patch32"
    [gemma]="google/embeddinggemma-300m"
    [qwen3-vl]="Qwen/Qwen3-VL-Embedding-2B"
)
declare -A BATCH_SIZES=(
    [clip]=32
    [gemma]=32
    [qwen3-vl]=4
)
DEVICE="cuda"

cd "$(dirname "$0")/.."

echo "--- ArchAIaGPT: Building ALL Indexes ---"

export PYTHONPATH=$PYTHONPATH:.
export HF_HOME=$(pwd)/.hf_cache
mkdir -p "$HF_HOME"

for MODEL in "${MODELS[@]}"; do
    echo "Encoding model: ${MODEL} ..."
    python embeddings/build_index.py \
        --dataset     "${DATASET}" \
        --out_dir     "${INDEX_DIR}" \
        --images_root "${IMAGES_ROOT}" \
        --model_type  "${MODEL}" \
        --model_id    "${MODEL_IDS[$MODEL]}" \
        --batch_size  "${BATCH_SIZES[$MODEL]}" \
        --device      "${DEVICE}"
done

echo "Done! All indexes saved to ${INDEX_DIR}."

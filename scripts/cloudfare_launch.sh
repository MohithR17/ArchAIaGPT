#!/bin/bash
# Simple launch script (skips build)

set -e

source ~/pw/storage/archatagpt/bin/activate

DATASET="/home/mohithr/pw/storage/archaia_dataset_hf_v3"
IMAGES_ROOT="/home/urmid/archaia/"
GEN_BACKEND="openai"
PORT=7860
CLOUDFLARED="$(dirname "$0")/../cloudflared-linux-amd64"

cd "$(dirname "$0")/.."

echo "--- ArchAIaGPT: Launch Mode ---"

# Start Cloudflare tunnel in background
echo "Starting Cloudflare tunnel..."
"$CLOUDFLARED" tunnel --url http://localhost:${PORT} \
    --no-autoupdate \
    2>&1 | grep -E "trycloudflare.com|ERR" &
TUNNEL_PID=$!

# Give tunnel a moment to establish
sleep 3

echo ""
echo "================================================"
echo " App launching... Cloudflare URL printed above  "
echo " Share the trycloudflare.com link with others   "
echo "================================================"
echo ""

# Launch app WITHOUT --share (cloudflare replaces it)
python app.py \
    --dataset "${DATASET}" \
    --port ${PORT} \
    --gen_backend "${GEN_BACKEND}" \
    --images_root "${IMAGES_ROOT}"

# Cleanup tunnel when app exits
kill $TUNNEL_PID 2>/dev/null
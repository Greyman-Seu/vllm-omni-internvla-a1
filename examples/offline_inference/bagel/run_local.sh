#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$ROOT_DIR/scripts/bagel_common.sh"

MODEL="$(bagel_effective_model)"
MODALITY="${MODALITY:-text2text}"
PROMPT="${PROMPT:-What is the capital of France?}"

echo "Running Bagel offline inference"
echo "Model: $MODEL"
echo "Modality: $MODALITY"
echo "HF endpoint: $HF_ENDPOINT"
echo "HF cache: $HF_HOME"

if ! bagel_check_cuda; then
  echo "CUDA is not available in the bagel environment. Exiting before offline inference." >&2
  exit 1
fi

export HF_ENDPOINT
export HF_HOME

if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx 'bagel'; then
  conda run -n bagel python "$ROOT_DIR/examples/offline_inference/bagel/end2end.py" \
    --model "$MODEL" \
    --modality "$MODALITY" \
    --prompts "$PROMPT"
else
  python "$ROOT_DIR/examples/offline_inference/bagel/end2end.py" \
    --model "$MODEL" \
    --modality "$MODALITY" \
    --prompts "$PROMPT"
fi

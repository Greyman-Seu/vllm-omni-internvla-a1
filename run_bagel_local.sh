#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"
MODEL_DIR="${MODEL:-${HOME}/models/BAGEL-7B-MoT}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export MODALITY="${MODALITY:-text2text}"
export PROMPT="${PROMPT:-What is the capital of France?}"
export MODEL="${MODEL_DIR}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Missing conda init script: ${CONDA_SH}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Model config not found: ${MODEL_DIR}/config.json" >&2
  exit 1
fi

if [[ ! -f "${MODEL_DIR}/ae.safetensors" ]]; then
  echo "Missing model weight: ${MODEL_DIR}/ae.safetensors" >&2
  exit 1
fi

if [[ ! -f "${MODEL_DIR}/ema.safetensors" ]]; then
  echo "Missing model weight: ${MODEL_DIR}/ema.safetensors" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate bagel

python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" || {
  echo "CUDA is not available in the bagel environment." >&2
  exit 1
}

cd "${ROOT_DIR}"

echo "Running Bagel local inference"
echo "Model: ${MODEL}"
echo "Modality: ${MODALITY}"
echo "Prompt: ${PROMPT}"
echo "HF endpoint: ${HF_ENDPOINT}"
echo "HF cache: ${HF_HOME}"
echo "CUDA available. Starting model initialization..."

exec python -u "${ROOT_DIR}/examples/offline_inference/bagel/end2end.py" \
  --model "${MODEL}" \
  --modality "${MODALITY}" \
  --prompts "${PROMPT}"

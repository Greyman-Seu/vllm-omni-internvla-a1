#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWENA1_PROCESSOR_DIR="${QWENA1_PROCESSOR_DIR:-/mnt/shared-storage-user/internvla/shared_model_weights/Qwen3/Qwen3-VL-2B-Instruct}"
QWENA1_COSMOS_DIR="${QWENA1_COSMOS_DIR:-/home/zhuyangkun/models/Cosmos-Tokenizer-CI8x8}"

export QWENA1_PROCESSOR_DIR
export QWENA1_COSMOS_DIR

python "$ROOT_DIR/run_qwena1_vllm_infer.py" "$@"

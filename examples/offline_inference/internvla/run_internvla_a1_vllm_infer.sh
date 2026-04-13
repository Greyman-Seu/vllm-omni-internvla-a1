#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../../.." && pwd)"
# hf download Jia-Zeng/InternVLA-A1-3B-FineTuned-Place_Markpen
INTERNVLA_A1_MODEL_DIR="${INTERNVLA_A1_MODEL_DIR:-/home/zhuyangkun/data/vllm_a1/new_data/InternVLA-A1-3B-ft-pen}"
# hf download InternRobotics/InternData-A1 real_lerobotv30/genie1/Genie1-Place_Markpen.tar.gz --repo-type dataset --local-dir temp_data_dir
INTERNVLA_A1_DATASET_DIR="${INTERNVLA_A1_DATASET_DIR:-/home/zhuyangkun/data/vllm_a1/new_data/Genie1-Place_Markpen}"
INTERNVLA_A1_PROCESSOR_DIR="${INTERNVLA_A1_PROCESSOR_DIR:-/mnt/shared-storage-user/internvla/shared_model_weights/Qwen3/Qwen3-VL-2B-Instruct}"
INTERNVLA_A1_COSMOS_DIR="${INTERNVLA_A1_COSMOS_DIR:-/home/zhuyangkun/models/Cosmos-Tokenizer-CI8x8}"
INTERNVLA_A1_OUTPUT_DIR="${INTERNVLA_A1_OUTPUT_DIR:-$REPO_ROOT/outputs/internvla_a1/vllm_infer}"

export INTERNVLA_A1_MODEL_DIR
export INTERNVLA_A1_DATASET_DIR
export INTERNVLA_A1_PROCESSOR_DIR
export INTERNVLA_A1_COSMOS_DIR

python "$ROOT_DIR/run_internvla_a1_vllm_infer.py" \
  --model-dir "$INTERNVLA_A1_MODEL_DIR" \
  --dataset-dir "$INTERNVLA_A1_DATASET_DIR" \
  --output-dir "$INTERNVLA_A1_OUTPUT_DIR" \
  --num-episodes "${INTERNVLA_A1_NUM_EPISODES:-1}" \
  --attn-implementation eager \
  "$@"

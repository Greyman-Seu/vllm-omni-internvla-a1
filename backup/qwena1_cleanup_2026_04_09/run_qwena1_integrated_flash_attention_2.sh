#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_DIR="${MODEL_DIR:-/home/zhuyangkun/data/vllm_a1/2026_02_13_10_22_12-qwena1-a2d_real_A2D_Put_the_pen_from_the_table_into_the_pen_holder-delta-scratch-060000/pretrained_model}"
DATASET_DIR="${DATASET_DIR:-/home/zhuyangkun/data/vllm_a1/pick_marker_pen_inference_rollouts_v30}"
QWENA1_PROCESSOR_DIR="${QWENA1_PROCESSOR_DIR:-/mnt/shared-storage-user/internvla/shared_model_weights/Qwen3/Qwen3-VL-2B-Instruct}"
QWENA1_COSMOS_DIR="${QWENA1_COSMOS_DIR:-/home/zhuyangkun/models/Cosmos-Tokenizer-CI8x8}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
NUM_SAMPLES="${NUM_SAMPLES:-2}"
SEED="${SEED:-1234}"

export QWENA1_PROCESSOR_DIR
export QWENA1_COSMOS_DIR

python "$ROOT_DIR/run_qwena1_integrated_infer.py" \
  --mode both \
  --model-dir "$MODEL_DIR" \
  --dataset-dir "$DATASET_DIR" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --num-samples "$NUM_SAMPLES" \
  --seed "$SEED" \
  --attn-implementation flash_attention_2 \
  "$@"

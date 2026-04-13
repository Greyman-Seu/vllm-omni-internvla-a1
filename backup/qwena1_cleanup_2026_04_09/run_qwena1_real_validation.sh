#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_STANDALONE_ROOT="$ROOT_DIR/internvla-a1-new/internvla-a1"
if [[ ! -d "$DEFAULT_STANDALONE_ROOT" ]]; then
  DEFAULT_STANDALONE_ROOT="$ROOT_DIR/internvla-a1"
fi
DEFAULT_MODEL_DIR="/mnt/petrelfs/zhuyangkun/vllm_a1/data/InternVLA-A1-3B-ft-pen"
if [[ ! -d "$DEFAULT_MODEL_DIR" ]]; then
  DEFAULT_MODEL_DIR="/home/zhuyangkun/data/vllm_a1/2026_02_13_10_22_12-qwena1-a2d_real_A2D_Put_the_pen_from_the_table_into_the_pen_holder-delta-scratch-060000/pretrained_model"
fi
DEFAULT_DATASET_DIR="/mnt/petrelfs/zhuyangkun/vllm_a1/data/Genie1-Place_Markpen"
if [[ ! -d "$DEFAULT_DATASET_DIR" ]]; then
  DEFAULT_DATASET_DIR="/home/zhuyangkun/data/vllm_a1/pick_marker_pen_inference_rollouts_v30"
fi

MODEL_DIR="${MODEL_DIR:-$DEFAULT_MODEL_DIR}"
DATASET_DIR="${DATASET_DIR:-$DEFAULT_DATASET_DIR}"
STANDALONE_ROOT="${STANDALONE_ROOT:-$DEFAULT_STANDALONE_ROOT}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/zhuyangkun/data/vllm_a1/qwena1_validation_outputs}"
QWENA1_PROCESSOR_DIR="${QWENA1_PROCESSOR_DIR:-/mnt/shared-storage-user/internvla/shared_model_weights/Qwen3/Qwen3-VL-2B-Instruct}"
QWENA1_COSMOS_DIR="${QWENA1_COSMOS_DIR:-/home/zhuyangkun/models/Cosmos-Tokenizer-CI8x8}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
NUM_SAMPLES="${NUM_SAMPLES:-2}"
SEED="${SEED:-1234}"

export QWENA1_PROCESSOR_DIR
export QWENA1_COSMOS_DIR

mkdir -p "$OUTPUT_DIR"

echo "[1/5] fake-input smoke"
python "$ROOT_DIR/run_qwena1_fake_infer.py" \
  --mode both \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --batch-size 1 \
  --seed "$SEED" \
  > "$OUTPUT_DIR/fake_input_smoke.json"

echo "[2/5] integrated direct vs registry"
python "$ROOT_DIR/run_qwena1_integrated_infer.py" \
  --mode both \
  --model-dir "$MODEL_DIR" \
  --dataset-dir "$DATASET_DIR" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --num-samples "$NUM_SAMPLES" \
  --seed "$SEED" \
  > "$OUTPUT_DIR/integrated_direct_vs_registry.json"

echo "[3/5] integrated vs standalone baseline compare (eager, no regional compile)"
python "$ROOT_DIR/compare_qwena1_integrated_vs_standalone.py" \
  --standalone-root "$STANDALONE_ROOT" \
  --model-dir "$MODEL_DIR" \
  --dataset-dir "$DATASET_DIR" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --num-samples "$NUM_SAMPLES" \
  --seed "$SEED" \
  --attn-implementation eager \
  --output-json "$OUTPUT_DIR/compare_eager.json" \
  | tee "$OUTPUT_DIR/compare_eager.stdout"

echo "[4/5] integrated attention backend sweep"
python "$ROOT_DIR/run_qwena1_integrated_infer.py" \
  --mode both \
  --model-dir "$MODEL_DIR" \
  --dataset-dir "$DATASET_DIR" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --num-samples "$NUM_SAMPLES" \
  --seed "$SEED" \
  --attn-implementation sdpa \
  > "$OUTPUT_DIR/integrated_sdpa.json"

if python "$ROOT_DIR/run_qwena1_integrated_infer.py" \
  --mode both \
  --model-dir "$MODEL_DIR" \
  --dataset-dir "$DATASET_DIR" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --num-samples "$NUM_SAMPLES" \
  --seed "$SEED" \
  --attn-implementation flash_attention_2 \
  > "$OUTPUT_DIR/integrated_flash_attention_2.json"; then
  :
else
  echo "flash_attention_2 is unavailable in the current environment; skipped." \
    | tee "$OUTPUT_DIR/integrated_flash_attention_2.status"
fi

echo "[5/5] regional compile compare"
python "$ROOT_DIR/compare_qwena1_integrated_vs_standalone.py" \
  --standalone-root "$STANDALONE_ROOT" \
  --model-dir "$MODEL_DIR" \
  --dataset-dir "$DATASET_DIR" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --num-samples "$NUM_SAMPLES" \
  --seed "$SEED" \
  --attn-implementation eager \
  --enable-regional-compile \
  --output-json "$OUTPUT_DIR/compare_eager_regional_compile.json" \
  | tee "$OUTPUT_DIR/compare_eager_regional_compile.stdout"

echo "Validation outputs written to: $OUTPUT_DIR"

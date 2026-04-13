#!/usr/bin/env bash

ROOT_DIR="/mnt/petrelfs/zhuyangkun/workspace/vllm-omni"
PY_SCRIPT="$ROOT_DIR/compare_qwena1_precision.py"

usage() {
  cat <<'EOF'
Usage:
  ./compare_qwena1_precision.sh [python_args]

Examples:
  ./compare_qwena1_precision.sh
  ./compare_qwena1_precision.sh --num-samples 3 --dtype bfloat16
  ./compare_qwena1_precision.sh --repo-root /mnt/petrelfs/zhuyangkun/workspace/lerobot_lab

Notes:
  - This shell script only activates the lerobot_lab environment and forwards args.
  - The real entrypoint is: compare_qwena1_precision.py
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  echo
  source /mnt/petrelfs/zhuyangkun/envs/miniconda3/etc/profile.d/conda.sh
  conda activate lerobot_lab
  cd "$ROOT_DIR"
  python "$PY_SCRIPT" --help
  exit 0
fi

source /mnt/petrelfs/zhuyangkun/envs/miniconda3/etc/profile.d/conda.sh
conda activate lerobot_lab

cd "$ROOT_DIR"
python "$PY_SCRIPT" "$@"

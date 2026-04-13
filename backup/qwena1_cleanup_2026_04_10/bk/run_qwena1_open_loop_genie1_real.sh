#!/usr/bin/env bash

ROOT_DIR="/mnt/petrelfs/zhuyangkun/workspace/vllm-omni/vllm-omni-internvla-a1/internvla-a1"
PY_SCRIPT="$ROOT_DIR/run_qwena1_open_loop_genie1_real.py"

usage() {
  cat <<'EOF'
Usage:
  ./run_qwena1_open_loop_genie1_real.sh [python_args]

Examples:
  ./run_qwena1_open_loop_genie1_real.sh
  ./run_qwena1_open_loop_genie1_real.sh --mode repo --num-episodes 1
  ./run_qwena1_open_loop_genie1_real.sh --mode standalone --num-episodes 1

Notes:
  - This shell script only activates the lerobot_lab environment and forwards args.
  - The real entrypoint is: run_qwena1_open_loop_genie1_real.py
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

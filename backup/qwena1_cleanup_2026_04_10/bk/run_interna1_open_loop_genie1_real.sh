#!/usr/bin/env bash

ROOT_DIR="/mnt/petrelfs/zhuyangkun/workspace/vllm-omni/vllm-omni-internvla-a1/internvla-a1"
PY_SCRIPT="$ROOT_DIR/run_interna1_open_loop_genie1_real.py"

usage() {
  cat <<'EOF'
Usage:
  ./run_interna1_open_loop_genie1_real.sh [python_args]

Examples:
  ./run_interna1_open_loop_genie1_real.sh
  ./run_interna1_open_loop_genie1_real.sh --num-episodes 1
  ./run_interna1_open_loop_genie1_real.sh --checkpoint Jia-Zeng/InternVLA-A1-3B-FineTuned-Place_Markpen

Notes:
  - This shell script only activates the lerobot_lab environment and forwards args.
  - The real entrypoint is: run_interna1_open_loop_genie1_real.py
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

#!/usr/bin/env bash

ROOT_DIR="/mnt/petrelfs/zhuyangkun/workspace/vllm-omni"
PY_SCRIPT="$ROOT_DIR/run_qwena1_repo_infer.py"

usage() {
  cat <<'EOF'
Usage:
  ./run_qwena1_repo_infer.sh [python_args]

Examples:
  ./run_qwena1_repo_infer.sh
  ./run_qwena1_repo_infer.sh --num-samples 1 --seed 1234
  ./run_qwena1_repo_infer.sh --repo-root /mnt/petrelfs/zhuyangkun/workspace/lerobot_lab

Notes:
  - This shell script only activates the lerobot_lab environment and forwards args.
  - The real entrypoint is: run_qwena1_repo_infer.py
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

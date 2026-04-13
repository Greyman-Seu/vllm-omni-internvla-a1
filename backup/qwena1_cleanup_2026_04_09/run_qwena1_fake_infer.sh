#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$ROOT_DIR/run_qwena1_fake_infer.py"

python "$PY_SCRIPT" "$@"

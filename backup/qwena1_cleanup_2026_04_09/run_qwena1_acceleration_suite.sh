#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[1/4] eager baseline compare"
bash "$ROOT_DIR/run_qwena1_compare_eager.sh"

echo "[2/4] sdpa integrated"
bash "$ROOT_DIR/run_qwena1_integrated_sdpa.sh"

echo "[3/4] eager + regional_compile compare"
bash "$ROOT_DIR/run_qwena1_compare_eager_regional_compile.sh"

echo "[4/4] sdpa + regional_compile integrated"
bash "$ROOT_DIR/run_qwena1_integrated_sdpa_regional_compile.sh"

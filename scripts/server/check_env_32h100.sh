#!/usr/bin/env bash
# Quick environment probe on the 32×H100 training server.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/tiger/verl_envs/v080_qwen35_cu128/bin/python}"

echo "=== verl-opd-qwen3-32h100 env check ==="
echo "ROOT:        ${ROOT}"
echo "PYTHON_BIN:  ${PYTHON_BIN}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "WARN: Python not found at ${PYTHON_BIN}"
else
  "${PYTHON_BIN}" --version
  "${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true
  "${PYTHON_BIN}" -c "import ray; print('ray', ray.__version__)" || echo "WARN: ray not importable"
fi

VER=$(cat "${ROOT}/verl/version/version" 2>/dev/null || echo "unknown")
echo "verl version file: ${VER}"

for var in STUDENT_MODEL TEACHER_MODEL TRAIN_FILE TEST_FILE CKPT_ROOT; do
  if [[ -n "${!var:-}" ]]; then
    echo "${var}=${!var}"
    if [[ "${var}" != CKPT_ROOT ]] && [[ ! -e "${!var}" ]]; then
      echo "  WARN: path does not exist"
    fi
  fi
done

if [[ -n "${RAY_ADDRESS:-}" ]]; then
  echo "RAY_ADDRESS=${RAY_ADDRESS}"
else
  echo "WARN: RAY_ADDRESS is not set; training will use Ray local/default init"
fi

if command -v nvidia-smi &>/dev/null; then
  echo "--- nvidia-smi (first GPU) ---"
  nvidia-smi -L | head -3
else
  echo "WARN: nvidia-smi not in PATH"
fi

if command -v ray &>/dev/null; then
  echo "--- ray status ---"
  ray status 2>/dev/null | head -20 || echo "ray status failed (cluster down?)"
fi

echo "=== check done ==="

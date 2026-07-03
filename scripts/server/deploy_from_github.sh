#!/usr/bin/env bash
# Clone verl-opd-qwen3-32h100 from GitHub and install on the training server.
set -euo pipefail

REPO_URL="${1:?Usage: $0 <github-repo-url> [install-dir]}"
INSTALL_DIR="${2:-/opt/tiger/verl-opd-qwen3-32h100}"
PYTHON_BIN="${PYTHON_BIN:-/opt/tiger/verl_envs/v080_qwen35_cu128/bin/python}"

if [[ -f /opt/tiger/verl_env.sh ]]; then
  # Provides the cluster's proxy/CA settings for GitHub and pip.
  # shellcheck disable=SC1091
  source /opt/tiger/verl_env.sh
fi

echo "==> Clone ${REPO_URL} -> ${INSTALL_DIR}"
if [[ -d "${INSTALL_DIR}/.git" ]]; then
  echo "    Directory exists; git pull"
  git -C "${INSTALL_DIR}" pull --ff-only
else
  git clone "${REPO_URL}" "${INSTALL_DIR}"
fi

cd "${INSTALL_DIR}"
echo "==> Install editable (${PYTHON_BIN})"
"${PYTHON_BIN}" -m pip install -e . --no-deps

if [[ -d /opt/tiger ]]; then
  if [[ -e /opt/tiger/verl && ! -L /opt/tiger/verl ]]; then
    BACKUP_DIR="/opt/tiger/verl.backup.$(date +%Y%m%d_%H%M%S)"
    echo "==> Backup existing /opt/tiger/verl -> ${BACKUP_DIR}"
    mv /opt/tiger/verl "${BACKUP_DIR}"
  fi
  echo "==> Symlink /opt/tiger/verl -> ${INSTALL_DIR}"
  ln -sfn "${INSTALL_DIR}" /opt/tiger/verl
fi

echo "==> Done. Next:"
echo "    cp examples/on_policy_distillation_trainer/env_32h100.example .env.32h100"
echo "    bash scripts/server/check_env_32h100.sh"

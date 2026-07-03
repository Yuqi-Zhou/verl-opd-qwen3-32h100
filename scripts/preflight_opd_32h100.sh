#!/usr/bin/env bash
# Preflight checks before launching 32xH100 OPD training.
# Usage: bash scripts/preflight_opd_32h100.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== verl OPD 32xH100 preflight =="

if [[ -f verl/version/version ]]; then
  echo "verl version: $(cat verl/version/version)"
else
  echo "verl version: (verl/version/version not found)"
fi

required=(
  "examples/on_policy_distillation_trainer/run_qwen3_30b_a3b_235b_moe_fully_async_32h100.sh"
  "CODEX_README.md"
  "verl/experimental/fully_async_policy/fully_async_main.py"
)
for f in "${required[@]}"; do
  [[ -f "$f" ]] || { echo "MISSING: $f"; exit 1; }
  echo "OK: $f"
done

: "${STUDENT_MODEL:=}"
: "${TEACHER_MODEL:=}"
if [[ -z "$STUDENT_MODEL" || -z "$TEACHER_MODEL" ]]; then
  echo "WARN: set STUDENT_MODEL and TEACHER_MODEL before training"
else
  [[ -d "$STUDENT_MODEL" || -f "$STUDENT_MODEL/config.json" ]] && echo "OK: STUDENT_MODEL" || echo "WARN: STUDENT_MODEL path not found: $STUDENT_MODEL"
  [[ -d "$TEACHER_MODEL" || -f "$TEACHER_MODEL/config.json" ]] && echo "OK: TEACHER_MODEL" || echo "WARN: TEACHER_MODEL path not found: $TEACHER_MODEL"
fi

echo ""
echo "GPU budget check (expected 32 total):"
echo "  rollout:  ${NGPUS_ROLLOUT:-8}"
echo "  train:    ${NGPUS_TRAIN:-8}"
echo "  teacher:  $((${TEACHER_NNODES:-2} * ${TEACHER_NGPUS_PER_NODE:-8}))"
echo ""
echo "Teacher footprint: replicas=${TEACHER_REPLICAS:-2} x TP=${TEACHER_TP:-8} = $((${TEACHER_REPLICAS:-2} * ${TEACHER_TP:-8})) (must equal teacher pool)"
echo ""
echo "Preflight done. See CODEX_README.md for launch steps."

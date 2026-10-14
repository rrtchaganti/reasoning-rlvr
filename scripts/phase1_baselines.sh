#!/usr/bin/env bash
# Phase 1 local runs: perturbed GSM8K, EXP-001 greedy rows (8 configs), EXP-003 partial-prompt probe.
# Usage (inside WSL2, venv active, clean tree):  bash scripts/phase1_baselines.sh [--allow-dirty]
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== perturbed GSM8K (seed 0)"
rlvr-perturb-gsm8k --seed 0
echo "== EXP-001 greedy baselines"
for cfg in configs/eval/base_*.yaml; do
  echo "---- $cfg"
  rlvr-eval --config "$cfg" "$@"
done
echo "== EXP-003 partial-prompt probe"
rlvr-probe-partial --config configs/probe/partial_prompt_1.7b.yaml "$@"
echo "== done"

#!/usr/bin/env bash
# Run CI's own commands in an environment built the way CI builds it: CPU torch plus the dev extras,
# no GPU extras (vllm, trl, peft). Catches tests that reach a GPU-extra import without importorskip,
# which the laptop gate (full GPU venv, slow tests deselected) cannot see. Run before opening a PR.
# The venv is built once in ~/venvs/ci_mimic (about 1 GB, a few minutes); later runs reuse it.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${CI_VENV:-$HOME/venvs/ci_mimic}
if [ ! -x "$VENV/bin/python" ]; then
  uv venv -q -p 3.12 "$VENV"
  VIRTUAL_ENV="$VENV" uv pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  VIRTUAL_ENV="$VENV" uv pip install -q -e ".[dev]"
fi
# The editable install records the checkout it was made from; re-point it at this one every run
# (a venv built from a since-deleted worktree imported nothing: 13 collection errors, 24 Oct 2026).
VIRTUAL_ENV="$VENV" uv pip install -q --no-deps -e .
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export CUDA_VISIBLE_DEVICES=""
echo "== ruff check"; ruff check .
echo "== ruff format"; ruff format --check .
echo "== pytest -m 'not gpu' (slow tests included, as in CI)"; pytest -q -m "not gpu" -p no:cacheprovider
echo "== CI commands passed"

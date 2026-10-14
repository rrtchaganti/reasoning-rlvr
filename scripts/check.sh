#!/usr/bin/env bash
# The pre-commit gate, runnable by hand: lint, format check, fast CPU tests.
# Usage (inside WSL2, venv active):  bash scripts/check.sh        # add --gpu to include GPU tests
set -euo pipefail
cd "$(dirname "$0")/.."
MARK='not gpu and not slow'
if [[ "${1:-}" == "--gpu" ]]; then MARK='not slow'; fi
echo "== ruff check";        ruff check .
echo "== ruff format";       ruff format --check .
echo "== pytest ($MARK)";    pytest -q -m "$MARK" -p no:cacheprovider
echo "== all checks passed"

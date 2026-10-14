#!/usr/bin/env bash
# Build the Python environment for reasoning-rlvr. Works without sudo (WSL2 laptop or a rental box).
#
#   bash scripts/setup_env.sh            # GPU environment: vllm, trl, peft, wandb + dev tools
#   bash scripts/setup_env.sh --cpu      # CI-style environment: package + dev tools only
#
# Why uv and a uv-managed Python: distro Pythons ship without pip, venv and the C headers that
# Triton needs to compile its runtime extension (FAILURES_AND_LEARNINGS.md, 14 Oct 2026).
# A uv-managed CPython is self-contained and identical on every machine.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${RLVR_VENV:-$HOME/venvs/rlvr}"
PY_VERSION="$(cat .python-version 2>/dev/null || echo 3.12)"
MODE="${1:-gpu}"

if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  echo "== installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "== python $PY_VERSION (uv-managed)"
uv python install "$PY_VERSION"
if [ ! -d "$VENV" ]; then
  uv venv "$VENV" --python "$PY_VERSION" --python-preference only-managed
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python --version

echo "== header check (Triton needs Python.h)"
python - <<'PY'
import os, sys, sysconfig
inc = sysconfig.get_paths()["include"]
ok = os.path.exists(os.path.join(inc, "Python.h"))
print(f"{inc}  Python.h: {'found' if ok else 'MISSING'}")
sys.exit(0 if ok else 1)
PY

if [ "$MODE" == "--cpu" ]; then
  echo "== installing package + dev extras (CPU)"
  uv pip install -e ".[dev]"
else
  echo "== installing vllm (brings its pinned torch)"
  uv pip install vllm
  echo "== installing package + dev + gpu extras"
  uv pip install -e ".[dev,gpu]"
  echo "== verify"
  python - <<'PY'
import torch, vllm, transformers, trl, math_verify, datasets
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("vllm", vllm.__version__, "transformers", transformers.__version__, "trl", trl.__version__)
PY
  if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    # @triton.jit needs a real .py file (not stdin), so the kernel test lives in its own script.
    python scripts/triton_check.py
  fi
fi

echo "== lock file"
uv pip freeze | grep -v '^rlvr' > requirements-lock.txt
echo "wrote requirements-lock.txt ($(wc -l < requirements-lock.txt) lines)"
echo "== done. Activate with:  source $VENV/bin/activate"

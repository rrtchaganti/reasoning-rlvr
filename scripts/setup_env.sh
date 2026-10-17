#!/usr/bin/env bash
# Build the Python environment for reasoning-rlvr. Works without sudo (WSL2 laptop or a rental box).
#
#   bash scripts/setup_env.sh            # GPU environment: vllm, trl, peft, wandb + dev tools
#   bash scripts/setup_env.sh --cpu      # CI-style environment: package + dev tools only
#   RLVR_VLLM_CUDA=cu129 bash scripts/setup_env.sh
#                                        # driver older than CUDA 13 (e.g. E2E's 570 driver, CUDA 12.8):
#                                        # vLLM's +cu129 release wheel, torch family via uv's cu129 backend,
#                                        # same vLLM version as the lock file (docs/rental-setup.md 2c)
#   RLVR_WRITE_LOCK=1 bash scripts/setup_env.sh
#                                        # laptop only: rewrite the tracked requirements-lock.txt
#
# Why uv and a uv-managed Python: distro Pythons ship without pip, venv and the C headers that
# Triton needs to compile its runtime extension (FAILURES_AND_LEARNINGS.md, 14 Oct 2026).
# A uv-managed CPython is self-contained and identical on every machine.
set -euo pipefail
cd "$(dirname "$0")/.."
VENV="${RLVR_VENV:-$HOME/venvs/rlvr}"
PY_VERSION="$(cat .python-version 2>/dev/null || echo 3.12)"
MODE="${1:-gpu}"
VLLM_CUDA="${RLVR_VLLM_CUDA:-}"
# The box installs the vLLM version the laptop was locked at, never "latest".
VLLM_VERSION="${RLVR_VLLM_VERSION:-$(grep -E '^vllm==' requirements-lock.txt 2>/dev/null | cut -d= -f3 || true)}"

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
  if [ -n "$VLLM_CUDA" ]; then
    [ -n "$VLLM_VERSION" ] || { echo "RLVR_VLLM_VERSION needed (no vllm== line in requirements-lock.txt)"; exit 1; }
    # uv's torch backend sends only the torch family (torch, triton, nvidia-*) to the PyTorch index
    # for this CUDA build and everything else to PyPI. An --extra-index-url would not do: that index
    # also hosts an old `packaging`, and uv stops at the first index that has a package (17 Oct).
    # The vLLM wheel is the matching release asset (vllm-<ver>+<cuda>-...manylinux_2_28_<arch>.whl).
    export UV_TORCH_BACKEND="$VLLM_CUDA"
    WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${VLLM_CUDA}-cp38-abi3-manylinux_2_28_$(uname -m).whl"
    echo "== installing vllm ${VLLM_VERSION}+${VLLM_CUDA} from the release wheel; torch backend ${UV_TORCH_BACKEND}"
    uv pip install "$WHEEL"
  else
    echo "== installing vllm ${VLLM_VERSION:-latest} (brings its pinned torch)"
    uv pip install "vllm${VLLM_VERSION:+==$VLLM_VERSION}"
  fi
  echo "== installing package + dev + gpu extras"
  if [ "${RLVR_WRITE_LOCK:-0}" != "1" ] && [ -f requirements-lock.txt ]; then
    # Pin the extras to the laptop's lock (the first E2E node picked up trl 1.16 against the tested
    # 1.15). `==2.13.0` in the lock still accepts the box's torch 2.13.0+cu129: a local version label
    # satisfies a plain == specifier, so torch's CUDA build is the one difference allowed to remain.
    uv pip install -e ".[dev,gpu]" --constraint requirements-lock.txt
  else
    uv pip install -e ".[dev,gpu]"
  fi
  echo "== verify"
  python - <<'PY'
import torch, vllm, transformers, trl, math_verify, datasets
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("vllm", vllm.__version__, "transformers", transformers.__version__, "trl", trl.__version__)
PY
  if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    # @triton.jit needs a real .py file (not stdin), so the kernel test lives in its own script.
    python scripts/triton_check.py
  fi
fi

if [ "${RLVR_WRITE_LOCK:-0}" = "1" ]; then
  echo "== lock file (tracked)"
  uv pip freeze | grep -v '^rlvr' > requirements-lock.txt
  echo "wrote requirements-lock.txt ($(wc -l < requirements-lock.txt) lines)"
else
  # Never dirty the tree on a box: the harness refuses to run on uncommitted changes. The freeze
  # goes to an ignored file; its torch/vllm lines are what COMPUTE.md records for the session.
  mkdir -p outputs
  LOCK_OUT="outputs/requirements-lock.$(hostname).txt"
  uv pip freeze | grep -v '^rlvr' > "$LOCK_OUT"
  echo "== wrote $LOCK_OUT; every difference from the tracked lock (nvidia-* libraries excluded):"
  diff <(grep -v -E '^(nvidia-|rlvr)' requirements-lock.txt | sed 's/+cu[0-9]*//') \
       <(grep -v -E '^(nvidia-|rlvr)' "$LOCK_OUT" | sed 's#^vllm @ .*/vllm-\([0-9.]*\)+.*#vllm==\1#; s/+cu[0-9]*//') \
    || true
fi
echo "== done. Activate with:  source $VENV/bin/activate"

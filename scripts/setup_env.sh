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
    # The PyTorch index serves triton (and sometimes torch) from download-r2.pytorch.org, which the TIR
    # H100 node reached at under 1 MB/s with multi-minute stalls on 20 Oct 2026 (uv hung three times,
    # 87 min lost). Any torch-*/triton-* wheel under $RLVR_HOME/wheels (fetched with `curl -C -` and
    # sha256-checked against the index's #sha256= fragment) installs first, so uv has nothing to fetch
    # from that host. The versions must be the ones the vllm wheel pins (torch 2.13.0+cu129, triton 3.7.1).
    for w in "${RLVR_HOME:-$HOME}"/wheels/torch-*.whl "${RLVR_HOME:-$HOME}"/wheels/triton-*.whl; do
      [ -f "$w" ] && { echo "== pre-installing $w"; uv pip install --no-deps "$w"; }
    done
    WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${VLLM_CUDA}-cp38-abi3-manylinux_2_28_$(uname -m).whl"
    echo "== installing vllm ${VLLM_VERSION}+${VLLM_CUDA} from the release wheel; torch backend ${UV_TORCH_BACKEND}"
    uv pip install "$WHEEL"
    # torchcodec is a vllm dependency used only for audio decoding. uv's torch backend does not route it
    # to the CUDA-specific index, so PyPI's wheel (built for torch 2.13's default CUDA 13) lands in a
    # cu129 venv and `import vllm` dies with "libnvrtc.so.13: cannot open shared object file" (TIR H100,
    # 18 Oct 2026). vllm's import of it is guarded, so the module may simply be absent.
    uv pip uninstall torchcodec || true
  else
    echo "== installing vllm ${VLLM_VERSION:-latest} (brings its pinned torch)"
    uv pip install "vllm${VLLM_VERSION:+==$VLLM_VERSION}"
  fi
  echo "== installing package + dev + gpu extras"
  if [ "${RLVR_WRITE_LOCK:-0}" != "1" ] && [ -f requirements-lock.txt ]; then
    # Pin the extras to the laptop's lock (the first E2E node picked up trl 1.16 against the tested
    # 1.15). `==2.13.0` in the lock still accepts the box's torch 2.13.0+cu129: a local version label
    # satisfies a plain == specifier, so torch's CUDA build is the one difference allowed to remain.
    # Named pins only, minus the families that differ by CUDA build and are pinned by the vllm
    # wheel itself (torch*, triton, nvidia-*, cuda-*, flashinfer, xformers): on the first node the
    # lock's cuda-bindings 13.x clashed with torch+cu129's 12.9 requirement.
    mkdir -p outputs
    grep -E '^[A-Za-z0-9_.-]+==' requirements-lock.txt \
      | grep -v -i -E '^(torch|triton|nvidia-|cuda-|flashinfer|xformers|vllm)' > outputs/constraints.txt
    uv pip install -e ".[dev,gpu]" --constraint outputs/constraints.txt
  else
    uv pip install -e ".[dev,gpu]"
  fi
  # The extras install re-resolves vllm's dependencies and brings torchcodec back (20 Oct 2026: the
  # rft dev eval died on libnvrtc.so.13). Remove it again here, after the last install into the venv.
  if [ -n "$VLLM_CUDA" ]; then uv pip uninstall torchcodec || true; fi
  # flash-attention: no prebuilt wheel exists for the locked torch 2.13 (Dao-AILab stops at torch 2.8),
  # so a wheel is built once per image and kept on the persistent disk (13 min on the TIR H100 image
  # with its nvcc 12.9, 19 Oct 2026; the build recipe is in docs/rental-setup.md section 2b). Any
  # flash_attn wheel under $RLVR_HOME/wheels (or the file named by RLVR_FLASH_WHEEL) installs here in
  # a second; without one, training runs with sdpa and packing=false (RLVR_TRAIN_EXTRA in the handoff).
  FLASH_WHEEL="${RLVR_FLASH_WHEEL:-}"
  if [ -z "$FLASH_WHEEL" ]; then
    for w in "${RLVR_HOME:-$HOME}"/wheels/flash_attn-*.whl; do [ -f "$w" ] && FLASH_WHEEL="$w" && break; done
  fi
  if [ -n "$FLASH_WHEEL" ] && [ -f "$FLASH_WHEEL" ]; then
    echo "== installing flash-attn from $FLASH_WHEEL"
    uv pip install "$FLASH_WHEEL" --no-deps
  else
    echo "== no flash-attn wheel found (RLVR_FLASH_WHEEL unset, none under ${RLVR_HOME:-$HOME}/wheels): sdpa + packing=false for training"
  fi
  echo "== verify"
  python - <<'PY'
import torch, vllm, transformers, trl, math_verify, datasets
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("vllm", vllm.__version__, "transformers", transformers.__version__, "trl", trl.__version__)
try:
    import flash_attn
    print("flash_attn", flash_attn.__version__)
except Exception as e:  # noqa: BLE001
    print("flash_attn absent:", type(e).__name__)
import importlib.util, os
if os.environ.get("UV_TORCH_BACKEND", "").startswith("cu12") and importlib.util.find_spec("torchcodec"):
    raise SystemExit("torchcodec is installed in a cu12x venv: it needs CUDA 13 and breaks `import vllm` at run time")
PY
  if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    # @triton.jit needs a real .py file (not stdin), so the kernel test lives in its own script.
    python scripts/triton_check.py
  fi
fi

if [ "${RLVR_WRITE_LOCK:-0}" = "1" ]; then
  echo "== lock file (tracked)"
  uv pip freeze | grep -v -E '^(rlvr|-e |file:)' > requirements-lock.txt   # no editable-install line
  echo "wrote requirements-lock.txt ($(wc -l < requirements-lock.txt) lines)"
else
  # Never dirty the tree on a box: the harness refuses to run on uncommitted changes. The freeze
  # goes to an ignored file; its torch/vllm lines are what COMPUTE.md records for the session.
  mkdir -p outputs
  LOCK_OUT="outputs/requirements-lock.$(hostname).txt"
  uv pip freeze | grep -v -E '^(rlvr|-e |file:)' > "$LOCK_OUT"
  echo "== wrote $LOCK_OUT; every difference from the tracked lock (nvidia-* and cuda-* libraries excluded):"
  diff <(grep -v -i -E '^(nvidia-|cuda-|rlvr)' requirements-lock.txt | sed 's/+cu[0-9]*//') \
       <(grep -v -i -E '^(nvidia-|cuda-|rlvr)' "$LOCK_OUT" | sed 's#^vllm @ .*/vllm-\([0-9.]*\)+.*#vllm==\1#; s/+cu[0-9]*//') \
    || true
fi
echo "== done. Activate with:  source $VENV/bin/activate"

#!/usr/bin/env bash
# Source this on every E2E node after the volume is mounted (docs/rental-setup.md, section 2c):
#   source /mnt/rlvr/reasoning-rlvr/scripts/e2e_env.sh
# Everything lives at fixed paths on the volume, so the environment built once on the cheap node
# works unchanged on the A100: the venv, uv's CPython (the venv points at it by path), the wheel
# cache and the Hugging Face cache. Secrets are not here: export HF_TOKEN and WANDB_API_KEY in the
# shell of each session.
export RLVR_HOME="${RLVR_HOME:-/mnt/rlvr}"
export RLVR_VENV="$RLVR_HOME/venv"
export UV_PYTHON_INSTALL_DIR="$RLVR_HOME/uv-python"
export UV_CACHE_DIR="$RLVR_HOME/uv-cache"
export HF_HOME="$RLVR_HOME/hf"
export HF_HUB_DISABLE_XET=1
export WANDB_PROJECT="${WANDB_PROJECT:-reasoning-rlvr}"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR" "$HF_HOME"
if [ -f "$RLVR_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$RLVR_VENV/bin/activate"
  echo "venv active: $(python --version 2>&1) at $RLVR_VENV"
else
  echo "no venv at $RLVR_VENV yet: run  bash scripts/setup_env.sh  (first time only)"
fi
echo "HF_HOME=$HF_HOME   repo: $RLVR_HOME/reasoning-rlvr"

#!/usr/bin/env bash
# Source this on every rented node, from the repo root or anywhere:
#   source scripts/e2e_env.sh
# Everything lives at fixed paths under RLVR_HOME, so the environment built once keeps working on
# every later start of the same disk: the venv, uv's CPython (the venv points at it by path), the
# wheel cache and the Hugging Face cache. Default RLVR_HOME:
#   /home/jovyan/rlvr  on a TIR notebook (E2E's managed platform, chosen 18 Oct 2026): the workspace
#                      disk at /home/jovyan persists across Stop and Start (docs/rental-setup.md, 2d)
#   /mnt/rlvr          on a MyAccount node with the block volume (section 2c; the 17 Oct L4 session)
# Secrets are not here: export HF_TOKEN and WANDB_API_KEY in the shell of each session.
if [ -z "${RLVR_HOME:-}" ]; then
  if [ -d /home/jovyan ]; then RLVR_HOME=/home/jovyan/rlvr; else RLVR_HOME=/mnt/rlvr; fi
fi
export RLVR_HOME
export RLVR_VENV="$RLVR_HOME/venv"
export UV_PYTHON_INSTALL_DIR="$RLVR_HOME/uv-python"
export UV_CACHE_DIR="$RLVR_HOME/uv-cache"
export HF_HOME="$RLVR_HOME/hf"
export HF_HUB_DISABLE_XET=1
export WANDB_PROJECT="${WANDB_PROJECT:-reasoning-rlvr}"
export PATH="$HOME/.local/bin:$PATH"
# Price of this node in INR per hour; scripts/project_sampling.py reads it as its default --rate.
# TIR H100 SXM 80 GB: 379.05 (18 Oct 2026). MyAccount A100 80 GB was 189, L4 49. Override when the
# notebook runs on another plan (export RLVR_NODE_RATE=189 before sourcing).
export RLVR_NODE_RATE="${RLVR_NODE_RATE:-379}"
# E2E images put their own CUDA toolkit (12.6 on the 17 Oct 2026 L4 image) on LD_LIBRARY_PATH.
# torch and vLLM ship the CUDA libraries they were built against (12.9 here); a system libcudart
# ahead of them on the search path is exactly the mismatch we do not want. The toolkit's nvcc stays
# on PATH but nothing in this project compiles with it (flash-attn is not built on E2E, see 2c).
# One exception: the TIR H100 image (18 Oct 2026) runs a CUDA 12.9 user space on a 550.127.08 host
# driver (CUDA 12.4). That only works through NVIDIA's forward-compatibility libcuda in the image's
# cuda compat directory, which must stay first on LD_LIBRARY_PATH. Keep that directory, drop the rest.
RLVR_CUDA_COMPAT=""
for d in /usr/local/cuda/compat /usr/local/cuda-*/compat; do
  if [ -e "$d/libcuda.so.1" ]; then RLVR_CUDA_COMPAT="$d"; break; fi
done
if [ -n "$RLVR_CUDA_COMPAT" ]; then
  export LD_LIBRARY_PATH="$RLVR_CUDA_COMPAT"
  echo "cuda forward-compat libcuda kept on LD_LIBRARY_PATH: $RLVR_CUDA_COMPAT"
else
  unset LD_LIBRARY_PATH
fi
export RLVR_CUDA_COMPAT
# The TIR image preinstalls vLLM 0.15.1 and Transformers 4.57.6 for its own Python. A PYTHONPATH that
# points at them would shadow the locked versions inside the venv; the venv is self-contained.
unset PYTHONPATH
mkdir -p "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR" "$HF_HOME"
if [ -f "$RLVR_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$RLVR_VENV/bin/activate"
  echo "venv active: $(python --version 2>&1) at $RLVR_VENV"
else
  echo "no venv at $RLVR_VENV yet: run  RLVR_VLLM_CUDA=cu129 bash scripts/setup_env.sh  (first time only; see docs/rental-setup.md 2d)"
fi
echo "RLVR_HOME=$RLVR_HOME   HF_HOME=$HF_HOME   repo: $RLVR_HOME/reasoning-rlvr   node rate: INR $RLVR_NODE_RATE/h"

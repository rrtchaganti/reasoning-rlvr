# Next session on the TIR H100 notebook: setup in the fewest idle minutes

Written on the node, 19 Oct 2026 12:50 UTC, from what the 18-19 Oct session actually ran (its session log and
notes are in the laptop's `../h100_session_archive/outputs/`); folded into the repo the same day with the
node's workarounds removed where the fix has landed on `phase-3-rft` (flags override configs at any value,
numeric YAML strings are cast, `setup_env.sh` removes torchcodec and installs a cached flash-attn wheel). Two paths: **A** when the workspace disk is the same (everything
below persists under /home/jovyan/rlvr; about 5 minutes to the first GPU job) and **B** when the disk is new
(about 45 minutes, of which 25 are the environment build and 13 the flash-attention build; both are CPU-only,
so start the first GPU job as soon as the venv imports).

What persists on the workspace disk (18-19 Oct state):

| path | what | size |
|---|---|---|
| /home/jovyan/rlvr/reasoning-rlvr | the repo, branch phase-3-rft at bb6f2a6, tree clean of tracked changes | small |
| /home/jovyan/rlvr/venv | Python 3.12.12 venv: torch 2.13.0+cu129, vLLM 0.31.0+cu129, transformers 5.17.0, trl 1.15.0, flash-attn 2.8.3.post2, pip, ninja | 12 GB |
| /home/jovyan/rlvr/uv-python, uv-cache | uv's CPython and wheel cache (rebuild is fast while they exist) | 12 GB |
| /home/jovyan/rlvr/hf | HF_HOME: Qwen3-1.7B-Base and Qwen3-0.6B-Base, datasets | 11 GB |
| /home/jovyan/rlvr/wheels/flash_attn-2.8.3.post2-cp312-cp312-linux_x86_64.whl | the flash-attention wheel built here (sm_90, torch 2.13 cu129) | 60 MB |
| outputs/ (git-ignored) | data sets, evals, samples, every final/ checkpoint, the shim outputs/bin/rlvr-sft, patches, logs, notes | 83 GB used in all |

Node facts to re-check at start (they drove every decision): driver 550.127.08 with the CUDA 12.9 forward-compat
libcuda in /usr/local/cuda/compat (scripts/e2e_env.sh keeps it first on LD_LIBRARY_PATH); nvcc 12.9 at
/usr/local/cuda/bin; cgroup limits 26 vCPUs and 250 GB RAM (nproc lies: 256); rate INR 379/h.

## Path A: same disk (about 5 minutes)

Step 0, before Start: have HF_TOKEN and WANDB_API_KEY ready; nothing else is needed from the laptop unless the
repo moved on (then the laptop pushes and the node fetches, step 3).

```bash
# 1. secrets, in the notebook terminal (never into a file). The tmux server inherits the environment of the shell
#    that starts it, so export BOTH before the first tmux command, and also set them in tmux's global environment
#    so later sessions and key rotations work (this session lost 40 min to a stale key in a shell copy).
export HF_TOKEN=...   # paste
export WANDB_API_KEY=...   # paste
tmux start-server; tmux set-environment -g HF_TOKEN "$HF_TOKEN"; tmux set-environment -g WANDB_API_KEY "$WANDB_API_KEY"

# 2. facts and disk (30 s)
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; df -h /home/jovyan | tail -1
cat /sys/fs/cgroup/cpu.max; cat /sys/fs/cgroup/memory.max

# 3. repo (read-only token on the node; never commit or push here)
cd /home/jovyan/rlvr/reasoning-rlvr && git status --short --untracked-files=no && git fetch -q && git log --oneline -1
#    if the laptop pushed new commits:  git checkout <branch> && git pull --ff-only

# 4. environment: the venv is already there, e2e_env.sh activates it (10 s)
source scripts/e2e_env.sh
python -c "import torch, vllm, transformers, trl, flash_attn; print(torch.__version__, torch.cuda.is_available(), vllm.__version__, flash_attn.__version__)"
python scripts/triton_check.py            # must print "triton compile + launch: ok"

# 5. data and models are on the disk; these commands only verify (hash check, no rebuild when present)
ls outputs/data/sft/train.jsonl outputs/data/gsm8k_perturbed* 2>/dev/null; ls /home/jovyan/rlvr/hf/hub | grep Qwen
#    if anything is missing:  bash scripts/phase2_rental.sh data   (4.5 min)  /  hf download Qwen/Qwen3-1.7B-Base

# 6. the learning-rate shim of the 18 Oct session (outputs/bin/rlvr-sft) is obsolete since the string-to-float
#    cast landed in src/rlvr/train/sft.py (19 Oct): delete it so it cannot shadow the venv's entry point
rm -f outputs/bin/rlvr-sft

# 7. first GPU job, inside tmux, started with -e HF_TOKEN only (WANDB_API_KEY comes from the tmux global env)
tmux new -d -s <name> -e HF_TOKEN="$HF_TOKEN" "bash /home/jovyan/rlvr/reasoning-rlvr/outputs/<runner>.sh"
```

Training flags that use the card (pre-registered effective batch 64): with flash-attention and the configs'
`packing: true`, `--set attn_implementation=flash_attention_2 --set per_device_batch=8 --set grad_accum=8`
(about 50 GB, 11.8 s/step). Without flash (fallback only): `--set packing=false --set attn_implementation=sdpa
--set per_device_batch=16 --set grad_accum=4` (70 GB, 15 s/step). Set them through RLVR_TRAIN_EXTRA for the
phase scripts. The Hub repo names: HF_REPO_SFT=Raviteja546/reasoning-rlvr-sft, HF_REPO_SAMPLES=
Raviteja546/reasoning-rlvr-samples, HF_REPO_RFT=Raviteja546/reasoning-rlvr-rft.

Since 19 Oct an explicit `rlvr-eval` flag overrides the config whatever value it carries (the session's 4,095
and 6,145 workarounds are no longer needed), and a `--limit` on gsm8k_perturbed keeps the problems paired with
gsm8k's subset. Still confirm the first set's name and the cap in the log within the first minutes of a launch.

## Path B: new disk (about 15 minutes with the Hub wheels, 20 Oct 2026; 45 minutes without them)

Run steps 1-3 of path A first (git clone instead of fetch: the repo URL and read-only token are on the laptop;
`mkdir -p /home/jovyan/rlvr && cd /home/jovyan/rlvr && git clone <url> reasoning-rlvr && git checkout <branch>`).

```bash
cd /home/jovyan/rlvr/reasoning-rlvr
# 3b. (20 Oct 2026) the three wheels the build needs are on the Hub: torch 2.13.0+cu129, triton 3.7.1 and the
#     flash-attn 2.8.3.post2 build for this image, with sha256.txt (1.1 GB, about 1 min at 28 MB/s). Fetch them
#     first: setup_env.sh installs any torch-*/triton-*/flash_attn-* wheel it finds under /home/jovyan/rlvr/wheels,
#     so uv fetches neither torch nor triton from download-r2.pytorch.org (under 1 MB/s with stalls from this node;
#     87 minutes lost on 20 Oct) and flash-attn is not compiled again (19 min).
hf download Raviteja546/reasoning-rlvr-samples --repo-type dataset --include "wheels/*" --local-dir /home/jovyan/rlvr
(cd /home/jovyan/rlvr/wheels && sha256sum -c sha256.txt)      # all three must say OK
# 4. environment build, cu129 route (2.5 min with the wheels present, 25 min without; CPU + network only), and the
#    model download in parallel (Qwen3-1.7B-Base only: the 0.6B student is not used before Phase 6)
tmux new -d -s build -e HF_TOKEN="$HF_TOKEN" "source scripts/e2e_env.sh && RLVR_VLLM_CUDA=cu129 bash scripts/setup_env.sh > outputs/h100_setup_env.log 2>&1; echo EXIT=\$? >> outputs/h100_setup_env.log"
tmux new -d -s dl -e HF_TOKEN="$HF_TOKEN" "export HF_HOME=/home/jovyan/rlvr/hf HF_HUB_DISABLE_XET=1; hf download Qwen/Qwen3-1.7B-Base > outputs/h100_download.log 2>&1; echo EXIT=\$? >> outputs/h100_download.log"
#    wait for the last line of outputs/h100_setup_env.log to be EXIT=0 (anchored watcher), then:
#    (if the build stalls: `du -sm /home/jovyan/rlvr/uv-cache` twice five minutes apart; flat means kill the tmux
#    session and rerun the same line, uv keeps what it has)
source scripts/e2e_env.sh
python scripts/triton_check.py              # setup_env.sh already removed torchcodec (PyPI's wheel needs CUDA 13)

# 5. flash-attention: setup_env.sh installs any wheel under /home/jovyan/rlvr/wheels by itself; rebuild it only
#    when the wheel is gone (13 min, CPU only, 12 jobs), then rerun the install line
ls /home/jovyan/rlvr/wheels/flash_attn-*.whl && uv pip install /home/jovyan/rlvr/wheels/flash_attn-*.whl --no-deps
#    rebuild when the wheel is gone (needs nvcc 12.9 on the image; the venv must already hold torch):
#    uv pip install ninja packaging psutil pip && mkdir -p /home/jovyan/rlvr/wheels && \
#    MAX_JOBS=12 FLASH_ATTN_CUDA_ARCHS=90 TORCH_CUDA_ARCH_LIST=9.0 CUDA_HOME=/usr/local/cuda nice -n 10 \
#      python -m pip wheel flash-attn==2.8.3.post2 --no-build-isolation --no-deps --no-cache-dir -w /home/jovyan/rlvr/wheels \
#      && uv pip install /home/jovyan/rlvr/wheels/flash_attn-*.whl --no-deps
#    (the build can run in a tmux session while the first GPU job runs with sdpa; switch to flash afterwards)
python - <<'PY'
import torch; from flash_attn import flash_attn_func
q = torch.randn(2, 256, 16, 128, device="cuda", dtype=torch.bfloat16); k = torch.randn_like(q); v = torch.randn_like(q)
o = flash_attn_func(q, k, v, causal=True)
r = torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=True).transpose(1,2)
print("flash vs sdpa max abs diff", (o - r).abs().max().item(), "(expect < 0.05)")
PY

# 6. data (4.5 min; hashes b5afb83e6aacafcf and 45b3fa4862b11e2a are checked by the script)
bash scripts/phase2_rental.sh data 2>&1 | tee outputs/h100_data.log

# 7. the shim (recreate from the archive's outputs/bin/rlvr-sft if outputs/ is empty), then step 7 of path A
mkdir -p outputs/bin && export PATH=$PWD/outputs/bin:$PATH
```

Checkpoints and data a new disk would need back from the Hub (`hf download <repo> --include "<path>/*" --local-dir ...`):
SFT final at the root of Raviteja546/reasoning-rlvr-sft (ablations under ablations/), RFT final at the root of
Raviteja546/reasoning-rlvr-rft (seeds under seeds/), sample sets and the archives (session notes, evals, RFT data,
patches, shim) in the dataset Raviteja546/reasoning-rlvr-samples.

## Timings measured here, for planning

| job | time |
|---|---|
| environment build (setup_env.sh, cu129) | 25 min on 18 Oct; 2.5 min on 20 Oct with torch and triton pre-installed from the Hub wheels; 87 min on 20 Oct when download-r2.pytorch.org stalled |
| model downloads (two Qwen3 bases) | 5 min, in parallel with the build |
| flash-attention wheel build | 13 min (12 jobs) alone, 19 min at nice 10 beside a GPU job; install from the wheel 1 s; the wheel is on the Hub since 20 Oct |
| data step | 4.5 min |
| smoke eval | about 5 min incl. vLLM start |
| greedy think-tag eval, five sets | 8 min; three templates + gap + probe 22 min |
| SFT 23k x 2 epochs (flash + packing) | about 1.8 h (sdpa unpacked: 3.0 h) |
| RFT 11k kept examples, 1 epoch | 12 min |
| vLLM start to first token | about 1 min per rlvr-eval call |

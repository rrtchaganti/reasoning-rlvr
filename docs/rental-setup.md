# Rented GPU: setup, and the Phase 2 session (EXP-002 + SFT)

Written during Phase 1 (14 Oct 2026), extended for Phase 2 (15 Oct). Applies to every rented
session in Phases 1-7; the per-phase commands change, the setup does not.

## 1. What to rent

- One 80 GB card. Chosen 18 Oct 2026: a TIR H100 SXM 80 GB at INR 379.05/h (section 2d), 2.0x the
  A100 80 GB price for about 1.8x the speed on this workload (bf16 training about 2x, vLLM sampling
  1.6-2x), so about the same rupees per experiment at half the wall-clock. The earlier rule "H100 only
  within 1.5x" compared hours, not experiments; rupees per experiment is the right comparison.
  Everything here fits in 40 GB; the 80 GB is for the n = 64 KV cache and for training without
  gradient checkpointing games.
- Image / template: Ubuntu 22.04 or 24.04 with an **NVIDIA driver of the 580 series or newer**
  (`nvidia-smi` must show "CUDA Version: 13.x"). The vLLM wheel we pin (`vllm 0.31.0`) brings a CUDA 13
  build of torch; on a 535/550/570 driver it fails at the first CUDA call with "driver too old", exactly
  as it did on the laptop before the driver update. If no such image exists on the provider, run
  `RLVR_VLLM_CUDA=cu129 bash scripts/setup_env.sh`: it installs vLLM's `+cu129` release wheel of the
  locked version with torch from the cu129 index (torch 2.13 has no cu128 build; CUDA 12.9 binaries run
  on a 12.8 driver through minor-version compatibility); record the versions in `COMPUTE.md` because
  they differ from the lock file and the run_info of those runs will say so.
- Disk: 60 GB is enough for an eval-only session (weights 3.4 GB, HF cache, 64-sample outputs about
  2-3 GB per eval set). The Phase 2 + 3 training session peaks near 150 GB: fp32 master weights plus
  Adam states make every intermediate trainer checkpoint about 20 GB (section 2d has the breakdown);
  the TIR notebook has 300 GB.
- Provider chosen 16 Oct 2026: E2E Networks (INR billing). Route from 18 Oct 2026: TIR, the managed
  notebook platform, one notebook for the whole project (section 2d). The 17 Oct L4 session ran on a
  MyAccount node (section 2c, kept as the record). MyAccount images (checked 17 Oct on the L4:
  Ubuntu 24.04.3, driver 570.133.20, CUDA 12.8, toolkit 12.6 on `LD_LIBRARY_PATH`, system Python 3.12.3,
  no pip, no torch) take the cu129 route above.

## 2. First ten minutes on the box

```bash
git clone https://github.com/rrtchaganti/reasoning-rlvr.git && cd reasoning-rlvr
git checkout phase-2-sft                  # the phase branch in question
bash scripts/setup_env.sh                 # uv, Python 3.12, vllm, project; prints the Triton check
source ~/venvs/rlvr/bin/activate
python scripts/triton_check.py            # must print "triton compile + launch: ok"
export HF_TOKEN=...   WANDB_API_KEY=...    # from your password manager, never into the repo
export HF_HOME=~/.cache/huggingface
rlvr-perturb-gsm8k --seed 0               # regenerates outputs/data/gsm8k_perturbed.jsonl (hash must match data/manifests/)
```

Check the manifest hash printed at the end equals the one in `data/manifests/gsm8k_perturbed.json`
(`b5afb83e6aacafcf`); if it does not, stop, the eval set is not the same.

## 2b. Training extras (Phase 2 onward): flash attention and Weights & Biases

The SFT configs pack sequences (`packing: true`), which is only correct with a flash-attention backend
(`rlvr-sft` refuses to start otherwise; the reason is in `tests/test_ref_sft_loss.py`). Install one:

```bash
uv pip install flash-attn --no-build-isolation      # prebuilt wheel if one matches torch 2.13/cu13; else it compiles (20-40 min on an A100 box)
python -c "import flash_attn; print(flash_attn.__version__)"
```

No prebuilt wheel matches the locked torch 2.13 (Dao-AILab's 2.8.3.post2 stops at torch 2.8), so on
a rented box the wheel is built once per image and kept on the persistent disk; `setup_env.sh`
installs any `flash_attn-*.whl` it finds under `$RLVR_HOME/wheels` (or `RLVR_FLASH_WHEEL`). The
recipe that worked on the TIR H100 image (19 Oct 2026; nvcc 12.9 matches torch's CUDA; 13 minutes
on 12 jobs, CPU only, so it can run while a GPU job is on):

```bash
uv pip install ninja packaging psutil pip && mkdir -p "$RLVR_HOME/wheels"
MAX_JOBS=12 FLASH_ATTN_CUDA_ARCHS=90 TORCH_CUDA_ARCH_LIST=9.0 CUDA_HOME=/usr/local/cuda nice -n 10 \
  python -m pip wheel flash-attn==2.8.3.post2 --no-build-isolation --no-deps --no-cache-dir -w "$RLVR_HOME/wheels"
uv pip install "$RLVR_HOME"/wheels/flash_attn-*.whl --no-deps
```

Verify with the kernel test in `docs/e2e-session-handoff.md` (flash versus sdpa, max abs diff under
0.05) before the first training step. The kernels hub route (`uv pip install kernels`,
`attn_implementation: kernels-community/flash-attn`) returned 404/401 with the fine-grained token on
19 Oct; untested with a classic read token. Last resort: `RLVR_TRAIN_EXTRA="--set packing=false
--set attn_implementation=sdpa --set per_device_batch=16 --set grad_accum=4"` (correct, about 1.5-2x
slower because of padding; the Phase 2 runs of 18 Oct ran this way) and say so in `COMPUTE.md`.

`report_to: wandb` in the train configs needs `wandb login` once (or `WANDB_API_KEY` exported);
set `WANDB_PROJECT=reasoning-rlvr`. To run without it: `--set report_to=none`.

## 2c. E2E Networks MyAccount: one volume, nodes come and go (chosen 16 Oct 2026; superseded by 2d on 18 Oct 2026)

Kept as the record of the L4 session (17-18 Oct 2026). The block-volume request was withdrawn on
18 Oct; the active routine is section 2d.

**Facts that shape the routine** (E2E documentation, read 16 Oct 2026; re-check before the first
invoice):

- A powered-off GPU node on MyAccount keeps billing until it is **terminated**. End every session
  with Terminate, never Power off. (TIR, E2E's managed platform, says stopped nodes are free; we do
  not use TIR.)
- Block-storage volumes outlive the node: create once, attach to a running or powered-off node, detach,
  re-attach to another node (GPU and C3 CPU plans). Attach plus mount is a few minutes.
- Saved images need a powered-off node and are billed for the whole root disk; not used.
- Prices (INR per hour, on-demand, 16 Oct 2026): L4 24 GB 49; A30 24 GB 90; A40 48 GB 96; V100 32 GB
  100 (no bf16, unusable); L40S 48 GB 102; A100 40 GB 179; A100 80 GB 189; H100 not offered. The
  USD 100-170 budget is about INR 8,400-14,300.
- A cold start (uv, wheels, flash-attn, model and dataset downloads, data builds) costs about one
  A100-hour. It happens **once, on the L4**, with the volume attached; every later node mounts the
  volume and is ready in ten minutes.

**Layout on the volume** (mounted at `/mnt/rlvr`; fixed paths so the venv works on any node):

```
/mnt/rlvr/reasoning-rlvr   the repo clone; outputs/ inside it holds checkpoints, samples, logs
/mnt/rlvr/venv             the Python environment            (RLVR_VENV)
/mnt/rlvr/uv-python        uv's CPython, which the venv points at by path   (UV_PYTHON_INSTALL_DIR)
/mnt/rlvr/uv-cache         wheel cache                       (UV_CACHE_DIR)
/mnt/rlvr/hf               models and datasets               (HF_HOME)
```

`scripts/e2e_env.sh` exports these and activates the venv; `scripts/e2e_attach.sh` finds and
mounts the volume. Secrets never go on the volume: export `HF_TOKEN` and `WANDB_API_KEY` in the
shell of each session.

**Cold start, once, on an L4 (about 1.5 h, INR 75):**

1. MyAccount -> Storage -> Block Storage -> create 250 GB in the region the GPU nodes use.
2. Launch an L4 node from an Ubuntu 22.04 (or 24.04) CUDA image. Write the image name down: every
   later node uses the same image, so the venv's assumptions hold. Attach the volume (Block Storage ->
   Action -> Attach -> this node).
3. In the SSH session (the clone asks for your GitHub user name and a personal access token, the repo
   is private):

```bash
sudo apt-get install -y git tmux
git clone https://github.com/rrtchaganti/reasoning-rlvr.git /tmp/rr      # for the mount script only
sudo bash /tmp/rr/scripts/e2e_attach.sh         # finds the blank disk, asks YES before formatting, mounts /mnt/rlvr
git clone https://github.com/rrtchaganti/reasoning-rlvr.git /mnt/rlvr/reasoning-rlvr
cd /mnt/rlvr/reasoning-rlvr && git checkout phase-2-sft
source scripts/e2e_env.sh
nvidia-smi | head -4                            # read "CUDA Version": 13.x or 12.8
bash scripts/setup_env.sh                       # CUDA 13.x driver: exactly the laptop's environment
RLVR_VLLM_CUDA=cu129 bash scripts/setup_env.sh  # CUDA 12.8 driver (E2E, 17 Oct): vLLM +cu129 wheel, torch cu129
python scripts/triton_check.py                  # must print "triton compile + launch: ok"
# flash-attn: no prebuilt wheel exists for torch 2.13 (Dao-AILab 2.8.3.post2 stops at torch 2.8). The 17 Oct L4
# image's toolkit (12.6) did not match torch's CUDA (12.9); the TIR H100 image (nvcc 12.9) built the wheel in
# 13 min (section 2b) and setup_env.sh installs it from $RLVR_HOME/wheels on later starts. Without a wheel,
# training runs with RLVR_TRAIN_EXTRA="--set packing=false --set attn_implementation=sdpa" (correct, about
# 1.5-2x slower; COMPUTE.md says so).
bash scripts/phase2_rental.sh data              # perturbed set + SFT set, both hashes checked
hf download Qwen/Qwen3-1.7B-Base && hf download Qwen/Qwen3-0.6B-Base
rlvr-eval --model Qwen/Qwen3-1.7B-Base --evals gsm8k --template few_shot_cot --limit 20 --max-tokens 256 --out outputs/eval/e2e_smoke
```

The cu129 line is the one step not verified before the first node (the wheel
`vllm-0.31.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl` and `torch-2.13.0+cu129` both exist, checked
17 Oct). If `import vllm` or the Triton check fails afterwards, paste the error before trying anything
else. Do this on the L4, never on the A100. `setup_env.sh` writes its freeze to
`outputs/requirements-lock.<hostname>.txt` (ignored) and prints the torch/vllm lines that differ from
the tracked lock: those go into the `COMPUTE.md` row.

4. From the laptop, copy the local base reference so the session skips it:
   `scp -r outputs/eval/base_1.7b_think_tag_4096 <user>@<node-ip>:/mnt/rlvr/reasoning-rlvr/outputs/eval/`
5. Terminate the L4 (MyAccount -> Node -> Terminate). The volume stays.

**Every session:**

```bash
# MyAccount: launch the node (same image), attach the volume, ssh in
git clone https://github.com/rrtchaganti/reasoning-rlvr.git /tmp/rr && sudo bash /tmp/rr/scripts/e2e_attach.sh
cd /mnt/rlvr/reasoning-rlvr && source scripts/e2e_env.sh && git pull
export HF_TOKEN=...  WANDB_API_KEY=...
tmux new -s phase2
bash scripts/phase2_rental.sh 2>&1 | tee outputs/phase2_rental.log
```

Before Terminate: the report step's uploads finished; `metrics.json`, `report.md`, `run_info.json`,
`train_log.jsonl` and `outputs/reports/**` copied to the laptop (section 4); the `COMPUTE.md` row
written from the invoice line. An idle A100 hour is four L4 hours: write the next phase's code on
the laptop while a session runs, never on the node.

**Without the volume** (ticket pending, 17 Oct): the same layout on the root disk. `mkdir -p /mnt/rlvr`
and run the cold start as written; when the volume arrives, attach it, then as root:
`mkdir /mnt/rlvr_vol && mkfs.ext4 -q -L rlvr /dev/vdb && mount /dev/vdb /mnt/rlvr_vol && rsync -aH /mnt/rlvr/ /mnt/rlvr_vol/`,
then `umount /mnt/rlvr_vol && mv /mnt/rlvr /mnt/rlvr_rootdisk && bash scripts/e2e_attach.sh` (it mounts
the labelled disk at `/mnt/rlvr`). Paths do not change, so the venv keeps working. Until that copy exists,
everything on the node is lost if the node is terminated: push samples and checkpoints to the Hub first.

**Which node for which phase** (hours are the plan's A100 estimates):

| phase | work | node | hours | INR |
|---|---|---|---|---|
| once | cold start | L4 | 1.5 | 75 |
| 2 | EXP-002, sweep, main, SFT sampling, LoRA, scaling, report | A100 80 GB | 15 | 2,850 |
| 2 (option) | sweep, main, LoRA, scaling on the L40S if the speed check passes | L40S | 10 of the 15 | saves about 900 |
| 3 | RFT sampling + train | A100 80 GB | 3 | 570 |
| 4 | GRPO main, three ablations, seed repeat (vLLM colocated with an fp32-master 1.7B policy: 80 GB only) | A100 80 GB | 36 | 6,800 |
| 5 | DPO | A100 80 GB or L40S | 3 | 300-570 |
| 6 | test-time-compute sampling | A100 80 GB | 3 | 570 |
| 7 | 0.6B distillation | L4 or L40S | 2-3 | 100-300 |

About 42 A100-hours plus 10-15 cheaper hours: INR 9,300-10,700 before storage. The GRPO seed repeat
(INR 1,500) is the first run to drop.

**The L40S option, with its rule.** On an L40S (48 GB), start with
`RLVR_TRAIN_EXTRA="--set per_device_batch=2 --set grad_accum=32" bash scripts/phase2_rental.sh speed`:
it runs the first sweep run (10 % of the data) and `scripts/project_run_time.py` projects the main
run from the measured tokens per second. Rule, fixed before any run: stay on the L40S if the main
run projects to at most 8 hours (overnight, about INR 800 against INR 760-1,130 on the A100),
otherwise terminate, attach the volume to an A100 80 GB and continue with `bash scripts/phase2_rental.sh`;
the finished sweep run is reused either way (same effective batch of 64 sequences). The `RLVR_TRAIN_EXTRA`
line stays on for every training step on a 48 GB card.

## 2d. TIR notebook route: one notebook for the whole project (chosen 18 Oct 2026)

Replaces 2c. TIR is E2E's managed notebook platform (JupyterLab in a container, GPU plans billed per
minute). Chosen after the MyAccount block-volume request was withdrawn on 18 Oct 2026.

**Facts (TIR documentation, read 18 Oct 2026):**

- A stopped notebook is not billed for compute, only for storage; billing runs per minute while it
  is running. The workspace disk at `/home/jovyan` persists across Stop and Start and is lost on
  Delete. An ephemeral disk at `/home/user` (50 GB) is lost on restart: nothing goes there.
- The workspace disk bills per GB-month, stopped or not. Measured on the 19 Oct 2026 usage report:
  `Ceph_Block_Storage_NOTEBOOK` at INR 10 per GB-month on a billable size of 340 GB (the whole disk;
  no free allowance appeared on the invoice), i.e. about INR 112 a day or INR 3,400 a month while
  the notebook merely exists. A notebook kept between phases therefore costs about as much per
  month as nine H100-hours; the decision rule is in section 5. MyAccount block volumes attach to
  node plans only, not to notebooks.
- SSH is an option per notebook at creation (SSH key registered in TIR, "enable SSH"); without it the
  JupyterLab file browser moves files.
- Plan: NVIDIA H100 SXM 80 GB, 26 vCPU, 250 GB RAM, 300 GB workspace, INR 379.05 per hour. Image:
  "vLLM v0.15.1, CUDA 12.9, Transformers 4.57.6, Python 3.12.12, Ubuntu 22.04". The image's stack is
  not used: `setup_env.sh` builds the locked environment (vLLM 0.31.0+cu129, torch 2.13.0+cu129) in
  its own venv on a uv-managed Python, and `e2e_env.sh` unsets `PYTHONPATH` and `LD_LIBRARY_PATH` so
  nothing of the image leaks in. User `jovyan`, no root: no `apt`, no `sudo`.

**The disk is the volume.** One notebook lives for the whole project. Between phases it is Stopped,
not Deleted: the venv, the model and dataset caches, every `final/` checkpoint and every sample set
stay in place, and the next session starts in ten minutes without downloads. The off-node copies
(Hub uploads from the `report` steps, the laptop archive of metrics and reports) are made before
every Stop, so the disk is a convenience, never the only copy. The notebook is deleted after the
last phase's archive is verified on the laptop.

**Layout** (`/home/jovyan/rlvr`; `scripts/e2e_env.sh` picks it when `/home/jovyan` exists):

```
/home/jovyan/rlvr/reasoning-rlvr   the repo clone; outputs/ inside it holds checkpoints, samples, logs
/home/jovyan/rlvr/venv             the Python environment            (RLVR_VENV)
/home/jovyan/rlvr/uv-python        uv's CPython, which the venv points at by path   (UV_PYTHON_INSTALL_DIR)
/home/jovyan/rlvr/uv-cache         wheel cache                       (UV_CACHE_DIR)
/home/jovyan/rlvr/hf               models and datasets               (HF_HOME)
```

**Disk budget for the Phase 2 + 3 session** (peak, before pruning):

| what | GB |
|---|---|
| venv, uv Python, wheel cache | 15 |
| HF cache: two Qwen3 models, the OpenR1 subset, eval sets | 10 |
| `main` run: up to three intermediate checkpoints (fp32 weights 6.8 + Adam states 13.6 each) and `final` | 70 |
| sweep, LoRA, scaling, RFT and control runs: finals only (each run has fewer steps than its save interval) | 50 |
| samples and evals (n = 64 sets, the RFT n = 8 pool) | 3 |
| **peak** | **about 150** |

Pruning rule: after a session's `report` step and the verified archive, `rm -rf outputs/train/*/checkpoint-*`
(every `final/` stays). Listed in the handoff (section 4), so the node session does it without asking.
`df -h /home/jovyan` at the start of every session; above 240 GB used, prune before running.

**tmux without root.** `which tmux conda screen`: tmux present, use it; absent with conda present,
`conda install -y -c conda-forge tmux` (the image's conda is user-writable); else `screen`; failing
all three, `nohup bash scripts/<phase>_rental.sh <step> > outputs/<log> 2>&1 &` plus `tail -f`. The
session scripts are resumable, so a lost terminal costs nothing either way.

**Cold start (once per notebook life, 20-30 min, about INR 150):**

```bash
mkdir -p /home/jovyan/rlvr && cd /home/jovyan/rlvr
git clone https://github.com/rrtchaganti/reasoning-rlvr.git     # user name + read-only token at the prompt; never `git config credential.helper store`
cd reasoning-rlvr && git checkout <phase branch>
source scripts/e2e_env.sh
nvidia-smi | head -4                                            # "CUDA Version": 12.9 -> cu129 route; 13.x -> plain setup_env.sh
RLVR_VLLM_CUDA=cu129 bash scripts/setup_env.sh
source scripts/e2e_env.sh && python scripts/triton_check.py      # must print "triton compile + launch: ok"
export HF_TOKEN=...  WANDB_API_KEY=...                          # typed, never in a file
bash scripts/phase2_rental.sh data                              # perturbed set + SFT set, both hashes checked
hf download Qwen/Qwen3-1.7B-Base && hf download Qwen/Qwen3-0.6B-Base
rlvr-eval --model Qwen/Qwen3-1.7B-Base --evals gsm8k --template few_shot_cot --limit 20 --max-tokens 256 --out outputs/eval/h100_smoke
```

**Every later session:** Start the notebook, open a terminal, then

```bash
cd /home/jovyan/rlvr/reasoning-rlvr && source scripts/e2e_env.sh
git pull && git checkout <phase branch> && git pull
uv pip install -e . --no-deps                                   # registers the phase's new console scripts
git status --short && df -h /home/jovyan
export HF_TOKEN=...  WANDB_API_KEY=...
tmux new -s phaseN
bash scripts/phaseN_rental.sh 2>&1 | tee outputs/phaseN_rental.log
```

**End of a session:** uploads done, archive to the laptop verified (section 4), prune, then Stop from
the TIR console; hours and the storage line go into `COMPUTE.md`. If the H100 plan is unavailable at a
later Start, "Update Plan" to an A100 80 GB (same paths, same venv) and `export RLVR_NODE_RATE=189`.

**Which session for which phase (H100 hours, A100 estimates divided by about 1.8):**

| phase | work | H100 h | INR |
|---|---|---|---|
| 2 + 3 | Countdown, sweep, main, SFT sampling, LoRA, scaling, report; RFT sampling, data, train, evals, control, report | 15-18 | 5,700-6,800 |
| 4 | GRPO main, three ablations, seed repeat | 18-22 | 6,800-8,300 |
| 5 | DPO | 1.5-2 | 600-800 |
| 6 | test-time-compute sampling | 1.5-2 | 600-800 |
| 7 | 0.6B distillation (a cheaper plan via Update Plan if TIR allows it on a stopped notebook; unverified) | 1.5 | 300-600 |
| disk | 270 GB billable for about three weeks | | about 1,000 |

About 38-46 H100-hours: INR 15,000-18,000 with the disk, against the plan's INR 8,400-14,300 ceiling
(USD 100-170), once the Countdown measurement (87 M tokens) and the 2.0x card price are in. Drop list,
in order, decided at the end of the Phase 2 + 3 session from measured rates: the GRPO seed repeat
(about INR 1,500), Countdown on 200 problems instead of 500 for the SFT and RFT checkpoints (about
INR 1,000), the LoRA SFT run (about INR 700). Never the baselines or the CIs.

## 3. The Phase 2 session (about 15 hours of A100, resumable)

One script runs the whole session in order and skips every step whose output already exists, so a
dropped SSH connection costs nothing: reconnect, `tmux attach`, rerun the same command.

```bash
tmux new -s phase2
export HF_TOKEN=... WANDB_API_KEY=... WANDB_PROJECT=reasoning-rlvr
bash scripts/phase2_rental.sh 2>&1 | tee outputs/phase2_rental.log      # all steps
bash scripts/phase2_rental.sh sweep                                      # or one step at a time
```

Steps and rough A100 time: `data` (10 min, CPU: regenerates the perturbed set and the SFT set and
checks both hashes against the manifests; stop if either differs), `exp002` (base n = 64
sampling; the four 4-shot-CoT sets and the base think-tag reference already exist and are copied in, so only
Countdown runs: 87 M tokens, 5-8 A100-hours by the measurement in section 3a), `sweep` (1.5 h: three LR points on 10 % of
the data, dev evals, `pick_lr.py` writes `outputs/train/chosen_lr.txt`), `main` (4-6 h: the full
fine-tune with the chosen LR, three eval templates, perturbation gap, partial-prompt probe), `sample`
(2 h: n = 64 for the SFT checkpoint), `lora` (3 h), `scaling` (3 h: 25 % and 50 %), `report` (10
min: paired tables, figures, model card, uploads). If money runs short, drop `scaling`, then `lora`;
never `main` or its evals.

The base think-tag reference in `exp002` was already run locally on 15 Oct (3 h 55 min on the 4060;
`outputs/eval/base_1.7b_think_tag_4096/`). Copy that folder to the box before starting
(`scp -r outputs/eval/base_1.7b_think_tag_4096 box:reasoning-rlvr/outputs/eval/`); the step then skips it and
takes about 2.5 h. Budget note: with `max_model_len: 4096` the think-tag rows get 4096 minus the prompt
(3,865-3,994 new tokens); the SFT checkpoint configs use the same window on purpose, so the paired rows
share the cap per problem, and `rlvr-eval` prints the effective budget per eval (journal, section 14).

Set `HF_DRY_RUN=1` to run the report step without uploading; `HF_REPO_SFT` and `HF_REPO_SAMPLES`
override the repo names.

**Memory.** The full fine-tunes keep fp32 master weights (`load_dtype: float32`, bf16 autocast):
about 27 GB for weights, gradients and Adam states of the 1.7B before activations, which fits the
80 GB card with room for the 4 x 4096-token packed micro-batch. On a 40 GB card halve
`per_device_batch` and double `grad_accum` (`--set per_device_batch=2 --set grad_accum=32`). The LoRA
run carries about 5 GB extra for the tag-row training (fp32 copy, gradient and Adam moments of the whole
embedding matrix; the 0.6B smoke went from 1.6 to 4.5 GB peak).
Watch the first log lines of every training step: `[rlvr-sft] tag token '</think>'` must say
`"untrained": true, "initialised": true`, and `run_info.json` must end with `train_delta_norm` above
0 for both tags; otherwise the checkpoint cannot close its think block (journal, section 12).

## 3a. The EXP-002 part on its own (about 3 hours of A100)

```bash
tmux new -s sample                        # survive SSH drops
rlvr-eval --config configs/sample/base_1.7b_n64.yaml 2>&1 | tee outputs/samples/n64_main.log
rlvr-eval --config configs/sample/base_1.7b_n64_countdown.yaml 2>&1 | tee outputs/samples/n64_countdown.log
python scripts/plot_pass_at_k.py outputs/samples/base_1.7b_n64_few_shot_cot outputs/samples/base_1.7b_n64_countdown \
    --greedy outputs/eval/base_1.7b_few_shot_cot --out outputs/reports/figures/phase-1-pass-at-k.png
```

Token budget, measured on an L4 on 17 Oct 2026 (journal, section 17): the four 4-shot-CoT sets are
34.3 M generated tokens (mean sample length 154 on GSM8K, 359 on MATH-500, 926 on AIME24) and took
5 h 10 min at 3,134 down to 964 output tokens/s. Countdown is the expensive one: the base's think-tag
completions average about 2,700 tokens (two thirds hit the 4096 cap), so 500 x 64 is about 87 M tokens;
the L4 did 816 tokens/s on it (22 h), and an A100 80 GB with four times the KV cache should do
3,000-5,000 tokens/s, 5-8 hours. The rate depends on the completion length, not only on the GPU and the
window: estimate per set, and read the rate from each set's own progress line
(`python scripts/project_sampling.py <log> --from-metrics <nearest metrics.json> --rate <INR/h> --max-hours 8`
prints the projection and a continue/stop verdict). The `--no-batch-invariant` default in these configs is deliberate (throughput; the
stored samples are the reproducibility artefact).

## 3b. The Phase 3 session (about 5-7 hours of A100, right after Phase 2 on the same node)

`scripts/phase3_rental.sh` mirrors the Phase 2 script: `smoke` (file hand-offs on 20 problems),
`sample` (the SFT checkpoint samples its own training prompts, n = 8, T = 1.0, 1.5-3 h), `data`
(`rlvr-build-rft`: solve rates, the kept set, the learnable-prompt file for Phase 4, decontamination;
CPU), `train` (one epoch from `sft`, 0.5-1 h), `eval` (the three templates, gap, probe), `sample64`
(n = 64 for the RFT checkpoint; Countdown skippable with `RLVR_SKIP_COUNTDOWN64=1`), `control`
(EXP-010b, one more SFT epoch on as many fresh traces, about 1 h) and `report`. The sampling cap and
the ten-minute rule are in `docs/e2e-session-handoff.md`, section 7; the plan and the decisions in
`docs/phase-3-plan.md`.

## 4. Before shutting the box down

1. Copy back to the laptop (`rsync -av box:reasoning-rlvr/outputs/ outputs/ --include='*/' --include='metrics.json' --include='report.md' --include='run_info.json' --include='train_log.jsonl' --include='*.md' --include='*.png' --include='*.json' --exclude='*'`, or `scp` the same files): every `metrics.json`, `report.md`, `run_info.json`, `train_log.jsonl`, `outputs/reports/**`. Not the weights (on the Hub) and not the samples (on the Hub dataset repo). Commit the reports and figures on the phase branch.
2. Upload the sample JSONL files to the project's HF dataset repo with the commit hash in the README
   (`huggingface-cli upload`); the files are git-ignored on purpose.
3. Append `COMPUTE.md`: provider, GPU, wall-clock from `run_info.json` (`wall_clock_s`), the invoice
   line, tokens generated (sum of `tokens.mean x num_problems x n` per eval).
4. Note anything that differed from the lock file (driver, torch build) in the same row.

## 5. JupyterLab as the node UI, and Claude Code inside it (added 17 Oct 2026)

On a TIR notebook (section 2d) steps 1-3 fall away: the notebook is a JupyterLab server reached
through the TIR console, the terminal comes from its launcher, and
`curl -fsSL https://claude.ai/install.sh | bash` puts Claude Code in `~/.local/bin` (on PATH through
`e2e_env.sh`; its login in `~/.claude` persists on the workspace disk across Stop and Start). tmux as
in 2d. Step 5 is unchanged with the `/home/jovyan/rlvr` paths. The rest of this section is the
MyAccount route of the L4 session.

A browser file manager and terminals for the node, reached through an SSH tunnel so port 8888 is never
open to the internet. JupyterLab is installed as an isolated uv tool, outside the project venv, so the
lock file stays untouched. Claude Code is a native binary in `/root/.local/bin`, also outside the venv.

1. Laptop, PowerShell (keep this window open; it is the tunnel and a shell on the node):

   ```
   ssh -L 8888:localhost:8888 root@<node-ip>
   ```

2. Node, once per node:

   ```bash
   cd /mnt/rlvr/reasoning-rlvr && source scripts/e2e_env.sh
   curl -fsSL https://claude.ai/install.sh | bash && claude --version
   uv tool install jupyterlab
   ```

3. Node, start JupyterLab in its own tmux session (survives SSH drops), then print its URL:

   ```bash
   tmux new -d -s jupyter 'source /mnt/rlvr/reasoning-rlvr/scripts/e2e_env.sh && jupyter-lab --no-browser --ip=127.0.0.1 --port=8888 --allow-root --ServerApp.root_dir=/mnt/rlvr'
   sleep 5 && jupyter-lab list
   ```

4. Laptop browser: open the printed `http://localhost:8888/lab?token=...` URL.

5. In JupyterLab: File, New, Terminal. In that terminal:

   ```bash
   tmux new -s claude
   cd /mnt/rlvr/reasoning-rlvr && source scripts/e2e_env.sh && claude
   ```

   First time: type `/login` inside Claude Code, open the printed URL on the laptop, paste the code back.
   Detach with Ctrl-b then d; `tmux attach -t claude` returns. If the browser tab or the tunnel drops, the
   tmux sessions keep running; reconnect the tunnel and reopen the URL.

   `uv tool install` exposes only the package's own executables (`jupyter-lab`), not the generic `jupyter`
   launcher from jupyter-core, so the commands above use `jupyter-lab` (17 Oct 2026).

# Rented GPU: setup, and the Phase 2 session (EXP-002 + SFT)

Written during Phase 1 (14 Oct 2026), extended for Phase 2 (15 Oct). Applies to every rented
session in Phases 1-7; the per-phase commands change, the setup does not.

## 1. What to rent

- One A100 80 GB (or H100 80 GB if the hourly price is within 1.5x; everything here fits in 40 GB, the
  80 GB is for the n = 64 KV cache and later for training without gradient checkpointing games).
- Image / template: Ubuntu 22.04 or 24.04 with an **NVIDIA driver of the 580 series or newer**
  (`nvidia-smi` must show "CUDA Version: 13.x"). The vLLM wheel we pin (`vllm 0.31.0`) brings a CUDA 13
  build of torch; on a 535/550/570 driver it fails at the first CUDA call with "driver too old", exactly
  as it did on the laptop before the driver update. If no such image exists on the provider, run
  `scripts/setup_env.sh` and then re-pin torch with `uv pip install torch --torch-backend=cu128` plus
  the matching `vllm` wheel for cu128; record the versions in `COMPUTE.md` because they differ from the
  lock file and the run_info of those runs will say so.
- Disk: 60 GB is enough (weights 3.4 GB, HF cache, 64-sample outputs about 2-3 GB per eval set).
- Provider chosen 16 Oct 2026: E2E Networks (MyAccount GPU nodes, on-demand, INR billing). Section 2c
  has the volume layout, the prices and the per-session routine. Its A100 image reports CUDA 12.8,
  so the cu128 route in section 2 applies unless an image with a 580-series driver is offered.

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

If no wheel matches and the compile is not worth the time, use the kernels hub build instead:
`uv pip install kernels` and set `attn_implementation: kernels-community/flash-attn` in the train
config (the trainer accepts it as a flash variant). Last resort: `packing: false` in every train
config (correct, about 1.5-2x slower because of padding) and say so in `COMPUTE.md`.

`report_to: wandb` in the train configs needs `wandb login` once (or `WANDB_API_KEY` exported);
set `WANDB_PROJECT=reasoning-rlvr`. To run without it: `--set report_to=none`.

## 2c. E2E Networks: one volume, nodes come and go (chosen 16 Oct 2026)

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
UV_TORCH_BACKEND=cu128 bash scripts/setup_env.sh   # CUDA 12.8 driver instead: cu128 torch; note versions in COMPUTE.md
python scripts/triton_check.py                  # must print "triton compile + launch: ok"
uv pip install flash-attn --no-build-isolation  # section 2b; a compile here is 20-40 min at INR 49/h, not INR 189/h
bash scripts/phase2_rental.sh data              # perturbed set + SFT set, both hashes checked
hf download Qwen/Qwen3-1.7B-Base && hf download Qwen/Qwen3-0.6B-Base
rlvr-eval --model Qwen/Qwen3-1.7B-Base --evals gsm8k --template few_shot_cot --limit 20 --max-tokens 256 --out outputs/eval/e2e_smoke
```

The cu128 line is the one step not verified before the first node: if `import vllm` fails with a
CUDA or ABI error after it, install the cu128 variant of the same vLLM version
(`uv pip install vllm==0.31.0 --torch-backend=cu128`, see the vLLM GPU installation page for the
index of that release) and record the versions in `COMPUTE.md`. Do this on the L4, never on the A100.

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
checks both hashes against the manifests; stop if either differs), `exp002` (3 h: base n = 64
sampling + the base think-tag reference at 4096 tokens), `sweep` (1.5 h: three LR points on 10 % of
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

Token budget estimate: 2679 problems x 64 samples x about 250 tokens plus Countdown 500 x 64 x about
600 tokens, roughly 60 M generated tokens; vLLM on an A100 does 5-10 k tokens/s for a 1.7B model, so
2-3.5 hours. The `--no-batch-invariant` default in these configs is deliberate (throughput; the
stored samples are the reproducibility artefact).

## 4. Before shutting the box down

1. Copy back to the laptop (`rsync -av box:reasoning-rlvr/outputs/ outputs/ --include='*/' --include='metrics.json' --include='report.md' --include='run_info.json' --include='train_log.jsonl' --include='*.md' --include='*.png' --include='*.json' --exclude='*'`, or `scp` the same files): every `metrics.json`, `report.md`, `run_info.json`, `train_log.jsonl`, `outputs/reports/**`. Not the weights (on the Hub) and not the samples (on the Hub dataset repo). Commit the reports and figures on the phase branch.
2. Upload the sample JSONL files to the project's HF dataset repo with the commit hash in the README
   (`huggingface-cli upload`); the files are git-ignored on purpose.
3. Append `COMPUTE.md`: provider, GPU, wall-clock from `run_info.json` (`wall_clock_s`), the invoice
   line, tokens generated (sum of `tokens.mean x num_problems x n` per eval).
4. Note anything that differed from the lock file (driver, torch build) in the same row.

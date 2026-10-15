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
- Providers used before on this machine: none yet. RunPod and Lambda both offer 580-series images at
  the time of writing; check the template's driver line before paying.

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

Set `HF_DRY_RUN=1` to run the report step without uploading; `HF_REPO_SFT` and `HF_REPO_SAMPLES`
override the repo names.

**Memory.** The full fine-tunes keep fp32 master weights (`load_dtype: float32`, bf16 autocast):
about 27 GB for weights, gradients and Adam states of the 1.7B before activations, which fits the
80 GB card with room for the 4 x 4096-token packed micro-batch. On a 40 GB card halve
`per_device_batch` and double `grad_accum` (`--set per_device_batch=2 --set grad_accum=32`).
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

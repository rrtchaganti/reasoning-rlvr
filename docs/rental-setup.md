# Rented GPU: setup and the EXP-002 sampling session

Written during Phase 1 (14 Oct 2026). Applies to every rented session
in Phases 1-7; the per-phase commands change, the setup does not.

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
git checkout phase-1-baselines            # or the phase branch in question
bash scripts/setup_env.sh                 # uv, Python 3.12, vllm, project; prints the Triton check
source ~/venvs/rlvr/bin/activate
python scripts/triton_check.py            # must print "triton compile + launch: ok"
export HF_TOKEN=...   WANDB_API_KEY=...    # from your password manager, never into the repo
export HF_HOME=~/.cache/huggingface
rlvr-perturb-gsm8k --seed 0               # regenerates outputs/data/gsm8k_perturbed.jsonl (hash must match data/manifests/)
```

Check the manifest hash printed at the end equals the one in `data/manifests/gsm8k_perturbed.json`
(`b5afb83e6aacafcf`); if it does not, stop, the eval set is not the same.

## 3. The EXP-002 session (about 3 hours of A100)

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

1. Copy `outputs/samples/**/metrics.json`, `report.md`, `run_info.json` and the figure back to the
   laptop (`scp` or `rsync`); commit the metrics, report and figure on the phase branch.
2. Upload the sample JSONL files to the project's HF dataset repo with the commit hash in the README
   (`huggingface-cli upload`); the files are git-ignored on purpose.
3. Append `COMPUTE.md`: provider, GPU, wall-clock from `run_info.json` (`wall_clock_s`), the invoice
   line, tokens generated (sum of `tokens.mean x num_problems x n` per eval).
4. Note anything that differed from the lock file (driver, torch build) in the same row.

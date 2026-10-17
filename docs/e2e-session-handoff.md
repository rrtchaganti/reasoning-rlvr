# E2E node session: state and next steps (updated 18 Oct 2026, laptop session)

For the Claude Code session that runs on the next E2E node (the A100 80 GB, or an L4 if the volume
arrives first). The laptop session keeps the journal, the ledger and the commits; this file is how it
hands work over. Read `CLAUDE.md` first. The previous version of this file (17 Oct) drove the L4
session; the L4 node is terminated and its results are folded in below.

## 0. State on 18 Oct 2026

- Branch `phase-2-sft` on GitHub; a fresh clone has everything. The data-step fix is committed
  (b34c25b): `rlvr-build-sft` leaves the tracked manifest and report alone when the rebuilt set has
  the same hash and count, and writes its own copy to `outputs/data/sft/rebuild_manifest.json`. The
  L4 session's patch 0001 did the same thing and is superseded; do not apply it.
- Phase 2 data: perturbed GSM8K hash `b5afb83e6aacafcf`, SFT set hash `45b3fa4862b11e2a`
  (23,000 examples). Regenerated and verified on the L4 on 17 Oct 2026 (371 s).
- EXP-002 (base 1.7B, n = 64, T 0.6, top-p 0.95, 4096 max tokens), run on the L4 at commit ade4f6a:
  - the 4-shot-CoT config (gsm8k, gsm8k_perturbed, math500, aime24) is DONE: 5 h 10 min, 34.3 M
    generated tokens. The laptop holds `outputs/samples/base_1.7b_n64_few_shot_cot/` (140 MB); it is
    copied onto the next node (section 2, step 3) so the harness skips it.

    | eval | pass@1 (mean sample) | pass@64 [95% CI] | maj@64 | trunc % | greedy 4-shot (Phase 1) |
    |---|---|---|---|---|---|
    | gsm8k (1319) | 71.6 | 98.8 [98.2, 99.4] | 85.7 | 0.1 | 74.5 |
    | gsm8k_perturbed (830) | 64.5 | 90.2 [88.2, 92.2] | 77.0 | 0.1 | 67.8 |
    | math500 (500) | 36.3 | 87.8 [84.6, 90.4] | 57.6 | 4.3 | 41.4 |
    | aime24 (30) | 0.6 | 20.0 [6.7, 36.7] | 0.0 | 15.9 | 0.0 |

  - the Countdown config (think-tag, 500 problems x 64, `configs/sample/base_1.7b_n64_countdown.yaml`)
    is NOT DONE: stopped on the L4 at 2,368 of 32,000 sequences after 1 h 36 min (816 output tok/s,
    2.55 s per sequence, 21 h more projected). No partial samples exist; the rerun starts from zero.
    The decision of 17 Oct stands: `max_tokens: 4096` unchanged. Cost on the A100: see section 1.
  - the base think-tag reference `outputs/eval/base_1.7b_think_tag_4096/` exists on the laptop;
    `exp002` skips it when present (copy it too, section 2 step 3).
- Not started anywhere: `sweep` (EXP-005), `main` (EXP-006), `sample`, `lora` (EXP-007), `scaling`
  (EXP-008), `report`. All need the 80 GB card except `lora`, which fits 24 GB with
  `--set per_device_batch=2 --set grad_accum=32`.
- E2E volume ticket still open on 18 Oct. Without it the layout goes on the root disk at the same
  paths (`docs/rental-setup.md`, section 2c, "Without the volume"), and nothing survives Terminate:
  the Hub uploads in `report` and the archive of section 4 happen before the node goes.
- Environment route on E2E images (driver 570.133.20, CUDA 12.8): `RLVR_VLLM_CUDA=cu129 bash
  scripts/setup_env.sh` gives torch 2.13.0+cu129 and vLLM 0.31.0+cu129, the laptop's versions
  otherwise. No flash-attn: every training step carries `RLVR_TRAIN_EXTRA="--set packing=false"`.

## 1. Countdown at n = 64: the cost, and the decision the user takes before the A100 session

The base model's Countdown completions do not end: in the greedy think-tag run at the 4096 window,
65.6 % hit the cap and the mean length was 2,716 tokens (`outputs/eval/base_1.7b_think_tag_4096/
countdown/metrics.json`). At n = 64 that is about 32,000 x 2,700 = 87 M generated tokens, more than
the four finished sets together (34 M). The L4 managed 816 output tokens/s on it (its 15 GiB KV cache
holds 23 sequences of 6,144 tokens). The A100 80 GB holds about four times as many sequences and has
six times the memory bandwidth; a 3,000-5,000 tokens/s guess gives 5-8 hours, INR 950-1,500. The
`sample` step repeats Countdown for the SFT checkpoint; if its traces are shorter it is cheaper, if
they also loop it is the same again.

Options, in the order the laptop session recommends:

1. Keep the pre-registered config and measure: run Countdown first on the A100 (section 3, step 4
   runs it first anyway). After ten minutes read the "output: N toks/s" figure from the progress
   line; remaining hours = (32,000 - done) x 2,700 / N / 3,600. Above 8 hours: stop (Ctrl-C in the
   tmux pane, the step is resumable from zero) and ask the user.
2. Reduce the problem count (`--set countdown_n=200`), keeping n = 64 and the cap: the pass@k curve
   keeps its shape, the CI widens (500 -> 200 problems). Recorded as a deviation in `experiments.md`.
3. Cut `max_tokens` to 2048 (the L4 session's rejected patch 0002): halves the cost; the evidence
   that no correct greedy answer needed more than 711 tokens is for the base only, and the SFT
   checkpoint may need more. Rejected on 17 Oct; stays rejected unless the user says otherwise.

Whichever option runs is written into `outputs/<node>_session_notes.md` with the measured rate.

## 2. Laptop side, before the node is launched

1. Done on 18 Oct: samples in `outputs/samples/base_1.7b_n64_few_shot_cot/`, figure
   `outputs/reports/figures/phase-1-pass-at-k.png`, journal section 17, `COMPUTE.md` rows, EXP-002
   partial result in `experiments.md`, Phase 1 report addendum.
2. Done on 18 Oct: the L4 node's invoice (10 h 33 min, INR 516.95, 17 Oct 13:28 to 18 Oct 00:01 IST)
   is in `COMPUTE.md`.
3. When the node is up, from the laptop (PowerShell, repo folder; about 160 MB):

   ```powershell
   scp -r outputs/eval/base_1.7b_think_tag_4096 root@<node-ip>:/mnt/rlvr/reasoning-rlvr/outputs/eval/
   scp -r outputs/samples/base_1.7b_n64_few_shot_cot root@<node-ip>:/mnt/rlvr/reasoning-rlvr/outputs/samples/
   ```

   The target folders must exist first (`mkdir -p outputs/eval outputs/samples` on the node, after
   the clone).

## 3. Node session (A100 80 GB): steps, in order

1. Set up per `docs/rental-setup.md` section 2c: `git clone` of `phase-2-sft` (type the GitHub
   token), `RLVR_VLLM_CUDA=cu129 bash scripts/setup_env.sh`, `source scripts/e2e_env.sh`,
   `python scripts/triton_check.py`. Export `HF_TOKEN` and `WANDB_API_KEY` in the shell (never in
   a file). If the volume exists, `sudo bash scripts/e2e_attach.sh` first; otherwise
   `mkdir -p /mnt/rlvr` and the same paths on the root disk.
2. Data, downloads, smoke (about 20 min):

   ```bash
   cd /mnt/rlvr/reasoning-rlvr && source scripts/e2e_env.sh
   bash scripts/phase2_rental.sh data 2>&1 | tee outputs/a100_data.log      # hashes b5afb83e6aacafcf and 45b3fa4862b11e2a
   git status --short                                                       # must print nothing
   hf download Qwen/Qwen3-1.7B-Base && hf download Qwen/Qwen3-0.6B-Base
   rlvr-eval --model Qwen/Qwen3-1.7B-Base --evals gsm8k --template few_shot_cot --limit 20 --max-tokens 256 --out outputs/eval/a100_smoke
   ```

   If either hash differs, or `git status` is not clean after the data step, stop and report.
3. Confirm the two copies from the laptop (section 2 step 3) are in place:
   `ls outputs/samples/base_1.7b_n64_few_shot_cot/metrics.json outputs/eval/base_1.7b_think_tag_4096/metrics.json`.
   Without them `exp002` spends about 7 A100-hours regenerating what exists.
4. Inside tmux, EXP-002's Countdown half (section 1 decides how), then the rest of the phase:

   ```bash
   tmux new -s phase2
   cd /mnt/rlvr/reasoning-rlvr && source scripts/e2e_env.sh
   export RLVR_TRAIN_EXTRA="--set packing=false"
   bash scripts/phase2_rental.sh exp002  2>&1 | tee outputs/a100_exp002.log    # Countdown only; the other two pieces are skipped
   bash scripts/phase2_rental.sh sweep   2>&1 | tee outputs/a100_sweep.log
   bash scripts/phase2_rental.sh main    2>&1 | tee outputs/a100_main.log
   bash scripts/phase2_rental.sh sample  2>&1 | tee outputs/a100_sample.log
   bash scripts/phase2_rental.sh lora    2>&1 | tee outputs/a100_lora.log
   bash scripts/phase2_rental.sh scaling 2>&1 | tee outputs/a100_scaling.log
   bash scripts/phase2_rental.sh report  2>&1 | tee outputs/a100_report.log
   ```

   Each step is resumable (rerun the same command after a drop). Hours and memory notes:
   `docs/rental-setup.md` section 3. Watch the first lines of every training step:
   `[rlvr-sft] tag token '</think>'` must say `"untrained": true, "initialised": true`, and
   `run_info.json` must end with `train_delta_norm` above 0 for both tags.
5. Keep `outputs/a100_session_log.md` (where / why / ran / result / undo) and
   `outputs/a100_session_notes.md` (throughput per step, peak memory, surprises) as the L4 session
   did. Fixes found on the node become patch files in `outputs/patches/`, never edits to tracked files.

## 4. Before Terminate

1. `report` finished its Hub uploads (checkpoints and the sample sets, including
   `base_1.7b_n64_few_shot_cot` and `base_1.7b_n64_countdown`).
2. One archive to the laptop, the way the L4 session did it: every `metrics.json`, `report.md`,
   `run_info.json`, `train_log.jsonl`, `outputs/reports/**`, the two session files, the logs and
   `outputs/patches/`. Verified on the laptop before anything is deleted.
3. The user terminates the node from the console and notes the hours for `COMPUTE.md`.

## 5. When the volume arrives

As root, with the volume attached (it appears as `/dev/vdb`; check with `lsblk`):

```bash
mkdir -p /mnt/rlvr_vol && mkfs.ext4 -q -L rlvr /dev/vdb && mount /dev/vdb /mnt/rlvr_vol
rsync -aH --info=progress2 /mnt/rlvr/ /mnt/rlvr_vol/
umount /mnt/rlvr_vol && mv /mnt/rlvr /mnt/rlvr_rootdisk && bash /mnt/rlvr_rootdisk/reasoning-rlvr/scripts/e2e_attach.sh
```

Between steps only, never while a step writes to `outputs/`. Paths do not change, so the venv keeps
working. If the volume arrives before the A100 session, the cheaper order is the one in
`docs/rental-setup.md` section 2c: cold start on an L4 with the volume (1.5 h), terminate, then the
A100 mounts it and starts at step 3.4 within ten minutes.

## 6. Rules (summary of CLAUDE.md)

- Nodes never edit tracked files, commit or push; they hold a read-only token. `git status` before
  every step.
- Secrets only as environment variables typed into the shell of each session.
- Long jobs inside `tmux`; the session scripts are resumable.
- A node is either running a step or being terminated. Powered-off E2E nodes still bill.
- Ask the user before terminating, deleting outputs, or starting anything beyond the steps above
  that costs more than about one GPU-hour.
- Every git, environment and install action logged with where / why / ran / result / undo.

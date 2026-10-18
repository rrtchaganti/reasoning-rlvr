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
- Phase 3 (RFT) code is ready on branch `phase-3-rft` (18 Oct 2026) and runs on the same node right
  after Phase 2's `report` step: section 7. The node clones `phase-3-rft`, which contains every
  Phase 2 commit.
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
   runs it first anyway). After ten minutes, from a second tmux pane:
   `python scripts/project_sampling.py outputs/a100_exp002.log --from-metrics outputs/eval/base_1.7b_think_tag_4096/countdown/metrics.json --rate 189 --max-hours 8`
   (remaining sequences x the greedy run's mean length / this set's output rate; tqdm's ETA
   underestimates because the short completions finish first). On "STOP": Ctrl-C in the job's pane
   (the step restarts from zero later) and ask the user.
2. Reduce the problem count (`--set countdown_n=200`), keeping n = 64 and the cap: the pass@k curve
   keeps its shape, the CI widens (500 -> 200 problems). Recorded as a deviation in `experiments.md`.
3. Cut `max_tokens` to 2048 (the L4 session's rejected patch 0002): halves the cost; the evidence
   that no correct greedy answer needed more than 711 tokens is for the base only, and the SFT
   checkpoint may need more. Rejected on 17 Oct; stays rejected unless the user says otherwise.

Whichever option runs is written into `outputs/<node>_session_notes.md` with the measured rate.

## 2. Laptop side, before the node is launched

1. Done on 18 Oct: samples in `outputs/samples/base_1.7b_n64_few_shot_cot/`, figure
   `outputs/reports/figures/phase-1-pass-at-k.png`, journal section 17, `COMPUTE.md` rows, EXP-002
   partial result in `experiments.md`, Phase 1 report addendum. Also on 18 Oct: `scripts/project_sampling.py`
   (the ten-minute rule of section 1, option 1), and `--all-samples` modes of `perturbation_gap.py` and
   `compare_checkpoints.py` (paired pass@k; the `report` step runs them on the base and SFT samples).
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

1. Set up per `docs/rental-setup.md` section 2c: `git clone` then `git checkout phase-3-rft` (the
   Phase 2 code plus Phase 3; type the GitHub token), `RLVR_VLLM_CUDA=cu129 bash scripts/setup_env.sh`, `source scripts/e2e_env.sh`,
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

## 3b. Between the phases

Phase 2's `report` step must have finished (uploads done) before Phase 3 starts; copy the Phase 2
archive to the laptop (section 4, step 2) *before* step 7 so a node loss during Phase 3 costs only
Phase 3. Nothing else is needed from the laptop: every Phase 3 decision that depends on a Phase 2
number is a rule in section 7.

## 4. Before Terminate

1. `report` finished its Hub uploads (checkpoints and the sample sets, including
   `base_1.7b_n64_few_shot_cot` and `base_1.7b_n64_countdown`).
2. One archive to the laptop, the way the L4 session did it: every `metrics.json`, `report.md`,
   `run_info.json`, `train_log.jsonl`, `outputs/reports/**`, the two session files, the logs and
   `outputs/patches/`. Verified on the laptop before anything is deleted.
3. The user terminates the node from the console and notes the hours for `COMPUTE.md`.

## 7. Phase 3 on the same node, right after Phase 2 (added 18 Oct 2026)

Plan and pre-registration: `docs/phase-3-plan.md`, `experiments.md` EXP-009, EXP-010, EXP-010b.
Script: `scripts/phase3_rental.sh` (steps `sample`, `data`, `train`, `eval`, `sample64`, `control`,
`report`; resumable; `smoke` first). Needs `outputs/train/sft_1p7b/final`, the Phase 2 think-tag eval
(`outputs/eval/sft_1.7b_think_tag/metrics.json`) and, for the paired pass@k, the Phase 2 SFT samples.

```bash
cd /mnt/rlvr/reasoning-rlvr && source scripts/e2e_env.sh && git status --short   # clean, on phase-3-rft
export RLVR_TRAIN_EXTRA="--set packing=false"
bash scripts/phase3_rental.sh smoke  2>&1 | tee outputs/phase3_smoke.log     # 5 min: 20 problems, 3 steps
bash scripts/phase3_rental.sh        2>&1 | tee outputs/phase3_rental.log    # about 5-7 h (section 5 of the plan)
```

**Rules the node session applies on its own (no laptop round-trip):**

1. *Sampling cap* (plan, decision 2): the script picks 2048 or 4096 from the SFT think-tag eval's
   p90; `RLVR_RFT_MAX_TOKENS` overrides it, with the reason written in the notes.
2. *Ten-minute rule on `sample`*: after ten minutes,
   `python scripts/project_sampling.py outputs/phase3_rental.log --from-metrics outputs/eval/sft_1.7b_think_tag/metrics.json --eval math500 --rate 189 --max-hours 3`.
   On STOP: Ctrl-C, then rerun the sampling with a smaller pool and note it:
   `rlvr-eval --config configs/sample/sft_1.7b_rft_n8.yaml --max-tokens <cap> --pool-limit gsm8k_train=2000 --pool-limit math_train=2500`
   (then `bash scripts/phase3_rental.sh` continues from `data`). The pool change is a deviation from
   EXP-009's config and is reported as such.
3. *EXP-009 falsifier* (more than half of MATH levels 1-3 at rate 0): continue with the smaller kept
   set; the finding goes in the notes and the laptop rewrites the Phase 4 prompt set.
4. *Countdown at n = 64 for the RFT checkpoint*: skip it (`RLVR_SKIP_COUNTDOWN64=1`) when the SFT
   Countdown half was skipped in Phase 2, or when the first ten minutes project more than 4 hours.
5. *Control run (EXP-010b)*: runs by default; skip only when the session is above 24 node-hours
   in total, and say so.
6. *Session budget*: Phase 2 (about 15 h) plus Phase 3 (about 5-7 h) is about 22 A100-hours,
   INR 4,200. Ask the user before the node passes 26 hours.

**Freedom and limits.** The node session may take any decision needed to finish these steps within
`CLAUDE.md`: it reads logs, reruns resumable steps, changes environment variables and command-line
flags, writes patches into `outputs/patches/` for the laptop, and records every decision with its
evidence in `outputs/a100_session_notes.md`. It does not edit tracked files, commit or push, does
not start work outside these steps that costs more than about one GPU-hour, and does not terminate
the node. When a rule above does not cover a situation and the choice costs more than an hour or
changes a pre-registered config, it asks the user in the chat and waits.

**Hand-back for Phase 3** (same archive routine as section 4): `outputs/data/rft/manifest.json`,
`solve_rate.jsonl`, `learnable_prompts.jsonl`, `outputs/reports/phase-3-*`, the figures, every
`metrics.json`, `report.md`, `run_info.json`, `train_log.jsonl` under `outputs/eval/rft_*`,
`outputs/train/rft_1p7b`, `outputs/train/sft_1p7b_control`, `outputs/samples/rft_*` and
`outputs/samples/sft_1.7b_rft_n8` (metrics only; the samples go to the Hub), plus the session files.

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

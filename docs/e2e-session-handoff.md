# E2E node session: state and next steps (updated 18 Oct 2026, laptop session)

For the Claude Code session that runs on the TIR H100 notebook (E2E's managed platform; chosen
18 Oct 2026, `docs/rental-setup.md` section 2d). The laptop session keeps the journal, the ledger
and the commits; this file is how it hands work over. Read `CLAUDE.md` first. The 17 Oct version of
this file drove the L4 session on a MyAccount node; the L4 is terminated and its results are folded
in below. Hours quoted in `docs/phase-3-plan.md`, `experiments.md` and `docs/rental-setup.md`
section 3 are A100 80 GB units; this card does about 1.8x that speed (section 8 has the H100 figures).

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
    copied onto the node (section 2, step 3) so the harness skips it.

    | eval | pass@1 (mean sample) | pass@64 [95% CI] | maj@64 | trunc % | greedy 4-shot (Phase 1) |
    |---|---|---|---|---|---|
    | gsm8k (1319) | 71.6 | 98.8 [98.2, 99.4] | 85.7 | 0.1 | 74.5 |
    | gsm8k_perturbed (830) | 64.5 | 90.2 [88.2, 92.2] | 77.0 | 0.1 | 67.8 |
    | math500 (500) | 36.3 | 87.8 [84.6, 90.4] | 57.6 | 4.3 | 41.4 |
    | aime24 (30) | 0.6 | 20.0 [6.7, 36.7] | 0.0 | 15.9 | 0.0 |

  - the Countdown config (think-tag, 500 problems x 64, `configs/sample/base_1.7b_n64_countdown.yaml`)
    is NOT DONE: stopped on the L4 at 2,368 of 32,000 sequences after 1 h 36 min (816 output tok/s,
    2.55 s per sequence, 21 h more projected). No partial samples exist; the rerun starts from zero.
    The decision of 17 Oct stands: `max_tokens: 4096` unchanged. Cost on this card: see section 1.
  - the base think-tag reference `outputs/eval/base_1.7b_think_tag_4096/` exists on the laptop;
    `exp002` skips it when present (copy it too, section 2 step 3).
- Not started anywhere: `sweep` (EXP-005), `main` (EXP-006), `sample`, `lora` (EXP-007), `scaling`
  (EXP-008), `report`.
- Phase 3 (RFT) code is ready on branch `phase-3-rft` (18 Oct 2026) and runs on the same node right
  after Phase 2's `report` step: section 7. The node clones `phase-3-rft`, which contains every
  Phase 2 commit.
- Storage: no block volume (the MyAccount volume request was withdrawn on 18 Oct). The notebook's
  300 GB workspace disk at `/home/jovyan` is the persistent store for the whole project:
  `/home/jovyan/rlvr/{reasoning-rlvr,venv,uv-python,uv-cache,hf}`. It survives Stop and Start, not
  Delete; the ephemeral disk at `/home/user` is lost on restart and is never used. The Hub uploads
  and the laptop archive stay the off-node copies (section 4).
- Node: TIR "NVIDIA H100 SXM 80 GB" plan, 26 vCPU, 250 GB RAM, INR 379.05 per hour (2.0x the
  A100 80 GB; about 1.8x the speed on this workload, so about the same rupees per experiment at
  half the wall-clock). Pre-built image "vLLM v0.15.1, CUDA 12.9, Transformers 4.57.6, Python
  3.12.12, Ubuntu 22.04"; none of that stack is used. A container as user `jovyan`: no root, no
  `apt`, no `sudo`, no `e2e_attach.sh`.
- Environment route: `RLVR_VLLM_CUDA=cu129 bash scripts/setup_env.sh` builds the project venv
  (torch 2.13.0+cu129, vLLM 0.31.0+cu129, trl 1.15.0 pinned) exactly as on the L4; if `nvidia-smi`
  reads "CUDA Version: 13.x", plain `bash scripts/setup_env.sh`. No flash-attn either way: every
  training step carries `RLVR_TRAIN_EXTRA="--set packing=false"`.

## 1. Countdown at n = 64: the cost, and the decision the user takes before the session

The base model's Countdown completions do not end: in the greedy think-tag run at the 4096 window,
65.6 % hit the cap and the mean length was 2,716 tokens (`outputs/eval/base_1.7b_think_tag_4096/
countdown/metrics.json`). At n = 64 that is about 32,000 x 2,700 = 87 M generated tokens, more than
the four finished sets together (34 M). The L4 managed 816 output tokens/s on it (its 15 GiB KV cache
holds 23 sequences of 6,144 tokens). The H100 80 GB holds about four times as many sequences and has
eleven times the memory bandwidth; a 5,000-9,000 tokens/s guess gives 3-5 hours, INR 1,100-1,900.
The `sample` step repeats Countdown for the SFT checkpoint; if its traces are shorter it is cheaper,
if they also loop it is the same again.

Options, in the order the laptop session recommends:

1. Keep the pre-registered config and measure: run Countdown first (section 3, step 5 runs it first
   anyway). After ten minutes, from a second tmux pane or terminal:
   `python scripts/project_sampling.py outputs/h100_exp002.log --from-metrics outputs/eval/base_1.7b_think_tag_4096/countdown/metrics.json --max-hours 4`
   (remaining sequences x the greedy run's mean length / this set's output rate; the price comes
   from `RLVR_NODE_RATE`; tqdm's ETA underestimates because the short completions finish first;
   4 H100-hours is the same money as the 8 A100-hours of the earlier rule). On "STOP": Ctrl-C in the
   job's pane (the step restarts from zero later) and ask the user.
2. Reduce the problem count (`--set countdown_n=200`), keeping n = 64 and the cap: the pass@k curve
   keeps its shape, the CI widens (500 -> 200 problems). Recorded as a deviation in `experiments.md`.
3. Cut `max_tokens` to 2048 (the L4 session's rejected patch 0002): halves the cost; the evidence
   that no correct greedy answer needed more than 711 tokens is for the base only, and the SFT
   checkpoint may need more. Rejected on 17 Oct; stays rejected unless the user says otherwise.

Whichever option runs is written into `outputs/h100_session_notes.md` with the measured rate.

## 2. Laptop side, before the notebook is launched

1. Done on 18 Oct: samples in `outputs/samples/base_1.7b_n64_few_shot_cot/`, figure
   `outputs/reports/figures/phase-1-pass-at-k.png`, journal section 17, `COMPUTE.md` rows, EXP-002
   partial result in `experiments.md`, Phase 1 report addendum, `scripts/project_sampling.py` (the
   ten-minute rule of section 1, option 1), the `--all-samples` modes of `perturbation_gap.py` and
   `compare_checkpoints.py` (paired pass@k; the `report` step runs them on the base and SFT samples),
   the L4 invoice in `COMPUTE.md`, and this TIR plan (`docs/rental-setup.md` 2d).
2. Creating the notebook (TIR console): the H100 SXM 80 GB plan, the pre-built vLLM image, 300 GB
   workspace; if the form offers SSH, add the laptop's public key and enable it (gives `scp` for
   step 3, otherwise the JupyterLab file browser does the transfer). Write the plan name, the image
   name and the storage rate per GB-month shown on the form into `COMPUTE.md`.
3. When the notebook is up and the clone exists (section 3, step 2), copy the two folders (about
   160 MB) from the laptop. With SSH enabled (PowerShell, repo folder):

   ```powershell
   scp -r outputs/eval/base_1.7b_think_tag_4096 jovyan@<notebook-ip>:/home/jovyan/rlvr/reasoning-rlvr/outputs/eval/
   scp -r outputs/samples/base_1.7b_n64_few_shot_cot jovyan@<notebook-ip>:/home/jovyan/rlvr/reasoning-rlvr/outputs/samples/
   ```

   Without SSH: zip on the laptop, drag into JupyterLab, unzip on the node:

   ```powershell
   Compress-Archive -Path outputs/eval/base_1.7b_think_tag_4096 -DestinationPath base_ref.zip
   Compress-Archive -Path outputs/samples/base_1.7b_n64_few_shot_cot -DestinationPath base_n64.zip
   ```

   Drop both files into `rlvr/reasoning-rlvr/outputs/` in the JupyterLab file browser, then on the node
   `cd /home/jovyan/rlvr/reasoning-rlvr/outputs && unzip -q base_ref.zip -d eval && unzip -q base_n64.zip -d samples && rm base_ref.zip base_n64.zip`.
   The target folders must exist first (`mkdir -p outputs/eval outputs/samples` after the clone).

## 3. Node session (TIR H100): steps, in order

1. First terminal (JupyterLab launcher, "Terminal"). Facts, written into the notes:

   ```bash
   nvidia-smi | head -4; nproc; free -g | head -2; df -h /home/jovyan /home/user; echo "PYTHONPATH=$PYTHONPATH"
   which python conda tmux screen sudo
   ```

   `tmux`: present, use it. Absent and `conda` present: `conda install -y -c conda-forge tmux`
   (the image's conda is writable by `jovyan`; no root needed). Otherwise `screen`. Failing all
   three, run each step as `nohup bash scripts/phase2_rental.sh <step> > outputs/h100_<step>.log 2>&1 &`
   and follow it with `tail -f`; the scripts are resumable, so a lost terminal costs nothing.
2. Clone and build (once per notebook life, about 20-30 minutes, INR 130-190):

   ```bash
   mkdir -p /home/jovyan/rlvr && cd /home/jovyan/rlvr
   git clone https://github.com/rrtchaganti/reasoning-rlvr.git     # user name + read-only token at the prompt; never `git config credential.helper store`
   cd reasoning-rlvr && git checkout phase-3-rft                    # Phase 2 code plus Phase 3
   source scripts/e2e_env.sh                                       # RLVR_HOME=/home/jovyan/rlvr; unsets PYTHONPATH and LD_LIBRARY_PATH
   RLVR_VLLM_CUDA=cu129 bash scripts/setup_env.sh                  # plain `bash scripts/setup_env.sh` if nvidia-smi says CUDA 13.x
   source scripts/e2e_env.sh && python scripts/triton_check.py      # must print "triton compile + launch: ok"
   export HF_TOKEN=...  WANDB_API_KEY=...                          # typed into the shell, never into a file
   ```

   `setup_env.sh` prints the torch/vLLM lines that differ from the tracked lock; they go into the
   notes for the `COMPUTE.md` row.
3. Data, downloads, smoke (about 20 min):

   ```bash
   cd /home/jovyan/rlvr/reasoning-rlvr && source scripts/e2e_env.sh
   bash scripts/phase2_rental.sh data 2>&1 | tee outputs/h100_data.log      # hashes b5afb83e6aacafcf and 45b3fa4862b11e2a
   git status --short                                                       # must print nothing
   hf download Qwen/Qwen3-1.7B-Base && hf download Qwen/Qwen3-0.6B-Base
   rlvr-eval --model Qwen/Qwen3-1.7B-Base --evals gsm8k --template few_shot_cot --limit 20 --max-tokens 256 --out outputs/eval/h100_smoke
   ```

   If either hash differs, or `git status` is not clean after the data step, stop and report.
4. Confirm the two copies from the laptop (section 2 step 3) are in place:
   `ls outputs/samples/base_1.7b_n64_few_shot_cot/metrics.json outputs/eval/base_1.7b_think_tag_4096/metrics.json`.
   Without them `exp002` spends about 4 H100-hours regenerating what exists.
5. Inside tmux, EXP-002's Countdown half (section 1 decides how), then the rest of the phase:

   ```bash
   tmux new -s phase2
   cd /home/jovyan/rlvr/reasoning-rlvr && source scripts/e2e_env.sh
   export RLVR_TRAIN_EXTRA="--set packing=false"
   bash scripts/phase2_rental.sh exp002  2>&1 | tee outputs/h100_exp002.log    # Countdown only; the other two pieces are skipped
   bash scripts/phase2_rental.sh sweep   2>&1 | tee outputs/h100_sweep.log
   bash scripts/phase2_rental.sh main    2>&1 | tee outputs/h100_main.log
   bash scripts/phase2_rental.sh sample  2>&1 | tee outputs/h100_sample.log
   bash scripts/phase2_rental.sh lora    2>&1 | tee outputs/h100_lora.log
   bash scripts/phase2_rental.sh scaling 2>&1 | tee outputs/h100_scaling.log
   bash scripts/phase2_rental.sh report  2>&1 | tee outputs/h100_report.log
   ```

   Each step is resumable (rerun the same command after a drop). Hours and memory notes:
   `docs/rental-setup.md` section 3 (A100 units) and section 8 below. Watch the first lines of every
   training step: `[rlvr-sft] tag token '</think>'` must say `"untrained": true, "initialised": true`,
   and `run_info.json` must end with `train_delta_norm` above 0 for both tags.
6. Keep `outputs/h100_session_log.md` (where / why / ran / result / undo) and
   `outputs/h100_session_notes.md` (throughput per step, peak memory, surprises) as the L4 session
   did. Fixes found on the node become patch files in `outputs/patches/`, never edits to tracked files.

## 3b. Between the phases

Phase 2's `report` step must have finished (uploads done) before Phase 3 starts. Then the node
session packs the Phase 2 archive (the file list of section 4, step 2) and puts a copy off the node
without waiting for the user, so a node loss during Phase 3 costs only Phase 3:

```bash
cd /home/jovyan/rlvr/reasoning-rlvr && tar czf /home/jovyan/rlvr/h100_phase2_archive.tar.gz \
  $(find outputs -name metrics.json -o -name report.md -o -name run_info.json -o -name train_log.jsonl -o -name '*.log' -o -name '*.md' -o -name '*.png' | grep -v '^outputs/samples/.*/samples.jsonl') outputs/reports outputs/patches 2>/dev/null
ls -lh /home/jovyan/rlvr/h100_phase2_archive.tar.gz
hf upload rrtchaganti/reasoning-rlvr-samples /home/jovyan/rlvr/h100_phase2_archive.tar.gz archives/h100_phase2_archive.tar.gz --repo-type dataset
```

The session tells the user the file is ready; the user downloads it from the JupyterLab file browser
(`rlvr/` folder, right-click, Download) whenever convenient, and Phase 3 starts at once. Nothing else
is needed from the laptop: every Phase 3 decision that depends on a Phase 2 number is a rule in
section 7.

## 4. End of the session: archive, prune, Stop (not Delete)

1. `report` finished its Hub uploads (checkpoints and the sample sets, including
   `base_1.7b_n64_few_shot_cot` and `base_1.7b_n64_countdown`).
2. One archive for the laptop: every `metrics.json`, `report.md`, `run_info.json`,
   `train_log.jsonl`, `outputs/reports/**`, `outputs/data/rft/*.json*` (not the samples), the two
   session files, the logs and `outputs/patches/`. The node session packs it the way section 3b does
   (`/home/jovyan/rlvr/h100_session_archive.tar.gz`), uploads the copy to the Hub under `archives/`,
   prints the size, and tells the user. The user downloads the file from the JupyterLab file
   browser (or `scp` with SSH), checks the size matches, and extracts it next to the laptop repo
   (`../h100_session_archive/`) for the laptop session to fold in. Verified before anything is deleted.
3. Prune, after step 2 is verified: `rm -rf outputs/train/*/checkpoint-*` (the intermediate trainer
   checkpoints, about 20 GB each with their Adam states; every `final/` stays). This is a listed step
   and needs no further asking; deleting anything else does. `df -h /home/jovyan` before and after,
   in the notes.
4. The user **Stops** the notebook from the TIR console (compute billing stops; the disk keeps
   billing per GB-month) and notes the running hours for `COMPUTE.md`. Delete is for the end of the
   project only (section 5).

## 5. Between phases: the stopped notebook as the volume

- While stopped, the laptop session does the code, the docs, the merge and the next branch; the
  disk keeps the venv, the model and dataset caches, every `final/` checkpoint and every sample set.
- Next session: Start the notebook from the console (same plan; if the H100 is not available,
  "Update Plan" to an A100 80 GB if the console allows it: same paths, same cu129 venv, and
  `export RLVR_NODE_RATE=189` before sourcing the env script), open a terminal, then:

  ```bash
  cd /home/jovyan/rlvr/reasoning-rlvr && source scripts/e2e_env.sh
  git pull && git checkout <phase branch> && git pull             # read-only token at the prompt
  uv pip install -e . --no-deps                                    # registers new console scripts
  git status --short                                               # must print nothing
  df -h /home/jovyan                                               # above 240 GB used: prune first (section 4, step 3)
  export HF_TOKEN=...  WANDB_API_KEY=...
  ```

  Ten minutes, no downloads. Then the phase script inside tmux.
- The notebook is deleted only after the last phase's archive is verified on the laptop; the user
  does it from the console.

## 6. Rules (summary of CLAUDE.md)

- Nodes never edit tracked files, commit or push; they hold a read-only token. `git status` before
  every step.
- Secrets only as environment variables typed into the shell of each session.
- Long jobs inside `tmux` (or the fallback of section 3, step 1); the session scripts are resumable.
- A running notebook bills whether or not a step runs; a stopped one bills only its disk. The notebook
  is either running a step or stopped.
- Ask the user before stopping, deleting outputs other than section 4 step 3, or starting anything
  beyond the steps above that costs more than about one GPU-hour.
- Every git, environment and install action logged with where / why / ran / result / undo.

## 7. Phase 3 on the same node, right after Phase 2 (added 18 Oct 2026)

Plan and pre-registration: `docs/phase-3-plan.md`, `experiments.md` EXP-009, EXP-010, EXP-010b.
Script: `scripts/phase3_rental.sh` (steps `sample`, `data`, `train`, `eval`, `sample64`, `control`,
`report`; resumable; `smoke` first). Needs `outputs/train/sft_1p7b/final`, the Phase 2 think-tag eval
(`outputs/eval/sft_1.7b_think_tag/metrics.json`) and, for the paired pass@k, the Phase 2 SFT samples.

```bash
cd /home/jovyan/rlvr/reasoning-rlvr && source scripts/e2e_env.sh && git status --short   # clean, on phase-3-rft
export RLVR_TRAIN_EXTRA="--set packing=false"
bash scripts/phase3_rental.sh smoke  2>&1 | tee outputs/phase3_smoke.log     # 5 min: 20 problems, 3 steps
bash scripts/phase3_rental.sh        2>&1 | tee outputs/phase3_rental.log    # about 3-4 h on this card (5-7 A100-hours in the plan)
```

**Rules the node session applies on its own (no laptop round-trip):**

1. *Sampling cap* (plan, decision 2): the script picks 2048 or 4096 from the SFT think-tag eval's
   p90; `RLVR_RFT_MAX_TOKENS` overrides it, with the reason written in the notes.
2. *Ten-minute rule on `sample`*: after ten minutes,
   `python scripts/project_sampling.py outputs/phase3_rental.log --from-metrics outputs/eval/sft_1.7b_think_tag/metrics.json --eval math500 --max-hours 2`.
   On STOP: Ctrl-C, then rerun the sampling with a smaller pool and note it:
   `rlvr-eval --config configs/sample/sft_1.7b_rft_n8.yaml --max-tokens <cap> --pool-limit gsm8k_train=2000 --pool-limit math_train=2500`
   (then `bash scripts/phase3_rental.sh` continues from `data`). The pool change is a deviation from
   EXP-009's config and is reported as such.
3. *EXP-009 falsifier* (more than half of MATH levels 1-3 at rate 0): continue with the smaller kept
   set; the finding goes in the notes and the laptop rewrites the Phase 4 prompt set.
4. *Countdown at n = 64 for the RFT checkpoint*: skip it (`RLVR_SKIP_COUNTDOWN64=1`) when the SFT
   Countdown half was skipped in Phase 2, or when the first ten minutes project more than 2.5 hours.
5. *Control run (EXP-010b)*: runs by default; skip only when the session is above 18 node-hours
   in total, and say so.
6. *Session budget*: Phase 2 (about 11-13 h on this card) plus Phase 3 (about 3-4 h) plus setup and
   archiving (about 1 h) is about 15-18 H100-hours, INR 5,700-6,800. Ask the user before the
   notebook passes 20 running hours (INR 7,600).

**Freedom and limits.** The node session may take any decision needed to finish these steps within
`CLAUDE.md`: it reads logs, reruns resumable steps, changes environment variables and command-line
flags, writes patches into `outputs/patches/` for the laptop, and records every decision with its
evidence in `outputs/h100_session_notes.md`. It does not edit tracked files, commit or push, does
not start work outside these steps that costs more than about one GPU-hour, and does not stop or
delete the notebook. When a rule above does not cover a situation and the choice costs more than an
hour or changes a pre-registered config, it asks the user in the chat and waits.

**Hand-back for Phase 3** (same archive routine as section 4): `outputs/data/rft/manifest.json`,
`solve_rate.jsonl`, `learnable_prompts.jsonl`, `outputs/reports/phase-3-*`, the figures, every
`metrics.json`, `report.md`, `run_info.json`, `train_log.jsonl` under `outputs/eval/rft_*`,
`outputs/train/rft_1p7b`, `outputs/train/sft_1p7b_control`, `outputs/samples/rft_*` and
`outputs/samples/sft_1.7b_rft_n8` (metrics only; the samples go to the Hub), plus the session files.

## 8. Hours on this card

A100 80 GB estimates (`docs/rental-setup.md` section 3, `docs/phase-3-plan.md` section 5) divided by
about 1.8: bf16 training runs about 2x faster on an H100 SXM, vLLM sampling about 1.6-2x (memory
bandwidth 3.35 against 2.0 TB/s). Unverified until the first step; the notes record the measured
ratio per step so the later phases are re-estimated from data.

| step | A100 h (plan) | H100 h (expected) |
|---|---|---|
| setup, data, downloads, smoke | 0.7 | 0.7 |
| `exp002` Countdown, 87 M tokens | 5-8 | 3-5 |
| `sweep` | 1.5 | 1 |
| `main` + three evals + gap + probe | 4-6 | 2.5-3.5 |
| `sample` (SFT n = 64, two sets) | 2 | 1-1.5 |
| `lora` | 3 | 1.5-2 |
| `scaling` | 3 | 1.5-2 |
| `report`, archive | 0.5 | 0.5 |
| Phase 3 (`smoke` to `report`) | 5-7 | 3-4 |
| **session** | **25-32** | **15-18** |

The earlier "about 15 h" for Phase 2 predates the Countdown measurement; the A100 column is the
honest sum. Disk peak for the session is about 150 GB of the 300 (fp32 master weights plus Adam
states are about 20 GB per intermediate checkpoint; `docs/rental-setup.md` 2d has the breakdown).

## 9. Starting the node session (the user's steps, then the opening message)

The user does the minimum by hand: create the notebook, clone, install Claude Code, export the
secrets, start Claude Code inside tmux. The node session does everything else from section 3 step 1
onward (environment build included), so a failing wheel or a wrong driver is handled by the session
that can read the error.

Laptop, PowerShell in the repo folder, before launching (the two archives for section 2 step 3):

```powershell
Compress-Archive -Path outputs/eval/base_1.7b_think_tag_4096 -DestinationPath ..\base_ref.zip
Compress-Archive -Path outputs/samples/base_1.7b_n64_few_shot_cot -DestinationPath ..\base_n64.zip
```

Node, first JupyterLab terminal (the clone asks for the GitHub user name and the read-only token):

```bash
mkdir -p /home/jovyan/rlvr && cd /home/jovyan/rlvr
git clone https://github.com/rrtchaganti/reasoning-rlvr.git
cd reasoning-rlvr && git checkout phase-3-rft && mkdir -p outputs/eval outputs/samples
curl -fsSL https://claude.ai/install.sh | bash
source scripts/e2e_env.sh && claude --version
which tmux || conda install -y -c conda-forge tmux
tmux new -s claude
```

Inside the tmux window (the secrets are typed here and nowhere else):

```bash
cd /home/jovyan/rlvr/reasoning-rlvr && source scripts/e2e_env.sh
export HF_TOKEN=hf_...
export WANDB_API_KEY=...
claude
```

First run: `/login`, open the printed URL on the laptop, paste the code back. Meanwhile, in the
JupyterLab file browser, open `rlvr/reasoning-rlvr/outputs/` and upload `base_ref.zip` and
`base_n64.zip` from the laptop (the session unzips them). Detach from tmux with Ctrl-b then d;
`tmux attach -t claude` returns; a closed browser tab changes nothing.

Opening message to paste into Claude Code on the node:

```text
You are the Claude Code session ON the E2E TIR H100 notebook for the reasoning-rlvr project
(18 Oct 2026). The laptop session wrote your instructions: read CLAUDE.md, then
docs/e2e-session-handoff.md in full, before running anything.

State of this notebook: the repo is cloned at /home/jovyan/rlvr/reasoning-rlvr on branch
phase-3-rft; HF_TOKEN and WANDB_API_KEY are exported in this shell (never print them, never
write them to a file); base_ref.zip and base_n64.zip are (or will shortly be) in outputs/,
to be unzipped as handoff section 2 step 3 says; nothing else has been done.

Your job, in order: handoff section 3 (steps 1 to 6: facts, the environment build by the cu129
route, the data step with its two hash checks, the downloads, the smoke eval, the two copies,
then Phase 2 inside tmux), section 3b, section 7 (Phase 3, by its rules), section 4 (archive
and prune; I do the Stop myself). Countdown follows section 1, option 1, unless I say otherwise.

Rules: never edit tracked files, commit or push; `git status --short` before every step; long
jobs inside tmux; every action in outputs/h100_session_log.md (where / why / ran / result /
undo) and every measurement or surprise in outputs/h100_session_notes.md; ask me before
anything that costs more than about one GPU-hour beyond the handoff's steps, before deleting
anything other than the listed prune, and before the notebook passes 20 running hours. Report
failures with their output and name skipped steps as skipped.

Start by printing the facts of section 3 step 1 and telling me what the next hour holds.
```

While the session runs, the user approves its permission prompts (or allows the repeated ones
for the session), keeps the tab or tmux reachable, and at the end does section 4 step 4 (Stop)
and carries the two session files and the archive back to the laptop session.

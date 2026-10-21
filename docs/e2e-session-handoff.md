# Node session handoff: Phase 4 (GRPO, shape B) on a TIR H100 notebook, session 2

Written 20 Oct 2026 on the laptop for session 1 and rewritten 21 Oct for session 2, for the Claude
Code session that runs on the node. Session 1's handoff text is in the git history of this file;
its results are in `outputs/reports/phase-4.md` sections 1-9. The plan this session executes is
`docs/phase-4-plan.md`; the pre-registration is `experiments.md` EXP-014a, EXP-014b, EXP-014c. The
node never edits tracked files, commits or pushes (`CLAUDE.md`).

## 0. State on 21 Oct 2026

- **The notebook exists and is Stopped** (the user's Stop at about 01:20 UTC on 21 Oct after
  downloading the session 1 archive). Its disk holds the venv (torch 2.13.0+cu129, vLLM 0.31.0,
  TRL 1.15.0, flash-attn 2.8.3.post2), the three wheels under `/home/jovyan/rlvr/wheels`, the HF
  cache, the SFT and RFT finals, the sample sets, the rebuilt data (hashes checked on 20 Oct), the
  `rft` dev eval, the two Phase 3 ablation checkpoints, the debugging run, the main run and the
  seed run with their checkpoints and evals, and the n = 32 samples. Session 2 therefore starts
  with path A of `docs/h100-next-session.md` (about 5 minutes, no build). If the notebook was
  Deleted instead, path B with the Hub wheels (about 15 minutes to the first GPU job).
- **On the Hub (namespace `Raviteja546`, public):** `reasoning-rlvr-grpo` (EXP-012 final at the
  root with a placeholder card; `seeds/seed1`), `reasoning-rlvr-rft` (RFT final at the root;
  `seeds/`, `controls/`, `ablations/nofilter`, `ablations/random2`), `reasoning-rlvr-sft`, and the
  dataset `reasoning-rlvr-samples` (sample sets, `grpo_1.7b_n32_think_tag_t06`, `wheels/` with
  `sha256.txt`, the session archives under `archives/`).
- **In the repo (branch `phase-4-grpo`, at or after fb856a8):** the session 1 results folded in
  (`outputs/reports/phase-4.md`, the paired tables `outputs/reports/phase-4-compare-*.md`, the
  figures); the config and the preflight now carry `per_device_batch: 16` and
  `vllm_gpu_memory_utilization: 0.30` (the session 1 values), so `RLVR_GRPO_EXTRA` is no longer
  needed and the preflight would refuse a config that differs; `scripts/phase4_rental.sh` exports
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` itself, excludes `controls/` and `ablations/`
  from the RFT download, and runs a paired compare only when its file is missing; the hacking
  watch's repeated-answer flag counts repeated *incorrect* answers; `rlvr-build-rft` writes its
  report next to the data. **The node must `git pull --ff-only` before anything else** (the user
  types the token); every run of session 2 is on the pulled commit.
- **Budget:** the project has spent about INR 16,500 (sessions to 21 Oct plus the disk) of the
  INR 22,000 ceiling. Session 2 is about 12 node-hours, INR 4,550, at the measured rates: about
  INR 700 of headroom if the GPU is never idle. No run is repeated without asking; the drop order is
  plan section 8 (A3, then A1). Rate INR 379.05 per hour; the disk about INR 107 a day.
- **The noise row the ablations are read against** (seed 1 minus seed 0, paired, think-tag greedy):
  GSM8K 0.3, perturbed GSM8K 0.2, MATH-500 4.8, Countdown 0.6 (`outputs/reports/phase-4.md`,
  section 5). An ablation delta smaller than these is "within noise" whatever its sign.

## 1. Restarting the node (the user, then the agent)

1. TIR console: Start the Stopped notebook (same SSH key). Nothing else is created.
2. In the notebook terminal, before anything else (the tmux server inherits this shell's
   environment, so the exports come first):

   ```bash
   export HF_TOKEN=hf_...            # typed, never into a file; BEFORE tmux starts
   export WANDB_API_KEY=...
   tmux start-server; tmux set-environment -g HF_TOKEN "$HF_TOKEN"; tmux set-environment -g WANDB_API_KEY "$WANDB_API_KEY"
   cd /home/jovyan/rlvr/reasoning-rlvr && git status --short && git pull --ff-only && git log --oneline -1
   source scripts/e2e_env.sh
   python -c "import torch, vllm, flash_attn; print(torch.__version__, vllm.__version__, flash_attn.__version__)"
   python -c "import importlib.util as u; assert u.find_spec('torchcodec') is None, 'run: uv pip uninstall torchcodec'"
   python scripts/triton_check.py                # must print: triton compile + launch: ok
   tmux new -s claude
   ```

   The status line prints nothing (clean tree); the pull prompts for the user name and the read-
   only token; the log line shows fb856a8 or later. Inside tmux:
   `cd /home/jovyan/rlvr/reasoning-rlvr && source scripts/e2e_env.sh && claude --permission-mode auto`
   (the skip-permissions flag is refused for root; auto mode is what session 1 ran in), then the
   opening message of section 7. If `claude` is missing after a Delete: `curl -fsSL
   https://claude.ai/install.sh | bash` first, then `/login`.
3. If the notebook was Deleted: handoff section 1 of 20 Oct (in git history) for the clone and the
   install, then path B of `docs/h100-next-session.md` with the wheels step (3b), then `setup`
   downloads everything the disk lacks from the Hub.

## 2. Session 2 on the node, in order

Every step is one `bash scripts/phase4_rental.sh <step>` call inside its own tmux session, chained
on the previous step's `EXIT=` line in its log (section 3, rule 4), with the GPU never idle between
them. Estimates are session 1's measured rates (56 s per GRPO step, 3 h 18 min per 200-step run
with its four dev evals, 8 min per five-set greedy eval).

| # | step | what | estimate | notes for the session |
|---|---|---|---|---|
| 1 | section 1 | Start, secrets, pull, env check | 5 min | the pull is the user's; nothing runs before `git log` shows the fold-in commit |
| 2 | `setup` | skips what is on the disk; re-asserts the manifest hashes and the pinned eval subsets | 2 min | a mismatch means the harness changed: stop and ask |
| 3 | `a1` | EXP-014a: preflight, 200 steps with `kl_beta=0`, four dev evals | 3.3 h | the ten-minute rule at step 10 against 3.3 h (bound 5.0 h); the KL is logged, not penalised |
| 4 | `a2` | EXP-014b: `dynamic_sampling=true`, `epsilon_high=0.28`; over-generates 2x the prompts and keeps the 32 informative groups | 4.0 h | the sampling half is about 1.3-1.6x session 1's; the bound is 6.0 h; `dynamic_sampling_log.jsonl` is a result (groups generated, informative, filled) |
| 5 | `a3` | EXP-014c: `loss_type=dr_grpo`, `scale_rewards=none` | 3.3 h | same rule; the length curve is the result |
| 6 | `evals` | the five-set greedy think-tag eval of each ablation's final checkpoint, one process per checkpoint | 25 min | main's evals and samples exist and are skipped |
| 7 | `report` | the paired compares of each ablation against main and against `rft` (only those whose file is missing: the laptop computed the rest), the model card from this report's tables, the uploads (main's card and samples; the ablation finals under `ablations/`) | 20 min | `outputs/eval/rft_1.7b_think_tag` is not on this disk; the compares against `rft` are skipped there and the laptop computes them from the archive, as in session 1 |
| 8 | section 5 | hand-back, archive, then the user's **Delete** (session 2 is the last) | 20 min | |

About 12 node-hours, INR 4,550. The order a1, a2, a3 is the plan's; if time runs short the drop
order is A3 first, then A1 (plan section 8); A2 stays because its question (does the prompt filter
leave enough variance) is the one the session 1 curves raised (30 % zero-variance groups throughout).
If the user is away, the session continues through the table on its own; it stops and asks only
where the table says so, or where a rule of section 3 is about to be broken.

## 3. Rules for the node session (the user's, 19 Oct 2026; `docs/phase-4-plan.md` section 7)

1. **Check before launch, every time.** `scripts/phase4_preflight.py --run <row>` runs before every
   training step (the session script does it) and its output goes into the notes. A `FAIL` line
   stops the step; the session asks the user rather than fixing the config to make it pass. The
   knob table now includes `per_device_batch` 16 and `vllm_gpu_memory_utilization` 0.30.
2. **Same subsets as the method compared against.** The pinned hashes (GSM8K `58c8ded1c644f332`,
   MATH-500 `5ca9795b45227236`, Countdown `9698ed78ea4e0491`) are checked by `--subsets` in `setup`.
   Every ablation is evaluated on the same five full sets as the main run, same configs.
3. **The whole GPU, every step.** One `rlvr-eval` process per checkpoint with every set in its
   config; no eval of one set "for a look". The training micro-batch is 16 completions for every
   run alike (session 1's value, pinned); it does not change.
4. **No idle gaps.** Steps run as `tmux new -d -s <step> -c /home/jovyan/rlvr/reasoning-rlvr -e HF_TOKEN="$HF_TOKEN" "source scripts/e2e_env.sh && bash scripts/phase4_rental.sh <step> > outputs/h100_p4_<step>.log 2>&1; echo EXIT=\$? >> outputs/h100_p4_<step>.log"`
   and the next step starts when the previous log's last line is `EXIT=0`. The marker is written
   by that launcher's `echo EXIT=$?` after the script has exited (`set -e` turns any failed command
   into a non-zero code), never by the script itself. The watcher is
   `while ! tail -n 1 outputs/h100_p4_<step>.log | grep -q '^EXIT=[0-9]*$'; do sleep 60; done`
   (60-90 s; it costs the GPU nothing): the last line, anchored. `EXIT=` with anything but 0 means
   stop and ask. Never wait on a tmux session name. When a step ends, the notes get its actual time
   against the table above; idle minutes are counted.
5. **The ten-minute rule on every GRPO run.** At step 10 the log's step time gives the projection;
   above 1.5x the table's estimate the run is stopped (`tmux kill-session -t =<step>`) and the user
   asked. Inside it, the run continues without a question. An OOM at micro-batch 16: one relaunch
   at 8 without asking (the user's standing permission of 20 Oct), recorded; an OOM at 8 stops the
   GRPO runs and the session writes the hand-back with what exists.
6. **Decisions by written rules.** The pre-registered falsifiers, the noise row of section 0 and the
   drop order (plan section 8) decide; anything they do not cover and that costs more than an hour
   is a question to the user. A GRPO run whose reward log shows the hacking flags continues to its
   end and is reported as such with its incident log; it is not restarted with a changed reward.
   The repeated-answer flag now needs repeated *incorrect* answers; a run that raises it is read
   with the other four flags before it is called hacking.
7. **`CLAUDE.md`:** no edits to tracked files (fixes go to `outputs/patches/` as patch files), no
   commits or pushes, secrets in the shell only, every action in `outputs/h100_p4_session_log.md`
   (where / why / ran / result / undo; continue session 1's numbering) and every measurement in
   `outputs/h100_p4_session_notes.md`, archive before Delete.

## 4. Measured rates (session 1, 20 Oct 2026)

| unit of work | measured | used for |
|---|---|---|
| GRPO step, shape B, micro-batch 16, MATH-heavy mix | 56 s (60 s at step 10, 54 s on a GSM8K mix) | every 200-step run: 3 h 08 min + 9 min of dev evals |
| dynamic sampling (A2) | not measured; 2x the completions per step, so 1.3-1.6x on the sampling half | A2 estimate 4.0 h, bound 6.0 h |
| five-set greedy think-tag eval | 7.5-8 min | each ablation's final checkpoint |
| three-template eval set + gap + probe | 19 min | (session 1 only) |
| n = 32, T = 0.6 on two 300-problem subsets | 28 min | (session 1 only) |
| RFT-size training | 12-14 min | (session 1 only) |
| path A restart / path B with the Hub wheels | 5 min / about 15 min | step 1 |
| Hub upload of a 6.5 GB final | about 4 min | the three ablation finals |

## 5. End of the session: hand-back, archive, uploads, Delete

1. Uploads (inside `report`): the main checkpoint's card to `Raviteja546/reasoning-rlvr-grpo`; the
   ablation finals under `ablations/{a1_nokl,a2_dynamic,a3_drgrpo}` of the same repo, each with its
   `run_info.json`, `train_log.jsonl`, `reward_log.jsonl` (and `dynamic_sampling_log.jsonl`) inside
   `final/`; `hf upload` in one tmux session per repo.
2. `outputs/h100_p4_handback.md` continued (session 2 section): the step table with actual times,
   every run's headline numbers with CIs and the file they came from, the preflight outputs, the
   per-step curves' summary at steps 1, 50, 100, 150, 200, the hacking-watch verdicts, A2's
   dynamic-sampling counts (groups generated, informative, filled) and its sampling time per step
   against main's, A3's length and truncation per step against main's, Hub and W&B URLs,
   `pip freeze`, `df -h`, node-hours and rupees.
3. The archive: every `metrics.json`, `report.md`, `run_info.json`, `train_log.jsonl`,
   `reward_log.jsonl`, `dynamic_sampling_log.jsonl`, `prompt_ids.json`, the per-sample files of
   every greedy eval, `outputs/reports/**`, `cards/grpo.md`, `outputs/patches/`, the session files,
   the hand-back, the logs; `tar czf /home/jovyan/rlvr/h100_p4_session2_archive.tar.gz`, its sha256
   and `tar tzf | wc -l` printed, a copy uploaded to the samples dataset under `archives/`. The
   session 1 lesson: the hand-back is finished before the archive is built, or the addendum has to
   be fetched separately.
4. The user downloads it, checks the hash, then **Deletes** the notebook: session 2 is the last
   session of Phase 4 and the disk bills about INR 107 a day. The agent never Stops or Deletes on
   its own: it finishes step 3, prints the hash, and asks.

## 6. What the laptop does afterwards

The paired compares of each ablation against `rft`, the ablation table, sections 10 onward of
`outputs/reports/phase-4.md`, the registry entries EXP-014a-c, the ledger, the journal, the
engineering notes, the PR body. Nothing of that is the node's.

## 7. The opening message for the node session

```
You are the node session of the reasoning-rlvr project, on the TIR H100 notebook that ran session 1,
branch phase-4-grpo. Read CLAUDE.md, then docs/e2e-session-handoff.md (this file, rewritten for
session 2) and docs/phase-4-plan.md; outputs/reports/phase-4.md holds session 1's results and the
noise row. Continue outputs/h100_p4_session_log.md (where / why / ran / result / undo, numbering on
from row 48) and outputs/h100_p4_session_notes.md (measurements, actual-vs-estimate after every
step, surprises) from the first command. Secrets are in the shell and in tmux's global environment;
never write them anywhere.

State: the repo at /home/jovyan/rlvr/reasoning-rlvr is on phase-4-grpo at the commit I just pulled
(git log shows it); the venv, wheels, data, checkpoints and session 1's outputs are on the disk;
nothing is running. Confirm with git status (clean), the env check of handoff section 1, and
nvidia-smi (idle).

Job: session 2 of handoff section 2, steps 2 to 8, in order, each inside its own tmux session
chained on the previous step's EXIT line (anchored watcher, rule 4), with the GPU never idle. Rules:
handoff section 3, all seven, and the ten-minute rule with the bounds in the table. Stop and ask me
where the table says so, where a preflight fails, where a hash does not match, or where a rule would
be broken; otherwise continue without questions. Start with `setup`, tell me what it skipped and
what it checked, then launch `a1` and give me the step-10 projection.
```

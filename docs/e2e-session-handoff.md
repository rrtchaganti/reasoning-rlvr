# Node session handoff: Phase 4 (GRPO, shape B) on a TIR H100 notebook

Written 20 Oct 2026 on the laptop for the Claude Code session that runs on the node. The previous
handoff (Phases 2 and 3, 18-19 Oct) is in the git history of this file; its results are in
`outputs/reports/phase-2.md` and `phase-3.md`. The plan this session executes is
`docs/phase-4-plan.md`; the pre-registration is `experiments.md` EXP-010g, EXP-010h, EXP-012,
EXP-013, EXP-014a-c. Two sessions: session 1 (setup, the two Phase 3 ablations, the debugging run,
the main run, the seed repeat, the evals) and session 2 (the three ablations, the report). The node
never edits tracked files, commits or pushes (`CLAUDE.md`).

## 0. State on 20 Oct 2026

- **No notebook exists.** The 18-19 Oct notebook was deleted; every session starts from the Hub
  (path B of `docs/h100-next-session.md`, about 45 minutes, GPU idle for 25 of them).
- **On the Hub (namespace `Raviteja546`, public):** `reasoning-rlvr-sft` (SFT final at the root,
  ablations under `ablations/`), `reasoning-rlvr-rft` (RFT final at the root, `seeds/`), the dataset
  `reasoning-rlvr-samples` (every n = 64 and n = 32 sample set, the EXP-009 samples
  `sft_1.7b_rft_n8`, the two session archives under `archives/`).
- **In the repo (branch `phase-4-grpo`):** `rlvr-grpo` (`src/rlvr/train/grpo.py`), the two builder
  options of `rlvr-build-rft`, `scripts/phase4_preflight.py`, `scripts/phase4_rental.sh`,
  `configs/train/grpo_1p7b.yaml`, the GRPO eval and sample configs, `data/manifests/rft_iter1.json`
  (the hashes the rebuilt RFT data must reproduce: kept set `f44edf33a26c55fc`, learnable prompts
  `281bdfe31176411b`).
- **Done on the laptop before any node time:** the correctness check of the reference GRPO loss
  against TRL (`scripts/grpo_correctness_check.py`; its table is in the Phase 4 journal). If the
  journal has no such table, stop and ask: the node does not run on unchecked machinery.
- **Budget:** the project has spent about INR 11,450; the ceiling is INR 22,000, set by the user on
  20 Oct (`docs/phase-4-plan.md`, section 5). Phase 4 at shape B is about 26 node-hours, INR 9,800,
  which leaves about INR 700 (under two H100-hours) of headroom for the whole phase: no run is
  repeated without asking. Rate INR 379.05 per hour; the disk INR 112 a day.

## 1. Laptop side, before the notebook is created (the user)

1. The correctness check has run locally and its table is in `docs/phase-4-journal.md`.
2. The ceiling for Phase 4 is INR 22,000, written in `docs/phase-4-plan.md`, section 5 (done 20 Oct).
3. Create the notebook as on 18 Oct: TIR, "NVIDIA H100 SXM 80 GB", the same pre-built image
   (vLLM v0.15.1, CUDA 12.9, Python 3.12.12), SSH enabled with the existing key
   `C:\Users\chaga\.ssh\tir_rlvr.pub`. The disk size does not matter beyond 150 GB.
4. In the notebook terminal, before anything else:

   ```bash
   export HF_TOKEN=hf_...            # typed, never into a file; BEFORE tmux starts
   export WANDB_API_KEY=...
   tmux start-server; tmux set-environment -g HF_TOKEN "$HF_TOKEN"; tmux set-environment -g WANDB_API_KEY "$WANDB_API_KEY"
   mkdir -p /home/jovyan/rlvr && cd /home/jovyan/rlvr
   git clone https://github.com/rrtchaganti/reasoning-rlvr.git   # user name + read-only token at the prompt
   cd reasoning-rlvr && git checkout phase-4-grpo
   curl -fsSL https://claude.ai/install.sh | bash
   source scripts/e2e_env.sh && claude --version
   tmux new -s claude
   ```

   Inside tmux: `cd /home/jovyan/rlvr/reasoning-rlvr && source scripts/e2e_env.sh && claude`,
   `/login`, then the opening message of section 7.

## 2. Session 1 on the node, in order

Every step below is one `bash scripts/phase4_rental.sh <step>` call inside its own tmux session,
chained on the previous step's `EXIT=` line in its log (section 3, rule 4), with the GPU never idle
between them. The estimates are from the 18-19 Oct session's rates; the debugging run replaces them
with measured ones before the main run starts.

| # | step | what | estimate | notes for the session |
|---|---|---|---|---|
| 1 | path B of `docs/h100-next-session.md` | venv (cu129 route, 25 min), flash-attn wheel (13 min, CPU, concurrent), Triton check | 0.75 h | `setup_env.sh` now removes torchcodec and installs the wheel itself when it is under `$RLVR_HOME/wheels`; the wheel is gone with the disk, so build it (the recipe is in the guide) while the venv finishes |
| 2 | `setup` | downloads the SFT and RFT finals and the three sample sets from the Hub, rebuilds the perturbed GSM8K, the SFT set and the RFT data, **asserts the manifest hashes**, runs the `rft` dev-split eval (3 min), checks the pinned eval subsets | 0.3 h | the hash assertion stops the step on a mismatch: stop and ask; the dev number goes into the notes as the baseline of every dev curve |
| 3 | `smoke` | `rlvr-grpo` for 2 steps on 4 prompts, 256-token cap, vLLM colocated | 5 min | plumbing only: the reward log, the checkpoint folder, W&B |
| 4 | `p3ablations` | EXP-010g and EXP-010h: build, train (12 min each), think-tag eval (10 min each) | 0.75 h | both manifests record `keep_incorrect`, `select`, `fraction_incorrect`; the eval runs all five sets in one process |
| 5 | `debug` | 50 GRPO steps on GSM8K-heavy prompts; prints seconds per step, the 200-step projection, peak memory, reward and finished-rate at step 1 and 50 | 1.0 h | **the measurement step**: if the projection for a 200-step run is above 5.7 h (1.5x the 3.8 h estimate), stop and ask before `main`; if vLLM runs out of memory, lower `vllm_gpu_memory_utilization` by 0.05 through `RLVR_GRPO_EXTRA` and say so; the micro-batch stays 32 unless the card is under 70 % used, which the notes record |
| 6 | `main` | EXP-012: preflight, 200 steps, then the dev evals of the four checkpoints | 3.8 h + 12 min | the ten-minute rule at step 10 (section 4); the preflight output is pasted into the notes |
| 7 | `seed` | EXP-013: seed 1 + dev evals | 4.0 h | same |
| 8 | `evals` | the main checkpoint's three greedy templates, gap, probe; the pinned-subset check; the n = 32 trained-format samples; the seed checkpoint's five-set greedy eval | 1.4 h | all sets of a checkpoint in one process |
| 9 | section 5 | archive, uploads, hand-back, Delete | 0.5 h | |

Session 1 is about 12 node-hours, INR 4,600. If the user is away, the session continues through
the table on its own; it stops and asks only where the table says so, or where a rule of section 3
is about to be broken.

## 3. Rules for the node session (the user's, 19 Oct 2026; `docs/phase-4-plan.md` section 7)

1. **Check before launch, every time.** `scripts/phase4_preflight.py --run <row>` runs before every
   training step (the session script does it) and its output goes into the notes. A `FAIL` line
   stops the step; the session asks the user rather than fixing the config to make it pass.
2. **Same subsets as the method compared against.** The pinned hashes (GSM8K `58c8ded1c644f332`,
   MATH-500 `5ca9795b45227236`, Countdown `9698ed78ea4e0491`) are checked by `--subsets` in `setup`
   and again before the n = 32 sampling. A mismatch means the harness changed; stop and ask.
3. **The whole GPU, every step.** One `rlvr-eval` process per checkpoint with every set in its
   config; the sampling run with both sets; no eval of one set "for a look". The training
   micro-batch is 32 completions; it changes only if the debugging run shows the card under 70 %
   used, and then once, for every run alike.
4. **No idle gaps.** Steps run as `tmux new -d -s <step> -e HF_TOKEN="$HF_TOKEN" "bash scripts/phase4_rental.sh <step> > outputs/h100_p4_<step>.log 2>&1; echo EXIT=\$? >> outputs/h100_p4_<step>.log"`
   and the next step starts when the previous log's last line is `EXIT=0`. The watcher is
   `while ! grep -q '^EXIT=' outputs/h100_p4_<step>.log; do sleep 60; done` (60-90 s; it costs the
   GPU nothing). Never wait on a tmux session name (the 27-minute slip of 18 Oct). When a step ends,
   the notes get its actual time against the table above; idle minutes are counted.
5. **The ten-minute rule on every GRPO run.** At step 10 the log's step time gives the projection;
   above 1.5x the plan's estimate the run is stopped (`tmux kill-session -t =<step>`) and the user
   asked. Inside it, the run continues without a question.
6. **Decisions by written rules.** The pre-registered falsifiers, the EXP-013 noise row and the
   drop order (plan section 8) decide; anything they do not cover and that costs more than an hour
   is a question to the user. A GRPO run whose reward log shows the hacking flags (format-only
   outputs, repeated answers, length growth without accuracy, truncation rising) continues to its
   end and is reported as a failure with its incident log; it is not restarted with a changed reward.
7. **`CLAUDE.md`:** no edits to tracked files (fixes go to `outputs/patches/` as patch files), no
   commits or pushes, secrets in the shell only, every action in `outputs/h100_p4_session_log.md`
   (where / why / ran / result / undo) and every measurement in `outputs/h100_p4_session_notes.md`,
   archive before Delete.

## 4. Estimates and the measured rates behind them

| unit of work | measured 18-19 Oct | used for |
|---|---|---|
| sampling, long completions | 11-12k output tokens/s | GRPO rollouts: 256 x about 1,200 tokens = 0.3 M tokens, about 30 s per step |
| packed training with flash-attn | 70 M tokens/h | the policy update on about 0.35 M tokens, about 20 s per step |
| RFT-size training (11,145 traces) | 12 min | EXP-010g, EXP-010h |
| five-set greedy think-tag eval | 8-10 min | every checkpoint |
| dev-split greedy eval (500 problems) | 3 min | the `rft` baseline and four checkpoints per run |
| n = 32, T = 0.6 on two 300-problem subsets | 43 min | the main checkpoint |
| environment build / flash wheel / downloads | 25 / 13 / 5 min | path B |

GRPO's per-step time is the unknown: 65 s is the arithmetic, the debugging run is the measurement.
Rupees at INR 379.05 per hour: session 1 about 4,600, session 2 about 5,200.

## 5. End of a session: archive, uploads, hand-back, Delete

1. Uploads: the main checkpoint and its card to `Raviteja546/reasoning-rlvr-grpo` (the `report`
   step does it in session 2; in session 1 `push_to_hub.py model --run outputs/train/grpo_1p7b
   --repo Raviteja546/reasoning-rlvr-grpo --base-model-id Raviteja546/reasoning-rlvr-rft`), the
   seed and ablation finals under `seeds/` and `ablations/`, the n = 32 samples and the two Phase 3
   ablation finals (`ablations/` of the RFT repo) and their data manifests.
2. `outputs/h100_p4_handback.md` as on 19 Oct: the step table with actual times, every run's
   headline numbers with CIs and the file they came from, the preflight outputs, the per-step
   curves' summary (reward, finished rate, length, KL, zero-variance groups) at steps 1, 50, 100,
   150, 200 for every run, the hacking-watch verdicts, Hub and W&B URLs, `pip freeze`, `df -h`,
   node-hours and rupees.
3. The archive: every `metrics.json`, `report.md`, `run_info.json`, `train_log.jsonl`,
   `reward_log.jsonl`, `dynamic_sampling_log.jsonl`, `prompt_ids.json`, the per-sample files of
   every greedy eval, `outputs/reports/**`, the data manifests, `outputs/patches/`, the two session
   files, the hand-back, the logs; `tar czf /home/jovyan/rlvr/h100_p4_session<N>_archive.tar.gz`, its
   sha256 and `tar tzf | wc -l` printed, a copy uploaded to the samples dataset under `archives/`.
4. The user downloads it (JupyterLab file browser), checks the hash, then **Deletes** the notebook
   (not Stops: the disk is INR 112 a day and everything is on the Hub).

## 6. Session 2

Path B again, then `setup` (hashes and the pinned subsets), `a1`, `a2`, `a3` (3.8 h each; A2 about
1.3-1.6x on the sampling half, which the ten-minute rule allows for), `evals` (the three five-set
greedy evals), `report` (the paired tables against the main run and against `rft`, the pass@k table,
the card, the uploads), then section 5. About 13 node-hours, INR 5,200. Drop order if the money
runs short: plan section 8 (A3, then A1; never the evals or the CIs).

## 7. The opening message for the node session

```
You are the node session of the reasoning-rlvr project, on a TIR H100 notebook, branch phase-4-grpo.
Read CLAUDE.md, then docs/e2e-session-handoff.md (this file) and docs/phase-4-plan.md. Keep
outputs/h100_p4_session_log.md (where / why / ran / result / undo) and outputs/h100_p4_session_notes.md
(measurements, actual-vs-estimate after every step, surprises) from the first command. Secrets are in
the shell and in tmux's global environment; never write them anywhere.

State: the repo is cloned at /home/jovyan/rlvr/reasoning-rlvr on branch phase-4-grpo at the latest
commit; the shell is root with HOME=/home/jovyan; no venv yet; nothing has been downloaded.

Job: session 1 of handoff section 2, steps 1 to 9, in order, each inside its own tmux session chained
on the previous step's EXIT line, with the GPU never idle. Rules: handoff section 3, all seven, and
the ten-minute rule of section 4. Stop and ask me where the table says so, where a preflight fails,
where a hash does not match, or where a rule would be broken; otherwise continue without questions.
Start by building the environment (path B of docs/h100-next-session.md) with the model downloads in
parallel, tell me the measured build time, and give me the plan for the next hour.
```

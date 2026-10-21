# Phase 4 plan: GRPO with a verifiable reward, shape B (decided 19 Oct 2026)

Branch `phase-4-grpo`, opened from the `phase-3-rft` tip (c10f29c) because Phases 2 and 3 are not
merged yet; the laptop merges `phase-3-rft` into it whenever Phase 3 moves and rebases onto `main`
after the Phase 3 merge. Pre-registration: `experiments.md` EXP-010g, EXP-010h (two Phase 3
ablations that run first), EXP-012, EXP-013, EXP-014a-c. This page holds what Phase 3 settled, the
chosen size, the evaluation rules, the session plan with its budget, and the rules the node session
runs under. Nothing here runs before the code of section 6 exists and the local correctness check
has passed.

## 1. What Phase 3 settled

- **Start checkpoint: `rft` seed 0** (`Raviteja546/reasoning-rlvr-rft`, the pre-registered EXP-010
  run). The EXP-010 falsifier did not fire, so Phase 3's decision 6 applies. The two extra seeds are
  the noise floor (up to 3 points between seeds), not candidates. The dev-split think-tag accuracy
  of `rft` was not measured on the node; it is the first GPU job of the session (3 minutes) and the
  number every per-step dev eval is read against.
- **Prompt set:** the learnable prompts of EXP-009 (`outputs/data/rft/learnable_prompts.jsonl`,
  4,606 prompts with 0 < solve rate < 1 for the SFT policy at n = 8, T = 1.0, hash
  281bdfe31176411b in `data/manifests/rft_iter1.json`; rebuilt on the node from the Hub samples
  `sft_1.7b_rft_n8` by `rlvr-build-rft`, hash checked). PLAN.md's Phase 4 set is MATH levels 1-3 in
  full (2,267 learnable) plus a 20 % slice of the GSM8K-train learnable set (468 of 2,339, seeded):
  2,735 prompts. At 32 prompts per step and 200 steps the run draws 6,400 prompts, about 2.3 passes.
  Caveat: the rates were measured for the SFT policy at a 2,048-token cap; some excluded rate-0
  problems are "cannot finish in 2,048", and `rft` solves more of the set than `sft` did. Not
  re-measured (a 1.3-hour run); the zero-variance-group fraction per step is the live measurement.
- **The reward must not pay for loops.** Reward 1 only when the completion finished (not truncated),
  has the format (one closed think block, one answer block) and is correct under math-verify;
  otherwise 0. Format rate, truncation rate, correct rate and length are logged separately every
  step, because a reward of 0 hides which of the three failed.
- **The bar GRPO must clear:** +3 to +6 points at greedy in the trained format over `sft` is what
  one 12-minute RFT run bought. GRPO's row is read against `rft` (paired), with the sampling hours
  as the compute-matched comparison.

## 2. Shape B: the size of every run

| knob | value | why |
|---|---|---|
| start | `rft` seed 0 | section 1 |
| prompts per step | 32 | half of PLAN.md's 64; halves every run |
| samples per prompt (G) | 8 | PLAN.md |
| completions per step | 256 | |
| max new tokens | 1,536 | PLAN.md said 2,048; the finished-only reward makes a shorter cap a training signal, and `rft`'s think-tag mean is 1,426 on GSM8K; truncation at step 0 is expected at 30-40 % and must fall |
| temperature | 1.0, top-p 1.0 | PLAN.md |
| learning rate | 1e-6, constant after 10 warm-up steps | PLAN.md |
| KL beta | 0.04 to the `rft` reference (A1 sets 0) | PLAN.md |
| updates per batch | 1 (on-policy) | PLAN.md; with one update the PPO clip never activates (section 5, A2) |
| steps | 200 | PLAN.md said 400-600; most RLVR gain at this scale is in the first 100-200 steps, and the dev curve every 50 steps says whether it plateaued |
| dev eval | every 50 steps: greedy think-tag on gsm8k_dev 250 + math_dev 250, about 3 min | the checkpoint rule and the plateau check |
| checkpoints | steps 100 and 200 (final) | |
| backend | TRL GRPOTrainer, vLLM colocated, bf16 weights, flash-attn for the policy forward | the correctness check of section 6 ties it to the reference loop |

Estimated cost per run on the H100 at the session's measured rates: 256 completions x about
1,200 tokens = 0.3 M generated tokens per step (about 30 s at 10-12k tokens/s), policy and reference
forward-backward on about 0.35 M tokens (about 20 s), weight sync and logging (about 10-15 s):
about 65 s per step, 3.6 h for 200 steps plus 12 min of dev evals. **Measured at step 10 of the
debugging run and written into the session notes before the main run starts.**

## 3. Evaluation rules (the user's instructions of 19 Oct, made concrete)

1. **Same problems as the previous method, always.** Every comparison is paired on shared ids.
   Greedy evals use the full sets (GSM8K 1,319, perturbed 830, MATH-500 500, Countdown 500, AIME24
   30), the same the Phase 2 and 3 checkpoints were scored on. Sampling evals use the seeded
   300-problem subsets the RFT runs used, pinned by their manifest hashes, which the harness
   reproduces from `--limit 300 --seed 0`: GSM8K `58c8ded1c644f332`, MATH-500 `5ca9795b45227236`,
   perturbed GSM8K `a4f583879e249526` (now paired with GSM8K's draw after the 19 Oct fix, so the
   perturbed subset of a new run has about 190 problems, all paired), Countdown `9698ed78ea4e0491`.
   The preflight (section 6) compares the hashes before any sampling run and refuses on a mismatch.
2. **Reduce a set only when the CI stays usable.** Half-widths of a paired 95 % CI on a 2-5 point
   effect: GSM8K full about 2.3, at 300 problems about 5.0; MATH-500 full about 4.4, at 300 about
   5.7. Rule: use the smallest set whose half-width is under half the expected effect; for greedy
   that is the full set (a five-set greedy eval costs 8-10 minutes, so there is nothing to save), for
   sampling the 300-problem subsets (where the cost is in n, not in the problem count).
3. **Inference in one process per checkpoint.** All five greedy sets in one `rlvr-eval` call (vLLM
   loads once, the KV cache is shared); both sampling sets in one call. Never one set per process,
   never a set left for later.
4. **What each run gets.** The main checkpoint: the three greedy templates plus gap and probe
   (22 min), the think-tag n = 32, T = 0.6 sampling on the pinned subsets (45 min, paired with the
   `rft` and `sft` samples on the Hub) for the pass@k and new-versus-sharpened counts; the 4-shot
   n = 64 curve only if the session is under budget (1.4 h). Seed repeat and ablation checkpoints:
   the five-set greedy think-tag eval only (10 min each); their other quantities are the per-step
   training statistics.

## 4. Pre-registered experiments (full text in `experiments.md`)

| id | run | question | read against |
|---|---|---|---|
| EXP-010g | RFT without the correctness filter (same count of finished, well-formed self-generated traces, correct or not) | selection or self-distillation? | `rft`, control_tok, `sft` |
| EXP-010h | RFT with two *random* correct traces per problem instead of the shortest two | how much of the stopping effect is length selection? | `rft` |
| EXP-012 | GRPO main run (section 2) | does RL beat RFT, and is it sharpening or capability? | `rft` (paired), its three seeds |
| EXP-013 | the same config, seed 1 | GRPO's own seed noise | EXP-012 |
| EXP-014a | KL beta 0 | is the reference model needed? | EXP-012, EXP-013 |
| EXP-014b | dynamic sampling (drop zero-variance groups, resample to fill the batch); clip-higher set to 0.28 but inert at one update per batch | does the prompt filter leave enough variance? | EXP-012, EXP-013 |
| EXP-014c | Dr. GRPO loss (no per-sequence length normalisation, no std normalisation) | does the standard normalisation grow length? | EXP-012, EXP-013 | *deferred 21 Oct (section 9)* |
| EXP-015 | the 2,048-token cap of the original plan, everything else EXP-012's | is the absence of a non-termination gain (report section 4.5) an artefact of the 1,536 cap's length pressure? | `rft` and EXP-012 (the both-finished paired difference, pass@16), EXP-013 |

Every ablation is one seed and is read against the EXP-013 noise row: a difference smaller than the
seed-to-seed difference is reported as "within noise", whatever its sign.

## 5. Session plan and budget (two sessions; the second only after the first is read)

The notebook was deleted on 19 Oct (the user's call; the disk cost INR 112 a day), so each session
starts with path B of `docs/h100-next-session.md`: environment build 25 min, flash-attn wheel 13 min
(CPU, concurrent), model and sample downloads from the Hub, data step 5 min; about 45 minutes,
GPU idle for 25 of them. Rates INR 379.05 per hour.

| session 1 (plan, 20 Oct) | hours | INR | actual (20-21 Oct) |
|---|---|---|---|
| rebuild (path B) and the `rft` dev-split eval | 0.8 | 300 | 1.9 h: the build stalled 88 min (INR 555 idle); setup 14 min |
| EXP-010g and EXP-010h (two 12-minute trainings + two 10-minute evals) | 0.75 | 285 | 0.9 h |
| debugging run on GSM8K-train, 50 steps, rate measurement (PLAN.md step 2; not a result) | 1.0 | 380 | 1.0 h incl. two OOM attempts; 54.1 s/step |
| EXP-012 main run (200 steps + 4 dev evals) | 3.8 | 1,440 | 3.3 h (56.3 s/step) |
| EXP-012 evals (greedy x3 + gap + probe; n = 32 trained-format sampling) | 1.2 | 455 | 0.8 h |
| EXP-013 seed repeat + its greedy eval | 4.0 | 1,520 | 3.4 h |
| archive, Hub uploads, Stop (handoff section 5 item 4) | 0.5 | 190 | 0.2 h of work + 1.5 h idle until the Stop |
| **session 1** | **12.0** | **4,570** | **13.3 h, INR 5,030** (Stop 07:00 IST) |

| session 2 (re-estimated 21 Oct at session 1's rates) | hours | INR |
|---|---|---|
| path A restart (Stopped notebook: secrets, pull, env check) and `setup` (hashes only) | 0.15 | 55 |
| EXP-014a (3.3 h), EXP-014b (4.0 h: dynamic sampling over-generates 2x), EXP-015 (4.3 h at a projected 73 s/step; EXP-014c deferred, section 9), each with its four dev evals | 11.6 | 4,400 |
| EXP-015's n = 32 trained-format sampling on the pinned subsets | 0.5 | 190 |
| three five-set greedy evals | 0.4 | 150 |
| `report` (ablation compares, card, uploads), hand-back, archive, Delete | 0.7 | 265 |
| **session 2 (with EXP-015 in EXP-014c's place, 21 Oct)** | **13.35** | **5,060** |

Session 1's work itself (9 h 55 min of GPU time) came in under the plan; the overrun is the
environment build's stall and the idle tail until the Stop. Session 2 is estimated at the measured
56 s per step; the 20 Oct estimate of 13.75 h / INR 5,215 (path B, 3.8 h per run) is superseded.
The notebook is Stopped between the sessions (the user's point on 20 Oct: the disk at about INR 107
a day is cheaper than the INR 285 rebuild for a gap under 2.5 days, and the local checkpoints survive).

Phase 4 total at these numbers about 25 node-hours, INR 9,450 (USD 113). Project to date about
INR 16,600 after session 1; after session 2 about INR 21,700 plus the disk days, so session 2 fits
the ceiling with about INR 300 to spare only if the GPU is never idle; the drop order of section 8
(EXP-015's sampling, then A1; A3 is deferred) is the answer if it is not. On 20 Oct the arithmetic was: project to date about
INR 11,450; after Phase 4 about INR 21,300 (USD 254), so the ceiling had to rise from the INR 14,300
confirmed on 13 Oct.

**Decided 20 Oct 2026: the ceiling is INR 22,000 (USD 262).** The user chose B on 19 Oct knowing
the five runs' cost; the extras above (two rebuilds, the Phase 3 ablations, the debugging run, the
evals) were priced before the ceiling was set. Headroom over the estimate is about INR 700, under
two H100-hours: no 200-step run can be repeated inside it, so a failed run is handled by the drop
order of section 8 and the ten-minute rule of section 7, not by a rerun. COMPUTE.md carries the
running total against this ceiling.
If session 1 shows GRPO at or below `rft` (the EXP-012 falsifier), session 2 still runs: the
ablations are about the method, not about beating RFT. **Session 1's verdict (21 Oct): the falsifier
did not fire (+4.1 GSM8K, +7.0 MATH-500, paired); the noise row is 0.3 / 0.2 / 4.8 / 0.6; session 2
runs all three ablations at session 1's settings (micro-batch 16, vLLM 0.30, pinned by the
preflight).** Results in `outputs/reports/phase-4.md`.

## 6. Code to write on the laptop before session 1

1. `rlvr-grpo` (`src/rlvr/train/grpo.py`): a TRL `GRPOTrainer` wrapper in the style of `rlvr-sft`:
   YAML config with `--set` overrides, the reward function of section 1 (finished, format,
   correct; the three parts logged), the learnable-prompt loader with the GSM8K slice, vLLM
   colocate, the per-step statistics of PLAN.md step 5 (reward, correct rate, format rate,
   truncation rate, length mean/p90, entropy, KL, clip fraction, zero-advantage-group fraction,
   gradient norm, tokens generated, step wall-clock) to W&B and to `train_log.jsonl`, the dev eval
   every 50 steps through `rlvr-eval`, `run_info.json` with the prompt-file hash and the reference
   checkpoint, the three ablation switches (`kl_beta`, `dynamic_sampling` + `clip_higher`,
   `loss=dr_grpo`), and the reward-hacking watch of PLAN.md step 6 (format-only outputs, repeated
   final answers, length growth without accuracy growth, "Wait" counts) written per step.
2. **The correctness check** (PLAN.md step 1): the own reference loop (`src/rlvr/ref/grpo_loss.py`,
   `advantages_kl.py`, tested since Phase 0) against TRL on a 50-prompt toy set with identical
   seeds on the laptop's GPU with Qwen3-0.6B-Base; per-step losses and rewards must agree; the
   result is a table in the engineering notes before any node time. The Dr. GRPO and KL-free
   variants are checked the same way (the reference loop has the switches).
3. `rlvr-build-rft --keep-incorrect` and `--per-problem-select {shortest,random}` for EXP-010g and
   EXP-010h, with tests.
4. `scripts/phase4_preflight.py`: prints the resolved config of a run and asserts, before launch,
   the prompt-file hash, the start checkpoint's `run_info` commit, the eval subsets' manifest hashes
   (section 3), the step count, cap, LR, beta, G, prompts per step and seed against a table in this
   page; exits non-zero on any mismatch. The session script calls it before every training and
   sampling step.
5. `scripts/phase4_rental.sh`: `smoke`, `p3ablations`, `debug`, `main`, `seed`, `a1`, `a2`, `a3`,
   `evals`, `report`; resumable; every step chained on the previous step's EXIT marker.
6. The handoff for the node session (`docs/e2e-session-handoff.md` rewritten for Phase 4) with the
   rules of section 7, and `cards/grpo.md` from the report step.

## 7. Rules for the node session (the user's observations of 19 Oct, in force for every run)

1. **Check before launch, every time.** The preflight of section 6 runs before every training and
   sampling step and its output goes into the session notes; a mismatch stops the step and asks.
   No run starts on a config nobody compared with the plan's table.
2. **Same subsets as the method being compared against.** The pinned manifest hashes of section 3;
   a new run's subset hash must equal the old run's or the comparison is not made.
3. **The whole GPU, every step.** All eval sets of a checkpoint in one `rlvr-eval` process; both
   sampling sets in one; the training micro-batch sized to fill the card (measured once in the
   debugging run and fixed for every run after); vLLM's memory fraction set so sampling and
   training do not thrash. No set run on its own "for a quick look".
4. **No idle gaps.** Steps are chained on EXIT markers in their logs, never on tmux session names
   (the 27-minute slip of 18 Oct). Any watcher loop polls at 60-90 seconds (a `sleep` plus a
   `tail` costs the GPU nothing; the user's limit is under two minutes), reports the step's actual
   time against the plan's estimate when it ends, and counts idle minutes in the notes.
5. **The ten-minute rule on every run.** At step 10 the projected wall-clock and rupees are
   written; above 1.5x the plan's estimate the run stops and the user is asked. Within it, the run
   continues without a question.
6. **Decisions by written rules, not by taste.** The pre-registered falsifiers, the noise row, and
   the drop order (section 8) are the rules; a situation none covers is a question to the user.
7. The standing rules of `CLAUDE.md`: no edits to tracked files, no commits, secrets in the shell
   only, every action logged with where / why / ran / result / undo, archive before Delete.

## 8. Drop order if money or time runs short

1. The 4-shot n = 64 curve of the main checkpoint (optional already).
2. (EXP-014c is already deferred, section 9.) Inside session 2: EXP-015's n = 32 sampling first (0.5 h), then EXP-014a (KL off): their questions are about the method; the
   seed repeat stays because without it no ablation is readable.
3. EXP-013 last, and then the ablation table says its deltas have no measured noise floor (PLAN.md
   section 8's own rule).
Never dropped: the main run's full evals, the pinned subsets, the CIs.

## 9. Decisions after session 1 (21 Oct 2026)

1. **EXP-015 replaces EXP-014c in session 2** (the user's decision after the termination
   decomposition of `outputs/reports/phase-4.md` section 4.5). The whole paired GRPO gain over `rft`
   sits on problems `rft` truncated; among problems both policies finished nothing moved on
   MATH-500. Two readings compete: on-policy methods at this scale teach stopping and little else,
   or the finished-only reward under a 1,536-token cap (the p90 length pinned at the cap, a quarter
   of completions cut and scored 0) selected for brevity over solving. The second is testable for one
   run (about 4.3 h plus 0.5 h of sampling) and bears on the project's central claim; the Dr. GRPO
   question bears on the method's length behaviour, which the finished-only reward may hide anyway.
   EXP-014c's pre-registration stands, deferred. The full reasoning chain is
   `docs/research-narrative.md`.
2. **Session 2 budget with EXP-015:** about 13.4 node-hours, INR 5,060, which lands the project within
   about INR 200 of the INR 22,000 ceiling if the GPU is never idle. Inside session 2 the drop order
   is EXP-015's n = 32 sampling, then EXP-014a (section 8).
3. **Documentation rule (the user, 21 Oct):** no run goes into a session table without its link in
   `docs/research-narrative.md` (what was known, the intuition, the question, the prediction and
   falsifier, the id) and its entry in `experiments.md`.
4. **Outcome (21 Oct 2026, session 2 run as this table says; `outputs/reports/phase-4.md` sections
   10-18).** EXP-014a's hypothesis failed in sign (KL off is better in the trained format, by
   termination, and 15x further from `rft`; nothing under zero-shot); EXP-014b's falsifier fired
   (within noise at 1.5x the step time); EXP-015's falsifier fired (the cap is not what limits the
   reasoning gain). 13.45 node-hours, INR 5,100, nothing dropped; the notebook Deleted the same day.
   The next plan is `docs/phase-5-plan.md`, built from `docs/research-narrative.md` "Open now".

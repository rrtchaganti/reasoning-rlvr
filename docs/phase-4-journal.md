# Phase 4 journal: GRPO with a verifiable reward

Chronological, as the other phase journals: what was decided and built, in order, with the reasons
given at the time. The plan is `docs/phase-4-plan.md`; the pre-registration EXP-010g, EXP-010h,
EXP-012, EXP-013, EXP-014a-c in `experiments.md`.

## 1. Context on 19-20 Oct: the size question before the code

Phase 3 closed on 19 Oct with RFT beating SFT by +3.3 GSM8K / +5.8 MATH-500 in the trained format
(three seeds, a token-matched control), the pass@k ceiling untouched, and a quarter of SFT
generations looping. Phase 4's start checkpoint followed from the plan's decision 6: `rft` seed 0.

The budget did not follow. PLAN.md's Phase 4 (64 prompts x 8 samples per step, 500 steps, five runs
of that size) priced at the rates the 18-19 Oct session had measured came to about 90 H100-hours,
INR 34,000, because the trained model writes 1,400-2,600 tokens per completion where the base wrote
150-900, and RL sampling is the dominant cost. The user's reaction was the right one ("are you
kidding me"), and the plan was not changed until they had chosen: three shapes were priced (as
planned, INR 34,000; half size with every run kept, INR 6,500 for the runs; minimal with two runs,
INR 2,700). Asked what the half-size shape buys over the minimal one, the answer was the seed repeat
(without it no ablation difference is readable) and the KL and loss-normalisation ablations (the
design questions the interview story needs). The user chose the half-size shape, B: 32 prompts x 8,
1,536-token cap, 200 steps, main + seed + three ablations, in two node sessions.

At the same time the user set rules for the node session from what they had watched on 18-19 Oct:
compare two methods on the same subset, pinned; reduce a subset only while the CI stays usable; check
hyperparameters and subsets before every launch; use the whole GPU (all sets in one process);
background watchers poll under two minutes; no idle gaps. These became section 7 of the plan, section 3
of the handoff, and the preflight script. The pre-registration was written and committed (cc0caa1)
before any code, with one honest note in EXP-014b: at one optimiser step per generation the PPO clip
never activates, so the DAPO ablation tests dynamic sampling and the clip-higher setting is inert.

The notebook had been deleted by the user on 19 Oct (the disk cost INR 112 a day), so both sessions
start from the Hub. With the rebuilds, the debugging run, the evals and the two Phase 3 ablations the
honest Phase 4 total is about 26 node-hours, INR 9,800, not the INR 6,500 of the runs alone; the plan
says so and leaves the ceiling to the user.

## 2. What was built (20 Oct, laptop, no GPU)

- **`rlvr-grpo` (`src/rlvr/train/grpo.py`).** A wrapper around TRL 1.15's `GRPOTrainer` in the shape
  of `rlvr-sft`: YAML config with `--set` overrides and the same numeric-string cast, the prompt set
  rebuilt from the learnable-prompt file with the shared think-tag template and checked against the
  committed manifest's `learnable_hash` (count and hash; a mismatch refuses to run), the GSM8K slice
  drawn by seed, and `run_info.json` with the prompt hash and the prompt ids used. The reward is one
  function with three parts logged separately: finished (the last token is EOS or the completion is
  under the cap), format (one closed think block, one answer block, the eval harness's own check),
  correct (the harness's extraction and math-verify grading); reward 1 only when all three hold. The
  per-call statistics (reward, the three rates, format-only rate, mean and p90 length, "Wait" counts,
  repeated-answer groups, zero-variance groups) go to `reward_log.jsonl` and, through TRL's
  `log_metric` hook, to W&B; a `hacking_flags` function reads PLAN.md step 6's signals off the log
  at the end (length growth without accuracy, format-only outputs, repeated answers, Wait-spam,
  rising truncation) and writes the verdict into `run_info.json`.
- **TRL's names, mapped.** Reading `GRPOConfig` in the installed TRL: `loss_type='grpo'` is the
  published per-sequence normalisation (the main run), `'dr_grpo'` the constant normalisation,
  `'dapo'` the token-level one, and TRL's *default* is `dapo`, so the main run sets `grpo`
  explicitly; `scale_rewards='group'` divides by the group std, `'none'` does not; `epsilon_high`
  is DAPO's clip-higher; `num_iterations=1` with one optimiser step per generation keeps the run
  on-policy (ratio 1, clip inert, `clip_ratio` logged to show it). Dynamic sampling is not in TRL
  (the word appears once, in a comment), so it is a subclass: over-generate `oversample` x the
  prompts, score them all through TRL's own path, keep the first 32 groups whose advantages are not
  all zero, fill from the front when there are fewer, and subselect every tensor of the returned
  batch by row. The optimiser step therefore sees the same number of completions in every run; the
  extra cost is sampling and scoring, and the informative fraction is logged per step.
- **The dev evaluation every 50 steps** is not inside the trainer. TRL's own `eval_strategy`
  samples completions on the eval prompts at the training temperature, which is a sampled reward,
  not the greedy dev accuracy the LR rule and the curve want. Instead the trainer saves weights-only
  checkpoints every 50 steps (`save_only_model`; about 7 GB each in fp32) and the session script runs
  `rlvr-eval` on each with the dev config, one process per checkpoint, 3 minutes each. Same
  evidence, no second engine inside the training process, the GPU doing one thing at a time.
- **Memory arithmetic for the H100:** fp32 master weights 6.8 GB, gradients 6.8, Adam 13.6, the
  frozen reference in bf16 3.4, vLLM at 35 % of the card 28 GB for the KV cache, activations with
  gradient checkpointing a few GB at a 32-completion micro-batch: about 60 GB. The debugging run
  measures it; `RLVR_GRPO_EXTRA` lowers the vLLM fraction if it must.
- **`rlvr-build-rft --keep-incorrect`** (EXP-010g: the correctness filter off, every other filter on;
  `meta.correct` recorded per example, `fraction_incorrect` in the manifest) and
  **`--per-problem-select random --select-seed N`** (EXP-010h: two usable samples drawn at random per
  problem with a per-problem seed, instead of the shortest two). Tests on hand-built samples for
  both, including that the random draw never picks an incorrect sample when the filter is on.
- **`scripts/phase4_preflight.py`.** Prints the resolved config and refuses a launch that differs
  from the plan's row for the named run (every shape-B knob and the one the ablation changes,
  including the output directory, so an ablation cannot overwrite the main run's folder), that uses a
  prompt file whose hash or count differs from the manifest, or that starts from a checkpoint whose
  `run_info.json` is not the RFT run at commit bb6f2a6; `--subsets` reproduces the pinned
  300-problem subsets through the harness and compares the hashes. A test loads the committed
  config with each row's overrides and asserts every row passes and a wrong knob is named.
- **`scripts/phase4_rental.sh`.** `setup` (Hub downloads, the data rebuilt and its hashes asserted
  against the committed manifest, the `rft` dev baseline, the subset check), `smoke`, `p3ablations`,
  `debug` (prints seconds per step and the 200-step projection), `main`, `seed`, `a1`-`a3` (each:
  preflight, train, the dev evals of its checkpoints), `evals` (all sets of a checkpoint in one
  process), `report`; `session1` and `session2` chain them. Every ablation is a `--set` override of
  the one config file, so the row in the preflight table is the single source of what each run is.
- **`scripts/grpo_correctness_check.py`** (PLAN.md step 1). Builds a small `GRPOTrainer` (HF
  generate, LoRA, Qwen3-0.6B-Base, 8 prompts x 4 samples, 256 tokens) and, for each generation batch
  TRL produces, recomputes with `src/rlvr/ref`: the group advantages from the re-scored rewards
  against TRL's `advantages`; the policy's per-token log-probs with a plain forward pass and the
  loss with `grpo_loss` against TRL's `_compute_loss` on the same tensors; the k3 KL against TRL's
  logged `kl`. Three settings: the main run's, A3's (Dr. GRPO), A1's (no KL). One constant differs
  by construction: Dr. GRPO divides by B x max_completion_length in TRL and by B x T_padded in the
  reference; the script rescales and says so. The check runs on the laptop's GPU from the user's
  terminal (the Claude Code shell cannot hold a multi-minute GPU job here); its table goes into this
  journal and the engineering notes before any node time.
- **The handoff rewritten for Phase 4** (`docs/e2e-session-handoff.md`): state, the user's laptop
  steps, the session-1 table with estimates and the measurement step, the seven rules with the exact
  tmux and watcher commands, the rates behind the estimates, the archive and Delete routine, session 2,
  the opening message.

Tests: 21 new (config, prompt set and manifest, the reward's three parts and the cap edge cases, the
statistics and the hacking flags, the group selection of dynamic sampling, the TRL config for shape B
and the ablations, the preflight against the committed config, the two builder options). Not done on
the laptop: the correctness check itself (the user's terminal, about 15 minutes with the model
download), and the node smoke of `rlvr-grpo` with vLLM colocated, which is the first GPU step of
session 1.

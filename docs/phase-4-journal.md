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

## 3. The correctness check, four runs to get it right (20 Oct, the user's terminal)

PLAN.md Phase 4 step 1 asks for the own reference loop to be checked against the library before
any node time. The check went through four runs on the laptop's 8 GB GPU, each exposing something.

1. **Out of memory** in the check's own forward pass: 32 completions through the 0.6B model at
   once with full-vocabulary logits is about 10 GB. TRL had generated and scored the batch fine.
   Fix: both sides of the comparison run four sequences at a time.
2. **`current_gradient_accumulation_steps` missing**: TRL's `_compute_loss` divides every
   micro-batch loss by a counter that only the Trainer loop sets. The chunked check sets it to 1,
   so each chunk returns its own mean and the average over equal chunks is the batch value.
3. **A check with no teeth.** Loss values and KL matched TRL to ten decimals, and the dynamic
   table showed zero informative groups, because the real reward was 0 for every completion: the
   base 0.6B never writes a finished, well-formed, correct think-tag answer in 256 tokens. With all
   advantages zero the policy term is never exercised, and at ratio 1 the published GRPO loss
   *value* is exactly 0 for any centred advantages, so a matching value proves little. Fix: TRL
   trains on a deterministic mixed reward in the check (the last hex digit of the completion's
   SHA-1; the real reward's rate is still reported), and the check compares the *gradients* of
   TRL's loss and of the reference loss with respect to the LoRA weights on the same batch.
4. **The run that counts** (Qwen3-0.6B-Base, LoRA r 16, 8 prompts x 4 samples, 256 tokens, bf16,
   HF generate, three generation batches per setting):

| setting | step | real reward rate | zero-adv groups | mean reward | adv max diff | ref loss | TRL loss | loss diff | grad cosine | grad max diff | grad rel diff | ref KL | TRL KL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main (grpo, group std, beta 0.04) | 0 | 0.000 | 0.12 | 0.500 | 2.97e-04 | 0.000038 | 0.000038 | 1.40e-09 | 0.999914 | 2.15e-03 | 1.31e-02 | 0.000873 | 0.000870 |
| main | 1 | 0.000 | 0.00 | 0.438 | 2.97e-04 | 0.000004 | 0.000004 | 9.31e-09 | 0.999903 | 1.46e-03 | 1.39e-02 | 0.000119 | 0.000094 |
| main | 2 | 0.000 | 0.00 | 0.562 | 2.97e-04 | 0.000035 | 0.000035 | 3.73e-09 | 0.999903 | 1.32e-03 | 1.39e-02 | 0.000847 | 0.000854 |
| A3 (dr_grpo, no scaling) | 0 | 0.000 | 0.12 | 0.500 | 0.00e+00 | -0.022495 | -0.022495 | 1.04e-09 | 0.999917 | 6.25e-04 | 1.29e-02 | 0.000873 | 0.000870 |
| A3 | 1 | 0.000 | 0.00 | 0.438 | 0.00e+00 | -0.027035 | -0.027035 | 3.35e-10 | 0.999902 | 1.24e-03 | 1.40e-02 | 0.000119 | 0.000094 |
| A3 | 2 | 0.000 | 0.00 | 0.562 | 0.00e+00 | 0.018490 | 0.018490 | 2.68e-09 | 0.999907 | 5.10e-04 | 1.36e-02 | 0.000847 | 0.000854 |
| A1 (beta 0) | 0 | 0.000 | 0.12 | 0.500 | 2.97e-04 | 0.000000 | 0.000000 | 1.86e-09 | 0.999920 | 2.15e-03 | 1.27e-02 | - | - |
| A1 | 1 | 0.000 | 0.00 | 0.438 | 2.97e-04 | 0.000000 | 0.000000 | 3.73e-09 | 0.999907 | 1.58e-03 | 1.36e-02 | - | - |
| A1 | 2 | 0.000 | 0.00 | 0.562 | 2.97e-04 | 0.000000 | 0.000000 | 3.73e-09 | 0.999880 | 1.71e-03 | 1.55e-02 | - | - |

| dynamic sampling (oversample 2) | step | rows returned / expected | shapes consistent | groups generated | informative | filled | kept zero-advantage |
|---|---|---|---|---|---|---|---|
| | 0 | 32 / 32 | True | 16 | 16 | 0 | 0 |
| | 1 | 32 / 32 | True | 16 | 15 | 0 | 0 |
| | 2 | 32 / 32 | True | 16 | 15 | 0 | 0 |

Reading. The loss values agree to 1e-9 in every setting, including Dr. GRPO's constant
normalisation (after the one rescaling the script documents: TRL divides by B x
max_completion_length, the reference by B x T_padded). The advantages agree exactly without std
scaling and to 3e-4 with it: TRL adds 1e-4 to the group std, the reference 1e-6, a convention and
not an error; the reference's default stays, the difference is stated here. The KL (k3) agrees
with TRL's logged value to 2e-5 (TRL's log is a per-micro-batch average in bf16). The gradients
with respect to the LoRA weights have cosine 0.9999 and a relative norm difference of 1.3-1.6 % in
every setting, the same in the KL-free setting, which points at bf16 forward noise between two
log-softmax paths rather than at a term that one side has and the other lacks (a missing term
would show up as a setting-dependent gap). A `--dtype float32` option was added to confirm this; if
the fp32 run brings the relative difference under 1e-3 the check is closed. The dynamic-sampling
subclass returns exactly the planned number of rows with every tensor subselected alike, and with a
mixed reward 15-16 of the 16 over-generated groups are informative, so the selection path works
where the earlier all-zero run could not show it. The real reward's rate was 0 on this model and
cap, as expected; the real reward is covered by the CPU unit tests (finished, format, correct, the
cap edge cases) and will first fire on the node's smoke with the RFT checkpoint.

**The fp32 confirmation (20 Oct, two more runs).** The first fp32 attempt made the gap *larger*
(loss diff 1e-5, gradient relative difference 5-11 %), because `--dtype float32` had changed only
the weights: TRL's forward still ran under bf16 autocast (`bf16=True` in the trainer config) while
the reference forward ran in fp32, two precisions compared against each other. With autocast off
for the fp32 check (59b5e1e), on 4 prompts x 4 samples at 192 tokens, chunk 1:

| setting | step | adv max diff | loss diff | grad cosine | grad max diff | grad rel diff |
|---|---|---|---|---|---|---|
| main, fp32 | 0 | 2.97e-04 (the 1e-4 vs 1e-6 std epsilon) | 7.45e-09 | 1.000000 | 1.66e-05 | 1.44e-05 |
| main, fp32 | 1 | 1.48e-04 | 3.73e-09 | 1.000000 | 9.58e-06 | 1.28e-05 |

The gradients agree to 1.4e-5 in relative norm: the 1.3 % of the bf16 run was bf16 forward noise,
as suspected, and the reference loss and TRL's are the same function of the same inputs. One more
thing the fp32 run shows: its KL is exactly 0, because no optimiser step is taken between the
generation batches and the policy *is* the reference. The bf16 run's KL of 8e-4 was therefore
numerical noise between two forwards of the same weights, matched on both sides because both read
the same noise through the same k3 formula; the KL term's arithmetic is checked, a KL from an
actual policy drift is not, and the k3 unit tests in `tests/test_ref_advantages_kl.py` are what
cover that. The check is closed; PLAN.md Phase 4 step 1 is done.

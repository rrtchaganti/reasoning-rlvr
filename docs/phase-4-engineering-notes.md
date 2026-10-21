# Phase 4 engineering notes: GRPO with a verifiable reward, and what the ablations were for

Branch `phase-4-grpo` (from `phase-3-rft` c10f29c). Code on the laptop 19-20 Oct 2026; the runs on
one TIR H100 notebook in two sessions (20-21 Oct, 21 Oct); written up 21 Oct 2026. Results:
`outputs/reports/phase-4.md`. Chronology: `docs/phase-4-journal.md`. The chain of questions:
`docs/research-narrative.md`. Pre-registration: `experiments.md` EXP-010g, 010h, 012, 013, 014a-c, 015.

## 1. What was built (laptop, no GPU)

- **`rlvr-grpo`** (`src/rlvr/train/grpo.py`): a wrapper over TRL's `GRPOTrainer` with vLLM colocated.
  It loads the learnable-prompt file and checks it against the committed manifest (count and hash),
  draws a seeded fifth of GSM8K-train beside the MATH prompts, builds the think-tag prompt, and scores
  every completion with `score_completion`: finished (EOS before the cap), format (exactly one think
  block then one answer block), correct (the harness's extractor and grader). Reward 1 only when all
  three hold. Per-step statistics go to a sink (`reward_log.jsonl`): reward, finished and format
  rates, correct-of-finished, token counts and p90, zero-variance group rate, repeated-answer rates
  (plain and incorrect-only), "Wait" per completion, format-only rate. `run_info.json` carries the
  config hash, the prompt hash, first/last-step signal, peak memory and the hacking watch.
- **Dynamic sampling** as a subclass: generate `oversample` x the prompts, keep the informative
  (non-zero-variance) groups up to the step's size, fill with zero-advantage groups only when short,
  log every step's counts (`dynamic_sampling_log.jsonl`). `epsilon_high` is passed through to TRL.
- **The hacking watch** (`hacking_flags`): five flags read off the first and last 20-step windows:
  length growth without accuracy, format-only outputs above 0.2, groups agreeing on one *wrong* answer
  above 0.5 (the correctness term was added after session 1, where the plain agreement rate rose with
  accuracy and flagged healthy runs), "Wait" spam, truncation rising.
- **`scripts/phase4_preflight.py`**: refuses a launch whose resolved knobs differ from the plan's row
  (shape B plus the one ablation switch), whose prompt file does not match the manifest, whose start
  checkpoint is not the RFT run, or whose pinned eval subsets do not reproduce their hashes. Tests in
  `tests/test_phase4_preflight.py`.
- **`scripts/phase4_rental.sh`**: the resumable session script (`setup`, `smoke`, `p3ablations`,
  `debug`, `main`, `seed`, `a1`, `a2`, `a3`, `cap`, `evals`, `report`); each step skips what exists,
  runs the preflight first, writes `EXIT=` only through the launcher's `echo`; the report step
  computes the paired tables that are missing, regenerates the card and uploads the finals.
- **`scripts/grpo_correctness_check.py`**: the reference loop (`src/rlvr/ref/`: group advantages,
  k1/k2/k3 KL, the clipped loss with three aggregations) against TRL on the same batch, fp32 and
  bf16; loss to 1e-9, gradients to 1.4e-5 in fp32 (journal section 3).
- **`scripts/decompose_gain.py`**: the per-problem split of a paired gain by the reference's
  truncation, with bootstrap CIs (report section 4.5). **`scripts/plot_grpo_curves.py`**: six-panel
  training and dev curves for any set of runs. `rlvr-build-rft --keep-incorrect` and
  `--per-problem-select random` for the two Phase 3 ablations.
- **`scripts/push_to_hub.py`**: a GRPO branch of the model card (added 21 Oct after the node's card
  printed the SFT fields as None), with a test.

## 2. The reward, and what each term is for

Reward = finished AND format AND correct. *Correct* is the harness's grade on the answer block, the
same code path as every eval, so a reward hack on the extractor would show up as an eval gain too
(and the eval has the perturbed twin and the probe against it). *Format* makes the output parseable
and stops tag stuffing (exactly one of each block). *Finished* is the term that matters: a completion
cut at the cap scores 0 whatever it holds, which is the only way to make the policy stop looping,
and also the structural bias of the whole phase (report sections 4.5, 14): the reward pays for
brevity before it pays for correctness. Nothing else was added (no partial format credit, no length
shaping), so the decomposition could be read cleanly. The alternative, a correctness-only reward with
truncated completions scored on their extracted answer, would have rewarded loops that happen to
contain the number and was rejected at the plan stage.

## 3. Shape B, the memory arithmetic, and the micro-batch

32 prompts x 8 samples, 1,536-token cap, 200 steps, LR 1e-6 constant after 10 warm-up steps, KL beta
0.04, one optimiser step per generation batch (so the ratio is 1 and the clip never activates; the
"clip region 0" row is a check, not a result). The plan's shape (64 x 8, 2,048, 400-600 steps) cost
90 H100-hours; B cost 3.1 hours a run. The cut's risk (cap, steps) was named in the plan and the cap
half was tested by EXP-015.

Memory at micro-batch 32 did not fit beside vLLM at 0.30 of the card: fp32 policy weights 6.9 GB,
grads 6.9, Adam 13.8, the reference 6.9, vLLM 23.7, about 58 GB fixed, and the logits of 32 x 1,700
tokens x 151,936 vocab in fp32 are about 33 GB. Micro-batch 16 with gradient accumulation 16 gives the
same 256-completion update (TRL's `grpo` loss averages per-sequence losses over the micro-batch and
the trainer averages the accumulated micro-batches) at a peak of 70 GB. 24 is not valid (256 must be
a multiple). These are the config's values now, pinned by the preflight.

## 4. Eval protocol and the noise row

Every checkpoint: five-set greedy think-tag eval in one `rlvr-eval` process (GSM8K 1,319, perturbed
830, MATH-500 500, Countdown 500, AIME24 30; 4,096-token window), paired against `rft` and against the
main run on shared problem ids with bootstrap CIs; the main run also under 4-shot CoT and zero-shot,
with the perturbation gap and the partial-prompt probe; n = 32 at T = 0.6 on the pinned 300-problem
subsets (hashes checked before sampling) for pass@k against `rft`'s and `sft`'s sample sets. The dev
split (250 + 250) every 50 steps gives the training curve; its half-width is about 5 points, so it
orders checkpoints only coarsely.

The seed repeat (EXP-013) is the noise row every ablation is read against: 0.3 / 0.2 / 4.8 / 0.6.
It was pre-registered as "under 2 / 3 points" and came out at 4.8 on MATH-500, which is the single
most useful number of the phase for reading anyone's one-seed MATH-500 claims at this scale.

## 5. The session rules, and who decided what

The user's rules of 19 Oct (plan section 7): same subsets across methods; preflight before every
run; whole GPU per process; watchers under two minutes; no idle gaps; anchored `EXIT=` watchers on
the step logs. The user decided: shape B (19 Oct), the ceiling (INR 22,000, 20 Oct), micro-batch 16
(15:23 UTC 20 Oct), Stop rather than Delete between sessions, EXP-015 in A3's slot (21 Oct), the two
additions before the Delete (21 Oct), the Delete. The node agent decided nothing that cost more than
an hour and never Stopped or Deleted. The laptop session computed every paired table from the
archives rather than trusting the node's reading, and found one thing the node's reading missed
(the card).

## 6. Budget

Plan: about 26 H100-hours, INR 9,800, two sessions. Actual: 26.7 node-hours, INR 10,130, plus about
INR 160 of disk (`COMPUTE.md`). The overrun is session 1's 88-minute environment build stall and the
idle tails; the GPU work itself finished inside the estimates in both sessions, and nothing was
dropped. Project at about INR 21,750 of 22,000 after the Delete.

## 7. Known limitations and debts

1. **One seed per ablation.** The ablation deltas are read against a noise row from one seed pair;
   a 4.8-point MATH-500 row makes every MATH-500 ablation claim coarse. Three seeds at 100 steps would
   have been a better spend than one at 200 (the reward plateaued by step 50).
2. **`train_log.jsonl` is written once, at the end of the run** (`grpo.py`, the block after
   `trainer.train()`); the live per-step time is only in the stdout log. A per-step append through a
   `TrainerCallback.on_log` is the fix; the handoff names the stdout log meanwhile.
3. **TRL 1.15 skips the reference model at `kl_beta=0`**, so A1 has no in-training KL; the offline
   estimator (`outputs/kl_offline.py` in the session 2 archive, not yet in `scripts/`) re-tokenises
   TRL's decoded completions, so the ids can differ from the sampled ones where the decode is not
   round-trip exact. It reproduces TRL's logged KL for the two runs that have one.
4. **`zero_variance_group_rate` under dynamic sampling is over the generated groups**, not the kept
   batch; the report reads it as one minus the informative fraction.
5. **The card's `trunc %` and `extract-fail %` are maxima over sets** (Countdown dominates), noted in
   the card; a per-set column is the fix. The GRPO card on the Hub (commit 7e6e7acb) is the SFT-shaped
   one until the regenerated `cards/grpo.md` is pushed with the user's token.
6. **The session script's `report` did not upload the ablation finals**; added after session 2 ran
   them by hand. Untested on a node.
7. **The Phase 3 controls trained on a seen subset** (carried debt); **EXP-010g has no power**
   (EXP-010i proposed); **EXP-014c not run**.
8. **No think-block length in the logs**: reasoning collapse was checked after the fact from the
   completion parquet files (no completion under 400 characters in a1's last ten steps), not from a
   logged statistic.
9. `kl_offline.py` and the budget-forcing eval option (EXP-016) are not in the harness yet.

## 8. Results and what they mean (21 Oct 2026; `outputs/reports/phase-4.md`)

GRPO from `rft` beats `rft` paired on every set with 500 or more problems in the trained format
(+4.1 / +4.8 / +7.0 / +7.0); decomposed per problem, the whole gain sits on problems `rft` truncated
and the both-finished difference is inside its CI. The three session 2 runs and the seed repeat
decompose the same way; the 2,048 cap does not move the both-finished difference or pass@16; the
reference model off buys more of the same stopping, 15x further from `rft`, and nothing under
zero-shot; dynamic sampling buys nothing. Against the base model's best prompt the pipeline adds +6
on GSM8K and +11 on MATH-500, nearly all of it SFT's. The method was stable and no hacking flag fired.
What it does not show: a reasoning gain from GRPO at this scale (the only both-finished cells clear of
zero are on perturbed GSM8K, single-seed); whether harder prompts would change that (EXP-018).

## 9. Interview question bank for this phase

1. *Why GRPO and not PPO?* Binary verifiable reward, no value model, group mean as the baseline; the
   plan's table of alternatives. The reference implementation agrees with TRL to 1.4e-5.
2. *Your gain is +7 on MATH-500. How do you know it is not just stopping?* The per-problem split: the
   gain sits on problems the previous policy truncated; both-finished 0.0 [-4.3, 3.9]; the loops do
   not finish at 8,192 tokens; the cap test and the KL-off run reproduce the shape.
3. *Is the cap the reason?* EXP-015: 2,048 tokens, same both-finished, same pass@16, more truncation
   at eval. Pre-registered prediction and falsifier; the falsifier fired.
4. *What did removing the KL penalty do?* 15x the drift, half the length, +5.6 on MATH-500 in the
   trained format, all on truncated problems, zero under zero-shot. The penalty was a brake on stopping.
5. *Why did dynamic sampling not help?* The solve-rate filter had already left 32 informative groups
   in 199 of 200 steps; DAPO's fix solves a problem the data pipeline had solved.
6. *What is your noise floor?* A seed repeat: 0.3 / 0.2 / 4.8 / 0.6. Most one-seed MATH-500 claims at
   this scale are inside it.
7. *Where could reward hacking come from with a verifiable reward?* The cap (brevity), the loss
   normalisations (Dr. GRPO), the extractor (tag stuffing, hedging, post-answer babble, each scored 0
   here), reasoning collapse, memorisation; which were measured and which were not.
8. *Why micro-batch 16?* The logits of 32 x 1,700 x 152k vocab in fp32 are 33 GB; 24 is not a divisor
   of 256; accumulation gives the same update.
9. *What would you do differently?* Three seeds at 100 steps; harder prompts from the start; the
   2,048 cap; log the think length; append the train log per step.
10. *What is the cheapest way to get what GRPO got here?* Unknown yet: EXP-016 (a decoding rule, free)
    and EXP-019 (DPO on pairs, 20 minutes) are the pre-registered tests.

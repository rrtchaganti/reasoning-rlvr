# The plan after Phase 4: Phases 5 to 7, one numbering, no further rental

Written 22 Oct 2026 after the Phase 5 merge (PR #6, tag `phase-5`), at the user's request for one
plan that starts fresh from the Phase 4 result and does not rename phases again. It supersedes the
option table of `docs/phase-5-plan.md` (options B-D are not taken; EXP-018 stays pre-registered and
unrun) and PLAN.md's Phases 5-8, whose mapping is in section 6. **Status: proposed, awaiting the
user's edits; no code before that.** Rules unchanged: narrative link and registry entry before any
run (`docs/research-narrative.md`, `experiments.md`); preflight before every GPU run; every number
with its CI; every deviation where the number is quoted; every mechanism sentence with its number.

## 0. Starting point

Four phases done and merged. The finding: in the trained format every post-SFT gain is termination;
a 2,048-token cap, no KL penalty and dynamic sampling each bought more or less of the same stopping
and nothing on problems both policies finish (`outputs/reports/phase-4.md` section 18). Spend about
INR 21,750 of the INR 22,000 ceiling; nothing on E2E. Two questions follow: how cheaply can stopping
be bought, and what exactly changed in the text. A third, whether harder prompts would make GRPO
teach something a rule cannot (EXP-018, about INR 1,700), needs a rental and is not run, by the
user's decision of 22 Oct 2026; it stays in the narrative's open table as the next experiment.

## 1. Phase 5: test-time compute, the cost of termination (INR 0)

**Question.** Is the RL policy's number reachable by spending tokens differently, and at what cost?

**5A, done (tag `phase-5`; `outputs/reports/phase-5.md`).** EXP-017: voting over two `rft` samples
matches `grpo` greedy at 2.6x the tokens, four beat it at 5x; from k = 4 the maj@k curves of the four
trained checkpoints coincide. EXP-016: a forced stop on `sft` at 1,536 tokens recovers all of RFT's
gain and 70-77 % of GRPO's cumulative gain; on `rft` at 2,048 it matches GRPO within the CI at fewer
tokens; forced answers are right about half the time.

**5B, next, branch `phase-5b-test-time-compute`, all on the laptop.**

| order | run | id | what it decides | laptop time |
|---|---|---|---|---|
| 1 | the two Phase 5 debts: `grpo` under the stop rule at 1,536 and 2,048; the unforced `sft` and `rft` runs on this machine paired against the H100 runs | EXP-020 | whether GRPO's remaining loops hold anything; whether the cross-machine caveat of EXP-016 is material | about 2.5 h |
| 2 | the "Wait" variant of budget forcing: at the budget append "\nWait" inside the think block, allow up to 512 more tokens, then force the stop; against the immediate stop at the same budget and at the same total tokens | EXP-021 | whether sequential extension (s1's move) helps a looping model, or only the stop does | about 2 h |
| 3 | an outcome reward model: a LoRA head on Qwen3-0.6B-Base trained on about 20,000 labelled traces from the Phase 3 solve-rate pass, held-out pair accuracy reported; then best-of-N and reward-weighted vote against majority vote on the n = 32 sample sets of `sft`, `rft`, `grpo` | EXP-022 | when a learned verifier beats voting, and whether best-of-N plateaus with a weak ORM at this scale | about 6-10 h, mostly unattended |

Output: the D6 chart with greedy, forced stop, "Wait", majority vote, weighted vote and best-of-N on
the same axes (tokens generated per problem against accuracy, CIs), and report sections 4-6. The
open PRM comparison of PLAN.md is dropped: nothing that fits 8 GB is worth comparing.

**What has to be built (laptop, with tests).** `--force-continue "Wait" --force-continue-tokens N`
in `rlvr-eval` (a third pass between the first and the forced stop); `rlvr-build-orm` (pairs and
labels from the Phase 3 sample files, held-out split by problem, manifest with hash) and `rlvr-orm`
(LoRA head on the 0.6B base with TRL or plain transformers, bf16, gradient checkpointing, the
project's `run_info`); `rlvr-score` (the ORM over a sample set, one score per sample, written next
to `samples.jsonl`); `scripts/plot_accuracy_vs_tokens.py` extended with best-of-N and weighted-vote
series; `scripts/phase5b_laptop.sh` (resumable, the EXP-016 pattern).

## 2. Phase 6: the traces (INR 0, no GPU)

**Question.** What did the training change in the text? The decomposition says where the gain sits;
nothing yet says what a loop is, where it starts, or what GRPO removed.

**Run.** EXP-023, descriptive but pre-registered: from the first-sample files of base, `sft`, `rft`,
`grpo`, KL-off and the cap run (think-tag and zero-shot where they exist, GSM8K and MATH-500), per
checkpoint: think-block length distributions; loop onset (the position of the first repeated 50-gram)
and the repeat unit's length; the rate of a boxed answer before the loop (59 % of `sft`'s truncated
zero-shot MATH-500 completions); counts of "Wait" and verification phrases per completion (the
training logs showed 5.9 to 3.6 per completion in the main run and 0.65 in KL-off); what the forced
answers of EXP-016 were extracted from and how far into the loop. Predictions in the registry; the
one that matters: GRPO's remaining loops have the same structure as SFT's (same repeat-unit lengths,
same onset distribution), only fewer of them.

**Output.** `outputs/reports/phase-6.md` with one table and one figure that put a number under every
mechanism sentence in the write-up; `scripts/analyse_traces.py` with tests.

## 3. Phase 7: write-up and the interview packet (INR 0)

The README's final pass (the question, the one table, the chart); the two-page packet of PLAN.md
section 10 with measured numbers; the paper-defence notes tied to our data (GRPO/DeepSeekMath, DAPO,
Dr. GRPO, Yue et al., s1); `experiments.md`, `COMPUTE.md` and `FAILURES_AND_LEARNINGS.md` reviewed;
the model cards re-pushed; the resume bullets; a blog post if the user wants one. Exit: someone else
can clone the repo, run one script and reproduce the base-model table.

## 4. Dropped, with the reason (also in the narrative)

- **DPO (PLAN.md Phase 5, EXP-019).** Its question, on-policy against off-policy at equal compute,
  lost its object: the on-policy stages bought termination and a free decoding rule recovers it, so
  DPO's best outcome is "matches a rule for INR 800". RFT already serves as the off-policy
  comparator and is decomposed. The pairs available (finished-correct against looping) share long
  prefixes, the regime where DPO's likelihood displacement degrades both responses, so a negative
  result would say little.
- **Distillation to 0.6B (PLAN.md Phase 7, D4's fourth checkpoint, D7).** A generic measurement
  with no thread from the data, needing a rental (about INR 1,200) or two days of laptop LoRA. The one
  non-generic version, "does the termination behaviour distil" (a student on the GRPO model's short
  finished traces against one on the long distilled traces), is recorded as an open question.
- **Quantisation and serving (PLAN.md Phase 7, D7).** An engineering deliverable whose laptop
  numbers would represent nothing a reviewer wants.
- **EXP-018 (harder prompts) and EXP-010i, EXP-014c.** Pre-registered, priced, unrun by the budget
  decision; the first is the next experiment the write-up names.

## 5. Order and time

Phase 5B about three laptop days (the reward model's training is the long pole), Phase 6 one to two
days, Phase 7 two days. No new ceiling. Every laptop GPU job runs detached inside WSL (the EXP-016
pattern) with the lid open.

## 6. Mapping to PLAN.md

| PLAN.md | this plan | state |
|---|---|---|
| Phase 5, DPO | dropped | section 4 |
| Phase 6, test-time compute | Phase 5 (5A done, 5B next) | section 1 |
| Phase 7, distillation, quantisation, serving | Phase 6, the traces; distillation and serving dropped | sections 2, 4 |
| Phase 8, write-up | Phase 7 | section 3 |
| D4's `distilled-0.6b` and D7 | not delivered; the reason in section 4 | |
| D6 | delivered in full by 5A + 5B | |

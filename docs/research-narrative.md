# The chain of questions: what we knew, what we suspected, what we asked, what we ran, what it said

Started 21 Oct 2026, after Phase 4 session 1, at the user's request: no experiment is launched
unless this file says what we knew before it, what we believed, which question it answers, and
what result would change our mind. The other documents hold the parts: `experiments.md` is the
pre-registration (hypothesis, metric, falsifier per run), `outputs/reports/phase-N.md` the numbers
with their CIs, `docs/phase-N-journal.md` the chronology, `docs/phase-N-engineering-notes.md` what
was built and how it is tested, `COMPUTE.md` the cost. This file is the thread through them: one
section per phase, in the order the questions arose. Every number here is quoted from a report and
carries its experiment id, so the source is one hop away.

**How to add a link.** Before a run goes into a session table: (1) the results that motivate it, with
ids; (2) the intuition, stated as a mechanism; (3) the question, as one sentence; (4) the
prediction, and the result that would falsify it; (5) the run's id in `experiments.md`. After the
run: (6) what it said, and (7) what it raised. A run without (1)-(5) does not run.

---

## Phase 0 (14 Oct 2026): a harness we can trust

**Known before.** Nothing measured. The project's claim is methodological: controlled experiments
with CIs, paired comparisons, from-scratch references with tests, an honest ledger. None of that is
possible if the evaluation itself is noisy or wrong.

**Intuition.** Most "RL improved reasoning" numbers in the wild are single greedy runs with no CI,
on an extractor that silently changes what counts as correct, under a sampler that is not
deterministic across batch sizes. The first thing to build is an evaluation whose numbers
reproduce bit for bit and whose extraction rules are logged.

**Questions.** Does the harness agree with itself (three identical greedy evals)? Do the from-scratch
GRPO pieces (pass@k, group advantages, k1/k2/k3 KL, the clipped loss with its three aggregations)
match their definitions under tests?

**Run.** EXP-000: three greedy evals of the 0.6B base on 100-problem subsets. They differed until
vLLM's batch-invariant mode became the default; then they were bit-identical. 95 tests on CPU.

**What it raised.** The harness is calibrated against itself, not against the world: Phase 1 has to
reproduce published numbers before any checkpoint is compared to anything.

## Phase 1 (14-18 Oct 2026): where the base model stands, and how much is latent

**Known before.** Phase 0's harness. Qwen3's technical report gives 4-shot-CoT numbers for the two
base models.

**Intuitions.** (a) A base model that has seen instruction-style maths in pretraining will do better
with a bare instruction than with four Minerva shots. (b) The think-tag format the RL phases need
is foreign to the base model and will cost accuracy before any training. (c) Most of what RL will
later "add" is already reachable by sampling: the pass@k ceiling is the reference point for the
sharpening-versus-capability question (Yue et al.). (d) GSM8K is partly memorised in every open
model; a perturbed copy (numbers and names changed) measures how much.

**Questions.** Q1's baseline row (where each prompt puts the base model); Q5's ceiling (pass@64 per
set); Q9's controls (perturbation gap, partial-prompt completion rate) before any training touches
the model.

**Runs and what they said.**
- EXP-001 (calibration): 4-shot-CoT GSM8K 74.5 [72.0, 77.0] against the report's 75.4; MATH-500
  41.4 against 43.5 on full MATH. Inside the pre-registered tolerance; the harness is calibrated.
- EXP-001 (the prompt table): zero-shot beats 4-shot-CoT by 3 points on GSM8K and 12 on MATH-500
  (77.6 / 53.6 against 74.5 / 41.4): intuition (a) held. Think-tag greedy is 61.3 / 43.0 with 36 % of
  GSM8K completions looping to the cap: intuition (b) held, and harder than expected. Countdown
  under think-tag is 1.2 %.
- EXP-002 (the ceiling, n = 64): pass@64 98.8 GSM8K, 87.8 MATH-500, 63.8 Countdown, 20.0 AIME24.
  Against greedy 77.6 / 53.6 / 1.2, the latent headroom is large everywhere and enormous on
  Countdown (intuition (c)): a method that moves greedy without moving pass@64 is sharpening.
- EXP-003 and EXP-003b: the base model's perturbation gap and partial-prompt completion rates, and
  the names-only control that separates recall of a problem from arithmetic on changed numbers.
  These are the rows every later checkpoint's controls are read against (Q9).

**What it raised.** The think-tag deficit is the elephant: before RL can be judged, the format has
to be taught, and the teaching (SFT) will be judged on whether it closes that deficit rather than
on absolute accuracy. And the zero-shot rows set a bar the whole pipeline will be held to: a
trained checkpoint that loses to the base model's own best prompt has not added anything a user
could not get for free.

## Phase 2 (15-19 Oct 2026): SFT on distilled traces, and what the format costs

**Known before.** Phase 1's rows. Distilled chain-of-thought data (OpenR1-Math-220k) exists in the
think-tag shape.

**Intuitions.** (a) SFT on competent traces teaches the format and a style; the gain in the trained
format will be large, the gain under the model's best prompt small. (b) More distilled data helps
with diminishing returns (a scaling curve, not a cliff). (c) The learning rate matters more than the
schedule at 23k examples; a three-point sweep on a subset picks it. (d) Training data must be
checked for eval overlap before a single number is believed (Q9).

**Questions.** Q1's SFT term; Q10 (which MATH levels move); the data-scaling slope (EXP-008); LoRA
versus full fine-tune at the same data (EXP-007; cut for budget, the pre-registration stands).

**Runs and what they said.**
- EXP-004 (overlap): 0 contaminated GSM8K items, 3 of 500 MATH-500 (dropped from training), 0
  AIME24. The eval numbers are not training-set recall.
- EXP-005 (LR sweep on 10 %): 1e-5 wins.
- EXP-006 (SFT, 23k traces, 2 epochs): think-tag greedy 74.8 GSM8K (+13.0 [10.0, 15.8] over the base
  in that format), 50.6 MATH-500 (+5.8 [0.6, 11.0]), 3.8 Countdown. Zero-shot 82.2 / 63.2: the
  trained format still loses to the model's own best prompt (intuition (a) held), because 28 % of
  GSM8K and 52 % of MATH-500 think-tag generations never terminate, at any cap, mostly a greedy
  degeneration (at T = 0.6 the truncation is 13 %).
- EXP-008 (data scaling, paired): 25 % to 100 % of the data buys +4.0 [1.4, 6.7] on GSM8K and +2.8
  [-1.4, 7.0] on MATH-500 (inconclusive). Intuition (b) held on GSM8K; MATH-500 is under-powered.
- EXP-002's SFT half: pass@64 unchanged on GSM8K, +5.2 on MATH-500. The format training did not
  move the ceiling on the easy set.

**What it raised.** The loops. A quarter to a half of the trained format's generations do not
finish. Every later gain in that format will be read through this: did the method make the model
better at mathematics, or better at stopping? That question, first written here, turns out to be
the question of the whole project (Phase 4, section 4.5 of its report). It also fixed the Phase 3
design: rejection sampling keeps only *finished* correct traces, so it will select for stopping
whether or not that was the intent, and the controls have to separate that from the data effect.

## Phase 3 (18-19 Oct 2026): one round of selection

**Known before.** Phase 2's checkpoint and its loops; the base ceiling.

**Intuitions.** (a) The SFT policy already solves many training prompts at T = 1.0; keeping its own
correct finished traces and training on them (RFT) is a cheap form of on-policy learning that
should move greedy more than it moves pass@k (sharpening). (b) Keeping the *shortest* correct traces
will push the policy to stop. (c) The gain must be compared to "the same number of gradient steps
on the distilled data" or it is just more training. (d) Only prompts the policy solves sometimes
but not always (solve rate in (0, 1)) carry signal for RL; the solve-rate distribution tells us how
many there are.

**Questions.** Q1's RFT term; Q5 for RFT (sharpening or capability); Q7 (transfer to Countdown);
Q9 (the controls after training); and "is it the data or the steps?".

**Runs and what they said.**
- EXP-009 (solve rates, n = 8 at T = 1.0 on 7,388 prompts): 4,606 learnable prompts (hash
  281bdfe31176411b); the kept set 11,145 traces (two shortest correct per problem, under a 2,048
  cap: a harness bug set the cap, recorded as a deviation).
- EXP-010 (RFT vs SFT, paired, think-tag): +3.3 [1.0, 5.8] GSM8K, +3.0 perturbed, +5.8 [1.6, 10.2]
  MATH-500, +3.2 Countdown; truncation 28 to 15 % and 52 to 40 %. Three seeds (EXP-010d): +4.6
  (3.3-5.5) and +4.3 (2.6-5.8).
- EXP-010b and EXP-010e (the controls): the same examples or the same tokens of *distilled* data
  gain nothing (74.8 to 74.1 GSM8K). Both controls trained on a seen subset of the SFT data, a
  design error recorded in the report; the claim they support is "not more steps on the distilled
  distribution", not "not fresh traces".
- EXP-010c (8,192-token cap): the truncated generations do not finish with more room, they loop;
  accuracy moves by under a point. The loops are not slow solutions.
- EXP-010f (pass@k in the trained format, T = 0.6, n = 32): a +2.3 / +2.8 pass@1 gain that persists
  at +2.0-2.5 to k = 32 on MATH-500; under 4-shot CoT the SFT and RFT curves coincide at every k.
- Q7: no Countdown transfer at any cap. Q9: gap and probe unchanged.

**What it raised.** Two mechanisms are confounded in RFT's gain: the correctness filter and the
"finished, short" filter. Pulling them apart needs two cheap ablations (no correctness filter;
random instead of shortest traces), written as EXP-010g and EXP-010h and run in Phase 4 session 1.
And the question Phase 2 raised sharpened: RFT's gain looked like termination too (truncation fell
by the same order as accuracy rose), but it had not been decomposed. Phase 4 would start from `rft`
(the pre-registered decision), so GRPO would be judged against a policy that had already learnt
some stopping.

## Phase 4, session 1 (20-21 Oct 2026): GRPO against selection

**Known before.** Everything above; a reference GRPO loop checked against TRL to 1.4e-5 in fp32
(the laptop, 20 Oct); the measured H100 rates.

**Intuitions, written in the plan before the runs.** (a) Two hundred steps of GRPO at 32 x 8 will beat
`rft` by 2-5 points on MATH-500 and 1-3 on GSM8K in the trained format, mostly as sharpening, with
a small persistent pass@k gain like RFT's. (b) A reward of 1 only for finished, well-formed, correct
completions will push truncation below 10 % / 25 %. (c) Countdown will not move (no Countdown in the
prompts). (d) The learnable-prompt filter made for `sft` will leave 20-40 % of groups with zero
variance for the `rft`-initialised policy. (e) The seed-to-seed difference of a 200-step run is
under 2 / 3 points. (f) The budget: shape B (32 x 8, 1,536 cap, 200 steps) instead of the plan's
64 x 8, 2,048 cap, 400-600 steps, because the latter cost 90 H100-hours; the cut was a cost decision,
and its risk (the cap and the step count) was named in the plan.

**Questions.** Q1's GRPO term; Q5 after RL; Q7; Q9; Q10; EXP-013's noise floor for the ablations;
the two Phase 3 ablations.

**Runs and what they said.**
- EXP-010g (no correctness filter): lands on `rft` (+0.2 GSM8K, -1.4 MATH-500, CIs covering 0),
  but the set it trained on was only 6.9 % incorrect because the shortest finished traces are mostly
  correct ones. Inconclusive by construction; a matched-incorrect set is the test with power.
- EXP-010h (random two instead of shortest two): -2.4 [-4.7, -0.1] on GSM8K with truncation 7 points
  higher; MATH-500 unchanged. About half of RFT's GSM8K stopping effect is length selection; on
  MATH-500 none of it is.
- EXP-012 (GRPO vs `rft`, paired, think-tag): +4.1 [2.0, 6.1] GSM8K, +4.8 [1.9, 7.6] perturbed,
  +7.0 [2.8, 11.0] MATH-500, +7.0 [3.6, 10.4] Countdown. Intuitions (a) and (b) held and were
  exceeded; (c) failed: Countdown doubled; (d) held at 30 %, flat through training.
- EXP-013 (seed 1): -0.3 / -0.2 / -4.8 [-9.0, -0.8] / +0.6 against seed 0. Intuition (e) failed on
  MATH-500; the noise row is 0.3 / 0.2 / 4.8 / 0.6.
- The decomposition (the user's question, report section 4.5): the whole paired gain sits on
  problems `rft` truncated (+4.2 of +4.1 GSM8K, +10.2 of +7.0 MATH-500, +11.2 of +7.0 Countdown);
  among problems both policies finished the paired difference is +1.1 [-0.6, 2.8] and 0.0 [-4.3,
  3.9]. With EXP-010c (the loops do not finish at 8,192 either), the policy change at greedy is
  "stop looping and answer". The same split on Phase 3's RFT step shows the same shape: both
  post-SFT stages at this scale are termination gains in the trained format.
- Q5 after RL (pass@k, T = 0.6): GSM8K pure sharpening (pass@16 -0.3; no problem newly solved in 32
  tries); MATH-500 +1.9 [0.4, 3.5] at k = 16 and 10 problems newly solved against 2 lost: a small
  widening, the same size RFT's was.
- Q9: gap 11.1 (`rft` 11.6), probe unchanged. Q10: levels 1-3 carry the gain, level 4 none.
- The method: stable, KL 0.006, no hacking signal (the one flag that fired lacked a correctness
  term and now has one), reward plateaued by step 50, three quarters of the steps bought little.

**What session 1 raised.** Against the base model's best prompt the pipeline stands at 84.3 / 64.4
(zero-shot) against 77.9 / 53.8: +6 on GSM8K and +11 on MATH-500, so the three stages did add
something a user could not get for free; but in the trained format, where the methods were
scored, every gain since SFT is termination. Two readings compete: (i) at 1.7B and this budget,
on-policy methods teach stopping and little else; (ii) the finished-only reward under a 1,536-token
cap, with the p90 length pinned at the cap throughout training and a quarter of completions scored
0 for being cut, *selected for brevity over solving*, and the missing reasoning gain is an artefact
of the cut budget. Reading (ii) is testable for one run's cost, and it bears on the project's
central claim; the normalisation ablation (A3, Dr. GRPO) bears on the method's length behaviour.
The user chose, on 21 Oct, to run the cap test (EXP-015) in session 2 in A3's place and defer A3.

## Phase 4, session 2 (21 Oct 2026): three knobs, and the cap test

**Known before.** Session 1's decomposition; the noise row (0.3 / 0.2 / 4.8 / 0.6); the measured rates.
The session 2 table as pre-registered: KL off (EXP-014a), dynamic sampling (EXP-014b), the 2,048 cap
(EXP-015) in A3's slot.

**Intuitions, written before the runs (the table of 21 Oct, kept in `docs/phase-4-plan.md` section 9).**
(a) The KL penalty is a safety margin, not a source of accuracy, at 200 steps: KL off lands within
noise with 2-5x the drift. (b) Refilling the batch with informative groups is the same optimiser steps
on more signal: +1-3 on MATH-500 at 1.3-1.6x the sampling. (c) The cap turned the reward into a
brevity reward; with 2,048 tokens the both-finished difference rises by at least 2 and pass@16 by 3.

**Runs and what they said.**
- EXP-014a (KL off): the hypothesis failed in sign. +2.2 / +3.3 / +5.6 over the main run on GSM8K,
  perturbed and MATH-500 (the last above the noise row), at half the length and a third of the
  truncation; offline KL to `rft` 15x the main run's (0.098 against 0.006 per token). Decomposed, +5.4
  of the +5.6 sits on problems the main run truncated (both-finished +2.7 [-1.3, 6.5]); under zero-shot
  the gain is zero (-1.5 / +0.4 / +1.8, CIs covering 0) while the brevity carries. The penalty was a
  brake on stopping; without it the policy stops sooner and the trained-format score rises. One
  non-termination cell: perturbed GSM8K both-finished +2.2 [0.1, 4.3] (noise row 0.2), single-seed.
- EXP-014b (dynamic sampling): the falsifier fired. Inside the noise row everywhere at 1.5x the step
  time; the learnable-prompt filter left 32 informative groups in 199 of 200 steps (one zero-advantage
  fill in the run). DAPO's fix addresses a batch the filter had fixed.
- EXP-015 (2,048 cap): the falsifier fired on both counts. Both-finished against `rft` 0.0 [-4.1,
  4.1] (main 0.0 [-4.3, 3.9]); pass@16 +0.8 [-0.6, 2.3] (main +1.9 [0.4, 3.5]); against the main run
  inside the noise row on every set, with longer completions and *more* truncation at eval. The cap
  is not what limits the reasoning gain.

**What session 2 raised.** Four things.
1. The "termination only" result is a property of the method at this scale, data and budget: a longer
   cap, a looser policy (no KL) and a better-filled batch all buy more or less of the same stopping,
   and nothing on problems both policies finish. Reading (ii) of session 1 is closed; reading (i) stands.
2. No on-policy stage after SFT has moved the best-prompt (zero-shot) accuracy beyond noise: RFT -1.1 /
   +1.6, GRPO +3.2 / -0.4, KL off -1.5 / +1.8. The pipeline's template-independent gain is SFT's. The
   question for any further RL spend is whether the recipe can add anything a user sees without the
   trained format.
3. The prompts may be the limit rather than the method: the pool is MATH levels 1-3 plus GSM8K, the
   eval gains sit on levels 1-3, 30-50 % of groups were all-correct and the reward plateaued by step 50.
   The policy may have seen nothing it could not already do.
4. If the gain is stopping, the question becomes how cheaply stopping can be bought: a decoding rule
   (budget forcing on `sft`, no training), an off-policy preference pair (DPO on finished-versus-looping
   traces, no rollouts), rejection sampling (12 minutes), or GRPO (3 hours). The cost per point of
   termination is the comparison the plan's Phases 5 and 6 should make, and the first two cost nothing
   or almost nothing.

Not changed by session 2: the Countdown gain (every run +5 to +8 over `rft`, all on `rft`-truncated
problems), the perturbation gap and the probe (no memorisation signal), the seed row.

## Open now (21 Oct 2026, after session 2): the next phases

Each row is a candidate for the next plan page (`docs/phase-5-plan.md`); the entries exist in
`experiments.md` as "proposed" and none runs before the user's signal and, for the two that rent a
GPU, a new ceiling. Order: the free ones first, because they bear on whether any further rental is
worth it.

| run | what we know | the intuition | the question | prediction / falsifier | cost | id |
|---|---|---|---|---|---|---|
| budget forcing on `sft` and `rft` | every post-SFT gain in the trained format is termination; the loops hold partial derivations (EXP-010c: they do not finish at 8,192) | forcing `</think><answer>` at a budget makes the model answer from what it has; if that recovers the gain, the training taught what a rule provides | how much of RFT's and GRPO's gain does a decoding rule recover with no training? | `sft`+force recovers at least half of RFT's +3.3 / +5.8 and a third of GRPO's cumulative gain / under a quarter at every budget: the loops are not usable derivations | laptop, INR 0 | EXP-016 |
| accuracy against tokens from the existing samples | n = 32 and n = 64 sample sets of base, `sft`, `rft`, `grpo`, cap exist; pass@k says sharpening on GSM8K, at most a small widening on MATH-500 | at matched tokens, voting over the earlier checkpoint may equal the RL policy's greedy number | is the GRPO policy the cheapest way to its own accuracy? | maj@4 of `rft` at or above `grpo` greedy on MATH-500 / maj@k of `sft` and `rft` below `grpo` greedy at every k to 32: the first capability signal of the project | INR 0 | EXP-017 |
| GRPO on MATH levels 3-5 | gains sit on levels 1-3; level 4 flat; 30-50 % all-correct groups; reward flat after step 50 | the policy saw nothing it could not already do; harder prompts with solve rate in (0, 0.5] carry the gradient the easy pool lacked | is the termination-only result about the prompts? | levels 4-5 at least +4 paired over `rft` with both-finished at least +2 / both inside EXP-012's CIs: the prompts were not the limit either | about 4.5 H100-h, INR 1,700 | EXP-018 |
| DPO on finished-correct vs looping pairs | the EXP-009 samples of `sft` hold a correct finished trace and a looping one for most prompts; RFT's 12-minute step bought most of the stopping | a preference pair is the cheapest signal for "stop"; no rollouts, no reference forward at train time beyond `sft` | can an off-policy method buy the termination gain at a tenth of GRPO's cost? | within the noise row of `rft` with truncation at or below `rft`'s / below `rft` by more than noise with truncation not falling | about 1.5 H100-h, INR 800 | EXP-019 |
| Dr. GRPO normalisation (deferred) | lengths fell under the standard loss in every run; a1 shows the KL term, not the normalisation, held length up | the normalisation's length bias is invisible under a finished-only reward | does the normalisation grow length here? | the pre-registration stands | about 3.5 H100-h, INR 1,400 | EXP-014c |
| matched-incorrect RFT set | EXP-010g had no power (93 % correct by construction) | a set with as many wrong as right finished traces separates selection from self-distillation | is the correctness filter the mechanism of RFT? | below `rft` by 2 / 3 points with truncation unchanged / within noise: self-distillation plus stopping | about 0.5 H100-h, INR 300 | EXP-010i |

Distillation and serving (PLAN.md Phase 7) have no question pulling on them from the data and stay an
engineering deliverable, last or cut. Phase 6's reward-model best-of-N is dropped unless EXP-017 shows
voting matches the policy, in which case a verifier is the natural next ask.

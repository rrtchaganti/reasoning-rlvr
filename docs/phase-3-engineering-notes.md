# Phase 3 engineering notes: rejection-sampling fine-tuning

What was built, how it was tested, why it is the way it is, what it owes, and the questions it
should be able to answer in an interview. Companion to `outputs/reports/phase-3.md` (results),
`docs/phase-3-plan.md` (the plan and its decisions), `docs/phase-3-journal.md` (what happened, in
order) and `experiments.md` EXP-009 to EXP-010f. Written 19 Oct 2026.

## 1. What was built (18 Oct, laptop)

- `rlvr-eval` training-pool flags: `--levels 1,2,3`, `--exclude-dev`, `--pool-limit EVAL=N`. The
  pool is written into the eval set's source string and manifest, so a sample file says which
  problems it was drawn from.
- `rlvr-build-rft` (`src/rlvr/data/rft_data.py`): per-problem solve rates from an n-sample run, the
  histogram per source and MATH level, the kept set (correct, finished, one closed think block, one
  answer block, distinct reasoning, the shortest two per problem, prompt + completion under the
  cap) in the SFT schema with the think-tag prompt rebuilt from the loader, decontamination against
  every eval set with the Phase 2 detectors, a manifest the trainer checks, and the learnable-prompt
  file (0 < rate < 1) for Phase 4 with its own hash. Eleven tests on hand-built samples, including
  an end-to-end build with a planted contamination hit.
- Configs for the n = 8 pool, the RFT run, the control, the RFT evals and the n = 64 samples;
  `scripts/phase3_rental.sh` (smoke, sample, data, train, eval, sample64, control, report;
  resumable) with the cap rule and the Countdown skip as environment variables.
- On 19 Oct, after the session: the harness fixes the node had worked around (commit 3e055ad):
  flags override configs at any value, a derived set's `--limit` follows its parent's draw, numeric
  YAML strings are cast, the model card's `base_model` can be given as a Hub id, `setup_env.sh`
  removes `torchcodec` on the cu129 route and installs a cached flash-attn wheel.

## 2. The kept-set rules, and what each one is for

One line per rule, because each is a decision a reviewer may question:

- *Correct* (math-verify against the gold answer): the reward. Without it this is self-distillation.
- *Finished* (`finish_reason` is not `length`): a truncated trace has no answer to verify and, if it
  were kept, would teach the model to run on. 8,593 samples dropped for this alone.
- *One closed think block, one answer block*: the format the evals and the RL phases read. 19 dropped.
- *Distinct reasoning*: two samples with the same think text are one example.
- *Shortest two per problem*: the cap stops easy problems (8 of 8 correct) from dominating the set
  (18,709 dropped for the cap), and "shortest" is a deliberate bias toward traces that stop. It is
  also the main reason the RFT checkpoint's truncation fell; section 7 lists it as a confound.
- *Under the token cap*: training rows must fit the 4,096 window.
- *Decontaminated*: the Phase 2 detectors (13-gram with coverage, fuzzy 0.9) against every eval set
  and the dev split; 200 traces dropped, 133 of them because 36 MATH-500 problems have a
  near-duplicate among the MATH-train prompts.

## 3. The controls, and the one that was not what it said

The pre-registered control (EXP-010b) was "one more epoch from `sft` on a fresh seeded slice of the
OpenR1 set with as many examples as the RFT kept set". The code for it set `limit` and `subset_seed`
on the SFT training file, which is the 23,000 traces the SFT checkpoint had already trained on
twice; `train_ids.json` of each control is a subset of the SFT run's. The node's notes called the
data fresh because the script's comment did. It was found on the laptop by checking the ids.

Why it could not have been fresh: the Phase 2 funnel kept every correct, single-answer, under-cap
trace in the source, all 23,000. A fresh slice would have needed a second source or a looser
funnel, and the pre-registration never said which. The lesson is procedural: a control's data
provenance is a test (ids disjoint from the training run's), not a comment.

What the controls still establish, because they share RFT's backend, start, LR and schedule: at
equal tokens a third pass over the distilled traces moves the SFT checkpoint by about nothing and
leaves its truncation where it was; the same tokens of selected self-generated traces move it by
+4.0 and +7.2 and cut truncation by a quarter. "The gain is not more gradient steps on the distilled
distribution" is supported; "the on-policy data is the point" is the likely reading, untested.

## 4. Eval protocol for the RFT checkpoint

As Phase 2's (engineering notes, section 7), with three additions. Paired pass@k for n-sample runs
(`compare_checkpoints.py --all-samples`): the unbiased pass@k estimator per problem on both runs,
the paired bootstrap of the difference, and the new-versus-sharpened counts (solved at least once
in n tries by one run only). Seeded subsets (`--limit 300`) when the full set is unaffordable: the
pairing stays valid on the shared ids and the CI widens by about the square root of the ratio
(1.3x on MATH-500, 2x on GSM8K). And the post-hoc runs of the gap analysis, each with a written
"result that would change the conclusion" before it started (EXP-010c to EXP-010f), which is the
closest a post-hoc run can come to pre-registration.

Two protocol holes found in this phase and fixed: an `rlvr-eval` flag equal to its default lost
to the config (EXP-009 sampled at 2,048 tokens instead of 4,096; an n = 32 run sampled five sets
instead of two), and `--limit` on the perturbed GSM8K set drew a subset independent of GSM8K's,
leaving 67 pairs of 300 for the gap. Both have regression tests (3e055ad).

## 5. The session's rules, and who decided what

The node session ran under written rules (handoff section 7) and, from 04:05 UTC, four rules the
user replaced them with: keep or stop by the known reasoning-model trajectory; actual against
estimated time after every run; the budget may grow while the project is on track; no
near-duplicate hyperparameter runs. The user decided the 300-problem subsets (04:20) and the list
of post-hoc runs (08:33, on the laptop's recommendation). The node decided alone: to keep the
2,048-cap samples (05:02), to run the control under the new rules, and the order of the post-hoc
runs. Every decision has its reasons in the journal at the minute it was made.

## 6. Budget

The plan's Phase 3 line was 4.5-7 A100-hours without a Countdown sampling. Actual on the H100:
core 4.1 h (the training steps came in at a fifth of their estimates because packed 4,096-token
rows cover 11,145 traces in 53 steps; the sampling steps at two thirds), post-hoc runs 4.0 h, the
analysis 0.9 h of idle GPU, archive and Stop 1.3 h: about 10.3 node-hours, INR 3,900. Measured
rates for planning: RFT-size training 12 min; a five-set greedy think-tag eval 8-10 min; n = 8
sampling of 7,388 prompts at a 2,048 cap 1 h 19 min; n = 64 on 300-problem subsets of four sets
1 h 22 min; the full-set n = 64 of a long-writing checkpoint 3 h 45 min.

## 7. Known limitations and debts

- The control is not a fresh-data control (section 3). A real one needs a second source of
  distilled traces; the ablation that comes closest with the data we have is "RFT without the
  correctness filter" (same number of self-generated traces, kept at random), which separates
  selection from self-distillation. Needs a `--keep-incorrect` option in `rlvr-build-rft`; priced at
  20 minutes on the H100.
- Length selection is confounded with correctness selection: "shortest two per problem" and the
  2,048 cap both keep short traces, and the RFT checkpoint's main visible change is that it stops.
  Two experiments would separate them: EXP-009 at 4,096 (priced at 2.7 h, not run) and a kept set
  that takes two *random* correct traces per problem instead of the shortest.
- The MATH-500 level pattern (gain on levels 4-5, not the trained 1-3) is post hoc at n of 128-134
  per level and is a lead, not a finding.
- The SFT checkpoint has one seed; RFT's seed-to-seed spread (up to 3 points) is the measured noise
  floor, and a second SFT seed (2.2 h) would say whether the SFT side's noise is similar.
- The SFT checkpoint trained unpacked with sdpa and every Phase 3 run packed with flash attention.
  Same loss, different micro-batch composition; the controls share RFT's backend, so the causal
  comparison is matched and the SFT-versus-RFT one is not.
- The n = 64 curve for RFT is on 300-problem subsets; the gap at n = 64 for RFT is unusable (67
  pairs) and the fixed harness would need a 1.2 h rerun to replace it.
- The RFT Countdown n = 64 sampling and the second iteration (EXP-011) were dropped before the
  session; the plan's diminishing-return row is not reported.
- The think-tag format still loses to zero-shot on the RFT checkpoint (78.2 against 81.1 on GSM8K,
  56.4 against 64.8 on MATH-500). One round of RFT halved the looping; it did not remove it.
- `rlvr-build-rft` names its overlap files `sft_openr1__*.json` (the Phase 2 prefix) inside
  `outputs/data/rft/overlap/`; cosmetic, but a reader will be confused.
- The learnable-prompt file and the per-problem solve rates live under `outputs/` (git-ignored), on
  the node's disk and in the session archive; the manifest with both hashes is committed
  (`data/manifests/rft_iter1.json`). Phase 4 must check the hash before training on them.
- The node's clock ran about two hours ahead of the console's; per-step hours are the node's, the
  total is the invoice's.

## 8. Results and what they mean (19 Oct 2026; tables in `outputs/reports/phase-3.md`)

- **EXP-009.** The SFT policy solves 54 % of MATH levels 1-3 and 75 % of GSM8K-train per sample at
  T = 1.0 within 2,048 tokens; 67 % and 59 % of problems are learnable (0 < rate < 1); 16 % of MATH
  problems and 4 % of GSM8K ones are never solved in 8 tries. The falsifier is far off. A third of
  GSM8K samples and three fifths of MATH samples hit the cap, so "rate" is "within 2,048 tokens".
- **EXP-010.** RFT beats SFT in the trained format: +3.3 [1.0, 5.8] GSM8K, +3.0 perturbed, +5.8 [1.6,
  10.2] MATH-500, +3.2 Countdown, paired, greedy; over three seeds +4.6 (3.3-5.5) and +4.3 (2.6-5.8).
  Truncation falls 12-13 points on every set. The other two templates do not move. Gap and
  completion rates unchanged.
- **EXP-010b and 010e.** At equal tokens an extra pass over seen distilled traces gives nothing;
  RFT gives +4.0 and +7.2 over it. Not steps, data.
- **EXP-010c.** The format deficit is looping, not the cap: at 8,192 tokens truncation is still
  27 % and the Countdown outputs run to 7,900 tokens. The RFT gain is the same at either cap. No
  Countdown transfer.
- **EXP-010f.** In the trained format pass@k never drops; a 2-3 point gain persists to k = 32 on
  MATH-500; at T = 0.6 the pass@1 gain is +2.3 / +2.8. Under few-shot the curves coincide.
- **What it means.** One round of selecting the model's own correct, finished, short traces buys
  3-6 points at greedy in the trained format and halves the looping that Phase 2 installed; it does
  not change what the model can reach in 64 tries, and it does not transfer. The mechanism the
  evidence supports is "stop-and-be-right" selection, not new capability. For Phase 4 this is the
  bar GRPO must clear, and a warning: a reward that pays for a correct answer after a loop will
  train loops.

## 9. Interview question bank for this phase

- *What is RFT and why run it before GRPO?* Expert iteration with a binary reward and no policy
  gradient: sample, verify, keep, fine-tune. It gives the cheapest on-policy improvement, the
  solve-rate distribution that Phase 4's prompt filter needs, and a bar: if GRPO cannot beat +3 to
  +6 points at greedy with a 20-hour run, the machinery is not paying for itself.
- *Your control was supposed to use fresh traces. Did it?* No; it used a seeded subset of the
  traces the model had already seen twice, because the funnel had kept every usable trace in the
  source and my code never checked provenance. What it still shows: more steps on that
  distribution do nothing at equal tokens, and RFT does. What it cannot show: whether fresh
  distilled traces would have done the same. I wrote the deviation in three places and the lesson
  is that provenance is a test on ids, not a comment.
- *How much of the RFT gain is just the model learning to stop?* Most of the greedy gain:
  truncation falls from 28 to 15 % on GSM8K and the accuracy of finished outputs barely changes;
  at an 8k cap the gain is the same, so the loops never end. Under sampling, which already
  rescues most of the SFT loops, the residual gain is 2-3 points and significant, and it persists
  to k = 32. So: mostly stopping, with a small real residual.
- *You sampled at 2,048 tokens when your rule said 4,096. Why did you keep it?* A flag equal to
  its default lost to the config; found at the data step. The kept set was large and the
  falsifier far off; 2,048 is the plan's own default for short traces and only changes which
  correct traces are kept, in the direction RFT pushes anyway; a rerun was 2.7 H100-hours for a
  result that differs mainly in keeping longer traces. Reported as part of the method, with the
  confound (length selection) named. The bug has a test now.
- *Is the gain above seed noise?* On GSM8K yes: three seeds, every one significant, spread 2.2
  points under a 4.6 mean. On MATH-500 borderline: spread 3.2 against a 4.3 mean, one seed's CI
  touching zero. The seed-to-seed difference is the noise floor for every single-seed comparison
  in the study, and I say so where one is quoted.
- *Does RFT sharpen or add capability?* Under the few-shot template the pass@k curves coincide at
  every k. In the trained format RFT sits above SFT at every k, including k = 32 on MATH-500 (10
  problems only RFT solves against 4), which is not "sharpening with lost diversity"; it is a
  uniform shift consistent with fewer non-terminating generations. The honest sentence: no loss of
  pass@k, a small persistent gain in the trained format, no new reach under the neutral template.
- *Why did the gain land on MATH levels 4-5 when the data was levels 1-3?* I do not know; it is
  post hoc at n of 128-134 per level. The candidate explanation is that short-trace selection
  helps most where the SFT model rambles longest, and level 4-5 is where its truncation is worst.
  It is a lead, not a claim.
- *The base model with a plain prompt scores 77.9 on GSM8K and your RFT checkpoint 78.2 in its own
  format. What did two phases of training buy?* On GSM8K, honestly, the format cost about what the
  training gained; the trained checkpoints under zero-shot (82.2 SFT, 81.1 RFT) are the gain over
  the base's 77.9. On MATH-500 the pipeline adds about ten points under the best prompt (53.8 ->
  63.2 -> 64.8) and twelve in its own format. The format is a cost until something teaches the
  model to stop; RFT is half of that something.
- *What would a reviewer still ask for?* The fresh-data or unfiltered control; a random-two
  instead of shortest-two kept set; EXP-009 at 4,096; a second SFT seed; the full-set n = 64 for
  RFT. All priced in the gap analysis; the first is 20 minutes on the H100 once the builder has a
  flag.
- *You asked the node agent to review its own results. Why trust that?* I did not; I asked it to
  list claims with CI widths and flags and to price experiments with a written "result that would
  change the conclusion" before any ran, and I chose which to run and read each verdict against
  its rule. The analysis cost 52 minutes of idle GPU and the runs four hours; the answers to the
  first two interview questions above come from them.

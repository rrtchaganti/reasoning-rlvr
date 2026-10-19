# Phase 3 journal: rejection-sampling fine-tuning (started 18 Oct 2026)

Chronological record of the phase: the context, each decision with the explanation given at the
time, each failure, what replaced it. Summary later in `phase-3-engineering-notes.md`; numbers in
`outputs/reports/phase-3.md`; post-mortems in `FAILURES_AND_LEARNINGS.md`.

## 1. Context on 18 Oct: code before compute

Phase 2 is not merged and its training runs have not happened: the first E2E session (an L4,
17 Oct) did the environment, the data and four of EXP-002's five sets, and the A100 session that
runs the SFT experiments has not been launched. The user's call: no idle GPU between phases. Plan
the GPU work on the laptop, write and test the code here, and let the A100 node run Phase 2 and
Phase 3 back to back, with a Claude Code session on the node taking the decisions that depend on
Phase 2 numbers under rules written in advance.

So Phase 3 was planned (`docs/phase-3-plan.md`) and built on 18 Oct without a single Phase 3 number
existing. The plan went to the user with six decisions marked; the user answered with the arrangement
above, dropped the second-iteration run (EXP-011), left the compute-matched control (EXP-010b) to my
judgement, and the rest to me. My calls, and why, are in the plan's section 6: the control runs
(Q1 asks for equal-compute comparisons, and one A100-hour is the price of being able to say "not
just more training"); MATH levels 1-3 are sampled in full and GSM8K-train is capped at 4,000 (Phase 4
draws its prompts from the MATH pool, so solve-rate coverage there is worth the hours); the sampling
cap is a rule the script applies from the SFT model's measured length, not a number guessed now.

The branch question needed a deviation from `WORKFLOW.md`: a phase branch opens from `main`, but
`main` does not have Phase 2. `phase-3-rft` was opened from the `phase-2-sft` tip instead, the node
clones it (it is a superset), and it rebases onto `main` after the Phase 2 merge.

## 2. What was built (18 Oct, laptop, no GPU)

- Two pool flags in `rlvr-eval`: `--levels` (MATH levels), `--exclude-dev` (the 250 + 250 dev
  problems drawn from the train splits must never be trained on), `--pool-limit EVAL=N` (a seeded
  cap). The eval set's `source` string records each filter, so the manifest says what the pool was.
- `rlvr-build-rft` (`src/rlvr/data/rft_data.py`): solve rates per problem with the histogram
  (EXP-009), the kept set (correct, stopped, one closed think block, one answer block, distinct
  reasoning, the shortest two per problem, prompt + completion under the cap) in the SFT schema with
  the think-tag prompt rebuilt from the loader, decontamination with the Phase 2 detectors, the
  manifest the trainer checks, and the learnable-prompt file (0 < rate < 1) for Phase 4 with its own
  hash. Eleven tests on hand-built samples, including the end-to-end build with a planted
  contamination hit.
- Configs: the n = 8 sampling pool, the RFT training run (LR 5e-6, 1 epoch, from `sft`), the control
  run, the three RFT eval templates and the n = 64 sample configs (copies of the SFT ones).
- `scripts/phase3_rental.sh`: `smoke`, `sample`, `data`, `train`, `eval`, `sample64`, `control`,
  `report`; the sampling-cap rule and the Countdown skip as environment variables.
- Pre-registration EXP-009, EXP-010, EXP-010b in `experiments.md`; handoff section 7 with the node
  session's rules and freedom; rental guide section 3b; README row.

One thing the tests caught in my own test data: four fake MATH questions that differed by one digit
were all flagged by the fuzzy detector (Indel similarity above 0.9), which is the detector doing its
job on a careless fixture. The fixture now uses four different questions and plants one exact copy.

Not done on the laptop: the GPU smoke (`bash scripts/phase3_rental.sh smoke` with
`RLVR_SMOKE_MODEL=outputs/train/sft_0p6b_lora_smoke_v2/final`, about ten minutes on the 4060). The
node runs the same smoke against the real SFT checkpoint before the session, so a local run is a
convenience, not a gate; if the user runs it, the result goes here.

Later on 18 Oct: the node is a TIR H100 notebook, not an A100 (Phase 2 journal, section 18). The
plan's hours stay in A100 units; the handoff's section 8 converts them (Phase 3 about 3-4 hours on
this card) and its section 7 rules now carry the H100 figures.

## 3. The session reaches Phase 3: a smoke that finds nothing (19 Oct, 03:35-03:45 UTC, the node session)

Phase 2 closed at 03:35 UTC (Phase 2 journal, sections 19-26) and the chain moved straight into
`phase3_rental.sh smoke`: 20 problems, n = 2, 512 tokens, three training steps. `rlvr-build-rft`
refused with "no usable examples": the SFT checkpoint writes 1,900-2,900 tokens per think-tag trace
and no sample finished in 512. Expected in hindsight; the laptop had smoked the step on the 0.6B
LoRA checkpoint, which writes short traces. A second smoke at 2,048 tokens passed in three minutes
and did the thing the smoke was really for: the training hand-off with `flash_attention_2` and
`packing: true` worked on this box (run_info: packing safe, three steps in 19 s), so Phase 3 ran
the pre-registered packed config rather than Phase 2's sdpa fallback. The smoke's numbers (SFT on
its own training prompts at 2,048 tokens: MATH levels 1-3 50 % with 65 % truncation, GSM8K-train
62.5 % with 37.5 %, n = 40 each) were the first hint of what EXP-009 would find. The cap is 2,048
in the script now.

## 4. EXP-009, the ten-minute rule, and the cap that was not applied (03:43-05:03 UTC)

The cap rule read the SFT think-tag p90 (about 3,960) and chose 4,096; the script passed
`--max-tokens 4096`. At ten minutes the projection said 0.9 h more for the MATH half at 19,475
output tokens/s (short prompts, many sequences in flight), well under the 2-hour limit, so the full
pre-registered pool ran: no reduction of the prompt set. MATH levels 1-3 took 42 minutes against
54 estimated, GSM8K-train 37 minutes; 1 h 19 min in all against 1 h 50 min.

The data step then printed the surprise: the samples had a 2,048-token cap. `rlvr-eval`'s
config merge treated a flag equal to its parser default (4096 is the default) as "not given" and
let the config's 2,048 win; the chain log showed the intended command, so nobody saw it until the
manifest said `sampling_max_tokens: 2048`. The consequences were in the counts: 59.5 % of MATH-train
samples and 34.9 % of GSM8K-train samples truncated, every kept trace under 2,048 tokens. The node
session decided alone (the user's rules from 04:05: no wasted compute, no near-duplicate runs, keep
or stop by whether the model sits on the known trajectory): keep the run and train on it. The
reasons, as written at 05:02: the kept set was large and healthy (11,145 traces, 4,606 learnable
prompts, the falsifier far off at 7-23 % rate-0 by level), 2,048 is the plan's own default for
shorter traces, the cap only changes which correct traces are kept (the shorter ones, which is the
direction RFT pushes), and a rerun would have cost 2.5-3 h of sampling plus the retrain for a result
that differs mainly in keeping longer traces. Reported as a deviation, offered to the user as a
rerun, never started unasked. The harness fix with a test is commit 3e055ad (19 Oct, laptop).

The solve rates themselves were on trend: 0.71, 0.60, 0.44 by MATH level; GSM8K-train less
saturated than the hypothesis said (37 % at 8 of 8 against "more than half") because a third of
its samples hit the cap. Decontamination of the kept set dropped 200 traces, 133 of them because 36
MATH-500 problems have near-duplicates among the MATH-train prompts, a known property of MATH that
the Phase 2 source (OpenR1) had mostly avoided (3 problems).

## 5. Twelve minutes of training, and a report step that failed on a string (05:03-05:38 UTC)

The RFT run had been estimated at 30-75 minutes. With packing, 11,145 traces of 1,207 tokens on
average fill 53 rows of 4,096 tokens at an effective batch of 64, and the run took 11 min 53 s at
13.1 s per step, 32.9 GB. The evals took 22 minutes. The result was the RFT signature the trend
check had predicted at 04:05: pass@1 up on every in-domain set (+3.3, +3.0, +5.8 paired), truncation
down 12-13 points everywhere, mean length 1,887 -> 1,426 on GSM8K, the other two templates
unmoved, the perturbation gap and the probe unchanged. The report step then died in
`push_to_hub.py`: the card's `base_model` was `outputs/train/sft_1p7b/final`, the path the run
trained from, and the Hub validates that field as a model id (HTTP 400). The compare tables had
been written before the failure; a manual script wrote the card with the SFT repo id and uploaded
the checkpoint; the laptop fix is a `--base-model-id` flag (3e055ad). One idle minute. A runner
script was edited while its loop ran and bash read the shifted file after the loop, so the runner
died with a syntax error after the control step had finished and logged its exit: no data lost, a
lesson logged (never edit a running bash script).

## 6. The control, the subset decision, and the first pass@k curve (05:38-07:41 UTC)

Two decisions the user took while awake. At 04:05 they replaced the handoff's budget rules with
four of their own: keep or stop by the known reasoning-model trajectory; actual against estimated
time after every run; the budget may grow while the project is on track; no near-duplicate
hyperparameter runs. Under those rules the control run (which the handoff's 18-hour rule would have
skipped) became the first thing to run after the RFT eval, and the LoRA ablation and the RFT
Countdown sampling stayed cut. At 04:20 they chose `--limit 300` seeded subsets for the RFT n = 64
sampling (1 h 25 min against 3 h 45 min on the full sets), paired on the shared ids with the full SFT
samples.

The control ran at 8 x 8 (132 steps, 26 min) and gave the number the report needed: an extra epoch
of SFT moves GSM8K +1.2 and MATH-500 -0.4 with truncation unchanged, so RFT's +6.2 [1.6, 10.6] over
it on MATH-500 is selection, not gradient steps. The node's notes called the control's data "fresh";
section 8 says why it was not. The subset sampling took 1 h 22 min (estimate 1 h 25 min) and gave
the pre-registered pass@k answer: under the 4-shot-CoT template the SFT and RFT curves coincide at
every k. Phase 3's core was complete at 07:41 UTC: 4 h 06 min against the 4-4.5 h estimate, with the
control and the subset sampling included because the training and sampling steps had come in far
under their estimates. Every actual-versus-estimate pair went into the notes.

## 7. The gap analysis and the five approved runs (07:41-12:27 UTC)

The user asked, on the laptop, for the node session to read its own results as a reviewer would
before the archive: every claim with its CI width and its flags (one seed, one checkpoint, small n,
unpaired, deviation), every question the evidence could not answer, and for each the smallest
experiment with its exact command, its measured cost on this card, and the result that would
change the conclusion, sorted by evidence per rupee, nothing started without approval by name. The
analysis took 52 minutes of idle GPU (INR 330) and produced 16 claims, 11 questions and 13
candidate runs. The laptop approved five, in order, with reasons written here: the 8k-cap eval
(the first question any interviewer would ask about the think-tag deficit), two more RFT seeds
(every training claim rested on one seed and RFT retrains in 12 minutes), a token-matched control
(the example-matched one had seen 2.2x the tokens), the n = 32 curve in the trained format (the
headline "no new ability" had been measured under a template the model was never trained in), and
the base 4-shot-CoT and zero-shot rows at the SFT cap (Phase 2's paired tables had used 1,024-token
base rows). Not approved: the n = 8 temperature run (superseded), the full perturbed n = 64 (low
value), a second SFT seed (INR 830; the backend point is not a confound because RFT and the control
start from that same SFT checkpoint), the 4,096-cap rerun of EXP-009 (INR 1,020 to confirm a
deviation that the report states as part of the method), LoRA and the RFT Countdown sampling (the
agreed drop list), and the two ablations that need code (unfiltered RFT, tagless SFT).

The runs took 4.0 hours against 3.8 estimated and each verdict was read against its written rule:
the deficit is looping, not length (truncation 27 % at 8,192 tokens too); the GSM8K gain is above
seed noise and the MATH-500 gain borderline; at equal tokens the control gains nothing; in the
trained format pass@k never drops and a 2-3 point gain persists to k = 32, with the greedy gain
mostly the removal of degenerate loops; and a new reviewer's point from the base rows: on GSM8K the
base's plain zero-shot (77.9) equals RFT think-tag (78.2), so there the pipeline recovers what the
format cost, while on MATH-500 training adds about ten points under the best prompt. The flag bug
bit once more during the n = 32 run (`--evals gsm8k math500` equals the parser default, so the
config's five sets ran and the SFT half was stopped after three, 10 minutes lost). Budget at the
end of the runs: 26.6 node-hours on the node's clock, INR 10,090.

## 8. Archive, Stop, and the laptop's reading (19 Oct, 12:40 UTC onward)

The node pruned 58 GB of intermediate checkpoints on the user's word (every `final/` kept on the
disk: 83 GB used of 339), wrote the Hub inventory and a redacted copy of its own transcript, packed
the archive (68 MB, 768 files) with its sha256, and uploaded a copy to the samples dataset. The
user downloaded it through the JupyterLab browser, checked the hash, and Stopped the notebook; the
console's usage report (to 16:01 IST) plus the three hours after it put the notebook at 28.1 hours
and INR 10,640, the disk at INR 10 per GB-month on 340 GB. The node's clock ran about two hours
ahead of the console's, so the per-step hours in these notes are the node's and the total is the
invoice's.

On the laptop, reading the archive found one thing the node had not: both controls trained on a
seeded subset of the SFT training file, which is the data the SFT checkpoint had already seen for
two epochs (`train_ids.json` of each control is a subset of the SFT run's). The pre-registration
said "a fresh seeded slice of the OpenR1 set"; the Phase 3 code set `limit` and `subset_seed` on the
same 23,000-trace file and never checked, and no fresh traces of that quality exist in the source
(the Phase 2 funnel kept all of them). The control therefore answers "is the gain just more
gradient steps on the distilled distribution" (no) and not "is it just more distilled data"
(untestable here). It is written as a deviation in the registry, the report and the engineering
notes, and as a design error of mine, not of the node's. The rest of the day's laptop work: the
harness fixes with tests (3e055ad), the rental docs (77da2be), the Phase 2 close-out (eb06c79) and
its merge, then this branch's close-out: the report (`outputs/reports/phase-3.md`), the registry
entries including the five post-hoc experiments with their pre-written rules, the ledger, the
engineering notes, the model card and the RFT manifest, and the Phase 4 plan's first page.

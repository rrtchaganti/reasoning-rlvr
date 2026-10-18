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

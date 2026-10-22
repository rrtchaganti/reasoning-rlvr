# Phase 5 engineering notes: the cost of termination (option A: no rental)

Branch `phase-5-ttc` (from `main` 648df74, the Phase 4 merge). Built 21 Oct 2026, run 21-22 Oct on
the laptop (RTX 4060 8 GB, WSL2) and from existing sample sets; written up 22 Oct 2026. Results:
`outputs/reports/phase-5.md`. Chronology: `docs/phase-5-journal.md`. Chain of questions:
`docs/research-narrative.md` (Phase 5 section). Pre-registration: `experiments.md` EXP-016, EXP-017
(done), EXP-018, EXP-019, EXP-010i (proposed, awaiting a budget).

## 1. What was built

- **Budget forcing in the eval harness** (`src/rlvr/eval/force_stop.py`; `rlvr-eval --force-stop-at B
  --force-stop-continue N`). A second generation pass rather than a custom decoding loop: the first
  pass runs at `max_tokens=B`; every completion with `finish_reason == "length"` and no `</think>`
  gets the template's `force_suffix` ("\n</think>\n<answer>", the exact ending of every SFT trace and
  finished sample) appended to prompt + completion, and a second `generate` call with `n=1`,
  `max_tokens=N` and the same stop strings writes the answer. The merged completion carries both
  parts, the continuation's finish reason and `forced=True`; the per-sample record carries
  `forced_stop` (default False, so files written before the field load unchanged); the summary adds
  the forced rate, the forced answers' accuracy and their finished rate; `metrics.json` keeps the
  pass's stats (`force_stop`). The `think_tag` template is the only one with a suffix; the others
  raise. vLLM's prefix cache makes the second pass cheap (the continuation prompts share the first
  pass's prefix). Tests on a scripted sampler (`tests/test_force_stop.py`): the targets, the
  continuation config, the merge, the stats, grading of a forced answer, the CLI flags.
- **`scripts/plot_accuracy_vs_tokens.py`** (EXP-017): maj@k and pass@k per checkpoint against the
  expected generated tokens per problem, restricted to the problems every run shares, with paired
  CIs against a reference greedy run; per-problem maj@k re-implemented (the metrics module's
  `maj_at_k` returns only the mean, and the CI and the paired difference need the per-problem
  values). Test on synthetic sample files.
- **`scripts/plot_force_stop.py`** (EXP-016's figure): accuracy against mean tokens per budget with
  the unforced greedy runs as reference points; a Markdown table with every number. Test on synthetic
  metrics files.
- **`scripts/phase5_laptop.sh`**: `smoke` (the 0.6B base at budget 64 on 8 problems), `download`
  (the root checkpoints only; vLLM's Hub loader pulls every `model.safetensors` in a repo, and the
  SFT repo holds six more under `ablations/` and `controls/`), `exp016` (checkpoint-outer: download
  one, run its five budgets, then the next), `exp016-ctrl` (the unforced runs on this machine,
  optional, not run). Resumable: a run with `metrics.json` is skipped.

## 2. Design decisions

- **Force as a stop, not as a "wait".** s1's budget forcing appends "Wait" to extend reasoning; the
  question here is the opposite, whether the loops already hold a usable derivation. Appending the
  closing tags reads what is there.
- **The suffix matches the trained ending exactly.** "\n</think>\n<answer>" is how every finished
  think-tag completion ends; the model's continuation (" 42 </answer>") is then in-distribution. A
  different suffix would test the rule and the format shift at once.
- **128 continuation tokens.** The answer block is short; 99-100 % of forced completions finished
  within 128 tokens at every budget, so the number does not bind.
- **Paired against the archive's greedy runs (H100, 4,096 window), not re-run on the laptop.** Same
  checkpoint, same harness, same batch-invariant vLLM, a different machine. `exp016-ctrl` (the
  unforced run on the laptop) exists to measure the machine effect and was left out for time; the
  EXP-000 result (batch-invariant greedy decoding is bit-identical across runs on one machine) does
  not cover two machines, so a cross-machine difference of a few tenths of a point is possible and
  is listed as a caveat in the report.
- **Budgets 768-3,072** bracket the training caps of Phase 4 (1,536 and 2,048) and the SFT model's
  mean finished length (about 1,900 on GSM8K, 2,800 on MATH-500).

## 3. Eval protocol

As Phase 4: greedy, think-tag, GSM8K (1,319) and MATH-500 (500), 4,096 window, batch-invariant vLLM;
paired bootstrap CIs on shared problem ids against `sft`, `rft` and `grpo`; the Phase 4
decomposition (`scripts/decompose_gain.py`) of each forced run against its own unforced checkpoint.
EXP-017 on the pinned 300-problem subsets with paired CIs against `grpo` greedy.

## 4. Laptop runs: what it took

Ten forced-stop evals (two checkpoints, five budgets) at 30-66 minutes each on the 4060 (longer
budgets take longer; the MATH-500 half dominates), about 8 hours of GPU in all, plus the two
checkpoint downloads (6.9 GB each, fp32; 0.4-0.65 MB/s on the night of 21 Oct, about 3 MB/s the
next morning). The job ran detached inside WSL (`setsid nohup`) because the session's own
background shells get reaped on low host memory; the laptop's hibernate-on-mains timeout was set
to never for the run and restored after. A closed lid suspended the laptop for most of the night
(the detached job resumed on wake with nothing lost); the lid action is separate from the idle
timeouts and was not changed.

## 5. Known limitations and debts

1. **Cross-machine pairing** (section 2): the unforced references are the H100 runs; `exp016-ctrl`
   would measure the laptop-versus-H100 greedy difference. Not run.
2. **One decode per checkpoint and budget**; greedy, so no seed row. The CIs are over problems.
3. **The rule is applied to the SFT and RFT checkpoints only**; forcing the GRPO checkpoint (which
   truncates 5 / 21 %) was not run and would be the natural next cell.
4. **maj@k in EXP-017 spends k full samples**; no early stopping or adaptive allocation, so the
   token axis is an upper bound on what a smarter sampler would pay.
5. **`exp016-ctrl` and the GRPO forced run** are the two cheapest additions; both laptop-only.
6. The `force_stop` second pass re-sends the full prompt + completion; on a backend without prefix
   caching it would double the cost of the forced fraction.

## 6. Results and what they mean

`outputs/reports/phase-5.md` sections 1-2. The one-line version: a decoding rule on the SFT
checkpoint recovers all of RFT's trained-format gain and most of GRPO's cumulative gain at fewer
tokens than either, and on the RFT checkpoint it matches GRPO within the CI at fewer tokens; voting
over two to four samples of the earlier checkpoints matches or beats the GRPO policy's greedy
number. The training taught, to within the CIs, what a stop rule provides.

## 7. Interview question bank for this phase

1. *How do you know the loops hold a usable derivation?* Forced answers at budget 1,536 are right
   about half the time on both sets (chance for a numeric answer is near zero), and the forced run
   gains 7-13 points on the problems the unforced run truncated.
2. *Why not just lower max_tokens?* A lower cap scores the cut completion as it stands (no answer
   block, extraction failure); forcing writes the answer block from the derivation so far. The
   difference is the whole result.
3. *Does forcing cost anything?* Yes: on problems the unforced run would have finished after the
   budget, the forced answer is worse (the both-finished paired difference is negative at every
   budget); the net is positive because the truncated set is larger. The budget trades the two.
4. *Is GRPO then useless?* No: it reaches the same accuracy in one greedy sample at its own token
   count, which is the cheapest way to that number below about 2.5x the tokens; what it did not do
   is move the accuracy-against-tokens curve.
5. *What would you run next?* The GRPO checkpoint under the same rule; the laptop control run; then
   the prompt-difficulty question (EXP-018), which the free experiments cannot answer.

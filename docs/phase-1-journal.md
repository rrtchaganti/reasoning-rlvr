# Phase 1 journal: baselines and contamination probes (14 Oct 2026)

Chronological story of the phase; the summary is in `phase-1-engineering-notes.md`, the numbers in
`outputs/reports/phase-1.md`. Written retrospectively on 15 Oct 2026 from the actions log, the failures
log and the chat record.

## Context

Phase 0 left a calibrated harness on the 0.6B model with 100-problem subsets. Phase 1 had to establish,
for both base models and on the full eval sets, (a) greedy rows for four prompt templates, (b) a
calibration against the Qwen3 technical report (75.44 on GSM8K and 43.50 on MATH for the 1.7B base,
4-shot CoT), (c) the contamination baselines every later checkpoint is compared to, and (d) the n = 64
pass@k ceiling, which needs a rented A100. The plan put the greedy rows and probes on the local GPU and
the n = 64 sampling on the rental.

Three experiments were pre-registered before any run: EXP-001 (greedy baselines with the calibration
falsifier: more than 3 points off the report means the harness is wrong), EXP-002 (pass@k ceiling at
n = 64), EXP-003 (three contamination probes with falsifiers: completion rate above 5 % or a perturbation
gap above 8 points means GSM8K is treated as recall-contaminated).

You also asked, on this day, for per-phase engineering notes (what was built, how tested, why), which
became step 6 of the per-phase routine and a PR checklist item; the Phase 0 notes were written then.

## Building the perturbed GSM8K

GSM-Symbolic and GSM-Plus were made by hand or with an LLM. This project needed a set that regenerates
from a seed on any machine with a hash check, so it is built from GSM8K's own calculator annotations
(`<<16-3=13>>`), which form a small arithmetic program per problem. Each literal in the program is
classified as a reference to an earlier result, a perturbable number from the question, or a constant;
the perturbable numbers are resampled in [x/2, 2x], the program is re-run, and the last result is the new
gold. Every kept problem passes a self-check: the program with the original numbers must reproduce every
annotation and the official gold.

The generator was run three times while reading outputs by eye, and each read found a hole:

1. **Steps done in prose.** "Thrice as much" was computed in the text, so the chain was `40 + 120 = 160`;
   the 40 moved, the 120 did not, and the self-check passed. Rule added: every literal above 12 must be
   explained (reference, perturbable number, number written in the question as digits or a word such as
   "fifty" or "dozen", or a unit constant like 60, 100, 365); otherwise `unexplained_constant`. 90 of
   1319 problems dropped.
2. **Fractional intermediates where the story has none.** Cents were allowed everywhere, so 17
   students split into 2 groups gave 8.5 students per group. Rule: fractions only when the question
   mentions money or already contains a decimal. A test over 30 seeds checks the division stays exact.
3. Also: zero intermediates rejected, no collisions with other question numbers, percentages never
   perturbed (their decimal form hides in the chain), ambiguous literals (equal to both a question number
   and an earlier result) refused (139 problems).

Final coverage 830 / 1319 (63 %), every drop reason counted, manifest hash `b5afb83e6aacafcf`. The
manifest is committed; the data is regenerated on every machine and the hash compared.

## The partial-prompt probe

Feed the first 60 % of a problem's words, continue greedily, score the head of the continuation against
the held-out 40 % with normalised Indel similarity cut to the held-out length (a bug found by the identity
test: extra text after a verbatim continuation lowered the score). Wilson intervals, and a chance baseline
from shuffled pairs. The test data had to be un-templated: templated sentences made the chance baseline
match the hits, which was the boilerplate problem in miniature.

## The first 1.7B run, and who runs long jobs

The first 1.7B eval was launched as an assistant background job. Thirty seconds after the vLLM engine
initialised it was stopped by the assistant's memory guard (Windows low on memory with WSL holding 9 GB).
Same mechanism as the Phase 0 install. The rule became firm: every multi-minute GPU job is launched by you
from PowerShell with a `tee` to a log, and the assistant does not edit tracked files while it runs (the
harness would refuse the next run on a dirty tree). Cost: 17 minutes, mostly a weight download that
survived in the cache.

You ran the few-shot-CoT 1.7B config: 423 s (75 s engine start-up, 348 s sampling for 2679 problems).
GSM8K 74.5 [72.0, 77.0] against the report's 75.44: calibration passed. MATH-500 41.4 against 43.50 on
the full MATH test set. The remaining seven configs plus the probe ran as one batch from your terminal
(about an hour; the tee log went to the WSL home directory by a path mistake, the results were fine).

## A grader rule that was measured and rejected

On that first run, 230 of 1319 GSM8K answers came from the last-number fallback, and reading the wrong
ones found a real error ("15 liters of water in the remaining 24 liters" extracted as 24). A
"conclusion sentence" rule (first number of the last sentence starting So / Therefore / Thus) looked
obviously better on six examples. Evaluated on the stored samples before touching code: it changed 34
extractions, fixed 6, broke 10. Rejected. The fallback's share (17 %) is printed in every table instead.

## The perturbation gap fired the falsifier, and a control inverted the reading

Paired on 830 problems, 1.7B 4-shot-CoT: original 78.0 vs perturbed 67.8, gap 10.1 [7.0, 13.3], flips
131 one way and 47 the other. Above the pre-registered 8-point falsifier for every chain-of-thought prompt
on the 1.7B model. The naive conclusion would have been "GSM8K is memorised". Two explanations were
written down instead: recall of the problems' surface form, or brittleness to larger numbers (the
resampled values are on average larger). A names-only variant separates them: names carry no arithmetic.
EXP-003b was pre-registered with its own falsifier (names-only gap above 5 points means recall).

Building the control hit two incidents in one afternoon:

- Widening the name pools changed the numbers-and-names set (825 problems, new hash) because names and
  numbers draw from the same seeded stream; a different pool size consumes a different number of draws.
  Eight baseline runs had already used the 830-problem set. Fix: the default swap keeps the original
  30 + 30 pools byte for byte (hash verified), the widened pools are used only by `--names-only`.
  Lesson: a seeded generator's output is a function of every draw; any change is a new dataset version.
- A patch applied through a shell heredoc turned the regex `\b` into literal backspace bytes, so the
  name pattern matched nothing and the set changed again (829). Found with `cat -A`; fixed with an
  exact-bytes script. Rule: no backslashes through heredocs, ever.

You ran the 3-minute control: names-only gap −0.4 [−3.3, 2.5] on 516 problems (flips 28 / 30, which is
the 11 % per-problem noise floor of a single greedy eval), while numbers-and-names on the same 371
problems cost 10.8 [6.2, 15.4]. Swapping names does nothing; changing numbers costs ten points. The gap is
arithmetic brittleness, not recall. GSM8K is not treated as memorised; the perturbed set stays the primary
reasoning-versus-recall metric for every checkpoint.

## The other results

- Zero-shot beats 4-shot-CoT on both models by 3 points on GSM8K and 10-12 on MATH-500; the base model
  writes its own long chain of thought and the Minerva shots pull it toward terse solutions. Both rows
  are kept for every checkpoint: 4-shot-CoT for comparability with the report, zero-shot as best prompt.
- Answer-only 4-shot collapses to 20.5 / 9.8 on GSM8K: the floor.
- The think-tag prompt (the RL starting point) is poor: 1.7B GSM8K 61.3 with 36 % of completions running
  into the 1024-token limit in repetition loops, Countdown 1.2 %; the 0.6B fails to produce the tag
  format on 12-34 % of problems.
- MATH-500 by level (1.7B, 4-shot-CoT): 67.4 / 64.4 / 51.4 / 35.2 / 15.7.
- Partial-prompt completion at similarity 0.9: GSM8K 2.1 % (1.1 % on held-outs of 12+ words), MATH-500
  16.3 % (11.6 %). The restricted rate was added after reading the hits: boilerplate endings ("Express
  your answer as a common fraction") inflated the headline. MATH-500 hits include genuine verbatim LaTeX;
  GSM8K hits are sentence structure with different numbers. The MATH falsifier fires, so MATH-500 gains in
  later phases are read next to the level breakdown and the Countdown transfer.

## The deferral decision and the merge

EXP-002 needs about three A100-hours. You chose to merge Phase 1 with that criterion marked "not met,
because no rented GPU yet" and run it at the start of the Phase 2 rental session, after setting up a
cloud provider. The deferral was written in four places (registry status, plan status line, README row,
PR body), a self-review of the 43-file diff was done (no CRLF, no leftovers, no secrets), the rental guide
`docs/rental-setup.md` was written (driver 580+ / CUDA 13 or re-pin torch; hash check of the perturbed set
on the box). PR #2 merged with a merge commit (fd70ca0), tag `phase-1`. 116 tests. Local compute 2.34
GPU-hours total, USD 0.

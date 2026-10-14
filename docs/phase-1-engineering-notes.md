# Phase 1 engineering notes: baselines and contamination probes

Companion to `outputs/reports/phase-1.md` (results) and the 14 Oct entries in
`FAILURES_AND_LEARNINGS.md`. What was built, how it is tested, why it is designed this way, and the
questions it invites. Written at the end of the local part of Phase 1 (14 Oct 2026); the n = 64
sampling section will be appended after the rental session.

---

## 1. What was built

| piece | file | tests |
|---|---|---|
| Perturbed GSM8K generator with recomputed golds | `src/rlvr/data/perturb_gsm8k.py` (`rlvr-perturb-gsm8k`) | `tests/test_perturb_gsm8k.py` (12 tests + 1 slow) |
| Partial-prompt completion probe | `src/rlvr/data/partial_prompt.py` (`rlvr-probe-partial`) | `tests/test_partial_prompt.py` |
| Boilerplate-resistant probe re-report | `scripts/probe_report.py` | exercised on the real run |
| Paired perturbation gap | `scripts/perturbation_gap.py` | `tests/test_perturbation_gap.py` |
| Baseline tables (D2) | `scripts/collect_baselines.py` | `tests/test_collect_baselines.py` |
| pass@k / maj@k figure (EXP-002) | `scripts/plot_pass_at_k.py` | `tests/test_plot_pass_at_k.py` |
| Configs | `configs/eval/base_*.yaml` (9), `configs/sample/*.yaml` (2), `configs/probe/*.yaml` (1) | config merge tested in Phase 0 |
| Rental guide | `docs/rental-setup.md` | - |

Suite at the end of the phase: 116 tests, about 25 s on CPU.

## 2. The perturbed-GSM8K generator: design and why each rule exists

GSM-Symbolic and GSM-Plus made their perturbed sets by hand or with an LLM. This project needs a
set that regenerates from a seed on any machine with a hash check, so it is built from GSM8K's own
calculator annotations (`<<16-3=13>>`), which form a small arithmetic program per problem.

**Classification of literals.** Each number in the program is a *reference* (equal to an earlier
annotated result), a *symbol* (equal to a perturbable number written in the question), or a
*constant*. Resample the symbols, re-run the program, take the last result as the new gold.

**Why every rule is there** (each one was added after a concrete failure, all in the failures log):

1. *Self-check*: the program with the original numbers must reproduce every annotation and the
   official `#### gold`. Catches annotations that were rounded or steps the writer skipped.
2. *Ambiguity*: a literal equal to both a question number and an earlier result is refused
   (which one did the writer mean?). 139 of 1319 problems.
3. *Unexplained constants*: a literal above 12 that is not a reference, a symbol, a number written
   in the question (digits or words such as "fifty", "dozen"), or a unit constant (60, 100, 365...)
   means the solution did a step in prose; found by eyeballing ("thrice as much" computed in the
   text, the 120 stayed while the 40 moved). 90 problems.
4. *No percentages*: "20 %" hides in the program as 0.2 or 20/100, so it is never perturbed.
5. *Strictly positive intermediates; integral unless the story is about money* ("8.5 students per
   group" was accepted by the first version because money problems need cents). 
6. *No collisions*: a resampled value may not equal another number in the question.
7. *Floor and unit constants*: numbers below 3, and 7 / 12 / 24 / 60 / 100 / 365 and friends, are
   structure, not quantity, and stay.

Coverage 830 / 1319 (63 %) with every drop reason counted in the report JSON. Semantically odd but
arithmetically correct stories survive ("2 quarters" of basketball, "2/6 of the oranges"); they do
not affect the gold and are documented rather than filtered.

**Reproducibility detail that bit us.** Name swaps draw from the same random stream as number
resampling, so widening the name pools changed which numbers were drawn and silently produced a
different set (825 problems, new hash) while the eight baseline runs had used the 830-problem one.
The fix keeps the original 30 + 30 name pools for the default set, uses the widened pools only for
the names-only control, and the generator's output hash is checked against the committed manifest
on every machine (`rlvr-perturb-gsm8k --seed 0` must print `b5afb83e6aacafcf`). Lesson: a seeded
generator's output is a function of *every* draw, so any change to any draw is a new dataset
version; version it or isolate the streams.

**Tests.** Closed-form cases: a hand-built two-step problem whose perturbed answer must equal the
recomputed formula; each rejection reason on a crafted input (`step_mismatch`,
`last_step_not_gold`, `no_steps`, `ambiguous_literal`, `unexplained_constant`, `no_names`);
integrality over 30 seeds with the division required to stay exact and no collisions; comma
formatting preserved; percentages not perturbed; word numbers explaining constants; name swaps
consistent and within gender; names-only mode keeps numbers and gold; a slow test that at least
half of the real test split survives.

## 3. The partial-prompt probe

Feed the first 60 % of the words, continue greedily (192 tokens, stop at a blank line), score the
head of the continuation against the held-out 40 % with normalised Indel similarity, cut to the
held-out length so extra text after a verbatim continuation does not lower the score (a bug found
by the identity test). Rates at 0.9 and 0.75 with Wilson intervals, and a chance baseline from
shuffled pairs.

The chance baseline is *not* a boilerplate control: it pairs a continuation with another problem's
ending, which rarely shares the same formula. Boilerplate endings ("Express your answer as a common
fraction") inflate the headline rate, so `scripts/probe_report.py` adds the rate restricted to
held-out parts of at least 12 words. Both are reported; the restricted one is the honest number
(MATH-500 11.6 %, GSM8K 1.1 %). Reading the hits by eye remains part of the method.

**Tests.** Word-boundary split with at least one held-out word; similarity identity, extra-text
invariance, unrelated text below 0.6; Wilson interval sanity; a synthetic report where five verbatim
and fifteen random continuations give exactly 5 / 20 at 0.9 with chance below count. The test data
had to be un-templated: templated sentences made the chance baseline match the hits, which was the
boilerplate problem in miniature.

## 4. The paired gap and the flip table

Accuracy differences between two eval sets on the *same* problems are paired by construction
(`meta.original_id`), so the CI is the paired bootstrap from Phase 0. The 2 x 2 table (solved both /
only original / only variant / neither) is reported next to the gap because the same 10-point gap
can be 10 % flipping one way or 30 % flipping both ways. The names-only control turned this into a
measurement of the noise floor: 11 % of problems flip under a change that cannot matter.

**Tests.** A synthetic pair set with known flip counts and an unpaired id; pairing failure raises.

## 5. Experimental design decisions worth defending

- **Fixed prompts, both comparability and best-prompt rows.** The 4-shot-CoT row matches the Qwen3
  report's protocol and calibrated within a point; the zero-shot row is better by up to 12 points
  on MATH-500. Both are kept for every checkpoint rather than switching the headline mid-project.
- **Calibration as a falsifier, not a formality.** If 4-shot-CoT GSM8K had landed outside 3 points
  of 75.44, nothing downstream would have run. It landed at 74.5.
- **A control before a conclusion.** The perturbation gap fired the pre-registered falsifier; the
  honest move was not to declare GSM8K contaminated but to pre-register the names-only control and
  run it (3 GPU-minutes). It inverted the interpretation: brittleness, not recall.
- **Grader changes are measured before they are adopted.** The conclusion-sentence rule looked
  obviously better on six examples and lost 0.3 points on 1319; rejected.
- **Batch-invariant decoding stays on** for every reported row, at a measured 13 % cost.

## 6. Known limitations and debts

- Perturbed numbers are drawn from [x/2, 2x], so the perturbed set is on average harder; the gap
  therefore measures "robustness to larger numbers", which is still the reasoning-versus-recall
  signal we want, but it is not a pure size-matched GSM-Symbolic replica. A size-matched variant
  (same digit count) would be a one-line change to `_sample_value` if needed.
- Names-only control covers 516 problems (names from fixed pools); pronouns stay consistent within
  gender lists, but rare names are not swapped.
- MATH-500 prose answers (10-21 % extraction failures) are reported, not graded; a stricter
  instruction or a fallback would change the numbers and is deferred to a `fix/` decision.
- EXP-002 not run; its code is tested on fake metrics only.
- 1024 max tokens truncates 27-60 % of AIME attempts; AIME is a sanity row only, as planned.

## 7. Interview question bank for this phase

- *How did you build a perturbed GSM8K without an LLM and how do you know the golds are right?*
  Section 2: calculator annotations as a program, four classes of literal, the self-check, and the
  two holes found by reading outputs (prose steps, 8.5 students).
- *Your perturbation gap was 10 points. Is the model memorising GSM8K?* No: names-only gap is
  -0.4 [-3.3, 2.5] on the same problems where numbers cost 10.8. It is brittleness to resampled
  numbers. The partial-prompt probe agrees: 1 % verbatim continuation on GSM8K, 12 % on MATH-500.
- *Why not just use GSM-Symbolic?* Reproducible from a seed with a hash, pairs with the original by
  id for paired CIs, no external dependency; and the rules are inspectable.
- *How do you separate boilerplate from memorisation in a completion probe?* Restrict to long
  held-outs and read the hits; the shuffled-pair chance baseline alone does not do it.
- *Why does zero-shot beat few-shot on a base model?* The shots impose a terse Minerva style and a
  truncation budget; Qwen3-Base has instruction-style maths in pretraining and writes its own chain
  of thought. Both rows are kept.
- *What is the noise floor of a single greedy eval?* 11 % of problems flip under a semantically
  null change; Phase 0 showed the engine itself flips 16-69 % of texts without batch-invariant mode.
- *How did you make sure a dataset generator stays reproducible across machines?* Seeded RNG,
  committed manifest with a content hash, regeneration check on every box; and the incident where
  widening a name list changed the set.
- *What would you do differently?* Isolate the random streams per component from the start; write
  the boilerplate-resistant probe rate first; size-match the perturbed numbers.

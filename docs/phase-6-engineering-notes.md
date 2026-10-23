# Phase 6 engineering notes: the traces

Branch `phase-5b-dev`; run 22-23 Oct 2026 on the laptop's CPU, no GPU. Results:
`outputs/reports/phase-6.md`. Chronology: `docs/phase-5-journal.md` section 5. Pre-registration:
`experiments.md` EXP-023.

## 1. What was built

`scripts/analyse_traces.py`, tested in `tests/test_analyse_traces.py` on synthetic completions:

- **Loop detection.** `first_repeat(words, n=50)`: the earliest 50-word window that recurs later in
  the completion and the distance to its next occurrence. One pass with a dictionary of window
  tuples; linear in the completion's length. Exact matching on whitespace-split words.
- **Per completion:** words, the think block's share of the text, has-loop, onset (fraction of
  words), repeat unit (words), answer before the onset, answer by the end of the first cycle (before
  onset + unit + 50 words), "Wait" count, verification-phrase count.
- **Per run and eval:** quantiles over truncated and finished completions separately; for each pair
  of runs the two-sample Kolmogorov-Smirnov distance of unit length and onset among truncated loops;
  for forced-stop runs, whether the model's own text (before the appended tags) already stated an
  answer, and the forced answers' accuracy split by that.
- A Markdown table, a JSON (the per-completion lists stripped) and a figure (onset and unit
  histograms) per invocation.

## 2. Design decisions

- **Words, not tokens.** The sample files hold text, and re-tokenising every completion of every run
  for a lexical measure buys nothing; units are reported in words with a rough token conversion.
- **50-word windows.** Short enough to catch a repeated step (the units turned out to be 100-190
  words), long enough that formulaic phrases ("Let me check the calculation again") do not register
  as loops. Paraphrased cycles are missed, so the loop rates are lower bounds.
- **First-cycle measure added after the first run.** Under zero-shot the whole solution repeats, so
  the onset is at the start and the answer inside the first cycle; the pre-registered "answer before
  the loop" reads 10-13 % there while 59 % of those completions are right. Both measures are kept and
  the change is recorded as a deviation.
- **The regex for an answer** (`\boxed{`, `<answer>`, "the answer is") is deliberately broad; it can
  count a hedge as an answer. The forced split uses the same regex on the text before the tags.

## 3. Known limitations

1. Lexical loops only; a model that paraphrases its cycle is counted as not looping.
2. First samples of greedy runs only; sampled completions (T = 0.6) loop far less and are not
   analysed here.
3. The KS distances are descriptive (no multiple-comparison correction); with 59-79 GRPO loops the
   onset distances of 0.14-0.16 are within sampling noise of zero.
4. "Verification phrases" is a hand-written list.

## 4. Interview question bank for this phase

1. *What is a loop, concretely?* A 100-190-word step of the derivation re-derived from about word
   500 (GSM8K) or 800 (MATH-500), with "Wait" every few sentences; 96-99 % of truncated GSM8K
   completions of a trained checkpoint contain one.
2. *Why does more room not help?* The loop starts at a fixed word position whatever the window; at
   8,192 tokens the same loop just runs more cycles.
3. *Did GRPO change the loops?* It made them rarer, not different: KS 0.11-0.16 against SFT's on unit
   length and onset.
4. *Why are truncated zero-shot completions usually right?* The trained checkpoints rewrite the whole
   solution after boxing the answer; the box is at the end of the first cycle.
5. *What would you measure next?* Paraphrase-robust loop detection (embedding similarity between
   paragraphs), and the same analysis on sampled completions.

# Phase 6 report: the traces

Branch `phase-5b-dev` (merged into `phase-5b-test-time-compute`). Run 22-23 Oct 2026 on the laptop's
CPU, no GPU, from the first-sample files of every greedy run in the archives and on the laptop.
Pre-registration: `experiments.md` EXP-023. Script: `scripts/analyse_traces.py` (tested). Tables:
`outputs/reports/phase-6-traces.{md,json,png}` (think-tag) and `phase-6-traces-zero_shot.{md,json,png}`.

**Question.** Every result of Phases 3 to 5 is about termination: the on-policy stages gained on the
problems the previous policy did not finish, and a decoding rule that forces the stop recovers most
of that gain. Nothing yet described the text. What is a loop, where does it start, how often does an
answer precede it, and what did the training change?

**Definitions.** A completion *has a loop* when some 50-word window recurs later in it. The *onset*
is the position of the first such window as a fraction of the completion's words; the *unit* is the
distance in words to its next occurrence. *Answer before the loop*: a boxed answer, an answer tag or
"the answer is" appears before the onset. *Answer by the end of the first cycle*: the same, before
onset + unit + 50 words. The second measure was added after the first run (23 Oct 2026) because the
pre-registered one misreads the zero-shot shape (section 3); both are reported.

## 1. What a loop is (think-tag, greedy, 4,096 window)

| run | truncated GSM8K / MATH-500 | of those with a loop | loop onset, median fraction (about words) | repeat unit, median words | answer before the loop / by the first cycle |
|---|---|---|---|---|---|
| base | 35.8 / 20.0 % | 97.2 / 85.0 % | 0.04 / 0.06 (80 / 120) | 4 / 44 | 0.0 / 0.0 and 1.2 / 1.2 % |
| `sft` | 28.1 / 52.0 % | 99.2 / 87.3 % | 0.20 / 0.35 (520 / 780) | 133 / 140 | 32.7 / 42.5 and 14.5 / 25.6 % |
| `rft` | 15.2 / 39.6 % | 98.5 / 88.4 % | 0.18 / 0.33 | 119 / 123 | 29.9 / 45.2 and 21.7 / 28.0 % |
| `grpo` | 4.6 / 20.8 % | 96.7 / 76.0 % | 0.19 / 0.31 | 116 / 143 | 22.0 / 32.2 and 8.9 / 15.2 % |
| KL off | 2.1 / 13.4 % | 96.4 / 89.6 % | 0.13 / 0.33 | 116 / 188 | 14.8 / 29.6 and 6.7 / 11.7 % |
| cap 2,048 | 9.6 / 26.2 % | 98.4 / 85.5 % | 0.20 / 0.33 | 102 / 140 | 29.6 / 39.2 and 15.2 / 20.5 % |
| `sft` at 8,192 (300 problems) | 27.0 / 46.3 % | 100 / 97.8 % | 0.09 / 0.19 (500 / 890) | 131 / 144 | 29.6 / 43.2 and 16.9 / 25.0 % |
| `rft` at 8,192 | 14.0 / 36.0 % | 100 / 95.4 % | 0.08 / 0.17 | 154 / 126 | 21.4 / 42.9 and 29.1 / 32.0 % |

Four things the table says.

1. **A truncated completion of a trained checkpoint is almost always a loop**: 96-99 % on GSM8K and
   76-90 % on MATH-500 contain a recurring 50-word window. The repeat unit is a paragraph of 100-190
   words, a whole step of the derivation re-derived, not a stutter. The base model's loops are a
   different object: a unit of 4 words on GSM8K starting in the first 4 % (degenerate repetition of
   a short phrase from the start).
2. **The loop starts at a fixed position, not at the end of the budget.** For `sft` the first
   recurring window begins around word 520 on GSM8K and 780 on MATH-500 at the 4,096 window, and
   around word 500 and 890 at the 8,192 window: the onset moved from 20 % to 9 % of the words
   because the completion doubled, not because the loop started later. More room adds cycles, not
   progress, which is the mechanism behind EXP-010c (the loops do not finish at 8,192).
3. **The training changed how often a loop happens, not what a loop is.** Among truncated
   completions, the Kolmogorov-Smirnov distance between `sft` and `grpo` is 0.11 / 0.11 on the
   repeat-unit length and 0.14 / 0.16 on the onset (GSM8K / MATH-500; `sft` against `rft` 0.07-0.10;
   `rft` against `grpo` 0.08-0.11). The one structural shift: a quarter of GRPO's remaining
   MATH-500 truncations (24 %) are long derivations without a recurring window, against 12-13 % for
   `sft` and 12 % for `rft`; GRPO removed loops faster than it removed long solutions.
4. **About a third of GSM8K loops follow a stated answer.** In `sft`'s trained format, 32.7 % of
   truncated GSM8K completions state an answer before the loop begins (14.5 % on MATH-500), which
   matches the share of truncated completions the extractor scores right (31.1 % / 13.5 %,
   `phase-4-truncated-accuracy.md`). These were already counted right without any training or rule;
   the termination gain of RFT, GRPO and the stop rule comes from the other two thirds, where the
   loop began before an answer was written.

## 2. Wait and verification phrases

| run | "Wait" per truncated / finished completion, GSM8K | MATH-500 | verification phrases, truncated / finished, GSM8K | MATH-500 |
|---|---|---|---|---|
| `sft` | 24.3 / 2.1 | 20.6 / 2.9 | 9.6 / 2.6 | 7.7 / 4.2 |
| `rft` | 26.3 / 1.7 | 23.3 / 2.6 | 9.7 / 2.3 | 6.7 / 3.9 |
| `grpo` | 27.6 / 1.5 | 24.0 / 2.0 | 4.7 / 1.9 | 4.7 / 2.9 |
| KL off | 17.0 / 0.2 | 8.1 / 0.6 | 10.7 / 0.3 | 4.0 / 0.8 |
| cap 2,048 | 22.1 / 1.6 | 15.8 / 2.1 | 9.7 / 2.1 | 5.6 / 3.2 |

A loop is where the "Wait"s are: a truncated completion carries 16-28 of them in every run but one
(KL off on MATH-500: 8), a finished one 0.2-2.9.
Per finished completion they fall in the order the training logs showed (`sft` 2.1, `grpo` 1.5, KL
off 0.2 on GSM8K); per truncated completion they do not fall from `sft` to `grpo`. The fall in the
per-completion average that the GRPO training log recorded (5.9 to 3.6) is mostly the shrinking share
of loops, plus a smaller fall in the finished completions; KL off cut both.

## 3. Under zero-shot the whole solution repeats

| run | truncated GSM8K / MATH-500 | onset, median fraction | unit, median words | answer before the loop | answer by the first cycle | finished completions with a repeat |
|---|---|---|---|---|---|---|
| `sft` | 17.3 / 59.2 % | 0.10 / 0.02 | 127 / 215 | 39.6 / 9.7 % | 90.7 / 87.2 % | 25.2 / 81.4 % |
| `rft` | 19.8 / 61.8 % | 0.09 / 0.02 | 118 / 210 | 40.8 / 11.3 % | 92.7 / 89.1 % | 26.5 / 75.9 % |
| `grpo` | 16.0 / 65.0 % | 0.08 / 0.02 | 120 / 205 | 39.0 / 12.9 % | 95.2 / 94.0 % | 23.5 / 76.0 % |
| KL off | 5.6 / 23.2 % | 0.10 / 0.08 | 66 / 172 | 54.1 / 15.8 % | 90.5 / 81.6 % | 19.0 / 69.3 % |

Under the zero-shot template the trained checkpoints write the full solution, box the answer, and
start again from the first sentence: on truncated MATH-500 completions the recurring window begins
in the first 2 % of the words (median) and the unit is the whole solution (about 210 words). Many
*finished* zero-shot completions show the same behaviour cut short: of 204 finished `sft` MATH-500
completions, 166 (81 %) contain a recurring window and in 88 (43 %) it begins within the first 5 %
of the words, the solution written twice and then a stop (median unit 234 words). A *truncated*
completion is the same solution written until the window ends. The boxed answer sits at the end of the
first cycle (87-94 % of truncated MATH-500 completions), which is why truncated zero-shot
completions are right 59-63 % of the time (`phase-4-truncated-accuracy.md`) and why the zero-shot
accuracy is barely affected by its 59-65 % truncation. The pre-registered "answer before the loop"
measure reads 10-13 % here because the onset is at the start; the first-cycle measure is the right
one for this shape. The repeat is a failure to stop after the answer, the same failure as the
trained format's, on a different template.

## 4. Where the forced answers come from (EXP-016's runs)

| forced run | forced GSM8K / MATH-500 | forced answers right | own text already stated an answer | right when it had / had not (GSM8K; MATH-500) | share of the forced-right that had |
|---|---|---|---|---|---|
| `sft` + stop at 1,536 | 437 / 390 | 54.5 / 49.7 % | 52.2 / 34.1 % | 64.0 / 44.0 and 75.2 / 36.6 % | 61.3 / 51.5 % |
| `sft` + stop at 3,072 | 340 / 268 | 47.9 / 36.6 % | 46.8 / 35.8 % | 55.3 / 41.4 and 54.2 / 26.7 % | 54.0 / 53.1 % |
| `rft` + stop at 2,048 | 217 / 266 | 39.6 / 38.0 % | 58.1 / 33.8 % | 47.6 / 28.6 and 62.2 / 25.6 % | 69.8 / 55.4 % |
| `grpo` + stop at 2,048 | 66 / 181 | 16.7 / 21.5 % | 43.9 / 33.1 % | 17.2 / 16.2 and 41.7 / 11.6 % | 45.5 / 64.1 % |

Half to two thirds of the forced answers that are right had already been stated somewhere in the
model's own text before the tags were appended; the stop rule then makes the model commit to it in
the answer block, where the extractor reads it first. The rest are produced at the stop from the
partial derivation, and they are right 26-44 % of the time for `sft` and `rft`, far above chance. For
GRPO both rates are low (16-17 % on GSM8K): what GRPO leaves looping is harder material, which is
EXP-020's result seen from the text.

## 5. Against the pre-registration (EXP-023)

| clause | prediction | result | verdict |
|---|---|---|---|
| 1 | a repeated unit of 20-200 tokens starting in the second half of the window (onset after 60 % for `sft`) | unit 102-188 words (about 130-250 tokens); onset at 20-35 % of the words, a fixed position near word 500-900 | unit roughly held (at the top of the range); onset failed: loops start early and at a fixed place |
| 2 | `grpo`'s loops have `sft`'s structure: KS under 0.15 on unit length and onset | 0.11 / 0.11 on the unit, 0.14 / 0.16 on the onset; falsifier (KS over 0.3 on the unit) not near | held on three cells of four, the fourth at 0.16 |
| 3 | an answer precedes the loop in under 15 % of trained-format truncations and over 50 % under zero-shot | trained format 32.7 % GSM8K, 14.5 % MATH-500; zero-shot 10-41 % before the loop but 87-95 % by the end of the first cycle | failed on GSM8K as stated; the zero-shot clause holds only with the first-cycle measure added after the first run; the falsifier's reading ("the termination gain is mostly extraction") does not follow, because the completions with an answer before the loop are already scored right without any training |
| 4 | "Wait" and verification phrases fall from `sft` to `grpo` to KL off, concentrated in truncated completions | they fall in that order in finished completions; per truncated completion they do not fall from `sft` to `grpo`; the average falls because loops became rare | half held |
| 5 | the forced-right answers mostly come from a derivation that had reached a result before the loop | 51-70 % of forced-right answers had been stated before the tags (46 % for GRPO on GSM8K) | held weakly: a majority, not most by a wide margin |

## 6. What Phase 6 says

The termination gain has a mechanism that can be read in the text. A trained checkpoint, in either
template, sometimes fails to stop: in the trained format it re-derives a step of 100-190 words from
about word 500 (GSM8K) or 800 (MATH-500) on, with "Wait" every few sentences; under zero-shot it
rewrites the whole solution after boxing the answer. More room adds cycles. RFT and GRPO made that
failure rarer without changing its shape; a stop rule cuts it at a budget and makes the model
commit to an answer it has usually half-reached. Where the loop follows an answer (a third of GSM8K
loops), nothing needed fixing; where it begins before one, the stop rule and the training buy the
same points. This is the text-level account of every termination number in Phases 3 to 5.

**Caveats.** The loop detector is lexical (exact 50-word windows); paraphrased cycles are missed,
which makes the loop rates lower bounds. Words, not tokens, are the unit. The answer regex counts
"the answer is" phrases that can be hedges. The first-cycle measure was added after seeing the
first run, for the reason stated in section 3, and both measures are reported.

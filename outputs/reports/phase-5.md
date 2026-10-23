# Phase 5 report: the cost of termination

Branch `phase-5-test-time-compute` (from `main` 648df74, the Phase 4 merge). Option A of `docs/phase-5-plan.md`
(the user's choice, 21 Oct 2026): no rental; EXP-017 from the existing sample sets (laptop, no GPU)
and EXP-016 budget forcing on the laptop's RTX 4060. Numbers are 95 % bootstrap CIs over problems,
paired on shared problem ids unless stated. Pre-registration: `experiments.md` EXP-016, EXP-017.
The chain of questions: `docs/research-narrative.md`, "Open now (after session 2)".

## 1. EXP-017: accuracy against generated tokens (21 Oct 2026)

**Question.** Phase 4 found that every post-SFT gain in the trained format is termination. If so,
the RL policy's greedy number should be reachable by spending the same tokens differently: sampling
an earlier checkpoint several times and voting. At matched generated tokens, is the GRPO policy
the cheapest way to its own accuracy?

**Data.** No new sampling. Think-tag, T = 0.6, top-p 0.95, n = 32 on the pinned 300-problem subsets
(hashes 58c8ded1c644f332 GSM8K, 5ca9795b45227236 MATH-500) for `sft`, `rft`, `grpo` (EXP-012) and
the cap run (EXP-015); 4-shot CoT, T = 0.6, n = 64 for the base model, `sft` and `rft` (the `rft`
set on 300-problem subsets, so the shared set is 300 there too); greedy think-tag runs of `sft`,
`rft` and `grpo` from the archives. `scripts/plot_accuracy_vs_tokens.py` restricts every run to the
problems all runs share, computes maj@k and pass@k with bootstrap CIs over problems (maj@k averaged
over 20 random k-subsets per problem, extraction failures not voting), the expected generated tokens
per problem at k samples (k x the problem's mean sample length), and the paired CI of each maj@k and
pass@k against `grpo` greedy on the same problems. Figures and full tables:
`figures/phase-5-accuracy-vs-tokens-think_tag.{png,md,json}`, `-few_shot_cot.{png,md,json}`.

**Think-tag (the trained format).** Tokens per problem in brackets.

| run | GSM8K | tokens | vs `grpo` greedy | MATH-500 | tokens | vs `grpo` greedy |
|---|---|---|---|---|---|---|
| `grpo` greedy | 84.7 [80.3, 88.7] | 999 | | 64.3 [58.7, 69.7] | 1,967 | |
| `rft` greedy | 80.7 | 1,379 | | 59.0 | 2,575 | |
| `sft` greedy | 75.3 | 1,866 | | 54.0 | 2,831 | |
| `rft` maj@2 | 82.4 | 2,561 | -2.3 [-5.2, 0.7] | 64.8 | 5,025 | +0.4 [-3.4, 4.2] |
| `rft` maj@4 | 88.6 | 5,122 | **+3.9 [1.1, 6.7]** | 71.9 | 10,049 | **+7.6 [3.8, 11.4]** |
| `sft` maj@2 | 80.3 | 3,375 | -4.4 [-7.2, -1.6] | 63.2 | 5,699 | -1.2 [-4.9, 2.5] |
| `sft` maj@4 | 86.7 | 6,750 | +2.1 [-0.7, 4.9] | 70.8 | 11,399 | +6.5 [2.6, 10.5] |
| `grpo` maj@4 | 88.9 | 3,739 | +4.2 [1.3, 7.2] | 73.6 | 7,534 | +9.3 [5.3, 13.1] |
| `sft` / `rft` / `grpo` / cap maj@32 | 90.7 / 91.7 / 91.3 / 91.7 | 30-54k | | 81.7 / 80.7 / 81.0 / 82.0 | 60-91k | |

Three things the table says. (1) Voting over two `rft` samples matches `grpo` greedy within the
CI at 2.6x the tokens on both sets, and four samples beat it with CIs clear of zero at 5x. (2) From
k = 4 on, the maj@k curves of all four trained checkpoints coincide within their CIs on both sets:
under sampling at T = 0.6 (which removes most of the greedy looping; Phase 2 measured 13 % truncation
at T = 0.6 against 28 % greedy for `sft`) the checkpoints are the same model, and the RFT and GRPO
gains are gains at the greedy point of the curve, not shifts of the curve. (3) `grpo`'s own maj@4
beats its greedy by 4 and 9 points at 4x the tokens: the policy did not become a model whose single
sample is as good as its vote.

**4-shot CoT (the base model's native prompt), n = 64.** The untrained base model's maj@16 on GSM8K
(2,438 tokens per problem, since 4-shot answers are short) is +0.5 [-3.0, 4.1] against `grpo`
greedy, and maj@64 (9,754 tokens) +2.7 [-1.0, 6.7]: sixteen votes over the base model at 2.4x the
tokens match the whole pipeline's greedy GSM8K number. On MATH-500 the base never gets there (maj@64
59.0, -5.3 [-10.7, 0.0]); `sft` under 4-shot does from k = 8 (+4.5 [0.2, 9.1] at 7,727 tokens), and
`rft` under 4-shot sits on `sft`'s curve at every k (the Phase 3 finding again).

**What EXP-017 says, against its pre-registration.** The hypothesis (maj@4 of `rft` at or above
`grpo` greedy on MATH-500; maj@4 of `sft` matching it on GSM8K) held; the falsifier (maj@k of `sft`
and `rft` below `grpo` greedy at every k to 32, the first capability signal) did not fire. The GRPO
policy is the cheapest way to its own number only below about 2.5x its token budget. Read with
Phase 4's decomposition: the termination gain is real and cheap to use (one greedy sample), and it is
worth exactly what two to four samples of the previous checkpoint cost. On MATH-500 the thing votes
over the base model cannot buy is SFT's.

**Caveats.** 300-problem subsets, so paired CIs are 3-4 points wide and a 2-point claim is not
resolvable here; maj@k spends k full samples with no early stopping or adaptive allocation, so the
token axis is an upper bound on what a smarter sampler would pay; the greedy runs were made on the
H100 and the sample runs on the H100 too (same vLLM, batch-invariant), but the sampled and greedy
decodes of one checkpoint are different draws of the same policy, not the same decode; the 4-shot
figure's shared set is the `rft` run's 300-problem subset, not the think-tag pinned subset.

## 2. EXP-016: budget forcing (21-22 Oct 2026, laptop RTX 4060)

**Question.** If RFT and GRPO taught the policy to "stop and answer", a decoding rule that forces
the stop may buy the same points with no training. At a token budget B, every completion that has
not closed its think block gets "\n</think>\n<answer>" appended (the exact ending of the trained
format) and up to 128 more tokens to write the answer (`src/rlvr/eval/force_stop.py`). Run on the
`sft` and `rft` checkpoints at B in {768, 1,024, 1,536, 2,048, 3,072}, greedy, think-tag, GSM8K and
MATH-500 in full, 4,096 window, batch-invariant vLLM; paired against the archive's unforced greedy
runs of `sft`, `rft` and `grpo` (H100, same harness) on shared problem ids
(`outputs/reports/phase-5-compare-*.md`), and decomposed against the checkpoint's own unforced run
(`phase-5-gain-decomposition-*.md`). Figure and the full table:
`figures/phase-5-force-stop.{png,md,json}`.

**Pre-registered.** `sft`+force at B = 1,536 recovers at least half of RFT's gain (+3.3 / +5.8 over
`sft`) and a third of GRPO's cumulative gain (+7.5 / +12.8 over `sft`); `rft`+force recovers at
least half of GRPO's gain over `rft` (+4.1 / +7.0); the forced answers are right about as often as
the policies' own finished answers. Falsifier: under a quarter of RFT's gain at every B, or forced
answers at chance.

**The sweep.** Unforced greedy at 4,096 for reference: `sft` 74.8 / 50.6 (1,887 / 2,910 tokens),
`rft` 78.2 / 56.4 (1,426 / 2,601), `grpo` 82.3 / 63.4 (999 / 1,944).

| run | B | GSM8K | tokens | forced | forced right | vs own | vs `rft` | vs `grpo` | MATH-500 | tokens | forced | forced right | vs own | vs `rft` | vs `grpo` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `sft`+force | 768 | 77.8 [75.4, 80.0] | 752 | 81 % | 73 % | +3.0 [0.9, 5.2] | -0.4 [-2.5, 1.7] | -4.5 [-6.5, -2.4] | 50.6 [46.4, 54.8] | 776 | 99 % | 50 % | 0.0 [-4.4, 4.4] | -5.8 [-10.2, -1.4] | -12.8 [-17.0, -8.8] |
| `sft`+force | 1,024 | 79.8 [77.6, 81.9] | 930 | 58 % | 68 % | +4.9 [3.0, 7.1] | +1.6 [-0.5, 3.6] | -2.5 [-4.3, -0.5] | 56.2 [52.0, 60.8] | 1,023 | 93 % | 53 % | +5.6 [1.4, 9.6] | -0.2 [-4.4, 4.0] | -7.2 [-11.2, -3.2] |
| `sft`+force | 1,536 | 80.7 [78.5, 82.8] | 1,152 | 33 % | 55 % | **+5.8 [3.9, 8.0]** | +2.5 [0.5, 4.5] | -1.6 [-3.4, 0.3] | 59.6 [55.6, 63.8] | 1,459 | 78 % | 50 % | **+9.0 [4.8, 13.0]** | +3.2 [-1.2, 7.6] | -3.8 [-7.8, 0.0] |
| `sft`+force | 2,048 | 80.6 [78.3, 82.7] | 1,308 | 28 % | 49 % | +5.8 [3.9, 7.9] | +2.4 [0.5, 4.5] | -1.7 [-3.5, 0.2] | 61.0 [56.8, 65.2] | 1,824 | 64 % | 43 % | +10.4 [6.4, 14.4] | +4.6 [0.4, 9.0] | -2.4 [-6.2, 1.6] |
| `sft`+force | 3,072 | 80.8 [78.6, 82.9] | 1,581 | 26 % | 48 % | +6.0 [4.0, 8.1] | +2.7 [0.6, 4.6] | -1.4 [-3.3, 0.5] | 62.2 [58.0, 66.4] | 2,419 | 54 % | 37 % | +11.6 [7.2, 15.6] | +5.8 [1.4, 10.4] | -1.2 [-5.4, 2.8] |
| `rft`+force | 768 | 78.2 [75.9, 80.4] | 735 | 71 % | 71 % | 0.0 [-2.1, 2.1] | (own) | -4.1 [-6.1, -2.1] | 51.6 [47.4, 56.0] | 776 | 98 % | 51 % | -4.8 [-9.2, -0.6] | (own) | -11.8 [-15.8, -8.0] |
| `rft`+force | 1,024 | 79.8 [77.6, 82.0] | 878 | 43 % | 61 % | +1.7 [-0.4, 3.7] | (own) | -2.4 [-4.5, -0.5] | 58.0 [53.8, 62.4] | 1,013 | 87 % | 52 % | +1.6 [-2.4, 5.8] | (own) | -5.4 [-9.2, -1.8] |
| `rft`+force | 1,536 | 80.8 [78.6, 82.9] | 1,033 | 21 % | 44 % | +2.7 [0.6, 4.6] | (own) | -1.4 [-3.6, 0.5] | 61.8 [57.6, 66.0] | 1,408 | 67 % | 45 % | +5.4 [1.6, 9.4] | (own) | -1.6 [-5.2, 2.0] |
| `rft`+force | 2,048 | 80.9 [78.8, 83.0] | 1,125 | 17 % | 40 % | +2.7 [0.8, 4.7] | (own) | -1.4 [-3.4, 0.6] | 64.0 [59.8, 68.2] | 1,710 | 53 % | 38 % | **+7.6 [3.6, 11.6]** | (own) | **+0.6 [-3.2, 4.4]** |
| `rft`+force | 3,072 | 80.9 [78.7, 83.0] | 1,282 | 15 % | 37 % | +2.7 [0.8, 4.7] | (own) | -1.4 [-3.4, 0.6] | 64.6 [60.4, 68.8] | 2,188 | 43 % | 30 % | +8.2 [4.0, 12.4] | (own) | +1.2 [-2.6, 4.8] |

"forced" is the share of completions that hit B inside the think block; "forced right" the
accuracy of those forced answers; 99-100 % of forced completions finished within the 128 extra
tokens at every budget, and truncation at the 4,096 window is 0-1.6 % in every run.

**What the sweep says, clause by clause.**
- *Recovery.* `sft`+force at 1,536 gains +5.8 / +9.0 over `sft`: 176 % / 155 % of RFT's gain and
  77 % / 70 % of GRPO's cumulative gain; at 3,072, +6.0 / +11.6, or 80 % / 91 % of GRPO's. The first
  two clauses held with room. `rft`+force at 2,048 gains +2.7 / +7.6 over `rft`: 66 % / 109 % of
  GRPO's gain over `rft`; the third clause held.
- *Against GRPO at matched tokens.* `rft`+force at 2,048 is +0.6 [-3.2, 4.4] on MATH-500 at 1,710
  tokens (GRPO 1,944) and -1.4 [-3.4, 0.6] on GSM8K at 1,125 (GRPO 999); `sft`+force at 3,072 is
  -1.2 [-5.4, 2.8] and -1.4 [-3.3, 0.5]. Every CI covers zero from 1,536 up on GSM8K and from 2,048
  up on MATH-500. GRPO keeps a point and a half on GSM8K at 10-30 % fewer tokens, inside the noise.
- *The forced answers* are right about half the time at 1,536 (55 % / 50 % for `sft`), against 31 %
  / 13.5 % for the same checkpoint's truncated completions scored through the extractor's fallback
  at 4,096 (`phase-4-truncated-accuracy.md`) and about 90 % for its finished answers: the fourth
  clause failed. The loops hold a usable derivation, not a finished one, and the
  rate falls with the budget (what is still looping at 3,072 is right 37-48 % of the time on GSM8K
  and 30-37 % on MATH-500).
- *The cost of the rule.* Decomposed against the unforced run (`phase-5-gain-decomposition-*`),
  the gain sits on the problems the unforced run truncated (+7.3 [5.6, 9.1] GSM8K, +13.2 [10.0,
  16.6] MATH-500 for `sft` at 1,536) and the forced run *loses* on problems the unforced run would
  have finished after the budget (-1.4 / -4.2; both-finished paired difference -2.1 [-3.7, -0.4]
  and -8.4 [-13.0, -4.2]). The budget trades the two: on GSM8K the net saturates by 1,536 (80.7,
  80.6, 80.8), on MATH-500 it is still rising at 3,072 (62.2) because `sft`'s finished derivations
  there are long (mean 2,900 tokens). At 768 the rule gains nothing on MATH-500 (0.0 [-4.4, 4.4]).
- *The shape of the curve* (`figures/phase-5-force-stop.png`): the two forced checkpoints trace one
  curve on each set; `rft`'s points sit 100-300 tokens to the left of `sft`'s at the same budget
  (it loops less, so fewer completions are cut), and `grpo`'s greedy star lies on that curve on
  MATH-500 and a point and a half above it on GSM8K.

**Conclusion, against the pre-registration.** The hypothesis held on three of four clauses and the
falsifier did not fire. A stop rule on the SFT checkpoint recovers all of RFT's trained-format gain
(on the same machine +4.6 / +7.2 against RFT's +2.9 / +4.6; section 2b) and most of GRPO's at fewer
tokens than either; on the RFT checkpoint it matches GRPO within the CI at fewer tokens on MATH-500
and within a point and a half on GSM8K (cross-machine cells; section 2b). Read with the Phase 4
decomposition and EXP-017: the training taught, to within the CIs, what a decoding rule provides,
and what it did not teach (the forced answers are right half the time, the policies' own finished
answers nine times in ten) is the part a rule cannot supply, which is also the part no run of
Phase 4 moved.

**Caveats.** The unforced references are the H100 runs and the forced runs are the laptop's (same
checkpoint, harness and batch-invariant vLLM; a cross-machine greedy difference of a few tenths of
a point is possible and `exp016-ctrl` would measure it, not run); one decode per cell, no seed row;
the rule was not applied to the GRPO checkpoint (5 / 21 % truncation), the natural next cell.

### 2b. Correction (23 Oct 2026): the same-machine re-pairing

The laptop control run (EXP-020b, section 4) measured the caveat above and it is not a few tenths:
the same checkpoint decoded greedily on the laptop and on the H100 differs on 15-23 % of problems
(net +0.6 to +1.8 points). Every "vs own" and "vs `rft`" cell of the table above therefore pairs two
machines. Re-paired against the laptop's own unforced runs (`phase-5-compare-{sft,rft}-force*-vs-{sft,rft}-laptop.md`,
`phase-5-compare-sft-force*-vs-rft-laptop.md`):

| run | B | GSM8K vs own | MATH-500 vs own | GSM8K vs `rft` | MATH-500 vs `rft` |
|---|---|---|---|---|---|
| `sft`+force | 768 | +1.7 [0.2, 3.3] | -1.8 [-5.6, 2.0] | -1.1 [-3.3, 1.0] | -6.4 [-10.8, -2.4] |
| `sft`+force | 1,024 | +3.7 [2.4, 5.2] | +3.8 [0.4, 7.4] | +0.8 [-1.1, 2.9] | -0.8 [-4.8, 3.4] |
| `sft`+force | 1,536 | **+4.6 [3.4, 6.0]** | **+7.2 [4.0, 10.4]** | +1.7 [-0.2, 3.7] | +2.6 [-1.6, 6.6] |
| `sft`+force | 2,048 | +4.5 [3.4, 5.8] | +8.6 [6.0, 11.4] | +1.7 [-0.3, 3.6] | +4.0 [0.0, 7.8] |
| `sft`+force | 3,072 | +4.8 [3.6, 6.1] | +9.8 [7.2, 12.4] | +1.9 [-0.1, 3.9] | +5.2 [0.8, 9.4] |
| `rft`+force | 768 | -0.8 [-1.8, 0.4] | -5.4 [-9.0, -2.0] | (own) | (own) |
| `rft`+force | 1,024 | +0.9 [-0.1, 1.8] | +1.0 [-2.0, 3.8] | (own) | (own) |
| `rft`+force | 1,536 | +1.9 [1.1, 2.7] | +4.8 [2.4, 7.4] | (own) | (own) |
| `rft`+force | 2,048 | +2.0 [1.2, 2.7] | **+7.0 [4.6, 9.6]** | (own) | (own) |
| `rft`+force | 3,072 | +2.0 [1.2, 2.7] | +7.6 [5.4, 10.0] | (own) | (own) |

RFT's own gain over SFT on the same machine is +2.9 [0.8, 5.0] / +4.6 [0.6, 9.0]
(`phase-5-compare-rft-vs-sft-laptop.md`). What changes:

- *Recovery.* `sft`+force at 1,536 gains +4.6 / +7.2 over `sft` (first quoted +5.8 / +9.0), 159 % /
  157 % of RFT's same-machine gain, and against the laptop's `rft` run it is +1.7 [-0.2, 3.7] /
  +2.6 [-1.6, 6.6]: at least RFT's level, above it by an amount inside the CI. Clause 1 holds.
  Against GRPO only the cross-machine cells exist; the GRPO laptop control is queued (section 4).
- *The cost of the rule was mostly the machine term.* On one machine a forced run is identical to
  the unforced run on every completion shorter than the budget (the same greedy decode), so the rule
  can only change completions that would have run past B. Same-machine decompositions
  (`phase-5-gain-decomposition-{sft,rft}-force*-vs-{sft,rft}-laptop.md`): `sft`+force at 1,536 gains
  +4.7 / +8.8 on the problems `sft` truncated and loses -0.1 / -1.6 on the problems it finished,
  both-finished -0.1 [-0.3, 0.0] / -2.7 [-5.4, -0.4]; at 3,072 the loss is 0.0 / +0.2; `rft`+force at
  2,048 loses 0.0 / -0.6. The first decomposition's -1.4 / -4.2 on finished problems and -2.1 / -8.4
  both-finished were the 15-23 % cross-machine flips, not the rule. The real cost is the MATH-500
  completions that would have finished between B and the window, about 1.6 points at 1,536.
- Unchanged: the forced answers' accuracy (computed within each run), the shape of the curve, and
  the conclusion. The figure's stars for `sft` and `rft` are the H100 runs; the laptop's own
  unforced points are 76.0 / 52.4 (`sft`) and 78.9 / 57.0 (`rft`).

## 3. What option A says together

1. The GRPO policy's greedy number is reachable without GRPO: by voting over two `rft` samples at
   2.6x the tokens (EXP-017), or by forcing the stop on `rft` at 2,048 tokens at 0.9x the tokens on
   MATH-500 (EXP-016), or on `sft` at 3,072 tokens.
2. Under sampling at T = 0.6 the four trained checkpoints are one model (EXP-017); under forced
   stopping the two pre-GRPO checkpoints trace one curve (EXP-016). The on-policy stages moved the
   greedy point along a curve that the checkpoints share.
3. What the pipeline added that neither rule nor votes recover is SFT's MATH-500 capability over
   the base model (EXP-017's 4-shot panel), and the half of forced answers that are wrong.
4. Cost per point of trained-format termination gain, for the record: a stop rule INR 0 and no
   training; RFT 0.2 H100-hours; GRPO 3.1 H100-hours.

What option A cannot answer is whether harder prompts would have made GRPO teach something a rule
cannot (EXP-018), which is the one training question left in `docs/phase-5-plan.md`.

## 4. EXP-020: the GRPO checkpoint under the stop rule, and the laptop control run (22-23 Oct 2026)

Phase 5B, `docs/plan-after-phase-4.md` section 1; laptop RTX 4060; `outputs/phase5b_exp020.log`.

**(b) The laptop control run.** The unforced `sft` and `rft` think-tag greedy runs, made again on
the laptop at the 4,096 window, against the H100 runs of the same checkpoints
(`phase-5-compare-{sft,rft}-laptop-vs-h100.md`):

| checkpoint | GSM8K laptop / H100 | paired | problems flipping | MATH-500 laptop / H100 | paired | problems flipping |
|---|---|---|---|---|---|---|
| `sft` | 76.0 / 74.8 | +1.2 [-0.9, 3.5] | 232 (17.6 %) | 52.4 / 50.6 | +1.8 [-2.6, 5.8] | 117 (23.4 %) |
| `rft` | 78.9 / 78.2 | +0.8 [-1.3, 2.8] | 196 (14.9 %) | 57.0 / 56.4 | +0.6 [-3.2, 4.8] | 103 (20.6 %) |

The pre-registered hypothesis (within 0.5 points, under 2 % flipping) failed and the falsifier
(over 1 point or over 5 % flipping) fired. Batch-invariant greedy decoding is bit-identical across
runs on one machine (EXP-000), not across GPUs: a near-tie in the argmax early in a long completion
sends the two decodes apart. The net difference is inside its CI, the flip rate is not small, and the
laptop loops a little less (truncation 25.7 / 48.2 % for `sft` against 28.1 / 52.0 %). EXP-016 is
re-paired on the same machine in section 2b; `FAILURES_AND_LEARNINGS.md` records the incident.

**(a) GRPO under the stop rule** (`phase-5-compare-grpo-force{1536,2048}-vs-grpo.md`):

| B | GSM8K | tokens | forced | vs `grpo` greedy | MATH-500 | tokens | forced | vs `grpo` greedy |
|---|---|---|---|---|---|---|---|---|
| 1,536 | 82.5 [80.5, 84.5] | 867 | 7.4 % | +0.2 [-1.4, 1.9] | 62.6 [58.4, 66.6] | 1,285 | 47 % | -0.8 [-4.4, 2.8] |
| 2,048 | 82.5 [80.5, 84.5] | 897 | 5.0 % | +0.2 [-1.4, 1.9] | 63.6 [59.4, 67.6] | 1,493 | 36 % | +0.2 [-3.4, 3.6] |

GRPO's unforced greedy run spends 999 / 1,944 tokens. The pre-registered prediction (+1 to +3 on
MATH-500) did not appear and the falsifier fired as stated: forcing the stop on GRPO buys the same
accuracy at 10-25 % fewer tokens and nothing more. The forced answers are right 17 / 22 % of the time
(GSM8K / MATH-500), far below `sft`'s 55 / 50 % (Phase 6, section 4): what GRPO leaves looping is
harder material. **Provisional:** the `grpo` greedy reference is the H100 run, so these cells carry
the machine term of (b), and the first decomposition (+4.2 on GRPO-truncated MATH-500 problems,
-5.0 on finished ones) cannot be read as the rule's cost (section 2b shows that on one machine the
rule leaves every completion shorter than the budget unchanged). The unforced GRPO run on the
laptop is queued after EXP-022 and will replace these cells.

## 5. EXP-021: "Wait" before the stop (23 Oct 2026)

At budget 1,536, every completion still inside its think block got "\nWait" appended and up to 512
more tokens; those that closed the think block and finished stayed finished, the rest got the forced
stop (`--force-continue`, tested). Same-machine pairings throughout
(`phase-5-compare-{sft,rft}-wait-vs-force{1536,2048}.md`).

| run | closed the think block on their own | GSM8K | tokens | vs stop at 1,536 | vs stop at 2,048 | MATH-500 | tokens | vs stop at 1,536 | vs stop at 2,048 |
|---|---|---|---|---|---|---|---|---|---|
| `sft` + Wait | 60 of 437 / 59 of 390 (14 / 15 %) | 80.8 [78.7, 82.9] | 1,311 | +0.2 [-0.2, 0.6] | +0.2 [-0.2, 0.7] | 61.2 [57.2, 65.4] | 1,833 | +1.6 [-0.4, 3.8] | +0.2 [-1.4, 1.8] |
| `rft` + Wait | 47 of 271 / 63 of 336 (17 / 19 %) | 80.9 [78.7, 83.0] | 1,128 | +0.1 [-0.5, 0.6] | 0.0 [-0.5, 0.5] | 63.4 [59.2, 67.6] | 1,724 | +1.6 [-0.2, 3.4] | -0.6 [-2.2, 1.4] |

The stop at 2,048 spends almost exactly the tokens the Wait condition spends (1,308 against 1,311 for
`sft` on GSM8K; 1,824 against 1,833 on MATH-500), so it is the matched comparison: the nudge lands
within a fraction of a point of it on all four cells. Against the immediate stop at 1,536 it gains on
MATH-500 what the extra tokens buy (+1.6, CIs touching 0). Of the completions nudged, 14-19 % closed
their think block on their own; the rest kept looping until the forced stop. Against the
pre-registration: within the noise of the immediate stop (held), under 20 % closing unaided (held),
below the stop at B + 512 (failed: level, not below); the falsifier (beats the same-budget stop by
over 2 with a CI clear of 0) did not fire. A looping model given "Wait" mostly loops on;
sequential extension buys nothing a larger budget does not. Phase 6 shows why: the loop is a
re-derivation of the same step from a fixed point, and "Wait" is already what the model writes
inside it, 16-28 times per truncated completion.

## 6. EXP-022: an outcome reward model against voting (running)

The first run (23 Oct 2026, 03:01-09:31 UTC) trained a LoRA head on Qwen3-0.6B-Base on 20,000
labelled `sft` samples (58 % correct) and reached a held-out accuracy of 87.9 % and AUROC 0.938 on
2,207 examples from problems it never saw, then hung while saving (`FAILURES_AND_LEARNINGS.md`, 23
Oct). The rerun with the fix started at 11:30 UTC; its held-out numbers, best-of-k and the
reward-weighted vote against majority vote follow here when the scoring lands.

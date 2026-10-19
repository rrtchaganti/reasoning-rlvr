# Phase 2 report: SFT on distilled traces

Branch `phase-2-sft`, started 15 Oct 2026. Section 1 comes from the local CPU work (15 Oct); sections 2-7 from
the rented sessions: an L4 on 17 Oct and the TIR H100 on 18-19 Oct (EXP-002 deferred from Phase 1,
EXP-005 to EXP-008; EXP-007 cut). Written up 19 Oct 2026.
Numbers are greedy, batch-invariant vLLM, 95 % bootstrap CI over problems unless stated.

## 1. Training data (EXP-004 and the data report)

Built by `rlvr-build-sft` from `open-r1/OpenR1-Math-220k` (`default`, revision e4e141ec) at commit
04779ec; full funnel and tables in `phase-2-data.md`.

| stage | count |
|---|---|
| rows in the source | 93,733 |
| no math-verify-correct trace | 28,627 |
| trace longer than 4096 tokens (prompt + completion) | 34,650 |
| multi-part (several boxes) or malformed trace | 6,678 |
| duplicate problem | 721 |
| overlap with an eval set or the dev split | 57 |
| **kept** | **23,000** (61.8 M tokens; mean 2,689, p90 3,790 tokens; hash `45b3fa4862b11e2a`) |

Sources kept: olympiads 76 %, cn_contest 15 %, aops_forum 6 %, amc_aime 2 %. Problem types: algebra
47 %, number theory 15 %, geometry 14 %, combinatorics 11 %, calculus 6 %. The extracted `<answer>`
value agrees with the source gold under math-verify on 99.8 % of a 2,000-example sample; the
disagreements are notation (`\dfrac` vs `\frac`, a missing `+C`).

**Training-data overlap (D3 part a).** Contaminated = a 13-gram hit covering at least half of the
eval problem, or a fuzzy match at similarity 0.9 or above; every hit item was dropped from training
whether or not it counts.

| eval set | problems | any 13-gram | covering ≥ 50 % | fuzzy | **contaminated** | train items dropped |
|---|---|---|---|---|---|---|
| GSM8K-test | 1319 | 0 | 0 | 0 | **0** | 0 |
| GSM8K perturbed / names-only / dev | 830 / 516 / 250 | 0 | 0 | 0 | **0** | 0 |
| MATH-500 | 500 | 6 | 2 | 2 | **3 (0.6 %)** | 31 |
| MATH-dev | 250 | 9 | 2 | 1 | **2 (0.8 %)** | 37 |
| AIME24 | 30 | 8 | 0 | 0 | **0** | 36 |

The eight AIME24 any-gram hits are one template sentence ("where m and n are relatively prime positive
integers, find m + n", in several variants), not copies; the first two builds would have reported them
as 27 % contamination (journal, section 7). The three MATH-500 problems are intermediate_algebra 90
(verbatim), 986 (paraphrase at 0.91) and 1365. GSM8K is absent from the SFT source, so every GSM8K row
after SFT measures transfer from olympiad-style data, not recall.

## 2. Compute

Local CPU: four data builds (25 + 7 + 5 + 5 min). Local GPU (RTX 4060, USD 0): trainer tests on a tiny
model, two 0.6B LoRA smokes with their 100-problem evals (0.4 h), and the base think-tag reference at
the 4096 window on all five eval sets (3.9 h; 5.3 M generated tokens).

Rented, E2E Networks L4 24 GB (INR 49/h, 17 Oct 2026): environment build, data build and smoke (1.5 h),
the four 4-shot-CoT sets of EXP-002 (5.2 h, 34.3 M generated tokens), and a stopped Countdown run
(1.6 h, discarded); 10.55 node-hours, INR 524.30 on the invoice.

Rented, E2E TIR H100 SXM 80 GB notebook (INR 379.05/h, 18-19 Oct 2026, one session for Phases 2 and
3): the Phase 2 part ran from 09:50 UTC on 18 Oct to 03:35 UTC on 19 Oct, 17.75 node-hours, about
INR 6,730, against the handoff's 10-12 hours. Breakdown: setup, data and smoke 0.6 h; EXP-002's
Countdown half 1.57 h (62.3 M generated tokens); LR sweep 0.7 h (18.5 M trained tokens); main SFT
run with its three evals, gap and probe 3.4 h (123.6 M trained, 13 M generated); SFT n = 64
few-shot sampling 3.77 h (165.5 M generated); SFT Countdown n = 64 3.07 h (119.4 M); LoRA 0.2 h
(stopped at step 20); data scaling 2.8 h (92.6 M trained, 16 M generated); report and archive
0.1 h; idle and debugging 1.75 h (a tmux name-prefix slip cost 27 idle minutes, a stale W&B key 40).
The two base greedy rows at the SFT cap (section 5) were 5 minutes of the Phase 3 part. Training ran
with exact sdpa attention and no packing (no flash-attention wheel exists for the locked torch
2.13; one was built on the node for Phase 3), at a 16 x 4 micro-batch that filled the card: 15.2 s
per optimiser step, 72.9 GB reserved. Rows in `COMPUTE.md`; the session's record in
`docs/phase-2-journal.md`, sections 19-27.

## 3. Deferred from Phase 1: EXP-002, the base model's pass@k ceiling *(complete)*

Four sets ran on the L4 on 17 Oct 2026 (commit ade4f6a) and the Countdown half on the H100 on 18 Oct
(commit bb6f2a6, think-tag, 500 generated problems x 64, T = 0.6, top-p 0.95, 4096 new tokens,
1 h 34 min, 62.3 M generated tokens, about INR 595). The full table, the figure
(`figures/phase-1-pass-at-k.png`, now with the Countdown panel) and the reading are in the Phase 1
report's addendum (section 7) and in `experiments.md`. The numbers this phase uses: pass@64 of the
base is 98.8 [98.2, 99.4] on GSM8K, 90.2 [88.2, 92.2] on perturbed GSM8K, 87.8 [84.6, 90.4] on
MATH-500, 20.0 [6.7, 36.7] on AIME24 and 63.8 [59.6, 68.0] on Countdown; maj@64 85.7, 77.0, 57.6,
0.0 and 11.0; mean sample accuracy at T = 0.6 is 3-5 points below greedy on the math sets. Countdown
is the set where sampling matters most: greedy 1.2 %, mean sample 2.8 %, pass@64 63.8 %, with 43 % of
samples hitting the cap (mean 1,946 tokens, median 466). The base can find the right expression in 64
tries for two problems in three, and almost never on the first.

## 4. EXP-005: learning-rate sweep on the dev split

Three runs on the seeded 10 % subset (2,300 examples, 1 epoch, 36 optimiser steps each, 10.5 min per
run at 16.5 s per step, 33 GB), think-tag greedy on the dev split (250 GSM8K-train + 250 MATH-train
problems, 4096 new tokens). The rule was written before the runs: highest pooled dev accuracy, ties
within 2 points to 1e-5.

| LR | pooled dev (n = 500) | GSM8K-dev | MATH-dev | tag-format failure GSM8K / MATH | final train loss |
|---|---|---|---|---|---|
| 5e-6 | 44.0 | 54.0 [47.6, 60.4] | 34.0 [28.0, 40.0] | 100 % / 100 % | 0.470 |
| **1e-5** | **53.8** | 65.2 | 42.4 | 79 % / 78 % | 0.452 |
| 2e-5 | 50.6 | 64.8 | 36.4 | 58 % / 68 % | 0.442 |

Winner 1e-5, as hypothesised. 5e-6 is below the other two beyond their CIs (about +-4.4 at n = 500);
1e-5 and 2e-5 are within one CI of each other, so the sweep is half a null result: the LR matters at
the low end and not at the top. The surprise is the tag-format column: after 36 steps every checkpoint
writes long think blocks and most never close them inside 4096 tokens (truncation 87 % on GSM8K-dev at
5e-6, median length at the cap). The failure falls with the LR (100 -> 79 -> 58 %), i.e. the format is
still being learned at this step count, and the main run (720 steps) is where it had to resolve. It
did only partly (section 5). The hypothesised failure mode of 2e-5, "erases base knowledge on MATH-dev
levels 1-2", is not visible at this resolution; the deficit of 2e-5 is on MATH-dev as a whole (36.4
vs 42.4, inside the CI).

## 5. EXP-006: base vs SFT, paired

**Run.** `configs/train/sft_1p7b.yaml` at commit bb6f2a6: full fine-tune of Qwen3-1.7B-Base, lr 1e-5,
2 epochs on the 23,000 traces (61.8 M tokens per epoch), effective batch 64 (16 x 4), AdamW
(0.9, 0.95), weight decay 0.1, cosine to 10 %, 3 % warmup, fp32 master weights under bf16 autocast,
gradient checkpointing, sdpa attention, no packing. 720 steps in 3 h 02 min on the H100, final train
loss 0.413, peak 51.6 GB allocated. The `<think>` and `</think>` rows were untrained in the base
(93 duplicate filler rows), were initialised from their plain-text pieces and moved by 0.0078 and
0.0531 (L2 norm of the delta); no duplicate rows remain. Checkpoint:
`Raviteja546/reasoning-rlvr-sft` on the Hub (card: `cards/sft.md`), `outputs/train/sft_1p7b/final`
on the node's disk.

**Think-tag, paired at the shared 4096 window** (`phase-2-compare-think_tag.md`):

| eval | pairs | base | SFT | SFT - base [paired CI] | only base / only SFT | truncated base / SFT |
|---|---|---|---|---|---|---|
| GSM8K | 1319 | 61.8 [59.3, 64.4] | 74.8 [72.5, 77.1] | **+13.0 [10.0, 15.8]** | 121 / 293 | 35.8 / 28.1 % |
| GSM8K perturbed | 830 | 51.4 [48.1, 54.7] | 66.0 [62.8, 69.3] | **+14.6 [11.0, 18.1]** | 75 / 196 | 35.8 / 28.8 % |
| MATH-500 | 500 | 44.8 [40.8, 49.2] | 50.6 [46.4, 55.0] | **+5.8 [0.6, 11.0]** | 76 / 105 | 20.0 / 52.0 % |
| Countdown | 500 | 1.2 [0.4, 2.2] | 3.8 [2.2, 5.6] | **+2.6 [0.8, 4.6]** | 5 / 18 | 65.6 / 94.8 % |
| AIME24 | 30 | 6.7 [0.0, 16.7] | 3.3 [0.0, 10.0] | -3.3 [-13.3, 6.7] | 2 / 1 | 30.0 / 96.7 % |

MATH-500 by level, SFT minus base (think-tag): L1 +23.3 [4.7, 41.9] (n 43), L2 +11.1 [0.0, 22.2]
(90), L3 +2.9 [-10.5, 15.2] (105), L4 +9.4 [-1.6, 20.3] (128), L5 -4.5 [-11.9, 3.0] (134). The gain
sits on levels 1-2, not on 3-5 as hypothesised, and level 5 moves the wrong way inside its CI. The
caveat of amendment (b) stands under this table: the base side reads an untrained `<think>` token, so
the paired think-tag gain flatters SFT; the fair bars are the base's own 4-shot-CoT rows.

**The other two templates, paired** (`phase-2-compare-few_shot_cot.md`, `phase-2-compare-zero_shot.md`;
the base rows were rerun on 19 Oct at the SFT cap, 4,095 new tokens, because the Phase 1 rows used 1,024;
the base truncates 0.5-9 % at either cap, so the one-token difference from 4,096 is immaterial):

| template | eval | base | SFT | SFT - base [paired CI] |
|---|---|---|---|---|
| 4-shot-CoT | GSM8K | 73.5 [71.2, 75.9] | 79.4 [77.3, 81.5] | **+5.8 [3.6, 8.2]** |
| 4-shot-CoT | GSM8K perturbed | 67.7 [64.6, 70.8] | 71.4 [68.2, 74.6] | **+3.7 [1.1, 6.6]** |
| 4-shot-CoT | MATH-500 | 40.2 [36.0, 44.6] | 56.8 [52.4, 61.4] | **+16.6 [12.4, 21.2]** |
| zero-shot | GSM8K | 77.9 [75.7, 80.1] | 82.2 [80.1, 84.2] | **+4.3 [1.8, 6.6]** |
| zero-shot | GSM8K perturbed | 69.9 [66.7, 73.0] | 72.7 [69.6, 75.5] | **+2.8 [0.1, 5.5]** |
| zero-shot | MATH-500 | 53.8 [49.4, 58.4] | 63.2 [59.0, 67.2] | **+9.4 [5.0, 14.0]** |

Under 4-shot-CoT the MATH-500 gain is on every level (L1 +16.3, L2 +21.1, L3 +15.2, L4 +21.1, L5
+10.4 [2.2, 17.9]); under zero-shot it is on levels 4-5 (+14.1, +15.7) with levels 1-3 inside their CIs.
Against the pre-registered bars: GSM8K think-tag 74.8 against "at least 75" (0.2 under, inside the
CI; the base's own 4-shot-CoT row is 73.5 at this cap and 74.5 at 1,024); MATH-500 50.6 against "at
least 50"; Countdown 3.8 against "under 10". Neither falsifier (GSM8K under 70, MATH-500 under 46;
levels 1-2 falling) fires.

**The trained format is the worst of the three.** On the SFT checkpoint think-tag scores 74.8 on
GSM8K and 50.6 on MATH-500 where zero-shot scores 82.2 and 63.2, although think-tag is the training
format. Length is the symptom: think-tag completions average 1,887 tokens on GSM8K (p90 at the cap)
and 2,910 on MATH-500, 28 % and 52 % truncate; Countdown truncates 95 % and AIME24 97 %. The
hypothesis said tag-format failures would fall below 1 %; the truncated outputs are exactly the ones
that never close the think block, so the failure rate is the truncation rate. Two runs made on the
same node in the Phase 3 part say what this is (`docs/phase-3-journal.md`; `phase-3-compare-think_tag_8k.md`,
`phase-3-compare-think_tag_n32_t06.md`): at an 8,192-token cap the SFT checkpoint scores 75.0 on a
300-problem GSM8K subset with truncation still at 27 % (mean 3,011 tokens; Countdown 95 % at a mean
of 7,880), so the truncated generations do not finish at twice the budget, they loop; and at T = 0.6
the same checkpoint's sampled pass@1 is 80.3 with truncation 13 %, so most of the loops are a
greedy-decoding degeneration, not a lost ability. The reading for this phase: SFT taught the style
and a quarter of the time the style does not stop; the zero-shot row is the better measure of what
the checkpoint knows, the think-tag row of what the format costs.

**Perturbation gap, think-tag, 830 pairs:** original 77.3 [74.3, 80.2] vs perturbed 66.0 [62.8,
69.5], gap 11.3 [7.7, 14.7] points, flip rate 26.7 % (both 484, only original 158, only perturbed 64,
neither 124). The base's gap at the same window is 10.4; SFT did not close it (hypothesis: within 3
points of the base; met), and under 4-shot-CoT sampling the gap is 10.0 [8.3, 11.8] at pass@1 and
9.3 [7.2, 11.4] at pass@64 (`phase-2-perturbation-gap-sft-n64.md`; base 10.2 and 8.6). The data has
no name perturbations and the number brittleness is untouched.

**Partial-prompt completion** (first 60 % of the problem, 192 greedy tokens, similarity 0.9): GSM8K
2.2 % [1.5, 3.1] (base 2.1 %), MATH-500 14.1 % [11.3, 17.4] (base 16.3 %), chance 0 on both. The
rates did not rise (hypothesis met): SFT on a MATH-derived source added no measurable recall of
MATH-500 problems, consistent with the 3-problem contamination count of section 1.

**pass@k and maj@k** (4-shot-CoT, n = 64, T = 0.6, top-p 0.95, 4096 new tokens, 3 h 45 min;
`figures/phase-2-pass-at-k-sft.png`, paired in `phase-2-compare-pass_at_k.md`):

| eval | k | base | SFT | SFT - base [paired CI] |
|---|---|---|---|---|
| GSM8K | 1 | 71.6 | 71.0 | -0.5 [-1.4, 0.4] |
| GSM8K | 64 | 98.8 | 98.2 | -0.6 [-1.3, 0.1] |
| GSM8K perturbed | 1 | 64.5 | 63.0 | -1.4 [-2.6, -0.3] |
| GSM8K perturbed | 64 | 90.2 | 88.8 | -1.4 [-2.8, -0.1] |
| MATH-500 | 1 | 36.3 | 51.6 | **+15.3 [13.7, 16.9]** |
| MATH-500 | 64 | 87.8 | 93.0 | **+5.2 [2.4, 8.0]** |
| AIME24 | 1 | 0.6 | 2.7 | +2.1 [0.6, 4.0] |
| AIME24 | 64 | 20.0 | 26.7 | +6.7 [-6.7, 20.0] |

maj@64: GSM8K 87.6 (base 85.7), perturbed 78.3 (77.0), MATH-500 72.6 (57.6), AIME24 6.7 (0.0).
Sampling at T = 0.6 doubles the SFT checkpoint's completion length (930-1,960 tokens against 429-886
greedy) and pushes truncation to 15-16 % on the math sets where the base had none. Under this
template SFT changes nothing on GSM8K at any k (the ceiling is unchanged within CI, slightly lower on
the perturbed set) and lifts MATH-500 by 15 points at pass@1 and 5 at pass@64, i.e. mostly
sharpening with a small amount of new reach (39 problems only SFT solves in 64 tries against 13 only
the base). The hypothesis "pass@64 rises by less than pass@1" holds on every set.

**Countdown at n = 64** (think-tag, 500 x 64, 3 h 04 min): pass@1 11.9 [10.8, 13.1], pass@8 44.9,
pass@64 62.0 [57.8, 66.0], maj@64 47.8; truncation 79 %, mean 3,731 tokens. The base's ceiling is
63.8 with 43 % truncation. SFT left the ability where it was and made the model finish far less
often: the greedy row (3.8 %) is a length failure, not a search failure.

## 6. EXP-007 (LoRA vs full) and EXP-008 (data scaling)

**EXP-007 was not run.** The LoRA run started at 00:30 UTC on 19 Oct at 19.2 s per step (26 % slower
than the full fine-tune at the same micro-batch, because the embedding rows train in fp32 next to the
adapters), 3 h 50 min projected; with the session already at 14.8 node-hours and Phase 3 still to
come, it was stopped at step 20 (INR 45) in favour of the data-scaling curve, and the user confirmed
the cut at 01:50 UTC. Reported as not met, because of budget; the pre-registered config and the
amendment stand for a later session.

**EXP-008** (`figures/phase-2-data-scaling.png`; same LR, 2 epochs, nested seeded subsets, same
backend and micro-batch as the main run; 25 %: 180 steps, 46 min; 50 %: 360 steps, 1 h 31 min):

| eval | base | 25 % (5,750) | 50 % (11,500) | 100 % (23,000) |
|---|---|---|---|---|
| GSM8K | 61.8 [59.2, 64.4] | 70.8 [68.4, 73.2] | 72.1 [69.7, 74.5] | 74.8 [72.6, 77.2] |
| GSM8K perturbed | 51.4 [48.1, 54.8] | 64.1 [60.8, 67.6] | 61.3 [58.0, 64.6] | 66.0 [62.8, 69.5] |
| MATH-500 | 44.8 [40.4, 49.2] | 47.8 [43.4, 52.0] | 44.0 [39.8, 48.4] | 50.6 [46.4, 54.8] |
| Countdown | 1.2 [0.4, 2.2] | 1.6 [0.6, 2.8] | 2.0 [0.8, 3.4] | 3.8 [2.2, 5.6] |
| MATH-500 truncation | 20.0 % | 57.0 % | 58.4 % | 52.0 % |

Paired on the shared problems (computed on the laptop from the archived samples;
`phase-2-compare-data25-vs-100.md`, `-data50-vs-100.md`, `-data25-vs-50.md`): 100 % minus 25 % is
+4.0 [1.4, 6.7] on GSM8K, +1.9 [-1.3, 5.1] on perturbed GSM8K, +2.8 [-1.4, 7.0] on MATH-500, +2.2
[0.4, 4.2] on Countdown; 100 % minus 50 % is +2.7 [0.2, 5.5], +4.7 [1.4, 8.0], +6.6 [2.4, 11.2] and
+1.8 [0.0, 3.6]; 50 % minus 25 % is inside its CI everywhere (MATH-500 -3.8 [-8.4, 0.6]). Reading:
GSM8K rises with data at every step (the hypothesis said it would gain fewer than 3 points from 25 to
100 %; it gained 4.0), and most of the base-to-SFT gain is there at 25 % (9 of 13 points). MATH-500
is flat within noise across the three sizes: the 100 % point clears the 25 % point by 2.8 with a CI
that includes zero, so neither the hypothesis (at least 5) nor the falsifier (under 2 with a CI
including 0) is met; the honest statement is "inconclusive, and small if present". Truncation on
MATH-500 jumps from 20 % (base) to 52-58 % at every SFT size: the think-tag SFT teaches long
thinking on MATH-style problems at the first 5,750 examples and more data does not teach it to
stop. Countdown is flat at 2-4 % as hypothesised. Checkpoints: `ablations/sft_1p7b_data25` and
`sft_1p7b_data50` under the SFT Hub repo (with the three sweep finals).

## 7. What Phase 3 needs from here

The `sft` checkpoint (`Raviteja546/reasoning-rlvr-sft`, main final at the root; on the node's disk
at `outputs/train/sft_1p7b/final`), its think-tag eval (`outputs/eval/sft_1.7b_think_tag/metrics.json`,
the cap rule reads the p90), the training prompts with their overlap drops (`outputs/data/sft/train.jsonl`,
regenerated from the manifest), the dev split, the SFT n = 64 samples for the paired pass@k
(`sft_1.7b_n64_few_shot_cot` in the samples dataset `Raviteja546/reasoning-rlvr-samples`), and the
paired comparison script for every later checkpoint. The finding that matters most for Phase 3 is the
last paragraph of section 5: a quarter of the SFT checkpoint's think-tag generations never terminate,
so a method that selects finished, correct traces (RFT) has an obvious lever, and a Phase 3 gain
must be read next to the truncation column.

# Full results

The long form of the README: the ten questions the plan pre-registered and their answers, how each
phase's question arose, what each later phase found, and every result table. Each number carries a
95 % bootstrap CI over problems; comparisons between checkpoints are paired on shared problem ids;
greedy, batch-invariant vLLM unless stated. Every quoted claim re-derives from its per-sample files
in the evidence index (`outputs/reports/evidence/README.md`). The per-phase reports in
`outputs/reports/` hold the full tables; the chain of questions with the predictions written before
each run is [`research-narrative.md`](research-narrative.md).

## 1. The plan's ten questions, answered

`PLAN.md` section 6 fixed these on 13 Oct 2026, before any training.

| id | question | answer, with the number |
|---|---|---|
| Q1 | How much does each stage add? | In the trained format: SFT +13.0 / +5.8 (GSM8K / MATH-500), RFT +3.3 / +5.8, GRPO +4.1 / +7.0 (each against the previous checkpoint, CIs clear of 0; EXP-006, 010, 012). Decomposed, the on-policy gains sit on problems the previous policy did not finish; on problems both finished, the difference is inside its CI (README table). Cost per stage: RFT's training 12 minutes, a GRPO run 3.1 H100-hours |
| Q2 | Is the KL term needed at this scale? | Not for stability at 200 steps: with it off the run did not diverge, landed +2.2 / +5.6 above the main run in the trained format, at half the length and 15 times further from RFT, with the gain again on problems the main run truncated and nothing under zero-shot (EXP-014a; the hypothesis "within noise" failed in sign) |
| Q3 | Do dynamic sampling and clip-higher help? | No: within noise at 1.5x the compute (+1.7 [-0.1, 3.6] / -2.0 [-6.0, 2.0]); the prompt filter had already left 32 informative groups in 199 steps of 200 (EXP-014b, falsifier fired) |
| Q4 | Does length normalisation cause length growth? | Not measured: Dr. GRPO (EXP-014c) was deferred for the cap test, pre-registered and unrun |
| Q5 | Does RL add capability or sharpen? | On GSM8K, sharpening only: GRPO over RFT at k = 16 is -0.3 [-0.9, 0.3], no problem newly solved in 32 tries. On MATH-500, a small capability signal: +1.9 [0.4, 3.5] at k = 16, 10 problems newly solved and 2 lost (EXP-012) |
| Q6 | Where does the RL policy sit against voting? | Voting over two RFT samples matches the GRPO policy's greedy number at 2.6x the tokens; four beat it at 5x (+3.9 [1.1, 6.7] / +7.6 [3.8, 11.4]; EXP-017). With the 0.6B verifier, the better of two RFT samples beats it (+3.0 [0.1, 6.0] / +4.0 [0.2, 7.7]; EXP-022). A stop rule on RFT at 2,048 tokens is level with it on one machine (-1.2 [-3.2, 0.8] / +2.4 [-1.6, 6.6]) |
| Q7 | Does training on GSM8K and MATH transfer to Countdown? | Countdown rose (think-tag 3.8 after SFT, 7.0 after RFT, 14.0 after GRPO, +7.0 [3.6, 10.4] for GRPO) where the plan predicted no move; decomposed, the GRPO gain sits on problems RFT truncated (+11.2 [8.6, 14.0] there), so it is the same termination, not a transferred skill |
| Q8 | What survives distillation and 4-bit quantisation? | Dropped (22 Oct 2026; `docs/plan-after-phase-4.md` section 4) |
| Q9 | Are the gains reasoning or recall? | No memorisation signal: the perturbed-GSM8K gap is 10.1 [7.0, 13.3] for the base, 11.6 after RFT, 11.1 [8.2, 14.1] after GRPO; a names-only control moves nothing (-0.4 [-3.3, 2.5]); partial-prompt verbatim completion is unchanged by training (EXP-003, 003b) |
| Q10 | Which MATH levels does each stage move? | GRPO over RFT: level 1 +11.6 [0.0, 23.3], 2 +6.7 [-2.2, 14.5], 3 +13.3 [3.8, 22.9], 4 0.0 [-8.6, 9.4], 5 +7.5 [-0.7, 15.7]; the RL prompts were levels 1-3. Against SFT, every level gains (`outputs/reports/phase-4.md` section 4.1) |

## 2. How the question arose, phase by phase

Each phase's question came from the previous phase's numbers.

1. **Phase 0.** Three identical greedy evals differed by up to 6 points because vLLM's default scheduler
   is not batch-invariant; with batch-invariant mode they are bit-identical. The harness is deterministic
   before any model is compared (EXP-000).
2. **Phase 1.** The base model does 12 points better on MATH-500 with a bare instruction than with four
   Minerva shots, and the think-tag format the RL stages need costs it 16 points on GSM8K because 36 %
   of generations loop. pass@64 is 98.8 / 87.8 / 63.8 (GSM8K / MATH-500 / Countdown), so the latent
   headroom is large everywhere (EXP-001, 002). Two bars were set here: every trained model is read
   against the base model's best prompt, and every gain in the trained format is read against the loops.
3. **Phase 2.** SFT on distilled traces teaches the format (+13.0 on GSM8K in that format) but the
   trained format still loses to zero-shot because a quarter to a half of its generations never
   terminate (EXP-006). Data scaling buys +4.0 [1.4, 6.7] on GSM8K from 25 to 100 % of the data
   (EXP-008). The question written at the end of this phase, "better at mathematics or better at
   stopping", became the project's question.
4. **Phase 3.** Rejection sampling keeps only finished correct traces, so it selects for stopping whether
   intended or not. The controls were designed for that: the same tokens of distilled data gain nothing
   (EXP-010b, 010e), the loops do not finish at 8,192 tokens (EXP-010c), and the gain persists to k = 32
   in the trained format but the SFT and RFT pass@k curves coincide under 4-shot CoT (EXP-010f). Two
   mechanisms were still confounded, the correctness filter and the length filter, and two ablations were
   added to separate them (EXP-010g, 010h).
5. **Phase 4, session 1.** GRPO from the RFT checkpoint, 32 prompts x 8 samples, 200 steps, reward 1 only
   for a finished, well-formed, correct answer. Beats RFT on every set; the per-problem decomposition;
   a seed repeat for the noise row; pass@k for sharpening against capability. The method was stable
   (KL 0.006, no clipping, no hacking signal), the reward plateaued by step 50, and 30 % of groups had
   zero variance throughout.
6. **Phase 4, session 2.** Was the termination-only result an artefact of the 1,536-token cap, the KL
   term or wasted groups? Three one-knob runs said no (section 3).
7. **Phase 5.** If the gain is termination, what is it worth? A stop rule, votes and a verifier, all
   without training (section 3).
8. **Phase 6.** What is a loop, in the text? (section 3)
9. **Phase 7.** The write-up, and a check of the whole: a fresh clone reproduces the base-model table
   (EXP-024).

## 3. What the later phases found

**Session 2 (21 Oct 2026; `outputs/reports/phase-4.md` sections 10-18).** Three one-knob runs from the
same checkpoint, read against the noise row. The pre-registered prediction for the cap run was that the
both-finished paired difference against RFT would rise by at least 2 points and pass@16 by at least 3;
the falsifier was both inside the main run's CIs. Both-finished came out at 0.0 [-4.1, 4.1] and pass@16
at +0.8 [-0.6, 2.3], so the falsifier fired. The reference model off was predicted to land within
noise; it landed above the main run on three sets, at half the length, 15 times further from RFT, with
the gain again on problems the main run truncated and nothing under zero-shot. Dynamic sampling, built
for the third of groups with zero variance, changed nothing: the prompt filter had already left 32
informative groups in 199 steps of 200.

**Phase 5, option A (21-22 Oct 2026, no rental; `outputs/reports/phase-5.md`).** EXP-017, from the
existing sample sets: voting over two RFT samples matches the GRPO policy's greedy number at 2.6x the
tokens and four samples beat it at 5x (+3.9 [1.1, 6.7] GSM8K, +7.6 [3.8, 11.4] MATH-500); from four
samples on, the majority-vote curves of SFT, RFT, GRPO and the cap run coincide; the untrained base
under a 4-shot prompt with 16 votes matches GRPO's greedy GSM8K number and never its MATH-500 number.
EXP-016, on the laptop: forcing the stop (append the closing tags at a token budget, no training) on
the SFT checkpoint gains +4.6 [3.4, 6.0] GSM8K and +7.2 [4.0, 10.4] MATH-500 at 1,536 tokens against
the same checkpoint decoded on the same machine, more than RFT's whole same-machine gain (+2.9 / +4.6)
and 75 % / 78 % of GRPO's (+6.1 / +9.2). The forced answers are right about half the time, so the
loops hold a usable derivation, not a finished one. The first-quoted EXP-016 numbers (+5.8 / +9.0)
paired laptop runs with H100 runs; EXP-020 showed why that matters.

**Phase 5B (22-24 Oct 2026, laptop, no rental).** EXP-020: the same checkpoint decoded greedily on the
H100 and on the laptop differs on 10-23 % of problems (three checkpoints, net -1.8 to +1.8 points), so
batch-invariant decoding is reproducible per machine, not across GPUs; every comparison against an
unforced checkpoint was re-paired on one machine, and the "cost of the rule" first reported was mostly
that machine term. Forcing the stop on the GRPO checkpoint at 2,048 gains +2.0 [0.6, 3.6] on MATH-500
at 27 % fewer tokens against its own same-machine run (the first, cross-machine pairing read this as
nothing), and with RFT and GRPO both stopped at 2,048, GRPO's MATH-500 gain over RFT is gone (-0.4
[-4.4, 3.4]; +1.6 [-0.4, 3.6] on GSM8K). EXP-021: appending "Wait" before the stop equals the plain
stop at the same total tokens on all four cells; a looping model given "Wait" mostly loops on.
EXP-022, an outcome reward model trained on the laptop (a LoRA head on Qwen3-0.6B-Base, 20,000 of
SFT's labelled samples), reached 89.4 % held-out accuracy (AUROC 0.944): with it, two RFT samples beat
the GRPO policy's greedy number (+3.0 [0.1, 6.0] / +4.0 [0.2, 7.7]), but its arg-max is over-optimised
as k grows (best-of-32 is 12-13 points below majority-of-32 on MATH-500), while the reward-weighted
vote stays at or above majority vote at every k.

**Phase 6 (23 Oct 2026, no GPU; `outputs/reports/phase-6.md`).** In the trained format 96-99 % of
truncated GSM8K completions (76-90 % on MATH-500) contain a recurring 50-word window; the repeated
unit is a 100-190-word step; it starts near word 500 (GSM8K) or 800 (MATH-500) whatever the window,
so more room adds cycles, not progress. The training changed how often a loop happens, not what it
is (Kolmogorov-Smirnov distances SFT against GRPO 0.11-0.16). Under zero-shot the trained checkpoints
rewrite their whole solution after boxing the answer, which is why truncated zero-shot completions
are usually right. Half to two thirds of the forced answers that are right had already been stated
before the stop.

## 4. The tables

**Base models (Phase 1, commit ffd3018).** pass@1 %, full eval sets, 1,024-token window.

| model | prompt | GSM8K | GSM8K perturbed | MATH-500 | Countdown |
|---|---|---|---|---|---|
| Qwen3-1.7B-Base | 4-shot-CoT | 74.5 [72.0, 77.0] | 67.8 [64.8, 71.1] | 41.4 [37.0, 45.8] | - |
| Qwen3-1.7B-Base | zero-shot | 77.6 [75.3, 79.9] | 70.0 [67.0, 73.1] | 53.6 [49.4, 57.6] | - |
| Qwen3-1.7B-Base | think-tag | 61.3 [58.7, 63.8] | 50.8 [47.6, 54.5] | 43.0 [38.8, 47.4] | 1.2 [0.4, 2.2] |
| Qwen3-0.6B-Base | 4-shot-CoT | 61.9 [59.3, 64.6] | 56.5 [53.1, 59.9] | 28.0 [24.0, 31.8] | - |
| Qwen3-0.6B-Base | zero-shot | 64.9 [62.3, 67.5] | 59.5 [56.1, 62.9] | 37.6 [33.4, 41.6] | - |

**After training** (Qwen3-1.7B-Base, greedy, 4,096-token window; paired differences against the previous
checkpoint on shared problems; 18-21 Oct 2026, TIR H100):

| checkpoint | prompt | GSM8K | GSM8K perturbed | MATH-500 | Countdown | truncated GSM8K / MATH-500 |
|---|---|---|---|---|---|---|
| base | think-tag | 61.8 [59.2, 64.4] | 51.4 [48.1, 54.8] | 44.8 [40.4, 49.2] | 1.2 [0.4, 2.2] | 36 / 20 % |
| SFT (EXP-006) | think-tag | 74.8 (+13.0 [10.0, 15.8]) | 66.0 (+14.6) | 50.6 (+5.8 [0.6, 11.0]) | 3.8 (+2.6) | 28 / 52 % |
| RFT (EXP-010) | think-tag | 78.2 (+3.3 [1.0, 5.8]) | 69.0 (+3.0) | 56.4 (+5.8 [1.6, 10.2]) | 7.0 (+3.2) | 15 / 40 % |
| GRPO (EXP-012) | think-tag | 82.3 (+4.1 [2.0, 6.1]) | 73.9 (+4.8 [1.9, 7.6]) | 63.4 (+7.0 [2.8, 11.0]) | 14.0 (+7.0 [3.6, 10.4]) | 5 / 21 % |
| GRPO seed 1 (EXP-013) | think-tag | 82.0 (-0.3 vs seed 0) | 73.6 (-0.2) | 58.6 (-4.8 [-9.0, -0.8]) | 14.6 (+0.6) | |
| base | zero-shot | 77.9 [75.6, 80.1] | 69.9 | 53.8 [49.4, 58.2] | - | 1-9 % |
| SFT | zero-shot | 82.2 (+4.3 [1.8, 6.6]) | 72.7 | 63.2 (+9.4 [5.0, 14.0]) | - | 17 / 59 % |
| RFT | zero-shot | 81.1 (-1.1) | 74.7 (+2.0) | 64.8 (+1.6) | - | |
| GRPO | zero-shot | 84.3 (+3.2 [1.5, 4.8]) | 74.7 | 64.4 (-0.4) | - | 16 / 65 % |
| GRPO, KL off (EXP-014a) | think-tag | 84.5 (+2.2 [0.4, 4.0] vs GRPO) | 77.1 (+3.3) | 69.0 (+5.6 [1.6, 9.4]) | 14.8 (+0.8) | 2 / 13 % |
| GRPO, KL off | zero-shot | 82.8 (-1.5 [-3.3, 0.3] vs GRPO) | 75.1 (+0.4) | 66.2 (+1.8 [-2.0, 5.4]) | - | 6 / 23 % |
| GRPO, dynamic sampling (EXP-014b) | think-tag | 84.0 (+1.7 [-0.1, 3.6] vs GRPO) | 74.8 (+1.0) | 61.4 (-2.0 [-6.0, 2.0]) | 12.0 (-2.0) | 4 / 22 % |
| GRPO, 2,048 cap (EXP-015) | think-tag | 82.5 (+0.2 [-1.7, 2.2] vs GRPO) | 72.3 (-1.6) | 60.8 (-2.6 [-6.4, 1.2]) | 12.4 (-1.6) | 10 / 26 % |

**pass@k.** 4-shot-CoT, n = 64, T = 0.6: base 98.8 / 90.2 / 87.8 / 63.8 (GSM8K / perturbed / MATH-500 /
Countdown under think-tag), SFT 98.2 / 88.8 / 93.0 / 62.0, RFT (300-problem subsets) 98.7 / 92.3 / 94.0.
In the trained format at T = 0.6, n = 32, on pinned 300-problem subsets: RFT's gain over SFT persists at
+2.0-2.5 to k = 32 on MATH-500; GRPO over RFT is -0.3 [-0.9, 0.3] at k = 16 on GSM8K (no problem newly
solved in 32 tries) and +1.9 [0.4, 3.5] on MATH-500 (10 problems newly solved, 2 lost); the 2,048-cap run
over RFT +0.8 [-0.6, 2.3] at k = 16 on MATH-500 (6 newly solved, 3 lost). Figures:
`outputs/reports/figures/`.

**The stop rule (Phases 5 and 5B; greedy think-tag, full sets, all on the laptop).** Forced at 768 /
1,024 / 1,536 / 2,048 / 3,072 tokens: SFT GSM8K 77.8 / 79.8 / 80.7 / 80.6 / 80.8 (752 to 1,581 mean
tokens), MATH-500 50.6 / 56.2 / 59.6 / 61.0 / 62.2 (776 to 2,419); RFT GSM8K 78.2 / 79.8 / 80.8 / 80.9 /
80.9, MATH-500 51.6 / 58.0 / 61.8 / 64.0 / 64.6; GRPO at 1,536 / 2,048: GSM8K 82.5 / 82.5, MATH-500 62.6
/ 63.6. Unforced on the laptop: SFT 76.0 / 52.4, RFT 78.9 / 57.0, GRPO 82.1 / 61.6 (on the H100 74.8 /
50.6, 78.2 / 56.4, 82.3 / 63.4). Same-machine pairings: `outputs/reports/phase-5.md` sections 2b and 4;
full table `outputs/reports/figures/phase-5-force-stop.md`; the final chart's table
`outputs/reports/figures/final-accuracy-vs-tokens.md`.

**Controls.** Perturbation gap on GSM8K (numbers resampled, paired on 830 problems): 10.1 [7.0, 13.3] for
the base, 11.6 after RFT, 11.1 [8.2, 14.1] after GRPO; a names-only control moves nothing (-0.4 [-3.3,
2.5]). Partial-prompt verbatim completion: GSM8K 1.1 %, MATH-500 11.6 % (long held-outs), unchanged by
training. Phase 3 ablations: the same tokens of distilled data instead of the selected traces gain nothing
(74.8 to 74.1 GSM8K); unfiltered traces land on RFT but were 93 % correct by construction (EXP-010g, no
power); random instead of shortest traces -2.4 [-4.7, -0.1] GSM8K (EXP-010h). GRPO training: reward 0.34
to 0.68, finished rate 0.35 to 0.75, mean completion 1,406 to 1,201 tokens with p90 at the cap, KL 0.006,
clip fraction 0, zero-variance groups 30 % flat, 56 s/step on one H100. KL off: offline KL to RFT 0.098
per token against the main run's 0.006, finished rate 0.96, mean completion 611 tokens; dynamic
sampling: 43.5 informative groups of 64 per step, one zero-advantage fill in 200 steps, 85 s/step; the
2,048 cap: finished rate 0.82, p90 at the cap throughout, 72 s/step.

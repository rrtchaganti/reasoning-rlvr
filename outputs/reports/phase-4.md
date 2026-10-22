# Phase 4 report: GRPO from the RFT checkpoint, shape B

Branch `phase-4-grpo`. Code and the reference-vs-TRL correctness check on the laptop, 20 Oct 2026;
session 1 (the two Phase 3 ablations, the debugging run, the main run, the seed repeat, the evals)
on the E2E TIR H100 SXM 80 GB notebook on 20-21 Oct 2026 at commit a8b9f69; session 2 (EXP-014a,
EXP-014b, EXP-015, their evals and the cap run's sampling) on the same notebook on 21 Oct 2026 at
commit 8e94fb5. Sections 1-9 written 21 Oct 2026 from the session 1 archive, sections 10-18 the
same day from the session 2 archive (`../h100_p4_session2_archive`, sha256 683c621e...). Numbers are greedy,
batch-invariant vLLM, 95 % bootstrap CIs over problems, paired on shared problem ids unless stated;
the paired tables were computed on the laptop from the per-sample files
(`outputs/reports/phase-4-compare-*.md`, `phase-3-compare-{nofilter,random2}-vs-rft.md`).
Pre-registration: `experiments.md` EXP-010g, EXP-010h, EXP-012, EXP-013 (this report) and
EXP-014a, EXP-014b, EXP-015 (sections 10-18; EXP-014c deferred). Plan: `docs/phase-4-plan.md`.

## 1. Summary (session 1)

Two hundred steps of GRPO from `rft` (32 prompts x 8 samples per step on the learnable prompts,
MATH levels 1-3 in full plus a seeded fifth of GSM8K-train; 1,536-token cap; reward 1 only for a
finished, well-formed, correct completion; KL 0.04 to `rft`) beat `rft` in the trained think-tag
format on every set with 500 or more problems, paired: **GSM8K +4.1 [2.0, 6.1], perturbed GSM8K
+4.8 [1.9, 7.6], MATH-500 +7.0 [2.8, 11.0], Countdown +7.0 [3.6, 10.4]**. The EXP-012 falsifier
(GRPO at or below `rft`) did not fire. The seed repeat reproduces the GSM8K-family and Countdown
gains to within a point (+3.8, +4.6, +7.6) but lands at +2.2 [-2.0, 6.6] on MATH-500, so the
**MATH-500 gain is positive in both seeds and its size is uncertain**: the noise row the ablations
are read against is 0.3 points on the GSM8K family, 0.6 on Countdown and 4.8 on MATH-500.

What the gain is made of. Decomposed per problem by whether the `rft` completion finished (section
4.5), **the whole paired gain sits on problems `rft` truncated**: +4.2 [3.0, 5.5] of the +4.1 on
GSM8K, +10.2 [7.4, 13.2] of the +7.0 on MATH-500, +11.2 [8.6, 14.0] of the +7.0 on Countdown; on the
problems `rft` finished the net is -0.2, -3.2 and -4.2 (GRPO truncates a few that `rft` finished),
and on problems both runs finished the paired accuracy difference is +1.1 [-0.6, 2.8] on GSM8K and
0.0 [-4.3, 3.9] on MATH-500. Truncation at 4,096 tokens fell from 15 % to 5 % on GSM8K and from 40 %
to 21 % on MATH-500, with completions 30 % shorter. Under the 4-shot CoT template the gain is 0.0 on
GSM8K and +1.8 [-1.0, 4.8] on MATH-500; under zero-shot +3.2 [1.5, 4.8] on GSM8K and -0.4 on
MATH-500. Phase 3's RFT step decomposes the same way (section 4.5): at this scale both post-SFT
stages are termination gains in the trained format. The pass@k
curves at T = 0.6 (section 6) say what Phase 3 said of RFT: on GSM8K pure sharpening (pass@16 -0.3,
no problem newly solved in 32 tries), on MATH-500 mostly sharpening with a small persistent gain
(+1.9 [0.4, 3.5] at k = 16; 10 problems solved by GRPO in 32 tries that `rft` never solved, against
2 the other way). The perturbation gap (11.1 [8.2, 14.1] against `rft`'s 11.6) and the partial-prompt
probe are unchanged: no new memorisation signal. The reward-hacking watch flagged only the
repeated-answer rate, which rose with accuracy; the other four signals were off in both runs, and
the flag now carries a correctness term (section 7).

The two Phase 3 ablations: training on the model's own finished traces without the correctness
filter lands on `rft` (+0.2 [-2.0, 2.4] GSM8K, -1.4 [-5.8, 3.0] MATH-500) but the set it trained on
was 93 % correct by construction, so the experiment has no power to separate selection from
self-distillation (section 3); choosing two *random* correct traces per problem instead of the two
shortest costs 2.4 [0.1, 4.7] points on GSM8K and leaves GSM8K truncation 7 points higher, with
MATH-500 unchanged: about half of RFT's stopping effect on GSM8K is length selection.

Session 1 cost 13.3 node-hours including the Stop (INR 5,030), against 12.0 planned; the work itself
finished inside the estimate and the overrun is 88 idle minutes of a stalled environment build plus
the idle time until the user's Stop. Micro-batch 32 did not fit beside vLLM; every GRPO run used
micro-batch 16 with gradient accumulation 16 (the same 256-completion update).

**Session 2 (21 Oct 2026; sections 10-18).** Three one-knob runs from the same start, read against the
noise row. **KL off (EXP-014a)** beats the main run on GSM8K (+2.2 [0.4, 4.0]), perturbed GSM8K
(+3.3 [1.1, 5.5]) and MATH-500 (+5.6 [1.6, 9.4]) at half the completion length and a third of the
truncation; decomposed, the gain is again on problems the main run truncated (+5.4 of +5.6 on
MATH-500; both-finished +2.7 [-1.3, 6.5]), so the penalty was a brake on the termination gain, not a
source of accuracy; one non-termination signal, +2.2 [0.1, 4.3] on perturbed GSM8K among problems both
finished, is small and single-seed. **Dynamic sampling (EXP-014b)** is inside the noise row on every
set at 1.5x the step time; the learnable-prompt filter left 32 informative groups in 199 of 200 steps,
so its falsifier fired. **The 2,048-token cap (EXP-015)** is inside the main run's CIs everywhere, with
pass@16 on MATH-500 at +0.8 [-0.6, 2.3] against `rft` and the both-finished paired difference against
the main run at +1.2 [-3.0, 5.2]: its falsifier fired, the cap is not what limits the reasoning gain,
and the "termination only" reading of EXP-012 stands as a property of the method at this scale and
budget rather than of the cut. Session 2 cost 13.0 node-hours (INR 4,930), nothing dropped; Phase 4
in all 26.3 node-hours, INR 9,960, plus the disk.

## 2. The runs as they were made

| run | what | commit | steps | wall-clock | s/step | peak GB | W&B |
|---|---|---|---|---|---|---|---|
| `rft_1p7b_nofilter` (EXP-010g) | RFT on 11,581 finished traces, correct or not, shortest two per problem | a8b9f69 | 56 | 12 min | 13 | 48 | fid4o2yz |
| `rft_1p7b_random2` (EXP-010h) | RFT on 11,145 correct traces, two at random per problem | a8b9f69 | 56 | 14 min | 15 | 48 | u4xl8kxf |
| `grpo_1p7b_debug` | 50 GRPO steps on all 4,606 learnable prompts, GSM8K fraction 1.0; the rate measurement (not a result) | a8b9f69 | 50 | 45 min | 54.1 | 70.3 | uot02yla |
| `grpo_1p7b` (EXP-012, main, seed 0) | shape B, 2,735 prompts (468 GSM8K + 2,267 MATH) | a8b9f69 | 200 | 3 h 08 min | 56.3 | 70.4 | 4b2hf5vo |
| `grpo_1p7b_seed1` (EXP-013) | the same with seed 1 | a8b9f69 | 200 | 3 h 07 min | 56.2 | 70.4 | w0dhqitz |

Shape B as pre-registered (`docs/phase-4-plan.md` section 2) with one change made on the node:
`per_device_batch` 16 instead of 32 (two out-of-memory failures in the debugging run at 32: the
fp32 logits of 32 completions at the cap are about 33 GB on top of 58 GB of weights, optimiser
state, reference model and vLLM's 30 % share), `vllm_gpu_memory_utilization` 0.30 instead of 0.35,
and PyTorch's expandable-segments allocator. Gradient accumulation is then 16 micro-batches per
optimiser step, so the update is the same 256-completion batch; TRL's `grpo` loss averages
per-sequence losses inside a micro-batch and the trainer averages the accumulated micro-batches, so
2 x 16 equals 1 x 32 up to floating-point rounding. The user decided this at 15:23 UTC on 20 Oct for
every run alike, before the main run; the committed config and the preflight now carry the two
values (fb856a8). Every launch passed the preflight: the knob table, the prompt-file hash
281bdfe31176411b with 2,735 prompts used (4,606 for the debugging run), the start checkpoint
`rft_1p7b` from commit bb6f2a6, and the pinned eval subsets (gsm8k 58c8ded1c644f332, math500
5ca9795b45227236, countdown 9698ed78ea4e0491) checked in `setup` and again before the sampling.

The debugging run measured 54.1 s per step at micro-batch 16, projecting 3.3 h for 200 steps with
dev evals against the plan's 3.8 h (1.5x bound 5.7 h), so the main run proceeded under the
ten-minute rule; both 200-step runs then ran at 56.2-56.3 s per step (60 s at step 10, falling as
completions shortened). The `rft` dev-split baseline every dev curve is read against: gsm8k_dev
86.0 [81.6, 90.0], math_dev 62.4 [56.4, 68.0] (250 + 250 training-split problems, greedy think-tag,
4,096 tokens, 8.8 % and 33.2 % truncation).

## 3. The two Phase 3 ablations: selection, self-distillation and length

Paired against `rft` (think-tag greedy, full sets, `phase-3-compare-{nofilter,random2}-vs-rft.md`):

| eval | pairs | `rft` | nofilter (EXP-010g) | nofilter - rft | random2 (EXP-010h) | random2 - rft |
|---|---|---|---|---|---|---|
| gsm8k | 1,319 | 78.2 | 78.3 | +0.2 [-2.0, 2.4] | 75.7 | **-2.4 [-4.7, -0.1]** |
| gsm8k_perturbed | 830 | 69.0 | 68.0 | -1.1 [-3.9, 1.7] | 68.0 | -1.1 [-4.1, 1.8] |
| math500 | 500 | 56.4 | 55.0 | -1.4 [-5.8, 3.0] | 57.8 | +1.4 [-3.2, 6.0] |
| countdown | 500 | 7.0 | 5.4 | -1.6 [-4.4, 1.2] | 5.8 | -1.2 [-3.8, 1.6] |

Truncation at 4,096 tokens and mean length, think-tag greedy (`sft` 28.1 % / 1,887 tokens on GSM8K,
52.0 % / 2,910 on MATH-500):

| run | kept-set tokens (mean, p90) | gsm8k trunc. / tokens | math500 trunc. / tokens |
|---|---|---|---|
| `rft` | 1,207 / 1,779 | 15.2 % / 1,426 | 39.6 % / 2,601 |
| nofilter | 1,211 / 1,789 | 15.3 % / 1,429 | 38.8 % / 2,546 |
| random2 | 1,438 / 2,001 | 22.0 % / 1,708 | 39.4 % / 2,660 |

**EXP-010g, no correctness filter.** The hypothesis (at most half of RFT's gain) and the falsifier
(within 1 point of `rft` on both sets with CIs covering 0) were written for a set in which a
substantial fraction of traces would be wrong. The set that the builder produced with the filter
off has 11,581 examples of which **6.9 % are incorrect**: the shortest two finished, well-formed
traces of a learnable problem are mostly correct ones, because wrong traces on these problems tend
to be the long, looping ones. The checkpoint lands on `rft` (GSM8K +0.2, inside the falsifier's
1-point band with the CI covering 0; MATH-500 -1.4, outside the band but with the CI covering 0),
with identical truncation and length. Read literally, the GSM8K half of the falsifier fires. Read
honestly, the experiment cannot separate the two mechanisms: 800 wrong traces among 11.6k is a 7 %
contamination of the kept set, and a 7 % contamination costing under a point is what either
hypothesis predicts. The Phase 3 wording ("one round of selection") stands, with this experiment
recorded as inconclusive by construction. A test with power needs a set whose incorrect fraction is
matched to the correct one (two incorrect traces per problem where they exist, or a 50/50 draw),
about 25 minutes on the H100; it is not in session 2's budget and is listed as a debt.

**EXP-010h, random two instead of shortest two.** The hypothesis was accuracy within 2 points of
`rft` and a truncation reduction at least 3 points smaller; the falsifier was truncation within 3
points of `rft`'s on both sets. On GSM8K the random-two checkpoint is 2.4 [0.1, 4.7] points below
`rft` (the CI excludes 0, just outside the hypothesised 2) and its truncation is 22.0 % against 15.2
% (6.8 points higher: of RFT's 12.9-point fall from `sft`, about half is length selection); on
MATH-500 accuracy is +1.4 [-3.2, 6.0] and truncation is the same 39 %. The falsifier did not fire;
the hypothesis holds on MATH-500 and misses on GSM8K by the accuracy margin. The kept set is 19 %
longer on average (1,438 vs 1,207 tokens) and the checkpoint's GSM8K completions 20 % longer; the
accuracy cost on GSM8K goes with the extra truncation (the gap between the two runs' "only" counts
is 137 vs 105 problems). Conclusion: on GSM8K, RFT's stopping effect is about half length selection
and half "finished traces at all"; on MATH-500, where even the shortest correct traces are long,
selecting for length buys nothing and the stopping effect is entirely the finished-only filter.

## 4. EXP-012: GRPO against `rft`, paired

### 4.1 Think-tag greedy, the headline (`phase-4-compare-think_tag.md`)

| eval | pairs | `rft` | grpo | grpo - rft [paired CI] | only rft / only grpo | trunc. rft -> grpo | tokens rft -> grpo |
|---|---|---|---|---|---|---|---|
| gsm8k | 1,319 | 78.2 [76.0, 80.3] | 82.3 [80.1, 84.3] | **+4.1 [2.0, 6.1]** | 66 / 120 | 15.2 -> 4.6 % | 1,426 -> 999 |
| gsm8k_perturbed | 830 | 69.0 [65.8, 72.2] | 73.9 [70.8, 76.7] | **+4.8 [1.9, 7.6]** | 46 / 86 | 16.9 -> 6.3 % | 1,530 -> 1,076 |
| math500 | 500 | 56.4 [52.0, 60.4] | 63.4 [59.0, 67.4] | **+7.0 [2.8, 11.0]** | 39 / 74 | 39.6 -> 20.8 % | 2,601 -> 1,944 |
| countdown | 500 | 7.0 [4.8, 9.4] | 14.0 [11.2, 17.2] | **+7.0 [3.6, 10.4]** | 22 / 57 | 87.8 -> 72.6 % | 3,654 -> 3,188 |
| aime24 | 30 | 0.0 | 3.3 [0.0, 10.0] | +3.3 [0.0, 10.0] | 0 / 1 | 93 -> 57 % | 3,841 -> 3,253 |

Against `sft` (`phase-4-compare-sft-think_tag.md`): +7.4 [5.2, 9.6] GSM8K, +7.8 [5.1, 10.5]
perturbed, +12.8 [8.8, 16.8] MATH-500, +10.2 [7.0, 13.8] Countdown. The three-stage picture in the
trained format on MATH-500: `sft` 50.6, `rft` 56.4, `grpo` 63.4; on GSM8K 74.8, 78.2, 82.3. The
pipeline from the base model's think-tag 43.0 / 61.3 (Phase 1) adds 20 and 21 points in this
format; against the base model's best prompt (zero-shot 53.8 MATH-500, 77.9 GSM8K; Phase 1) the
checkpoint's own zero-shot rows (64.4, 84.3) add 11 on MATH-500 and 6 on GSM8K (section 4.3). (An
earlier draft of this sentence quoted `sft`'s zero-shot 63.2 as the base's; corrected 21 Oct.)

MATH-500 by level, grpo - rft (paired; 43 / 90 / 105 / 128 / 134 problems): level 1 +11.6 [0.0,
23.3], level 2 +6.7 [-2.2, 14.5], level 3 +13.3 [3.8, 22.9], level 4 0.0 [-8.6, 9.4], level 5 +7.5
[-0.7, 15.7]. The RL prompts were levels 1-3; levels 1-3 carry the gain (CIs excluding 0 at levels
1 and 3), level 4 none, level 5 a positive point estimate with the CI touching 0. Against `sft`,
every level gains (+18.6, +5.6, +16.2, +11.7, +14.2); the level-4 flatness is specific to the GRPO
step, where `rft` had gained most (+11.7 at level 4 in Phase 3).

### 4.2 Hypotheses and falsifiers, read one by one

| pre-registered (EXP-012) | observed | verdict |
|---|---|---|
| +2 to +5 MATH-500, +1 to +3 GSM8K, one CI excluding 0 | +7.0 [2.8, 11.0], +4.1 [2.0, 6.1] | larger than hypothesised; both CIs exclude 0 |
| truncation under 10 % GSM8K, under 25 % MATH-500 | 4.6 %, 20.8 % | met |
| dev accuracy rises within 100 steps, plateaus by 200 | dev curves within their 4-6 point CIs at every mark (section 7); no plateau call possible from 250 problems | not decidable at this dev size |
| pass@32 moves under 2 on GSM8K, 0-3 on MATH-500 | GSM8K: pass@16 -0.3 [-0.9, 0.3], no problem newly solved in 32 tries; MATH-500 pass@16 +1.9 [0.4, 3.5], +2.7 at n = 32 by the flip table | met (sharpening on GSM8K; small persistent gain on MATH-500) |
| Countdown within its CI | +7.0 [3.6, 10.4], from 7.0 to 14.0 | **not met: Countdown doubled** |
| perturbation gap within 3 points of `rft`'s | 11.1 [8.2, 14.1] vs 11.6 [8.2, 14.9] (greedy, 830 pairs) | met (the pre-registration quoted `rft`'s n = 64 gap, 9.2; the greedy gap is the like-for-like one) |
| probe rates within Phase 3 values | gsm8k 2.6 % [1.9, 3.6] vs 2.4 %; math500 14.1 % [11.3, 17.4] vs 14.3 % | met |
| length falls or holds while accuracy rises | mean tokens 1,406 -> 1,201 in training; -30 % at eval; accuracy up | met |
| **falsifier 1:** grpo - rft at or below 0 on both sets, CIs excluding +1 | +4.1 and +7.0 | did not fire |
| **falsifier 2 (method):** truncation at step 200 above step 0, or format rate under 95 %, or length +30 % with flat dev | training truncation 65 % -> 25 %; format rate among finished completions 100 % at every step; eval tag-format rate 95.4 % on GSM8K and 79.2 % on MATH-500; length -15 % | did not fire on the intended reading; see the note |

The note on falsifier 2. "Format rate below 95 %" was written without saying whether a truncated
completion counts as badly formatted. At the 4,096-token eval the tag-format rate on MATH-500 is
79.2 % because 20.8 % of completions never reach the closing tag, which is truncation, not format
gaming; among finished completions the rate is 100 % in every reward call of both runs
(`format_rate` equals `finished_rate` at every logged step). The intended signal (the policy
emitting the tags without the work) is the `format_only_rate`, which stayed under 2 %. The rule is
recorded as under-specified and read by its intent; a reader who takes it literally has the 79 %
and the explanation side by side.

The Countdown surprise. The hypothesis said Countdown would stay within its CI because no
Countdown problem is in the RL prompts. It doubled in both seeds (+7.0 and +7.6, seed difference
0.6) with truncation falling from 88 % to 73 % and length from 3,654 to 3,188 tokens: the policy
learnt to finish, and on Countdown, where 90 % of `sft`'s completions never finished, finishing is
most of the game. The pass@64 ceiling of the base model on Countdown is 63.8 (Phase 1); greedy
think-tag at 14.0 is far from it, so this is a termination gain on a task the model can already
solve one time in four at sampling, not new arithmetic. It is the one place where "learning to
stop" transferred across tasks, and it argues that the stopping behaviour is a property of the
policy, not of the trained distribution.

### 4.3 The other two templates (`phase-4-compare-few_shot_cot.md`, `-zero_shot.md`)

| template | gsm8k | gsm8k_perturbed | math500 |
|---|---|---|---|
| 4-shot CoT, grpo - rft | +0.1 [-1.5, 1.8] | 0.0 [-1.9, 1.9] | +1.8 [-1.0, 4.8] |
| zero-shot, grpo - rft | +3.2 [1.5, 4.8] | 0.0 [-1.8, 1.8] | -0.4 [-3.8, 3.0] |
| absolute, grpo: 4-shot / zero-shot / think-tag | 79.3 / 84.3 / 82.3 | 71.4 / 74.7 / 73.9 | 57.4 / 64.4 / 63.4 |

The gain lives in the trained format and, on GSM8K only, in zero-shot. Under 4-shot CoT nothing
moved; under zero-shot MATH-500 (65 % truncation at 4,096 tokens, as for `rft`) nothing moved
either. The trained-format gain on MATH-500 coincides with a 19-point fall in truncation in that
format and no change in truncation under the other templates: the GRPO step taught the policy to
finish inside the think-tag format, which is where its reward lived, and the accuracy it buys is
largely the accuracy of answers that now arrive. Zero-shot remains the best prompt for this
checkpoint on both sets (84.3 and 64.4), as it was for `sft` and `rft`; the format gap it opened in
Phase 2 (zero-shot minus think-tag: `sft` +7.4 on GSM8K, `rft` +2.9) has narrowed to +2.0.

### 4.4 Memorisation controls

Perturbation gap, think-tag greedy, 830 pairs: original 84.9 [82.3, 87.3], perturbed 73.9 [70.7,
77.0], **gap 11.1 [8.2, 14.1]**, flip rate 0.202 (`sft` 11.3 / 0.267, `rft` 11.6 / 0.251). The gap
is the same within a point across the three stages; the flip rate fell with accuracy. Partial-prompt
probe (fraction 0.6, 192 tokens): GSM8K mean similarity 0.591 (chance 0.404), rate at 0.9 2.6 % [1.9,
3.6]; MATH-500 0.615 (chance 0.295), 14.1 % [11.3, 17.4]; `sft` 0.589 / 2.2 % and 0.613 / 14.1 %;
`rft` 0.590 / 2.4 % and 0.616 / 14.3 %. Nothing in Phase 4 moved either control.

### 4.5 Where the gain is: the paired gain split by termination (`phase-4-gain-decomposition.md`)

The user asked how "mostly by learning to finish" was known rather than assumed. It was an
inference from correlations (truncation fell where accuracy rose; the other templates did not
move; the finished rate rose in training), and the per-sample files allow a direct split. For
every shared problem, the first sample of each run gives (finished, correct); the paired gain in
points of the set is the sum of per-problem differences, and it splits exactly into the part on
problems the `rft` completion truncated and the part on problems it finished
(`scripts/decompose_gain.py`, 95 % bootstrap CIs over problems):

| eval | net gain | on problems `rft` truncated | on problems `rft` finished | both finished (n) | accuracy there, `rft` -> grpo | paired diff there |
|---|---|---|---|---|---|---|
| gsm8k | +4.1 [2.0, 6.1] | **+4.2 [3.0, 5.5]** | -0.2 [-1.7, 1.4] | 1,089 | 88.5 -> 89.6 | +1.1 [-0.6, 2.8] |
| gsm8k_perturbed | +4.8 [2.2, 7.6] | **+4.9 [3.3, 6.7]** | -0.1 [-2.2, 1.9] | 668 | 78.7 -> 80.1 | +1.3 [-0.9, 3.7] |
| math500 | +7.0 [2.8, 11.2] | **+10.2 [7.4, 13.2]** | -3.2 [-6.0, -0.2] | 279 | 86.7 -> 86.7 | 0.0 [-4.3, 3.9] |
| countdown | +7.0 [3.8, 10.6] | **+11.2 [8.6, 14.0]** | -4.2 [-6.0, -2.4] | 25 | 68.0 -> 56.0 | -12.0 [-28.0, 4.0] |

The cells behind the MATH-500 row: of 117 problems `rft` truncated and GRPO finished, GRPO is right
on 54 that `rft` was not and wrong on 4 the other way; of 279 problems both finished, 18 flip each
way; of 23 problems `rft` finished and GRPO truncated, `rft` is right on 16. Seed 1 decomposes the
same way (+8.0 [5.4, 10.8] on `rft`-truncated problems, -5.8 on `rft`-finished, both-finished
-1.5 [-5.6, 2.6]; `phase-4-gain-decomposition-seed1.md`): the 4.8-point seed difference on
MATH-500 is a difference in how many `rft`-finished problems seed 1 truncates (34 against 23) and
loses, not a difference in accuracy among finished answers. Under zero-shot, where GRPO changed
truncation little, the split is flat (`-zero_shot.md`: GSM8K +1.7 and +1.5, MATH-500 +0.8 and
-1.2, every CI covering or touching 0).

Two caveats keep this descriptive. Truncation is an outcome of each policy, so the cells are not
fixed strata and a policy that finishes more also changes which problems land in each cell. And
"the gain is on problems `rft` truncated" does not by itself say `rft` would have failed them with
more tokens; the Phase 3 cap experiment (EXP-010c, `phase-3.md` section 6) does: at an 8,192-token
cap `rft`'s truncated generations still do not finish, they loop, and its accuracy moves by under
a point. So the problems GRPO now gets right are ones `rft` was looping on, and GRPO's policy
change is to stop looping and answer; among problems both policies answer, GRPO is no more
accurate than `rft` on MATH-500 and about a point more accurate on GSM8K, inside the CI.

**What a truncated completion is worth** (`phase-4-truncated-accuracy.md`, computed 22 Oct 2026
after a wording error was found in the learning session). A completion cut at the window is not
scored as wrong: the extractor falls back to a final phrase, the last number or a `\boxed{}`
inside the loop, and only an extraction failure is scored wrong. First samples, greedy, 4,096
window:

| run | GSM8K truncated | right when truncated / finished | MATH-500 truncated | right when truncated / finished |
|---|---|---|---|---|
| `sft` think-tag | 28.1 % | 31.1 % / 91.9 % | 52.0 % | 13.5 % / 90.8 % |
| `rft` think-tag | 15.2 % | 25.0 % / 87.7 % | 39.6 % | 11.6 % / 85.8 % |
| `grpo` think-tag | 4.6 % | 9.8 % / 85.8 % | 20.8 % | 4.8 % / 78.8 % |
| `sft` zero-shot | 17.3 % | 67.1 % / 85.3 % | 59.2 % | 58.8 % / 69.6 % |
| `grpo` zero-shot | 16.0 % | 73.0 % / 86.5 % | 65.0 % | 62.8 % / 67.4 % |

So in the trained format a truncated completion is right one time in three on GSM8K and one in
eight on MATH-500 (the GSM8K ones mostly through a final phrase inside the loop; 68 % of the
MATH-500 ones yield nothing), against nine in ten when finished; under zero-shot the box usually
precedes the loop and a truncated completion is right six times in ten. This is why the
termination gain is worth what the decomposition says and not the whole truncation rate: the
problems `rft` truncated were not all lost, only two thirds to seven eighths of them. Only the GRPO
training reward scores a cut completion 0.

The same split applied to Phase 3's RFT step (`phase-3-gain-decomposition.md`, rft - sft): GSM8K
+6.9 [5.2, 8.6] on `sft`-truncated problems and -3.6 on `sft`-finished, MATH-500 +12.2 [8.8, 15.6]
and -6.4, both-finished paired diff -0.8 and -2.5 (CIs covering 0). Both post-SFT stages at this
scale are termination gains in the trained format; what pass@k adds (section 6) is that the
sampled policy also reaches a few more MATH-500 problems than `rft` did in 32 tries, the one
signal that is not termination.

## 5. EXP-013: the seed repeat, the noise row

Seed 1 minus seed 0, paired on the full sets (`phase-4-compare-seed1-vs-main.md`), and seed 1
against `rft` (`phase-4-compare-think_tag-seed1.md`):

| eval | seed 0 (main) | seed 1 | seed 1 - seed 0 [paired CI] | seed 1 - rft [paired CI] | seed 0 - rft |
|---|---|---|---|---|---|
| gsm8k | 82.3 | 82.0 | -0.3 [-2.3, 1.6] | +3.8 [1.6, 6.0] | +4.1 |
| gsm8k_perturbed | 73.9 | 73.6 | -0.2 [-2.5, 2.2] | +4.6 [1.9, 7.2] | +4.8 |
| math500 | 63.4 | 58.6 | **-4.8 [-9.0, -0.8]** | +2.2 [-2.0, 6.6] | +7.0 |
| countdown | 14.0 | 14.6 | +0.6 [-3.2, 4.4] | +7.6 [4.2, 11.0] | +7.0 |
| aime24 | 3.3 | 0.0 | -3.3 [-10.0, 0.0] | 0.0 | +3.3 |

The hypothesis was a seed-to-seed difference under 2 points on GSM8K and under 3 on MATH-500. GSM8K
and its perturbed set move by 0.3, Countdown by 0.6; MATH-500 moves by 4.8 points with a CI that
excludes 0, which fails the hypothesis and is the number every MATH-500 ablation delta is read
against. The falsifier (a seed difference larger than the gain over `rft` on the same set) did not
fire by the letter on MATH-500 (4.8 against seed 0's 7.0), but seed 1's own gain there is +2.2 with
a CI covering 0, so the honest statement is: GRPO's MATH-500 gain over `rft` is positive in both
seeds and its size is between 2 and 7 points, not resolvable with one seed per arm. By level the
seeds differ most at level 5 (-9.7 [-18.7, -1.5]) and level 3 (-5.7); seed 1's truncation on
MATH-500 is 26.0 % against seed 0's 20.8 %, which accounts for part of it. The dev curves of the two
seeds cross within their step-to-step noise, as hypothesised, and their training statistics are
alike (section 7): the seed difference is in which 500 test problems the final policy gets right,
not in the run's behaviour.

**The noise row for session 2:** an ablation delta smaller than 0.3 (GSM8K), 0.2 (perturbed), 4.8
(MATH-500), 0.6 (Countdown) is "within noise" whatever its sign. The GSM8K rows are tight enough that
any ablation effect of a point will be visible; MATH-500 will separate only effects above 5 points.

## 6. Sharpening or capability: pass@k in the trained format (`phase-4-compare-think_tag_n32_t06.md`)

n = 32 samples at T = 0.6, think-tag, the pinned 300-problem subsets (hashes checked before
sampling); paired against the `rft` and `sft` samples of Phase 3 (fetched from the Hub):

| eval | k | `rft` | grpo | grpo - rft [paired CI] | `sft` | grpo - sft |
|---|---|---|---|---|---|---|
| gsm8k | 1 | 82.6 | 83.9 | +1.3 [0.4, 2.1] | 80.3 | +3.6 [2.7, 4.6] |
| gsm8k | 4 | 95.0 | 94.8 | -0.2 [-0.9, 0.6] | 94.1 | +0.8 [-0.1, 1.6] |
| gsm8k | 16 | 98.3 | 98.0 | -0.3 [-0.9, 0.3] | 97.8 | +0.3 [-0.6, 1.2] |
| math500 | 1 | 61.6 | 64.7 | +3.1 [2.1, 4.1] | 58.8 | +5.9 [4.7, 7.1] |
| math500 | 4 | 80.1 | 82.0 | +2.0 [0.9, 3.1] | 78.0 | +4.0 [2.7, 5.3] |
| math500 | 16 | 88.7 | 90.6 | +1.9 [0.4, 3.5] | 86.2 | +4.4 [2.5, 6.4] |

Solved at least once in 32 tries, `rft` / grpo: GSM8K both 296, only `rft` 0, only grpo 0, neither
4; MATH-500 both 270, only `rft` 2, only grpo 10, neither 18. Against `sft`: MATH-500 only `sft` 3,
only grpo 17. Figure: `figures/phase-4-pass-at-k-grpo.png` (maj@k and the greedy reference line).

Reading. On GSM8K the curves meet by k = 4: GRPO moved probability mass onto answers `rft` could
already produce, and no problem entered the solvable set. On MATH-500 the curves do not meet: +1.9
at k = 16 with the CI excluding 0, and ten problems that `rft` never solved in 32 tries that GRPO
solves, against two the other way. The persistent gain is the same size RFT had over `sft` at k = 32
in Phase 3 (+2.0-2.5) and is again small next to the pass@1 gain, so the answer to the project's
central question at this scale is the same after RL as after selection: mostly sharpening, with a
small, CI-excluding-zero widening of what the policy can reach on MATH-500 in its own format. The
sampled pass@1 at T = 0.6 (83.9 and 64.7) tracks greedy (82.3 and 63.4): the checkpoint's gain is
not a greedy artefact. Truncation at T = 0.6 is 1.4 % on GSM8K and 12.5 % on MATH-500 (`rft` in
Phase 3: 2-3 % and 15-18 %).

## 7. Training dynamics and the hacking watch (`figures/phase-4-grpo-curves.png`)

Centred nine-step window means, main (seed 0) and seed 1 (`figures/phase-4-grpo-curves.md`):

| statistic | step 1 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|
| reward, main / seed 1 | 0.354 / 0.352 | 0.548 / 0.602 | 0.605 / 0.621 | 0.627 / 0.633 | 0.625 / 0.574 |
| finished rate | 0.370 / 0.387 | 0.593 / 0.656 | 0.658 / 0.680 | 0.685 / 0.693 | 0.701 / 0.637 |
| mean tokens | 1,393 / 1,384 | 1,264 / 1,212 | 1,220 / 1,207 | 1,216 / 1,203 | 1,197 / 1,254 |
| KL to `rft` (k3) | 0.0008 / 0.0008 | 0.0037 / 0.0040 | 0.0049 / 0.0051 | 0.0051 / 0.0052 | 0.0061 / 0.0053 |
| zero-variance groups | 0.325 / 0.263 | 0.264 / 0.292 | 0.323 / 0.309 | 0.260 / 0.288 | 0.294 / 0.306 |

Both runs do the same thing: the reward rises from 0.35 to about 0.6 in the first 30 steps, mostly
through the finished rate (0.35 to 0.65) while the mean length falls from 1,400 to 1,200 tokens
(p90 at the 1,536 cap throughout), then drifts up slowly to 0.62-0.63 by step 150 as the correct
rate among finished completions climbs from 0.52 to 0.79; the KL to the reference grows to 0.005-
0.006, the entropy falls from 0.51 to 0.43, the gradient norm sits at 0.13-0.25 and the PPO clip
region is 0.0 at every step (one optimiser step per batch: the ratio is 1 on-policy and the clip is
inert, as the plan says). Zero-variance groups averaged 30.0 % in both runs with no trend: the
learnable-prompt filter, made for `sft`, leaves about five informative groups in eight for the
`rft`-initialised policy at every point of training, which is the quantity EXP-014b's dynamic
sampling acts on. The two runs part only in the last quarter, where seed 1's reward and finished
rate dip (0.57 and 0.64 at step 200 against 0.63 and 0.70) and its length rises; its MATH-500
truncation at eval is 5 points higher than seed 0's, which is the visible side of the 4.8-point
seed difference.

Dev curves (greedy think-tag, 250 + 250 training-split problems, CIs of 4 and 6 points): gsm8k_dev
86.0 at `rft`, then 86.8 / 84.4 / 88.0 / 87.6 (main) and 87.2 / 84.4 / 84.0 / 86.0 (seed 1) at steps
50 / 100 / 150 / 200; math_dev 62.4, then 66.8 / 63.2 / 64.4 / 65.2 and 68.0 / 66.0 / 67.6 / 68.8.
Every point is inside the CI of the `rft` value; the one clear dev signal is MATH-dev truncation
at 4,096 tokens falling from 33.2 % to 16.4 % (main) and 18.0 % (seed 1) by step 50 and staying
there. The checkpoint rule (best dev) would have picked step 150 on GSM8K-dev and step 50 on
MATH-dev; the pre-registered final checkpoint is what was evaluated, and this dev split is too
small to pick checkpoints with.

**Hacking watch.** Of the five flags (length growth without accuracy, format-only outputs,
repeated answers, Wait spam, truncation rising) only `repeated_answers` was raised, in every run,
because the rate of groups agreeing on one answer rose with accuracy (0.375 to 0.625 in the main
run, 0.188 to 0.531 in seed 1) and the flag had no correctness term. The flag now counts groups
agreeing on one *wrong* answer (fb856a8); re-read on the session logs it would not have fired. The
other four signals were false throughout: length fell 15 % while the correct rate among finished
completions rose, format-only outputs stayed under 2 %, "Wait" per completion fell from 5.9 to
3.6, and truncation fell. Verdict: no reward hacking in either run.

## 8. Deviations and decisions, in one place

1. **Micro-batch 16, vLLM share 0.30, expandable segments** for every GRPO run (section 2), the
   user's decision at 15:23 UTC on 20 Oct after two OOMs at the pre-registered 32. Same update,
   same optimiser step; the config and the preflight now pin the values. Session 2 runs the same.
2. **The environment build stalled for 87 minutes** (the PyTorch index serves triton from a host this
   node reached at under 1 MB/s; uv never timed out). Fixed on the node with curl-fetched,
   sha256-checked torch and triton wheels; those wheels and the flash-attn wheel are on the Hub
   (`wheels/` of the samples dataset) and `setup_env.sh` installs them first when present. 88 GPU-
   idle minutes, INR 555.
3. **EXP-010g has no power** (section 3): 6.9 % incorrect traces by construction. Recorded as
   inconclusive; a matched-incorrect set is a debt.
4. **The step order and the watcher.** Before the session a bug in the session script printed an
   `EXIT=` line after training and before the dev evals; the user noticed, the fix landed (a8b9f69)
   and the node pulled it before `smoke`, so every run of record is on the fixed commit.
5. **`rlvr-build-rft` rewrote the tracked Phase 3 data report** (its date line) on the node, which
   made the harness refuse the next step; the node restored the file and relaunched (3 idle
   minutes). The builder now writes its report next to the data (fb856a8).
6. **The paired tables and the termination decomposition were computed on the laptop**, not by the node's `report` step: the `rft` eval
   folders were not on the new disk (they are in the Phase 3 archive), and the n = 32 sample files
   of `rft` and `sft` were fetched from the Hub dataset. The same script, the same per-sample files.
7. **Falsifier 2's format clause** was under-specified (section 4.2) and is read by its intent.
8. **The Countdown hypothesis** ("within its CI") was wrong (section 4.2): the task doubled through
   termination. Reported as a hypothesis miss, not re-explained away.
9. **The seed-repeat hypothesis** (under 3 points on MATH-500) failed at 4.8 points; the MATH-500
   gain is reported as "positive, size uncertain" and the ablations will be read against 4.8.
10. **The dev split** (250 + 250) is too small for its intended plateau and checkpoint-selection
    roles; both are reported as not decidable rather than decided.

## 9. Compute (session 1)

Node-hours from the first tmux marker (12:14 UTC, 20 Oct) to the user's Stop (01:30 UTC, 07:00 IST,
21 Oct): 13.3 h, INR 5,030, against the plan's 12.0 h / INR 4,570. GPU work 13:52-23:47 UTC
(9 h 55 min) with about 17 idle minutes inside the steps; the environment build to the first GPU
job 1 h 28 min (all idle); uploads, hand-back and archive 13 minutes; idle until the Stop
1 h 30 min. Per step: setup 14 min, smoke 2, the two Phase 3 ablations 53, debugging run 62
(including two OOM attempts and the question), main 3 h 18, seed 3 h 18, evals 56 (the n = 32
sampling 28 min, a five-set greedy eval 7.5-8 min, the three-template set 19 min). Rows in
`COMPUTE.md`. Rates for session 2: 56 s per GRPO step at micro-batch 16 on the MATH-heavy mix, 3 h
18 min per 200-step run with four dev evals, 8 min per five-set greedy eval, path A restart 5
minutes. Project to date about INR 16,600 of the INR 22,000 ceiling.

## 10. Session 2: the three ablations as they were made

Node started by the user about 03:25 UTC on 21 Oct (path A, no build); `git pull --ff-only` to 8e94fb5;
every step launched by an anchored EXIT watcher within 60 s of the previous step's `EXIT=0`; no
preflight failed, no run was relaunched, no OOM, nothing dropped (`../h100_p4_session2_addendum/
outputs/h100_p4_handback.md`, S2.1-S2.12; log rows 49-81). Each run is EXP-012's config plus its one
pre-registered switch, seed 0, micro-batch 16, vLLM 0.30.

| run | switch | commit | steps | wall-clock | s/step | peak GB | W&B |
|---|---|---|---|---|---|---|---|
| `grpo_1p7b_a1_nokl` (EXP-014a) | `kl_beta=0` | 8e94fb5 | 200 | 2 h 24 min | 40.3 | 63.9 | qaqbv90s |
| `grpo_1p7b_a2_dynamic` (EXP-014b) | `dynamic_sampling=true`, `epsilon_high=0.28` (inert) | 8e94fb5 | 200 | 4 h 55 min | 85.3 | 70.4 | m5kzgl0x |
| `grpo_1p7b_cap2048` (EXP-015) | `max_new_tokens=2048` | 8e94fb5 | 200 | 4 h 10 min | 71.8 | 73.4 | gvxhkztz |

a1 ran 15 s/step faster than the main run because TRL 1.15 does not instantiate the reference model at
`kl_beta=0` (section 16); a2's step is 1.5x the main run's because 64 groups are generated for the 32
kept; the cap run's 1.28x matched the 73 s projection. Evals: three five-set greedy think-tag evals
(7-9 min each) and the cap run's n = 32, T = 0.6 sampling on the pinned subsets (36 min; hashes
re-checked before it). The `report` step wrote the compares against the main run, the cap sampling
compares and the cap-vs-main decomposition on the node; the compares against `rft` (its eval folders
were not on that disk) and the a1/a2/cap decompositions against `rft`, plus a1 against the main run,
were computed on the laptop from the archive (`phase-4-compare-{a1,a2,cap}-vs-rft.md`,
`phase-4-gain-decomposition-{a1,a2,cap}-vs-rft.md`, `phase-4-gain-decomposition-a1-vs-main.md`).

## 11. The ablation table

**Finals, greedy think-tag, full sets** (`outputs/eval/grpo_1p7b_{a1_nokl,a2_dynamic,cap2048}_think_tag/
metrics.json`; main and seed 1 from session 1):

| set | n | main (EXP-012) | seed 1 (EXP-013) | a1 KL 0 | a2 dynamic | cap 2,048 |
|---|---|---|---|---|---|---|
| gsm8k | 1,319 | 82.3 [80.2, 84.2] | 82.0 | 84.5 [82.5, 86.6] | 84.0 [82.0, 86.1] | 82.5 [80.4, 84.5] |
| gsm8k_perturbed | 830 | 73.9 [70.7, 77.0] | 73.6 | 77.1 [74.2, 80.1] | 74.8 [71.8, 78.0] | 72.3 [69.4, 75.4] |
| math500 | 500 | 63.4 [59.4, 67.6] | 58.6 | 69.0 [65.0, 73.0] | 61.4 [57.2, 65.6] | 60.8 [56.6, 64.8] |
| countdown | 500 | 14.0 [11.0, 17.0] | 14.6 | 14.8 [11.6, 18.2] | 12.0 [9.2, 15.0] | 12.4 [9.6, 15.2] |
| aime24 | 30 | 3.3 [0.0, 10.0] | 0.0 | 6.7 [0.0, 16.7] | 3.3 [0.0, 10.0] | 6.7 [0.0, 16.7] |
| truncation gsm8k / math500 | | 4.6 / 20.8 % | 6.7 / 26.0 % | **2.1 / 13.4 %** | 3.8 / 22.4 % | 9.6 / 26.2 % |
| mean tokens gsm8k / math500 | | 999 / 1,944 | 1,066 / 2,118 | **490 / 1,248** | 938 / 1,940 | 1,241 / 2,199 |

**Paired against the main run** (`phase-4-compare-{a1,a2,cap}-vs-main.md`), read against the EXP-013
noise row (0.3 / 0.2 / 4.8 / 0.6):

| set | a1 - main | a2 - main | cap - main |
|---|---|---|---|
| gsm8k | **+2.2 [0.4, 4.0]** | +1.7 [-0.1, 3.6] | +0.2 [-1.7, 2.2] |
| gsm8k_perturbed | **+3.3 [1.1, 5.5]** | +1.0 [-1.4, 3.4] | -1.6 [-4.1, 1.1] |
| math500 | **+5.6 [1.6, 9.4]** (above the noise row) | -2.0 [-6.0, 2.0] (inside it) | -2.6 [-6.4, 1.2] (inside it) |
| countdown | +0.8 [-3.0, 4.6] | -2.0 [-5.6, 1.2] | -1.6 [-5.2, 2.2] |
| aime24 | +3.3 [0.0, 10.0] | 0.0 | +3.3 [0.0, 10.0] |

**Paired against `rft`** (`phase-4-compare-{a1,a2,cap}-vs-rft.md`; main's row from section 4 for
reference):

| set | main - rft | a1 - rft | a2 - rft | cap - rft |
|---|---|---|---|---|
| gsm8k | +4.1 [2.0, 6.1] | **+6.3 [4.2, 8.3]** | +5.8 [3.8, 8.0] | +4.3 [2.1, 6.6] |
| gsm8k_perturbed | +4.8 [1.9, 7.6] | **+8.1 [5.5, 10.6]** | +5.8 [3.0, 8.4] | +3.3 [0.5, 6.0] |
| math500 | +7.0 [2.8, 11.0] | **+12.6 [8.0, 16.8]** | +5.0 [0.8, 9.4] | +4.4 [0.2, 8.6] |
| countdown | +7.0 [3.6, 10.4] | +7.8 [4.2, 11.6] | +5.0 [1.6, 8.6] | +5.4 [2.2, 8.8] |

**The decompositions** (`scripts/decompose_gain.py`, as section 4.5; reference truncation defines the
split; both-finished is the paired accuracy difference on problems both policies answered):

| comparison | set | net | on problems the reference truncated | on problems it finished | both finished (n) | paired diff there |
|---|---|---|---|---|---|---|
| a1 - main | gsm8k | +2.2 | +2.0 [1.3, 2.7] | +0.2 [-1.5, 2.0] | 1,241 | +1.0 [-0.7, 2.7] |
| a1 - main | gsm8k_perturbed | +3.3 | +1.8 [0.8, 2.8] | +1.4 [-0.6, 3.6] | 764 | **+2.2 [0.1, 4.3]** |
| a1 - main | math500 | +5.6 | +5.4 [3.4, 7.6] | +0.2 [-3.0, 3.2] | 372 | +2.7 [-1.3, 6.5] |
| a1 - main | countdown | +0.8 | +8.4 [6.0, 11.0] | -7.6 [-10.4, -4.8] | 46 | +15.2 [6.5, 26.1] |
| a1 - rft | gsm8k | +6.3 | +5.5 [4.2, 6.9] | +0.8 [-0.8, 2.4] | 1,103 | +1.4 [-0.5, 3.4] |
| a1 - rft | gsm8k_perturbed | +8.1 | +5.5 [3.9, 7.3] | +2.5 [0.6, 4.5] | 682 | **+3.5 [1.3, 5.7]** |
| a1 - rft | math500 | +12.6 | +12.8 [10.0, 16.0] | -0.2 [-3.2, 2.8] | 291 | +2.4 [-2.1, 7.2] |
| a2 - rft | gsm8k | +5.8 | +5.3 [3.9, 6.7] | +0.5 [-1.0, 2.0] | 1,093 | +1.6 [-0.1, 3.3] |
| a2 - rft | math500 | +5.0 | +8.8 [5.8, 11.8] | -3.8 [-6.8, -0.8] | 277 | -0.7 [-5.8, 4.0] |
| cap - rft | gsm8k | +4.3 | +5.9 [4.5, 7.3] | -1.6 [-3.4, 0.2] | 1,047 | +0.6 [-1.3, 2.5] |
| cap - rft | gsm8k_perturbed | +3.3 | +4.0 [2.4, 5.7] | -0.7 [-3.0, 1.4] | 641 | +2.5 [0.2, 4.8] |
| cap - rft | math500 | +4.4 | +9.0 [5.8, 12.2] | -4.6 [-7.4, -1.8] | 268 | 0.0 [-4.1, 4.1] |
| cap - main | gsm8k | +0.2 | +2.4 [1.6, 3.3] | -2.1 [-3.9, -0.4] | 1,149 | +1.1 [-0.6, 2.9] |
| cap - main | math500 | -2.6 | +3.2 [1.6, 4.8] | -5.8 [-9.4, -2.2] | 330 | +1.2 [-3.0, 5.2] |

Every session 2 checkpoint's gain over `rft` on MATH-500 sits on problems `rft` truncated, as the main
run's did; on problems both finished no MATH-500 difference has a CI clear of zero. The perturbed
GSM8K both-finished cells of a1 (+2.2 against the main run, +3.5 against `rft`, noise row 0.2) and the
cap run (+2.5 against `rft`) are the only non-termination cells with CIs clear of zero in the whole
phase; they are small, single-seed, and one of fifteen rows in this table, and are reported as a signal
to test, not a result.

Dev curves (greedy think-tag, gsm8k_dev 250 + math_dev 250; `rft` 86.0 / 62.4; half-widths about 5):

| checkpoint | gsm8k_dev ck50 / 100 / 150 / 200 | math_dev ck50 / 100 / 150 / 200 | math truncation at ck200 | math mean tokens at ck200 |
|---|---|---|---|---|
| main (EXP-012) | 86.8 / 84.4 / 88.0 / 87.6 | 66.8 / 63.2 / 64.4 / 65.2 | 16.4 % | 1,875 |
| seed 1 (EXP-013) | 87.2 / 84.4 / 84.0 / 86.0 | 68.0 / 66.0 / 67.6 / 68.8 | 18.0 % | 1,986 |
| a1 KL 0 | 88.0 / 89.2 / 88.4 / 88.0 | 67.6 / 67.6 / 74.4 / 74.0 | 10.4 % | 1,136 |
| a2 dynamic | 84.8 / 87.6 / 87.2 / 88.8 | 63.6 / 67.6 / 64.4 / 67.6 | 15.6 % | 1,865 |
| cap 2,048 | 81.2 / 86.4 / 86.4 / 84.0 | 65.2 / 65.6 / 63.6 / 60.8 | 24.0 % | 2,225 |

## 12. EXP-014a: the reference model off

**Pre-registered.** Within the noise row of the main run; KL to `rft` (logged, not penalised) 2-5x
the main run's; length and truncation within 10 % of the main run's. Falsifier: worse on MATH-500 by
more than the noise row with the KL diverging. The parenthesis in the falsifier ("better by more than
the noise row: the penalty costs accuracy here, also reported") is the case that occurred.

**What happened.** The hypothesis failed on every clause. a1 is above the main run on the GSM8K family
and on MATH-500 with CIs clear of zero and the MATH-500 delta (5.6) above the noise row (4.8); its
completions are half the length (490 against 999 tokens on GSM8K, 1,248 against 1,944 on MATH-500) and
its truncation a third (2.1 / 13.4 % against 4.6 / 20.8 %). In training (section 15) the finished rate
rose to 0.96 against the main run's 0.75, the mean completion fell to 611 tokens against 1,197, the
p90 left the cap at step 100 (the only run of the phase where it did), and zero-variance groups rose
to half (more all-correct groups). No KL was logged: TRL 1.15 does not run the reference forward at
beta 0 (section 16). **Offline KL against `rft`** (the node's `outputs/kl_offline.py` on 512 logged
completions from steps 191-200 of each run, re-tokenised, both models in bf16; k3 per token with a
bootstrap CI over completions; `outputs/reports/phase-4-kl-offline.json`): a1 0.0977 [0.0905,
0.1074], main 0.0062 [0.0061, 0.0064], cap 0.0043 [0.0042, 0.0046]; per sequence 49.5, 7.4 and 6.3
nats. The estimator reproduces TRL's logged in-training KL where one exists (main 0.0062 at step
200, cap 0.0041), so the scale is right: **a1 is 15x further from `rft` per token and 7x per
sequence**, far beyond the pre-registered 2-5x. The penalty at beta 0.04 held the other runs near
0.006 nats per token.

**Decomposed** (section 11), the gain over the main run is the termination pattern a third time: +5.4
of +5.6 on MATH-500 sits on problems the main run truncated and a1 finished; the both-finished paired
difference is +2.7 [-1.3, 6.5] on MATH-500 and +1.0 [-0.7, 2.7] on GSM8K. Against `rft` the picture is
the main run's at larger scale: +12.8 of +12.6 on `rft`-truncated MATH-500 problems, both-finished
+2.4 [-2.1, 7.2]. The exception is perturbed GSM8K, where the both-finished difference is +2.2 [0.1,
4.3] against the main run and +3.5 [1.3, 5.7] against `rft` (noise row 0.2). MATH-500 by level against
`rft`: +16.3, +12.2, +22.9 on levels 1-3, +8.6 [0.8, 17.2] on level 4 (the main run: 0.0 [-8.6, 9.4]),
+7.5 [-1.5, 16.4] on level 5.

**What the completions look like.** From the completion logs of the last ten steps (2,560 training
completions): mean 1,718 characters (p10 985, p90 2,802) against the main run's 3,450 (p10 2,317);
none under 400 characters in either run. The policy did not collapse to an empty think block; it
writes shorter derivations and stops. The format-only rate (tags present, nothing correct) ran
0.09-0.16 from step 50 against the main run's 0.03-0.07, below the 0.2 flag.

**Reading.** At beta 0.04 the KL term is a brake on the one thing the reward pays for at this scale,
stopping. Without it the policy shortens faster and finishes more, and the eval gain over the main run
is the extra finishing. The reference model is not needed for stability at 200 steps and this
learning rate: no hacking flag, entropy 0.31 at the end (main 0.43), the zero-variance groups rose
because more groups were all correct, not all wrong. **Conclusion:** the pre-registered hypothesis
("the penalty is a safety margin, not a source of accuracy") was wrong in sign: the penalty cost
accuracy in the trained format by holding length up, and the cost was termination. One seed; the
MATH-500 delta is one noise row above zero; the perturbed GSM8K both-finished cell is the candidate
non-termination effect and needs a seed before it is called one.

**Outside the trained format, nothing.** a1's zero-shot greedy eval (addendum 2;
`outputs/reports/phase-4-compare-a1-vs-main-zero_shot.md`, paired against the main run's zero-shot
folder on the full sets): GSM8K -1.5 [-3.3, 0.3] (82.8 against 84.3), perturbed +0.4 [-1.8, 2.4],
MATH-500 +1.8 [-2.0, 5.4] (66.2 against 64.4), AIME24 -6.7 [-16.7, 0.0]; every CI covers zero. The
brevity carries (truncation 5.6 / 23.2 % against the main run's 16.0 / 65.0 %, completions 40 %
shorter), the accuracy does not. Dropping the reference term let the policy drift 15x further and get
much shorter, which bought a trained-format gain and no template-independent one: the same shape as
the main run's gain over `rft` in session 1 (+3.2 / -0.4 under zero-shot, section 4.3).

## 13. EXP-014b: dynamic sampling

**Pre-registered.** Zero-variance fraction 20-40 % and rising; dynamic sampling +1-3 on MATH-500 at
1.3-1.6x the sampling time. Falsifier: within the noise row on both sets.

**What happened.** The falsifier fired. a2 is inside the noise row and inside its CI on every set
(+1.7 [-0.1, 3.6] GSM8K, -2.0 [-6.0, 2.0] MATH-500 against the main run); against `rft` it reproduces
the main run's gains (+5.8 / +5.0) with the same decomposition (MATH-500 +8.8 on `rft`-truncated
problems, both-finished -0.7 [-5.8, 4.0]). The mechanism did run: 64 groups generated per step, a mean
43.5 informative (fraction 0.68, min 0.48 at step 195), and **one step in 200 (step 195) filled one
group with zero advantage**; the main run's own zero-variance fraction was 30 % and flat, not rising.
Step time 85.3 s against 56.3 (1.5x, inside the pre-registered 1.3-1.6x). Clip-high 0.28 was inert at
every step (`clip_ratio/high_mean` 0), as pre-registered. The logged `zero_variance_group_rate` of
this run is over the 64 generated groups (0 over the kept 32 by construction; section 16).

**Conclusion.** With the learnable-prompt filter (solve rate in (0, 1) for the SFT policy) the batch
already carries 70 % informative groups and 32 of them in 199 of 200 steps; dropping the rest and
refilling buys nothing measurable here at 1.5x the cost. The DAPO fix addresses a batch the filter
had already fixed. What it does not test: a harder prompt pool, where the main run's 30 % all-correct
groups would become all-wrong ones.

## 14. EXP-015: the 2,048-token cap

**Pre-registered.** Training truncation at step 200 below 15 % (EXP-012: 25 %); mean length above
1,200; net MATH-500 gain over `rft` at least +7.0; the both-finished paired difference over `rft` up by
at least 2 (EXP-012: 0.0 [-4.3, 3.9]) and pass@16 over `rft` up by at least 3 (EXP-012: +1.9 [0.4,
3.5]). Falsifier: both-finished inside EXP-012's CI and pass@16 within 1 point of EXP-012's.

**What happened.** The falsifier fired on both counts. Both-finished against `rft` on MATH-500: 0.0
[-4.1, 4.1] (EXP-012: 0.0 [-4.3, 3.9]); against the main run +1.2 [-3.0, 5.2]. pass@16 on the pinned
MATH-500 subset against `rft`: +0.8 [-0.6, 2.3] (EXP-012: +1.9 [0.4, 3.5]); against the main run -1.2
[-2.7, 0.3]; solved in 32 tries, 300 problems: both 269, only `rft` 3, only cap 6 (EXP-012: 10 and 2).
The net gains over `rft` (+4.3 / +3.3 / +4.4 / +5.4) are below the main run's on every set and the
MATH-500 one (+4.4 [0.2, 8.6]) is below the pre-registered +7.0; the decomposition puts +9.0 of it on
`rft`-truncated problems and -4.6 on `rft`-finished ones. Training (section 15): the finished rate
started 16 points higher than the main run's (the longer cap cuts fewer) and ended 7 higher (0.82),
the mean length fell 1,690 to 1,458 (14 %, the main run's 15 %), the p90 stayed at the 2,048 cap at
every step, the training truncation ended at 18 %, not below 15 %, and the KL to `rft` ended lower
(0.0041 against 0.0062). At eval the cap policy writes longer completions (1,241 / 2,199 tokens against
999 / 1,944) and truncates *more* at the 4,096 window (9.6 / 26.2 % against 4.6 / 20.8 %); its dev
MATH-500 curve ends at 60.8, below `rft`'s 62.4, with 24 % truncation.

**Conclusion.** The cap is not what limits the reasoning gain. With 33 % more room the policy learns
the same thing, stop and answer, more slowly and less completely, and nothing moves on the problems
both policies finish or in the number of problems reachable in 32 samples. The "termination only"
reading of EXP-012 is a property of the method at this scale, data and budget, not of the cut from
2,048 to 1,536 tokens. What the run does not test: a cap long enough that the loops themselves finish
(EXP-010c says 8,192 is not that cap either), or a prompt pool where finishing is not the binding
constraint (section 18).

## 15. Training dynamics and the hacking watch, session 2 (`figures/phase-4-grpo-curves-ablations.png`)

Centred nine-step window means at steps 1, 50, 100, 150, 200 (`figures/phase-4-grpo-curves-ablations.md`;
a2's reward is over its 32 kept groups, which exclude the all-correct groups that raise the main run's
mean, so its level is not comparable):

| run | reward | finished rate | mean tokens | KL to `rft` | zero-variance groups |
|---|---|---|---|---|---|
| main | 0.354 / 0.548 / 0.605 / 0.627 / 0.625 | 0.370 / 0.593 / 0.658 / 0.685 / 0.701 | 1,393 / 1,264 / 1,220 / 1,216 / 1,197 | 0.0008 / 0.0037 / 0.0049 / 0.0051 / 0.0061 | 0.325 / 0.264 / 0.323 / 0.260 / 0.294 |
| a1 KL 0 | 0.352 / 0.731 / 0.804 / 0.859 / 0.869 | 0.366 / 0.857 / 0.908 / 0.945 / 0.965 | 1,396 / 964 / 820 / 674 / 611 | not logged | 0.344 / 0.375 / 0.417 / 0.556 / 0.506 |
| a2 dynamic | 0.398 / 0.658 / 0.652 / 0.682 / 0.695 | 0.419 / 0.720 / 0.727 / 0.742 / 0.769 | 1,367 / 1,159 / 1,166 / 1,153 / 1,141 | 0.0008 / 0.0052 / 0.0062 / 0.0065 / 0.0072 | 0.284 / 0.326 / 0.280 / 0.339 / 0.359 (of 64 generated) |
| cap 2,048 | 0.515 / 0.677 / 0.688 / 0.719 / 0.711 | 0.560 / 0.747 / 0.758 / 0.789 / 0.787 | 1,659 / 1,462 / 1,423 / 1,447 / 1,469 | 0.0008 / 0.0031 / 0.0039 / 0.0040 / 0.0041 | 0.237 / 0.295 / 0.306 / 0.299 / 0.338 |

Entropy at step 200: main 0.427, a1 0.315, a2 0.425, cap 0.451. "Wait" per completion at step 200:
main 3.6, a1 0.65, a2 3.4, cap 4.7. The p90 length is at the cap at every step of every run except
a1, whose p90 falls to 866 by step 200. Correct-of-finished at step 200: main 0.79, a1 0.89, a2 0.77
(kept groups), cap 0.81.

**Hacking watch** (`run_info.json`, with the correctness term of fb856a8): no flag on any of the three
runs. Length growth without accuracy, format-only outputs above 0.2, repeated incorrect answers above
0.5, "Wait" spam, truncation rising: all false; first-to-last window tokens / correct rate: a1 1,298 /
0.65 to 603 / 0.87, a2 1,254 / 0.68 to 1,149 / 0.79, cap 1,531 / 0.72 to 1,451 / 0.79. The
repeated-*incorrect* rate is 0.00-0.03 at every sampled step of every run (the flag that fired on the
session 1 runs was reading repeated *correct* answers). a1's format-only rate (0.09-0.16 from step
50) is the one statistic that moved toward a flag without reaching it (section 12).

## 16. Deviations and incidents, session 2 (the node's S2.10, and what changed on the laptop)

1. **The step-10 watcher for a1 waited on `train_log.jsonl`, which `rlvr-grpo` writes once at the end
   of the run**; the live `step_time` is only in the stdout log. The projection was computed 65 min
   late (0.90x the estimate; the run was never at risk); a2's and the cap run's were on time from the
   stdout log. Laptop: handoff rule 5 now names the stdout log as the live source; a per-step
   `train_log.jsonl` is a debt (engineering notes).
2. **TRL 1.15 does not run the reference model at `kl_beta=0`**: a1 has no logged KL and ran 40 s/step.
   The config comment and the handoff said the KL would be logged; both corrected. The drift was
   estimated offline before the Delete (section 12; the node's `kl_offline.py`, kept in the archive
   with its first run's output, whose bootstrap CI was degenerate and was fixed in a rerun).
3. **a2's logged `zero_variance_group_rate` is over the 64 generated groups**, not the 32 kept; this
   report reads it as "1 minus the informative fraction" and takes the kept-batch fact (0 by
   construction, one fill in 200 steps) from `dynamic_sampling_log.jsonl`.
4. **The ablation finals' uploads were not in `scripts/phase4_rental.sh report`**; the node ran the
   three `hf upload` calls by hand (`ablations/{a1_nokl,a2_dynamic,cap2048}` of
   `Raviteja546/reasoning-rlvr-grpo`, commits af382d1b, 204ebba5, a1ba6694). The step now uploads
   any seed or ablation final it finds.
5. **The GRPO model card the `report` step generated was SFT-shaped** (data hash `None`, epochs `None`,
   "cross-entropy"), and that card went to the Hub (commit 7e6e7acb). `scripts/push_to_hub.py` now
   writes a GRPO section from the run's own `run_info` (prompts, shape, reward, training signal), with
   a test; `cards/grpo.md` is regenerated from the archive; the Hub copy is replaced when the user
   next pushes with their token (`hf upload Raviteja546/reasoning-rlvr-grpo cards/grpo.md README.md`).
6. **Micro-batch:** the user asked mid-run whether 24 would fit; 24 is not valid (256 completions per
   step must be a multiple of the micro-batch), 32 did not fit on 20 Oct, so 16 stays (plan section 6).
7. **Disk at 96 %** at the end of the session (six 33 GB run folders); a seventh run would not have fit.
   The notebook is to be Deleted after the laptop has the archive (it has).
8. The cap run's n = 32 sampling took 36 min against the 28 estimated (longer samples); the compares
   against `rft` were skipped on the node (folders absent) and computed here, as the handoff said.
9. **Two additions after the archive, at the user's request (16:33-16:48 UTC):** the offline KL of the
   a1, main and cap finals against `rft` and a zero-shot greedy eval of the a1 final (4.4 min), about
   8 GPU-minutes and INR 95, delivered as `h100_p4_session2_addendum2.tar.gz` (sha256 38fae1df...).
   Then the user Deleted the notebook.

## 17. Compute, session 2 and Phase 4

| | plan | actual |
|---|---|---|
| GPU work, `setup` launch to `report` EXIT | 12.65 h | 12 h 29 min (03:30:44 to 16:00:07 UTC), under 5 min idle between steps |
| the three GRPO runs | 11.6 h | 11 h 29 min (a1 2.40, a2 4.91, cap 4.18) |
| evals and report | 1.25 h | 1.0 h |
| tail: uploads, hand-back, archive | 0.33 h | about 0.4 h (to about 16:25) |
| node-hours, Start (about 03:25) to the archive (about 16:25) | 13.4 h / INR 5,060 | **about 13.0 h / INR 4,930** |

The drop order (plan section 8) was not needed. Phase 4 in all: session 1 13.27 h (INR 5,030) plus
session 2 about 13.0 h (INR 4,930) = 26.3 node-hours, INR 9,960, plus the notebook's disk (about
INR 107 a day from 20 Oct to the Delete), the two additions of addendum 2 (about INR 95) and the idle
time from 16:48 UTC to the user's Delete on 21 Oct. Project to date about INR 21,700 at the Delete,
against the INR 22,000 ceiling (`COMPUTE.md`).

## 18. What Phase 4 supports, and what it does not

**Supported by the evidence.**
- GRPO with a finished-and-correct reward from `rft` beats `rft` paired on every set with 500 or more
  problems, in the trained format (section 4); the seed repeat reproduces the GSM8K-family and Countdown
  gains within a point and the MATH-500 gain with a 4.8-point spread (section 5).
- The gain in the trained format is a termination gain. Four independent reads agree: the per-problem
  decomposition of the main run, the seed, a1, a2 and the cap run (every MATH-500 both-finished cell
  inside its CI); EXP-010c (the loops do not finish at 8,192 tokens); the cap test (33 % more room
  changes nothing on finished problems or in pass@16); and the KL-off run (removing the brake on
  shortening buys more of the same gain). Phase 3's RFT step decomposes the same way.
- Outside the trained format the pipeline's gain over the base model's best prompt is +6 on GSM8K and
  +11 on MATH-500 (SFT most of it; sections 4.1, 4.3).
- The method is stable at this learning rate with or without the reference model: KL to `rft` under
  0.01, no clipping, no hacking flag with the corrected watch, entropy not collapsing (0.31-0.45).
- Dynamic sampling is unnecessary when the prompt pool is already filtered by solve rate.
- pass@k: GSM8K pure sharpening in every run; MATH-500 a small widening in the main run (+1.9 [0.4,
  3.5] at k = 16) that the cap run does not reproduce (+0.8 [-0.6, 2.3]); the two are inside each
  other's CIs, so the honest statement is "at most a small widening".

**Not supported, or not tested.**
- The cap hypothesis (EXP-015) and the dynamic-sampling hypothesis (EXP-014b) failed their
  pre-registered tests; the KL hypothesis (EXP-014a) failed in sign.
- Any claim about reasoning gains from GRPO at this scale: the only both-finished cells with CIs clear
  of zero are on perturbed GSM8K (a1 and the cap run, +2.2 to +3.5), single-seed, one row in fifteen.
- The loss normalisation (EXP-014c, Dr. GRPO) was not run; the selection-versus-self-distillation
  question of RFT (EXP-010g) still has no powered test.
- Removing the KL term buys nothing outside the trained format (a1's zero-shot row, section 12);
  no on-policy stage after SFT has moved the best-prompt accuracy beyond the noise (RFT -1.1 / +1.6,
  GRPO +3.2 / -0.4, a1 -1.5 / +1.8 on GSM8K / MATH-500).
- Whether the termination-only result is about the *prompts* rather than the method: the pool is
  MATH levels 1-3 plus GSM8K, the eval gains sit on levels 1-3, and 30-50 % of groups were all-correct.
  A run on level 3-5 prompts is the test, and it is the first candidate for the next plan
  (`docs/research-narrative.md`, "Open now").

The chain of questions this phase answered, and the ones it raised, are in `docs/research-narrative.md`;
the engineering record in `docs/phase-4-engineering-notes.md`; the chronology in `docs/phase-4-journal.md`.

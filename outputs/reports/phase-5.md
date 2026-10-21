# Phase 5 report: the cost of termination

Branch `phase-5-ttc` (from `main` 648df74, the Phase 4 merge). Option A of `docs/phase-5-plan.md`
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

## 2. EXP-016: budget forcing (pending; runs on the laptop from `scripts/phase5_laptop.sh`)

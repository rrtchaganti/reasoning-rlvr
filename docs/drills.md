# From-scratch drills

`PLAN.md` section 10.2's list: forty-five minutes each, no libraries beyond PyTorch, the code must run.
Each drill names the reference in this repo to check an attempt against, the test that pins its
behaviour, and the mistake this project actually made or caught in that piece, which is the part
an interviewer is most likely to probe.

| drill | reference | test | status |
|---|---|---|---|
| Unbiased pass@k from (n, c), and why the naive estimate is biased | `src/rlvr/ref/pass_at_k.py` | `tests/test_ref_pass_at_k.py` (brute force, edge cases, the naive estimate biased low) | in the repo |
| GRPO loss: group-normalised advantages, PPO clipping, k3 KL, padded batch with a completion mask | `src/rlvr/ref/grpo_loss.py`, `advantages.py`, `kl.py` | `tests/test_ref_grpo_loss.py`, `tests/test_ref_advantages_kl.py` | in the repo; matched TRL's loss to 1e-9 and gradients to 1.4e-5 in fp32 (`scripts/grpo_correctness_check.py`) |
| Majority vote with answer normalisation | `majority_vote` in `src/rlvr/eval/metrics.py` | `tests/test_eval_metrics.py::test_majority_vote_ties_and_none` | in the repo |
| Greedy and top-p sampling over an HF causal LM with a KV cache | `src/rlvr/ref/sampling.py` | `tests/test_ref_sampling.py` (matches `generate`, matches the no-cache forward, stops at EOS, reproducible) | in the repo; the cache is the model's `past_key_values`, not a hand-written one |
| Causal self-attention forward pass with a mask | none | none | not in the repo: the one drill to write from nothing |
| Bootstrap CI for accuracy, and a paired bootstrap for a difference | `bootstrap_ci`, `paired_bootstrap_diff` in `src/rlvr/eval/metrics.py` | `tests/test_eval_metrics.py` (coverage, paired tighter than unpaired, zero for identical runs) | in the repo |
| Length-aware packing with a block-diagonal attention mask | `src/rlvr/ref/sft_loss.py` | `tests/test_ref_sft_loss.py` (packed with the block mask equals unpacked; a plain causal mask does not) | in the repo |

## What each drill is really testing

**pass@k.** The estimator is 1 - C(n-c, k) / C(n, k), computed as a running product to avoid huge
binomials. The naive "fraction of k-subsets drawn once" is biased low; the test shows it. The
project's use: pass@k at k = 16 and 32 on the n = 32 sample sets separated sharpening from new
solutions (EXP-012: no problem newly solved on GSM8K; 10 on MATH-500).

**GRPO loss.** Three choices hide in one line: the advantage's normalisation (divide by the group
std or not), the aggregation (mean per sequence then over sequences, or over all tokens), and where
the KL enters (here in the loss, k3, against the frozen start). The tests pin each one: per-sequence
aggregation weights a short completion's tokens more (`test_sequence_agg_weights_short_completions_more`),
std normalisation inflates low-variance groups (`test_std_normalisation_inflates_low_variance_groups`),
and with one update per batch the ratio is 1 and the gradient equals REINFORCE with a group-mean
baseline (`test_gradient_equals_reinforce_with_baseline_when_ratio_is_one`). The project's runs used
the published per-sequence loss with group-std scaling (`configs/train/grpo_1p7b.yaml`); a constant
group gives zero signal, which is why 30 % zero-variance groups mattered.

**Majority vote.** Normalise before counting, let extraction failures abstain, break ties
deterministically (first seen). At k = 2 a vote cannot break a tie, which is where the 0.6B verifier
earned its +3 to +6 points (EXP-022).

**Sampling with a cache.** The first forward pass takes the whole prompt and returns the cache; each
later step feeds one token. Greedy decoding with a cache must equal the no-cache forward token for
token; batched decoding on a GPU need not, which is the lesson of EXP-000 (vLLM's default scheduler
moved greedy accuracy by up to 6 points until batch-invariant mode).

**Attention.** Scaled dot product, a causal mask added as -inf (or the dtype's minimum) before the
softmax, never multiplied in after it. The project's version of the mistake is in the packing drill.

**Bootstrap.** Resample problems, not tokens or samples: the problem is the unit of replication. For
two checkpoints on the same problems, bootstrap the per-problem difference; the paired interval is
much tighter than two separate ones, which is what lets a 2-point gain on 500 problems be read at
all.

**Packing.** First-fit into fixed-length rows, positions restarting per sequence, and a
block-diagonal causal mask. The bug this project caught (`FAILURES_AND_LEARNINGS.md`, 15 Oct 2026):
a boolean 4D mask passed to transformers 5.17 was *added* to the attention scores, True and False
becoming +1 and +0 instead of 0 and -inf, so nothing was masked; the packed losses were off by up to
0.04 nats. The test that packed equals unpacked caught it before any training run.

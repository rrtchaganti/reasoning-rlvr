# Quick reference: what goes in, what comes out, and the knobs, per phase

Companion to `PLAN.md`. One screen per phase: a concrete input/output pair, the loss, the optimiser settings, and the few numbers to watch. Values are the planned defaults; the YAML in `configs/` is the source of truth once runs start, and measured values replace the placeholders here.

Written 9 Oct 2026.

---

## 0. One-page hyperparameter table

| Phase | Model in → out | Objective / loss | LR (schedule) | Batch | Epochs / steps | Max len | Other |
|---|---|---|---|---|---|---|---|
| 0 Harness | any checkpoint → JSON metrics | none (inference) | – | vLLM auto | – | 4096 new tokens | T=0 greedy; T=0.6, top-p 0.95, n=64 for pass@k |
| 1 Baselines | Qwen3-1.7B-Base → same | none | – | – | – | 4096 | 0-shot, 4-shot, 4-shot-CoT, think-tag prompts |
| 2 SFT | Base → `sft` | CE on response tokens only | 1e-5 full FT, sweep {5e-6, 1e-5, 2e-5}; cosine, 3 % warmup | 64 sequences effective (packed) | 2 epochs | 4096 | AdamW β=(0.9, 0.95), wd 0.1, clip 1.0, bf16, grad ckpt. LoRA variant: r 64, α 128, LR 2e-4 |
| 3 RFT | `sft` → `rft` | CE on kept correct traces | 5e-6, cosine | 64 | 1 epoch | 4096 | sampling: 8/problem, T=1.0, top-p 1.0; keep ≤ 2 correct per problem |
| 4 GRPO | `rft` → `grpo` | clipped PG + β·KL(k3) | 1e-6 constant, 10 warmup steps | 64 prompts × G=8 = 512 rollouts/step | 500 steps | 2048 new | β 0.04, ε 0.2, μ=1 (one update per batch), T=1.0, AdamW, clip 1.0, vLLM colocate |
| 4 A1 | same | β = 0 | same | same | same | same | drop the reference model |
| 4 A2 | same | ε_low 0.2 / ε_high 0.28, dynamic sampling | same | resample until 64 non-zero-variance groups | same | same | DAPO |
| 4 A3 | same | token-level mean, no length norm, no std norm | same | same | same | same | Dr. GRPO |
| 5 DPO | `sft` → `dpo` | −log σ(β(Δ_chosen − Δ_rejected)) | 5e-7, cosine | 32 pairs | 1 epoch | 4096 | β sweep {0.05, 0.1, 0.3}; ref = `sft`; one on-policy round |
| 6 ORM | Qwen3-0.6B-Base + head → `orm` | BCE on correct/incorrect | 1e-5, cosine | 64 | 1 epoch | 4096 | pairs/labels from Phase 3 samples |
| 6 TTC | `sft`, `grpo` → curves | none | – | – | – | budgets {256, 512, 1024, 2048, 4096} | maj@k, best-of-N, budget forcing with ≤ 2 "Wait" |
| 7 Distil | 0.6B-Base → `distilled-0.6b` | CE on teacher traces | 2e-5 full FT, cosine | 64 | 2 epochs | 4096 | teacher = `grpo` correct traces; then AWQ 4-bit, group 128 |

Compute budget reminder: ≈ 53-55 A100-hours total after the 13 Oct decision of one seed per ablation plus one seed-repeat of the main GRPO run (`COMPUTE.md` is the ledger).

---

## Phase 0: evaluation harness

**Input (what the harness is given):** a problem and a prompt template.

```
# GSM8K test item (as stored)
question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and
          bakes muffins for her friends every day with four. She sells the remainder at the
          farmers' market daily for $2 per fresh duck egg. How much in dollars does she make
          every day at the farmers' market?
answer:   18
```

```
# prompt actually sent to the model (think-tag template, used for every trained checkpoint)
A conversation between User and Assistant. The Assistant first thinks about the reasoning
process in the mind and then provides the User with the answer. The reasoning process is
enclosed within <think> </think> and the answer within <answer> </answer>.
User: Janet's ducks lay 16 eggs per day. ... How much in dollars does she make every day at the farmers' market?
Assistant:
```

**Output (what the model returns, what the harness extracts):**

```
<think>
16 eggs per day. She uses 3 + 4 = 7. Remaining 16 - 7 = 9 eggs. 9 * $2 = $18.
</think>
<answer>18</answer>
```

```
# per-problem record written by the harness
{"id": "gsm8k_test_0", "pred": "18", "gold": "18", "correct": true, "extracted": true,
 "format_ok": true, "n_tokens": 61, "truncated": false}
```

**Other eval inputs.**

```
# MATH-500
problem: Find the value of x such that \sqrt{x+7} = 9.      gold: 74      level: 1
# Perturbed GSM8K (memorisation control: same logic, new names and numbers, re-solved to verify)
original:  Janet's ducks lay 16 eggs per day. She eats three ... bakes ... with four ... $2 per egg.   gold: 18
perturbed: Priya's hens lay 21 eggs per day. She eats five ... bakes ... with six ... $3 per egg.     gold: 30
# Countdown (generated, with a programmatic verifier)
numbers: [2, 3, 5, 7]   target: 29
expected output: <answer>7*5 - 3*2</answer>      verifier: uses each number once, evaluates to 29
# AIME24 (sanity row only)
```

**Metrics computed per eval set.**

- pass@1 greedy = mean(correct) at T=0.
- pass@k, unbiased, from n=64 samples with c correct: `1 - C(n-c, k) / C(n, k)`.
- maj@k = majority vote over the first k samples after answer normalisation.
- extraction-failure rate, format rate, truncation rate, mean / p90 tokens.
- 95 % CI: bootstrap over problems, 2000 resamples. Paired bootstrap on per-problem differences when comparing two models.

**Numbers to watch:** three repeated greedy evals should agree within the reported run-to-run variation; inspect 50 random "wrong" samples by eye for extraction bugs.

---

## Phase 1: baselines

**Input:** the same problems with four templates: zero-shot, 4-shot, 4-shot-CoT, think-tag.

```
# 4-shot-CoT prompt (abbreviated)
Question: <example 1>  Answer: <worked solution> The answer is 12.
Question: <example 2>  Answer: ... The answer is 7.
Question: <example 3>  ...
Question: <example 4>  ...
Question: Janet's ducks lay 16 eggs per day. ...
Answer:
```

**Output:** the completion; last number after "The answer is" or last `\boxed{}` is extracted.

**Sampling for pass@k:** T=0.6, top-p 0.95, n=64, seed fixed, max 4096 new tokens. All samples saved to `outputs/samples/base/*.jsonl` (reused for DPO pairs and for the capability-vs-sharpening analysis).

**Deliverable rows (placeholders):**

| template | GSM8K pass@1 | MATH-500 pass@1 | Countdown | GSM8K pass@64 | MATH-500 pass@64 |
|---|---|---|---|---|---|
| 4-shot-CoT | ~75 (report: 75.44) | ~43 (report: 43.50 on full MATH) | ? | ? | ? |

**Contamination probes run here:** 13-gram overlap of every training file vs every eval file; partial-prompt completion test (feed the first 60 % of a test problem, measure verbatim continuation of the rest); GSM8K perturbed variants (GSM-Symbolic / GSM-Plus style) vs original.

---

## Phase 2: SFT distillation

**Input training example (one sequence, after filtering OpenR1-Math-220k to verified-correct, ≤ 4096 tokens, deduplicated, eval-decontaminated):**

```
[prompt, loss masked]
A conversation between User and Assistant. ... <template as above>
User: A rectangle has perimeter 30 and one side of length 7. What is its area?
Assistant:
[response, loss applied]
<think>
Perimeter = 2(l + w) = 30, so l + w = 15. One side is 7, so the other is 8. Area = 7 * 8 = 56.
</think>
<answer>56</answer>
```

**Output:** checkpoint `sft`. Evaluated with the think-tag template.

**Loss.** Token-level cross-entropy over response tokens only:
`L = -(1/N_resp) Σ_{t ∈ response} log p_θ(y_t | y_<t, x)`.
Packing with a block-diagonal attention mask so packed samples cannot attend to each other.

**Hyperparameters.** Full fine-tune bf16, AdamW (0.9, 0.95), weight decay 0.1, LR 1e-5 after a sweep over {5e-6, 1e-5, 2e-5} on a 10 % subset, cosine to 10 % of peak, 3 % warmup, effective batch 64 packed sequences of 4096, 2 epochs, gradient clipping 1.0, gradient checkpointing, flash attention. Checkpoint selection on the dev split (500 held-out train problems), never on test.
LoRA comparison run: r 64, α 128, dropout 0.05, all linear layers, LR 2e-4, same data and epochs.
Data-scaling ablation: 25 % / 50 % / 100 % of the filtered set, one seed each.

**Numbers to watch:** train and dev loss; dev pass@1 every 500 steps; MATH level 1-2 accuracy (a drop means the LR is erasing base knowledge); format rate should reach > 99 %; response-length distribution vs the training traces.

---

## Phase 3: rejection-sampling fine-tuning (expert iteration)

**Input:** prompts from GSM8K-train + MATH-train levels 1-3; the `sft` model samples 8 responses each at T=1.0.

```
problem: A train travels 120 km in 1.5 hours. What is its average speed in km/h?   gold: 80
sample 1: <think>120 / 1.5 = 80</think><answer>80</answer>        correct
sample 2: <think>120 * 1.5 = 180</think><answer>180</answer>      wrong
sample 3: <think>Speed = distance / time = 120/1.5 = 80</think><answer>80</answer>   correct
... (8 total; this problem: 5 correct of 8 -> solve rate 0.625)
```

**Filter:** keep correct samples, at most 2 per problem (prefer the shortest two), drop problems with solve rate 0 (nothing to learn from) and record the solve-rate histogram.

**Output:** `rft` checkpoint trained from `sft` on the kept set. Side output: `solve_rate.jsonl` used to select learnable prompts for GRPO (0 < rate < 1).

**Loss and hyperparameters.** Same cross-entropy as Phase 2, LR 5e-6 cosine, 1 epoch, batch 64, from `sft` weights. One iteration; a second iteration only if time allows, reported as the diminishing return.

**Numbers to watch:** fraction of problems with solve rate 0 and 1; kept-set size; pass@1 and pass@64 vs `sft` (paired CI).

---

## Phase 4: GRPO with a verifiable reward

**Input:** one prompt (think-tag template, from the learnable set) → G = 8 sampled completions at T=1.0, max 2048 new tokens.

```
prompt: ... User: What is the smallest positive integer that is divisible by 6, 8 and 10?  Assistant:
gold: 120
```

**Reward per completion** (`math-verify` on the `<answer>` block, +0.1 if both tag pairs are well-formed):

| rollout | answer | format ok | correct | reward r_i |
|---|---|---|---|---|
| 1 | 120 | yes | yes | 1.1 |
| 2 | 240 | yes | no | 0.1 |
| 3 | 120 | yes | yes | 1.1 |
| 4 | 120 | no (missing `</think>`) | yes | 1.0 |
| 5 | 60 | yes | no | 0.1 |
| 6 | 120 | yes | yes | 1.1 |
| 7 | (truncated at 2048) | no | no | 0.0 |
| 8 | 120 | yes | yes | 1.1 |

**Group advantage:** `A_i = (r_i - mean(r)) / (std(r) + 1e-4)`. Here mean = 0.7125, std ≈ 0.49, so A ≈ +0.79 for rollouts 1, 3, 6, 8; +0.59 for 4; −1.25 for 2 and 5; −1.45 for 7. A group where all eight are correct (or all wrong) has A = 0 for every token: no gradient, wasted compute. That fraction is logged every step.

**Loss (per token t of rollout i, averaged over all completion tokens in the batch):**

```
ratio_it = π_θ(y_it | ·) / π_old(y_it | ·)            (= 1 exactly when μ = 1, one update per batch)
L_PG     = - min( ratio_it · A_i , clip(ratio_it, 1-ε, 1+ε) · A_i )
KL_it    = exp(ref_logp - logp) - (ref_logp - logp) - 1   (k3 estimator, ≥ 0, ref = rft with adapters off / frozen copy)
L        = mean_tokens( L_PG + β · KL_it )
```

**Prompts:** a 50-100 step debugging run on GSM8K-train first (reward, parsing, vLLM sync, logging; not reported), then the main run on MATH-train levels 1-3 with solve rate in (0, 1) plus a 20 % learnable GSM8K slice. Headline eval: MATH-500 by level; GSM8K, perturbed GSM8K and Countdown alongside.

**Hyperparameters (main run):** from `rft`; LR 1e-6 constant with 10 warmup steps; AdamW (0.9, 0.99), weight decay 0; G 8; 64 prompts per step (512 rollouts); μ = 1; ε 0.2; β 0.04; T 1.0, top-p 1.0; max 2048 new tokens; gradient clipping 1.0; 500 steps; bf16; vLLM colocate with 40 % of GPU memory; eval on the dev split every 50 steps; checkpoint every 100.

**Ablations (1 seed each, one knob each, read against the seed-repeat of the main run):**

| ID | change | hypothesis | what would falsify it |
|---|---|---|---|
| A1 | β = 0 | KL is not needed at 1.7B for 500 steps | divergence, entropy collapse, or dev accuracy below the β=0.04 run outside the CI |
| A2 | ε_high 0.28 + dynamic sampling | entropy stays higher, more accuracy per generated token | entropy curves overlap; same accuracy per token |
| A3 | Dr. GRPO loss (no /|y| per sequence, no /std) | response-length growth is a normalisation artefact | lengths grow the same way |

**Logged every step:** mean / min / max reward, correct rate, format rate, response length mean / p90, truncation rate, policy entropy, KL to reference, clip fraction, zero-advantage-group fraction, gradient norm, tokens generated, rollout and update wall-clock. Every 50 steps: dev pass@1.

**Reward-hacking watch list:** tags with empty think blocks; same final answer repeated regardless of problem; length rising while accuracy is flat; "Wait" chains with no new computation; truncation rate rising.

**Capability-vs-sharpening analysis (after training):** pass@k for k = 1..64 for `rft` and `grpo` on all evals, plus per-problem counts: solved by `grpo` at pass@1 but unsolved by `rft` at pass@64 (new) vs solved by `rft` at some k > 1 (sharpened).

---

## Phase 5: DPO comparison

**Input pair (same prompt, one correct and one incorrect sample from Phase 3):**

```
prompt:   ... User: A train travels 120 km in 1.5 hours. What is its average speed in km/h?  Assistant:
chosen:   <think>Speed = distance / time = 120/1.5 = 80</think><answer>80</answer>
rejected: <think>120 * 1.5 = 180</think><answer>180</answer>
```

**Loss.** `L = -log σ( β · [ (log π_θ(chosen) - log π_ref(chosen)) - (log π_θ(rejected) - log π_ref(rejected)) ] )`, sequence log-probs summed over response tokens, reference = frozen `sft`.

**Hyperparameters.** LR 5e-7 cosine, batch 32 pairs, 1 epoch, β ∈ {0.05, 0.1, 0.3}; one extra round where pairs are regenerated from the current policy (on-policy DPO). Compute matched to the GRPO main run.

**Numbers to watch:** reward margin, chosen and rejected log-prob trends (both falling = likelihood displacement), response length, pass@1 and pass@64.

---

## Phase 6: test-time compute

**Majority vote input/output:**

```
problem: ... smallest positive integer divisible by 6, 8 and 10?
samples (k=8): 120, 240, 120, 120, 60, 120, 120, 120   -> vote: 120 (6/8)   correct
```

**Best-of-N with the ORM:**

```
ORM input:  (prompt, full response) -> scalar score
ORM training label: 1 if the response's extracted answer is correct, else 0 (Phase 3 samples, held-out problems for eval)
ORM loss: binary cross-entropy;  LR 1e-5 cosine, batch 64, 1 epoch, Qwen3-0.6B-Base + scalar head on the last token
best-of-N: pick argmax ORM score among N samples; N ∈ {1, 2, 4, 8, 16, 32, 64}
```

**Budget forcing (s1):**

```
budget 512 thinking tokens: generation stops at </think> or 512 tokens; if the model tries to stop
early, append "Wait" and continue (at most 2 times); then force </think><answer> and generate the answer.
curve: accuracy vs budget ∈ {256, 512, 1024, 2048, 4096}
```

**Chart D6:** x = mean tokens generated per problem (log scale), y = accuracy with CI; one series each for greedy, maj@k, best-of-N, budget forcing, for `sft` and `grpo`.

---

## Phase 7: distillation, quantisation, serving

**Input:** correct traces generated by `grpo` on the training prompts (T=0.6, keep ≤ 2 per problem, same filters as Phase 2), formatted exactly as the Phase 2 SFT example.

**Output:** `distilled-0.6b` (Qwen3-0.6B-Base, full fine-tune, LR 2e-5 cosine, 2 epochs, batch 64, max 4096), then `distilled-0.6b-awq` (AWQ 4-bit, group size 128, calibration on 128 training prompts).

**Serving benchmark (vLLM):** batch sizes 1 / 8 / 32, 512-token prompts, 1024-token generation; report tokens/s, time-to-first-token, p50 latency for 1.7B-bf16, 0.6B-bf16, 0.6B-AWQ, alongside pass@1 on all evals.

**Numbers to watch:** accuracy loss teacher → student and bf16 → AWQ (flag if > 5 points at either step); throughput gain.

---

## Metric formulas kept in one place

- Unbiased pass@k (Chen et al. 2021): `pass@k = 1 - C(n-c, k) / C(n, k)`, computed in log space for stability.
- Majority vote: normalise each answer with `math-verify`, take the most frequent; ties broken by first occurrence.
- Bootstrap CI: resample problems with replacement 2000 times, report 2.5 and 97.5 percentiles.
- Paired difference: for models A and B on the same problems, bootstrap the mean of `correct_A - correct_B`.
- Policy entropy per step: `-Σ_v p(v) log p(v)` averaged over generated positions.
- Clip fraction: share of tokens where `ratio` left `[1-ε, 1+ε]`.
- KL estimators: k1 = `logp - ref_logp`; k2 = `0.5 (logp - ref_logp)^2`; k3 = `exp(ref_logp - logp) - (ref_logp - logp) - 1`.

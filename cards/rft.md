---
license: apache-2.0
base_model: Raviteja546/reasoning-rlvr-sft
tags: [reasoning, math, rft, rejection-sampling, qwen3, rlvr-study]
datasets: [openai/gsm8k, EleutherAI/hendrycks_math]
---

# reasoning-rlvr-rft

Checkpoint from the reasoning-rlvr study (https://github.com/rrtchaganti/reasoning-rlvr), commit `bb6f2a6e01b01397d3edd42861dbba16b88f1b58` on branch `phase-3-rft`. Prompt format: the think-tag template (`<think> ... </think> <answer> ... </answer>`) used by every eval and by the RL phases of the study; see the repo's `src/rlvr/eval/prompts.py`.

## Training

- Start: `Raviteja546/reasoning-rlvr-sft` (`outputs/train/sft_1p7b/final` on the training node); full fine-tune; bf16=True, packing=True (flash_attention_2).
- Data: the start checkpoint's own samples (8 per problem, T = 1.0) on GSM8K-train and MATH-train levels 1-3, the two shortest distinct correct traces per problem kept, built by `rlvr-build-rft` (manifest hash `f44edf33a26c55fc`, 11145 examples, 13448983 tokens per epoch).
- Optimiser: AdamW (0.9, 0.95), lr 5e-06, cosine_with_min_lr to 0.1 of peak, warmup 0.03, weight decay 0.1, grad clip 1.0, effective batch 64 sequences of 4096 tokens, 1 epochs (53 steps), seed 0.
- Loss: cross-entropy on completion tokens only; final train loss 0.4274690218691556.
- Compute: 0.20 GPU-hours on NVIDIA H100 80GB HBM3, peak memory 32.88 GB.

## Evaluation (greedy, batch-invariant vLLM, 95 % bootstrap CI over problems)

Sets: GSM8K test (1,319 problems); `gsm8k_perturbed`, 830 GSM8K test problems with the numbers and names resampled and the gold recomputed from the solution's calculator annotations, no LLM involved (the repo's `src/rlvr/data/perturb_gsm8k.py`); MATH-500 (500); Countdown (500, think-tag only); AIME 2024 (30, one problem = 3.3 points). One greedy completion per problem; the bracket resamples problems, not training seeds.

Accuracy (%):

| template | gsm8k | gsm8k_perturbed | math500 | countdown | aime24 |
|---|---|---|---|---|---|
| think_tag | 78.2 [75.9, 80.4] | 69.0 [65.8, 72.3] | 56.4 [52.2, 60.6] | 7.0 [4.8, 9.4] | 0.0 [0.0, 0.0] |
| few_shot_cot | 79.2 [77.0, 81.3] | 71.4 [68.3, 74.6] | 55.6 [51.4, 60.0] | - | 0.0 [0.0, 0.0] |
| zero_shot | 81.1 [78.9, 83.2] | 74.7 [71.7, 77.8] | 64.8 [60.6, 69.0] | - | 6.7 [0.0, 16.7] |

Truncated / extraction failed (% of completions in that set):

| template | gsm8k | gsm8k_perturbed | math500 | countdown | aime24 |
|---|---|---|---|---|---|
| think_tag | 15.2 / 0.0 | 16.9 / 0.0 | 39.6 / 26.8 | 87.8 / 87.8 | 93.3 / 0.0 |
| few_shot_cot | 10.2 / 0.0 | 10.2 / 0.0 | 20.0 / 8.6 | - | 46.7 / 0.0 |
| zero_shot | 19.8 / 0.0 | 21.1 / 0.0 | 61.8 / 10.4 | - | 63.3 / 0.0 |

Each rate is over that set's completions alone. A completion is truncated when prompt plus completion reach the 4,096-token context (the 4-shot prompt leaves the fewest tokens for the answer). Extraction failed means no rule found an answer after every fallback (the `<answer>` block, then the last `\boxed{}`, then a final phrase, then on the numeric sets the last number); those completions are scored wrong. On the numeric sets (both GSM8K sets and AIME 2024) only a completion with no number in it fails. Countdown accepts only the `<answer>` block or a `\boxed{}`, so a truncated Countdown completion almost always fails and the two rates nearly coincide. Truncation alone does not make a completion wrong: the fallback often still finds the right answer inside the loop.

Paired on the same problems against the checkpoint it was trained from, both under the think-tag template at the 4,096-token context:

| eval | pairs | sft | rft | rft - sft [paired CI] | only sft / only rft |
|---|---|---|---|---|---|
| gsm8k | 1319 | 74.8 [72.5, 77.1] | 78.2 [76.0, 80.3] | **3.3 [1.0, 5.8]** | 101 / 145 |
| gsm8k_perturbed | 830 | 66.0 [62.8, 69.3] | 69.0 [65.8, 72.2] | 3.0 [0.0, 5.9] | 62 / 87 |
| math500 | 500 | 50.6 [46.4, 55.0] | 56.4 [52.0, 60.4] | **5.8 [1.6, 10.2]** | 49 / 78 |
| countdown | 500 | 3.8 [2.2, 5.6] | 7.0 [4.8, 9.4] | **3.2 [0.6, 6.0]** | 16 / 32 |
| aime24 | 30 | 3.3 [0.0, 10.0] | 0.0 [0.0, 0.0] | -3.3 [-10.0, 0.0] | 1 / 0 |

MATH-500 by level, rft - sft:

| level | n | sft | rft | diff [paired CI] |
|---|---|---|---|---|
| 1 | 43 | 74.4 [60.5, 86.0] | 81.4 [69.8, 90.7] | 7.0 [-7.0, 18.6] |
| 2 | 90 | 76.7 [67.8, 85.6] | 75.6 [66.7, 84.4] | -1.1 [-11.1, 8.9] |
| 3 | 105 | 56.2 [46.7, 64.8] | 59.0 [49.5, 68.6] | 2.9 [-8.6, 14.3] |
| 4 | 128 | 48.4 [39.8, 57.0] | 60.2 [51.6, 68.8] | **11.7 [2.3, 21.1]** |
| 5 | 134 | 23.1 [16.4, 30.6] | 29.9 [22.4, 38.1] | 6.7 [0.0, 13.4] |

Bold marks a paired difference whose 95 % CI excludes zero. The accuracy brackets here differ by up to 0.4 from the table above because the paired bootstrap is a separate resampling.

## Known failure modes

- The samples were drawn at a 2,048-token cap (a flag bug; the plan said 4,096), so every kept trace is under 2,048 tokens; the checkpoint is evaluated at 4,096 and what does not finish is cut (truncation rates in the table; the repo's `outputs/reports/phase-3.md`).
- Answers are graded with math-verify on the `<answer>` block, falling back to the last `\boxed{}`, a final phrase or (numeric sets) the last number; what no rule extracts is scored wrong. A completion cut at the window is scored the same way, so truncated completions are often still right (the SFT checkpoint's truncated GSM8K completions: 31 %; the repo's `outputs/reports/phase-4-truncated-accuracy.md`).
- 36 MATH-500 problems (7.2 %) have near duplicates among the MATH-train prompts (the known MATH train/test overlap); their traces, 133 items, were removed before training, with 73 more for 21 MATH-dev problems and 2 for one GSM8K problem (`outputs/reports/phase-3-data.md`). The evals are on the full sets.

## Stopping the loop at decode time (24 Oct 2026)

The rule: when a completion reaches B tokens inside its think block, append `\n</think>\n<answer>` and let the model write up to 128 more tokens (`rlvr-eval --force-stop-at B --force-stop-continue 128`).

At B = 2,048, against the same checkpoint decoded unforced on the same machine (RTX 4060): GSM8K +2.0 [1.2, 2.7], MATH-500 +7.0 [4.6, 9.6]. Against the GRPO checkpoint decoded unforced on that machine it is -1.2 [-3.2, 0.8] / +2.4 [-1.6, 6.6], at 1,125 / 1,710 mean tokens against 993 / 2,049; with both stopped at 2,048, GRPO over this checkpoint is +1.6 [-0.4, 3.6] / -0.4 [-4.4, 3.4] (GSM8K / MATH-500).

Greedy decoding with the project's batch-invariant harness is bit-identical on one machine, not across GPUs: this checkpoint decoded on an H100 and on an RTX 4060 differs on 15-21 % of problems, so compare numbers from the same machine (the repo's `outputs/reports/phase-5.md`, section 4).

## Reproduce

```bash
rlvr-eval --config configs/sample/sft_1.7b_rft_n8.yaml --max-tokens 2048  # the samples, at the cap used
rlvr-build-rft --samples outputs/samples/sft_1.7b_rft_n8
rlvr-sft --config configs/train/rft_1p7b.yaml
rlvr-eval --config configs/eval/rft_1.7b_think_tag.yaml
```

---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B-Base
tags: [reasoning, math, sft, qwen3, rlvr-study]
datasets: [open-r1/OpenR1-Math-220k]
---

# reasoning-rlvr-sft

Checkpoint from the reasoning-rlvr study (https://github.com/rrtchaganti/reasoning-rlvr), commit `bb6f2a6e01b01397d3edd42861dbba16b88f1b58` on branch `phase-3-rft`. Prompt format: the think-tag template (`<think> ... </think> <answer> ... </answer>`) used by every eval and by the RL phases of the study; see the repo's `src/rlvr/eval/prompts.py`.

## Training

- Base: `Qwen/Qwen3-1.7B-Base`; full fine-tune; bf16=True, packing=False (sdpa).
- Data: `outputs/data/sft/train.jsonl` built by `rlvr-build-sft` (manifest hash `45b3fa4862b11e2a`, 23000 of 23000 examples, subset fraction 1.0, 61796025 tokens per epoch).
- Optimiser: AdamW (0.9, 0.95), lr 1e-05, cosine_with_min_lr to 0.1 of peak, warmup 0.03, weight decay 0.1, grad clip 1.0, effective batch 64 sequences of 4096 tokens, 2 epochs (720 steps), seed 0.
- Loss: cross-entropy on completion tokens only; final train loss 0.41336831582917105.
- Compute: 3.04 GPU-hours on NVIDIA H100 80GB HBM3, peak memory 51.6 GB.

## Evaluation (greedy, batch-invariant vLLM, 95 % bootstrap CI over problems)

| template | gsm8k | gsm8k_perturbed | math500 | countdown | aime24 | trunc % | extract-fail % |
|---|---|---|---|---|---|---|---|
| think_tag | 74.8 [72.6, 77.2] | 66.0 [62.8, 69.5] | 50.6 [46.4, 54.8] | 3.8 [2.2, 5.6] | 3.3 [0.0, 10.0] | 96.7 | 94.8 |
| few_shot_cot | 79.4 [77.2, 81.5] | 71.4 [68.2, 74.5] | 56.8 [52.4, 61.2] | - | 0.0 [0.0, 0.0] | 63.3 | 7.4 |
| zero_shot | 82.2 [80.1, 84.4] | 72.7 [69.5, 75.8] | 63.2 [59.0, 67.4] | - | 6.7 [0.0, 16.7] | 59.2 | 10.0 |


Paired against the base model on the same problems:

| eval | pairs | base | sft | sft - base [paired CI] | only base / only sft |
|---|---|---|---|---|---|
| gsm8k | 1319 | 61.8 [59.3, 64.4] | 74.8 [72.5, 77.1] | **13.0 [10.0, 15.8]** | 121 / 293 |
| gsm8k_perturbed | 830 | 51.4 [48.1, 54.7] | 66.0 [62.8, 69.3] | **14.6 [11.0, 18.1]** | 75 / 196 |
| math500 | 500 | 44.8 [40.8, 49.2] | 50.6 [46.4, 55.0] | **5.8 [0.6, 11.0]** | 76 / 105 |
| countdown | 500 | 1.2 [0.4, 2.2] | 3.8 [2.2, 5.6] | **2.6 [0.8, 4.6]** | 5 / 18 |
| aime24 | 30 | 6.7 [0.0, 16.7] | 3.3 [0.0, 10.0] | **-3.3 [-13.3, 6.7]** | 2 / 1 |

MATH-500 by level, sft - base:

| level | n | base | sft | diff [paired CI] |
|---|---|---|---|---|
| 1 | 43 | 51.2 [34.9, 65.1] | 74.4 [60.5, 86.0] | 23.3 [4.7, 41.9] |
| 2 | 90 | 65.6 [55.6, 75.6] | 76.7 [67.8, 85.6] | 11.1 [0.0, 22.2] |
| 3 | 105 | 53.3 [43.8, 61.9] | 56.2 [46.7, 64.8] | 2.9 [-10.5, 15.2] |
| 4 | 128 | 39.1 [30.5, 47.7] | 48.4 [39.8, 57.0] | 9.4 [-1.6, 20.3] |
| 5 | 134 | 27.6 [20.1, 35.1] | 23.1 [16.4, 30.6] | -4.5 [-11.9, 3.0] |


## Known failure modes

- Trained on olympiad-style traces capped at 4096 tokens; longer reasoning is cut at the cap (truncation rate in the table).
- Answers are graded with math-verify on the `<answer>` block, falling back to the last `\boxed{}`, a final phrase or (numeric sets) the last number; what no rule extracts is scored wrong in the accuracy and the extraction-failure rate is reported beside it. A completion cut at the window is scored the same way, so truncated completions are often still right (the SFT checkpoint's truncated GSM8K completions: 31 %; the repo's `outputs/reports/phase-4-truncated-accuracy.md`).
- Three MATH-500 problems and no GSM8K problem were found in the SFT source and removed before training (`outputs/reports/phase-2-data.md`); the base model's own pretraining contamination is measured in the repo's Phase 1 report and is not addressed by this checkpoint.

## Stopping the loop at decode time (24 Oct 2026)

In the trained format a quarter to a half of this checkpoint's greedy completions loop until the window. The rule: when a completion reaches B tokens inside its think block, append `\n</think>\n<answer>` and let the model write up to 128 more tokens (`rlvr-eval --force-stop-at B --force-stop-continue 128`).

At B = 1,536, against the same checkpoint decoded unforced on the same machine (RTX 4060): GSM8K +4.6 [3.4, 6.0], MATH-500 +7.2 [4.0, 10.4], at 1,152 / 1,459 mean tokens; that is more than the RFT checkpoint's whole gain over this one and three quarters of the GRPO checkpoint's. The forced answers are right about half the time.

Greedy decoding with the project's batch-invariant harness is bit-identical on one machine, not across GPUs: this checkpoint decoded on an H100 and on an RTX 4060 differs on 18-23 % of problems, so compare numbers from the same machine (the repo's `outputs/reports/phase-5.md`, section 4).

## Reproduce

```bash
rlvr-build-sft                                   # regenerates the data, checks the manifest hash
rlvr-sft --config configs/train/sft_1p7b.yaml
rlvr-eval --config configs/eval/sft_1.7b_think_tag.yaml
```

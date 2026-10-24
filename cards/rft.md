---
license: apache-2.0
base_model: Raviteja546/reasoning-rlvr-sft
tags: [reasoning, math, sft, qwen3, rlvr-study]
datasets: [open-r1/OpenR1-Math-220k]
---

# reasoning-rlvr-rft

Checkpoint from the reasoning-rlvr study (https://github.com/rrtchaganti/reasoning-rlvr), commit `bb6f2a6e01b01397d3edd42861dbba16b88f1b58` on branch `phase-3-rft`. Prompt format: the think-tag template (`<think> ... </think> <answer> ... </answer>`) used by every eval and by the RL phases of the study; see the repo's `src/rlvr/eval/prompts.py`.

## Training

- Base: `outputs/train/sft_1p7b/final`; full fine-tune; bf16=True, packing=True (flash_attention_2).
- Data: `outputs/data/rft/train.jsonl` built by `rlvr-build-sft` (manifest hash `f44edf33a26c55fc`, 11145 of 11145 examples, subset fraction 1.0, 13448983 tokens per epoch).
- Optimiser: AdamW (0.9, 0.95), lr 5e-06, cosine_with_min_lr to 0.1 of peak, warmup 0.03, weight decay 0.1, grad clip 1.0, effective batch 64 sequences of 4096 tokens, 1 epochs (53 steps), seed 0.
- Loss: cross-entropy on completion tokens only; final train loss 0.4274690218691556.
- Compute: 0.20 GPU-hours on NVIDIA H100 80GB HBM3, peak memory 32.88 GB.

## Evaluation (greedy, batch-invariant vLLM, 95 % bootstrap CI over problems)

| template | gsm8k | gsm8k_perturbed | math500 | countdown | aime24 | trunc % | extract-fail % |
|---|---|---|---|---|---|---|---|
| think_tag | 78.2 [75.9, 80.4] | 69.0 [65.8, 72.3] | 56.4 [52.2, 60.6] | 7.0 [4.8, 9.4] | 0.0 [0.0, 0.0] | 93.3 | 87.8 |
| few_shot_cot | 79.2 [77.0, 81.3] | 71.4 [68.3, 74.6] | 55.6 [51.4, 60.0] | - | 0.0 [0.0, 0.0] | 46.7 | 8.6 |
| zero_shot | 81.1 [78.9, 83.2] | 74.7 [71.7, 77.8] | 64.8 [60.6, 69.0] | - | 6.7 [0.0, 16.7] | 63.3 | 10.4 |


Paired against the base model on the same problems:

| eval | pairs | sft | rft | rft - sft [paired CI] | only sft / only rft |
|---|---|---|---|---|---|
| gsm8k | 1319 | 74.8 [72.5, 77.1] | 78.2 [76.0, 80.3] | **3.3 [1.0, 5.8]** | 101 / 145 |
| gsm8k_perturbed | 830 | 66.0 [62.8, 69.3] | 69.0 [65.8, 72.2] | **3.0 [0.0, 5.9]** | 62 / 87 |
| math500 | 500 | 50.6 [46.4, 55.0] | 56.4 [52.0, 60.4] | **5.8 [1.6, 10.2]** | 49 / 78 |
| countdown | 500 | 3.8 [2.2, 5.6] | 7.0 [4.8, 9.4] | **3.2 [0.6, 6.0]** | 16 / 32 |
| aime24 | 30 | 3.3 [0.0, 10.0] | 0.0 [0.0, 0.0] | **-3.3 [-10.0, 0.0]** | 1 / 0 |

MATH-500 by level, rft - sft:

| level | n | sft | rft | diff [paired CI] |
|---|---|---|---|---|
| 1 | 43 | 74.4 [60.5, 86.0] | 81.4 [69.8, 90.7] | 7.0 [-7.0, 18.6] |
| 2 | 90 | 76.7 [67.8, 85.6] | 75.6 [66.7, 84.4] | -1.1 [-11.1, 8.9] |
| 3 | 105 | 56.2 [46.7, 64.8] | 59.0 [49.5, 68.6] | 2.9 [-8.6, 14.3] |
| 4 | 128 | 48.4 [39.8, 57.0] | 60.2 [51.6, 68.8] | 11.7 [2.3, 21.1] |
| 5 | 134 | 23.1 [16.4, 30.6] | 29.9 [22.4, 38.1] | 6.7 [0.0, 13.4] |


## Known failure modes

- Trained on olympiad-style traces capped at 4096 tokens; longer reasoning is cut at the cap (truncation rate in the table).
- Answers are graded with math-verify on the `<answer>` block, falling back to the last `\boxed{}`, a final phrase or (numeric sets) the last number; what no rule extracts is scored wrong in the accuracy and the extraction-failure rate is reported beside it. A completion cut at the window is scored the same way, so truncated completions are often still right (the SFT checkpoint's truncated GSM8K completions: 31 %; the repo's `outputs/reports/phase-4-truncated-accuracy.md`).
- Three MATH-500 problems and no GSM8K problem were found in the SFT source and removed before training (`outputs/reports/phase-2-data.md`); the base model's own pretraining contamination is measured in the repo's Phase 1 report and is not addressed by this checkpoint.

## Stopping the loop at decode time (24 Oct 2026)

The rule: when a completion reaches B tokens inside its think block, append `\n</think>\n<answer>` and let the model write up to 128 more tokens (`rlvr-eval --force-stop-at B --force-stop-continue 128`).

At B = 2,048, against the same checkpoint decoded unforced on the same machine (RTX 4060): GSM8K +2.0 [1.2, 2.7], MATH-500 +7.0 [4.6, 9.6]. Against the GRPO checkpoint decoded unforced on that machine it is -1.2 [-3.2, 0.8] / +2.4 [-1.6, 6.6], at 1,125 / 1,710 mean tokens against 993 / 2,049; with both stopped at 2,048, GRPO over this checkpoint is +1.6 [-0.4, 3.6] / -0.4 [-4.4, 3.4] (GSM8K / MATH-500).

Greedy decoding with the project's batch-invariant harness is bit-identical on one machine, not across GPUs: this checkpoint decoded on an H100 and on an RTX 4060 differs on 15-21 % of problems, so compare numbers from the same machine (the repo's `outputs/reports/phase-5.md`, section 4).

## Reproduce

```bash
rlvr-build-sft                                   # regenerates the data, checks the manifest hash
rlvr-sft --config configs/train/sft_1p7b.yaml
rlvr-eval --config configs/eval/sft_1.7b_think_tag.yaml
```

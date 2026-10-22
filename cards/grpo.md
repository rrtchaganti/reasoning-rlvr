---
license: apache-2.0
base_model: Raviteja546/reasoning-rlvr-rft
tags: [reasoning, math, grpo, rlvr, qwen3, rlvr-study]
datasets: [openai/gsm8k, EleutherAI/hendrycks_math]
---

# reasoning-rlvr-grpo

Checkpoint from the reasoning-rlvr study (https://github.com/rrtchaganti/reasoning-rlvr), commit `a8b9f6947facfc2f9a627252e438ca5c4bb2da06` on branch `phase-4-grpo`. Prompt format: the think-tag template (`<think> ... </think> <answer> ... </answer>`) used by every eval and by the RL phases of the study; see the repo's `src/rlvr/eval/prompts.py`.

## Training

- Start: `outputs/train/rft_1p7b/final`; full fine-tune; GRPO (TRL `GRPOTrainer` with vLLM colocated) with a verifiable reward: 1 only when the completion finished, has the think/answer format and is correct (`rlvr.train.grpo.score_completion`), else 0.
- Prompts: `outputs/data/rft/learnable_prompts.jsonl` (manifest `data/manifests/rft_iter1.json`, hash `281bdfe31176411b`); 2735 of 4606 prompts drawn over the run, by source {"gsm8k_train": 468, "math_train": 2267}.
- Shape: 32 prompts x 8 samples per step (256 completions), 1536-token cap, T = 1.0, 200 steps, micro-batch 16, one optimiser step per generation batch (clip active: False).
- Optimiser: AdamW (0.9, 0.99), lr 1e-06 constant_with_warmup (warmup 10 steps), weight decay 0.0, grad clip 1.0; loss `grpo` with `scale_rewards=group`, KL beta 0.04, epsilon 0.2, dynamic sampling False; bf16=True (flash_attention_2); seed 0.
- Training signal, first -> last step: reward 0.344 -> 0.684, finished rate 35 % -> 75 %, mean completion 1406 -> 1201 tokens; zero-variance groups 30 % of groups on average; hacking-watch flags raised: none (what each flag reads is in the phase report).
- Compute: 3.13 GPU-hours on NVIDIA H100 80GB HBM3, peak memory 70.35 GB.

## Evaluation (greedy, batch-invariant vLLM, 95 % bootstrap CI over problems)

| template | gsm8k | gsm8k_perturbed | math500 | countdown | aime24 | trunc % | extract-fail % |
|---|---|---|---|---|---|---|---|
| think_tag | 82.3 [80.2, 84.2] | 73.9 [70.7, 77.0] | 63.4 [59.4, 67.6] | 14.0 [11.0, 17.0] | 3.3 [0.0, 10.0] | 72.6 | 72.4 |
| few_shot_cot | 79.3 [77.2, 81.6] | 71.4 [68.3, 74.5] | 57.4 [53.2, 61.6] | - | 0.0 [0.0, 0.0] | 50.0 | 5.6 |
| zero_shot | 84.3 [82.4, 86.3] | 74.7 [71.7, 77.7] | 64.4 [60.4, 68.6] | - | 6.7 [0.0, 16.7] | 65.0 | 7.4 |

The `trunc %` and `extract-fail %` columns are the largest value over the sets of that row (Countdown under the think-tag template dominates both); per-set rates are in each eval's `metrics.json`.

Paired on the same problems against the checkpoint it was trained from:

| eval | pairs | rft | grpo | grpo - rft [paired CI] | only rft / only grpo |
|---|---|---|---|---|---|
| gsm8k | 1319 | 78.2 [76.0, 80.3] | 82.3 [80.1, 84.3] | **4.1 [2.0, 6.1]** | 66 / 120 |
| gsm8k_perturbed | 830 | 69.0 [65.8, 72.2] | 73.9 [70.8, 76.7] | **4.8 [1.9, 7.6]** | 46 / 86 |
| math500 | 500 | 56.4 [52.0, 60.4] | 63.4 [59.0, 67.4] | **7.0 [2.8, 11.0]** | 39 / 74 |
| countdown | 500 | 7.0 [4.8, 9.4] | 14.0 [11.2, 17.2] | **7.0 [3.6, 10.4]** | 22 / 57 |
| aime24 | 30 | 0.0 [0.0, 0.0] | 3.3 [0.0, 10.0] | **3.3 [0.0, 10.0]** | 0 / 1 |

MATH-500 by level, grpo - rft:

| level | n | rft | grpo | diff [paired CI] |
|---|---|---|---|---|
| 1 | 43 | 81.4 [69.8, 90.7] | 93.0 [83.7, 100.0] | 11.6 [0.0, 23.3] |
| 2 | 90 | 75.6 [66.7, 84.4] | 82.2 [73.3, 90.0] | 6.7 [-2.2, 14.5] |
| 3 | 105 | 59.0 [49.5, 68.6] | 72.4 [63.8, 81.0] | 13.3 [3.8, 22.9] |
| 4 | 128 | 60.2 [51.6, 68.8] | 60.2 [51.6, 68.8] | 0.0 [-8.6, 9.4] |
| 5 | 134 | 29.9 [22.4, 38.1] | 37.3 [29.1, 45.5] | 7.5 [-0.7, 15.7] |


## Known failure modes

- In the trained format the gain over the start checkpoint is a termination gain: the policy stops looping and answers; among problems both checkpoints finish the paired accuracy difference is inside its CI (`outputs/reports/phase-4.md`, section 4.5).
- Trained on olympiad-style traces capped at 4096 tokens; longer reasoning is cut at the cap (truncation rate in the table).
- Answers are graded with math-verify on the `<answer>` block, falling back to the last `\boxed{}`, a final phrase or (numeric sets) the last number; what no rule extracts is scored wrong in the accuracy and the extraction-failure rate is reported beside it. A completion cut at the window is scored the same way, so truncated completions are often still right (the SFT checkpoint's truncated GSM8K completions: 31 %; the repo's `outputs/reports/phase-4-truncated-accuracy.md`).
- Three MATH-500 problems and no GSM8K problem were found in the SFT source and removed before training (`outputs/reports/phase-2-data.md`); the base model's own pretraining contamination is measured in the repo's Phase 1 report and is not addressed by this checkpoint.

## Reproduce

```bash
python scripts/phase4_preflight.py --run main --config configs/train/grpo_1p7b.yaml
rlvr-grpo --config configs/train/grpo_1p7b.yaml
rlvr-eval --config configs/eval/grpo_1.7b_think_tag.yaml
```

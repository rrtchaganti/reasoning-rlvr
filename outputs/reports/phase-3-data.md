# Phase 3 RFT data: rft_iter1 (built 2026-10-18 22:03)

Samples from `outputs/samples/sft_1.7b_rft_n8` (run tag `bb6f2a6`, model `outputs/train/sft_1p7b/final`, n = 8, T = 1.0, max_tokens 2048).

## Solve rates (EXP-009)

| split | problems | rate 0 | rate 1 | learnable (0 < rate < 1) | mean rate |
|---|---|---|---|---|---|
| gsm8k_train | 4000 | 173 | 1488 | 2339 (58.5 %) | 0.750 |
| math_train | 3388 | 551 | 570 | 2267 (66.9 %) | 0.541 |
| math_train/level1 | 548 | 36 | 168 | 344 (62.8 %) | 0.706 |
| math_train/level2 | 1304 | 163 | 246 | 895 (68.6 %) | 0.595 |
| math_train/level3 | 1536 | 352 | 156 | 1028 (66.9 %) | 0.436 |

Learnable prompts written for Phase 4: 4606 (hash 281bdfe31176411b).

## Kept set (EXP-010 training data)

- examples: 11145 (at most 2 per problem, shortest distinct correct samples), content hash `f44edf33a26c55fc`
- by split: {'gsm8k_train': 6956, 'math_train': 4189}
- drops by reason: {'incorrect': 20438, 'over_per_problem_cap': 18709, 'truncated': 8593, 'think_tags': 14, 'format': 5}
- prompt + completion tokens: {'n': 11145, 'mean': 1206.7, 'p50': 1143, 'p90': 1779, 'p99': 2131, 'max': 2529, 'total': 13448983}

## Decontamination (D3 a)

| eval | hits (any n-gram) | covered | fuzzy | contaminated | train items dropped |
|---|---|---|---|---|---|
| gsm8k | 1 | 0 | 1 | 1 | 2 |
| math500 | 24 | 16 | 32 | 36 | 133 |
| aime24 | 0 | 0 | 0 | 0 | 0 |
| gsm8k_perturbed | 1 | 0 | 2 | 2 | 4 |
| gsm8k_dev | 1 | 0 | 0 | 0 | 2 |
| math_dev | 10 | 6 | 21 | 21 | 73 |

Dropped in total: 200.

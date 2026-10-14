# Phase 1 greedy baselines (EXP-001)

Commits: ffd3018. Greedy (T = 0), batch-invariant vLLM, 1024 max new tokens. pass@1 in %, 95 % bootstrap CI over problems.

## Accuracy

| model | template | gsm8k | gsm8k_perturbed | math500 | countdown | aime24 |
|---|---|---|---|---|---|---|
| Qwen3-1.7B-Base | zero_shot | 77.6 [75.3, 79.9] | 70.0 [67.0, 73.1] | 53.6 [49.4, 57.6] | - | 3.3 [0.0, 10.0] |
| Qwen3-1.7B-Base | few_shot | 20.5 [18.3, 22.6] | 13.6 [11.3, 15.9] | 17.4 [14.2, 20.8] | - | 3.3 [0.0, 10.0] |
| Qwen3-1.7B-Base | few_shot_cot | 74.5 [72.0, 77.0] | 67.8 [64.8, 71.1] | 41.4 [37.0, 45.8] | - | 0.0 [0.0, 0.0] |
| Qwen3-1.7B-Base | think_tag | 61.3 [58.7, 63.8] | 50.8 [47.6, 54.5] | 43.0 [38.8, 47.4] | 1.2 [0.4, 2.2] | 6.7 [0.0, 16.7] |
| Qwen3-0.6B-Base | zero_shot | 64.9 [62.3, 67.5] | 59.5 [56.1, 62.9] | 37.6 [33.4, 41.6] | - | 3.3 [0.0, 10.0] |
| Qwen3-0.6B-Base | few_shot | 9.8 [8.2, 11.4] | 7.3 [5.5, 9.2] | 11.0 [8.4, 13.8] | - | 0.0 [0.0, 0.0] |
| Qwen3-0.6B-Base | few_shot_cot | 61.9 [59.3, 64.6] | 56.5 [53.1, 59.9] | 28.0 [24.0, 31.8] | - | 0.0 [0.0, 0.0] |
| Qwen3-0.6B-Base | think_tag | 38.7 [36.1, 41.5] | 33.7 [30.6, 37.1] | 33.6 [29.6, 37.8] | 2.0 [1.0, 3.4] | 0.0 [0.0, 0.0] |

## Extraction failures / truncation / mean tokens

| model | template | gsm8k fail% / trunc% / tokens | gsm8k_perturbed fail% / trunc% / tokens | math500 fail% / trunc% / tokens | countdown fail% / trunc% / tokens | aime24 fail% / trunc% / tokens |
|---|---|---|---|---|---|---|
| Qwen3-1.7B-Base | zero_shot | 0.0 / 0.2 / 248 | 0.0 / 0.4 / 243 | 20.8 / 13.0 / 532 | - | 0.0 / 46.7 / 839 |
| Qwen3-1.7B-Base | few_shot | 0.0 / 0.1 / 20 | 0.0 / 0.0 / 17 | 0.0 / 0.2 / 17 | - | 0.0 / 0.0 / 14 |
| Qwen3-1.7B-Base | few_shot_cot | 0.0 / 0.4 / 154 | 0.0 / 0.4 / 148 | 10.0 / 7.4 / 232 | - | 0.0 / 26.7 / 468 |
| Qwen3-1.7B-Base | think_tag | 0.0 / 35.8 / 495 | 0.0 / 35.8 / 490 | 28.2 / 24.2 / 565 | 68.8 / 66.4 / 776 | 0.0 / 40.0 / 801 |
| Qwen3-0.6B-Base | zero_shot | 0.0 / 0.9 / 312 | 0.0 / 1.0 / 310 | 18.0 / 16.4 / 479 | - | 0.0 / 60.0 / 833 |
| Qwen3-0.6B-Base | few_shot | 0.0 / 0.0 / 9 | 0.0 / 0.0 / 9 | 0.0 / 0.6 / 22 | - | 0.0 / 0.0 / 14 |
| Qwen3-0.6B-Base | few_shot_cot | 0.0 / 0.8 / 142 | 0.0 / 0.6 / 138 | 17.2 / 10.6 / 240 | - | 0.0 / 26.7 / 462 |
| Qwen3-0.6B-Base | think_tag | 11.7 / 15.3 / 322 | 13.1 / 14.0 / 299 | 34.4 / 19.0 / 527 | 86.4 / 52.0 / 584 | 0.0 / 43.3 / 752 |

## MATH-500 by level

| model | template | L1 | L2 | L3 | L4 | L5 | all |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B-Base | zero_shot | 81.4 [69.8, 93.0] | 70.0 [60.0, 80.0] | 70.5 [61.9, 79.0] | 47.7 [39.1, 56.2] | 26.1 [18.7, 33.6] | 53.6 [49.4, 57.6] |
| Qwen3-1.7B-Base | few_shot | 37.2 [20.9, 53.5] | 31.1 [21.1, 41.1] | 21.9 [14.3, 30.5] | 8.6 [3.9, 13.3] | 6.7 [3.0, 11.2] | 17.4 [14.2, 20.8] |
| Qwen3-1.7B-Base | few_shot_cot | 67.4 [53.5, 81.4] | 64.4 [54.4, 74.4] | 51.4 [41.9, 61.0] | 35.2 [27.3, 43.0] | 15.7 [9.7, 22.4] | 41.4 [37.0, 45.8] |
| Qwen3-1.7B-Base | think_tag | 51.2 [37.2, 65.1] | 64.4 [54.4, 73.3] | 50.5 [41.0, 60.0] | 39.1 [31.2, 47.7] | 23.9 [17.2, 31.3] | 43.0 [38.8, 47.4] |
| Qwen3-0.6B-Base | zero_shot | 81.4 [69.8, 93.0] | 61.1 [51.1, 71.1] | 43.8 [34.3, 53.3] | 29.7 [22.6, 38.3] | 10.4 [6.0, 15.7] | 37.6 [33.4, 41.6] |
| Qwen3-0.6B-Base | few_shot | 27.9 [16.3, 41.9] | 22.2 [14.4, 31.1] | 10.5 [4.8, 16.2] | 7.0 [3.1, 11.7] | 2.2 [0.0, 5.2] | 11.0 [8.4, 13.8] |
| Qwen3-0.6B-Base | few_shot_cot | 55.8 [41.9, 69.8] | 54.4 [44.4, 65.6] | 34.3 [25.7, 42.9] | 14.1 [7.8, 20.3] | 9.7 [5.2, 14.9] | 28.0 [24.0, 31.8] |
| Qwen3-0.6B-Base | think_tag | 51.2 [37.2, 65.1] | 45.6 [35.6, 55.6] | 44.8 [35.2, 54.3] | 28.9 [21.9, 36.7] | 15.7 [10.4, 22.4] | 33.6 [29.6, 37.8] |

## Which extraction rule fired

| model | template | eval | extraction rule counts |
|---|---|---|---|
| Qwen3-1.7B-Base | zero_shot | gsm8k | boxed 980, last_number 187, final_phrase 152 |
| Qwen3-1.7B-Base | zero_shot | gsm8k_perturbed | boxed 624, last_number 113, final_phrase 93 |
| Qwen3-1.7B-Base | zero_shot | math500 | boxed 388, none 104, final_phrase 8 |
| Qwen3-1.7B-Base | zero_shot | aime24 | last_number 15, boxed 15 |
| Qwen3-1.7B-Base | few_shot | gsm8k | final_phrase 1242, last_number 77 |
| Qwen3-1.7B-Base | few_shot | gsm8k_perturbed | final_phrase 789, last_number 41 |
| Qwen3-1.7B-Base | few_shot | math500 | boxed 499, final_phrase 1 |
| Qwen3-1.7B-Base | few_shot | aime24 | boxed 30 |
| Qwen3-1.7B-Base | few_shot_cot | gsm8k | final_phrase 1089, last_number 230 |
| Qwen3-1.7B-Base | few_shot_cot | gsm8k_perturbed | final_phrase 666, last_number 164 |
| Qwen3-1.7B-Base | few_shot_cot | math500 | boxed 449, none 50, final_phrase 1 |
| Qwen3-1.7B-Base | few_shot_cot | aime24 | boxed 22, last_number 8 |
| Qwen3-1.7B-Base | think_tag | gsm8k | tag 759, last_number 558, final_phrase 2 |
| Qwen3-1.7B-Base | think_tag | gsm8k_perturbed | tag 459, last_number 369, final_phrase 2 |
| Qwen3-1.7B-Base | think_tag | math500 | tag 291, none 141, boxed 64, final_phrase 4 |
| Qwen3-1.7B-Base | think_tag | countdown | none 344, tag 156 |
| Qwen3-1.7B-Base | think_tag | aime24 | last_number 14, tag 13, boxed 3 |
| Qwen3-0.6B-Base | zero_shot | gsm8k | boxed 1302, last_number 15, final_phrase 2 |
| Qwen3-0.6B-Base | zero_shot | gsm8k_perturbed | boxed 818, last_number 10, final_phrase 2 |
| Qwen3-0.6B-Base | zero_shot | math500 | boxed 406, none 90, final_phrase 4 |
| Qwen3-0.6B-Base | zero_shot | aime24 | last_number 18, boxed 11, final_phrase 1 |
| Qwen3-0.6B-Base | few_shot | gsm8k | final_phrase 1312, last_number 7 |
| Qwen3-0.6B-Base | few_shot | gsm8k_perturbed | final_phrase 826, last_number 4 |
| Qwen3-0.6B-Base | few_shot | math500 | boxed 497, final_phrase 3 |
| Qwen3-0.6B-Base | few_shot | aime24 | boxed 30 |
| Qwen3-0.6B-Base | few_shot_cot | gsm8k | final_phrase 1243, last_number 60, boxed 16 |
| Qwen3-0.6B-Base | few_shot_cot | gsm8k_perturbed | final_phrase 786, last_number 36, boxed 8 |
| Qwen3-0.6B-Base | few_shot_cot | math500 | boxed 414, none 86 |
| Qwen3-0.6B-Base | few_shot_cot | aime24 | boxed 20, last_number 10 |
| Qwen3-0.6B-Base | think_tag | gsm8k | last_number 655, tag 264, boxed 191, none 154, final_phrase 55 |
| Qwen3-0.6B-Base | think_tag | gsm8k_perturbed | last_number 402, tag 171, boxed 111, none 109, final_phrase 37 |
| Qwen3-0.6B-Base | think_tag | math500 | boxed 266, none 172, tag 47, final_phrase 15 |
| Qwen3-0.6B-Base | think_tag | countdown | none 432, tag 62, boxed 6 |
| Qwen3-0.6B-Base | think_tag | aime24 | boxed 15, last_number 12, tag 2, final_phrase 1 |

| run | pairs | k | original pass@k [CI] | perturbed pass@k [CI] | gap (orig - pert) [paired CI] |
|---|---|---|---|---|---|
| sft_1.7b_n64_few_shot_cot | 830 | 1 | 73.0 [71.3, 74.8] | 63.0 [60.7, 65.3] | **10.0 [8.3, 11.8]** |
| sft_1.7b_n64_few_shot_cot | 830 | 4 | 91.8 [90.2, 93.3] | 81.0 [78.5, 83.5] | **10.8 [8.8, 12.9]** |
| sft_1.7b_n64_few_shot_cot | 830 | 16 | 96.0 [94.9, 97.2] | 86.0 [83.7, 88.1] | **10.1 [8.1, 12.1]** |
| sft_1.7b_n64_few_shot_cot | 830 | 64 | 98.1 [97.1, 98.9] | 88.8 [86.6, 90.8] | **9.3 [7.2, 11.4]** |

Solved at least once in 64 tries: both 732, only original 82, only perturbed 5, neither 11 (of 830 pairs).

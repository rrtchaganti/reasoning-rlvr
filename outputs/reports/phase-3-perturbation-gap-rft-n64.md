| run | pairs | k | original pass@k [CI] | perturbed pass@k [CI] | gap (orig - pert) [paired CI] |
|---|---|---|---|---|---|
| rft_1.7b_n64_few_shot_cot | 67 | 1 | 75.3 [69.7, 80.6] | 72.5 [65.3, 78.8] | **2.8 [-1.7, 8.1]** |
| rft_1.7b_n64_few_shot_cot | 67 | 4 | 93.9 [88.8, 97.7] | 89.7 [83.2, 95.7] | **4.2 [0.5, 8.9]** |
| rft_1.7b_n64_few_shot_cot | 67 | 16 | 97.4 [93.5, 99.9] | 92.9 [86.5, 98.2] | **4.5 [0.8, 9.1]** |
| rft_1.7b_n64_few_shot_cot | 67 | 64 | 98.5 [95.5, 100.0] | 94.0 [88.1, 98.5] | **4.5 [0.0, 9.0]** |

Solved at least once in 64 tries: both 63, only original 3, only perturbed 0, neither 1 (of 67 pairs).

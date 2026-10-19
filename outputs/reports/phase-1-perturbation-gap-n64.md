| run | pairs | k | original pass@k [CI] | perturbed pass@k [CI] | gap (orig - pert) [paired CI] |
|---|---|---|---|---|---|
| base_1.7b_n64_few_shot_cot | 830 | 1 | 74.6 [72.5, 76.6] | 64.5 [62.0, 67.0] | **10.2 [8.3, 12.0]** |
| base_1.7b_n64_few_shot_cot | 830 | 4 | 91.2 [89.6, 92.7] | 80.5 [78.0, 82.8] | **10.7 [8.7, 12.7]** |
| base_1.7b_n64_few_shot_cot | 830 | 16 | 96.6 [95.5, 97.6] | 86.9 [84.7, 89.0] | **9.7 [7.7, 11.6]** |
| base_1.7b_n64_few_shot_cot | 830 | 64 | 98.8 [98.0, 99.4] | 90.2 [88.2, 92.2] | **8.6 [6.6, 10.6]** |

Solved at least once in 64 tries: both 746, only original 74, only perturbed 3, neither 7 (of 830 pairs).

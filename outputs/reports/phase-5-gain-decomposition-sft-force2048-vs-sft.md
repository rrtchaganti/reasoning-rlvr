Paired gain sft+force2048 - sft, in points of each set, split by whether the sft completion finished (95 % bootstrap CIs over problems):

| eval | pairs | net gain | on problems sft truncated | on problems sft finished | both finished (n) | accuracy there, sft -> sft+force2048 | paired diff there |
|---|---|---|---|---|---|---|---|
| gsm8k | 1319 | +5.8 [+3.6, +7.9] | +7.4 [+5.7, +9.2] | -1.6 [-2.8, -0.5] | 949 | 91.9 -> 89.7 | -2.2 [-3.9, -0.6] |
| math500 | 500 | +10.4 [+6.4, +14.6] | +14.2 [+10.8, +17.8] | -3.8 [-6.0, -1.6] | 238 | 90.8 -> 83.2 | -7.6 [-12.2, -3.4] |

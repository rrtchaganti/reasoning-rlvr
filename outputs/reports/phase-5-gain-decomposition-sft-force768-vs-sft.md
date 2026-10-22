Paired gain sft+force768 - sft, in points of each set, split by whether the sft completion finished (95 % bootstrap CIs over problems):

| eval | pairs | net gain | on problems sft truncated | on problems sft finished | both finished (n) | accuracy there, sft -> sft+force768 | paired diff there |
|---|---|---|---|---|---|---|---|
| gsm8k | 1319 | +3.0 [+0.8, +5.2] | +6.7 [+5.2, +8.4] | -3.8 [-5.2, -2.4] | 938 | 92.0 -> 86.8 | -5.2 [-7.2, -3.3] |
| math500 | 500 | +0.0 [-4.4, +4.2] | +8.4 [+5.0, +11.8] | -8.4 [-11.0, -6.0] | 238 | 91.2 -> 73.9 | -17.2 [-22.7, -12.2] |

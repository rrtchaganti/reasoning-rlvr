Paired gain sft+force1024 - sft, in points of each set, split by whether the sft completion finished (95 % bootstrap CIs over problems):

| eval | pairs | net gain | on problems sft truncated | on problems sft finished | both finished (n) | accuracy there, sft -> sft+force1024 | paired diff there |
|---|---|---|---|---|---|---|---|
| gsm8k | 1319 | +4.9 [+2.8, +7.1] | +7.3 [+5.6, +9.0] | -2.4 [-3.6, -1.1] | 940 | 91.9 -> 88.5 | -3.4 [-5.2, -1.6] |
| math500 | 500 | +5.6 [+1.4, +9.6] | +11.0 [+7.6, +14.4] | -5.4 [-7.8, -3.0] | 237 | 91.1 -> 79.7 | -11.4 [-16.5, -6.8] |

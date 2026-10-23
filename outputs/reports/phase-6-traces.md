Loop = the first 50-word window that recurs later in the completion; onset = its position as a fraction of the completion's words; unit = words to its next occurrence. First samples, greedy.

**gsm8k**

| run | n | truncated % | truncated: words (median) | with a loop % | loop onset (median) | repeat unit words (median) | answer before loop % | answer by end of first cycle % | Wait / completion (trunc. / fin.) | verify phrases (trunc. / fin.) | finished with a loop % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 1319 | 35.8 | 2003 | 97.2 | 0.04 | 4 | 0.0 | 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 2.4 |
| sft | 1319 | 28.1 | 2618 | 99.2 | 0.20 | 133 | 32.7 | 42.5 | 24.3 / 2.1 | 9.6 / 2.6 | 4.0 |
| rft | 1319 | 15.2 | 2658 | 98.5 | 0.18 | 119 | 29.9 | 45.2 | 26.3 / 1.7 | 9.7 / 2.3 | 4.5 |
| grpo | 1319 | 4.6 | 2592 | 96.7 | 0.19 | 116 | 22.0 | 32.2 | 27.6 / 1.5 | 4.7 / 1.9 | 2.7 |
| kl_off | 1319 | 2.1 | 2698 | 96.4 | 0.13 | 116 | 14.8 | 29.6 | 17.0 / 0.2 | 10.7 / 0.3 | 0.7 |
| cap2048 | 1319 | 9.6 | 2536 | 98.4 | 0.20 | 102 | 29.6 | 39.2 | 22.1 / 1.6 | 9.7 / 2.1 | 3.0 |
| sft_8k | 300 | 27.0 | 5562 | 100.0 | 0.09 | 131 | 29.6 | 43.2 | 55.7 / 2.1 | 15.9 / 2.5 | 3.7 |
| rft_8k | 300 | 14.0 | 5532 | 100.0 | 0.08 | 154 | 21.4 | 42.9 | 54.3 / 1.7 | 16.9 / 2.3 | 4.7 |
| sft_force1536 | 1319 | 0.5 | 950 | 0.0 | - | - | - | - | 3.7 / 3.6 | 4.2 / 2.9 | 24.6 |
| rft_force2048 | 1319 | 0.1 | 1357 | 0.0 | - | - | - | - | 11.0 / 3.5 | 6.0 / 2.7 | 16.6 |
| sft_force3072 | 1319 | 0.0 | - | - | - | - | - | - | - / 6.2 | - / 3.9 | 28.7 |
| grpo_force2048 | 1319 | 0.1 | 1039 | 0.0 | - | - | - | - | 3.0 / 1.9 | 3.0 / 2.0 | 5.8 |

| forced run | forced | forced answers right % | own text already states an answer % | right when it did % | right when it did not % | share of forced-right with an answer before the tags % |
|---|---|---|---|---|---|---|
| sft_force1536 | 437 | 54.5 | 52.2 | 64.0 | 44.0 | 61.3 |
| rft_force2048 | 217 | 39.6 | 58.1 | 47.6 | 28.6 | 69.8 |
| sft_force3072 | 340 | 47.9 | 46.8 | 55.3 | 41.4 | 54.0 |
| grpo_force2048 | 66 | 16.7 | 43.9 | 17.2 | 16.2 | 45.5 |

| pair | KS distance, repeat-unit length (truncated) | KS distance, loop onset (truncated) | n |
|---|---|---|---|
| sft:grpo | 0.11 | 0.14 | 367 / 59 |
| sft:rft | 0.07 | 0.10 | 367 / 197 |
| rft:grpo | 0.10 | 0.08 | 197 / 59 |
| sft:kl_off | 0.13 | 0.37 | 367 / 27 |
| grpo:cap2048 | 0.14 | 0.15 | 59 / 125 |

**math500**

| run | n | truncated % | truncated: words (median) | with a loop % | loop onset (median) | repeat unit words (median) | answer before loop % | answer by end of first cycle % | Wait / completion (trunc. / fin.) | verify phrases (trunc. / fin.) | finished with a loop % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 500 | 20.0 | 1974 | 85.0 | 0.06 | 44 | 1.2 | 1.2 | 0.0 / 0.0 | 0.0 / 0.0 | 3.5 |
| sft | 500 | 52.0 | 2235 | 87.3 | 0.35 | 140 | 14.5 | 25.6 | 20.6 / 2.9 | 7.7 / 4.2 | 5.8 |
| rft | 500 | 39.6 | 2205 | 88.4 | 0.33 | 123 | 21.7 | 28.0 | 23.3 / 2.6 | 6.7 / 3.9 | 6.3 |
| grpo | 500 | 20.8 | 2216 | 76.0 | 0.31 | 143 | 8.9 | 15.2 | 24.0 / 2.0 | 4.7 / 2.9 | 3.3 |
| kl_off | 500 | 13.4 | 2193 | 89.6 | 0.33 | 188 | 6.7 | 11.7 | 8.1 / 0.6 | 4.0 / 0.8 | 4.2 |
| cap2048 | 500 | 26.2 | 2282 | 85.5 | 0.33 | 140 | 15.2 | 20.5 | 15.8 / 2.1 | 5.6 / 3.2 | 3.5 |
| sft_8k | 300 | 46.3 | 4695 | 97.8 | 0.19 | 144 | 16.9 | 25.0 | 42.9 / 3.1 | 15.4 / 4.5 | 9.3 |
| rft_8k | 300 | 36.0 | 4600 | 95.4 | 0.17 | 126 | 29.1 | 32.0 | 59.7 / 2.8 | 14.9 / 4.2 | 9.4 |
| sft_force1536 | 500 | 0.8 | 852 | 0.0 | - | - | - | - | 1.0 / 4.2 | 1.0 / 3.2 | 17.7 |
| rft_force2048 | 500 | 0.6 | 1065 | 33.3 | 0.70 | 8 | 0.0 | 0.0 | 1.0 / 4.6 | 2.0 / 3.5 | 21.7 |
| sft_force3072 | 500 | 0.2 | 1458 | 0.0 | - | - | - | - | 1.0 / 8.8 | 4.0 / 5.1 | 40.5 |
| grpo_force2048 | 500 | 0.2 | 737 | 0.0 | - | - | - | - | 1.0 / 3.7 | 2.0 / 2.7 | 16.6 |

| forced run | forced | forced answers right % | own text already states an answer % | right when it did % | right when it did not % | share of forced-right with an answer before the tags % |
|---|---|---|---|---|---|---|
| sft_force1536 | 390 | 49.7 | 34.1 | 75.2 | 36.6 | 51.5 |
| rft_force2048 | 266 | 38.0 | 33.8 | 62.2 | 25.6 | 55.4 |
| sft_force3072 | 268 | 36.6 | 35.8 | 54.2 | 26.7 | 53.1 |
| grpo_force2048 | 181 | 21.5 | 33.1 | 41.7 | 11.6 | 64.1 |

| pair | KS distance, repeat-unit length (truncated) | KS distance, loop onset (truncated) | n |
|---|---|---|---|
| sft:grpo | 0.11 | 0.16 | 227 / 79 |
| sft:rft | 0.07 | 0.07 | 227 / 175 |
| rft:grpo | 0.11 | 0.11 | 175 / 79 |
| sft:kl_off | 0.17 | 0.15 | 227 / 60 |
| grpo:cap2048 | 0.11 | 0.13 | 79 / 112 |

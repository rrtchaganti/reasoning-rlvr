Loop = the first 50-word window that recurs later in the completion; onset = its position as a fraction of the completion's words; unit = words to its next occurrence. First samples, greedy.

**gsm8k**

| run | n | truncated % | truncated: words (median) | with a loop % | loop onset (median) | repeat unit words (median) | answer before loop % | Wait / completion (trunc. / fin.) | verify phrases (trunc. / fin.) | finished with a loop % |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 1319 | 35.8 | 2003 | 97.2 | 0.04 | 4 | 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 2.4 |
| sft | 1319 | 28.1 | 2618 | 99.2 | 0.20 | 133 | 32.7 | 24.3 / 2.1 | 9.6 / 2.6 | 4.0 |
| rft | 1319 | 15.2 | 2658 | 98.5 | 0.18 | 119 | 29.9 | 26.3 / 1.7 | 9.7 / 2.3 | 4.5 |
| grpo | 1319 | 4.6 | 2592 | 96.7 | 0.19 | 116 | 22.0 | 27.6 / 1.5 | 4.7 / 1.9 | 2.7 |
| kl_off | 1319 | 2.1 | 2698 | 96.4 | 0.13 | 116 | 14.8 | 17.0 / 0.2 | 10.7 / 0.3 | 0.7 |
| cap2048 | 1319 | 9.6 | 2536 | 98.4 | 0.20 | 102 | 29.6 | 22.1 / 1.6 | 9.7 / 2.1 | 3.0 |
| sft_8k | 300 | 27.0 | 5562 | 100.0 | 0.09 | 131 | 29.6 | 55.7 / 2.1 | 15.9 / 2.5 | 3.7 |
| rft_8k | 300 | 14.0 | 5532 | 100.0 | 0.08 | 154 | 21.4 | 54.3 / 1.7 | 16.9 / 2.3 | 4.7 |
| sft_force1536 | 1319 | 0.5 | 950 | 0.0 | - | - | - | 3.7 / 3.6 | 4.2 / 2.9 | 24.6 |
| rft_force2048 | 1319 | 0.1 | 1357 | 0.0 | - | - | - | 11.0 / 3.5 | 6.0 / 2.7 | 16.6 |

| pair | KS distance, repeat-unit length (truncated) | KS distance, loop onset (truncated) | n |
|---|---|---|---|
| sft:grpo | 0.11 | 0.14 | 367 / 59 |
| sft:rft | 0.07 | 0.10 | 367 / 197 |
| rft:grpo | 0.10 | 0.08 | 197 / 59 |
| sft:kl_off | 0.13 | 0.37 | 367 / 27 |
| grpo:cap2048 | 0.14 | 0.15 | 59 / 125 |

**math500**

| run | n | truncated % | truncated: words (median) | with a loop % | loop onset (median) | repeat unit words (median) | answer before loop % | Wait / completion (trunc. / fin.) | verify phrases (trunc. / fin.) | finished with a loop % |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 500 | 20.0 | 1974 | 85.0 | 0.06 | 44 | 1.2 | 0.0 / 0.0 | 0.0 / 0.0 | 3.5 |
| sft | 500 | 52.0 | 2235 | 87.3 | 0.35 | 140 | 14.5 | 20.6 / 2.9 | 7.7 / 4.2 | 5.8 |
| rft | 500 | 39.6 | 2205 | 88.4 | 0.33 | 123 | 21.7 | 23.3 / 2.6 | 6.7 / 3.9 | 6.3 |
| grpo | 500 | 20.8 | 2216 | 76.0 | 0.31 | 143 | 8.9 | 24.0 / 2.0 | 4.7 / 2.9 | 3.3 |
| kl_off | 500 | 13.4 | 2193 | 89.6 | 0.33 | 188 | 6.7 | 8.1 / 0.6 | 4.0 / 0.8 | 4.2 |
| cap2048 | 500 | 26.2 | 2282 | 85.5 | 0.33 | 140 | 15.2 | 15.8 / 2.1 | 5.6 / 3.2 | 3.5 |
| sft_8k | 300 | 46.3 | 4695 | 97.8 | 0.19 | 144 | 16.9 | 42.9 / 3.1 | 15.4 / 4.5 | 9.3 |
| rft_8k | 300 | 36.0 | 4600 | 95.4 | 0.17 | 126 | 29.1 | 59.7 / 2.8 | 14.9 / 4.2 | 9.4 |
| sft_force1536 | 500 | 0.8 | 852 | 0.0 | - | - | - | 1.0 / 4.2 | 1.0 / 3.2 | 17.7 |
| rft_force2048 | 500 | 0.6 | 1065 | 33.3 | 0.70 | 8 | 0.0 | 1.0 / 4.6 | 2.0 / 3.5 | 21.7 |

| pair | KS distance, repeat-unit length (truncated) | KS distance, loop onset (truncated) | n |
|---|---|---|---|
| sft:grpo | 0.11 | 0.16 | 227 / 79 |
| sft:rft | 0.07 | 0.07 | 227 / 175 |
| rft:grpo | 0.11 | 0.11 | 175 / 79 |
| sft:kl_off | 0.17 | 0.15 | 227 / 60 |
| grpo:cap2048 | 0.11 | 0.13 | 79 / 112 |

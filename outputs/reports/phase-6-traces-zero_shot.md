Loop = the first 50-word window that recurs later in the completion; onset = its position as a fraction of the completion's words; unit = words to its next occurrence. First samples, greedy.

**gsm8k**

| run | n | truncated % | truncated: words (median) | with a loop % | loop onset (median) | repeat unit words (median) | answer before loop % | Wait / completion (trunc. / fin.) | verify phrases (trunc. / fin.) | finished with a loop % |
|---|---|---|---|---|---|---|---|---|---|---|
| sft | 1319 | 17.3 | 2195 | 98.7 | 0.10 | 127 | 39.6 | 3.0 / 0.0 | 1.1 / 0.1 | 25.2 |
| rft | 1319 | 19.8 | 2200 | 99.6 | 0.09 | 118 | 40.8 | 0.9 / 0.0 | 1.0 / 0.0 | 26.5 |
| grpo | 1319 | 16.0 | 2229 | 99.5 | 0.08 | 120 | 39.0 | 0.8 / 0.0 | 1.0 / 0.1 | 23.5 |
| kl_off | 1319 | 5.6 | 2024 | 100.0 | 0.10 | 66 | 54.1 | 0.1 / 0.0 | 0.5 / 0.0 | 19.0 |

| pair | KS distance, repeat-unit length (truncated) | KS distance, loop onset (truncated) | n |
|---|---|---|---|
| sft:grpo | 0.09 | 0.12 | 225 / 210 |

**math500**

| run | n | truncated % | truncated: words (median) | with a loop % | loop onset (median) | repeat unit words (median) | answer before loop % | Wait / completion (trunc. / fin.) | verify phrases (trunc. / fin.) | finished with a loop % |
|---|---|---|---|---|---|---|---|---|---|---|
| sft | 500 | 59.2 | 2104 | 98.0 | 0.02 | 215 | 9.7 | 0.0 / 0.0 | 0.9 / 0.3 | 81.4 |
| rft | 500 | 61.8 | 2075 | 97.7 | 0.02 | 210 | 11.3 | 0.0 / 0.0 | 0.9 / 0.3 | 75.9 |
| grpo | 500 | 65.0 | 2087 | 97.5 | 0.02 | 205 | 12.9 | 0.0 / 0.0 | 0.7 / 0.2 | 76.0 |
| kl_off | 500 | 23.2 | 2043 | 98.3 | 0.08 | 172 | 15.8 | 0.2 / 0.0 | 0.9 / 0.1 | 69.3 |

| pair | KS distance, repeat-unit length (truncated) | KS distance, loop onset (truncated) | n |
|---|---|---|---|
| sft:grpo | 0.06 | 0.07 | 290 / 317 |

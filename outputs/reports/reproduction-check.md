# Reproduction check (EXP-024)

| run | eval | pairs (ref / new) | same text | same extracted | same grade | pass@1 ref | pass@1 new | identical |
|---|---|---|---|---|---|---|---|---|
| base_1.7b_zero_shot | aime24 | 30 (30 / 30) | 30 | 30 | 30 | 3.3 | 3.3 | yes |
| base_1.7b_zero_shot | gsm8k | 1319 (1319 / 1319) | 1319 | 1319 | 1319 | 77.6 | 77.6 | yes |
| base_1.7b_zero_shot | gsm8k_perturbed | 830 (830 / 830) | 830 | 830 | 830 | 70.0 | 70.0 | yes |
| base_1.7b_zero_shot | math500 | 500 (500 / 500) | 500 | 500 | 500 | 53.6 | 53.6 | yes |
| base_1.7b_few_shot | aime24 | 30 (30 / 30) | 30 | 30 | 30 | 3.3 | 3.3 | yes |
| base_1.7b_few_shot | gsm8k | 1319 (1319 / 1319) | 1319 | 1319 | 1319 | 20.5 | 20.5 | yes |
| base_1.7b_few_shot | gsm8k_perturbed | 830 (830 / 830) | 830 | 830 | 830 | 13.6 | 13.6 | yes |
| base_1.7b_few_shot | math500 | 500 (500 / 500) | 500 | 500 | 500 | 17.4 | 17.4 | yes |
| base_1.7b_few_shot_cot | aime24 | 30 (30 / 30) | 30 | 30 | 30 | 0.0 | 0.0 | yes |
| base_1.7b_few_shot_cot | gsm8k | 1319 (1319 / 1319) | 1319 | 1319 | 1319 | 74.5 | 74.5 | yes |
| base_1.7b_few_shot_cot | gsm8k_perturbed | 830 (830 / 830) | 830 | 830 | 830 | 67.8 | 67.8 | yes |
| base_1.7b_few_shot_cot | math500 | 500 (500 / 500) | 500 | 500 | 500 | 41.4 | 41.4 | yes |
| base_1.7b_think_tag | aime24 | 30 (30 / 30) | 30 | 30 | 30 | 6.7 | 6.7 | yes |
| base_1.7b_think_tag | countdown | 500 (500 / 500) | 500 | 500 | 500 | 1.2 | 1.2 | yes |
| base_1.7b_think_tag | gsm8k | 1319 (1319 / 1319) | 1319 | 1319 | 1319 | 61.3 | 61.3 | yes |
| base_1.7b_think_tag | gsm8k_perturbed | 830 (830 / 830) | 830 | 830 | 830 | 50.8 | 50.8 | yes |
| base_1.7b_think_tag | math500 | 500 (500 / 500) | 500 | 500 | 500 | 43.0 | 43.0 | yes |

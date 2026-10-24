# The final chart: accuracy against generated tokens (pinned 300-problem subsets)

Greedy points are laptop runs (forced stop at a budget, or none); vote rows are the H100 n = 32 sample sets
(T = 0.6). Accuracy in %, 95 % bootstrap CI over problems; tokens are mean generated tokens per problem.

## GSM8K

| checkpoint | method | accuracy [95 % CI] | tokens |
|---|---|---|---|
| sft | greedy, stop at 768 (laptop) | 78.0 [73.0, 82.7] | 751 |
| sft | greedy, stop at 1,024 (laptop) | 81.3 [76.7, 85.7] | 928 |
| sft | greedy, stop at 1,536 (laptop) | 82.7 [78.0, 86.7] | 1,143 |
| sft | greedy, stop at 2,048 (laptop) | 83.0 [78.3, 87.0] | 1,290 |
| sft | greedy, stop at 3,072 (laptop) | 82.7 [78.0, 86.7] | 1,551 |
| sft | greedy, no stop (laptop) | 77.3 [72.3, 82.0] | 1,802 |
| sft | maj@1 (H100) | 80.0 [76.6, 83.0] | 1,688 |
| sft | maj@2 (H100) | 80.3 [77.0, 83.3] | 3,375 |
| sft | maj@4 (H100) | 86.7 [83.5, 89.9] | 6,750 |
| sft | maj@8 (H100) | 90.1 [87.2, 92.9] | 13,500 |
| sft | maj@16 (H100) | 90.5 [87.4, 93.4] | 27,000 |
| sft | maj@32 (H100) | 90.7 [87.0, 93.7] | 54,000 |
| rft | greedy, stop at 768 (laptop) | 79.7 [75.0, 84.0] | 736 |
| rft | greedy, stop at 1,024 (laptop) | 80.3 [76.0, 84.7] | 878 |
| rft | greedy, stop at 1,536 (laptop) | 82.3 [78.0, 86.7] | 1,022 |
| rft | greedy, stop at 2,048 (laptop) | 82.7 [78.3, 87.0] | 1,113 |
| rft | greedy, stop at 3,072 (laptop) | 82.3 [78.0, 86.7] | 1,274 |
| rft | greedy, no stop (laptop) | 80.7 [76.0, 85.3] | 1,430 |
| rft | maj@1 (H100) | 82.5 [79.4, 85.4] | 1,281 |
| rft | maj@2 (H100) | 82.4 [79.4, 85.2] | 2,561 |
| rft | maj@4 (H100) | 88.6 [85.6, 91.4] | 5,122 |
| rft | maj@8 (H100) | 90.4 [87.3, 93.1] | 10,245 |
| rft | maj@16 (H100) | 91.3 [88.3, 94.1] | 20,489 |
| rft | maj@32 (H100) | 91.7 [88.3, 94.3] | 40,979 |
| rft | verifier-weighted vote @1 (H100) | 82.5 | 1,281 |
| rft | verifier-weighted vote @2 (H100) | 87.6 | 2,561 |
| rft | verifier-weighted vote @4 (H100) | 90.3 | 5,122 |
| rft | verifier-weighted vote @8 (H100) | 91.8 | 10,245 |
| rft | verifier-weighted vote @16 (H100) | 92.5 | 20,489 |
| rft | verifier-weighted vote @32 (H100) | 93.3 | 40,979 |
| grpo | greedy, stop at 1,536 (laptop) | 83.7 [79.3, 87.7] | 859 |
| grpo | greedy, stop at 2,048 (laptop) | 83.7 [79.3, 87.7] | 892 |
| grpo | greedy, no stop (laptop) | 83.0 [78.7, 87.0] | 1,006 |
| grpo | maj@1 (H100) | 83.7 [80.5, 86.7] | 935 |
| grpo | maj@2 (H100) | 83.8 [80.7, 86.7] | 1,869 |
| grpo | maj@4 (H100) | 88.9 [86.0, 91.7] | 3,739 |
| grpo | maj@8 (H100) | 90.6 [87.7, 93.4] | 7,478 |
| grpo | maj@16 (H100) | 91.2 [88.2, 94.0] | 14,956 |
| grpo | maj@32 (H100) | 91.3 [88.0, 94.3] | 29,911 |

## MATH-500

| checkpoint | method | accuracy [95 % CI] | tokens |
|---|---|---|---|
| sft | greedy, stop at 768 (laptop) | 50.0 [44.3, 55.7] | 777 |
| sft | greedy, stop at 1,024 (laptop) | 54.7 [49.0, 60.0] | 1,023 |
| sft | greedy, stop at 1,536 (laptop) | 59.7 [54.3, 65.3] | 1,459 |
| sft | greedy, stop at 2,048 (laptop) | 61.0 [55.3, 66.3] | 1,827 |
| sft | greedy, stop at 3,072 (laptop) | 62.7 [57.3, 68.0] | 2,426 |
| sft | greedy, no stop (laptop) | 51.3 [45.7, 57.0] | 2,938 |
| sft | maj@1 (H100) | 59.2 [55.1, 63.5] | 2,850 |
| sft | maj@2 (H100) | 63.2 [58.9, 67.0] | 5,699 |
| sft | maj@4 (H100) | 70.8 [66.6, 74.9] | 11,399 |
| sft | maj@8 (H100) | 76.1 [71.5, 80.2] | 22,797 |
| sft | maj@16 (H100) | 79.3 [74.9, 83.2] | 45,594 |
| sft | maj@32 (H100) | 81.7 [77.3, 86.0] | 91,189 |
| rft | greedy, stop at 768 (laptop) | 51.7 [46.0, 57.3] | 776 |
| rft | greedy, stop at 1,024 (laptop) | 58.7 [53.0, 64.3] | 1,012 |
| rft | greedy, stop at 1,536 (laptop) | 63.0 [57.7, 68.3] | 1,401 |
| rft | greedy, stop at 2,048 (laptop) | 65.7 [60.3, 71.0] | 1,699 |
| rft | greedy, stop at 3,072 (laptop) | 66.0 [60.7, 71.3] | 2,166 |
| rft | greedy, no stop (laptop) | 57.7 [52.0, 63.3] | 2,567 |
| rft | maj@1 (H100) | 61.4 [57.3, 65.6] | 2,512 |
| rft | maj@2 (H100) | 64.8 [60.6, 68.5] | 5,025 |
| rft | maj@4 (H100) | 71.9 [67.8, 75.9] | 10,049 |
| rft | maj@8 (H100) | 76.8 [72.1, 80.9] | 20,098 |
| rft | maj@16 (H100) | 79.1 [74.6, 83.2] | 40,197 |
| rft | maj@32 (H100) | 80.7 [76.3, 85.0] | 80,394 |
| rft | verifier-weighted vote @1 (H100) | 61.4 | 2,512 |
| rft | verifier-weighted vote @2 (H100) | 68.4 | 5,025 |
| rft | verifier-weighted vote @4 (H100) | 73.4 | 10,049 |
| rft | verifier-weighted vote @8 (H100) | 77.4 | 20,098 |
| rft | verifier-weighted vote @16 (H100) | 79.0 | 40,197 |
| rft | verifier-weighted vote @32 (H100) | 80.0 | 80,394 |
| grpo | greedy, stop at 1,536 (laptop) | 63.7 [58.0, 69.0] | 1,278 |
| grpo | greedy, stop at 2,048 (laptop) | 64.7 [59.0, 70.3] | 1,482 |
| grpo | greedy, no stop (laptop) | 63.0 [57.3, 68.7] | 2,017 |
| grpo | maj@1 (H100) | 64.2 [60.0, 68.4] | 1,883 |
| grpo | maj@2 (H100) | 66.1 [62.1, 69.9] | 3,767 |
| grpo | maj@4 (H100) | 73.6 [69.4, 77.8] | 7,534 |
| grpo | maj@8 (H100) | 76.9 [72.3, 81.0] | 15,068 |
| grpo | maj@16 (H100) | 79.8 [75.5, 83.7] | 30,136 |
| grpo | maj@32 (H100) | 81.0 [76.3, 85.3] | 60,272 |

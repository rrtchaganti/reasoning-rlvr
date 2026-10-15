# Phase 2 SFT data report

Source `open-r1/OpenR1-Math-220k` config `default` revision `e4e141ec9dea9f8326f4d347be56105859b2bd68`; tokenizer `Qwen/Qwen3-1.7B-Base`; cap 4096 tokens; built at commit `04779ecf5e0ecbed7586000d9012f98697fa1c1c`.

## Funnel

| stage | count |
|---|---|
| rows in source | 93733 |
| dropped: too_long | 34650 |
| dropped: no_correct_generation | 28627 |
| dropped: multiple_boxed | 5965 |
| dropped: duplicate_problem | 721 |
| dropped: no_boxed_in_summary | 513 |
| dropped: no_single_think_close | 188 |
| dropped: unbalanced_boxed | 7 |
| dropped: nested_think | 3 |
| dropped: empty_summary | 2 |
| kept before decontamination | 23057 |
| dropped: eval overlap (union) | 57 |
| **kept** | **23000** |

## Length (prompt + completion, Qwen3 tokens)

| set | n | mean | p50 | p90 | p99 | max | total |
|---|---|---|---|---|---|---|---|
| all parsed candidates (before the cap) | 57707 | 5985.4 | 4939 | 11796 | 16691 | 22266 | 345400988 |
| kept | 23000 | 2686.8 | 2707 | 3789 | 4066 | 4096 | 61796025 |

## Training-data overlap (EXP-004, D3 part a)

| eval set | problems | any 13-gram | 13-gram covering >= 50 % | fuzzy >= 0.9 | fuzzy only | **contaminated** (covered or fuzzy) | % | train items dropped (any hit) |
|---|---|---|---|---|---|---|---|---|
| gsm8k | 1319 | 0 | 0 | 0 | 0 | **0** | 0.0 | 0 |
| math500 | 500 | 6 | 2 | 2 | 0 | **3** | 0.6 | 31 |
| aime24 | 30 | 8 | 0 | 0 | 0 | **0** | 0.0 | 36 |
| gsm8k_perturbed | 830 | 0 | 0 | 0 | 0 | **0** | 0.0 | 0 |
| gsm8k_names_only | 516 | 0 | 0 | 0 | 0 | **0** | 0.0 | 0 |
| gsm8k_dev | 250 | 0 | 0 | 0 | 0 | **0** | 0.0 | 0 |
| math_dev | 250 | 9 | 2 | 1 | 0 | **2** | 0.8 | 37 |

A 13-gram hit whose shared windows cover less than half of the eval problem is a template sentence (AIME's 'relatively prime positive integers, find m + n'), not a copy; those items are still dropped from training but not counted as contamination. Per-eval JSON with the hit ids: `outputs/reports/phase-2-overlap/`.

## Sources and problem types kept

| source | kept |
|---|---|
| olympiads | 17491 |
| cn_contest | 3469 |
| aops_forum | 1448 |
| amc_aime | 397 |
| inequalities | 118 |
| number_theory | 55 |
| olympiads_ref | 22 |

| problem type | kept |
|---|---|
| Algebra | 10722 |
| Number Theory | 3419 |
| Geometry | 3231 |
| Combinatorics | 2438 |
| Calculus | 1382 |
| Logic and Puzzles | 1132 |
| Inequalities | 456 |
| Other | 220 |

## Answer check

On a seeded sample of 2000 kept examples, our extracted `<answer>` value agrees with the source's `answer` field under math-verify in 1997 cases (99.8 %); disagreements are listed in the manifest for reading by eye.

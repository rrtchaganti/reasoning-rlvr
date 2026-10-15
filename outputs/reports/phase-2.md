# Phase 2 report: SFT on distilled traces

Branch `phase-2-sft`, started 15 Oct 2026. Sections 1-2 come from the local CPU work and are final;
sections 3-7 are filled in from the rental session (EXP-002 deferred from Phase 1, EXP-005 to EXP-008).
Numbers are greedy, batch-invariant vLLM, 95 % bootstrap CI over problems unless stated.

## 1. Training data (EXP-004 and the data report)

Built by `rlvr-build-sft` from `open-r1/OpenR1-Math-220k` (`default`, revision e4e141ec) at commit
04779ec; full funnel and tables in `phase-2-data.md`.

| stage | count |
|---|---|
| rows in the source | 93,733 |
| no math-verify-correct trace | 28,627 |
| trace longer than 4096 tokens (prompt + completion) | 34,650 |
| multi-part (several boxes) or malformed trace | 6,678 |
| duplicate problem | 721 |
| overlap with an eval set or the dev split | 57 |
| **kept** | **23,000** (61.8 M tokens; mean 2,689, p90 3,790 tokens; hash `45b3fa4862b11e2a`) |

Sources kept: olympiads 76 %, cn_contest 15 %, aops_forum 6 %, amc_aime 2 %. Problem types: algebra
47 %, number theory 15 %, geometry 14 %, combinatorics 11 %, calculus 6 %. The extracted `<answer>`
value agrees with the source gold under math-verify on 99.8 % of a 2,000-example sample; the
disagreements are notation (`\dfrac` vs `\frac`, a missing `+C`).

**Training-data overlap (D3 part a).** Contaminated = a 13-gram hit covering at least half of the
eval problem, or a fuzzy match at similarity 0.9 or above; every hit item was dropped from training
whether or not it counts.

| eval set | problems | any 13-gram | covering ≥ 50 % | fuzzy | **contaminated** | train items dropped |
|---|---|---|---|---|---|---|
| GSM8K-test | 1319 | 0 | 0 | 0 | **0** | 0 |
| GSM8K perturbed / names-only / dev | 830 / 516 / 250 | 0 | 0 | 0 | **0** | 0 |
| MATH-500 | 500 | 6 | 2 | 2 | **3 (0.6 %)** | 31 |
| MATH-dev | 250 | 9 | 2 | 1 | **2 (0.8 %)** | 37 |
| AIME24 | 30 | 8 | 0 | 0 | **0** | 36 |

The eight AIME24 any-gram hits are one template sentence ("where m and n are relatively prime positive
integers, find m + n", in several variants), not copies; the first two builds would have reported them
as 27 % contamination (journal, section 7). The three MATH-500 problems are intermediate_algebra 90
(verbatim), 986 (paraphrase at 0.91) and 1365. GSM8K is absent from the SFT source, so every GSM8K row
after SFT measures transfer from olympiad-style data, not recall.

## 2. Compute so far

Local CPU: four data builds (25 + 7 + 5 + 5 min). Local GPU (RTX 4060, USD 0): trainer tests on a tiny
model, two 0.6B LoRA smokes with their 100-problem evals (0.4 h), and the base think-tag reference at
the 4096 window on all five eval sets (3.9 h; 5.3 M generated tokens). The rental budget set by the data: about 15 A100-hours for the whole session
(section 8 of the engineering notes).

## 3. Deferred from Phase 1: EXP-002, the base model's pass@k ceiling *(after the rental)*

## 4. EXP-005: learning-rate sweep on the dev split *(after the rental)*

## 5. EXP-006: base vs SFT, paired *(after the rental)*

Think-tag at the 4096-token window for both (4096 minus the prompt, 3,865-3,994 new tokens, the same
cap per problem on both sides; EXP-006 amendment d); 4-shot-CoT and zero-shot rows as well; MATH-500 by
level; perturbation gap and partial-prompt completion on the SFT checkpoint; pass@k and maj@k curves.

The base side of the pair, run locally on 15 Oct (`outputs/eval/base_1.7b_think_tag_4096/`, 3 h 55 min
on the 4060), next to the Phase 1 row at 1024 new tokens:

| eval | n | base think-tag, 1024 (Phase 1) | base think-tag, 4096 window | truncated (4096) | tag format |
|---|---|---|---|---|---|
| GSM8K | 1319 | 61.3 [58.7, 63.8] | 61.8 [59.2, 64.4] | 35.8 % | 0 % |
| GSM8K perturbed | 830 | 50.8 [47.6, 54.5] | 51.4 [48.1, 54.8] | 35.8 % | 0 % |
| MATH-500 | 500 | 43.0 [38.8, 47.4] | 44.8 [40.4, 49.2] | 20.0 % | 0 % |
| Countdown | 500 | 1.2 [0.4, 2.2] | 1.2 [0.4, 2.2] | 65.6 % | 0 % |
| AIME24 | 30 | 6.7 [0.0, 16.7] | 6.7 [0.0, 16.7] | 30.0 % | 0 % |

Every GSM8K output truncated at 1024 was still running at the 4096 window (472 of 472; perturbed 297 of
297): the base's think-tag loops do not end, so its truncation rate is a property of the model, not of
the budget. The tag-format rate is 0 because the base cannot emit `</think>` (its embedding row is
untrained; engineering notes, section 5). This row is "the base reading an untrained token"; the fair
base comparison for the SFT numbers is the 4-shot-CoT row (GSM8K 74.5, MATH-500 41.4; Phase 1 report).

## 6. EXP-007 (LoRA vs full) and EXP-008 (data scaling) *(after the rental)*

## 7. What Phase 3 needs from here

The `sft` checkpoint (HF link after upload), the training prompts with their overlap drops
(`outputs/data/sft/train.jsonl`, regenerated from the manifest), the dev split, and the paired
comparison script for every later checkpoint.

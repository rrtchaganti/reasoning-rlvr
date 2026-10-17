# Phase 3 plan: rejection-sampling fine-tuning (written 18 Oct 2026, for sign-off before any code)

PLAN.md Phase 3 in one sentence: from the `sft` checkpoint, sample 8 answers per training problem,
keep the correct ones (at most 2 per problem), fine-tune once more, and measure what selection alone
buys against SFT. Side product: the per-problem solve-rate file that Phase 4 uses to pick learnable
prompts. Budget: about 3 A100-hours (PLAN.md section 8).

This document is the plan to edit before building. Decisions that are yours are marked **[decide]**;
everything else follows the plan and the Phase 2 code.

## 1. What already exists and is reused unchanged

- Sampling: `rlvr-eval` with `--template think_tag`, `--n 8`, `--temperature 1.0`; it already loads
  `gsm8k_train` (7,473) and `math_train` (7,500, with `meta.level`), writes one `samples.jsonl` per
  eval with `problem_id`, `text`, `correct`, `num_tokens`, `finish_reason`, `extracted`, `meta`.
- The training-example format: `rlvr.data.sft_data.sft_prompt` is byte-identical to the think-tag
  eval prompt, `sft_completion`-style completions end with `</think>` and `<answer> x </answer>`,
  and `reject_reason` holds the format rules (one answer, closed think block, token cap).
- The trainer: `rlvr-sft` takes any `model` path, any `data`/`manifest` pair (hash checked), LR,
  epochs; fp32 master weights under bf16 autocast; tag-row handling (the `sft` checkpoint has
  trained rows, so the initialiser reports `untrained: false` and does nothing).
- Decontamination: `rlvr.data.sft_data.decontaminate` against every eval set (13-gram coverage
  plus fuzzy), with the per-eval JSON report.
- Evaluation and comparison: `configs/eval/sft_1.7b_{think_tag,few_shot_cot,zero_shot}.yaml` with
  `--model`/`--out`, `compare_checkpoints.py` (paired, MATH by level; `--all-samples` for pass@k),
  `perturbation_gap.py`, `rlvr-probe-partial`, `configs/sample/sft_1.7b_n64*.yaml`,
  `plot_pass_at_k.py`, `push_to_hub.py`, `project_sampling.py` for the ten-minute rule.

## 2. Pipeline (the `phase3_rental.sh` steps, resumable like Phase 2's)

1. `sample`: `rlvr-eval --config configs/sample/sft_1.7b_rft_n8.yaml`
   (model `outputs/train/sft_1p7b/final`, evals `gsm8k_train math_train`, think-tag, n = 8,
   T = 1.0, top-p 1.0, seed 0, `max_tokens` **[decide, section 6]**, `max_model_len` 6144,
   batch-invariant off, out `outputs/samples/sft_1.7b_rft_n8/`). Two new harness flags:
   `--levels 1,2,3` (keeps problems whose `meta.level` is in the set; only evals with levels) and
   `--exclude-dev` (drops the 250 + 250 dev problems, which are drawn from these train splits with
   seed 20261015 and must never be trained on). Prompt pool after both: about 7,220 GSM8K-train and
   about 4,000 MATH-train levels 1-3 (the exact counts are printed and go in the report).
2. `data`: `rlvr-build-rft` (new, `src/rlvr/data/rft_data.py`). Reads the two `samples.jsonl`,
   groups by problem, and writes:
   - `outputs/data/rft/solve_rate.jsonl`: one line per problem `{id, source, level, n, c, rate,
     mean_tokens, truncated}`; this is the Phase 4 prompt-filter input, so its schema is fixed here.
   - `outputs/reports/figures/phase-3-solve-rate.png` and the counts of rate = 0, rate = 1, in
     between, per source and per MATH level.
   - `outputs/data/rft/train.jsonl` in the SFT schema (`id, source, problem_type, prompt,
     completion, answer, n_tokens`): per problem the correct samples that pass `reject_reason`,
     deduplicated on the normalised reasoning text (Yuan et al.: distinct paths, not raw count),
     the **shortest two** kept; problems with rate 0 contribute nothing. The prompt is rebuilt with
     `sft_prompt(question)` from the loader (the samples file carries the id, not the question).
   - decontamination of the kept prompts against every eval set with the Phase 2 code and the same
     coverage rule; dropped items counted in the report.
   - `outputs/data/rft/manifest.json` (content hash, counts, solve-rate summary, source commit,
     sampling config hash) and `outputs/reports/phase-3-data.md`. The trainer checks this manifest
     (`manifest: outputs/data/rft/manifest.json` in the config); the laptop session copies it to
     `data/manifests/rft_iter1.json` after the run for the record. The committed-manifest guard of
     b34c25b is not needed because the manifest is built on the box, once.
3. `train`: `rlvr-sft --config configs/train/rft_1p7b.yaml` (`model: outputs/train/sft_1p7b/final`,
   data and manifest above, LR 5e-6 cosine, 1 epoch, effective batch 64, `max_length` 4096,
   fp32 master weights, no packing on E2E, `out: outputs/train/rft_1p7b`).
4. `eval`: the three templates at the Phase 2 windows (`--model outputs/train/rft_1p7b/final`),
   the perturbation gap and the partial-prompt probe on the checkpoint, paired comparisons rft minus
   sft and rft minus base (`compare_checkpoints.py`).
5. `sample64`: `configs/sample/sft_1.7b_n64.yaml` and the Countdown one with `--model`/`--out`
   (`outputs/samples/rft_1.7b_n64_*`): the pass@k curves and the paired pass@k against the SFT
   samples (`--all-samples`), the per-problem "new capability vs sharpening" counts that Phase 4
   reuses. Countdown at n = 64 carries the same cost warning as EXP-002 (handoff, section 1).
6. `report`: tables, figures, model card, Hub upload of `rft` and the sample sets.

Optional, only if the hours allow **[decide]**: `iter2` repeats steps 1-3 from `rft` (the
diminishing-return row of the plan), and `control` runs the compute-matched SFT control of section 5.

## 3. Code to write (all unit-tested on fake samples; one local GPU smoke on the 0.6B LoRA checkpoint)

| piece | file | tests |
|---|---|---|
| `--levels`, `--exclude-dev` in the harness; dev ids exposed by `datasets.py` | `src/rlvr/eval/run_eval.py`, `src/rlvr/eval/datasets.py` | `tests/test_eval_datasets_prompts.py` (+2) |
| RFT builder: solve rates, selection (correct, format-checked, distinct, shortest two), prompt rebuild, decontamination, manifest, report, histogram | `src/rlvr/data/rft_data.py` (`rlvr-build-rft`) | `tests/test_rft_data.py` (about 10) |
| Learnable-prompt filter for Phase 4 (reads `solve_rate.jsonl`, keeps 0 < rate < 1, writes the prompt file with its own hash) | same module, `rlvr-build-rft --learnable-out` | same file |
| Session script with the six steps above, resumable | `scripts/phase3_rental.sh` | `tests/test_phase3_scripts.py` (parse only) |
| Configs | `configs/sample/sft_1.7b_rft_n8.yaml`, `configs/train/rft_1p7b.yaml`, `configs/sample/rft_1.7b_n64*.yaml` (or `--model/--out` on the SFT ones; the former keeps `run_info` self-describing) | config-load tests |
| Docs | `docs/phase-3-journal.md` (from day one), `docs/phase-3-engineering-notes.md` (at the end), `outputs/reports/phase-3.md` skeleton, rental guide section for Phase 3 | - |

Local GPU smoke before the rental (about 15 min on the 4060): sample n = 8 on 50 GSM8K-train
problems from `outputs/train/sft_0p6b_lora_smoke_v2/final`, build, train 5 steps, run the eval on
20 problems. This checks every file hand-off, not any number.

## 4. Pre-registration (goes into `experiments.md` once the branch is on top of the merged Phase 2)

**EXP-009 Solve-rate distribution of the SFT policy on its training prompts.** Q5 input, Phase 4
prompt filter. Hypothesis: GSM8K-train is mostly saturated for the SFT model at T = 1.0 (more than
half the problems at 8 of 8 correct, under 10 % at 0 of 8), MATH-train levels 1-3 are not (fewer
than a third at 8 of 8, 15-30 % at 0 of 8), so the learnable set (0 < rate < 1) is 2,000-4,000 MATH
problems and 1,500-3,000 GSM8K ones. Metric: the histogram per source and MATH level; the kept-set
size. Falsifier: more than half of MATH levels 1-3 at rate 0: the SFT policy cannot be improved by
selection on MATH and Phase 4's prompt set has to move to levels 1-2 or to GSM8K.

**EXP-010 RFT versus SFT, paired.** Q1 (the RFT term), Q5, Q7, Q9, Q10. Hypothesis: one iteration
of RFT from `sft` raises think-tag greedy pass@1 by 1-4 points on GSM8K and 1-3 on MATH-500 (paired
CI excluding 0 on at least one), concentrated on MATH levels 1-3 (the levels it trained on); pass@64
moves by less than 2 points (selection sharpens, it does not add capability); Countdown is flat
within its CI; truncation and format-failure rates do not rise; the perturbation gap and the
partial-prompt completion rate stay within the Phase 2 values (the RFT data is the model's own
output on decontaminated prompts, so recall has no new entry point). Metric: paired bootstrap on
shared problems, MATH by level, pass@k and maj@k curves k <= 64 against the SFT samples with the
paired pass@k, the new-vs-sharpened problem counts. Falsifier: rft minus sft at or below 0 on both
GSM8K and MATH-500 with CIs excluding +1: selection alone does nothing at this scale and the Phase 4
story starts from `sft`, not `rft` **[decide: whether Phase 4 then trains from sft]**. Config:
`configs/sample/sft_1.7b_rft_n8.yaml`, `configs/train/rft_1p7b.yaml`; rented A100; about 3 h.

**EXP-010b (optional) Compute-matched SFT control.** PLAN.md section 4.4 asks for SFT vs RFT at
equal GPU-hours. RFT costs sampling plus one epoch on about 10k examples; the cheapest fair control
is one more epoch of SFT from `sft` on a fresh seeded 10k-example slice of the OpenR1 set at the same
LR (about 1 A100-hour). Hypothesis: the control gains less than RFT on MATH-500 (the on-policy
data is the point). **[decide: run it, or report RFT at best effort only]**

**EXP-011 (optional) Second iteration.** From `rft`, same recipe; hypothesis: under half the gain
of the first iteration (ReST-EM's plateau). **[decide]**

## 5. Compute (A100 80 GB, INR 189/h)

| step | tokens | hours | note |
|---|---|---|---|
| sample n = 8 on about 11,200 prompts | 90k sequences x 300-600 tokens = 27-54 M | 1.5-3 | depends on how long the SFT model's think blocks are; the ten-minute rule applies |
| build | - | 0.1 | CPU |
| train 1 epoch on about 10k examples | about 5-10 M | 0.5-1 | no packing on E2E |
| evals (3 templates, gap, probe) | 3 M | 0.7 | as Phase 2's `main` evals |
| sample64 (four sets) | 34 M | 1.5-2 | the EXP-002 figure, A100 rate |
| sample64 Countdown | up to 87 M | 2-8 | only if the SFT Countdown samples were affordable |
| total without Countdown | | 4.5-7 | PLAN.md said 3; the n = 64 evals are the difference |

The prompt pool is the lever: `--limit-per-eval 4000` (seeded) halves the sampling hours and still
leaves about 2,000 learnable MATH prompts **[decide: full pool or 4,000 per source]**.

## 6. Decisions for you

1. **Branch.** Phase 2 is not merged, so `phase-3-rft` was opened from the `phase-2-sft` tip
   (commit 5ca7bb1), not from `main`. When Phase 2 merges, this branch is rebased onto `main`
   (WORKFLOW.md allows the rebase when `main` has moved). The alternative is to hold all Phase 3
   code until the merge; that costs the A100 session its chance to run Phase 3 right after Phase 2.
2. **Sampling budget for RFT**: `max_tokens` 4096 (PHASES_IO) or 2048 (what Phase 4 uses for its
   own sampling, and half the cost). The SFT traces are long (kept-set p90 3,789 tokens under the
   4096 cap, `data/manifests/sft_openr1.json`), so the SFT model may write long think blocks and
   nobody knows its length distribution until the Phase 2 `main` evals report it. My recommendation
   is a rule rather than a number: read mean and p90 of the SFT think-tag eval; 2048 if the p90 is
   under 2,048 (consistent with Phase 4, half the cost), otherwise 4096 with the pool capped
   (decision 3). The config carries 2048 as the default and the session script prints the rule.
3. **Prompt pool**: full (about 11,200) or capped at 4,000 per source (section 5).
4. **Dev exclusion** (`--exclude-dev`): I will do this without asking; the dev split picks the LR
   and the checkpoint, and training on it is the kind of leak the plan's section 4 exists to prevent.
5. **Optional runs**: EXP-010b (compute-matched control), EXP-011 (second iteration).
6. **If the EXP-010 falsifier fires**: Phase 4 from `sft` or still from `rft`.

## 7. Order of work on the laptop (no GPU until the local smoke)

1. Harness flags and dev-id exposure, with tests.
2. `rft_data.py` and its tests; the learnable-prompt filter.
3. Configs, `phase3_rental.sh`, journal section 1.
4. Local smoke on the 0.6B checkpoint (your terminal, about 15 min).
5. Pre-registration text into `experiments.md`, README row, rental guide, handoff section for the
   A100 session (Phase 2 steps, then Phase 3 steps on the same node if the volume is still absent).

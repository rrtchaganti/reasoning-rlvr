# Phase 0 report: scaffold, harness, from-scratch references

Branch `phase-0-scaffold-harness`, 14 Oct 2026. Local GPU only (RTX 4060 Laptop, 8 GB, WSL2).
Model used for the smoke checks: `Qwen/Qwen3-0.6B-Base`. **None of the numbers below are headline
results**: they are 100-problem subsets, chosen to exercise the harness, on the small model. The
full-set baselines on Qwen3-1.7B-Base are Phase 1.

## Exit criteria (PLAN.md, Phase 0)

| Criterion | State |
|---|---|
| `pytest` green | 95 tests pass on CPU (`scripts/check.sh`); CI runs the same on every push |
| Harness produces a report for 0.6B-Base with CIs | yes: `report.md`, per-eval `metrics.json`, `samples.jsonl`, `manifest.json`, `run_info.json` |
| Three repeated greedy evals agree to within the reported variation | yes, after making vLLM batch-invariant mode the default (EXP-000 below): three runs are bit-identical |

## What was built

- `src/rlvr/ref/`: from-scratch, tested reference implementations: unbiased pass@k, group
  advantages with the std switch (Dr. GRPO), k1/k2/k3 KL estimators, the GRPO clipped loss with
  sequence / token / Dr. GRPO aggregation and clip-higher, and a KV-cache sampling loop.
- `src/rlvr/eval/`: eval sets with content-hash manifests (GSM8K, MATH-500, AIME24, Countdown
  generator and AST verifier, perturbed GSM8K loader), four prompt templates, an extraction chain
  that records which rule fired, math-verify grading with a numeric fallback, bootstrap and paired
  bootstrap CIs, maj@k, pass@k curves, vLLM and HF samplers, the `rlvr-eval` CLI.
- `src/rlvr/data/contamination.py`: 13-gram and fuzzy overlap probe.
- `src/rlvr/utils/run_info.py`: every run records commit, dirty flag, config hash, seed, package
  versions, GPU, timings; runs on a dirty tree are refused unless `--allow-dirty`.
- Tooling: `scripts/setup_env.sh` (uv, no sudo), `scripts/triton_check.py`, `scripts/check.sh`,
  `scripts/inspect_samples.py`, `scripts/phase0_smoke.sh`, CI workflow, PR template, lock file.

## EXP-000: do three identical greedy evals agree?

Config `configs/eval/phase0_smoke.yaml`: few-shot CoT (GSM8K, MATH-500) and the think-tag prompt
(Countdown), n = 1, temperature 0, seed 0, 1024 max tokens, 100 problems per eval.

**Default vLLM mode (commit 1366950).** pass@1 in %, 100 problems each, so one problem = one point.

| eval | run 1 | run 2 | run 3 | spread | problems whose *text* differed across runs |
|---|---|---|---|---|---|
| GSM8K | 68 | 70 | 68 | 2 | 16 / 100 |
| MATH-500 | 25 | 27 | 31 | 6 | 55 / 100 |
| Countdown | 1 | 2 | 2 | 1 | 69 / 100 |

The hypothesis (spread well under 0.5 points) is rejected. The completions themselves differ, in
some cases from the first character, so this is numerical non-determinism in batched bf16
decoding (reduction order depends on batch composition), not grading. On a 500-problem set the
same effect would be about a fifth of the size in percentage points, but it would still be there,
and it would be correlated with nothing we control.

**vLLM batch-invariant mode (`VLLM_BATCH_INVARIANT=1`), two runs, same commit.**

| eval | run 1 | run 2 | problems whose text differed |
|---|---|---|---|
| GSM8K | 68 | 68 | 0 / 100 |
| MATH-500 | 33 | 33 | 0 / 100 |
| Countdown | 2 | 2 | 0 / 100 |

Cost on this GPU: sampling 68 s per run instead of about 60 s; engine start-up about one minute
longer (the mode disables some kernel autotuning and uses fixed-split attention).

**Decision.** Batch-invariant mode is now the default in `VLLMSampler` (commit f4d1360); every
reported number is bit-reproducible given the commit, config and seed. `--no-batch-invariant`
exists for throughput-only work such as the n = 64 sampling in Phase 1, where the per-sample
randomness dominates anyway and the samples are stored.

**Final check on the committed default (commit f4d1360), three runs.**

| eval | run 1 | run 2 | run 3 | spread | problems whose text differed | trunc % | extract-fail % |
|---|---|---|---|---|---|---|---|
| GSM8K | 68 [59, 77] | 68 | 68 | 0 | 0 / 100 | 0 | 0 |
| MATH-500 | 33 [23, 41] | 33 | 33 | 0 | 0 / 100 | 5 | 14 |
| Countdown | 2 [0, 5] | 2 | 2 | 0 | 0 / 100 | 51 | 86 |

Brackets are 95 % bootstrap CIs over problems. Per run: 150 s wall clock, of which about 83 s is
engine start-up and 68 s sampling (recorded in each `run_info.json`).

Note that batch-invariant mode changes the numbers themselves (MATH-500 33 vs 25-31): it is a
different, fixed reduction order, not a "more correct" one. The point is reproducibility, not
accuracy. Comparisons must always be made within one mode.

## What the first smoke run caught (all fixed, with regression tests)

1. **MATH prompts ended on the wrong token.** The shots contain `Solution:` followed by a newline,
   which the Qwen3 tokeniser merges into one token; the prompt ended on a bare colon. The base
   model put 24-49 % probability on end-of-text and 53 % of completions were empty. Prompts now
   end on the shots' token. Extraction failures on MATH-500: 61 % -> 12-14 %.
2. **`<think>` as the last prompt token.** It is a single added token the base model has barely
   seen; output after it was garbage. The prompt now ends `Assistant:` newline `<think>` newline.
3. **The `</answer>` stop string was stripped from the output**, so the tag rule could never fire.
   Both samplers now keep the stop string.
4. Minor: `$228 000` was extracted as `228`; space-grouped thousands are now accepted.

Details and the probing method are in `FAILURES_AND_LEARNINGS.md` (14 Oct entries).

## Grader spot-check by eye

- GSM8K: 12 of 12 wrong samples read were genuine model errors (wrong arithmetic or wrong setup);
  the extracted number was the model's stated answer in every case.
- MATH-500: 12 wrong-but-extracted samples read, all genuine (`20` for 720, `7` for 17, `30°` for
  90°). Remaining extraction failures split into answers stated in prose without `\boxed{}`
  (about 10 of 100, e.g. "the largest angle is $2x=120$ degrees") and 1024-token repetition loops
  (about 8 of 100). Both are reported as extraction failures, not as wrong answers.
- Countdown: 23 extracted equations checked against the verifier; the one accepted equation is
  valid, every rejected one is wrong for a real reason (copied the `(1 + 2) / 3` example, reused a
  number, wrong value). The 0.6B base nearly never closes `</think>` and 45-51 % of completions run
  to the token limit, consistent with TinyZero's observation that sub-1B models do not do
  Countdown before RL.

## Compute

See `COMPUTE.md`: about 0.6 local GPU-hours for Phase 0 (one failed 16-minute start, one 4.5-minute
run before the fixes, eight smoke runs of 2-3.5 minutes, one probe). USD 0.

## What Phase 1 needs from here

- The full-set base-model rows on Qwen3-1.7B-Base (greedy, all templates) run locally with the
  committed default; n = 64 sampling goes to the rental.
- The perturbed-GSM8K generator and the partial-prompt completion probe (new code).
- A `fix/` branch if the MATH prose-answer extraction gap turns out to matter for the 1.7B model;
  for the 0.6B smoke it is reported, not patched.

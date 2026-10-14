# Experiment registry

Pre-registered before each run; closed with a conclusion after. One entry per run or per
family of runs that share a hypothesis. Config hashes come from `run_info.json`.

Template:

```
## EXP-NNN  <title>                                               [phase N] [status: planned|running|done]
**Question.** (link to PLAN.md section 6 Q-id if any)
**Hypothesis.** What we expect and why.
**Metric.** Exactly what number decides it, on which eval, with what CI.
**Falsifier.** What result would make us reject the hypothesis.
**Config.** path(s) + config hash(es), seed(s), commit.
**Result.** The number(s), with CI. Link to report / W&B.
**Conclusion.** "At this scale and budget, X; this does not tell us Y."
```

---

## EXP-000  Harness self-consistency: three identical greedy evals agree      [phase 0] [status: done]
**Question.** Section 4.1: what is the run-to-run variation of a temperature-0 vLLM eval?
**Hypothesis.** Greedy decoding with a fixed seed is deterministic up to vLLM batching effects;
pass@1 on GSM8K-test varies by well under 0.5 points across three identical runs.
**Metric.** pass@1 on GSM8K-test and MATH-500 for Qwen3-0.6B-Base, few_shot_cot, three runs.
Report max minus min.
**Falsifier.** Spread above 1 point: the harness is not reproducible and every later
comparison needs this noise added to its CI.
**Config.** `configs/eval/phase0_smoke.yaml`; seeds 0, 0, 0; default vLLM mode at commit 1366950, batch-invariant mode at commit f4d1360. Local RTX 4060, vllm 0.31.0, torch 2.13.0.
**Attempt 1 (commit 9a2edfa, discarded).** One run only. It found three harness bugs rather than measuring noise: MATH-500 61 % extraction failures (prompt ended on the wrong token boundary, model emitted end-of-text), Countdown 100 % failures (pre-filled `<think>` as the last token; stop string stripped). GSM8K 67.0 [58.0, 76.0] on its 100-problem subset looked sane and 12 of 12 inspected wrong samples were genuine model errors. Fixed in the following commit; see FAILURES_AND_LEARNINGS.md (14 Oct). Output kept locally as `run1_before_prompt_fixes`.
**Result.** Default vLLM mode (1366950), pass@1 % on 100-problem subsets: GSM8K 68 / 70 / 68 (spread 2), MATH-500 25 / 27 / 31 (spread 6), Countdown 1 / 2 / 2 (spread 1). The completion *text* differed across runs on 16 / 55 / 69 of 100 problems, sometimes from the first character. Hypothesis rejected: the falsifier (spread above 1 point) fired on two of three evals.
With `VLLM_BATCH_INVARIANT=1` (two runs at 1366950, then three runs at f4d1360 where it is the default): GSM8K 68 / 68 / 68, MATH-500 33 / 33 / 33, Countdown 2 / 2 / 2; text identical on 300 / 300 problems. Cost: sampling 68 s vs ~60 s per run, engine start-up ~80 s vs ~70 s. Report: `outputs/reports/phase-0.md`.
**Conclusion.** vLLM greedy decoding is not reproducible run-to-run in its default mode at this scale; on 100-problem subsets the noise is 1-6 points, which would swamp most ablation effects. Batch-invariant mode removes it entirely and is now the harness default, so every reported number is a function of (commit, config, seed) only. This does not tell us anything about accuracy: batch-invariant mode is a different fixed reduction order (MATH-500 33 vs 25-31 here), so comparisons are only valid within one mode.

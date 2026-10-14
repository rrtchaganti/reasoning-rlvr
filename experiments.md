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

## EXP-000  Harness self-consistency: three identical greedy evals agree      [phase 0] [status: planned]
**Question.** Section 4.1: what is the run-to-run variation of a temperature-0 vLLM eval?
**Hypothesis.** Greedy decoding with a fixed seed is deterministic up to vLLM batching effects;
pass@1 on GSM8K-test varies by well under 0.5 points across three identical runs.
**Metric.** pass@1 on GSM8K-test and MATH-500 for Qwen3-0.6B-Base, few_shot_cot, three runs.
Report max minus min.
**Falsifier.** Spread above 1 point: the harness is not reproducible and every later
comparison needs this noise added to its CI.
**Config.** `configs/eval/phase0_smoke.yaml`; seeds 0, 0, 0; commit TBD (after the prompt fixes below).
**Attempt 1 (commit 9a2edfa, discarded).** One run only. It found three harness bugs rather than measuring noise: MATH-500 61 % extraction failures (prompt ended on the wrong token boundary, model emitted end-of-text), Countdown 100 % failures (pre-filled `<think>` as the last token; stop string stripped). GSM8K 67.0 [58.0, 76.0] on its 100-problem subset looked sane and 12 of 12 inspected wrong samples were genuine model errors. Fixed in the following commit; see FAILURES_AND_LEARNINGS.md (14 Oct). Output kept locally as `run1_before_prompt_fixes`.
**Result.** TBD
**Conclusion.** TBD

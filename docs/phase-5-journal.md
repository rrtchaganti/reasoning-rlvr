# Phase 5 journal: the cost of termination (chronological)

Context first, then what happened in order. Results: `outputs/reports/phase-5.md`. Plan:
`docs/phase-5-plan.md`. Pre-registration: `experiments.md` EXP-016 to EXP-019, EXP-010i.

## 1. Context (21 Oct 2026, after the Phase 4 merge)

Phase 4 closed with one finding and two questions. The finding: in the trained format every
post-SFT gain is termination, and a longer cap, no KL penalty and a better-filled batch each bought
more or less of the same stopping (report sections 4.5, 12-14, 18). The questions: how cheaply can
stopping be bought, and were the prompts the limit. The budget was spent to within about INR 250 of
the ceiling, so the plan page (`docs/phase-5-plan.md`) put the free runs first and priced the rest
as options A to D. The user chose **A** ("go on with A. Start now!"): EXP-016 budget forcing on the
laptop's 4060 and EXP-017 accuracy against tokens from the existing sample sets. Branch
`phase-5-test-time-compute` from `main` 648df74.

## 2. What was built first (21 Oct, laptop, no GPU)

- **Budget forcing in the harness** (`src/rlvr/eval/force_stop.py`; `rlvr-eval --force-stop-at B
  --force-stop-continue N`). Design choice: a second generation pass rather than a custom sampler
  loop. The first pass runs at `max_tokens=B`; every completion with `finish_reason == "length"` and
  no `</think>` in its text gets the template's `force_suffix` ("\n</think>\n<answer>", the exact
  ending of every SFT trace and finished sample) appended to prompt + completion, and a second
  `generate` with `n=1`, `max_tokens=N` and the same stop strings writes the answer. The merged
  completion carries both parts, the continuation's finish reason and `forced=True`; the per-sample
  record gets `forced_stop` (default False so old files load), the summary the forced rate, the
  forced answers' accuracy and their finished rate, and `metrics.json` the pass's stats. The template
  guard: only `think_tag` defines a suffix; the few-shot templates raise. Tests on a scripted sampler
  (`tests/test_force_stop.py`): the targets, the continuation config, the merge, the stats, the
  grading of a forced answer, the CLI flags.
- **`scripts/plot_accuracy_vs_tokens.py`** for EXP-017: maj@k and pass@k per checkpoint against the
  expected generated tokens per problem (k x the problem's mean sample length), restricted to the
  problems every run shares, paired against a reference greedy run. maj@k per problem is averaged
  over 20 random k-subsets (the metrics module's `maj_at_k` returns only the mean, and the CI and
  the paired difference need the per-problem values). Test on synthetic sample files.
- **`scripts/phase5_laptop.sh`**: `smoke` (0.6B base, 8 problems, budget 64), `exp016` (sft and rft
  at budgets 1,536, 1,024, 2,048, 768, 3,072 on GSM8K + MATH-500, greedy think-tag, 4,096 window,
  batch-invariant vLLM, resumable), `exp016-ctrl` (the unforced runs on this machine, optional).

## 3. EXP-017 (21 Oct, laptop, no GPU)

The Phase 3 archive's sample folders for `sft`/`rft` n = 32 and the three 4-shot n = 64 sets hold
metrics only; the per-sample files were fetched from the Hub into `../hub_samples` (as for the
session 1 fold-in). Two figures: think-tag n = 32 on the pinned subsets (sft, rft, grpo, cap) and
4-shot CoT n = 64 (base, sft, rft), both paired against `grpo` greedy on the 300 problems every
run shares (the `rft` 4-shot set was sampled on 300-problem subsets). The result is in the report,
section 1: voting over two `rft` samples matches `grpo` greedy at 2.6x the tokens, four beat it at
5x; from k = 4 the maj@k curves of the four trained checkpoints coincide; the base model under
4-shot with 16 votes matches `grpo`'s greedy GSM8K number at 2.4x the tokens and never reaches its
MATH-500 number. Committed as 379b67a with the EXP-016 code.

## 4. EXP-016 on the laptop (21-22 Oct, overnight)

The user went to sleep and asked the session to run option A's remaining runs itself. Done
detached inside WSL (`setsid nohup`, so the job survives the session's shell; the earlier lesson
was that the session's own background shells get reaped on low host memory), with Windows'
hibernate-on-mains timeout set from 3 h to never for the run and restored afterwards (logged in
`GIT_ACTIONS_LOG.md`). No tracked file was edited while the sweep ran, because every `rlvr-eval`
call refuses a dirty tree at its start.

- 18:14 UTC smoke on the 0.6B base: 6 of 8 completions forced at budget 64, 5 finished within 32
  more tokens, tails exactly "\n</think>\n<answer> 498 </answer>"; EXIT=0.
- 18:20 UTC the sweep launched (`outputs/phase5_exp016.log`): the two checkpoints from the Hub,
  then the ten runs in the order 1,536, 1,024, 2,048, 768, 3,072 for sft then rft at each budget.
- The first launch pulled the SFT repo's ablation and control checkpoints too (vLLM's Hub loader
  fetches every `model.safetensors`); stopped, the script given a `download` step with the rental
  script's excludes (e3f7e81), relaunched. The link then measured 0.4-0.65 MB/s to the Hub
  (1.3 MB/s in general); the Xet backend was no faster; the sweep was made checkpoint-outer so
  the SFT half would finish while RFT downloaded (1cf82a5), and relaunched again about 02:30 UTC
  on the log's clock.
- The lid was closed overnight (the user's father); the laptop slept with the clock frozen just
  after the first eval started (03:01 UTC on the log) and resumed on the lid opening in the
  morning; the detached job continued with nothing lost. The RFT download was started by hand in
  parallel with the SFT evals once the link was faster (about 3 MB/s), so the sweep's own
  download step found it present.
- Results as they arrived (log times): `sft` 1,536 at 03:01 (80.7 / 59.6), 1,024 at 03:49 (79.8 /
  56.2), 2,048 at 04:21 (80.6 / 61.0), 768 at 05:27 (77.8 / 50.6), 3,072 at 05:52 (80.8 / 62.2);
  `rft` 1,536 at 07:40 (80.8 / 61.8), 1,024 at 08:22 (79.8 / 58.0), 2,048 at 08:52 (80.9 / 64.0),
  768 at 09:47 (78.2 / 51.6), 3,072 at 10:11 (80.9 / 64.6); EXIT=0 at 11:34 UTC. The Claude Code
  session restarted once during the sweep (its monitors were lost; the job was not).
- 11:34-12:30 UTC: the paired tables against `sft`, `rft`, `grpo` and the decompositions computed
  on the laptop; the figure script written with a test; report section 2 and 3, the registry, the
  narrative, the ledger, these notes; hibernate-on-mains restored to 3 h.

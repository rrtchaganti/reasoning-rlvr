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

## 5. Phase 5B and Phase 6, overnight and the next day (22-23 Oct 2026)

**Context.** After the evidence index was merged the user resumed the paused plan
(`docs/plan-after-phase-4.md`) and, past midnight, asked for all its runs to go overnight. The new code
was written in a second worktree (`../rlvr_dev`, branch `phase-5b-dev`) while EXP-020 ran in the main
checkout with the existing scripts, so no tracked file changed under a running job: the "Wait" pass
in the harness, the reward model (`python -m rlvr.train.orm build | train | score`), best-of-k and
weighted vote in the chart script, `scripts/phase5b_laptop.sh`, and the trace analysis. A detached
chain waited for EXP-020 to finish, fast-forwarded the checkout and ran the rest.

**The night (UTC).**
- 18:55-00:57 EXP-020: the GRPO checkpoint forced at 1,536 and 2,048 (82.5 / 62.6 and 82.5 / 63.6,
  flat against its unforced 82.3 / 63.4), then the unforced `sft` and `rft` on the laptop (76.0 /
  52.4 and 78.9 / 57.0), each about two hours because they loop to the window.
- 00:57-03:00 EXP-021: "Wait" before the stop, `sft` 80.8 / 61.2, `rft` 80.9 / 63.4, the same as the
  plain stop at 2,048.
- 19:07 (22 Oct) EXP-023 ran on the laptop's CPU from the archives while the GPU worked.
- 03:00 EXP-022: 20,000 training examples built in under a minute; training 03:01-09:22, 16-36 s a
  step (batches padded to their longest example); held-out 87.9 %, AUROC 0.938 at 09:31.

**The morning.** The trainer froze in PEFT's save: a Hub lookup's DNS step blocked in the WSL kernel
(netlink dump, state D), with the weights only in memory. A Wi-Fi toggle and ten minutes did not
free it, `wsl --shutdown` waited on it, and the user stopped the VM's worker and restarted the WSL
and Hyper-V compute services from an elevated shell. WSL rebooted at 11:29 UTC. The fix (weights to
disk before evaluation and before any Hub call; offline steps; a fallback load from the plain state
dict, tested) went in as 3bd2898 and the reward model was rerun from 11:30.

**The second day (UTC).**
- 11:30-17:26 EXP-022 rerun: training 5.8 h (faster than the first run's 6.4 h on the same
  batches), held-out 89.4 %, AUROC 0.944; the weights were on disk before the held-out pass.
- 17:26-22:05 scoring of the six n = 32 sample sets (`sft`, `rft`, `grpo` on GSM8K and MATH-500),
  38-55 minutes each; the chart at 22:06.
- 22:07 the GRPO same-machine control failed at once: the main checkout was dirty (the user's
  one-line edit to `METHODS_AND_METRICS.md`, left untouched) and the harness refuses a dirty tree.
  A worktree did not help either (WSL's git cannot follow the worktree's Windows path), so the run
  went to a fresh WSL clone at the same commit and started at 22:09.
- While it ran, EXP-022 was folded in: best-of-k, the reward-weighted vote and majority vote on
  the same seeded draws; a `best_of_k` rule in the evidence index with two new claims, both
  re-derived.

**What the results changed.** The laptop control runs showed a machine term of 15-23 % flipped
problems, so EXP-016 was re-paired on one machine: the stop rule's gain on SFT is +4.6 / +7.2 (first
quoted +5.8 / +9.0), still above RFT's, and the "cost of the rule" first reported was mostly that
machine term, because on one machine a forced decode is identical to the unforced one below the
budget. EXP-020's GRPO comparison is cross-machine for the same reason; its same-machine control was
queued after the reward model. The trace analysis needed one measure added after its first run (the
answer by the end of the first cycle), because under zero-shot the whole solution is the repeating
unit; both measures are reported.

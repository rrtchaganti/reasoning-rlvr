# reasoning-rlvr

A small, fully reproducible study of the post-training recipe for mathematical reasoning (SFT, rejection sampling, GRPO with verifiable rewards, DPO, test-time compute, distillation) on one base model, Qwen3-1.7B-Base, with confidence intervals, paired comparisons, contamination probes and a compute ledger.

Plan: [`PLAN.md`](PLAN.md). Process: [`WORKFLOW.md`](WORKFLOW.md). Engineering notes per phase (what was built, how it is tested, why): [`docs/`](docs/README.md). Background: [`METHODS_AND_METRICS.md`](METHODS_AND_METRICS.md), [`GLOSSARY.md`](GLOSSARY.md), [`PHASES_IO_AND_HYPERPARAMS.md`](PHASES_IO_AND_HYPERPARAMS.md).

## Status

| Phase | Branch | State | Report |
|---|---|---|---|
| 0 Scaffold, harness, from-scratch references | `phase-0-scaffold-harness` | merged 14 Oct 2026, tag `phase-0` | [`outputs/reports/phase-0.md`](outputs/reports/phase-0.md) |
| 1 Baselines and pass@k ceiling | `phase-1-baselines` | local part done 14 Oct 2026 (EXP-001, 003, 003b); EXP-002 four of five sets done 17 Oct 2026 on a rented L4 (report addendum; base pass@64 98.8 GSM8K, 87.8 MATH-500); Countdown half runs in the A100 session | [`outputs/reports/phase-1.md`](outputs/reports/phase-1.md) |
| 2 SFT | `phase-2-sft` | code complete 15 Oct 2026 (data built: 23,000 traces, hash 45b3fa48; EXP-004 closed); first E2E session 17 Oct (environment, data, EXP-002); training runs wait for the A100 session | [`outputs/reports/phase-2.md`](outputs/reports/phase-2.md) |
| 3 RFT | `phase-3-rft` | code ready 18 Oct 2026 (sampling-pool flags, `rlvr-build-rft`, session script, EXP-009/010/010b pre-registered); runs on the A100 right after Phase 2 | (after the run) |
| 4 GRPO + ablations | | | |
| 5 DPO | | | |
| 6 Test-time compute | | | |
| 7 Distil, quantise, serve | | | |
| 8 Write-up | | | |

## Results

Each number carries a 95 % bootstrap CI over problems and the commit that produced it; greedy, batch-invariant vLLM.

**Base models (Phase 1, commit ffd3018).** pass@1 %, full eval sets.

| model | prompt | GSM8K | GSM8K perturbed | MATH-500 | Countdown |
|---|---|---|---|---|---|
| Qwen3-1.7B-Base | 4-shot-CoT | 74.5 [72.0, 77.0] | 67.8 [64.8, 71.1] | 41.4 [37.0, 45.8] | - |
| Qwen3-1.7B-Base | zero-shot | 77.6 [75.3, 79.9] | 70.0 [67.0, 73.1] | 53.6 [49.4, 57.6] | - |
| Qwen3-1.7B-Base | think-tag | 61.3 [58.7, 63.8] | 50.8 [47.6, 54.5] | 43.0 [38.8, 47.4] | 1.2 [0.4, 2.2] |
| Qwen3-0.6B-Base | 4-shot-CoT | 61.9 [59.3, 64.6] | 56.5 [53.1, 59.9] | 28.0 [24.0, 31.8] | - |
| Qwen3-0.6B-Base | zero-shot | 64.9 [62.3, 67.5] | 59.5 [56.1, 62.9] | 37.6 [33.4, 41.6] | - |

Calibration: the 4-shot-CoT rows match the Qwen3 technical report (75.44 / 59.59 on GSM8K) within 1-2 points. Perturbation gap on GSM8K (numbers resampled, paired on 830 problems): 10.1 [7.0, 13.3] for the 1.7B base; a names-only control moves nothing (-0.4 [-3.3, 2.5]), so the gap is arithmetic brittleness, not recall. Partial-prompt verbatim completion: GSM8K 1.1 %, MATH-500 11.6 % (long held-outs). Details: `outputs/reports/phase-1.md`.

Phase 0 produced one methodological result rather than a model result: in vLLM's default mode three
identical greedy evals of Qwen3-0.6B-Base differed by up to 6 points on 100-problem subsets because the
generated text itself changed between runs; with vLLM's batch-invariant mode (now the harness default)
they are bit-identical. Details in `outputs/reports/phase-0.md`, EXP-000 in `experiments.md`.

## Reproduce

```bash
# inside WSL2 Ubuntu or any Linux box with a CUDA GPU; no sudo needed
bash scripts/setup_env.sh            # uv + uv-managed Python 3.12 + vllm/trl + dev tools, writes requirements-lock.txt
source ~/venvs/rlvr/bin/activate
bash scripts/check.sh                # ruff + fast CPU tests (the pre-commit gate)
python scripts/triton_check.py       # GPU sanity: a Triton kernel compiles and runs
bash scripts/phase0_smoke.sh         # Phase 0 exit check on Qwen3-0.6B-Base
```

CPU-only (what CI does): `bash scripts/setup_env.sh --cpu` then `pytest -q -m "not gpu"`.

Per-phase commands are listed in `scripts/` as they are added.

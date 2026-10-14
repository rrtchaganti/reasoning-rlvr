# reasoning-rlvr

A small, fully reproducible study of the post-training recipe for mathematical reasoning (SFT, rejection sampling, GRPO with verifiable rewards, DPO, test-time compute, distillation) on one base model, Qwen3-1.7B-Base, with confidence intervals, paired comparisons, contamination probes and a compute ledger.

Plan: [`PLAN.md`](PLAN.md). Process: [`WORKFLOW.md`](WORKFLOW.md). Engineering notes per phase (what was built, how it is tested, why): [`docs/`](docs/README.md). Background: [`METHODS_AND_METRICS.md`](METHODS_AND_METRICS.md), [`GLOSSARY.md`](GLOSSARY.md), [`PHASES_IO_AND_HYPERPARAMS.md`](PHASES_IO_AND_HYPERPARAMS.md).

## Status

| Phase | Branch | State | Report |
|---|---|---|---|
| 0 Scaffold, harness, from-scratch references | `phase-0-scaffold-harness` | merged 14 Oct 2026, tag `phase-0` | [`outputs/reports/phase-0.md`](outputs/reports/phase-0.md) |
| 1 Baselines and pass@k ceiling | `phase-1-baselines` | in progress (started 14 Oct 2026); EXP-001..003 pre-registered | |
| 2 SFT | | | |
| 3 RFT | | | |
| 4 GRPO + ablations | | | |
| 5 DPO | | | |
| 6 Test-time compute | | | |
| 7 Distil, quantise, serve | | | |
| 8 Write-up | | | |

## Results

None yet. Tables appear here as phases merge; each number carries a 95 % CI and the tag of the commit that produced it.

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

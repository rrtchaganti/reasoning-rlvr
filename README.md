# reasoning-rlvr

A small, fully reproducible study of the post-training recipe for mathematical reasoning (SFT, rejection sampling, GRPO with verifiable rewards, DPO, test-time compute, distillation) on one base model, Qwen3-1.7B-Base, with confidence intervals, paired comparisons, contamination probes and a compute ledger.

Plan: [`PLAN.md`](PLAN.md). Process: [`WORKFLOW.md`](WORKFLOW.md). Background: [`METHODS_AND_METRICS.md`](METHODS_AND_METRICS.md), [`GLOSSARY.md`](GLOSSARY.md), [`PHASES_IO_AND_HYPERPARAMS.md`](PHASES_IO_AND_HYPERPARAMS.md).

## Status

| Phase | Branch | State | Report |
|---|---|---|---|
| 0 Scaffold, harness, from-scratch references | `phase-0-scaffold-harness` | in progress (started 14 Oct 2026) | |
| 1 Baselines and pass@k ceiling | | | |
| 2 SFT | | | |
| 3 RFT | | | |
| 4 GRPO + ablations | | | |
| 5 DPO | | | |
| 6 Test-time compute | | | |
| 7 Distil, quantise, serve | | | |
| 8 Write-up | | | |

## Results

None yet. Tables appear here as phases merge; each number carries a 95 % CI and the tag of the commit that produced it.

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

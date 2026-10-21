# reasoning-rlvr

A small, fully reproducible study of the post-training recipe for mathematical reasoning (SFT, rejection sampling, GRPO with verifiable rewards, DPO, test-time compute, distillation) on one base model, Qwen3-1.7B-Base, with confidence intervals, paired comparisons, contamination probes and a compute ledger.

Plan: [`PLAN.md`](PLAN.md). The chain of questions across phases: [`docs/research-narrative.md`](docs/research-narrative.md). Process: [`WORKFLOW.md`](WORKFLOW.md). Engineering notes per phase (what was built, how it is tested, why): [`docs/`](docs/README.md). Background: [`METHODS_AND_METRICS.md`](METHODS_AND_METRICS.md), [`GLOSSARY.md`](GLOSSARY.md), [`PHASES_IO_AND_HYPERPARAMS.md`](PHASES_IO_AND_HYPERPARAMS.md).

## Status

| Phase | Branch | State | Report |
|---|---|---|---|
| 0 Scaffold, harness, from-scratch references | `phase-0-scaffold-harness` | merged 14 Oct 2026, tag `phase-0` | [`outputs/reports/phase-0.md`](outputs/reports/phase-0.md) |
| 1 Baselines and pass@k ceiling | `phase-1-baselines` | merged 2026-10-14, tag `phase-1`; local part done 14 Oct 2026 (EXP-001, 003, 003b); EXP-002 complete 18 Oct 2026 (four sets on a rented L4, Countdown on the H100; report addendum: base pass@64 98.8 GSM8K, 87.8 MATH-500, 63.8 Countdown) | [`outputs/reports/phase-1.md`](outputs/reports/phase-1.md) |
| 2 SFT | `phase-2-sft` | merged 20 Oct 2026 (through `phase-3-rft`, PR #4), tag `phase-2`; runs done 18-19 Oct 2026 on a TIR H100 (EXP-005, 006, 008; EXP-007 LoRA cut for budget): SFT think-tag greedy 74.8 GSM8K (+13.0 paired), 50.6 MATH-500 (+5.8); zero-shot 82.2 / 63.2; pass@64 unchanged on GSM8K, +5.2 on MATH-500; a quarter of think-tag generations never terminate; checkpoint `Raviteja546/reasoning-rlvr-sft` | [`outputs/reports/phase-2.md`](outputs/reports/phase-2.md) |
| 3 RFT | `phase-3-rft` | merged 20 Oct 2026 (PR #4), tag `phase-3`; runs done 19 Oct 2026 on the same H100 (EXP-009, 010, 010b and four post-hoc runs): RFT beats SFT in the trained format by +3.3 GSM8K / +5.8 MATH-500 (paired; three seeds +4.6 / +4.3), halves the looping, no pass@k loss, no Countdown transfer; a token-matched control gains nothing; checkpoint `Raviteja546/reasoning-rlvr-rft` | [`outputs/reports/phase-3.md`](outputs/reports/phase-3.md) |
| 4 GRPO + ablations | `phase-4-grpo` | plan and pre-registration written 19 Oct 2026 (shape B: 32 x 8, 200 steps, main + seed + three ablations, two sessions, about 26 H100-hours; `docs/phase-4-plan.md`, EXP-010g/h, EXP-012, 013, 014a-c); code, preflight, session script and the reference-vs-TRL correctness check done 20 Oct 2026 (gradients agree to 1.4e-5 in fp32); ceiling raised to INR 22,000 the same day; **session 1 run 20-21 Oct 2026 on a TIR H100** (EXP-010g/h, 012, 013): GRPO beats RFT in the trained format on every set with n >= 500, paired (+4.1 GSM8K, +7.0 MATH-500, +7.0 Countdown); the whole paired gain sits on problems RFT truncated (truncation 15 -> 5 %, 40 -> 21 %; accuracy among problems both finished unchanged within CI); pass@k says sharpening on GSM8K and a small persistent gain on MATH-500; the seed repeat reproduces GSM8K and Countdown to within a point and lands 4.8 lower on MATH-500; checkpoint `Raviteja546/reasoning-rlvr-grpo`; next: session 2 (A1 KL off, A2 dynamic sampling, A3 Dr. GRPO) | [`outputs/reports/phase-4.md`](outputs/reports/phase-4.md) |
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

After training (Qwen3-1.7B-Base, greedy, 4,096-token window; paired differences against the previous checkpoint on shared problems; 18-19 Oct 2026, TIR H100):

| checkpoint | prompt | GSM8K | GSM8K perturbed | MATH-500 | Countdown | truncated GSM8K / MATH-500 |
|---|---|---|---|---|---|---|
| base | think-tag | 61.8 [59.2, 64.4] | 51.4 [48.1, 54.8] | 44.8 [40.4, 49.2] | 1.2 [0.4, 2.2] | 36 / 20 % |
| SFT (EXP-006) | think-tag | 74.8 (+13.0 [10.0, 15.8]) | 66.0 (+14.6) | 50.6 (+5.8 [0.6, 11.0]) | 3.8 (+2.6) | 28 / 52 % |
| RFT (EXP-010) | think-tag | 78.2 (+3.3 [1.0, 5.8]) | 69.0 (+3.0) | 56.4 (+5.8 [1.6, 10.2]) | 7.0 (+3.2) | 15 / 40 % |
| GRPO (EXP-012) | think-tag | 82.3 (+4.1 [2.0, 6.1]) | 73.9 (+4.8) | 63.4 (+7.0 [2.8, 11.0]) | 14.0 (+7.0 [3.6, 10.4]) | 5 / 21 % |
| base | zero-shot | 77.9 [75.6, 80.1] | 69.9 | 53.8 [49.4, 58.2] | - | 1-9 % |
| SFT | zero-shot | 82.2 (+4.3 [1.8, 6.6]) | 72.7 | 63.2 (+9.4 [5.0, 14.0]) | - | 17 / 59 % |
| RFT | zero-shot | 81.1 (-1.1) | 74.7 (+2.0) | 64.8 (+1.6) | - | |

pass@64 (4-shot-CoT, n = 64, T = 0.6): base 98.8 / 90.2 / 87.8 / 63.8 (Countdown, think-tag), SFT 98.2 / 88.8 / 93.0 / 62.0, RFT (300-problem subsets) 98.7 / 92.3 / 94.0. The trained format loses to a plain zero-shot prompt on its own checkpoints because a quarter (SFT) to a sixth (RFT) of its generations never terminate, at any cap. Details: `outputs/reports/phase-2.md`, `outputs/reports/phase-3.md`.

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

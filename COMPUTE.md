# Compute and cost ledger

One row per run. Hours are wall-clock on the stated GPU; cost is the provider's invoice line
(or hours × rate for local, at USD 0). Tokens generated and trained are from `run_info.json`
or W&B. Appended at the end of each session.

| Date | Phase | Run | GPU | Hours | USD | Tokens generated | Tokens trained | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-14 | 0 | smoke, failed engine start (Python.h) | RTX 4060 Laptop 8 GB | 0.27 | 0 | 0 | 0 | 16 min to the Triton compile error; no samples |
| 2026-10-14 | 0 | smoke run before prompt fixes (9a2edfa) | RTX 4060 Laptop 8 GB | 0.08 | 0 | ~0.06 M | 0 | discarded; found three harness bugs |
| 2026-10-14 | 0 | prompt probe (HF backend) | RTX 4060 Laptop 8 GB | 0.03 | 0 | <0.01 M | 0 | next-token distributions, 4 prompt variants |
| 2026-10-14 | 0 | EXP-000 default mode, 3 runs (1366950) | RTX 4060 Laptop 8 GB | 0.13 | 0 | ~0.26 M | 0 | 7 m 44 s; not reproducible |
| 2026-10-14 | 0 | EXP-000 batch-invariant, 2 runs (1366950) | RTX 4060 Laptop 8 GB | 0.11 | 0 | ~0.18 M | 0 | bit-identical |
| 2026-10-14 | 0 | EXP-000 final, 3 runs (f4d1360) | RTX 4060 Laptop 8 GB | 0.14 | 0 | ~0.27 M | 0 | 8 m 33 s; bit-identical; 150 s per run (83 s start-up) |
| 2026-10-14 | 1 | 1.7B few_shot_cot, lost run (memory guard) | RTX 4060 Laptop 8 GB | 0.30 | 0 | 0 | 0 | killed after engine init; weight download kept |
| 2026-10-14 | 1 | EXP-001 greedy baselines, 8 configs (ffd3018) | RTX 4060 Laptop 8 GB | 1.19 | 0 | ~4.9 M | 0 | 2679 problems x 8; think_tag 1.7B alone 26 min (long loops) |
| 2026-10-14 | 1 | EXP-003 partial-prompt probe (ffd3018) | RTX 4060 Laptop 8 GB | 0.04 | 0 | ~0.2 M | 0 | 1810 prefixes |
| 2026-10-14 | 1 | EXP-003b names-only control (585a586) | RTX 4060 Laptop 8 GB | 0.05 | 0 | ~0.08 M | 0 | 516 problems |
| 2026-10-15 | 2 | EXP-004 SFT data builds x4 + overlap (04779ec) | laptop CPU | 0 | 0 | 0 | 0 | 42 CPU-min total; 2.2 GB download once |
| 2026-10-15 | 2 | trainer GPU smoke, tiny model (d73e206) | RTX 4060 Laptop 8 GB | 0.01 | 0 | 0 | 0 | 2 steps, 18 s |
| 2026-10-15 | 2 | 0.6B LoRA smoke: train on 400 examples + 100-problem GSM8K think-tag eval (e84dba1, fa2f091) | RTX 4060 Laptop 8 GB | 0.17 | 0 | 0.16 M | 0.77 M | smoke, not a result; exposed the untrained think-token rows |
| 2026-10-15 | 2 | trainer GPU tests x4 (tag rows, LoRA merge, fp32 + autocast) | RTX 4060 Laptop 8 GB | 0.02 | 0 | 0 | 0 | tiny model |
| 2026-10-15 | 2 | 0.6B LoRA smoke v2 with the fixed trainer: train on 400 examples + 100-problem GSM8K think-tag eval (57625ea) | RTX 4060 Laptop 8 GB | 0.25 | 0 | 0.17 M | 0.77 M | smoke, not a result; tag rows initialised and trained, `</think>` emitted 5/100 (0 before); peak 4.48 GB |

| 2026-10-15 | 2 | EXP-006 paired reference: base 1.7B think-tag at the 4096 window, 5 evals (c994bde) | RTX 4060 Laptop 8 GB | 3.92 | 0 | 5.3 M | 0 | 372 output tok/s (KV cache holds ~6 sequences at 4096; 1,216 tok/s at 1024 in Phase 1); estimated 1 h, took 3 h 55 min; truncated outputs end at 3,865-3,994 tokens (window minus prompt, journal 14) |
| 2026-10-17 | 2 | E2E L4 node: environment (cu129 route), data build, model downloads, smoke eval (ade4f6a) | E2E L4 24 GB | 1.5 | INR 75 (USD 0.9) | <0.01 M | 0 | first E2E node, root disk, no volume; torch 2.13.0+cu129, vllm 0.31.0+cu129, trl 1.15.0 pinned; data step dirtied the tree (fixed in b34c25b) |
| 2026-10-17 | 1 (deferred) | EXP-002 four sets: base 1.7B n = 64, 4-shot-CoT on GSM8K, perturbed, MATH-500, AIME24 (ade4f6a) | E2E L4 24 GB | 5.16 | INR 253 (USD 3.0) | 34.3 M | 0 | output tok/s per set 3,134 / 3,082 / 1,171 / 964 (mean sample length 154 / 151 / 359 / 926); KV cache 15.3 GiB, 23 sequences at 6,144 |
| 2026-10-17 | 1 (deferred) | EXP-002 Countdown, stopped at 2,368 / 32,000 sequences (ade4f6a) | E2E L4 24 GB | 1.60 | INR 78 (USD 0.9) | ~4.7 M, discarded | 0 | 816 tok/s at about 2,700 tokens a sample, 21 h more projected; rerun on the A100 from zero |

Node row (pending the console figure): the L4 node existed from 17 Oct 2026 about 14:30 IST to its
termination on 18 Oct; the invoice bills every hour of that, including setup, the archive and idle
time, at INR 49/h. The GPU-time rows above sum to 8.3 h (INR 405); the node hours are the real cost.

**Running totals.** Rented: 8.26 GPU-h of runs, about INR 405 (USD 4.8); node-hours pending the
invoice. Local: 6.70 h.

**Budget.** USD 100-170 confirmed; estimate 53-55 A100-hours (PLAN.md section 8). First to drop
if over: the seed-repeat GRPO run, then the LoRA SFT run. Never the baselines or the CIs.

**Rates (E2E Networks, MyAccount on-demand, INR per hour, 16 Oct 2026).** L4 24 GB 49; A30 24 GB 90;
A40 48 GB 96; V100 32 GB 100; L40S 48 GB 102; A100 40 GB 179; A100 80 GB 189; H100 not offered. About
INR 84 per USD. A powered-off node still bills: sessions end with Terminate. Plan and per-phase
node choice: `docs/rental-setup.md`, section 2c. Rented rows below quote the invoice line in INR with
the USD conversion of the day.

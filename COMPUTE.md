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

**Running totals.** Rented: 0.0 h, USD 0. Local: 2.34 h.

**Budget.** USD 100-170 confirmed; estimate 53-55 A100-hours (PLAN.md section 8). First to drop
if over: the seed-repeat GRPO run, then the LoRA SFT run. Never the baselines or the CIs.

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

| 2026-10-17 | 2 | E2E L4 node, whole life (invoice: two consecutive lines, 13:28-16:01 INR 124.95 and 16:01-00:01 INR 392) | E2E L4 24 GB | 10.55 | INR 516.95 (USD 6.2) | - | - | the three rows above are inside this one; 2.3 h of it was setup, archive and idle time; terminated 18 Oct 00:01 IST |

The invoice line is the cost; the three run rows above break the same hours down and are not added
again. The console labels the plan "2 vCPU and 6 GB RAM"; the node had 25 vCPU and 100 GB. The usage
report of 19 Oct adds a last line for the same node, 00:01-00:10 IST, INR 7.35 (the minutes between
the last command and the Terminate): L4 total INR 524.30.

| 2026-10-16 | 2 | E2E MyAccount A30 node, 1.32 h, and its saved image `rlvr-env-v1-disk-0` (640 GB, INR 5 per GB-month, 4.1 GB-months billed so far); first provider attempt, nothing run | E2E A30 24 GB | 1.32 | INR 118.80 + 20.50 (USD 1.7) | 0 | 0 | from the usage report of 19 Oct 2026; the image keeps billing until deleted from MyAccount |
| 2026-10-18 | - | TIR dataset volume `rlvr-phase-2`, 250 GB, created 12:32 IST and deleted the same day | - | - | INR 10.20 | - | - | 1.02 GB-months at INR 10; not needed (the notebook disk is the volume) |

| 2026-10-18 | 2 | TIR H100: environment (cu129 route, 25 min), data build, model downloads, smoke (bb6f2a6) | TIR H100 SXM 80 GB | 0.60 | INR 227 (USD 2.7) | <0.01 M | 0 | torch 2.13.0+cu129, vllm 0.31.0+cu129, trl 1.15.0 per the lock; torchcodec removed (CUDA 13 wheel); no flash-attn in the Phase 2 part: every training run sdpa + packing=false |
| 2026-10-18 | 1 (deferred) | EXP-002 Countdown n = 64, base 1.7B think-tag, 500 x 64 (bb6f2a6) | TIR H100 SXM 80 GB | 1.57 | INR 595 (USD 7.1) | 62.3 M | 0 | 11,047 output tok/s (13.5x the L4); mean 1,946 tokens a sample, 43 % truncated; completes EXP-002 |
| 2026-10-18 | 2 | EXP-005 LR sweep: 3 runs of 36 steps on 2,300 examples + 3 dev evals (bb6f2a6) | TIR H100 SXM 80 GB | 0.73 | INR 277 (USD 3.3) | 5.0 M | 18.5 M | 16.5 s/step at 4 x 16, 33 GB; winner 1e-5 |
| 2026-10-18 | 2 | EXP-006 main SFT, 720 steps (3 h 02 m) + think-tag, 4-shot-CoT, zero-shot evals + gap + probe (bb6f2a6) | TIR H100 SXM 80 GB | 3.43 | INR 1,300 (USD 15.5) | 13.1 M | 123.6 M | 15.2 s/step at 16 x 4, 72.9 GB reserved (51.6 allocated), loss 0.413; W&B run dtahxhtt |
| 2026-10-18 | 2 | EXP-006 SFT n = 64, 4-shot-CoT, four sets (bb6f2a6) | TIR H100 SXM 80 GB | 3.77 | INR 1,429 (USD 17.0) | 165.5 M | 0 | 10.9-11.3k tok/s; sampled completions 2x the greedy length, 15-16 % truncated |
| 2026-10-18 | 2 | EXP-006 SFT Countdown n = 64, think-tag (bb6f2a6) | TIR H100 SXM 80 GB | 3.07 | INR 1,164 (USD 13.9) | 119.4 M | 0 | 79 % truncated, mean 3,731 tokens; pass@64 62.0 vs base 63.8 |
| 2026-10-19 | 2 | EXP-007 LoRA r = 64, stopped at step 20 of 720 (bb6f2a6) | TIR H100 SXM 80 GB | 0.20 | INR 76 (USD 0.9) | 0 | 3.4 M | 19.2 s/step (1.26x the full fine-tune), 51.7 GB; cut for budget, no checkpoint |
| 2026-10-19 | 2 | EXP-008 data scaling: 25 % (180 steps) + 50 % (360 steps) + two think-tag evals (bb6f2a6) | TIR H100 SXM 80 GB | 2.80 | INR 1,061 (USD 12.6) | 16.3 M | 92.6 M | 15.3 s/step at 16 x 4; W&B runs in project reasoning-rlvr |
| 2026-10-19 | 2 | Phase 2 report step: compare tables, figures, model card, Hub uploads, archive (bb6f2a6) | TIR H100 SXM 80 GB | 0.10 | INR 38 (USD 0.5) | 0 | 0 | `Raviteja546/reasoning-rlvr-sft` (main + ablations/), samples dataset, `archives/h100_phase2_archive.tar.gz` |
| 2026-10-19 | 2 | base 4-shot-CoT and zero-shot greedy at the SFT cap (4,095), for the paired Phase 2 tables (run in the Phase 3 part) | TIR H100 SXM 80 GB | 0.08 | INR 30 (USD 0.4) | 1.7 M | 0 | the Phase 1 rows used 1,024 tokens |
| 2026-10-18/19 | 2 | idle, debugging and the flash-attn build inside the Phase 2 part | TIR H100 SXM 80 GB | 1.75 | INR 663 (USD 7.9) | - | - | 27 idle min from a tmux name-prefix match, 40 from a stale W&B key, the YAML learning-rate string, the LoRA switch; the flash-attn wheel built on the CPU in 13 min while data50 trained |

Phase 2 part of the H100 session: 09:50 UTC 18 Oct to 03:35 UTC 19 Oct, 17.75 node-hours, about INR
6,730 (USD 80); the rows above are the breakdown (they sum to 18.1 h because the evals and the idle
minutes overlap at the step boundaries). The session continued into Phase 3 without a Stop; the
notebook's whole-life invoice line (28.1 h, INR 10,640) is recorded with the Phase 3 rows.

**Running totals (after Phase 2, 19 Oct 2026).** Rented: L4 10.55 node-hours (INR 524.30), A30
1.32 h (INR 118.80 + 20.50), the TIR dataset volume (INR 10.20), H100 Phase 2 part 17.75 h (about
INR 6,730): about INR 7,400 (USD 88). Local: 6.70 h. Budget used: about 52 % of the INR 14,300
(USD 170) ceiling after Phase 2; the whole session, Phase 3 included, brings the project to about
INR 11,450 (80 %) plus the notebook disk at about INR 112 a day while it exists.

**Budget.** USD 100-170 confirmed; estimate 53-55 A100-hours (PLAN.md section 8). First to drop
if over: the seed-repeat GRPO run, then the LoRA SFT run. Never the baselines or the CIs.

**Rates (E2E Networks, MyAccount on-demand, INR per hour, 16 Oct 2026).** L4 24 GB 49; A30 24 GB 90;
A40 48 GB 96; V100 32 GB 100; L40S 48 GB 102; A100 40 GB 179; A100 80 GB 189; H100 not offered. About
INR 84 per USD. A powered-off node still bills: sessions end with Terminate. Plan and per-phase
node choice: `docs/rental-setup.md`, section 2c. Rented rows below quote the invoice line in INR with
the USD conversion of the day.

**Rates from 18 Oct 2026 (E2E TIR, the managed notebook platform; `docs/rental-setup.md` 2d).** H100
SXM 80 GB plan INR 379.05 per hour (26 vCPU, 250 GB RAM), billed per minute while running; a stopped
notebook bills only its 300 GB workspace disk (per GB-month beyond 30 GB; the rate from the create
form goes here when the notebook is made). The node's price is exported as `RLVR_NODE_RATE`.
Projection: the Phase 2 + 3 session 15-18 H100-hours, INR 5,700-6,800; the whole plan 38-46
H100-hours plus disk, INR 15,000-18,000, above the ceiling unless the drop list in 2d applies.

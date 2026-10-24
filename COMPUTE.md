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

| Date | Phase | Run | GPU | Hours | USD | Tokens generated | Tokens trained | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-15 | 2 | EXP-006 paired reference: base 1.7B think-tag at the 4096 window, 5 evals (c994bde) | RTX 4060 Laptop 8 GB | 3.92 | 0 | 5.3 M | 0 | 372 output tok/s (KV cache holds ~6 sequences at 4096; 1,216 tok/s at 1024 in Phase 1); estimated 1 h, took 3 h 55 min; truncated outputs end at 3,865-3,994 tokens (window minus prompt, journal 14) |
| 2026-10-17 | 2 | E2E L4 node: environment (cu129 route), data build, model downloads, smoke eval (ade4f6a) | E2E L4 24 GB | 1.5 | INR 75 (USD 0.9) | <0.01 M | 0 | first E2E node, root disk, no volume; torch 2.13.0+cu129, vllm 0.31.0+cu129, trl 1.15.0 pinned; data step dirtied the tree (fixed in b34c25b) |
| 2026-10-17 | 1 (deferred) | EXP-002 four sets: base 1.7B n = 64, 4-shot-CoT on GSM8K, perturbed, MATH-500, AIME24 (ade4f6a) | E2E L4 24 GB | 5.16 | INR 253 (USD 3.0) | 34.3 M | 0 | output tok/s per set 3,134 / 3,082 / 1,171 / 964 (mean sample length 154 / 151 / 359 / 926); KV cache 15.3 GiB, 23 sequences at 6,144 |
| 2026-10-17 | 1 (deferred) | EXP-002 Countdown, stopped at 2,368 / 32,000 sequences (ade4f6a) | E2E L4 24 GB | 1.60 | INR 78 (USD 0.9) | ~4.7 M, discarded | 0 | 816 tok/s at about 2,700 tokens a sample, 21 h more projected; rerun on the A100 from zero |

| Date | Phase | Run | GPU | Hours | USD | Tokens generated | Tokens trained | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-17 | 2 | E2E L4 node, whole life (invoice: two consecutive lines, 13:28-16:01 INR 124.95 and 16:01-00:01 INR 392) | E2E L4 24 GB | 10.55 | INR 516.95 (USD 6.2) | - | - | the three rows above are inside this one; 2.3 h of it was setup, archive and idle time; terminated 18 Oct 00:01 IST |

The invoice line is the cost; the three run rows above break the same hours down and are not added
again. The console labels the plan "2 vCPU and 6 GB RAM"; the node had 25 vCPU and 100 GB. The usage
report of 19 Oct adds a last line for the same node, 00:01-00:10 IST, INR 7.35 (the minutes between
the last command and the Terminate): L4 total INR 524.30.

| Date | Phase | Run | GPU | Hours | USD | Tokens generated | Tokens trained | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-16 | 2 | E2E MyAccount A30 node, 1.32 h, and its saved image `rlvr-env-v1-disk-0` (640 GB, INR 5 per GB-month, 4.1 GB-months billed so far); first provider attempt, nothing run | E2E A30 24 GB | 1.32 | INR 118.80 + 20.50 (USD 1.7) | 0 | 0 | from the usage report of 19 Oct 2026; the image keeps billing until deleted from MyAccount |
| 2026-10-18 | - | TIR dataset volume `rlvr-phase-2`, 250 GB, created 12:32 IST and deleted the same day | - | - | INR 10.20 | - | - | 1.02 GB-months at INR 10; not needed (the notebook disk is the volume) |

| Date | Phase | Run | GPU | Hours | USD | Tokens generated | Tokens trained | Notes |
|---|---|---|---|---|---|---|---|---|
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

| Date | Phase | Run | GPU | Hours | USD | Tokens generated | Tokens trained | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-19 | 3 | Phase 3 smoke: first attempt at a 512-token cap found no correct trace; second at 2,048 passed (bb6f2a6) | TIR H100 SXM 80 GB | 0.17 | INR 64 (USD 0.8) | 0.3 M | 0.1 M | flash-attn self-test and the packed training path verified here |
| 2026-10-19 | 3 | EXP-009 sampling: SFT 1.7B think-tag, n = 8, T = 1.0, 7,388 prompts (MATH L1-3 3,388 + GSM8K-train 4,000) at a 2,048 cap (bb6f2a6) | TIR H100 SXM 80 GB | 1.31 | INR 497 (USD 5.9) | 95.6 M | 0 | 19.5-21.6k tok/s; the cap rule's 4,096 was lost to the flag bug (patch 0004, fixed 3e055ad) |
| 2026-10-19 | 3 | `rlvr-build-rft`: solve rates, kept set 11,145 (hash f44edf33a26c55fc), decontamination, learnable prompts (bb6f2a6) | CPU on the node | 0.02 | INR 8 | 0 | 0 | 25 s |
| 2026-10-19 | 3 | EXP-010 RFT training, 53 steps (bb6f2a6) | TIR H100 SXM 80 GB | 0.20 | INR 76 (USD 0.9) | 0 | 13.4 M | flash_attention_2 + packing, 4 x 16, 13.1 s/step, 32.9 GB, loss 0.427; W&B project reasoning-rlvr |
| 2026-10-19 | 3 | EXP-010 evals: think-tag, 4-shot-CoT, zero-shot, gap, probe (bb6f2a6) | TIR H100 SXM 80 GB | 0.37 | INR 140 (USD 1.7) | 12.3 M | 0 | 22 min |
| 2026-10-19 | 3 | Phase 3 report step: first attempt failed on the model card's base_model (patch 0005); manual card + upload | TIR H100 SXM 80 GB | 0.05 | INR 19 | 0 | 0 | `Raviteja546/reasoning-rlvr-rft` |
| 2026-10-19 | 3 | EXP-010b example-matched control: 132 steps + think-tag eval (bb6f2a6) | TIR H100 SXM 80 GB | 0.65 | INR 246 (USD 2.9) | 7.4 M | 29.8 M | 8 x 8 flash + packing, 11.8 s/step, 50.5 GB; the examples were a seen subset of the SFT data (deviation) |
| 2026-10-19 | 3 | EXP-010 n = 64 4-shot-CoT sampling of the RFT checkpoint on 300-problem subsets (bb6f2a6) | TIR H100 SXM 80 GB | 1.38 | INR 523 (USD 6.2) | 59.4 M | 0 | 11.3k tok/s; the user's call to subset (saves 2.5 h) |
| 2026-10-19 | 3 | gap analysis written on the node (GPU idle) | TIR H100 SXM 80 GB | 0.87 | INR 330 (USD 3.9) | - | - | 07:41-08:33 on the node's clock |
| 2026-10-19 | 3 | EXP-010c greedy think-tag at an 8,192 cap, SFT and RFT, subsets + Countdown (bb6f2a6) | TIR H100 SXM 80 GB | 0.47 | INR 178 (USD 2.1) | 8.8 M | 0 | |
| 2026-10-19 | 3 | EXP-010d RFT seeds 1 and 2: training + think-tag evals (bb6f2a6) | TIR H100 SXM 80 GB | 0.73 | INR 277 (USD 3.3) | 13.0 M | 26.9 M | 11 min per training run at 8 x 8; W&B runs 7kd2qfk4, jzxy2zoo |
| 2026-10-19 | 3 | EXP-010e token-matched control: 59 steps + think-tag eval (bb6f2a6) | TIR H100 SXM 80 GB | 0.42 | INR 159 (USD 1.9) | 7.5 M | 13.4 M | W&B run iagfhjwq |
| 2026-10-19 | 3 | EXP-010f think-tag n = 32, T = 0.6, SFT and RFT on 300-problem subsets (bb6f2a6) | TIR H100 SXM 80 GB | 2.20 | INR 834 (USD 9.9) | 97.6 M | 0 | includes 10 min lost to the flag bug (the SFT half sampled an extra set and was stopped) |
| 2026-10-19 | 3 | archive, Hub inventory, transcript scan, prune, the user's download and Stop | TIR H100 SXM 80 GB | 1.6 | INR 606 (USD 7.2) | - | - | about 1 h of it after the console's 16:01 IST report, per the user |

| Date | Phase | Run | GPU | Hours | USD | Tokens generated | Tokens trained | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-18/19 | 2 + 3 | TIR H100 notebook `rlvr`, whole life (usage report: 25.07 h to 16:01 IST 19 Oct, INR 9,502.78; about 3 h more to the Stop, INR 1,137) | TIR H100 SXM 80 GB | 28.1 | INR 10,640 (USD 127) | about 650 M | about 285 M | the Phase 2 and Phase 3 rows above are inside this one; the node clock ran about two hours ahead of the console's, so the per-step hours are the node's and the total is the invoice's; disk INR 118.40 to 16:01 IST and about INR 112 a day since |

| Date | Phase | Run | GPU | Hours | USD | Tokens generated | Tokens trained | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-20 | 4 | path B environment build on a new disk; venv stalled three times on the PyTorch index's triton host, fixed with curl-fetched sha256-checked wheels (a8b9f69) | TIR H100 SXM 80 GB (idle) | 1.6 | INR 606 (USD 7.2) | - | - | 12:14-13:52 UTC, 88 GPU-idle minutes against 25 planned; the wheels are on the Hub since |
| 2026-10-20 | 4 | `setup` (downloads, data rebuild with hashes asserted, `rft` dev eval, pinned subsets) and `smoke` | TIR H100 SXM 80 GB | 0.4 | INR 152 (USD 1.8) | 0.7 M | 0 | two relaunches of `setup` (dirty tree from the RFT builder's report; torchcodec), 3 idle minutes; the pull window 8 min |
| 2026-10-20 | 4 | EXP-010g and EXP-010h: build, train (12 and 14 min), five-set evals (10 and 13 min) (a8b9f69) | TIR H100 SXM 80 GB | 0.9 | INR 341 (USD 4.1) | 9 M | 30 M | W&B fid4o2yz, u4xl8kxf |
| 2026-10-20 | 4 | debugging run: two OOM attempts at micro-batch 32, the user's decision (micro-batch 16, vLLM 0.30), 50 steps at 54.1 s/step, one dev eval (a8b9f69) | TIR H100 SXM 80 GB | 1.03 | INR 390 (USD 4.6) | 16 M | 16 M | not a result; 5 idle minutes for the question; W&B uot02yla |
| 2026-10-20 | 4 | EXP-012 main run: 200 GRPO steps at 56.3 s/step + four dev evals (a8b9f69) | TIR H100 SXM 80 GB | 3.3 | INR 1,251 (USD 14.9) | 62 M | 62 M | peak 70.4 GB; W&B 4b2hf5vo |
| 2026-10-20 | 4 | EXP-013 seed 1: 200 steps at 56.2 s/step + four dev evals (a8b9f69) | TIR H100 SXM 80 GB | 3.3 | INR 1,251 (USD 14.9) | 62 M | 62 M | W&B w0dhqitz |
| 2026-10-20 | 4 | EXP-012 evals: three templates, gap, probe, subsets check, n = 32 think-tag samples (28 min); EXP-013 five-set eval (a8b9f69) | TIR H100 SXM 80 GB | 0.93 | INR 353 (USD 4.2) | 45 M | 0 | all sets of a checkpoint in one process |
| 2026-10-20/21 | 4 | uploads (four finals, 8.5 min in parallel), hand-back, archive (188 MB, 875 files), the wheels upload and the addendum; idle until the user's Stop at 01:30 UTC (07:00 IST) | TIR H100 SXM 80 GB (idle) | 1.72 | INR 652 (USD 7.8) | - | - | GPU idle from 23:47 UTC |

| Date | Phase | Run | GPU | Hours | USD | Tokens generated | Tokens trained | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-20/21 | 4 | TIR H100 notebook, session 1 whole life (12:14 UTC 20 Oct to the Stop at 01:30 UTC (07:00 IST) 21 Oct) | TIR H100 SXM 80 GB | 13.27 | INR 5,030 (USD 60) | about 195 M | about 170 M | the rows above are inside this one; plan 12.0 h / INR 4,570; the work itself 9 h 55 min of GPU time; the overrun is the build stall and the idle tail; the disk (320 GB) about INR 107 a day while the notebook exists, Stopped or not |

| Date | Phase | Run | GPU | Hours | USD | Tokens generated | Tokens trained | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-21 | 4 | session 2 start: the user's Start (about 03:25 UTC), secrets, pull to 8e94fb5, env check; `setup` (10 s, everything on disk, hashes re-checked) | TIR H100 SXM 80 GB | 0.10 | INR 38 | 0 | 0 | path A, no build |
| 2026-10-21 | 4 | EXP-014a `a1` (KL 0): 200 steps at 40.3 s/step (no reference forward at beta 0) + four dev evals (8e94fb5) | TIR H100 SXM 80 GB | 2.40 | INR 910 (USD 10.8) | about 45 M | about 31 M | peak 63.9 GB; projection 0.90x the estimate |
| 2026-10-21 | 4 | EXP-014b `a2` (dynamic sampling): 200 steps at 85.3 s/step (64 groups generated, 32 kept) + four dev evals | TIR H100 SXM 80 GB | 4.91 | INR 1,861 (USD 22.2) | about 118 M | about 59 M | peak 70.4 GB; 1.34x the estimate at step 10, 1.23x at the end |
| 2026-10-21 | 4 | EXP-015 `cap` (2,048 cap): 200 steps at 71.8 s/step + four dev evals | TIR H100 SXM 80 GB | 4.18 | INR 1,584 (USD 18.9) | about 77 M | about 75 M | peak 73.4 GB, no OOM at micro-batch 16; 1.01x the estimate |
| 2026-10-21 | 4 | `evals` (three five-set greedy evals, 7-9 min each; cap n = 32 sampling 36 min) and `report` (under 1 min) | TIR H100 SXM 80 GB | 1.00 | INR 379 (USD 4.5) | about 30 M | 0 | subsets re-checked twice |
| 2026-10-21 | 4 | uploads (three ablation finals, 12.5 min), hand-back, archive (1,774 paths), addendum | TIR H100 SXM 80 GB (GPU idle) | 0.42 | INR 159 (USD 1.9) | - | - | GPU idle from 15:59 UTC as planned |
| 2026-10-21 | 4 | additions at the user's request (16:33-16:48 UTC): offline KL of three finals vs `rft` (twice), a1 zero-shot eval (4.4 min), addendum 2 | TIR H100 SXM 80 GB | 0.25 | INR 95 (USD 1.1) | about 1 M | 0 | 8 GPU-minutes of work |
| 2026-10-21 | 4 | idle: Start to `setup` (about 5 min) and addendum 2 to the user's Delete (about 16:52 UTC) | TIR H100 SXM 80 GB (idle) | 0.15 | INR 57 | - | - | |

| Date | Phase | Run | GPU | Hours | USD | Tokens generated | Tokens trained | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-21 | 4 | TIR H100 notebook, session 2 whole life (the user's Start about 03:25 UTC to the Delete about 16:52 UTC, 21 Oct) | TIR H100 SXM 80 GB | 13.45 | INR 5,100 (USD 61) | about 270 M | about 165 M | the rows above are inside this one; plan 13.4 h / INR 5,060; GPU work 12 h 29 min with under 5 min idle between steps; nothing dropped, no run repeated; the disk billed about INR 107 a day from 20 Oct to the Delete (about INR 160) |

Phase 4 in all: 26.7 node-hours, INR 10,130 (sessions 1 and 2) plus about INR 160 of disk. The plan's Phase 4 line at the original shape was about 90 H100-hours; shape B with the cap test in A3's slot cost 30 % of that.

| Date | Phase | Run | GPU | Hours | USD | Tokens generated | Tokens trained | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-21 | 5 | EXP-017 accuracy against tokens from the existing sample sets (379b67a) | laptop CPU | 0 | 0 | 0 | 0 | about 2 CPU-minutes; five sample sets fetched from the Hub (about 1.5 GB) |
| 2026-10-21 | 5 | EXP-016 smoke: 0.6B base, 8 problems, budget 64 | RTX 4060 Laptop 8 GB | 0.05 | 0 | <0.01 M | 0 | the forced-stop path end to end |
| 2026-10-21/22 | 5 | EXP-016 sweep: sft and rft at five budgets, GSM8K + MATH-500, greedy think-tag (1cf82a5) | RTX 4060 Laptop 8 GB | 8.55 | 0 | about 5.5 M | 0 | ten runs of 30-66 min, 03:01-11:34 UTC on the log's clock; the laptop slept most of the night with the job suspended (lid closed), nothing lost; two fp32 checkpoints (6.9 GB each) downloaded at 0.4-3 MB/s, about 2 h of network |
| 2026-10-22/23 | 5B | EXP-020: grpo forced at 1,536 and 2,048; unforced sft and rft on the laptop (3fc00b3 era, phase5_laptop.sh) | RTX 4060 Laptop 8 GB | 5.6 | 0 | about 4 M | 0 | 18:55-00:57 UTC; the unforced runs loop to the window (about 2 h each) |
| 2026-10-23 | 5B | EXP-021: sft and rft with Wait before the stop at 1,536 (3fc00b3) | RTX 4060 Laptop 8 GB | 2.05 | 0 | about 1.5 M | 0 | 00:57-03:00 UTC |
| 2026-10-23 | 5B | EXP-022 first reward-model run: LoRA head on Qwen3-0.6B-Base, 20,000 examples, held-out eval | RTX 4060 Laptop 8 GB | 6.5 | 0 | 0 | about 30 M | 03:01-09:31 UTC; held-out 87.9 %, AUROC 0.938; weights lost in a WSL kernel hang during the save (FAILURES_AND_LEARNINGS.md, 23 Oct); 0.95 h idle until the restart |
| 2026-10-23 | 5B | EXP-022 rerun with the save fix (3bd2898): training, held-out eval, scoring of six n = 32 sample sets (57,600 samples), the chart | RTX 4060 Laptop 8 GB | 10.6 | 0 | 0 | about 30 M | 11:30-22:06 UTC; training 5.8 h (1,250 steps), held-out 89.4 %, AUROC 0.944; scoring 4.65 h (38-55 min a set); chart 1 min |
| 2026-10-23 | 5B | EXP-020 GRPO control: grpo unforced greedy think-tag on the laptop, GSM8K + MATH-500 (clean WSL clone at 0d10994) | RTX 4060 Laptop 8 GB | 1.1 | 0 | about 2.3 M | 0 | 22:09-23:16 UTC; a first launch at 22:07 was refused by the harness (the main checkout held an uncommitted edit) |
| 2026-10-24 | 7 | EXP-024 reproduction: fresh clone of 4548c00, fresh venv, the four 1.7B base greedy configs, the per-problem check | RTX 4060 Laptop 8 GB | 0.85 | 0 | about 3.3 M | 0 | 05:07-05:59 UTC on the WSL clock; environment build 1 min (uv cache), archive download 1 min, evals 0.82 h |
| 2026-10-23 | 6 | EXP-023 trace analysis, twice (the second with the first-cycle measure and the forced split) | laptop CPU | 0 | 0 | 0 | 0 | about 15 CPU-minutes |

Phase 3 part: about 10.3 node-hours, INR 3,900: core 4.1 h, the gap-analysis runs 4.0 h, the
analysis 0.9 h idle, archive and Stop 1.3 h. The plan's Phase 3 line was 4.5-7 A100-hours.

**Final totals (24 Oct 2026, end of Phase 7).** Rented, unchanged since the Delete of 21 Oct: L4 10.55
node-hours, A30 1.32 h, two TIR H100 notebooks 28.1 + 26.7 = 54.8 h, their disks and the dataset volume:
**about INR 21,750 (USD 259) of the INR 22,000 (USD 262) ceiling**. Local, at INR 0: about 15.3 laptop
GPU-hours to the end of Phase 5A, 25.9 in Phase 5B (EXP-020 5.6 + 1.1, EXP-021 2.05, EXP-022 6.5 lost + 10.6),
0.85 in Phase 7 (EXP-024): **about 42 laptop GPU-hours**; Phase 6 and the analyses CPU only. No run after
21 Oct rented anything.

**Running totals (after Phase 4 session 2 and the Delete, 21 Oct 2026).** Rented: L4 10.55 node-hours
(INR 524.30), A30 1.32 h (INR 118.80 + 20.50), the TIR dataset volume (INR 10.20), the first H100
notebook 28.1 h (INR 10,640) and its disk (INR 132 to the Stop), the second H100 notebook 13.27 +
13.45 = 26.7 h (INR 10,130) and its disk (about INR 160, 20 Oct to the Delete): **about INR 21,750
(USD 259)**. Local: 6.70 h plus the fold-in. Budget used: about 99 % of the INR 22,000 (USD 262)
ceiling the user set on 20 Oct; nothing on E2E is billing after the Delete (the invoice is the record;
the notebook rows are the node clock's). Any further rental needs a new ceiling
(`docs/phase-5-plan.md`). Phase 5 option A (21-22 Oct) cost INR 0: about 8.6 laptop GPU-hours and
2 CPU-minutes (local total about 15.3 h). The exact Delete minute was not recorded; the whole-life row carries the
chat's time and will be replaced by the invoice's.

**Running totals (after Phase 4 session 1, 21 Oct 2026).** Rented: L4 10.55 node-hours (INR 524.30),
A30 1.32 h (INR 118.80 + 20.50), the TIR dataset volume (INR 10.20), the first H100 notebook 28.1 h
(INR 10,640) and its disk (INR 132 to the Stop), the second H100 notebook's session 1 13.3 h
(INR 5,030) and its disk (about INR 107 a day from 20 Oct): about INR 16,600
(USD 196). Local: 6.70 h. Budget used: about 75 % of the INR 22,000 (USD 262) ceiling the user set on
20 Oct; session 2 at the measured rates is about 12 h, INR 4,550, which fits with about INR 700 to
spare if the GPU is never idle (plan section 5; drop order section 8).

**Running totals (after the session, 19 Oct 2026).** Rented: L4 10.55 node-hours (INR 524.30),
A30 1.32 h (INR 118.80 + 20.50), the TIR dataset volume (INR 10.20), the H100 notebook 28.1 h
(INR 10,640) and its disk (INR 132 to the Stop): about INR 11,450 (USD 136). Local: 6.70 h.
Budget used: about 80 % of the INR 14,300 (USD 170) ceiling of 13 Oct, about 52 % of the
INR 22,000 (USD 262) ceiling the user set on 20 Oct for Phase 4; the notebook disk adds about
INR 112 a day while it exists (INR 3,400 a month), which is nine H100-hours a month for keeping
the environment warm. The user said on 19 Oct that the budget may grow while the project is on
track; Phase 4 as planned in PLAN.md did not fit the remainder, and shape B of
`docs/phase-4-plan.md` (about 26 node-hours, INR 9,800) fits the raised ceiling with about
INR 700 to spare.

**Budget.** USD 100-170 confirmed on 13 Oct; raised to INR 22,000 (USD 262) on 20 Oct for Phase 4
at shape B (`docs/phase-4-plan.md`, section 5). Original estimate 53-55 A100-hours (PLAN.md section 8). First to drop
if over: the seed-repeat GRPO run, then the LoRA SFT run. Never the baselines or the CIs.

**Rates (E2E Networks, MyAccount on-demand, INR per hour, 16 Oct 2026).** L4 24 GB 49; A30 24 GB 90;
A40 48 GB 96; V100 32 GB 100; L40S 48 GB 102; A100 40 GB 179; A100 80 GB 189; H100 not offered. About
INR 84 per USD. A powered-off node still bills: sessions end with Terminate. Plan and per-phase
node choice: `docs/rental-setup.md`, section 2c. Rented rows below quote the invoice line in INR with
the USD conversion of the day.

**Rates from 18 Oct 2026 (E2E TIR, the managed notebook platform; `docs/rental-setup.md` 2d).** H100
SXM 80 GB plan INR 379.05 per hour (26 vCPU, 250 GB RAM), billed per minute while running; a stopped
notebook bills only its workspace disk: `Ceph_Block_Storage_NOTEBOOK`, INR 10 per GB-month on a
billable 340 GB (usage report of 19 Oct 2026), about INR 112 a day. The node's price is exported as
`RLVR_NODE_RATE`. Projection before the session: Phase 2 + 3 in 15-18 H100-hours, INR 5,700-6,800;
the whole plan 38-46 H100-hours plus disk, INR 15,000-18,000. Actual: 28.1 node-hours, INR 10,640
(the whole-life row above).

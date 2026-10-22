# Phase 5 plan: the cost of termination, and whether the prompts were the limit

Written 21 Oct 2026 after Phase 4 session 2, from `docs/research-narrative.md` "Open now (after
session 2)". **Status: option A chosen by the user on 21 Oct 2026 and run 21-22 Oct (EXP-016, EXP-017 done,
`outputs/reports/phase-5.md`); options B to D still await a budget.** Nothing here runs before (a) the user
picks a budget option in section 4 and (b) the Phase 4 PR is merged. Every run has its narrative link
and its `experiments.md` entry (EXP-016 to EXP-019, EXP-010i, EXP-014c); the rules of
`docs/phase-4-plan.md` section 7 and section 9 item 3 carry over.

## 0. What Phase 4 left

In the trained format every post-SFT gain is termination; a longer cap, no KL penalty and a
better-filled batch each buy more or less of the same stopping and nothing on the problems both
policies finish; no on-policy stage after SFT moved the best-prompt accuracy beyond noise
(`outputs/reports/phase-4.md` section 18). Two questions follow, in order of cost:

1. **How cheaply can stopping be bought?** If a decoding rule or an off-policy pair buys what RFT and
   GRPO bought, the three GPU-hours of GRPO were the expensive way to a format fix.
2. **Were the prompts the limit?** The pool was MATH levels 1-3 plus GSM8K; the eval gains sit on
   levels 1-3; a third to a half of the training groups were all-correct; the reward plateaued by
   step 50.

PLAN.md's Phase 5 (DPO) and Phase 6 (test-time compute) are re-cut around these two questions; Phase
7 (distillation, serving) stays an engineering deliverable, last or cut.

## 1. The runs, free ones first

| order | run | id | where | cost | what it decides |
|---|---|---|---|---|---|
| 1 | budget forcing on `sft` and `rft`: force `</think>\n<answer>` at a budget B, let the model finish | EXP-016 | laptop (4060, bf16, greedy; Phase 1 ran the 1.7B evals there) | INR 0, about 6 GPU-hours of laptop time | whether the training taught what a rule provides |
| 2 | accuracy against generated tokens: greedy, maj@k and pass@k per checkpoint from the existing n = 32 / 64 sample sets | EXP-017 | laptop, no GPU | INR 0 | whether the RL policy is the cheapest way to its own number |
| 3 | matched-incorrect rejection-sampling set (the powered EXP-010g) | EXP-010i | H100, inside the next session | about 0.5 h, INR 300 | whether the correctness filter is RFT's mechanism |
| 4 | DPO on finished-correct vs looping pairs from the EXP-009 samples of `sft` | EXP-019 | H100 | about 1.5 h, INR 800 | whether an off-policy pair buys the termination gain |
| 5 | GRPO on MATH levels 3-5 with `rft` solve rate in (0, 0.5] | EXP-018 | H100 | about 4.5 h, INR 1,700 | whether the prompts were the limit |
| 6 | Dr. GRPO normalisation (deferred from Phase 4) | EXP-014c | H100 | about 3.5 h, INR 1,400 | whether the normalisation's length bias is visible here |

Runs 1 and 2 go first because their answers change whether 3-6 are worth renting for: if a decoding
rule recovers most of the gain and voting over `rft` matches `grpo` greedy, the case for more RL at
this scale is weak and run 5 is the only one left with a question.

## 2. What has to be built (laptop, with tests, before any run)

- `rlvr-eval --force-stop-at B`: the sampler stops at B tokens if no `</think>` has been emitted,
  appends the closing tags, continues for up to 128 tokens; the per-sample file records whether the
  stop was forced. Unit test on a fake sampler; a smoke on the 0.6B base.
- `scripts/plot_accuracy_vs_tokens.py`: one chart per set, accuracy with CIs against mean generated
  tokens per problem, series per checkpoint (greedy, maj@k, pass@k), from sample files; test on a
  synthetic sample set.
- `rlvr-build-dpo`: pairs from the EXP-009 sample files (chosen: the shortest finished correct trace;
  rejected: a truncated trace of the same prompt), manifest with hash; `rlvr-dpo` over TRL's
  `DPOTrainer` with the project's run_info and preflight row. Tests on a toy sample file.
- `rlvr-build-rft --match-incorrect 0.5`: the matched-incorrect kept set; test.
- The level 3-5 prompt pool: a solve-rate pass of `rft` on MATH-train levels 3-5 (n = 8, T = 1.0,
  on the node, about 40 min, INR 250, inside EXP-018's cost) and a new manifest; `phase4_preflight.py`
  rows for `hard`, `dpo`, `a3`.
- `scripts/phase5_rental.sh` with the Phase 4 script's shape (resumable steps, anchored EXIT lines,
  preflight first, report step with the paired tables and the uploads); the handoff rewritten for
  path B (the Hub wheels, no disk).
- Carry-over fixes: `rlvr-grpo` appends `train_log.jsonl` per step; `kl_offline.py` into `scripts/`
  with a CPU test on a toy model; a per-set truncation column in the card.

About two laptop days.

## 3. Eval protocol

As Phase 4: five-set greedy think-tag evals in one process, paired against `sft`, `rft` and `grpo` on
shared ids with bootstrap CIs, the noise row 0.3 / 0.2 / 4.8 / 0.6, the decomposition for every
trained checkpoint, the pinned subsets for pass@k, the perturbation gap and the probe for any
checkpoint trained on new prompts (EXP-018). Zero-shot for every trained checkpoint, since the Phase 4
lesson is that the trained-format number and the best-prompt number disagree.

## 4. Budget options (the user's decision; rate INR 379.05 per H100 node-hour, setup from the Hub wheels about 0.3 h)

| option | runs | node-hours | rupees | new ceiling (project at about INR 21,750 now) |
|---|---|---|---|---|
| A | 1, 2 only | 0 | 0 | none needed |
| B | A + 3 + 4 in one session | about 2.5 | about INR 950 | INR 23,000 |
| C | B + 5 | about 7 | about INR 2,650 | INR 24,500 |
| D | C + 6 | about 10.5 | about INR 4,000 | INR 26,000 |

A single session for B, C or D (no Stop between runs; the node is Deleted at the end as in Phase 4).
Recommendation: **A now, then decide B or C on A's results**; D only if the Phase 4 debts are to be
closed for the write-up. The drop order inside a session: EXP-014c, then EXP-010i, then EXP-019; EXP-018
is never dropped once started.

## 5. What would change this plan

- EXP-016 recovering most of the gain: run 4 (DPO) loses its point (a free rule beats a cheap
  trainer); run 5 keeps its question.
- EXP-017 showing `rft` maj@4 at or above `grpo` greedy on MATH-500: the write-up's test-time-compute
  chart is the headline of Phase 5, and a verifier for best-of-N becomes the natural next ask.
- EXP-018 moving the both-finished difference: the project's central claim gets its first exception
  and the next question is scale of the pool, not the method.

## 6. Rules carried over (unchanged)

Narrative link and registry entry before any run; preflight before every run; same subsets across
methods; whole GPU per process; watchers under two minutes on the stdout logs; no idle gaps; the node
agent never Stops or Deletes; secrets only as environment variables; the harness refuses a dirty tree;
every deviation written where its number is quoted; the ledger row before the number is believed.

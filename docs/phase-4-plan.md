# Phase 4 plan: GRPO with a verifiable reward (first page, 19 Oct 2026)

Written on the laptop the day the Phase 2 + 3 session closed, before any Phase 4 code. This page
records the decisions Phase 3's results force, the budget arithmetic at measured H100 rates, and
the first steps of the next node session. The pre-registration (EXP-012 onward) is written once
the user has settled the two decisions marked **[decide]**; nothing below runs before that.

## 1. What Phase 3 settled

- **Start checkpoint: `rft` (seed 0, `Raviteja546/reasoning-rlvr-rft`).** The EXP-010 falsifier
  did not fire (RFT beats SFT by +3.3 GSM8K / +5.8 MATH-500 in the trained format, paired), so
  Phase 3's decision 6 applies. The two extra seeds are the noise floor (up to 3 points between
  seeds), not candidates. The dev-split think-tag accuracy of `rft` was not measured on the node;
  it costs one 3-minute eval and is run at the start of the Phase 4 session for the record.
- **Prompt set:** `outputs/data/rft/learnable_prompts.jsonl` (4,606 prompts with 0 < solve rate < 1
  for the SFT policy at n = 8, T = 1.0: 2,267 MATH levels 1-3 and 2,339 GSM8K-train; hash
  281bdfe31176411b in `data/manifests/rft_iter1.json`). PLAN.md asks for MATH levels 1-3 plus a
  20 % GSM8K slice. Caveat: the rates were measured at a 2,048-token cap, so some rate-0 problems
  (excluded) are "cannot finish in 2,048", and the set is the SFT policy's, not `rft`'s; a re-measure
  from `rft` at n = 8 is a 1.3-hour run and is not planned.
- **The reward must not pay for loops.** A quarter of SFT generations and a sixth of RFT
  generations never terminate at greedy, and the RFT gain was mostly the removal of loops. Phase
  4's reward is correct-and-finished (a truncated completion scores 0 whatever it contains), and
  response length, truncation rate and the "Wait"-count are logged per step as results.
- **The bar GRPO must clear:** +3 to +6 points at greedy in the trained format over `sft` is what
  one 12-minute RFT run bought. GRPO's row is read against `rft`, with the compute-matched control
  being `rft` itself plus the extra sampling hours.

## 2. Budget arithmetic at measured rates (the first [decide])

PLAN.md's Phase 4 main run: 64 prompts x G = 8 per step, up to 2,048 new tokens, 400-600 steps,
plus a seed repeat and three one-knob ablations at the same size. On this H100 with vLLM
colocated, at the session's measured rates (11-12k output tokens/s at these lengths; packed
training at about 70 M tokens/h):

| quantity | per step | 500 steps |
|---|---|---|
| completions | 512 | 256,000 |
| generated tokens (mean 1,500 from `rft`) | 0.77 M | 384 M |
| sampling time | about 65 s | 9 h |
| training + reference log-probs (about 0.9 M tokens) | about 50 s | 7 h |
| overhead (weight sync, eval every 50 steps) | about 15 s | 2 h |
| **wall clock** | **about 2.2 min** | **about 18 h, INR 6,900** |

Five runs of that size are 90 hours. The project has spent about INR 11,450 of its INR 14,300
ceiling, and the notebook disk costs INR 112 a day. The plan as written does not fit; the user said
on 19 Oct that the budget may grow while the project is on track. Three shapes, for the user to
choose from or amend:

| shape | main run | seed repeat | ablations | node-hours | INR | what it answers |
|---|---|---|---|---|---|---|
| A: as planned | 500 steps, 64 x 8 | yes | 3 | about 90 | 34,000 | everything in PLAN.md |
| B: half size | 200 steps, 32 x 8 (0.38 M tokens/step, 1.2 min) | yes | 3 | about 22 | 8,500 | the same questions with wider CIs; GRPO's gain above the RFT bar is probably still resolvable (3-6 points needs CIs of about 2.5 on GSM8K: the full test set at greedy gives that) |
| C: minimal | 200 steps, 32 x 8 | no | 1 (A2 dynamic sampling, the cheapest to argue for) | about 10 | 3,800 | does GRPO beat RFT; one component question; no noise row (the RFT seeds stand in, badly) |

Recommendation: **B**, in two sessions (main + seed first, about 9 hours; ablations after the
main run is read), with the two Phase 3 ablations below at the start of the first session. B needs
the ceiling raised by about INR 6,000 (USD 70) including the disk days; C fits the remainder and
answers the central question with the weakest evidence. **[decide: A, B or C, and the ceiling]**

## 3. First steps of the next session (code first, on the laptop)

1. **Two Phase 3 ablations that need code**, both from the gap analysis, priced on the H100:
   - EXP-010g, RFT without the correctness filter: `rlvr-build-rft --keep-incorrect` (same number
     of self-generated traces, random instead of correct), train, think-tag eval, paired against
     `rft`. 20 minutes, INR 130. If it matches `rft`, the Phase 3 gain is self-distillation of
     short traces; if it matches the control, the filter is the mechanism.
   - EXP-010h, random-two instead of shortest-two kept set (a `--per-problem-select random`
     option): separates length selection from correctness selection. 20 minutes, INR 130.
   - (Deferred, 2 hours, INR 760: SFT without the think tags, EXP-006b, a builder option and a
     tagless eval prompt. **[decide: now or never]**)
2. **The correctness check** (PLAN.md Phase 4 step 1): the own reference GRPO loop
   (`src/rlvr/ref/grpo_loss.py`, `advantages_kl.py`, tested since Phase 0) against TRL's
   `GRPOTrainer` on a 50-prompt toy set with identical seeds; per-step losses and rewards must
   agree. Runs on the laptop's GPU with the 0.6B model; no node time.
3. **The debugging run on GSM8K-train** (step 2): 50-100 steps from `rft` on the node, where the
   reward function, tag parsing, vLLM sync, logging and the zero-variance-group fraction are shaken
   out. About 1.5 hours; its numbers are not results.
4. The main run under the chosen shape, with eval on the dev split every 50 steps and a checkpoint
   every 100; the pre-registered metrics are PLAN.md's (reward, correct rate, length, truncation,
   entropy, KL, clip fraction, zero-advantage groups, gradient norm, tokens per step, step time).

## 4. Node session set-up

`docs/h100-next-session.md` (path A: the disk persists; about 5 minutes to the first GPU job).
Delete `outputs/bin/rlvr-sft` (the obsolete shim), `git pull` to the Phase 4 branch, `uv pip
install -e . --no-deps`, confirm `flash_attn` imports and `python scripts/triton_check.py`
passes, check the hash of `outputs/data/rft/learnable_prompts.jsonl` against the manifest, then the
`rft` dev-split eval (3 minutes) before any training step. Secrets before tmux; both keys into
tmux's global environment. The disk is INR 112 a day: if the user's decision takes more than about
two weeks, Delete the notebook (every checkpoint and sample set is on the Hub; the session
archive holds the rest) and rebuild from path B (45 minutes) when Phase 4 starts.

## 5. What goes into the pre-registration once section 2 is decided

EXP-012 (GRPO main run vs `rft`, paired, with the sharpening-versus-capability analysis of
PLAN.md step 7), EXP-013 (seed repeat, the noise row), EXP-014a-c (the ablations A1 KL beta, A2
DAPO clip-higher + dynamic sampling, A3 Dr. GRPO), each with hypothesis, metric, falsifier and
config, in `experiments.md` before the branch's first node session. Hypotheses to write against
Phase 3's numbers: GRPO from `rft` gains 2-5 points at greedy in the trained format over `rft` on
MATH-500 (the headroom is on levels 4-5 where RFT gained most), pass@64 unchanged within 2 points
(sharpening), truncation falls below 10 % under a finished-only reward, Countdown stays unusable,
the KL-free run matches the main run within noise at this step count, dynamic sampling matters
because the learnable set still holds 37 % rate-1 GSM8K problems the sampler wastes groups on.

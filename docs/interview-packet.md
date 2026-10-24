# reasoning-rlvr: the interview packet

Two pages to read before a conversation about this project. Each number is paired on shared problems
with a 95 % bootstrap CI and re-derives from its per-sample files (claim ids in brackets, evidence in
`outputs/reports/evidence/`).

## The study in six lines

- Qwen3-1.7B-Base; SFT on 23,000 distilled traces, one round of rejection-sampling fine-tuning (RFT),
  then GRPO (32 prompts x 8 samples, 200 steps, reward 1 only for a finished, well-formed, correct
  answer); GSM8K, MATH-500, Countdown, a perturbed GSM8K twin.
- Greedy, batch-invariant vLLM; every comparison paired per problem; a hypothesis and falsifier per
  run before it started (36 registry entries); a seed repeat as the noise row.
- The question: what does each stage buy, and is it reasoning, sharpening or termination?
- About 55 H100-hours and 11 L4-hours (USD 260) plus about 40 laptop GPU-hours, 14-24 Oct 2026.
- Three contamination probes before and after every stage; a reward-hacking watch on every RL run.
- One person, one GPU-day per method; every failure logged with its fix.

## Five results

1. **The on-policy gains are termination.** GRPO over RFT on MATH-500: +7.0 [2.8, 11.0], of which
   +10.2 [7.4, 13.2] on problems RFT did not finish and 0.0 [-4.3, 3.9] on problems both finished
   (P4-02); RFT over SFT decomposes the same way (P3-04).
2. **A free decoding rule recovers it.** Appending `</think><answer>` at 1,536 tokens lifts SFT by
   +4.6 [3.4, 6.0] / +7.2 [4.0, 10.4], more than RFT's whole gain (P5-08); with RFT and GRPO both
   stopped at 2,048, GRPO's MATH-500 lead is -0.4 [-4.4, 3.4] (P5-16).
3. **Votes and a small verifier reach the RL policy's number.** Two RFT samples voted match GRPO's
   greedy accuracy at 2.6x the tokens (P5-01); with a 0.6B verifier, the better of two beats it by
   +3.0 [0.1, 6.0] / +4.0 [0.2, 7.7] (P5-13); its best-of-32 collapses 12-13 points below
   majority-of-32 on MATH-500 (P5-14).
4. **A loop is one step repeated from a fixed point** (a 100-190-word unit starting near word 500 /
   800), and training made loops rarer without changing their shape (KS 0.11-0.16; P6-01).
5. **Determinism is per machine.** The same checkpoint flips 10-23 % of problems between an H100 and
   an RTX 4060 (P5-10, P5-18); re-pairing on one machine turned "GRPO gains nothing from the stop rule"
   into +2.0 [0.6, 3.6] on MATH-500 (P5-11).

## Predictions that failed, and what each taught

- Default vLLM greedy is deterministic: wrong by up to 6 points; batch-invariant mode, then the
  machine term (EXP-000, 020b). *Pair only within one machine.*
- Seed noise under 3 points on MATH-500: it was 4.8 (EXP-013). *One seed per arm cannot resolve a
  5-point gain on 500 problems.*
- Countdown flat under GRPO: it doubled, by termination (EXP-012). *Check the mechanism before calling
  it transfer.*
- KL off within noise: it landed above, by termination, 15x further from RFT (EXP-014a).
- Clip-higher and dynamic sampling help: no (EXP-014b). *With one update per batch the ratio is 1
  and clipping never acts.*
- A longer cap unlocks reasoning: no; the both-finished difference stayed 0.0 (EXP-015).
- Forced answers as good as finished ones: right about half the time for SFT and RFT, 17-22 % for
  GRPO (EXP-016, 020a). *Loops hold a partial derivation, not an answer.*

## The papers, in one line each (full notes: `docs/paper-notes.md`)

- **DeepSeekMath / GRPO:** agree; same objective, verifiable reward; RL sharpens (GSM8K: no problem
  newly solved at k = 16), and the pass@1 gain is termination.
- **DAPO:** clip-higher and dynamic sampling did nothing at 1.7B, for a measurable reason (clip
  fraction 0; filtered prompts already informative); KL off did not destabilise.
- **Dr. GRPO:** the template finding reproduces on Qwen3 (zero-shot beats 4-shot CoT by 12 on
  MATH-500); the loss-bias ablation is pre-registered, not run.
- **Yue et al.:** agree within k = 32 (sharpening, a small MATH-500 widening); we go further on pass@1
  (termination) and on votes (curves coincide from k = 4); the crossover regime is untested.
- **s1:** the stop half reproduces strongly; the "Wait" half does not (equal to the plain stop at the
  same tokens), matching later work.
- **DeepSeek-R1:** distillation is the largest step at small scale (+13.0); RL after it bought
  stopping.
- **STaR:** one round is selection for stopping; half its effect is the length filter (-2.4 for random
  traces).
- **Spurious rewards:** no contamination signal here, and their clipping mechanism cannot act in our
  runs, but our result invites their control: a finished-only reward, not run.

## Tradeoffs, cost, and what next

- **Spent where:** SFT and RFT one H100 session (28.1 h, INR 10,640); GRPO two sessions (26.7 h, INR
  10,130); everything after Phase 4 on the laptop at INR 0. A GRPO run is 3.1 H100-hours; RFT's training
  12 minutes; the stop rule nothing.
- **Accepted:** one seed per ablation with a measured noise row; 1,536-token training cap (tested by
  EXP-015); 300-problem subsets for sampled curves; no PRM; no distillation to 0.6B.
- **With 10x compute:** GRPO on MATH levels 3-5 (EXP-018) with two seeds; a finished-only-reward
  control; Dr. GRPO's loss; a matched-incorrect RFT set. **Not:** more steps of the same GRPO, whose
  reward plateaued by step 50.
- **What would change the conclusion:** GRPO on harder prompts moving the both-finished difference,
  or a reward-ablation showing the correctness term matters beyond stopping.

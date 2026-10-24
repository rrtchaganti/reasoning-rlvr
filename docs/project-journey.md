# The project journey, start to finish

The whole project in the order it happened, for reviewing it in one sitting. Each step reads the
same way: **the idea** (what we believed or wanted), **what we did**, **what happened** (the
result or the surprise), and **what it changed** (the decision or the code that came out of it).
Numbers are the ones quoted in the reports, paired on shared problems with 95 % bootstrap CIs where
two models are compared; "GSM8K / MATH-500" pairs are written as two numbers with a slash.

Where to go deeper: the reasoning behind each question is in `docs/research-narrative.md`; the
day-by-day detail in `docs/phase-N-journal.md`; the pre-registrations in `experiments.md`; the
incidents in `FAILURES_AND_LEARNINGS.md`; the results in `outputs/reports/phase-N.md` and, in short,
`README.md` and `docs/results.md`.

**Contents.** 1. Before the code · 2. Phase 0 · 3. Phase 1 · 4. Phase 2 · 5. Phase 3 · 6. Phase 4 ·
7. Phase 5A · 8. The 22 Oct turn · 9. Phase 5B · 10. Phase 6 · 11. Phase 7 · 12. The thread in one
table · 13. What the process taught · 14. What is still open.

---

## 1. Before the code (8-13 Oct 2026): the idea and the plan

**The idea.** Thousands of people have run a GRPO notebook; what a post-training team screens for
is whether someone can design a controlled experiment and reach a conclusion others can trust,
implement the core pieces from scratch, debug a run from its logs, read a paper critically, and
make cost decisions. So the deliverable was set as a small, fully reproducible study, not "a model
that got better at GSM8K": take one base model (Qwen3-1.7B-Base), apply each stage of the modern
reasoning recipe (SFT on distilled traces, rejection-sampling fine-tuning, RL with a verifiable
reward, then DPO, test-time compute, distillation), and measure what each stage buys, with CIs,
paired comparisons, contamination checks and a compute ledger. The sharpest question, from Yue et
al.: does RL add reasoning capacity, or only sharpen what the base model could already sample?

**What we did.** `PLAN.md` (written 8 Oct, amended 9 and 13 Oct): eight phases, nine deliverables
(D1-D9), ten pre-registered questions (Q1-Q10), rigour standards (section 4). Decisions confirmed on
13 Oct: Qwen3-1.7B-Base only, Qwen3-0.6B-Base as the student; a budget of USD 100-170; one seed per
ablation plus one seed repeat of the main GRPO run; TRL with vLLM for training; Countdown as the
out-of-distribution task; a perturbed GSM8K as the reasoning-versus-recall control; GSM8K and
MATH-500 as the headline sets. `WORKFLOW.md`: one branch and one pull request per phase, merged and
tagged; every git and environment action logged with how to undo it.

**What it set up.** The habit that shaped everything later: write the hypothesis and the result
that would prove it wrong before running anything.

## 2. Phase 0 (14 Oct): a harness we can trust

**The idea.** Most "RL improved reasoning" numbers are single greedy runs with no CI, graded by an
extractor nobody inspects, from a sampler that is not deterministic. Before any model is compared,
the evaluation must reproduce bit for bit and its grading must be logged.

**What we did.** Built the environment on a laptop (RTX 4060, 8 GB; Windows with WSL2 Ubuntu, no
sudo): `uv` and a uv-managed Python, vLLM, TRL. Wrote from-scratch references with tests (unbiased
pass@k, group advantages, the k1/k2/k3 KL estimators, the GRPO loss with three aggregations and
clip-higher, a KV-cache sampler checked against `generate`) and the evaluation harness `rlvr-eval`
(eval sets with hashed manifests, four prompt templates, an extraction chain that records which
rule fired, bootstrap CIs, maj@k, pass@k, a refusal to run on a dirty git tree).

**What happened.**
- The environment fought back: Git Bash rewrote Linux paths, WSL given 12 GB starved Windows and
  the install was killed, the Windows driver was too old for CUDA 13 (updated), Triton needed
  Python headers that only a sudo install provides (solved with a uv-managed Python that ships them).
- The first smoke run found three harness bugs at once: MATH-500 completions were empty because
  the prompt ended on a token boundary the few-shot examples never show; the think-tag prompt ended
  on a token the base model had barely seen; vLLM stripped the `</answer>` stop string, so the
  extractor could never match.
- **EXP-000, the reproducibility test, failed**: three identical greedy evals gave GSM8K 68 / 70 /
  68 and MATH-500 25 / 27 / 31; the generated text itself differed on up to 69 of 100 problems.
  Cause: batched bf16 decoding is not deterministic. vLLM's batch-invariant mode made three runs
  bit-identical.

**What it changed.** Batch-invariant decoding became the default for every number in the project.
Prompt endings are checked at the token level. Long GPU jobs run from the user's own terminal
(background jobs in the assistant's shell were killed when memory ran low). PR #1, tag `phase-0`.

## 3. Phase 1 (14-18 Oct): where the base model stands, and how much is latent

**The idea.** Before training, know (a) the base model's numbers under several prompts, (b) that
the harness matches the Qwen3 technical report, (c) how much accuracy is reachable just by
sampling (the pass@k ceiling), and (d) whether GSM8K is memorised.

**What we did.** Greedy runs of both base models under four templates on the full sets (laptop,
1.2 GPU-hours); a perturbed GSM8K built from the dataset's own calculator annotations, regenerated
from a seed with a hash check; a partial-prompt probe (show 60 % of a problem, see if the model
completes the rest verbatim); the pass@64 ceiling (needed a rented GPU, so deferred to Phase 2's
rental).

**What happened.**
- Calibration passed: 4-shot CoT GSM8K 74.5 against the report's 75.4.
- **Zero-shot beat 4-shot CoT** by 3 points on GSM8K and 12 on MATH-500 (77.6 / 53.6 against 74.5 /
  41.4): the base model writes its own long reasoning and the few-shot examples pull it short.
- **The think-tag format, which the RL phases need, cost the base model badly**: 61.3 / 43.0, with
  36 % of GSM8K completions looping until the token limit.
- The perturbation gap was 10.1 points, above the pre-registered 8-point "memorised" line. Instead
  of concluding "memorised", a names-only control was added (change only names, which carry no
  arithmetic): it moved nothing (-0.4). The gap is brittleness to new numbers, not recall.
- A tempting grader rule ("take the number in the last 'So...' sentence") was measured on stored
  samples before adopting it: it fixed 6 extractions and broke 10. Rejected.
- Later (17-18 Oct, rented): pass@64 is 98.8 GSM8K, 87.8 MATH-500, 63.8 Countdown. The headroom is
  large: a method that raises greedy accuracy without raising pass@64 is only sharpening.

**What it changed.** Two bars every later model is held to: the base model's own best prompt
(zero-shot), and the loops of the trained format. PR #2, tag `phase-1`.

## 4. Phase 2 (15-19 Oct): SFT on distilled traces, and the loops

**The idea.** Supervised fine-tuning on competent reasoning traces teaches the think-tag format and
a style; the gain should be large in that format and small under the best prompt.

**Step 1, the data (15 Oct, laptop).** Filtered OpenR1-Math-220k to verified-correct traces under
4,096 tokens, deduplicated, decontaminated against every eval set (3 of 500 MATH-500 problems
dropped; a 13-gram detector at first flagged AIME24 as 27 % contaminated, which turned out to be
template sentences and was fixed), with a dev split from the training sets for choosing the
learning rate. 23,000 examples, 61.8 M tokens, hashed manifest.

**Step 2, a local smoke that found the deepest bug (15 Oct).** A 0.6B LoRA smoke ran end to end,
but **not one output contained `</think>`**. Chasing it: the tokens were not being stripped; the
model never generated them. The base model's embedding rows for `<think>` and `</think>` were
bit-identical to dozens of padding rows: **the Base checkpoint had never trained them**, and LoRA
leaves embeddings frozen, so the model could not learn to emit them. Fix: initialise the two rows
from their text pieces and train them (with a hook that leaves every other row frozen). Caught at
the same time, before any rented run: pure-bf16 AdamW at a 1e-5 learning rate rounds most updates
to nothing, so full fine-tunes now keep fp32 master weights; and a boolean attention mask was being
*added* to the scores instead of applied, caught by the test that packed and unpacked losses match.

**Step 3, the 4,096-token base reference (15 Oct).** Estimated at an hour, took 3 h 55 min (the
throughput at long context is a third of the short-context one). Two findings: the base's loops
never end (all 472 GSM8K completions truncated at 1,024 tokens were still running at 4,096), and
vLLM's context window was silently clipping "4,096 new tokens" to 4,096 minus the prompt, now
warned about and recorded in every run.

**Step 4, renting compute (16-18 Oct).** The provider: E2E Networks. A cheap L4 node (INR 49 an
hour) ran the environment build and four of the five pass@64 sets, and showed the long completions
would make Phase 2 cost about twice the plan. The planned A100 route needed a storage volume the
provider had to assign by hand; when that stalled, the user moved to a TIR H100 notebook (INR 379 an
hour), with its own disk used as the persistent volume. A Claude Code session on the node ran the steps under written rules, and the
laptop session wrote code and docs.

**Step 5, training (18-19 Oct, H100).** A three-point learning-rate sweep (1e-5 won), the main SFT
run (720 steps, 3 h), the evals, the data-scaling curve; LoRA was stopped after 20 steps to keep
Phase 3 inside the budget line; flash-attention had no wheel for this torch and was built from
source in 13 minutes for Phase 3.

**What happened.** SFT gained +13.0 [10.0, 15.8] on GSM8K and +5.8 on MATH-500 over the base in the
think-tag format, but **the trained format still lost to the model's own zero-shot prompt** (74.8 /
50.6 against 82.2 / 63.2), because **28 % of GSM8K and 52 % of MATH-500 think-tag completions never
terminate**. More data helped GSM8K (+4.0 [1.4, 6.7] from 25 % to 100 %) and did not teach stopping.

**What it changed.** The project's real question was written here: **is a later gain "better at
mathematics" or "better at stopping"?** It also shaped Phase 3: rejection sampling keeps only
finished correct traces, so it selects for stopping whether intended or not, and the controls
must separate that.

## 5. Phase 3 (18-19 Oct): one round of selection (RFT)

**The idea.** Sample the SFT model on its own training problems, keep the shortest correct
finished traces, train on them: cheap on-policy learning that should move greedy accuracy more than
pass@k. Compare against "the same amount of training on the distilled data", or it is just more
training.

**What we did.** Code and plan were written on the laptop before any Phase 2 number existed; the
same H100 session ran Phase 3 straight after Phase 2. Solve rates per problem (EXP-009) gave 4,606
learnable prompts (solved sometimes, not always) and a kept set of 11,145 traces; RFT trained in 12
minutes.

**What happened.**
- A harness bug: a flag equal to its default value lost to the config, so EXP-009 sampled at 2,048
  tokens instead of 4,096. Kept, recorded as a deviation, fixed with a test.
- RFT beat SFT by +3.3 [1.0, 5.8] / +5.8 [1.6, 10.2], truncation fell 28 to 15 % and 52 to 40 %;
  three seeds agreed. The same tokens of distilled data gained nothing.
- The node session was asked to read its own results as a reviewer and propose the smallest runs
  that could change a conclusion; five were approved: the loops do not finish even at 8,192 tokens
  (they are loops, not slow solutions); the RFT gain is above seed noise on GSM8K; the gain
  persists to k = 32 in the trained format, while under 4-shot CoT the SFT and RFT pass@k curves
  coincide.
- Found on the laptop afterwards: the "fresh data" controls had trained on data the SFT model had
  already seen, a design error. It is recorded where the number is quoted; the control supports
  "not more steps on the same data", not "not more data".

**What it changed.** RFT's gain looked like termination, but two mechanisms were mixed: the
correctness filter and the "shortest finished" filter. Two ablations were written to separate them
(EXP-010g, 010h). Phases 2 and 3 merged on 20 Oct (PR #4), tags `phase-2`, `phase-3`. First rental
total: 28.1 H100-hours, INR 10,640.

## 6. Phase 4 (19-21 Oct): GRPO with a verifiable reward

**The idea.** RL with a reward of 1 only for a finished, well-formed, correct answer should beat
selection (RFT), mostly as sharpening.

**Step 1, the price shock (19-20 Oct).** The plan's GRPO shape (64 prompts x 8 samples, 500 steps,
five runs) priced at the measured rates came to about 90 H100-hours, INR 34,000: the trained model
writes 1,400-2,600 tokens per completion. Three shapes were priced; the user chose shape B (32 x 8,
1,536-token cap, 200 steps; a main run, a seed repeat and three one-knob ablations) and raised the
ceiling to INR 22,000. The user's rules for the node came from watching Phase 2-3: compare on pinned
subsets, check settings before every launch, use the whole GPU, poll often, no idle gaps.

**Step 2, the correctness check (20 Oct, laptop).** Before renting, the TRL trainer was checked
against the from-scratch GRPO loss. Four runs to get a check with teeth: the first matched loss
values only because every reward was 0 (and at ratio 1 the loss value is 0 for any centred
advantages), so the check moved to comparing *gradients* under a mixed reward. In fp32 the gradients
agree to 1.4e-5. Also noted at pre-registration: with one optimiser step per batch the PPO ratio is
1, so DAPO's clip-higher cannot act in these runs.

**Step 3, session 1 (20-21 Oct, H100).** A slow package mirror cost 88 minutes before the first GPU
job. The debugging run ran out of memory twice at micro-batch 32 (the logits over a 152k vocabulary
are the limit); the user chose micro-batch 16 for every run. Then the two RFT ablations, the main
GRPO run (200 steps, 3.1 h) and the seed repeat.

**What happened.**
- GRPO beat RFT on every set: +4.1 [2.0, 6.1] GSM8K, +7.0 [2.8, 11.0] MATH-500, and Countdown
  doubled (+7.0) where the plan predicted no move. Stable: KL 0.006, no clipping, no hacking flag;
  the reward plateaued by step 50; 30 % of groups had zero variance throughout.
- The seed repeat landed 4.8 points lower on MATH-500 (predicted under 3): the noise row.
- Random instead of shortest traces cost RFT 2.4 points on GSM8K: half of RFT's stopping effect is
  the length filter. The unfiltered-RFT ablation had no power (its set was 93 % correct anyway).
- **The user asked: "How do you know it is termination?"** The first recap had said "mostly by
  learning to finish", an inference from correlations. So it was computed: `decompose_gain.py`
  splits the paired gain by whether the previous model finished each problem. **The whole gain
  sits on problems RFT did not finish** (+10.2 of +7.0 on MATH-500); on problems both finished the
  difference is 0.0 [-4.3, 3.9]. RFT's step decomposes the same way. This became the project's
  central result.

**Step 4, the cap test (21 Oct).** One reading remained: the 1,536-token cap, with a quarter of
completions cut and scored 0, might have turned the reward into a brevity reward. The user chose to
spend the third ablation's slot on a 2,048-token run (EXP-015). The same day the user set a rule:
no run without its entry in a new `docs/research-narrative.md` (what was known, the intuition, the
question, the prediction, the falsifier).

**Step 5, session 2 (21 Oct, H100).** KL off, dynamic sampling, and the 2,048 cap.
- KL off was predicted to land within noise; it landed above the main run (+5.6 on MATH-500), at
  half the length, 15 times further from RFT, and again only on truncated problems, with nothing
  under zero-shot.
- Dynamic sampling changed nothing: the learnable-prompt filter had already left almost no wasted
  groups.
- The 2,048 cap moved neither the both-finished difference (0.0) nor pass@16: the cap was not the
  limit.

**What it changed.** "Termination only" is a property of the method at this scale, data and budget.
No stage after SFT moved the best-prompt (zero-shot) accuracy beyond noise. The spend reached about
INR 21,750 of 22,000, so the next questions had to be free: if the gain is stopping, how cheaply can
stopping be bought? PR #5, tag `phase-4`.

## 7. Phase 5A (21-22 Oct, laptop, INR 0): what termination is worth

**The idea.** If training mainly taught the model to stop, (a) forcing the stop at decode time
should recover the gain, and (b) voting over several samples of an earlier model should reach the
RL policy's number at some token cost.

**What we did.** Budget forcing in the harness: when a completion reaches B tokens inside its think
block, append `</think><answer>` and let the model answer (EXP-016, ten laptop runs overnight).
Accuracy against generated tokens from the existing sample sets (EXP-017, no GPU).

**What happened.**
- Voting over two RFT samples matched GRPO's greedy number at 2.6x the tokens; four beat it. From
  four samples on, the voting curves of SFT, RFT and GRPO coincide.
- Forcing the stop on SFT recovered more than RFT's whole gain (first quoted as +5.8 / +9.0; see
  Phase 5B for the corrected number). The forced answers were right about half the time: the loops
  hold a usable partial derivation, not a finished one.
- Practical: the laptop lid was closed overnight and the machine slept; the detached job resumed on
  wake with nothing lost.

**What it changed.** The cost per point of termination: a stop rule INR 0, RFT 12 minutes of
training, GRPO 3.1 H100-hours. PR #6, tag `phase-5`.

## 8. The 22 Oct turn: corrections, a new plan, and an evidence index

**A correction from reading the code.** In a learning session the user read the grader and found
two false sentences in the docs: an extraction failure is scored *wrong* (the docs said "never
wrong"), and a truncated completion is not automatically wrong (the extractor falls back; SFT's
truncated GSM8K completions are right 31 % of the time). Both corrected with the measured numbers
in the reports and the model cards.

**A new plan for the rest.** `docs/plan-after-phase-4.md`: no further rental. Dropped, with reasons:
DPO (a free rule already buys the termination gain, and DPO on pairs sharing long prefixes is the
regime where it degrades both responses), distillation and serving (no thread from the data). Kept:
the two debts of Phase 5 (GRPO under the rule; the cross-machine pairing), a "Wait" variant, an
outcome reward model, and a trace analysis.

**The evidence index.** The user asked whether every claim could be shown with its examples. Built:
`configs/evidence/claims.yaml` lists each quoted claim with its runs and a selection rule;
`scripts/evidence.py` re-derives the number from the per-sample files, compares it with the quoted
one and writes the problems and excerpts behind it. 0 disagreements. PR #7.

## 9. Phase 5B (22-24 Oct, laptop, INR 0): the debts, "Wait", a verifier

**The idea.** (a) GRPO's remaining loops hold readable answers too, so forcing it should gain 1-3
points on MATH-500. (b) The same checkpoint decoded on the H100 and on the laptop should differ by a
few tenths of a point. (c) "Wait" before the stop gives a loop nothing it lacks. (d) A verifier
trained on the policy's own samples beats voting at small k and plateaus at large k.

**What happened.**
- **(b) failed, and it mattered most.** The same checkpoint flips 15-23 % of problems between the two
  GPUs (later 10-23 % across three checkpoints). Batch invariance is per machine, not across GPUs.
  Every comparison against an unforced model was re-paired on one machine: the stop rule on SFT is
  +4.6 [3.4, 6.0] / +7.2 [4.0, 10.4] (not +5.8 / +9.0), still more than RFT's same-machine gain, and the
  "cost of the rule" first reported was mostly this machine term.
- **(a) reversed once controlled.** Paired against the H100 run, GRPO under the rule read "no gain"
  and was recorded as the falsifier firing. After an unforced GRPO run on the laptop: +2.0 [0.6, 3.6]
  on MATH-500 at 27 % fewer tokens. And with RFT and GRPO both stopped at 2,048, GRPO's MATH-500 lead
  is gone: -0.4 [-4.4, 3.4].
- **(c) held.** "Wait" equals the plain stop at the same total tokens; 14-19 % of nudged loops
  closed on their own.
- **(d) held in shape.** A LoRA verifier on Qwen3-0.6B-Base, trained 5.8 hours on 20,000 of SFT's
  labelled samples: 89.4 % held-out accuracy, AUROC 0.944. The better of two RFT samples by its score
  beats GRPO's greedy number (+3.0 / +4.0); but as an arg-max over 32 samples it falls 12-13 points
  below a plain vote on MATH-500 (a weak verifier is over-optimised), while its scores as vote
  weights stay at or above voting.
- Practical: the first verifier run lost 6.4 hours when a library save hung on a network lookup
  inside WSL; weights are now written to disk first. CI turned out red on every Phase 5B push (a test
  needed a package CI does not install); `scripts/ci_local.sh` now reproduces CI before every PR.

**What it changed.** Every paired claim needs both sides on one machine, which the project had
assumed since Phase 0 without checking. GRPO and the stop rule are mostly substitutes.

## 10. Phase 6 (22-23 Oct, laptop CPU): what a loop is

**The idea.** Every result so far is about termination, but nothing had described the text. The
guess: a loop is a repeated chunk starting late; training changes how often loops happen, not what
they are.

**What happened.** 96-99 % of truncated GSM8K completions (76-90 % on MATH-500) contain a recurring
50-word window; the repeated unit is a 100-190-word step; **the loop starts early, at a fixed word
position** (about 500 on GSM8K, 800 on MATH-500) whatever the window, so more room adds cycles, not
progress. SFT's and GRPO's loops have the same shape (Kolmogorov-Smirnov distance 0.11-0.16):
training made the failure rarer without changing it. Under zero-shot the trained models rewrite their
whole solution after boxing the answer. A measure was added after the first run because the
pre-registered one misread the zero-shot shape; both are reported.

**What it changed.** The central result got a text-level mechanism: a trained model that fails to
stop re-derives one step from a fixed point; the on-policy stages made that rarer; a stop rule cuts
it. Phases 5B and 6 merged as PR #8 (tags `phase-5b`, `phase-6`); the laptop-only files went to the
Hub with a hash.

## 11. Phase 7 (24 Oct): the write-up, and one check of the whole

**The idea.** The exit criterion from the first day: someone can clone the repo, run one script and
reproduce the base-model table. On the same machine with the locked versions, it should reproduce
bit for bit.

**What we did.** `scripts/reproduce_baselines.sh` (fresh clone, fresh environment, the archived
reference from the Hub with its hash checked, the four base configs, a per-problem comparison).
The README rewritten around five findings, one table and one chart; `docs/results.md` with the plan's
ten questions answered; an interview packet, paper notes on eight papers checked against the papers,
drills and derivations; the registry and incident log indexed; the model cards extended.

**What happened.** **11,216 of 11,216 completions identical** in text, extracted answer and grade;
the table identical to the digit. The paper notes surfaced the cheapest missing control: GRPO with a
reward for finishing alone, which would show whether the correctness term taught anything beyond
stopping. PR #9 (squash-merged), tag `phase-7`.

## 12. The thread in one table

| phase | the question | the answer | what it changed |
|---|---|---|---|
| 0 | Does the harness agree with itself? | Not until batch-invariant decoding (6-point drift before) | Every number is batch-invariant |
| 1 | Where does the base model stand? | Zero-shot is its best prompt; think-tag loops 36 %; pass@64 headroom is large | Two bars: the best prompt, and the loops |
| 2 | What does SFT buy? | +13.0 in the format, but a quarter to half of completions never stop | The project's question: mathematics or stopping? |
| 3 | Does selection beat more SFT? | Yes (+3.3 / +5.8), and truncation falls | Two filters to separate (correctness, length) |
| 4 | Does GRPO beat selection, and how? | Yes (+4.1 / +7.0), all on problems RFT did not finish; robust to cap, KL, sampling | "Termination only" at this scale |
| 5A | What is termination worth? | A free stop rule and two votes reach the trained numbers | Cost per point: rule INR 0, RFT minutes, GRPO hours |
| 5B | Do the comparisons hold on one machine? | 10-23 % of problems flip across GPUs; one verdict reversed; a verifier helps at k = 2 only | Pair within one machine; GRPO and the rule are substitutes |
| 6 | What is a loop? | One step re-derived from a fixed point; training changed frequency, not shape | A text-level mechanism for the result |
| 7 | Does it reproduce? | Bit for bit, 11,216 of 11,216 completions | The exit criterion met |

## 13. What the process taught (the lessons that recur)

- **Compute, don't infer.** "Mostly by learning to finish" was an inference until the
  decomposition computed it; the machine term was an assumption until a control measured it.
- **Pre-register, and keep the failures.** About ten predictions failed (determinism, the machine
  term, the seed noise, Countdown, KL off, dynamic sampling, the cap, forced-answer accuracy, the
  verifier at k = 4, loop onset); each one is in the registry with what it changed.
- **A control is only a control if it is checked.** The Phase 3 controls trained on seen data; the
  first correctness check passed because the reward was always 0; the first GRPO pairing crossed
  machines.
- **Estimate from measured rates.** "About an hour" took four; the plan's GRPO shape was three times
  the budget; the long completions of the trained model drove every cost.
- **Small tooling rules, learned once each.** Long jobs detached or in the user's terminal; no
  backslashes through shell heredocs; LF line endings; never edit a running script; never `git add
  -A` in a checkout that holds personal notes; secrets only as environment variables; read CI after
  the first push.

## 14. What is still open

- **GRPO on harder prompts** (MATH levels 3-5, EXP-018): the experiment that could change the
  conclusion, pre-registered and unrun for budget.
- **A finished-only reward control**: would GRPO rewarded only for stopping reproduce most of the
  gain? Proposed in Phase 7.
- **Dr. GRPO's loss normalisation** (EXP-014c) and a **matched-incorrect RFT set** (EXP-010i):
  pre-registered and unrun.
- **The blog post**, paused by the user.

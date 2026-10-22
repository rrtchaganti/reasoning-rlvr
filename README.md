# reasoning-rlvr: where does post-training gain come from?

A controlled study of the reasoning post-training recipe on one small base model. Starting from
Qwen3-1.7B-Base, each stage of the recipe (SFT on distilled traces, rejection-sampling fine-tuning,
GRPO with a verifiable reward) is trained at a stated compute budget and scored the same way: greedy,
paired on shared problems, 95 % bootstrap confidence intervals, a pre-registered hypothesis and
falsifier per run, three contamination probes, a seed-noise row, and a compute ledger. The question
the study asks is not "does the recipe work" but **what each stage buys, and whether what it buys is
reasoning, sharpening, or something cheaper**.

**The answer so far (21 Oct 2026, four phases run, about 55 H100 node-hours).** Every stage beats the one
before it in the format it was trained in, with CIs clear of zero. Decomposed per problem, the whole
paired gain of both on-policy stages (RFT, then GRPO) sits on problems the previous policy failed to
terminate; on problems both policies finished, accuracy did not move. Sampling tells the same story:
pass@k on GSM8K is pure sharpening, and on MATH-500 GRPO reaches a few problems RFT never solved in 32
tries (+1.9 [0.4, 3.5] at k = 16), the one signal that is not termination. Against the base model's
own best prompt the three stages add +6 on GSM8K and +11 on MATH-500, so the pipeline did add
something a user could not get for free, but **at 1.7B and one GPU-day per method, on-policy
post-training in the trained format taught the model to stop looping and answer, and little else**.
Session 2 tested whether that was an artefact of the 1,536-token training cap: a 2,048-token cap, the
reference model off, and dynamic sampling each bought more or less of the same stopping and nothing
on the problems both policies finish (EXP-015's falsifier fired; section 4), so the result is a
property of the method at this scale, data and budget. No on-policy stage after SFT moved the
best-prompt accuracy beyond noise. Phase 5, with no rental, then asked what the termination gain is
worth: **a decoding rule that forces the stop on the SFT checkpoint recovers all of RFT's gain and
70-91 % of GRPO's at fewer tokens than either, on the RFT checkpoint it matches GRPO within the CI,
and voting over two to four samples of the earlier checkpoints matches or beats the GRPO policy's
greedy number** (sections 4 and 6). To within the CIs, the training taught what a stop rule provides.

Chain of questions across phases: [`docs/research-narrative.md`](docs/research-narrative.md). Plan
and rigour standards: [`PLAN.md`](PLAN.md). Pre-registration: [`experiments.md`](experiments.md).
Process: [`WORKFLOW.md`](WORKFLOW.md). Engineering notes and journals per phase:
[`docs/`](docs/README.md). Background: [`METHODS_AND_METRICS.md`](METHODS_AND_METRICS.md),
[`GLOSSARY.md`](GLOSSARY.md), [`PHASES_IO_AND_HYPERPARAMS.md`](PHASES_IO_AND_HYPERPARAMS.md).
Checkpoints and sample sets: [`Raviteja546`](https://huggingface.co/Raviteja546) on Hugging Face.

## 1. The finding in one table

Qwen3-1.7B-Base, greedy, think-tag format (the format the RL stages train in), 4,096-token window.
Each stage's gain is paired against the previous checkpoint on shared problems. "On truncated" is the
part of that gain that sits on problems the previous policy did not finish; "both finished" is the
paired accuracy difference restricted to problems both policies answered (`scripts/decompose_gain.py`,
`outputs/reports/phase-4.md` section 4.5).

| stage (experiment) | GSM8K gain | on truncated | both finished | MATH-500 gain | on truncated | both finished | truncation GSM8K / MATH-500 |
|---|---|---|---|---|---|---|---|
| base, think-tag | 61.8 | | | 44.8 | | | 36 / 20 % |
| SFT, 23k distilled traces (EXP-006) | +13.0 [10.0, 15.8] | not split | | +5.8 [0.6, 11.0] | not split | | 28 / 52 % |
| RFT, one round of selection (EXP-010) | +3.3 [1.0, 5.8] | +6.9 [5.2, 8.6] | -0.8 [-2.5, 0.9] | +5.8 [1.6, 10.2] | +12.2 [8.8, 15.6] | -2.5 [-7.0, 1.5] | 15 / 40 % |
| GRPO, 200 steps (EXP-012) | +4.1 [2.0, 6.1] | +4.2 [3.0, 5.5] | +1.1 [-0.6, 2.8] | +7.0 [2.8, 11.0] | +10.2 [7.4, 13.2] | 0.0 [-4.3, 3.9] | 5 / 21 % |
| ablation, KL off (EXP-014a), against the GRPO run | +2.2 [0.4, 4.0] | +2.0 [1.3, 2.7] | +1.0 [-0.7, 2.7] | +5.6 [1.6, 9.4] | +5.4 [3.4, 7.6] | +2.7 [-1.3, 6.5] | 2 / 13 % |
| ablation, 2,048 cap (EXP-015), against the GRPO run | +0.2 [-1.7, 2.2] | +2.4 [1.6, 3.3] | +1.1 [-0.6, 2.9] | -2.6 [-6.4, 1.2] | +3.2 [1.6, 4.8] | +1.2 [-3.0, 5.2] | 10 / 26 % |
| final, think-tag | 82.3 | | | 63.4 | | | |
| final, zero-shot (the model's best prompt) | 84.3 (base 77.9) | | | 64.4 (base 53.8) | | | |

Three facts make the decomposition a conclusion rather than a correlation. The truncated generations
are loops, not slow solutions: at an 8,192-token cap they still do not finish and accuracy moves by
under a point (EXP-010c). Half of RFT's stopping effect on GSM8K is the length filter itself: choosing
two random correct traces per problem instead of the two shortest costs 2.4 [0.1, 4.7] points
(EXP-010h). And the seed repeat of GRPO lands 4.8 points lower on MATH-500 for the same reason: it
truncates 34 problems RFT had finished, against 23 for seed 0, with no difference in accuracy among
finished answers (EXP-013). The 4.8 is the project's noise row on MATH-500, larger than most MATH-500
gains people report at this scale with one seed. Session 2 added two more reads: with the reference
model off the policy drifts 15 times further from RFT, halves its length and gains again only on
problems the main run truncated, with nothing under zero-shot (EXP-014a); with a 2,048-token cap the
both-finished difference and pass@16 do not move (EXP-015). Dynamic sampling changed nothing
(EXP-014b).

## 2. How the question arose, phase by phase

Each phase's question came from the previous phase's numbers; the record is in
[`docs/research-narrative.md`](docs/research-narrative.md), with the predictions written before each run.

1. **Phase 0.** Three identical greedy evals differed by up to 6 points because vLLM's default scheduler
   is not batch-invariant; with batch-invariant mode they are bit-identical. The harness is deterministic
   before any model is compared (EXP-000).
2. **Phase 1.** The base model does 12 points better on MATH-500 with a bare instruction than with four
   Minerva shots, and the think-tag format the RL stages need costs it 16 points on GSM8K because 36 %
   of generations loop. pass@64 is 98.8 / 87.8 / 63.8 (GSM8K / MATH-500 / Countdown), so the latent
   headroom is large everywhere (EXP-001, 002). Two bars were set here: every trained model is read
   against the base model's best prompt, and every gain in the trained format is read against the loops.
3. **Phase 2.** SFT on distilled traces teaches the format (+13.0 on GSM8K in that format) but the
   trained format still loses to zero-shot because a quarter to a half of its generations never
   terminate (EXP-006). Data scaling buys +4.0 [1.4, 6.7] on GSM8K from 25 to 100 % of the data
   (EXP-008). The question written at the end of this phase, "better at mathematics or better at
   stopping", became the project's question.
4. **Phase 3.** Rejection sampling keeps only finished correct traces, so it selects for stopping whether
   intended or not. The controls were designed for that: the same tokens of distilled data gain nothing
   (EXP-010b, 010e), the loops do not finish at 8,192 tokens (EXP-010c), and the gain persists to k = 32
   in the trained format but the SFT and RFT pass@k curves coincide under 4-shot CoT (EXP-010f). Two
   mechanisms were still confounded, the correctness filter and the length filter, and two ablations were
   added to separate them (EXP-010g, 010h).
5. **Phase 4, session 1.** GRPO from the RFT checkpoint, 32 prompts x 8 samples, 200 steps, reward 1 only
   for a finished, well-formed, correct answer. Beats RFT on every set; the per-problem decomposition
   above; a seed repeat for the noise row; pass@k for sharpening versus capability. The method was stable
   (KL 0.006, no clipping, no hacking signal), the reward plateaued by step 50, and 30 % of groups had
   zero variance throughout.

## 3. What every number carries

- **Paired comparisons with CIs.** Every difference between checkpoints is computed per problem on shared
  ids with a 95 % bootstrap CI over problems; a difference whose CI covers zero is reported as such.
- **A deterministic harness.** `rlvr-eval` runs batch-invariant vLLM; three identical evals are
  bit-identical. Extraction rules are unit-tested and logged. The 4-shot-CoT rows reproduce the Qwen3
  technical report within 1-2 points (EXP-001).
- **Pre-registration.** Each run has a hypothesis, a metric and a falsifier in `experiments.md` before it
  starts. Failed predictions are kept: Countdown doubled under GRPO when the plan said it would not move;
  the seed-to-seed noise on MATH-500 was 4.8 against a predicted 3; EXP-010g had no power by construction.
- **Controls that cost compute.** Token-matched and step-matched controls, a cap experiment, a
  no-correctness-filter run, a random-selection run, a seed repeat. One design error is recorded where
  the number is quoted: the Phase 3 controls trained on a seen subset of the SFT data, so they support
  "not more steps on the distilled distribution", not "not fresh traces".
- **Three contamination probes**, on the base model and after every stage: 13-gram and fuzzy overlap
  between every training set and every eval set (3 of 500 MATH-500 items dropped from training), a
  perturbed GSM8K twin with a names-only control (the 10-point gap is arithmetic brittleness, not
  recall), and a partial-prompt completion probe. No stage added a memorisation signal.
- **From-scratch references, checked against the library.** `src/rlvr/ref/` holds the GRPO loss with its
  three aggregations, group advantages, the k1/k2/k3 KL estimators, unbiased pass@k, the SFT loss and a
  minimal sampler, each under tests. The TRL + vLLM trainer used for the scaled runs matches the
  reference loss to 1e-9 and its gradients to 1.4e-5 in fp32 (`scripts/grpo_correctness_check.py`).
- **A reward-hacking watch** on every RL run, five flags read off the per-step statistics: length growth
  without accuracy, format-only outputs, groups agreeing on one wrong answer, "wait" spam, truncation
  rising; plus the KL, clip-fraction and zero-variance-group curves. Diagnoses go to
  `FAILURES_AND_LEARNINGS.md` with the plot that revealed each.
- **A compute ledger.** `COMPUTE.md` lists every run's GPU-hours, tokens and cost; the project stands at
  about 55 H100 node-hours and 11 L4-hours, about USD 260, against a stated ceiling of USD 262.
- **Reproducibility.** Every run is a config plus a commit hash; a preflight compares the resolved
  config with the plan's table, the prompt file with its committed manifest hash, the start checkpoint
  with the run it came from, and the eval subsets with their pinned hashes before a rented GPU starts;
  the harness refuses to run on a dirty tree; CI runs ruff and the CPU tests (193 in the gate) on every
  push.

## 4. What session 2 and Phase 5 found, and what comes next

**Session 2 (21 Oct 2026; `outputs/reports/phase-4.md` sections 10-18).** Three one-knob runs from the
same checkpoint, read against the noise row. The pre-registered prediction for the cap run was that the
both-finished paired difference against RFT would rise by at least 2 points and pass@16 by at least 3;
the falsifier was both inside the main run's CIs. Both-finished came out at 0.0 [-4.1, 4.1] and pass@16
at +0.8 [-0.6, 2.3], so the falsifier fired. The reference model off was predicted to land within
noise; it landed above the main run on three sets, at half the length, 15 times further from RFT, with
the gain again on problems the main run truncated and nothing under zero-shot. Dynamic sampling, built
for the third of groups with zero variance, changed nothing: the prompt filter had already left 32
informative groups in 199 steps of 200. Every claim and every failed prediction is in the registry.

**Phase 5, option A (21-22 Oct 2026, no rental; `outputs/reports/phase-5.md`).** Two free tests of
what the termination gain is worth. EXP-017, from the existing sample sets: voting over two RFT
samples matches the GRPO policy's greedy number at 2.6x the tokens and four samples beat it at 5x
(+3.9 [1.1, 6.7] GSM8K, +7.6 [3.8, 11.4] MATH-500); from four samples on, the majority-vote curves of
SFT, RFT, GRPO and the cap run coincide; the untrained base under a 4-shot prompt with 16 votes
matches GRPO's greedy GSM8K number and never its MATH-500 number. EXP-016, on the laptop: forcing
the stop (append the closing tags at a token budget, no training) on the SFT checkpoint gains +5.8
[3.9, 8.0] GSM8K and +9.0 [4.8, 13.0] MATH-500 at 1,536 tokens, more than RFT's whole gain and 70-77 %
of GRPO's cumulative gain; on the RFT checkpoint at 2,048 it stands at -1.4 [-3.4, 0.6] and +0.6
[-3.2, 4.4] against GRPO at fewer tokens. The forced answers are right about half the time, so the
loops hold a usable derivation, not a finished one. Both pre-registered hypotheses held and neither
falsifier fired. Figures: `outputs/reports/figures/phase-5-*.png`.

**What comes next, each in the narrative and the registry before it runs** (`docs/phase-5-plan.md`,
options B to D, awaiting a budget). The one training question left is whether harder prompts would
make GRPO teach what a rule cannot: GRPO on MATH levels 3-5 (EXP-018). Also priced: the
matched-incorrect rejection-sampling set (EXP-010i), DPO on finished-versus-looping pairs (EXP-019,
now of less interest since a free rule buys the termination gain), Dr. GRPO's normalisation
(EXP-014c). Two free cells remain on the laptop: the GRPO checkpoint under the stop rule, and the
laptop control run for the cross-machine caveat.

## 5. Status

| Phase | State | Report |
|---|---|---|
| 0 Harness, from-scratch references | merged 14 Oct 2026, tag `phase-0` | [`phase-0.md`](outputs/reports/phase-0.md) |
| 1 Baselines, pass@k ceiling, contamination probes | merged 14 Oct 2026 (EXP-002 completed 18 Oct), tag `phase-1` | [`phase-1.md`](outputs/reports/phase-1.md) |
| 2 SFT on distilled traces | runs 18-19 Oct 2026 on a TIR H100; merged 20 Oct, tag `phase-2`; checkpoint `Raviteja546/reasoning-rlvr-sft` | [`phase-2.md`](outputs/reports/phase-2.md) |
| 3 Rejection-sampling fine-tuning | runs 19 Oct 2026; merged 20 Oct (PR #4), tag `phase-3`; checkpoint `Raviteja546/reasoning-rlvr-rft` | [`phase-3.md`](outputs/reports/phase-3.md) |
| 4 GRPO and ablations | branch `phase-4-grpo`; session 1 run 20-21 Oct 2026 (EXP-010g/h, 012, 013), session 2 run 21 Oct (EXP-014a/b, 015); 26.7 node-hours, INR 10,130; checkpoint `Raviteja546/reasoning-rlvr-grpo` with the ablation finals under `ablations/`; PR pending | [`phase-4.md`](outputs/reports/phase-4.md) |
| 5 Test-time compute, the cost of termination | 5A merged 22 Oct 2026 (PR #6, tag `phase-5`): budget forcing (EXP-016) and accuracy against tokens (EXP-017), laptop and existing samples, INR 0, both hypotheses held. 5B proposed (`docs/plan-after-phase-4.md`): the GRPO checkpoint under the stop rule and the laptop control run (EXP-020), "Wait" before the stop (EXP-021), an outcome reward model against voting (EXP-022); laptop only | [`phase-5.md`](outputs/reports/phase-5.md) |
| 6 The traces: what a loop is and what GRPO removed | proposed (EXP-023), no GPU | |
| 7 Write-up and the interview packet | after Phase 6 | |
| dropped | DPO (EXP-019), distillation, quantisation and serving; EXP-018 (harder prompts) pre-registered and unrun by the budget decision; reasons in `docs/plan-after-phase-4.md` section 4 | |
| 8 Write-up | | |

## 6. Full results

Each number carries a 95 % bootstrap CI over problems and the commit that produced it; greedy,
batch-invariant vLLM. Paired tables are in `outputs/reports/phase-N-compare-*.md`.

**Base models (Phase 1, commit ffd3018).** pass@1 %, full eval sets.

| model | prompt | GSM8K | GSM8K perturbed | MATH-500 | Countdown |
|---|---|---|---|---|---|
| Qwen3-1.7B-Base | 4-shot-CoT | 74.5 [72.0, 77.0] | 67.8 [64.8, 71.1] | 41.4 [37.0, 45.8] | - |
| Qwen3-1.7B-Base | zero-shot | 77.6 [75.3, 79.9] | 70.0 [67.0, 73.1] | 53.6 [49.4, 57.6] | - |
| Qwen3-1.7B-Base | think-tag | 61.3 [58.7, 63.8] | 50.8 [47.6, 54.5] | 43.0 [38.8, 47.4] | 1.2 [0.4, 2.2] |
| Qwen3-0.6B-Base | 4-shot-CoT | 61.9 [59.3, 64.6] | 56.5 [53.1, 59.9] | 28.0 [24.0, 31.8] | - |
| Qwen3-0.6B-Base | zero-shot | 64.9 [62.3, 67.5] | 59.5 [56.1, 62.9] | 37.6 [33.4, 41.6] | - |

**After training** (Qwen3-1.7B-Base, greedy, 4,096-token window; paired differences against the previous
checkpoint on shared problems; 18-21 Oct 2026, TIR H100):

| checkpoint | prompt | GSM8K | GSM8K perturbed | MATH-500 | Countdown | truncated GSM8K / MATH-500 |
|---|---|---|---|---|---|---|
| base | think-tag | 61.8 [59.2, 64.4] | 51.4 [48.1, 54.8] | 44.8 [40.4, 49.2] | 1.2 [0.4, 2.2] | 36 / 20 % |
| SFT (EXP-006) | think-tag | 74.8 (+13.0 [10.0, 15.8]) | 66.0 (+14.6) | 50.6 (+5.8 [0.6, 11.0]) | 3.8 (+2.6) | 28 / 52 % |
| RFT (EXP-010) | think-tag | 78.2 (+3.3 [1.0, 5.8]) | 69.0 (+3.0) | 56.4 (+5.8 [1.6, 10.2]) | 7.0 (+3.2) | 15 / 40 % |
| GRPO (EXP-012) | think-tag | 82.3 (+4.1 [2.0, 6.1]) | 73.9 (+4.8 [1.9, 7.6]) | 63.4 (+7.0 [2.8, 11.0]) | 14.0 (+7.0 [3.6, 10.4]) | 5 / 21 % |
| GRPO seed 1 (EXP-013) | think-tag | 82.0 (-0.3 vs seed 0) | 73.6 (-0.2) | 58.6 (-4.8 [-9.0, -0.8]) | 14.6 (+0.6) | |
| base | zero-shot | 77.9 [75.6, 80.1] | 69.9 | 53.8 [49.4, 58.2] | - | 1-9 % |
| SFT | zero-shot | 82.2 (+4.3 [1.8, 6.6]) | 72.7 | 63.2 (+9.4 [5.0, 14.0]) | - | 17 / 59 % |
| RFT | zero-shot | 81.1 (-1.1) | 74.7 (+2.0) | 64.8 (+1.6) | - | |
| GRPO | zero-shot | 84.3 (+3.2 [1.5, 4.8]) | 74.7 | 64.4 (-0.4) | - | 16 / 65 % |
| GRPO, KL off (EXP-014a) | think-tag | 84.5 (+2.2 [0.4, 4.0] vs GRPO) | 77.1 (+3.3) | 69.0 (+5.6 [1.6, 9.4]) | 14.8 (+0.8) | 2 / 13 % |
| GRPO, KL off | zero-shot | 82.8 (-1.5 [-3.3, 0.3] vs GRPO) | 75.1 (+0.4) | 66.2 (+1.8 [-2.0, 5.4]) | - | 6 / 23 % |
| GRPO, dynamic sampling (EXP-014b) | think-tag | 84.0 (+1.7 [-0.1, 3.6] vs GRPO) | 74.8 (+1.0) | 61.4 (-2.0 [-6.0, 2.0]) | 12.0 (-2.0) | 4 / 22 % |
| GRPO, 2,048 cap (EXP-015) | think-tag | 82.5 (+0.2 [-1.7, 2.2] vs GRPO) | 72.3 (-1.6) | 60.8 (-2.6 [-6.4, 1.2]) | 12.4 (-1.6) | 10 / 26 % |

**pass@k.** 4-shot-CoT, n = 64, T = 0.6: base 98.8 / 90.2 / 87.8 / 63.8 (GSM8K / perturbed / MATH-500 /
Countdown under think-tag), SFT 98.2 / 88.8 / 93.0 / 62.0, RFT (300-problem subsets) 98.7 / 92.3 / 94.0.
In the trained format at T = 0.6, n = 32, on pinned 300-problem subsets: RFT's gain over SFT persists at
+2.0-2.5 to k = 32 on MATH-500; GRPO over RFT is -0.3 [-0.9, 0.3] at k = 16 on GSM8K (no problem newly
solved in 32 tries) and +1.9 [0.4, 3.5] on MATH-500 (10 problems newly solved, 2 lost); the 2,048-cap run
over RFT +0.8 [-0.6, 2.3] at k = 16 on MATH-500 (6 newly solved, 3 lost). Figures:
`outputs/reports/figures/`.

**Budget forcing (Phase 5, EXP-016; greedy think-tag, full sets, laptop).** SFT with the stop forced at
768 / 1,024 / 1,536 / 2,048 / 3,072 tokens: GSM8K 77.8 / 79.8 / 80.7 / 80.6 / 80.8 (752 to 1,581 mean
tokens), MATH-500 50.6 / 56.2 / 59.6 / 61.0 / 62.2 (776 to 2,419); RFT forced: GSM8K 78.2 / 79.8 /
80.8 / 80.9 / 80.9, MATH-500 51.6 / 58.0 / 61.8 / 64.0 / 64.6. Paired against GRPO's greedy 82.3 /
63.4: RFT forced at 2,048 -1.4 [-3.4, 0.6] / +0.6 [-3.2, 4.4]. Full table: `outputs/reports/figures/
phase-5-force-stop.md`.

**Controls.** Perturbation gap on GSM8K (numbers resampled, paired on 830 problems): 10.1 [7.0, 13.3] for
the base, 11.6 after RFT, 11.1 [8.2, 14.1] after GRPO; a names-only control moves nothing (-0.4 [-3.3,
2.5]). Partial-prompt verbatim completion: GSM8K 1.1 %, MATH-500 11.6 % (long held-outs), unchanged by
training. Phase 3 ablations: the same tokens of distilled data instead of the selected traces gain nothing
(74.8 to 74.1 GSM8K); unfiltered traces land on RFT but were 93 % correct by construction (EXP-010g, no
power); random instead of shortest traces -2.4 [-4.7, -0.1] GSM8K (EXP-010h). GRPO training: reward 0.34
to 0.68, finished rate 0.35 to 0.75, mean completion 1,406 to 1,201 tokens with p90 at the cap, KL 0.006,
clip fraction 0, zero-variance groups 30 % flat, 56 s/step on one H100. KL off: offline KL to RFT 0.098
per token against the main run's 0.006, finished rate 0.96, mean completion 611 tokens; dynamic
sampling: 43.5 informative groups of 64 per step, one zero-advantage fill in 200 steps, 85 s/step; the
2,048 cap: finished rate 0.82, p90 at the cap throughout, 72 s/step.

## 7. Repository map

| where | what |
|---|---|
| `src/rlvr/eval/` | the harness: datasets, prompts, extraction, metrics, bootstrap CIs, batch-invariant sampler |
| `src/rlvr/ref/` | from-scratch references: GRPO loss, advantages, KL estimators, pass@k, SFT loss, sampler |
| `src/rlvr/train/` | `rlvr-sft` (TRL SFT, packed, flash attention) and `rlvr-grpo` (TRL GRPOTrainer + vLLM colocate, dynamic-sampling subclass, hacking watch) |
| `src/rlvr/data/` | SFT and RFT data builders, contamination, perturbed GSM8K, partial-prompt probe |
| `scripts/` | per-phase session scripts (rental and laptop, resumable), preflight checks, paired compares, the gain decomposition, the accuracy-against-tokens and budget-forcing analyses, plots |
| `tests/` | about 200 CPU tests plus GPU-marked ones; `scripts/check.sh` is the gate before every commit |
| `experiments.md` | pre-registration: hypothesis, metric, falsifier, result, conclusion per run |
| `outputs/reports/` | one report per phase with every table, plus the paired compare tables and figures |
| `docs/` | research narrative, per-phase engineering notes and journals, rental and handoff procedure |
| `cards/` | model cards with config, data, compute and the eval table with CIs |
| `COMPUTE.md`, `FAILURES_AND_LEARNINGS.md` | the ledger and the incident log |

## 8. Reproduce

```bash
# inside WSL2 Ubuntu or any Linux box with a CUDA GPU; no sudo needed
bash scripts/setup_env.sh            # uv + uv-managed Python 3.12 + vllm/trl + dev tools, writes requirements-lock.txt
source ~/venvs/rlvr/bin/activate
bash scripts/check.sh                # ruff + fast CPU tests (the pre-commit gate)
python scripts/triton_check.py       # GPU sanity: a Triton kernel compiles and runs
bash scripts/phase0_smoke.sh         # Phase 0 exit check on Qwen3-0.6B-Base
rlvr-eval --help                     # score any HF checkpoint on GSM8K, perturbed GSM8K, MATH-500, Countdown, AIME24
```

CPU-only (what CI does): `bash scripts/setup_env.sh --cpu` then `pytest -q -m "not gpu"`. The rented-GPU
sessions are `scripts/phase{1,2,3,4}_rental.sh`, each step resumable, with `docs/rental-setup.md` and
`docs/e2e-session-handoff.md` as the procedure.

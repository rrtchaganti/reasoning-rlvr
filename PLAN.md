# `reasoning-rlvr`: a 1.7B reasoning model trained end to end, to the standard a top lab would hold it to

Project 1 of `side_projects_plan.md`. Status: **decisions confirmed 13 Oct 2026, build starts 14 Oct**. Nothing built yet.
Written 8 Oct 2026, amended 9 Oct (contamination, headline metric) and 13 Oct (section 11 confirmations, local-GPU plan, one seed per ablation, new dates) and 20 Oct (section 11 item 2: the ceiling raised to INR 22,000 for Phase 4 at the shape of `docs/phase-4-plan.md`). The GRPO-for-Spark-TTS project is paused; its GRPO loop and failure-log habit carry over here.

---

## 0. What this project has to prove, and to whom

Interviewers at Anthropic and Google DeepMind are not going to be impressed that a 1.7B model got better at GSM8K. Thousands of people have run a GRPO notebook. What they screen for, across every public account of their loops (section 2), is whether the person can:

1. design a controlled experiment and reach a conclusion they and others can trust (that sentence is lifted from an Anthropic RL engineering posting),
2. implement the core pieces (attention, a loss, a sampler, a training loop, an evaluator) from scratch, correctly, under time pressure,
3. debug a training run from its logs and metrics,
4. read a paper critically: contribution, method, what the evidence does and does not support, what they would do next,
5. make engineering judgement calls under compute and time constraints, and say which tradeoffs they accepted,
6. write it up so that someone else can reproduce it.

So the deliverable is not "four checkpoints". It is **a small, fully reproducible study of the post-training recipe for reasoning (SFT, rejection sampling, RLVR, test-time compute, distillation) on one base model, with error bars, honest baselines, a contamination check, ablations that change one thing at a time, and a written record of every decision and every failure.** The checkpoints are a by-product.

The through-line to your background: this is the same GRPO loop you ran in production for TTS with WER and speaker-similarity rewards, now with a text-verifiable reward, and the same evaluation discipline (normalised metrics, held-out judges) carried over.

---

## 1. Objective and deliverables

**Objective.** Starting from one small base model (Qwen3-1.7B-Base), measure how much mathematical reasoning accuracy each stage of the modern recipe buys at a fixed, accounted compute budget, and whether reinforcement learning with verifiable rewards adds capability or only sharpens sampling. Every number carries a confidence interval(CI). Every comparison is paired on the same problems. Every run is reproducible from a config and a commit hash.

**Deliverables (definition of done).**

| # | Artefact | Standard |
|---|---|---|
| D1 | `rlvr-eval`, an evaluation harness | One command scores any HF checkpoint on GSM8K-test, MATH-500 and the OOD task; outputs JSON with per-problem results, pass@1 (greedy), unbiased pass@k for k in {1,4,16,64}, majority-vote@k, mean/median/p90 response length, truncation rate, tag-format rate, extraction-failure rate, and 95 % CIs (bootstrap over problems). Deterministic given seed and vLLM version. Unit-tested answer extraction. |
| D2 | Base-model baseline report | Zero-shot, 4-shot, 4-shot-CoT, and "think-tag" prompted baselines on all three evals with CIs, plus pass@k curves. This is the bar every trained model is compared against. |
| D3 | Contamination and memorisation report | Three probes, all run on the base model before training and on every checkpoint after: (a) 13-gram and fuzzy overlap between every training set used (SFT traces, RFT prompts, GRPO prompts) and every eval set, with overlapping items removed and counted; (b) partial-prompt completion test on GSM8K-test and MATH-500 (first 60 % of the problem shown, verbatim continuation rate measured); (c) perturbation gap: accuracy on original GSM8K items vs GSM-Symbolic / GSM-Plus style rewrites of the same items. Section 4.3. |
| D4 | Four checkpoints on Hugging Face | `sft`, `rft`, `grpo`, `distilled-0.6b` with model cards: training config, data, compute in GPU-hours, eval table with CIs, known failure modes. |
| D5 | GRPO ablation study | Three one-knob ablations, one seed each, plus a seed-repeat of the main GRPO config that gives the seed-to-seed noise row; reward / length / entropy / KL / eval-accuracy curves, paired comparison with CIs, and a written conclusion that states what the evidence supports and what it does not. |
| D6 | Test-time-compute chart | Accuracy vs generated tokens for greedy, majority vote k=1..64, best-of-N with a trained ORM, budget forcing, and the RL policy, on the same problems, same axes, with CIs. |
| D7 | Distillation and serving table | 0.6B student: accuracy loss vs the 1.7B teacher, AWQ 4-bit accuracy loss, vLLM throughput and latency at several batch sizes. |
| D8 | From-scratch reference implementations with tests | GRPO loss (token- and sequence-level), unbiased pass@k estimator, group advantage, k3 KL, answer normaliser, a minimal sampling loop with KV cache. Verified against the library implementation on a tiny run. |
| D9 | Write-up | README with every table; `FAILURES_AND_LEARNINGS.md`; a blog post; a 2-page "interview packet" (section 10). |

**Scope guard.** One base model. Math only (plus one OOD task). At most three GRPO ablations. No AIME chasing. No multi-node. If a stage overruns by more than one day, ship what exists with a plot and move on.

---

## 2. What top labs test, and how this project answers it

Sources are third-party guides and candidate reports (section 12); the loops change, so verify with the recruiter. The consistent signals:

| What the loop probes | Where it shows up | How this project trains and evidences it |
|---|---|---|
| **Controlled experiments, trustworthy conclusions** (Anthropic RL postings: "design controlled experiments and reach conclusions you and others can trust") | research discussion, hiring-manager judgement screen | Fixed eval protocol (section 4.2), paired bootstrap CIs, one seed per ablation with a measured seed-noise reference (section 4.5), one knob per ablation, pre-registered hypotheses in `experiments.md` before each run, negative results reported |
| **ML coding from scratch, no libraries, runs by the end** (DeepMind ML-coding round: attention, loss, sampler, training loop; Anthropic: build a small working system to a spec, refactor as constraints change) | timed coding rounds | D8 reference implementations written first, with tests, then compared to TRL/verl. The eval harness is written to a spec and refactored as evals are added. Weekly 45-minute drills derived from the project (section 10.2) |
| **Debugging a training run from logs** ("diagnose ML failures from logs and metrics") | ML coding/debugging round | Every run logs reward, length, entropy, KL, grad norm, clip fraction, zero-advantage-group fraction, truncation rate; `FAILURES_AND_LEARNINGS.md` records each diagnosis with the plot that revealed it |
| **Paper discussion**: contribution, method, limits, extension | paper round (both labs) | Section 10.1: five papers whose claims this project tests directly (GRPO/DeepSeekMath, DAPO, Dr. GRPO, "Does RL incentivize new reasoning" (Yue et al.), s1 budget forcing). You will have your own data on each claim |
| **Math and ML fundamentals** (probability, optimisation, RL theory: MDPs, policy gradient, variance reduction) | DeepMind skills/quiz round | Section 10.3 derivation list, each tied to a plot: REINFORCE to PPO to GRPO, why group mean is a baseline, bias of per-sequence length normalisation, why clipping bounds the update, k1/k2/k3 KL estimators, unbiased pass@k |
| **Engineering judgement and tradeoffs** | hiring-manager screen, systems design | Compute ledger (section 4.6): every run's GPU-hours and cost; decisions table with alternatives rejected (section 3) |
| **Communication** | everything | README tables, model cards, one chart per question, a blog post, and the 2-page packet |
| **Safety and values** (Anthropic) | values round | Not a math project's job, but: honest reporting of reward hacking and of what RL did *not* achieve, and a short section on evaluation gaming and why contamination checks matter, are the right instincts to show |

---

## 3. Decisions (with alternatives rejected)

| Decision | Choice | Alternatives rejected and why |
|---|---|---|
| Base model | **Qwen3-1.7B-Base**, confirmed 13 Oct; no newer-base check, the model is fixed for the whole study | Qwen3-1.7B (post-trained): already RL'd on math, so gains would be unreadable. Llama-3.2-1B: weaker math base, fewer tokens of headroom. 4B: doubles RL cost; 1.7B keeps every run on one GPU |
| Student | Qwen3-0.6B-Base | same family, same tokenizer, clean distillation story |
| RL algorithm | GRPO with verifiable reward; ablate DAPO (clip-higher + dynamic sampling) and Dr. GRPO (no length/std normalisation) | PPO with a value model: not needed for binary outcome rewards and doubles memory; RLOO/REINFORCE++: fine, but GRPO is the vocabulary interviewers use |
| RL library | **Own reference implementation** (from the TTS project, ~300 lines) for correctness checks and small runs; **TRL `GRPOTrainer` with vLLM colocate** for the scaled runs. verl is the fallback if TRL's vLLM path breaks on the rented box | Only-library: cannot answer "what does your loss do". Only-own: too slow for 1k-token rollouts at G=8 |
| Reward | `math-verify` exact match on the boxed/`<answer>` content, 1/0, plus +0.1 for correct tag format (ablate removing it) | LLM judge: not verifiable; PRM as reward: known to be hacked, kept for test-time best-of-N only |
| SFT data | OpenR1-Math-220k (or the current best open distilled set), filtered to correct, ≤ 4k tokens, deduplicated against all evals | Self-generated only: the base model's traces are too weak to teach format and style |
| Headline metric | **MATH-500, reported by difficulty level 1-5**, paired CIs. GSM8K-test is the secondary, "did we break anything" number. Reason: Qwen3-1.7B-Base already scores about 75 on GSM8K 4-shot-CoT (technical report) and its stage-2 pre-training used trillions of synthetic math tokens from Qwen2.5-Math, so GSM8K has little headroom and is contamination-suspect; MATH has 40-55 points of room and a level axis that shows *where* a method helps | GSM8K as headline: most GRPO groups would be all-correct (a community run on this exact base logged 304 of 500 steps with zero reward variance) and gains would be mostly format elicitation |
| Eval sets | GSM8K test (1319), MATH-500, a **perturbed GSM8K set** (GSM-Symbolic / GSM-Plus style rewrites of the same items) as the memorisation control, and **one OOD task**: Countdown (number game, programmatic verifier) | AIME24: 30 problems, near-zero at 1.7B, CI width makes it meaningless as a primary metric; kept as a sanity row only |
| Generation for eval | vLLM, temperature 0 for pass@1; T = 0.6, top-p 0.95 for pass@k with k=64 samples, fixed seed | HF generate: too slow for 64 samples × 1819 problems |
| Tracking | W&B project `reasoning-rlvr`, plus `experiments.md` (hypothesis, config hash, result, conclusion) committed to git | Only W&B: not reviewable offline |
| Compute | Local RTX 4060 Laptop (8 GB) inside WSL2 Ubuntu: harness, unit tests, greedy and small-k evals, contamination probes, the own-loop-vs-TRL correctness check on 0.6B, ORM LoRA, AWQ and latency benchmarks (section 3.1). Rented single A100-80GB / H100: all 1.7B training, 0.6B full fine-tune, pass@64 sampling | Multi-GPU: out of scope, adds failure surface without changing conclusions |

### 3.1 What the local 8 GB GPU can and cannot do (measured 13 Oct 2026)

Hardware as found: RTX 4060 Laptop GPU, 8188 MiB; WSL2 Ubuntu with the GPU visible, Python 3.12.3, CUDA 12.6 runtime libraries, no conda or uv yet; WSL memory at the default 7 GB (raise to 12 GB in `.wslconfig` before installing vLLM). vLLM runs only on Linux, so the whole project lives in WSL2 with one environment that matches the rental image; Windows is for editing.

Memory arithmetic for Qwen3-1.7B in bf16: weights about 3.4 GB; KV cache 112 KB per token (28 layers × 8 KV heads × 128 head-dim × K and V × 2 bytes); with the display and CUDA context taken out, about 3.5 GB is left for KV, roughly 30k cached tokens, so 7 concurrent sequences at 4096 tokens or about 30 at 1024. Decode is bandwidth-bound (about 256 GB/s): roughly 70 tokens/s single-stream, 0.5-1.5k tokens/s batched. Set `gpu_memory_utilization` near 0.85 and `max_model_len` 4096.

| Runs locally | Rented |
|---|---|
| Greedy evals of every eval set for any 1.7B or 0.6B checkpoint (about 1.5M generated tokens, under an hour) | n = 64 sampling for pass@k and maj@k (about 60M tokens per checkpoint on MATH-500 + GSM8K: 15-30 h locally, about 2 h on an A100) |
| pass@k with n = 8 for quick checks | SFT and RFT of 1.7B, every GRPO run, DPO |
| Contamination probes (overlap is CPU; partial-prompt completion is short greedy decoding) | Full fine-tune of 0.6B (AdamW states alone are about 10 GB); LoRA on 0.6B fits locally |
| Phase 4 step 1 correctness check: own loop vs TRL on Qwen3-0.6B-Base with LoRA, 50 prompts, 256-token completions, HF generate rather than vLLM colocate | Batch-32 serving benchmark for 1.7B (KV does not fit at 4k context) |
| ORM as LoRA on 0.6B; AWQ of 0.6B; vLLM latency at batch 1 and 8 for 0.6B and 1.7B; majority vote, best-of-N and budget-forcing post-processing from saved samples | |

Consequence for the schedule: the greedy rows of Phase 1 and the base-model probes run locally on 15 Oct before any rental; the first rented day is the n = 64 sampling only.

---

## 4. Rigour standards (the attention-to-detail section)

These are the rules every stage follows. They are the difference between a tutorial and a study.

### 4.1 Reproducibility
- Every run has a YAML config, a git commit hash, a seed, the library versions (`pip freeze` saved), the exact prompt template, and the vLLM version. All written into the run directory and the W&B config.
- Data is versioned by content hash; the exact filtered SFT/RFT/GRPO prompt files are saved with the run.
- Evaluation is deterministic at temperature 0 up to vLLM batching nondeterminism; report the observed run-to-run variation once (three identical greedy evals) so it is known. *Done, EXP-000 (14 Oct):* default mode moved pass@1 by 1-6 points on 100-problem subsets and changed the text on 16-69 % of problems; the harness therefore runs every vLLM eval with `VLLM_BATCH_INVARIANT=1`, under which three runs are bit-identical. Reported numbers are a function of (commit, config, seed).

### 4.2 Evaluation protocol (fixed before any training run)
- **Prompt template** is fixed per model family and written in the harness; the same template is used for every checkpoint. Changing the template is an experiment, not a tweak.
- **Answer extraction**: `math-verify` on the `<answer>` block; fall back to the last `\boxed{}`; count extraction failures separately and never silently as wrong. Unit tests cover fractions, decimals, negative numbers, units, percent, equivalent forms (`0.5` vs `1/2`), and trailing text.
- **Max new tokens** is fixed (4096) and the truncation rate is always reported; a model that gets truncated more is not "worse at math", it is longer, and the report must say which.
- **Metrics**: greedy pass@1; unbiased pass@k (Chen et al. 2021 estimator) from n = 64 samples at T = 0.6; majority vote@k from the same samples; mean and p90 tokens. Report per-dataset and per-MATH-level.
- **Uncertainty**: 95 % bootstrap CI over problems for every number. For comparisons between two models, **paired** bootstrap on the per-problem correctness difference, because the problems are shared. Binomial standard errors at the sample sizes in play (GSM8K 1319, MATH-500 500) are about ±1.2 and ±2.2 points, so single-digit differences on MATH-500 are not conclusions without pairing and seeds.
- **Seeds**: one seed per run (confirmed 13 Oct). The main GRPO config is run twice so the seed-to-seed difference is measured once and reported as the noise row of the ablation table; an ablation delta smaller than that row is reported as inconclusive, not as a finding. Headline recipe stages use 1 seed plus the eval CI, stated as such.

### 4.3 Contamination and memorisation
What is known (verified 9 Oct 2026 from the Qwen3 technical report and model card): Qwen3-1.7B-Base is pre-trained only, no SFT or RL, but its 5T-token "reasoning" pre-training stage up-weighted STEM and synthetic data, and trillions of synthetic tokens (textbooks, Q&A pairs, instructions) were generated with Qwen2.5-Math, a model itself trained on GSM8K and MATH. The report contains no decontamination statement. Independent work ("Spurious Rewards", "Reasoning or Memorization?", and a 2026 follow-up on Qwen3-8B) attributes part of RLVR gains on Qwen bases to memorised benchmarks. OpenR1-Math-220k, our SFT source, has been audited to contain roughly 20-30 MATH-500 problems. So contamination is a measured quantity in this project, not an assumption.

Three probes, each cheap, each run on the base model in Phase 1 and on every checkpoint afterwards:
1. **Training-data overlap.** 13-gram overlap and normalised fuzzy match (indel similarity > 0.9) between each training set (SFT traces, RFT prompts, GRPO prompts, distillation traces) and each eval set. Overlapping training items are dropped and the counts reported. Mandatory for OpenR1-Math-220k.
2. **Partial-prompt completion.** Show the model the first 60 % of each GSM8K-test and MATH-500 problem (no instruction, plain completion) and measure the rate at which it reproduces the remaining text near-verbatim (ROUGE-L > 0.7 against the true continuation). A memorised item gets completed; a fresh one does not. Reported as a rate per eval set, with ten examples.
3. **Perturbation gap.** Build a perturbed GSM8K-test set by changing names and numbers while keeping the solution logic (GSM-Symbolic / GSM-Plus style, generated by templates and verified by re-solving). Report accuracy on original vs perturbed for every checkpoint. A gap that widens after training means the recipe is rewarding recall; a gap that stays flat while both rise means it is rewarding reasoning.

What this changes in the write-up: GSM8K is never the headline; MATH-500 by level, the perturbation gap and Countdown carry the generalisation claim; and every "RL improved X" sentence is accompanied by the probe results for that checkpoint. This cannot fix what the base model has seen, and the write-up says so.

### 4.4 Baselines
- The strongest cheap baseline is always present: the base model with 4-shot CoT prompting, and majority vote over the base model at equal token budget. "RL beat the base" means nothing if prompting plus voting matches it.
- Compute-matched comparisons: SFT vs RFT vs GRPO reported at equal GPU-hours, and separately at "best effort".

### 4.5 Ablation discipline
- One knob per ablation. Pre-registered in `experiments.md`: hypothesis, metric, what result would falsify it. One seed each, compared against the measured seed-noise row from the main-config repeat (section 4.2). Curves for reward, response length, entropy, KL (to the SFT/RFT reference), clip fraction, fraction of zero-advantage groups, truncation rate, eval accuracy every N steps.
- Conclusions are written in the form "at this scale and budget, X; this does not tell us Y".

### 4.6 Compute and cost ledger
- `COMPUTE.md`: one row per run with GPU type, hours, cost, tokens generated, tokens trained. Final table: accuracy per GPU-hour by method. Interviewers ask "what did it cost" and "what would you do with 10×".

### 4.7 Code quality
- `src/rlvr/` package, `pip install -e .`, type hints, docstrings that say *why*, `pytest` for extraction, pass@k, advantage, loss, and KL; `ruff` clean. No notebooks in the critical path; notebooks are for inspection only.

### 4.8 Honesty
- Negative and null results stay in the README. Every reward-hacking incident, divergence, and bug that changed a number goes into `FAILURES_AND_LEARNINGS.md` with the plot or sample that exposed it.

---

## 5. Phases

Each phase lists: objective, steps, exit criterion, what usually goes wrong, and what you can say in an interview afterwards.

### Phase 0: Scaffold, harness, from-scratch references (2 days, local GPU in WSL2)

**Objective.** A tested evaluation harness and tested reference implementations exist before any model is trained, so that every later number is produced by code you trust.

**Steps.**
1. Repo layout (section 7), env inside WSL2 Ubuntu (`uv` or `conda`; `torch`, `transformers` ≥ 4.51 for Qwen3, `vllm`, `trl`, `math-verify`, `datasets`, `wandb`), `.wslconfig` memory raised to 12 GB, `pre-commit` with `ruff`. The same `requirements.txt` is what the rental box installs.
2. `rlvr/eval/`: dataset loaders (GSM8K test, MATH-500, Countdown generator with verifier, AIME24 sanity), prompt templates, vLLM sampler, extraction, scoring, bootstrap CIs, JSON report, Markdown table writer.
3. `rlvr/ref/`: from scratch, with tests: `pass_at_k(n, c, k)` unbiased estimator; `group_advantages`; `grpo_loss` with switches for token- vs sequence-level averaging and std normalisation on/off; `kl_k1/k2/k3`; `clip_fraction`; a minimal single-sequence sampling loop with KV cache over an HF model (used to explain decoding, not for speed).
4. Contamination script (13-gram + normalised exact match).
5. Run the harness on Qwen3-0.6B-Base locally to shake out bugs.

**Exit.** `pytest` green; harness produces a report for 0.6B-Base with CIs; three repeated greedy evals agree to within the reported variation.

**What goes wrong.** Extraction bugs that silently mark correct answers wrong (always inspect 50 random "wrong" samples by eye); vLLM version pinning; chat-template mismatches for a *base* model (use a plain completion prompt, not the chat template).

**Interview value.** "I wrote the pass@k estimator and the GRPO loss myself and tested them against TRL on a toy run" is a direct answer to the ML-coding round.

### Phase 1: Baselines and the pass@k ceiling (1 day, rented GPU)

**Objective.** Know exactly what the base model can do before touching it, including how much capability is latent in sampling.

**Steps.**
1. Qwen3-1.7B-Base: zero-shot, 4-shot, 4-shot-CoT, think-tag prompt, on GSM8K-test, perturbed GSM8K, MATH-500 (by level), Countdown, AIME24 sanity. Greedy.
2. Sampled: n = 64 at T = 0.6 on all evals; pass@k curve k = 1..64 and majority-vote@k curve. Token counts.
3. Save every sample (needed later for DPO pairs and for the "did RL change the distribution" analysis).
4. **Contamination probes on the base model** (section 4.3): partial-prompt completion rate on GSM8K-test and MATH-500; perturbation gap on GSM8K; training-data overlap report for every training file prepared so far. These are the baseline values every later checkpoint is compared against.
5. Reference points from the Qwen3 technical report (4-shot CoT): GSM8K 75.44, MATH 43.50 (full MATH test, not MATH-500) for the 1.7B base; 59.59 and 32.44 for the 0.6B base. If our 4-shot-CoT numbers land more than about 3 points away, the harness or the prompt is wrong, not the model.

**Exit.** D2 report. One figure: pass@k and maj@k vs k for all evals. D3 baseline rows filled.

*Status, 14 Oct 2026.* Steps 1, 4 and 5 done locally (D2 report, D3 rows, calibration passed). Steps 2 and 3 (n = 64 sampling, the figure, the saved samples) are marked "not met, because no rented GPU yet" at merge time and run first in the Phase 2 rental session (EXP-002 in `experiments.md`).

**Interview value.** The pass@64 ceiling is the reference point for the central question of the project (Yue et al.): if GRPO's pass@1 ends up below the base model's pass@64, RL sharpened; if pass@64 itself rises, RL added.

### Phase 2: SFT distillation (1.5 days)

**Objective.** Teach the format and a competent reasoning style from distilled traces, and measure what SFT alone buys.

**Steps.**
1. Data: filter OpenR1-Math-220k (or current best) to verified-correct traces, ≤ 4k tokens, dedup, drop eval-overlapping items (contamination report D3), rewrite into `<think>…</think><answer>…</answer>`. Report the size and length distribution.
2. Train 1.7B: full fine-tune in bf16 with gradient checkpointing and packing with cross-contamination masking; loss on assistant tokens only; LR sweep over 3 values on a 10 % subset first; 2 epochs; eval every 500 steps so the best checkpoint is chosen on a dev split (a held-out 500 of GSM8K-train + MATH-train), **not** on the test sets.
3. Also LoRA r=64 at the same budget, one run, to have the LoRA-vs-full number.
4. Ablate SFT data size (25 % / 50 % / 100 %) with one seed each: the data-scaling curve.

**Exit.** `sft` checkpoint on HF with card; table: base vs SFT on all evals with paired CIs; pass@k curve for SFT; data-scaling plot.

**What goes wrong.** Packing without proper attention masks (cross-contamination between packed samples) inflates loss; learning rate too high erases base knowledge (watch MATH level 1-2 accuracy); traces longer than the max length truncated mid-answer (filter by length first).

**Interview value.** Assistant-only masking, packing masks, LR sensitivity, the data-scaling curve, and why SFT on distilled traces is the dominant lever for small models (the R1 distillation finding).

### Phase 3: Rejection-sampling fine-tuning, expert iteration (1 day)

**Objective.** Show that RL with a binary reward can be done with no policy-gradient machinery, and measure how far it gets.

**Steps.**
1. From `sft`, sample 8 per problem on GSM8K-train + MATH-train (levels 1-3), T = 1.0. Keep correct traces; cap at 2 per problem to avoid easy-problem domination; log the per-problem solve rate distribution (this is also the GRPO prompt-selection input).
2. Retrain from `sft` on the kept traces (one iteration). Optionally a second iteration if time allows; report the diminishing return.
3. Compare to SFT at equal GPU-hours.

**Exit.** `rft` checkpoint and card; table and pass@k curve; solve-rate histogram.

**Interview value.** STaR/ReST/expert iteration vs policy gradient; why RFT loses the information in wrong samples; the solve-rate histogram as the reason dynamic sampling exists.

### Phase 4: GRPO with verifiable reward, plus three ablations (3.5 days)

**Objective.** The central experiment. Train the RL policy, then answer with evidence which GRPO components matter at this scale, and whether RL added capability or sharpened sampling.

**Steps.**
1. **Correctness check first.** Run the own reference GRPO loop and TRL `GRPOTrainer` on the same 50-prompt toy set with identical seeds and settings; the per-step losses and rewards should agree closely. Record the result. This is the day you can later say "I verified the library against my own implementation".
2. **Debugging run on GSM8K-train first** (50-100 steps): the base is already strong here, so this is where the reward function, tag parsing, vLLM sync and the logging are shaken out cheaply, and where the zero-variance-group fraction is measured. Its accuracy numbers are not reported as results.
3. **Main run** from `rft`: prompts = **MATH-train levels 1-3** filtered to solve rate in (0, 1) from Phase 3 (learnable prompts), plus a 20 % slice of GSM8K-train with solve rate in (0, 1) to keep the easy distribution covered; G = 8; 64 prompts per step; LR 1e-6; KL beta 0.04 to the `rft` reference; max 2048 new tokens; T = 1.0; one gradient step per batch; vLLM colocate; 400-600 steps; eval every 50 steps on the dev split; save every 100. The real RL budget goes to MATH because that is where the headroom and the level axis are.
4. **Seed repeat** of the main run with a second seed, same config: its delta against the main run is the seed-noise row that every ablation delta is read against.
5. **Ablations**, one seed each, one knob each, same prompts and steps as the main run:
   - A1: KL beta 0.04 vs 0 (do we need the reference model at all?).
   - A2: DAPO: clip-higher (ε_high 0.28) + dynamic sampling (drop zero-variance groups and resample) vs standard clipping.
   - A3: Dr. GRPO: token-level loss without per-sequence length normalisation and without std normalisation, vs standard GRPO.
   Logged for all: reward, correct-rate, response length (mean, p90, truncation rate), entropy, KL, clip fraction, zero-advantage-group fraction, gradient norm, tokens generated per step, step wall-clock, eval accuracy.
6. **Reward-hacking watch**: format-only gaming (tags without content), repeated final answers, length growth without accuracy growth, "Wait"-spam. Each incident gets a log entry and a sample.
7. **Capability vs sharpening**: pass@k curves (k to 64) for `rft` and `grpo` on all evals, same samples protocol. Also the per-problem view: how many problems did GRPO solve at pass@1 that RFT could not solve at pass@64 (new capability), vs solved at pass@1 that RFT solved only at higher k (sharpening).
8. **Contamination probes on `grpo`** (section 4.3): partial-prompt completion rate and the perturbation gap, next to the base-model values. The headline table reports MATH-500 by level with the perturbation gap and Countdown alongside, so a reader can see whether gains came with or without a memorisation signature.

**Exit.** `grpo` checkpoint and card; main table (MATH-500 by level headline, GSM8K, perturbed GSM8K, Countdown) with paired CIs; ablation table with single-seed paired CIs and the seed-noise row; curve figures; the pass@k before/after figure; D3 rows for `grpo`; a written conclusion for each ablation in the "supports / does not support" form.

**What goes wrong.** vLLM weight sync and memory on one GPU (set the vLLM memory fraction and batch size first); entropy collapse within 100 steps (watch the entropy curve, consider clip-higher); length explosion (Dr. GRPO ablation shows whether normalisation is the cause); all-correct groups wasting most of the compute on GSM8K (dynamic sampling); eval accuracy rising on train-like prompts and flat on Countdown (the generalisation story).

**Interview value.** Every question in the GRPO vocabulary answered with your own curve; the capability-vs-sharpening answer with your own pass@k data; one honest "the KL term did / did not matter at this scale" sentence.

### Phase 5: DPO comparison (1 day)

**Objective.** Put the offline method on the same chart at the same compute, so the on-policy vs off-policy question has a number.

**Steps.** Pairs from Phase 3 samples (correct vs incorrect on the same problem); DPO from `sft` with beta sweep over 3 values; one extra run with pairs regenerated from the current policy (one round of on-policy DPO). Equal GPU-hours to GRPO. Report accuracy, response length and pass@k.

**Exit.** DPO rows added to the main table and the test-time chart.

**Interview value.** Implicit reward, reference model, why off-policy pairs go stale, likelihood displacement.

### Phase 6: Test-time compute (1 day)

**Objective.** One chart that puts training-time and test-time compute on the same axis: accuracy vs tokens generated per problem.

**Steps.**
1. Majority vote k = 1..64 for `base`, `sft`, `rft`, `grpo` from the saved samples.
2. Outcome reward model: train a small ORM (Qwen3-0.6B-Base + scalar head) on Phase 3 samples; best-of-N with it for N up to 64; compare with majority vote and with an open PRM if one runs on the box.
3. Budget forcing (s1): cap thinking tokens at several budgets; force continuation with "Wait" up to two times; accuracy vs budget curve.
4. Chart: x = mean tokens generated per problem (log), y = accuracy with CI, one series per method; vertical marker for training compute equivalents is a stretch.

**Exit.** D6 chart and table; ORM accuracy on held-out pairs reported.

**Interview value.** Parallel vs sequential test-time scaling, when voting beats a reward model, why best-of-N with a weak ORM plateaus, and where the RL policy sits relative to voting over the SFT model.

### Phase 7: Distillation, quantisation, serving (1 day)

**Objective.** Measure what survives in a 0.6B student and what quantisation costs, and produce serving numbers.

**Steps.** SFT Qwen3-0.6B-Base on the `grpo` model's correct traces (same filtering as Phase 2); eval; AWQ 4-bit; vLLM throughput (tokens/s), TTFT and latency at batch 1/8/32 for 1.7B-bf16, 0.6B-bf16, 0.6B-AWQ; accuracy at each.

**Exit.** `distilled-0.6b` on HF with card; D7 table.

**Interview value.** The R1 finding that small models gain more from distillation than from RL, with your own numbers; quantisation error sources; prefill vs decode bound.

### Phase 8: Analysis, write-up, interview packet (1.5 days)

**Objective.** Turn the results into something an interviewer can read in ten minutes and probe for an hour.

**Steps.** README with all tables and figures; `experiments.md` finalised; `COMPUTE.md` final ledger; `FAILURES_AND_LEARNINGS.md` reviewed; blog post ("What actually moved the number on a 1.7B reasoning model"); model cards; the 2-page interview packet (section 10); resume bullets with measured numbers; LinkedIn/X post with the main chart.

**Exit.** Everything in section 1 checked off. Someone else can clone the repo, run one script, and reproduce the base-model baseline table.

---

## 6. Experiment registry (pre-registered questions)

| ID | Question | Metric | Evidence that would change your mind |
|---|---|---|---|
| Q1 | At equal GPU-hours, how much accuracy does SFT, RFT, GRPO each add? | paired Δ accuracy with CI on three evals | GRPO Δ inside the CI of RFT at equal compute |
| Q2 | Is the KL term needed at this scale? | A1: accuracy, length, entropy, KL curves | no KL run diverges or collapses entropy |
| Q3 | Do clip-higher and dynamic sampling help? | A2: entropy over steps, accuracy per generated token | no entropy difference; same accuracy per token |
| Q4 | Does length normalisation cause length growth? | A3: response length curves, accuracy | same length growth under Dr. GRPO loss |
| Q5 | Does RL add capability or sharpen? | pass@k curves k ≤ 64, per-problem new-vs-sharpened counts | pass@64 unchanged and all pass@1 gains come from problems RFT could already solve at some k |
| Q6 | Where does the RL policy sit vs voting over SFT? | D6 chart | maj@k over SFT matches GRPO pass@1 at a lower token cost |
| Q7 | Does training on GSM8K/MATH transfer to Countdown? | Countdown accuracy per stage | flat Countdown while in-distribution rises |
| Q8 | What survives distillation and 4-bit quantisation? | D7 | more than 5 points lost at either step |
| Q9 | Are the gains reasoning or recall? | perturbation gap (original − perturbed GSM8K) and partial-prompt completion rate per checkpoint, next to MATH-500 by level | the gap widens after SFT/RL, or completion rate rises, while MATH level 3-5 and Countdown stay flat |
| Q10 | Which MATH levels does each stage move? | MATH-500 accuracy by level 1-5 per checkpoint, paired CIs | all gains on levels 1-2 (sharpening of what the base already does) with levels 4-5 unchanged |

---

## 7. Repository layout

```
reasoning_rlvr/
  PLAN.md                      this file
  WORKFLOW.md                  git process: branch per phase, PRs into main, what is never committed
  .github/                     CI (ruff + pytest on CPU) and the phase PR template
  docs/                        engineering notes per phase: what was built, how it is tested, why, interview Q&A
  cards/                       model-card sources for the HF checkpoints
  data/manifests/              dataset ids, revisions, hashes (the data itself is git-ignored)
  README.md                    results, how to reproduce
  FAILURES_AND_LEARNINGS.md    log (same template as the TTS project)
  experiments.md               pre-registered hypotheses, config hashes, conclusions
  COMPUTE.md                   GPU-hours and cost ledger
  INTERVIEW_PACKET.md          section 10, filled with measured numbers at the end
  pyproject.toml, requirements.txt, .pre-commit-config.yaml
  configs/
    eval/{gsm8k,math500,countdown,aime24}.yaml
    sft_1p7b.yaml  sft_0p6b_distill.yaml  rft.yaml
    grpo_main.yaml  grpo_a1_nokl.yaml  grpo_a2_dapo.yaml  grpo_a3_drgrpo.yaml
    dpo.yaml  orm.yaml
  src/rlvr/
    data/        loaders, filtering, dedup, contamination, trace formatting, Countdown generator
    eval/        prompts, vllm sampler, extraction (math-verify), metrics, bootstrap, report
    ref/         from-scratch: pass_at_k, advantages, grpo_loss, kl, sampling loop (tested)
    train/       sft.py, rft.py, grpo.py (TRL), grpo_ref.py (own loop), dpo.py, orm.py
    tts/         majority vote, best-of-N, budget forcing
    serve/       awq quantise, vllm benchmark
  scripts/       one entry point per phase; run_all_evals.sh
  tests/         extraction, pass_at_k, advantages, loss, kl, contamination
  notebooks/     inspection only
  outputs/       runs, samples, reports (gitignored except reports/*.md)
```

---

## 8. Timeline (start Wed 14 Oct 2026, confirmed 13 Oct)

| Dates | Phase | GPU |
|---|---|---|
| 14-15 Oct | 0 Scaffold, harness, references, tests; greedy Phase 1 rows and base-model probes on the 15th | local (WSL2) |
| 16 Oct | 1 pass@64 sampling, D2 and D3 baseline reports | rented |
| 17-18 Oct | 2 SFT (+ data-scaling and LoRA runs overnight) | rented |
| 19 Oct | 3 RFT | rented |
| 20-22 Oct | 4 GRPO main + seed repeat + 3 ablations × 1 seed (overnight runs; write Phase 5/6 code while they train) | rented |
| 23 Oct | 5 DPO | rented |
| 24 Oct | 6 Test-time compute | rented (sampling) |
| 25 Oct | 7 Distil, quantise, serve | rented / local |
| 26-27 Oct | 8 Write-up, packet, resume update | none |

Compute estimate: SFT 1.7B ≈ 3 h; RFT sampling + train ≈ 3 h; GRPO main ≈ 8 h; seed repeat ≈ 8 h; 3 ablations × 1 seed ≈ 20 h; DPO ≈ 3 h; evals with n = 64 sampling ≈ 6 h total; distillation ≈ 2 h. About 53-55 A100-hours. At ~USD 1.5-2.5 per hour on a rental, roughly USD 80-140, inside the confirmed USD 100-170 budget with about 15 h of headroom reserved for reruns after bugs. If the headroom is gone, the seed repeat is the first run to drop, and the ablation table then says its deltas have no measured noise floor.

Interrupt rule from the master plan still applies: if an interview lands, go in with Phases 0-2 and the plan.

---

## 9. Risks and fallbacks

| Risk | Signal | Fallback |
|---|---|---|
| TRL + vLLM colocate breaks on the rented image | OOM or hang at the first rollout | verl; or own loop with HF generate at G = 4 for the main run only |
| Base model already saturates GSM8K | base 4-shot ≥ 85 | shift the headline to MATH-500 by level and Countdown; say why |
| Entropy collapse in GRPO | entropy < 0.1 within 100 steps, length shrinks | clip-higher, lower LR, keep KL |
| Length explosion / truncation | p90 length at max tokens | Dr. GRPO loss, length penalty with an explicit ablation row |
| Countdown too hard or too easy | accuracy ≈ 0 or ≈ 1 at base | adjust the number range; document the chosen difficulty |
| Rental budget exceeded | COMPUTE.md over 60 h | drop the seed-repeat run; then the LoRA SFT run; never drop the baselines or the CIs |

---

## 10. Interview packet (to be filled with measured numbers at the end)

### 10.1 Papers you will be able to defend with your own data
1. DeepSeekMath / GRPO (Shao et al. 2024): group baseline instead of a value model.
2. DAPO (Yu et al. 2025): clip-higher, dynamic sampling, token-level loss, overlong shaping.
3. Dr. GRPO, "Understanding R1-Zero-like training" (Liu et al. 2025): length and std normalisation bias.
4. "Does RL really incentivize reasoning capacity beyond the base model?" (Yue et al. 2025): pass@k sharpening argument.
5. s1 (Muennighoff et al. 2025): budget forcing and test-time scaling.
Plus for context: DeepSeek-R1 (distillation vs RL for small models), STaR (expert iteration), "Spurious rewards" (Shao et al. 2025) as a sceptical reading of RLVR gains on Qwen models, which is exactly why the contamination report and the OOD task exist.

For each: one paragraph of contribution, one of method, one of what the evidence shows, one of limitations, and one of what you measured that agrees or disagrees.

### 10.2 From-scratch drills derived from this project (45 minutes each, no libraries, code must run)
- Unbiased pass@k from (n, c) and why the naive estimate is biased.
- GRPO loss with group-normalised advantages, PPO clipping, and k3 KL, in plain PyTorch, for a batch of padded sequences with a completion mask.
- Majority vote with answer normalisation over a list of sampled answers.
- A greedy and a top-p sampling loop over an HF causal LM with a KV cache.
- Causal self-attention forward pass with a mask, from scratch.
- Bootstrap CI for accuracy and a paired bootstrap for the difference between two models.
- A length-aware packing routine that builds the block-diagonal attention mask.

### 10.3 Derivations and questions, each tied to a plot
- Policy gradient → REINFORCE with baseline → PPO clipped objective → GRPO; why group mean is a valid baseline and what it assumes about the group.
- Importance ratio equals 1 after one update per batch; what clipping does when it is not.
- Bias from dividing by sequence length and by group std (Dr. GRPO); which plot shows it.
- Why asymmetric clipping keeps entropy up; the entropy curve under A2.
- Zero-advantage groups and why they waste compute; the zero-advantage fraction curve.
- k1 / k2 / k3 KL estimators: bias, variance, sign.
- Capability vs sharpening: the pass@k figure and the per-problem counts.
- Parallel vs sequential test-time scaling; ORM vs PRM vs voting; the D6 chart.
- Distillation vs RL at 0.6B; quantisation error sources; prefill- vs decode-bound serving.
- What it cost, what you would do with 10× compute, and what you would not do.

---

## 11. Decisions (confirmed 13 Oct 2026)

1. **Base model**: Qwen3-1.7B-Base only, no newer-base check. Student Qwen3-0.6B-Base. Confirmed.
2. **Budget**: USD 100-170 confirmed. Revised estimate after decision 3 is about 53-55 A100-hours, USD 80-140 (section 8). **Amended 20 Oct 2026:** after Phases 2 and 3 (about INR 11,450 spent, on one H100 session instead of the A100-hours estimated here) the ceiling is raised to INR 22,000 (USD 262) for Phase 4 at shape B of `docs/phase-4-plan.md` (about 26 H100-hours, INR 9,800); the Phase 4 steps of section 7 are superseded by that plan's section 4.
3. **Seeds**: one seed per ablation. Confirmed. Added on top: the main GRPO config is run twice so that one seed-to-seed difference is measured and every ablation delta is read against it (sections 4.2, 4.5, Phase 4 step 4). Costs about 8 A100-hours and is the first run to drop if the budget runs short.
4. **Training stack**: TRL `GRPOTrainer` + vLLM colocate on the rental for scaled runs; own loop for the correctness check and toy comparison, run locally. Local RTX 4060 (8 GB) use is fixed in section 3.1: greedy and small-k inference, probes, 0.6B LoRA jobs, quantisation and latency benchmarks locally; n = 64 sampling and all 1.7B training rented.
5. **Eval sets**: Countdown as the OOD task and the perturbed-GSM8K set as the memorisation control. Confirmed.
6. **DPO phase**: kept, one day. Confirmed.
7. **Dates**: start 14 Oct 2026, finish 27 Oct 2026 (section 8).
8. **Headline**: MATH-500 by level, GSM8K secondary, RL budget on MATH levels 1-3. Confirmed.
9. **Open-pretraining-data control run** (OLMo-2-1B or SmolLM3-3B-Base): set aside, not scheduled, not budgeted. Revisit only if the Phase 1 probes show a large memorisation signature on Qwen3.

---

## 12. Sources for section 2 (third-party; verify with recruiters)

- Anthropic RE loop descriptions and the "controlled experiments" posting language: https://www.sundeepteki.org/advice/anthropic-research-engineer-interview-2026 ; https://www.getorava.com/interview-guides/anthropic ; https://www.glassdoor.com/Interview/Anthropic-Research-Engineer-Interview-Questions-EI_IE8109027.0,9_KO10,27.htm ; https://www.indexventures.com/startup-jobs/anthropic/research-engineer-rl-engineering/ ; https://mljobs.io/jobs/anthropic-research-engineer-production-model-post-training-5112018008
- DeepMind RE loop (ML coding from scratch, fundamentals, paper/project defence): https://igotanoffer.com/en/advice/google-deepmind-research-engineer-interview ; https://www.techinterview.org/post/3233474918/deepmind-interview-process-2026/ ; https://www.glassdoor.co.in/Interview/Google-DeepMind-Research-Engineer-Interview-Questions-EI_IE1596815.0,15_KO16,33.htm ; https://www.designgurus.io/answers/detail/what-is-the-google-deepmind-interview-process-like-round-by-round
- Cross-lab RE guide: https://www.sundeepteki.org/advice/the-ultimate-ai-research-engineer-interview-guide-cracking-openai-anthropic-google-deepmind-top-ai-labs
- Base model and contamination (section 4.3): Qwen3 technical report https://arxiv.org/abs/2505.09388 ; Qwen3-1.7B-Base card https://huggingface.co/Qwen/Qwen3-1.7B-Base ; Spurious Rewards https://arxiv.org/abs/2506.10947 ; Reasoning or Memorization? https://arxiv.org/abs/2507.10532 ; Spurious Rewards Paradox (Qwen3-8B probe) https://arxiv.org/abs/2601.11061 ; OpenR1-Math-220k MATH-500 overlap audits: https://ii.inc/web/blog/post/ii-thought and https://arxiv.org/abs/2504.11456 ; GSM-Symbolic https://arxiv.org/abs/2410.05229 ; community GRPO run on this base (zero-variance statistic) https://github.com/cppywh/mini-deepseek-r1-zero

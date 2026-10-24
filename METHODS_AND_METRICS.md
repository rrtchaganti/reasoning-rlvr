# Methods and metrics for reasoning post-training: what a researcher needs to know

Companion to `PLAN.md`, `PHASES_IO_AND_HYPERPARAMS.md` and `GLOSSARY.md`. The glossary gives one-line definitions; this file gives the mechanics, the failure modes, and the history. Part A covers the metrics we report and the training diagnostics we watch. Part B walks through the training-stage methods in the order they were proposed, because the later ones only make sense as fixes to the earlier ones. Part C maps all of it onto our phases.

Written 13 Oct 2026. Dates are first arXiv appearance to the month; verify before quoting in an interview.

---

# Part A: metrics

## A.1 Accuracy and pass@1

**Definition.** Fraction of problems whose extracted final answer matches the reference. With greedy decoding this is deterministic and is what papers usually mean by "accuracy". With sampling it is the expected accuracy of a single sample, estimated by averaging over n samples per problem (`mean_i mean_j correct_ij`), and it is then lower than greedy for most models because sampling occasionally picks a wrong branch.

**What to know.**
- *Greedy is a specific operating point, not "the model".* A model can have lower 'greedy accuracy' and higher 'sampled accuracy' than another. Report both and say which.
- *Temperature changes the number.* T=0.6 and T=1.0 give different pass@1. The Qwen3 and DeepSeek reports use T=0.6, top-p 0.95 for sampled evaluation; many RL papers train at T=1.0 and evaluate at 0.6. Fix the setting in the harness.
- *Extraction is part of the metric.* An answer the scorer cannot find is scored wrong in the accuracy, and the extraction-failure rate is reported next to it, so a low number can be read as a format problem or a math problem. In this project the fallback order (the `<answer>` tag, the last `\boxed{}`, a final phrase, the last number on numeric sets) means GSM8K almost never fails extraction, while the base model's zero-shot MATH-500 run leaves 104 of 500 answers (20.8 %) unextracted (`outputs/reports/phase-1-baselines.md`).
- *Truncation is part of the metric.* A model cut off at the max-token limit is not wrong at math, it is long. Report truncation rate next to accuracy, and keep the limit fixed across checkpoints.
- *Equivalence, not string match.* `1/2`, `0.5`, `\frac{1}{2}` and `.5` are one answer. `math-verify` does symbolic comparison; still inspect a sample of disagreements, because it also has edge cases (units, intervals, sets, multiple answers).
- *Prompt sensitivity.* For base models the template alone moves GSM8K by 5-15 points (the Dr. GRPO paper shows Qwen2.5 bases answering better with no template than with the R1 template). Treat the template as an experimental variable, fixed in advance.

**Statistical floor.** For an accuracy p on N problems the standard error is about `sqrt(p(1-p)/N)`. At N=1319 (GSM8K test) and p=0.75 that is 1.2 points; at N=500 (MATH-500) and p=0.5 it is 2.2 points; at N=30 (AIME24) and p=0.1 it is 5.5 points. A single-run difference smaller than about two standard errors is not a result.

## A.2 pass@k

**Definition.** The probability that at least one of k independent samples is correct. It measures coverage: how much correct behaviour the model's distribution contains, regardless of whether the model can pick it out.

**Estimator.** The naive way, drawing exactly k samples and checking whether any is correct, is unbiased but has high variance and wastes samples. The standard estimator (Chen et al. 2021, the Codex paper) draws n ≥ k samples, counts c correct, and computes

`pass@k = 1 − C(n−c, k) / C(n, k)`,

the probability that a random k-subset of the n samples contains no correct one, subtracted from 1. Averaged over problems. Compute the ratio of binomials in log space or with the running-product form, otherwise it overflows for n=64.

**What to know.**
- pass@k is monotone in k and saturates; the shape of the curve matters more than one point. A steep early rise means many problems are "almost solved"; a long flat tail means the model lacks the capability.
- pass@k needs the ground truth to count a success, so it is not an inference method. It is an upper bound on what any selection method (majority vote, reward model) could achieve with k samples.
- The central use in this project is Yue et al. (2025): if RL raises pass@1 but the RL model's pass@k falls below the base model's pass@k at large k, RL has *sharpened* the distribution onto answers the base could already produce, and has narrowed coverage. If pass@k rises at every k, RL added capability. The per-problem version (problems solved by the RL model at pass@1 that the base never solved in 64 samples) is the most convincing form.
- At k=64 the estimate for a single problem is coarse; the dataset-level average is fine, but per-problem claims need care.
- Use the *same* samples for pass@k and maj@k so the two curves are comparable.

## A.3 Majority vote, maj@k (self-consistency)

**Definition.** Sample k answers, normalise them, return the most frequent. Accuracy of that one returned answer (Wang et al. 2022).

**What to know.**
- It is a real inference method: no ground truth needed. It is the baseline any test-time method must beat, and the baseline any RL gain must beat at equal tokens.
- maj@k ≤ pass@k always, and typically saturates by k=16-32 on GSM8K-scale problems. If maj@k keeps rising at k=64 the model is diverse but inconsistent.
- It works only when the correct answer is the mode. On hard problems where the model is confidently wrong the same way, voting amplifies the error. Report problems where the vote was wrong with ≥ 75 % agreement: those are systematic failures, not sampling noise.
- Answer normalisation decides the result. Two votes for `0.5` and two for `1/2` must count as four.
- Cost axis: maj@k costs k full generations. Plot accuracy against mean tokens generated, not against k, so that a longer-thinking model and a voting model can be compared.

## A.4 Best-of-N with a reward model

**Definition.** Sample N answers, score each with an outcome reward model (ORM) or a process reward model (PRM), return the argmax.

**What to know.**
- Scales with N only as far as the verifier is accurate. Gao et al. (2022) measured reward-model over-optimisation as a function of N: beyond some N the selected answers exploit the verifier's errors and true accuracy falls. Expect the curve to bend, and report where.
- ORM vs PRM: an ORM labels whole responses (cheap labels from final-answer correctness); a PRM labels steps (expensive labels, Lightman et al. 2023) and aggregates them (min, product, last). PRMs can beat voting on hard problems but are easily gamed and generalise poorly across formats.
- A weak ORM often loses to majority vote. The honest comparison is ORM best-of-N vs maj@N vs "weighted vote" (sum ORM scores per answer) on the same samples.
- Evaluate the ORM itself on held-out problems (AUROC of correct vs incorrect) before using it.

## A.5 Confidence intervals and comparisons

- **Bootstrap CI.** Resample the problems with replacement (2000 times), recompute the metric, take the 2.5th and 97.5th percentiles. Works for any metric including pass@k and maj@k. For sampled metrics there are two sources of variance, problems and samples; bootstrapping problems only is standard and slightly optimistic.
- **Paired comparison.** Two models on the same problems: bootstrap the per-problem difference. The CI is much narrower than the difference of two independent CIs because problem difficulty cancels. Report "A − B = +3.1 [1.4, 4.9]". This is how a 3-point MATH-500 difference becomes a result.
- **Seeds.** Training is stochastic; the eval CI does not capture it. Two training seeds give a crude estimate of run-to-run variance, which for small RL runs can be 2-4 points on MATH-500. One seed plus an eval CI is an observation, not a conclusion, and should be labelled as such.
- **Multiple comparisons.** With three ablations × several evals, one of the differences will look significant by chance. Pre-register which comparison is primary.

## A.6 Contamination and memorisation metrics

- **n-gram overlap.** Fraction of eval items sharing a 13-gram (or 8-gram) with any training item after normalisation. Catches copies; misses paraphrases.
- **Fuzzy match.** Normalised indel or Levenshtein similarity above 0.9; catches light edits.
- **Partial-prompt completion.** Show 60 % of a test problem, measure near-verbatim continuation (ROUGE-L > 0.7). Directly tests memorisation of the eval text, independent of the training data we can see.
- **Perturbation gap.** Accuracy on original items minus accuracy on rewrites with changed numbers and names (GSM-Symbolic, Mirzadeh et al. 2024). A large gap means recall; a flat gap under training means the method taught reasoning rather than retrieval.

## A.7 Training diagnostics (RL)

Read these together; any one alone misleads.

| Signal | Healthy | Warning |
|---|---|---|
| Mean reward / correct rate on training prompts | rises, then plateaus | flat from step 0 (prompts too hard or too easy); sudden jump with no eval change (format gaming) |
| Response length | drifts up slowly or stays | rapid growth with flat accuracy (length bias, "Wait" spam); collapse (model stopped reasoning) |
| Truncation rate | near 0 | rising: lengths hitting the cap; those samples get reward 0 and distort the gradient |
| Policy entropy | declines gently | drops to near 0 within 100 steps (collapse: samples become identical, groups go zero-variance); rises sharply (instability) |
| KL to reference | small, slowly growing | step change (LR too high); exactly 0 forever (adapters not training) |
| Clip fraction | small (a few %) | large: updates too big relative to ε; with μ=1 it must be 0 |
| Zero-advantage group fraction | low (< 30 %) | high: compute is wasted; prompt set too easy; dynamic sampling needed |
| Gradient norm | stable | spikes (bad batch, numerical issue) |
| Eval accuracy (dev) | tracks training reward with lag | reward up, eval flat (overfitting prompts, hacking); eval down (forgetting) |
| Pass@k on dev | stable or up | pass@1 up and pass@16 down: sharpening at the cost of coverage |

---

# Part B: training-stage methods, in the order they were proposed

A one-line timeline first, then each method in detail. The pattern of the field is: a supervised method, a selection-based method that reuses the supervised machinery, a policy-gradient method that uses all samples, and then a series of fixes to the policy-gradient method's biases once people trained at scale and looked at the curves.

| When | Method | One line |
|---|---|---|
| 1992 | REINFORCE (Williams) | Policy gradient with a sampled return; baselines reduce variance |
| 2015-2016 | TRPO, GAE (Schulman et al.) | Trust-region updates; advantage estimation with a value function |
| Jul 2017 | PPO (Schulman et al.) | Clipped surrogate objective as a cheap trust region |
| Jun 2017, Sep 2020 | Preference RL (Christiano et al.), RLHF for summarisation (Stiennon et al.) | Learn a reward model from human comparisons, optimise with PPO + KL penalty |
| 2021-2022 | Instruction tuning / SFT (FLAN, T0, InstructGPT SFT) | Supervised fine-tuning on prompt-response pairs |
| Jan 2022 | Chain-of-thought prompting (Wei et al.) | Make the model write intermediate steps |
| Mar 2022 | Self-consistency (Wang et al.) | Sample many chains, majority vote |
| Mar 2022 | InstructGPT (Ouyang et al.) | SFT → reward model → PPO: the RLHF pipeline |
| Mar 2022 | STaR (Zelikman et al.) | Sample rationales, keep the ones that reach the right answer, fine-tune, repeat |
| Nov 2022 | Process vs outcome feedback (Uesato et al.) | Step-level supervision compared with final-answer supervision on GSM8K |
| Dec 2022 - 2023 | CoT distillation (Ho et al.; Hsieh et al.; Orca) | Fine-tune small models on reasoning traces from large models |
| May 2023 | Let's Verify Step by Step (Lightman et al.) | PRM800K; PRMs beat ORMs for best-of-N; origin of MATH-500 |
| May 2023 | DPO (Rafailov et al.) | Preference optimisation without a reward model or sampling |
| Aug 2023 | RFT (Yuan et al.), ReST (Gulcehre et al.) | Rejection-sampling fine-tuning at scale; distinct reasoning paths matter |
| Dec 2023 | ReST-EM (Singh et al.) | Expert iteration framed as EM; beats SFT on human data |
| Feb 2024 | RLOO revisited (Ahmadian et al.) | REINFORCE with leave-one-out baseline matches PPO for LLMs |
| Feb 2024 | GRPO (Shao et al., DeepSeekMath) | Group-mean baseline replaces the value model |
| Feb-May 2024 | KTO, ORPO, SimPO | DPO variants: no pairs, no reference model, length-normalised margin |
| Aug 2024 | Test-time compute scaling (Snell et al.) | Allocate inference compute by difficulty; search vs revision |
| Sep 2024 | o1 (OpenAI) | Long chain-of-thought trained with RL; test-time scaling shown publicly |
| Jan 2025 | DeepSeek-R1 / R1-Zero | RLVR with GRPO at scale from a base model; cold-start SFT; distillation to small models |
| Jan 2025 | Kimi k1.5; s1 (Muennighoff et al.) | Length penalty and curriculum; budget forcing with 1k examples |
| Mar 2025 | Dr. GRPO (Liu et al.) | Remove length and std normalisation biases |
| Mar 2025 | DAPO (Yu et al.) | Clip-higher, dynamic sampling, token-level loss, overlong shaping |
| Apr 2025 | Does RL incentivise new reasoning? (Yue et al.) | pass@k says RL sharpens, does not add, at least at this scale |
| May-Jun 2025 | Entropy mechanism (Cui et al.); high-entropy tokens (Wang et al.); ProRL (Liu et al.) | Entropy collapse as the limiting factor; where the learning signal lives; long-horizon RL |
| Jun 2025 | Spurious rewards (Shao et al.) | Random rewards improve Qwen2.5-Math: a warning about the base model |
| Jul 2025 | GSPO (Zheng et al., Qwen) | Sequence-level importance ratio for stability, especially MoE |
| May-Oct 2025 | On-policy distillation (Qwen3 report; Thinking Machines) | Student samples, teacher gives per-token log-prob targets |

## B.1 REINFORCE and the variance problem (1992)

**Idea.** For a policy π_θ, the gradient of expected reward is `E[ R(τ) ∇ log π_θ(τ) ]`. Sample trajectories, weight their log-probability gradient by their return. Subtracting any action-independent baseline b leaves the gradient unbiased and reduces variance: `E[(R − b) ∇ log π]`.

**Why it matters here.** Every method below is REINFORCE with a different choice of baseline, trust region and data reuse. For a language model, a trajectory is a response, the actions are tokens, and the reward usually arrives once at the end.

**Pros.** Simple, unbiased, no value function.
**Cons.** Variance scales badly with sequence length; a single sample per prompt gives a noisy signal; no protection against a large step destroying the policy.

## B.2 TRPO, GAE and PPO (2015-2017)

**Idea.** Limit how far each update moves the policy. TRPO constrains the KL between old and new policy exactly; PPO approximates that with a clipped surrogate:

`L = E[ min( r_t A_t , clip(r_t, 1−ε, 1+ε) A_t ) ]`, with `r_t = π_θ(a_t|s_t) / π_old(a_t|s_t)`.

Advantages A_t come from a learned value function V(s) with generalised advantage estimation (GAE), which trades bias for variance with λ.

**Pros.** Stable, sample-efficient enough to reuse each batch for several epochs (the ratio is what makes reuse legitimate), the default for RLHF from 2020 to 2024.
**Cons.** Needs a value model the size of the policy (memory and compute doubled), whose per-token value estimates are poor when the reward is a single number at the end. Many hyperparameters (λ, γ, ε, value loss weight, epochs). For LLMs with outcome rewards the value model is the weak part, which is what GRPO removes.

**Interview note.** With one gradient step per batch the ratio is exactly 1 and clipping is inert; clipping matters only when you reuse rollouts (μ > 1) or when generation and training policies differ (vLLM sync lag, MoE routing noise). GSPO (B.17) is about the second case.

## B.3 Preference learning and RLHF (2017, 2020, 2022)

**Idea.** Christiano et al. (2017) learn a reward model from human comparisons of trajectory pairs (Bradley-Terry: `P(a ≻ b) = σ(r(a) − r(b))`) and optimise the policy against it. Stiennon et al. (2020) and InstructGPT (Ouyang et al., 2022) applied this to language: SFT on demonstrations, reward model on comparisons, PPO with a per-token KL penalty to the SFT model to stop the policy from drifting into reward-model blind spots.

**Pros.** Works for goals that cannot be written as a verifier (helpfulness, tone). Established the SFT → RM → RL pipeline and the KL-to-reference idea used everywhere since.
**Cons.** Reward models are imperfect and get over-optimised (Gao et al., 2022: true quality falls as the policy pushes RM score higher); humans are expensive; three models in memory (policy, reference, reward, plus value).

**Where it sits for us.** Our reward is a verifier, so no reward model is needed for training; the ORM in Phase 6 is a test-time tool. The KL-to-reference and the over-optimisation lesson carry over unchanged.

## B.4 Supervised fine-tuning and instruction tuning (2021-2022)

**Idea.** Cross-entropy on (prompt, response) pairs, loss on response tokens only. FLAN and T0 showed multi-task instruction tuning generalises to unseen tasks; InstructGPT's first stage is SFT on human demonstrations.

**Mechanics that matter.** Chat template and special tokens; loss masking; packing with a block-diagonal mask so packed examples do not attend to each other; learning rate (too high erases pre-trained knowledge, too low fails to learn the format); epochs (2-3 on small sets, 1 on large); checkpoint selection on a dev set.

**Pros.** Cheap, stable, teaches format and style quickly, the base of everything that follows.
**Cons.** Limited to behaviours present in the data; imitates mistakes in the data; with teacher traces, the student learns the teacher's style whether or not it can execute it (small models trained on very long R1 traces often get worse); exposure bias (trained on gold prefixes, tested on its own). No signal about what *not* to do.

## B.5 Chain-of-thought prompting and self-consistency (Jan, Mar 2022)

**Idea.** Wei et al. showed that few-shot examples containing reasoning steps make large models produce steps and raise accuracy sharply on arithmetic and symbolic tasks; the effect appears only above a model-size threshold. Wang et al. sampled many chains and took the majority answer, gaining 10-20 points on GSM8K.

**Pros.** No training; shows the capability is latent; voting is a universal baseline.
**Cons.** Prompt-sensitive; costs tokens; voting fails when the mode is wrong; small models benefit less from CoT prompting, which is why we train.

## B.6 STaR: self-taught reasoner (Mar 2022)

**Idea.** Zelikman et al.: sample a rationale and answer per problem; keep those whose answer is correct; for failed problems, give the model the answer and ask it to produce a rationale ("rationalisation") and keep that too; fine-tune from the *original* model on the kept set; repeat.

**Pros.** First clean demonstration that a model can bootstrap its own reasoning data with only final-answer labels. No RL machinery.
**Cons.** Only learns from successes; easy problems dominate the kept set; rationalisation can teach post-hoc justifications; each iteration restarts from the base to avoid drift, which is expensive; performance plateaus after a few iterations.

## B.7 Process vs outcome supervision; PRMs (Nov 2022, May 2023)

**Idea.** Uesato et al. compared step-level and final-answer-level feedback on GSM8K. Lightman et al. built PRM800K, step-labelled solutions on MATH, trained a process reward model, and showed PRM best-of-N beats ORM best-of-N and voting at large N. MATH-500 is the 500-problem test subset they used.

**Pros.** Step rewards are denser and localise errors; PRMs make best-of-N scale further.
**Cons.** Step labels are expensive and format-bound; PRMs generalise poorly to other models' solution styles; as a training reward they are gamed (DeepSeek-R1 report explicitly tried and dropped PRMs for this reason); score aggregation across steps is a design choice with no clean answer. For us: ORM and PRM are test-time comparison points only.

## B.8 Chain-of-thought distillation (Dec 2022 onward)

**Idea.** Ho et al. ("LLMs are reasoning teachers"), Hsieh et al. ("Distilling step-by-step"), Orca: generate reasoning traces with a strong model, fine-tune a small model on them. DeepSeek-R1 (Jan 2025) made this the headline for small models: R1 traces distilled into Qwen and Llama models beat RL run directly on those models.

**Pros.** Most accuracy per GPU-hour for small models; the traces carry strategies the small model would not discover by RL from its own samples.
**Cons.** Ceiling is the teacher; style transfer without competence transfer when traces are too long or too hard for the student (filter by length and verify correctness); any contamination in the teacher's data is inherited; the student does not learn from its own mistakes. On-policy distillation (B.18) addresses the last point.

## B.9 DPO and its variants (May 2023; Feb-May 2024)

**Idea.** Rafailov et al. showed that the RLHF objective with a KL penalty has a closed-form optimal policy, and that the Bradley-Terry likelihood of a preference pair can be written directly in terms of the policy and the reference:

`L = −log σ( β [ log π_θ(y_w|x) − log π_ref(y_w|x) − log π_θ(y_l|x) + log π_ref(y_l|x) ] )`.

No reward model, no sampling during training, one frozen reference. β controls how far the policy may move. Variants: IPO (bounded objective), KTO (unpaired good/bad labels), ORPO (no reference model), SimPO (length-normalised margin, no reference).

**Pros.** Simple, stable, cheap; one training pass over stored pairs; good for style and helpfulness.
**Cons.** Off-policy: pairs come from an older policy and go stale as the policy moves; the implicit reward is only trusted near the data. The loss pushes the margin apart and often lowers the log-probability of *both* chosen and rejected responses (likelihood displacement), which can move mass to unseen, sometimes bad, responses. For reasoning it under-performs on-policy RL at equal compute in most reports; iterative or online DPO (regenerate pairs each round) closes part of the gap at the cost of becoming RL in disguise. Length bias: DPO prefers longer responses unless corrected (SimPO, length-regularised DPO).

**Where it sits for us.** Phase 5 is exactly this comparison: offline DPO from stored pairs vs one on-policy round vs GRPO, at equal GPU-hours.

## B.10 RFT, ReST, ReST-EM: rejection sampling at scale (Aug, Dec 2023)

**Idea.** Yuan et al. (RFT) sampled many solutions per GSM8K problem from an SFT model, kept correct ones, deduplicated by reasoning path, and fine-tuned; the gain scaled with the number of *distinct* correct paths, not raw samples. ReST (Gulcehre et al.) formalised grow/improve loops with a reward threshold for translation. ReST-EM (Singh et al.) framed it as expectation-maximisation and showed it beats fine-tuning on human data on MATH and code, with gains growing with model size but plateauing after 2-3 iterations.

**Pros.** Uses the SFT trainer; no policy-gradient tuning; naturally on-policy data; the solve-rate histogram it produces is the diagnostic for RL prompt selection.
**Cons.** Throws away the information in wrong samples; easy problems saturate the kept set unless capped; the gain per iteration decays fast; needs a verifier; the model can learn to produce many near-identical paths, so diversity filtering matters.

**Where it sits for us.** Phase 3, one iteration. It is also the ancestor of our GRPO prompt filter (keep prompts with solve rate strictly between 0 and 1).

## B.11 RLOO and REINFORCE revisited (Feb 2024)

**Idea.** Ahmadian et al. argued that for LLM fine-tuning, where the policy starts strong and the reward is per-sequence, PPO's value model and GAE are unnecessary. REINFORCE with a leave-one-out baseline (the mean reward of the other k−1 samples for the same prompt) matches PPO. ReMax (2023) used the greedy response's reward as the baseline; REINFORCE++ (Jan 2025) added PPO-style clipping and per-token KL to plain REINFORCE.

**Pros.** Simpler, less memory, fewer hyperparameters, competitive accuracy.
**Cons.** Needs several samples per prompt; without clipping, large steps can still destabilise.

**Why it matters.** GRPO is this idea with the group mean (including the sample itself) as baseline and the std as a scale, plus PPO clipping.

## B.12 GRPO (Feb 2024)

**Idea.** Shao et al. (DeepSeekMath): for each prompt sample a group of G responses, score them, and use the group statistics as the baseline:

`A_i = (r_i − mean(r_1..G)) / std(r_1..G)`,

then apply the PPO clipped objective per token with A_i shared across the tokens of response i, add a KL penalty to the reference (k3 estimator), and average. No value model.

**Pros.** Halves memory relative to PPO; the baseline is exact for the prompt rather than estimated; simple to implement; works with binary rewards; became the default after R1.
**Cons.**
- *Standard-deviation normalisation* up-weights prompts where the group is nearly all-right or all-wrong (small std) and down-weights medium-difficulty ones; Dr. GRPO argues this is a difficulty bias.
- *Per-sequence length normalisation* (dividing each response's token loss by its length, then averaging over responses) makes long wrong responses cheaper and short correct ones less rewarded: response length grows without accuracy.
- *Zero-variance groups* (all same reward) give zero advantage and waste compute, which is most of the compute on easy prompt sets.
- *Entropy collapse*: symmetric clipping suppresses the growth of low-probability tokens, so exploration dies.
- *Within-group baseline* means the gradient signal depends on G; small G is noisy, large G is expensive.
- The KL term's role at scale is unclear; several later works drop it.

**Where it sits for us.** Phase 4 main run. The three ablations each remove one of the cons above.

## B.13 Test-time compute scaling (Aug-Sep 2024)

**Idea.** Snell et al. studied how to spend a fixed inference budget: parallel sampling with a verifier, sequential revision, or tree search with a PRM, and showed the best allocation depends on problem difficulty; compute-optimal test-time scaling can beat a 14× larger model on easy and medium problems. OpenAI's o1 showed long RL-trained chains of thought scaling smoothly with test-time tokens.

**Pros.** Inference compute is adjustable per query; voting and verification are training-free.
**Cons.** Gains saturate; the verifier is the bottleneck; sequential methods need a model trained to revise; and comparisons must be on a tokens axis or they are meaningless.

**Where it sits for us.** Phase 6 chart: everything on one accuracy-vs-tokens plot.

## B.14 DeepSeek-R1 and R1-Zero; Kimi k1.5; s1 (Jan 2025)

**Idea.** R1-Zero: GRPO with rule-based rewards (accuracy and format) directly on a base model, no SFT; long chains, self-verification and "aha" behaviours emerged; readability and language mixing were poor. R1: a small cold-start SFT on curated long-CoT data, then reasoning RL, then rejection sampling plus SFT on broader data, then a second RL stage; and distillation of R1 outputs into 1.5B-70B models, which beat RL applied directly to those sizes. The report also records that PRMs and MCTS were tried and dropped. Kimi k1.5 added a length penalty and a difficulty curriculum, and found that long-to-short distillation preserved much of the gain. s1 showed that 1,000 carefully chosen SFT examples plus "budget forcing" (truncating or extending thinking with "Wait") gave controllable test-time scaling.

**Pros.** Showed RLVR works end to end at scale; the cold-start + RL + distillation recipe became the template; small-model practitioners learned that distillation is the better first move.
**Cons / open points.** The "aha moment" was later shown to exist in Qwen bases before any RL (Dr. GRPO paper); gains on small models from direct RL are modest; format rewards are gameable; the base model's pre-training determines most of what RL can elicit (B.16).

## B.15 Dr. GRPO and DAPO: fixing GRPO's biases (Mar 2025)

**Dr. GRPO (Liu et al., "Understanding R1-Zero-like training").** Two changes: remove the division by response length (use a constant normaliser) and remove the division by group std. Result: the same accuracy with much shorter responses and no length explosion; the paper also showed that the prompt template interacts strongly with Qwen2.5 bases and that these bases already produce "aha" behaviour.

**DAPO (Yu et al., ByteDance Seed).** Four changes, each ablated: *clip-higher* (ε_high = 0.28 vs ε_low = 0.2, so low-probability tokens can grow, preventing entropy collapse); *dynamic sampling* (resample until the batch has no zero-variance groups); *token-level policy-gradient loss* (long responses weigh proportionally, so long bad responses are penalised); *overlong reward shaping* (soft penalty approaching the length cap instead of a hard zero). They also dropped the KL term.

**Pros.** Each is a small code change with a measured effect; together they reproduce R1-style results with fewer steps and give the diagnostics (entropy, zero-variance fraction, length) that practitioners now watch.
**Cons.** Effects are measured at 32B on AIME; at 1.7B on MATH the magnitude is unknown, which is why we ablate rather than adopt. Clip-higher raises entropy but can destabilise if ε_high is too large; dynamic sampling costs extra generation; dropping KL removes the only anchor if the reward is gameable.

## B.16 Capability vs sharpening; spurious rewards; entropy (Apr-Jun 2025)

**Yue et al. ("Does RL really incentivize reasoning capacity beyond the base model?").** Across many models and tasks, RLVR-trained models have higher pass@1 but *lower* pass@k than their base models at large k. Interpretation: RL re-weights toward paths the base could already produce and narrows coverage. Distillation, by contrast, can raise pass@k because it imports new paths.

**Shao et al. ("Spurious rewards").** On Qwen2.5-Math, random rewards, incorrect rewards and format-only rewards produce large MATH-500 gains; on Llama and OLMo they do not. The mechanism they identify is elicitation of code-style reasoning the base already has; follow-ups attribute part of it to benchmark memorisation in Qwen pre-training. Lesson: an RLVR gain on a Qwen base is not evidence of learning until it is checked against a random-reward control, a perturbation test, and a non-Qwen base.

**Cui et al. ("Entropy mechanism").** Policy entropy falls predictably as reward rises, and the attainable reward is bounded by how much entropy is left; the fall is driven by high-covariance tokens (confident and reinforced). They propose clipping or penalising the covariance of those tokens to keep exploring. **Wang et al. ("Beyond the 80/20 rule")** show that most of the RL learning signal lives in a minority of high-entropy "forking" tokens; updating only those matches or beats full updates.

**ProRL (Liu et al.).** With KL control, reference-policy resets and thousands of steps, RL did expand pass@k on some tasks, especially those where the base was weak; a counterpoint to Yue et al. that depends on training much longer than most papers.

**What this means for a researcher.** Report pass@k, not just pass@1. Run a random-reward or format-only control if the base is Qwen. Watch entropy as the leading indicator. Say "elicited" rather than "learned" unless the pass@k curve and the perturbation gap support "learned".

## B.17 GSPO (Jul 2025)

**Idea.** Zheng et al. (Qwen): GRPO's per-token importance ratios are noisy and, for MoE models whose expert routing changes between the sampling and training forward passes, biased enough to collapse training. GSPO uses one sequence-level ratio, length-normalised, and clips sequences rather than tokens.

**Pros.** Stable at scale and for MoE; matches the unit of the reward (the sequence).
**Cons.** Coarser credit assignment; for small dense models trained with μ=1, the difference from GRPO is small because the ratios are near 1 anyway.

## B.18 On-policy distillation (May and Oct 2025)

**Idea.** Qwen3 report and the Thinking Machines write-up: the *student* samples responses; the teacher scores each student token with its own log-probabilities; the loss is the per-token reverse KL (or log-prob difference) to the teacher. Dense, on-policy, needs no verifier. Qwen3 reported roughly one tenth of the RL cost for the small models.

**Pros.** Learns from the student's own mistakes (unlike SFT distillation), dense signal (unlike RLVR), cheap, works where there is no verifier.
**Cons.** Needs a teacher that shares the tokenizer and is strictly better; inherits the teacher's biases and contamination; ceiling is the teacher.

**Where it sits for us.** Not in scope, but the natural "what next" for Phase 7 and the right answer to "how would you make the 0.6B student better".

---

# Part C: how the methods and metrics map onto the phases

| Phase | Method(s) from Part B | Primary metrics from Part A | The question it settles |
|---|---|---|---|
| 1 Baselines | CoT prompting, self-consistency (B.5) | pass@1 by template, pass@k and maj@k curves, contamination probes (A.6) | What is latent in the base; how contaminated are the evals |
| 2 SFT distillation | SFT (B.4), CoT distillation (B.8) | paired Δ vs base, pass@k, MATH by level, data-scaling curve | How much the teacher's traces buy; does pass@k rise (new paths) |
| 3 RFT | STaR / RFT / ReST-EM (B.6, B.10) | Δ vs SFT at equal GPU-hours, solve-rate histogram | What selection alone achieves; which prompts are learnable |
| 4 GRPO + ablations | PPO lineage (B.2), GRPO (B.12), Dr. GRPO and DAPO (B.15), entropy (B.16) | training diagnostics (A.7), paired Δ, pass@k before/after, perturbation gap | Which GRPO parts matter at 1.7B; capability vs sharpening |
| 5 DPO | DPO and variants (B.9) | Δ at equal compute, length, pass@k, chosen/rejected log-prob curves | On-policy vs off-policy at equal cost |
| 6 Test-time | self-consistency, best-of-N, PRM/ORM (B.7, B.13), budget forcing (B.14) | accuracy vs tokens chart, ORM AUROC, over-optimisation bend | Where RL sits relative to spending the same tokens at inference |
| 7 Distil + serve | CoT distillation (B.8); on-policy distillation as next step (B.18) | accuracy loss per step, throughput | What survives at 0.6B and 4-bit |

---

# Part D: reading order

If you read five papers, read these in this order: DeepSeekMath (GRPO), DeepSeek-R1, Dr. GRPO, DAPO, Yue et al. If you read five more: STaR, DPO, Lightman et al., Snell et al., Spurious Rewards. Background if asked to derive: Williams 1992, Schulman 2017 (PPO), Ahmadian 2024 (RLOO). Each citation key here appears in `PLAN.md` section 10 with what to say about it.

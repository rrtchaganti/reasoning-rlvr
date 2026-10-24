# Reasoning-RLVR Reading Guide: Theory Behind Each Resource, Phase by Phase

Oct 21, 2026 · @CHAGANTI RAJA RAVI TEJA

## Phase 0 — Evaluation and reference implementations

Every later method is read through these three tools: an unbiased pass@k, a low-variance KL estimate, and a deterministic eval.

### Chen et al. 2021 (Codex) — the unbiased pass@k estimator

pass@k is the probability that at least one of k samples solves a problem. The plug-in estimate 1 − (1 − p̂)^k with p̂ = c/n is biased and high-variance, because it is a nonlinear function of a noisy estimate. Codex's fix: draw n ≥ k samples, count c correct, and compute the probability that a k-subset drawn without replacement contains no correct sample.


$\text{pass@}k = \mathbb{E}_{\text{problems}}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$  

where $\mathbb{E}$ is expectation(mean/average) over problems.

This is exactly unbiased for any n ≥ k. It is computed stably as 1 − ∏ (1 − k/i) for i from n − c + 1 to n, rather than through huge binomials.

Example: n = 64, c = 4, k = 8 gives 1 − (56·55·54·53)/(64·63·62·61) ≈ 0.42. One budget of 64 samples yields the entire pass@1 … pass@64 curve, which is what Phase 1 needs.

### Schulman, "Approximating KL Divergence" — k1, k2, k3

You want KL(π\_θ ‖ π\_ref) but only have samples from π\_θ and per-token log-probs. Define r = π\_ref(x) / π\_θ(x) for x \~ π\_θ.

- k1 = −log r: unbiased, but single samples can be negative and variance is high.
- k2 = ½ (log r)²: always positive, low variance, but biased (correct only to second order when the policies are close).
- k3 = (r − 1) − log r: unbiased because E\_θ\[r\] = 1, non-negative because r − 1 ≥ log r, and low variance because r − 1 acts as a control variate negatively correlated with −log r.

At r = 2: k1 = −0.69 (negative), k2 = 0.24, k3 = 0.31. GRPO's KL penalty is k3 written out:


$D_{\text{KL}} = \frac{\pi_{\text{ref}}(o_t)}{\pi_\theta(o_t)} - \log\frac{\pi_{\text{ref}}(o_t)}{\pi_\theta(o_t)} - 1$

The "KL off" ablation in Phase 4 asks whether this term does anything for a verifiable-reward run.

### Thinking Machines, "Defeating Nondeterminism in LLM Inference" — batch invariance

The usual story (floating point is non-associative and GPUs are concurrent) is incomplete. Most forward-pass kernels are deterministic run to run for a fixed input. What changes is the batch your request shares the GPU with.

Kernels (matmul split-K, RMSNorm reductions, attention's KV-split) pick different reduction orders depending on batch size. The same prompt gets logits differing at \~1e-6, an argmax flips somewhere in a long generation, and greedy completions diverge from there. They saw 80 distinct outputs from 1,000 temperature-0 runs of the same prompt.

The fix is batch-invariant kernels: fixed tile configurations, no split-K, attention split by fixed KV-chunk size rather than a fixed number of splits. The cost is throughput. The payoff is that a paired comparison (same problem, two checkpoints) measures the checkpoint rather than the scheduler, and bootstrap CIs over problems mean something.

The SFT loss, three GRPO aggregations, group advantages and KV-cache sampler in `src/rlvr/ref/` are the equations of Phases 2 and 4 written once; they are covered there.

## Phase 1 — Baselines and the pass@k ceiling

The base model's pass@64 is the reference point every training phase is judged against.

### Qwen3 technical report

Two things matter here. First, the reported numbers for the 1.7B model under its stated decoding settings: temperature 0.6, top-p 0.95, top-k 20 for thinking mode. The report explicitly advises against greedy decoding in thinking mode because it loops. Reproducing those numbers within tolerance is the calibration check that the harness is right before any checkpoints are compared.

Second, how the small models were made. Qwen3-0.6B/1.7B/4B are not RL-trained; they come from "strong-to-weak distillation" from the 32B/235B teachers, off-policy (SFT on teacher outputs) then on-policy (student samples, KL to teacher logits). Your base already carries a distilled reasoning style, which shapes how to read what SFT/RFT/GRPO add on top.

### Yue et al. 2025, "Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?"

Plot pass@k for base vs RL-trained model as k runs 1 → 256+. RL wins at small k (pass@1 is far higher), but the curves cross: at large k the base matches or exceeds.

Interpretation: RL sharpens. It reallocates mass toward paths the base could already sample rather than creating new paths; the RL model's correct traces have low perplexity under the base. Distillation from a stronger teacher does move the ceiling.

They also find PPO, GRPO, RLOO and DAPO all land far below the base's pass@256, so the gap is sampling efficiency, not algorithm choice. This is the sharpening-versus-capability question the whole project is organised around: measure the base's pass@64 first, then ask of each training phase whether it moved pass@1 toward that ceiling or moved the ceiling itself.

### GSM-Symbolic (Mirzadeh et al. 2024)

A memorisation control. Turn GSM8K problems into templates and resample names and numbers. A model that solved the problem shouldn't care that Sophie has 23 apples instead of 15, but accuracy drops and variance across instantiations is large.

GSM-NoOp adds an irrelevant clause ("five of them were smaller than usual") and causes big drops, because models subtract the number anyway. The reading is that a lot of GSM8K accuracy is surface pattern-matching.

The perturbed-GSM8K set, n-gram overlap, and partial-prompt completion (give the first half of a test problem and see if the model finishes it verbatim) together separate "learned to reason" from "saw the test set".

## Phase 2 — SFT on distilled traces

SFT on another model's traces teaches format and style; whether it teaches reasoning is what the data-scaling curve decides.

### DeepSeek-R1 (distillation section) and OpenR1-Math-220k

Distillation here is sequence-level: generate \~800k traces with R1 (600k reasoning, 200k general), SFT smaller Qwen/Llama models on them, no RL. Their key control: RL directly on Qwen-32B-Base for 10k steps gave a model clearly worse than the one distilled from R1. For small models, copying a big model's traces beats discovering them by RL.

OpenR1-Math-220k is Hugging Face's open reproduction: 220k NuminaMath problems with 2–4 R1 traces each, verified by Math-Verify (symbolic answer check) and an LLM judge for the remainder. SFT on its 94k "default" split reproduces DeepSeek's distilled-7B numbers. The `<think>…</think>` convention is what think-tag prompting and SFT share.

### LIMA (Zhou et al. 2023) and s1 (Muennighoff et al. 2025) — the small-data view

LIMA: 1,000 curated examples on a 65B model matched RLHF'd models in human preference. This motivated the "superficial alignment hypothesis": pretraining gives capability, fine-tuning mostly teaches the format and style of using it.

s1 is the reasoning version. From 59k candidates, filter to 1k by quality, difficulty (drop what Qwen2.5-7B/32B already solve), and domain diversity; SFT Qwen2.5-32B for 26 minutes; the full 59k barely beats the 1k. For your data-scaling curve the implication is sharp: if accuracy saturates after a few thousand traces, Phase 2 bought format, not reasoning.

### Minerva (Lewkowycz et al. 2022) — the 4-shot CoT prompt convention

PaLM continued-pretrained on 38B tokens of arXiv and LaTeX-preserving web math. Beyond the model, the paper fixed the protocol everyone copies: a 4-shot prompt of worked solutions ending `Final Answer: The final answer is $…$. I hope it is correct.`, chain-of-thought at temperature 0 for the pass@1 number, and majority voting over samples (maj1@k) as the second number. 540B goes from 33.6% to 50.3% on MATH with maj@64. The project's 4-shot-CoT baseline uses this prompt so numbers are literature-comparable.

### Sub-topic mechanics

- Packing concatenates many traces per sequence using flash-attention's variable-length kernel (`cu_seqlens`), so tokens cannot attend across trace boundaries and position ids reset per trace.
- Overlap filtering drops any training trace whose problem shares long n-grams with an eval problem.
- The learning-rate sweep and data-scaling curve are the two controls that tell format gains from reasoning gains.

## Phase 3 — Rejection-sampling fine-tuning / expert iteration

All three papers are one algorithm with different emphasis: sample from the current model, keep what is correct, fine-tune on it, repeat.

### Yuan et al. 2023, "Scaling Relationship on Learning Mathematical Reasoning with LLMs" (the RFT paper)

Three findings. Pretraining loss is log-linearly related to GSM8K accuracy, and SFT data scales roughly log-linearly. RFT — sample k = 100 solutions at temperature 0.7 from the SFT model, keep correct ones, deduplicate by the sequence of equations so you keep distinct reasoning paths — beats SFT with no new human data.

The number of distinct correct paths per problem is the quantity that predicts gain. Pooling paths from several models (U-RFT) helps more, and RFT helps weak models more than strong ones.

Your "shortest two correct per problem" is one dedup rule. The solve-rate distribution tells you which prompts yield anything: 0/64 contributes nothing, 64/64 nothing new; the learnable prompts sit between.

### STaR (Zelikman et al. 2022)

The same loop on GPT-J, plus rationalization: for problems the model gets wrong, give the answer as a hint, let it generate a rationale backward, and keep it if the rationale reaches the answer. This recovers hard examples the plain filter discards.

Two details matter. Each iteration fine-tunes from the original model on the accumulated data, not from the last checkpoint, to limit drift. And the appendix shows the filtered-SFT gradient equals REINFORCE with an indicator reward:

$\nabla_\theta\, \mathbb{E}_{y \sim \pi_\theta}\big[\mathbb{1}[\text{correct}(y)]\big] \approx \mathbb{E}\big[\mathbb{1}[\text{correct}(y)]\, \nabla_\theta \log \pi_\theta(y \mid x)\big]$

So RFT is already policy-gradient RL, just without a baseline, without importance ratios, and with one update per sampling round.

### ReST-EM (Singh et al. 2023, "Beyond Human Data")

Frames the loop as expectation-maximisation. E-step: sample from the policy and weight by reward (binary → filter). M-step: maximise weighted log-likelihood, which with binary rewards is SFT on correct samples.

On PaLM-2 for MATH and APPS, model-generated data beats the human-written training solutions. Gains accrue over 2–3 iterations then overfit (they restart from base each iteration too), and gains transfer to held-out tasks. The "expert iteration" name is from Anthony et al. 2017: a sampler-plus-filter as expert, the network as apprentice imitating it.

### Why the controls exist

- Token-matched control: the same volume of SFT-model samples without the correctness filter. Tests whether the gain comes from correctness or merely from on-policy data.
- Correctness filter vs length filter: separates "learned to be right" from "learned to be short".
- Termination vs reasoning decomposition: per problem, was a newly solved problem previously wrong, or previously unfinished (looped past the cap)? The latter is a termination gain, not a reasoning gain.

## Phase 4 — GRPO with verifiable rewards and the ablation family

GRPO is PPO's clipped objective with a group-mean baseline in place of the critic; every ablation is a different normalisation or clip choice inside that one equation.

### PPO (Schulman et al. 2017)

The clipped surrogate. With ratio ρ\_t = π\_θ(a\_t | s\_t) / π\_old(a\_t | s\_t) and advantage Â\_t:


$L^{\text{CLIP}} = \mathbb{E}_t\left[\min\!\left(\rho_t \hat A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,\hat A_t\right)\right]$.

The min makes this a pessimistic lower bound on policy improvement. If Â > 0 and the ratio already exceeds 1 + ε, the clipped term is constant and the gradient vanishes; symmetrically for Â < 0 below 1 − ε. Example: Â = +1, ρ = 1.5, ε = 0.2 → min(1.5, 1.2) = 1.2, a constant, zero gradient.

That lets you take several gradient epochs on one batch of rollouts without ratios blowing up. In LLM RLHF, PPO runs token-level with a learned value model and GAE; the value model is what GRPO removes.

### DeepSeekMath (Shao et al. 2024) — GRPO's definition

For prompt q, sample a group of G outputs from π\_old, score each r\_i, and set a group-relative advantage applied to every token of output i:


$\hat A_i = \frac{r_i - \text{mean}(r_{1..G})}{\text{std}(r_{1..G})}$

The objective is PPO's clipped term with this advantage, averaged per sequence then per group, minus β times the k3 KL to the reference:
$J(\theta) = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \Big[\min\big(\rho_{i,t}\hat A_i,\ \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat A_i\big) - \beta\, D_{\text{KL}}\big[\pi_\theta \| \pi_{\text{ref}}\big]\Big]$

The group mean is a Monte-Carlo baseline replacing the critic, roughly halving memory. With G = 8, binary reward, 2 correct: correct outputs get Â = 0.75 / 0.43 ≈ +1.7, wrong ones ≈ −0.58. If all 8 agree, Â = 0 and the group contributes nothing.

Two more things feed the project. The "unified paradigm" section writes every method as a gradient coefficient GC times the log-prob gradient:

$\nabla_\theta J = \mathbb{E}\Big[\sum_t GC(q, o, t)\, \nabla_\theta \log \pi_\theta(o_t \mid q, o_{<t})\Big]$

SFT has GC = 1, RFT has GC = 1\[correct\], GRPO has GC = Â\_i with clip logic. That is the file's through-line as one equation. And they already observed RL raising Maj@K but not Pass@K, concluding it makes the distribution more robust rather than adding capability: Yue et al. a year early.

### DeepSeek-R1 — reward design and the "aha" claims

R1-Zero: pure GRPO on the V3 base with only rule-based rewards — accuracy (boxed answer matches; test cases for code) and format (think tags present). They deliberately avoided neural reward models, citing reward hacking and retraining cost.

Observed: response length grows through training and the model starts re-examining its own steps (the "aha moment"). But outputs were unreadable and mixed languages, so R1 proper adds a cold-start SFT of a few thousand long CoTs, a language-consistency reward during RL, a rejection-sampling SFT round, and a final RL for general preferences.

The "unsuccessful attempts" section is the candid part: PRMs (hard to define and label steps, reward-hackable) and MCTS (search space too large, value model hard to train). Your binary "finished and well-formed and correct" reward is the R1-Zero reward; Dr. GRPO and SimpleRL-Zoo below are the critiques of the "aha" reading.

### DAPO (Yu et al. 2025)

Four fixes to GRPO, each tied to an observed failure; the project tests the first two.

- Clip-higher (ε\_low = 0.2, ε\_high = 0.28): with a symmetric clip, a token at probability 0.01 can only rise to 0.012 per step while one at 0.9 can go to 1.0. The clip suppresses exactly the low-probability tokens exploration needs, and entropy collapses. A higher upper clip lets rare tokens grow.
- Dynamic sampling: groups where all G are right or all wrong have Â = 0 and waste batch. Keep sampling prompts until the batch is full of non-zero-variance groups. As training progresses more prompts saturate, so without this the effective batch shrinks.
- Token-level loss: GRPO's 1/|o\_i| averages each sequence first, so a token in a long response weighs less than one in a short response. DAPO averages over all tokens in the batch.
- Overlong reward shaping: instead of zero reward for truncation (which punishes long-but-sound reasoning), a soft linear penalty near the cap, or masking truncated samples from the loss.

They also drop the KL term entirely and reach 50 on AIME-24 with Qwen2.5-32B in half the steps of R1-Zero-32B.

### Dr. GRPO (Liu et al. 2025, "Understanding R1-Zero-Like Training")

Two biases hidden in GRPO's normalisation. The 1/|o\_i| is a length bias: for a wrong response (Â < 0) the per-token penalty shrinks with length, so the policy learns to make wrong answers longer; for right ones it prefers shorter. The std division is a difficulty bias: prompts with near-zero reward variance get amplified weights.

Dr. GRPO removes both — sum tokens with a constant normaliser, and Â\_i = r\_i − mean — and matches accuracy with shorter outputs.

Their other contribution is base-model analysis. Qwen2.5 already produces "wait, let me check" self-reflection before any RL, and templates matter a lot. So part of R1-Zero's length growth and "aha" is an artefact of objective plus base, not emergent reasoning. Your 2,048-token cap run is the same family of observation: if truncation means zero reward, the cap is a termination reward in disguise, and length shrinks for reasons unrelated to reasoning.

### Open-Reasoner-Zero and SimpleRL-Zoo — the reward-hacking watch

ORZ shows the minimal recipe scales: vanilla PPO with GAE γ = λ = 1, no KL, binary reward, on Qwen2.5-32B base, matching R1-Zero-32B in a tenth of the steps.

SimpleRL-Zoo runs zero-RL on ten base models (Llama, Mistral, DeepSeek, Qwen 0.5B–32B). The findings tell a small-model project what to watch: all improve, but length growth is often repetition or incoherence rather than reflection; strict format rewards can kill exploration in weak models; training-set difficulty has to match the model.

Concrete things to monitor: multiple boxed answers gaming the matcher, filler that pushes toward the cap, think tags wrapping empty reasoning.

### Spurious Rewards (Shao et al. 2025)

The cautionary result. Qwen2.5-Math-7B improves on MATH-500 by \~21 points with random rewards, \~16 with format-only rewards, \~25 with deliberately wrong labels, versus \~29 with real ones.

Mechanism: the model already reasons in Python-like code text \~65% of the time, and that behaviour correlates with correctness. GRPO's clipping is not reward-neutral: under random rewards the clipped objective still drifts toward the model's prior high-probability behaviours, and removing clipping removes the random-reward gain. Llama and OLMo get nothing from spurious rewards.

For a verifiable-reward project on a Qwen base this makes two control runs mandatory, random-reward and format-only. Only the margin over them is attributable to the verifier.

## Phase 5 — DPO as the contrast

DPO is the offline pairwise counterpart to GRPO's online group-relative update: no rollouts during training, but stale pairs and a margin-only constraint.

### Rafailov et al. 2023, DPO

Start from the KL-regularised RLHF objective, maximise E\[r(x, y)\] − β·KL(π ‖ π\_ref). Its optimum has a closed form:


$\pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x)\, \exp\!\big(r(x, y)/\beta\big)$


Invert it to express the reward in terms of the optimal policy:

$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$


Substitute into the Bradley–Terry likelihood and the intractable Z(x) cancels in the difference:


$L_{\text{DPO}} = -\,\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$

The policy's log-ratio is an implicit reward; no reward model, no rollouts, no critic. The gradient is ∇log π(y\_w) − ∇log π(y\_l) weighted by σ(r̂\_l − r̂\_w), i.e. by how wrong the implicit reward currently is.

Two costs. It is off-policy, so as the policy moves the pairs go stale. And it constrains only the margin, so log π(y\_w) and log π(y\_l) can fall together (likelihood displacement), which on reasoning data shows up as chosen traces becoming less likely.

### Iterative RPO (Pang et al. 2024)

DPO adapted to reasoning. Each iteration: sample K CoTs per problem from the current model, label by final answer, form (correct, incorrect) pairs, and train with L\_DPO + α·NLL(y\_w). The length-normalised NLL on the chosen trace is what stops its probability from falling; without it most of the gain vanishes. Initialise the next iteration from the current model (unlike STaR).

Llama-2-70B GSM8K goes 55.6 → 81.6. Your reframing to finished-correct vs looping pairs aims DPO at one failure mode, so the experiment tests whether offline pairwise training buys termination as cheaply as RFT did, with no rollouts during training.

## Phase 6 — Test-time compute

Every inference-time method is a point on one accuracy-versus-tokens chart, and the trained policies are the other points on it.

### Wang et al. 2022, self-consistency

Sample many CoTs at temperature, extract final answers, take the majority. The justification: marginalise over reasoning paths rather than trusting one, since correct answers are reachable by many different paths while wrong answers scatter.

PaLM-540B GSM8K goes 56.5 → 74.4 with 40 samples. Weighting votes by sequence probability does not beat the plain count. This is maj@k, the verifier-free parallel baseline every other test-time method must beat per token.

### Lightman et al. 2023, "Let's Verify Step by Step" — ORM vs PRM

An outcome reward model scores a whole solution from the final-answer label; a process reward model scores each step. They built PRM800K (800k human step labels on 75k GPT-4 solutions) and compared best-of-N on MATH. At N = 1,860: PRM 78.2%, ORM 72.4%, majority vote 69.6%, with the gap widening in N.

The PRM also penalises right-answer-wrong-reasoning, which an ORM rewards. Your Phase 6 trains an ORM (cheap: labels come from the verifier) for best-of-N; this paper sets the ceiling to expect and explains why a PRM would be the next step.

### Snell et al. 2024, "Scaling LLM Test-Time Compute Optimally"

Two ways to spend inference compute: improve the proposal (sequential revisions, the model edits its own answer) or search with a verifier (best-of-N, beam search, lookahead against a PRM).

The compute-optimal strategy depends on difficulty. On easy questions sequential revision beats parallel sampling; on hard ones parallel/search wins. Allocating by estimated difficulty gets best-of-N's accuracy with \~4× less compute.

In the FLOPs-matched comparison, test-time compute on a small model beats a 14× larger pretrained model on easy and medium problems, while pretraining still wins on the hardest.

### s1 (Muennighoff et al. 2025) — budget forcing

Control thinking length at decode time. To cap: force the end-of-thinking token and the answer. To extend: when the model tries to stop, suppress the end token and append "Wait", which makes it re-check and often fix its answer.

AIME accuracy rises with the number of forced "Wait"s before plateauing. They score test-time methods on control, scaling slope, and peak performance; sequential scaling via "Wait" beats parallel majority voting per token in their setting. In your chart this is the one knob that moves a single trained policy along the token axis without retraining.

## Phase 7 — Distillation, quantisation, serving

These are engineering deliverables: compress the 1.7B policy into 0.6B, shrink its weights to 4 bits, and serve it without wasting KV cache.

### Hinton et al. 2015, knowledge distillation

Train the student on the teacher's softened distribution q\_i = softmax(z\_i / T):

$L = \alpha\, T^2\, \mathrm{KL}\big(q^{\text{teacher}}_T \,\|\, q^{\text{student}}_T\big) + (1-\alpha)\, \mathrm{CE}(\text{hard labels})$

The T² compensates for the 1/T² scaling of soft-target gradients. The "dark knowledge" is in the relative probabilities of wrong classes — a 2 looks more like a 7 than a 3 — which one-hot labels discard.

For LLMs there are two regimes. Sequence-level KD trains on teacher samples, which is what R1 and your Phase 2 did. Logit KD applies a per-token KL over the full vocabulary and needs a shared tokenizer, which Qwen3-1.7B → 0.6B has. Qwen3 did off-policy logit KD then on-policy (student samples, KL to teacher logits); reverse-KL variants like MiniLLM are mode-seeking and tend to suit generation better than forward KL.

### AWQ (Lin et al. 2023)

Weight-only 4-bit quantisation for the memory-bound decode phase. Observation: \~1% of weight channels are salient, and you find them by activation magnitude, not weight magnitude. Keeping those in FP16 recovers most of the loss, but mixed precision is hardware-hostile.

Instead, scale: multiply the salient weight channel by s and divide the matching activation by s, an exact identity. The quantisation step Δ is set by the group max and barely changes, so the salient channel's relative error drops by roughly 1/s. Search s = s\_X^α over a small grid of α to minimise output MSE on a small calibration set.

No backprop, so no overfitting to calibration data, and it holds up on instruction-tuned models. Their TinyChat kernels give over 3× decode speedup versus FP16.

### vLLM (Kwon et al. 2023), PagedAttention

The KV cache is the serving bottleneck. Prior systems wasted 60–80% of it by pre-allocating contiguous space per request at maximum length: internal fragmentation, reserved-but-unused space, and external fragmentation between requests.

PagedAttention borrows OS virtual memory. KV tensors live in fixed-size blocks (e.g. 16 tokens), a per-sequence block table maps logical to physical blocks, blocks are allocated on demand, and waste is confined to the last block.

Side benefits: prefix sharing across requests via reference counts, copy-on-write for parallel sampling and beam search (relevant to your n = 64 evals), and preemption by swap or recompute. With continuous batching this gives 2–4× the throughput of FasterTransformer/Orca at equal latency. It is the same engine whose batch-size-dependent kernels Phase 0 had to make invariant.

## The through-line in one equation

DeepSeekMath's gradient-coefficient view ties the phases together: every training method is the same log-prob gradient with a different coefficient on a different data source.

$\nabla_\theta J = \mathbb{E}\Big[\sum_t GC(q, o, t)\, \nabla_\theta \log \pi_\theta(o_t \mid q, o_{<t})\Big]$

| Phase | Method | Data source | Gradient coefficient GC | Policy regime |
| --- | --- | --- | --- | --- |
| 2 | SFT | Someone else's traces (R1 via OpenR1) | 1 | Off-policy |
| 3 | RFT / expert iteration | Own samples, filtered | 1\[correct\] | On-policy, no gradient-based policy objective |
| 4 | GRPO | Own samples, grouped | Â\_i with clip and KL | On-policy, online |
| 5 | DPO / Iterative RPO | Own samples, paired | σ(r̂\_l − r̂\_w) on (y\_w − y\_l) + α·NLL | Off-policy, pairwise |
| 6 | Test-time compute | Inference samples | None — compute spent at decode | No training |

The question Yue et al. and Dr. GRPO force at every step is which of three things the coefficient bought: a new path (reasoning), more mass on an existing path (sharpening), or just stopping before the cap (termination).

# Paper-defence notes

The papers this project tests directly (`PLAN.md` section 10.1), each with its contribution, the
method details that matter for the comparison, its evidence, its limits, and what this project
measured that agrees or disagrees. Facts about each paper were checked against the paper itself on
24 Oct 2026 (arXiv abstract and full text; venues from the proceedings); figures that exist only in
plots are marked as such rather than quoted. Our numbers are greedy unless stated, Qwen3-1.7B-Base,
paired, with 95 % CIs; claim ids point into the evidence index.

None of these papers studies Qwen3-1.7B-Base. Three caveats they raise for Qwen-family models
(templates, contamination, spurious rewards) are the ones a reviewer is most likely to bring.

---

## 1. DeepSeekMath and GRPO (Shao et al., arXiv 2402.03300, Feb 2024)

**Contribution.** A 7B maths model from 120B tokens of continued pre-training, and GRPO, a PPO
variant that drops the critic: the baseline is the group of G outputs for the same question.

**Method that matters here.** Outcome rewards are normalised by subtracting the group mean and
dividing by the group std; that value is every token's advantage. The KL to the reference enters the
loss, not the reward, as the unbiased, always-positive estimator pi_ref/pi - log(pi_ref/pi) - 1 (k3).
The objective averages over each output's tokens (a 1/|o_i| factor) and then over outputs. The
reward was a learned reward model, retrained with a replay buffer, not a verifier.

**Evidence.** DeepSeekMath-RL 7B against the Instruct model: GSM8K 82.9 to 88.2, MATH 46.8 to 51.7
(chain of thought). The authors' own analysis: RL improves maj@K but not pass@K.

**Limits.** A learned reward model; one model family; the pass@K remark is a short analysis, not a
controlled study.

**What we measured.** Our GRPO is the published objective (per-sequence averaging, group-std scaling,
k3 KL in the loss at beta 0.04) with a verifiable reward instead of a learned one, and our reference
loss matches TRL's to 1e-9 (`scripts/grpo_correctness_check.py`). Their pass@K remark holds here on
GSM8K (GRPO over RFT at k = 16: -0.3 [-0.9, 0.3], no problem newly solved in 32 tries) and almost on
MATH-500 (+1.9 [0.4, 3.5], 10 newly solved, 2 lost; P4-08). What we add: the pass@1 gain itself
decomposes as termination (P4-02).

## 2. DAPO (Yu et al., arXiv 2503.14476, Mar 2025; NeurIPS 2025)

**Contribution.** An open large-scale RL recipe for long chain of thought from a base model
(Qwen2.5-32B), with code and the DAPO-Math-17K dataset.

**Method that matters here.** Four techniques: clip-higher (eps_low 0.2, eps_high 0.28); dynamic
sampling (drop groups that are all right or all wrong and sample until 0 < correct < G); a
token-level loss (normalise by all tokens in the group, so long responses weigh more); overlong
shaping (mask truncated samples, and a soft penalty falling to -1 over the last 4,096 of 20,480
tokens). The KL penalty is removed. Reward +1 / -1, rule-based.

**Evidence.** AIME 2024 avg@32: 50 for DAPO, 47 for DeepSeek-R1-Zero-Qwen-32B at half the steps, 30
for naive GRPO.

**Limits.** Maths only; the authors note the training reward correlates little with validation
accuracy.

**What we measured.** Two of the four, one knob at a time at 1.7B (EXP-014b): clip-higher at 0.28
with dynamic sampling changed nothing (+1.7 [-0.1, 3.6] / -2.0 [-6.0, 2.0]) at 1.5x the compute.
Both nulls have a mechanism. Clip-higher can only act when the ratio leaves the clip range, and with
one optimiser step per batch the ratio was 1 and the clip fraction 0 throughout
(`docs/derivations.md` item 4). Dynamic sampling only helps when many groups are uninformative, and
the learnable-prompt filter had already left 32 informative groups in 199 steps of 200. Removing the
KL (EXP-014a) did not destabilise anything: it landed +2.2 [0.4, 4.0] / +5.6 [1.6, 9.4] above the main
run, 15 times further from RFT, and the gain was again termination with nothing under zero-shot
(P4-05, P4-06). Our reward gave a truncated completion 0, a harder version of their overlong
penalty at a 1,536-token cap.

## 3. Dr. GRPO, "Understanding R1-Zero-Like Training" (Liu et al., arXiv 2503.20783, Mar 2025; COLM 2025)

**Contribution.** A critical look at base models and at GRPO in R1-Zero-style training, and an
unbiased objective.

**Method that matters here.** Two biases in GRPO: a response-level length bias from the 1/|o_i|
factor (a long wrong answer is penalised less per token, so wrong answers grow) and a question-level
difficulty bias from dividing by the group std (very easy and very hard questions weigh more). Dr.
GRPO removes both, replacing |o_i| by a constant such as the generation budget. On templates: Qwen2.5
models answer best with no template at all (Qwen2.5-Math-1.5B, average of five benchmarks: 7.9 with
the R1 template, 24.2 with Qwen's, 33.1 with none), which the authors read as a sign of
question-answer text in pretraining. Most base models already show the "aha" self-reflection.

**Evidence.** Oat-Zero-7B (Qwen2.5-Math-7B, Dr. GRPO, MATH levels 3-5): AIME 2024 43.3, MATH-500 80.0.

**Limits.** Self-reflection detectors can misclassify; generation capped at 3k tokens.

**What we measured.** Both biases are pinned by unit tests on our reference loss
(`test_sequence_agg_weights_short_completions_more`, `test_std_normalisation_inflates_low_variance_groups`),
but the ablation that would isolate them (EXP-014c) is pre-registered and unrun. Under the biased
loss our completions got shorter, not longer (1,406 to 1,201 tokens), because the reward gave a
truncated completion 0. The template finding reproduces on Qwen3-1.7B-Base: zero-shot 77.6 / 53.6
against 4-shot CoT 74.5 / 41.4 and think-tag 61.3 / 43.0 (GSM8K / MATH-500; P1-01), and every trained
checkpoint is therefore also read against the base model's best prompt. Their prompt pool (MATH
levels 3-5) is exactly our unrun EXP-018.

## 4. Yue et al., "Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?" (arXiv 2504.13837, Apr 2025; NeurIPS 2025 oral)

**Contribution.** RLVR models beat their base at small k and lose at large k: what RLVR samples, the
base already could. Distillation, by contrast, adds new reasoning patterns.

**Method.** pass@k up to k = 256-1024 at T = 0.6 across Qwen2.5 7B/14B/32B and Llama-3.1-8B
(SimpleRL-Zoo), Oat-Zero, DAPO-32B, code and vision models; six RL algorithms; a perplexity analysis.

**Evidence.** The curves cross as k reaches tens to hundreds; on AIME24 at k = 1024, 13.3 % of
problems are solved only by the base and none only by the RL model; the six algorithms perform
alike.

**Limits.** The authors: frontier pipelines are proprietary and newer techniques may change it.
Later work (ProRL, arXiv 2505.24864) reports prolonged RL beating the base across pass@k.

**What we measured.** Within k = 32 we see no crossing and almost no widening: GSM8K is pure
sharpening, MATH-500 a small widening (10 problems newly solved against 2). Our comparison is
GRPO against RFT, not against the base, and our sampled sets stop at k = 32, so the crossover regime
is not tested. Two results go further than theirs at our scale: the pass@1 gain is termination
(P4-02), and from k = 4 the majority-vote curves of SFT, RFT and GRPO coincide (EXP-017), so on our
data the RL policy's whole advantage is reachable by sampling the earlier checkpoints.

## 5. s1: Simple test-time scaling (Muennighoff et al., arXiv 2501.19393, Jan 2025; EMNLP 2025)

**Contribution.** About 1,000 curated examples plus control of decode length give test-time
scaling.

**Method.** s1K (1,000 questions from 59,029, filtered for quality, difficulty and diversity, traces
from Gemini Flash Thinking); SFT of Qwen2.5-32B-Instruct in 26 minutes on 16 H100s. Budget forcing:
for a maximum, append the end-of-thinking delimiter (optionally "Final Answer:") to force an answer;
for a minimum, suppress the delimiter and append "Wait" to extend.

**Evidence.** s1-32B with budget forcing: AIME24 56.7, MATH-500 93.0, GPQA 59.6 (without: 50.0, 92.6,
56.6).

**Limits.** The authors: gains flatten, the context window bounds them, and suppressing the end of
thinking too often causes repetitive loops. Later work (arXiv 2507.14419) attributes most of the
scaling to the maximum-length side and finds "Wait" makes the model oscillate.

**What we measured.** Both halves, on checkpoints that loop. The maximum side is our EXP-016: forcing
the stop on SFT at 1,536 gains +4.6 [3.4, 6.0] / +7.2 [4.0, 10.4] on the same machine (P5-08), more
than RFT's whole gain, and on GRPO at 2,048 +2.0 [0.6, 3.6] on MATH-500 (P5-11). The minimum side is
our EXP-021: "Wait" before the stop equals the plain stop at the same total tokens on all four cells
(P5-12), and 14-19 % of nudged loops closed on their own. Phase 6 says why: the loop is one step
re-derived from a fixed point, and "Wait" is already what the model writes inside it, 16-28 times
per truncated completion. Our reading agrees with the later work, not with "Wait" as a scaling knob.

## 6. DeepSeek-R1 (DeepSeek-AI, arXiv 2501.12948, Jan 2025; Nature 2025)

**Contribution.** R1-Zero (GRPO from a base model with rule-based accuracy and format rewards), R1
(a multi-stage pipeline), and six dense models distilled from R1 by SFT alone (about 800K samples).

**Evidence on distillation against RL** (v1, Qwen-32B-Base): RL for over 10K steps reaches AIME24
47.0, MATH-500 91.6; the distilled model 72.6 and 94.3. Small distilled models: Qwen-1.5B 28.9 / 83.9,
Qwen-7B 55.5 / 92.8 (AIME24 / MATH-500). The authors' conclusion: for small models distillation
beats large-scale RL. (The Nature version's tables were not checked.)

**Limits.** Readability and language mixing in R1-Zero; few-shot prompting degrades it; process
reward models and MCTS did not work for them.

**What we measured.** Our SFT on 23,000 distilled traces is the largest single step (+13.0 / +5.8 in
the trained format, P2-01), and both on-policy stages after it bought termination (P3-04, P4-02):
at 1.7B, the R1 conclusion in miniature, with the mechanism made explicit. Like them, we find
few-shot prompting hurts the base model on MATH-500 (P1-01). We did not distil to 0.6B (dropped).

## 7. STaR, the Self-Taught Reasoner (Zelikman et al., arXiv 2203.14465, Mar 2022; NeurIPS 2022)

**Contribution.** A model bootstraps reasoning by fine-tuning on its own rationales that reach the
right answer.

**Method.** Few-shot prompt GPT-J 6B for rationales; keep those with correct answers; fine-tune,
restarting from the original pre-trained model each iteration; repeat. Rationalization: for missed
questions, give the answer as a hint and train on the resulting rationale without the hint.

**Evidence.** CommonsenseQA 72.5 (68.8 without rationalization) against 73.0 for a 30x larger
fine-tuned GPT-3; GSM8K 10.7 against 5.8 for direct-answer fine-tuning.

**Limits.** Needs above-chance few-shot accuracy; binary tasks let poor rationales through.

**What we measured.** Our RFT is one STaR round without rationalization, continuing from the SFT
checkpoint rather than restarting, keeping the two shortest correct traces per problem. It gained
+3.3 [1.0, 5.8] / +5.8 [1.6, 10.2] (P3-01), all of it on problems SFT truncated (P3-04), and half of
its GSM8K stopping effect is the length filter, not the correctness filter: two random correct traces
instead of the two shortest cost 2.4 [0.1, 4.7] points (P3-06). Selection on "finished and correct"
is selection for stopping. The second round was dropped for budget (EXP-011).

## 8. Spurious Rewards (Shao et al., arXiv 2506.10947, Jun 2025)

**Contribution.** RLVR with GRPO raises Qwen2.5 maths scores even with random or incorrect rewards,
because it amplifies behaviours learned in pretraining.

**Method.** GRPO on DeepScaleR data, no KL, 16 rollouts per prompt, 300 steps, with ground-truth,
majority-vote, one-shot, incorrect, random and format-only rewards.

**Evidence.** Qwen2.5-Math-7B on MATH-500: +29.1 with the true reward, +24.1 with incorrect labels,
+21.4 with random rewards, +13.8 with a format reward. Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct
and OLMo2-7B stay flat or decline (per-model numbers only in figures). The mechanism they propose:
code-style reasoning rises from 65 % to over 90 %, and GRPO's clipping bias reinforces already-likely
tokens; without clipping, random rewards give no signal.

**Limits.** The authors leave the mechanism open and call spurious rewards a probe, not a method;
later work (arXiv 2507.10532) argues the Qwen2.5 gains reflect benchmark contamination.

**What we measured.** Three contamination probes found no memorisation signal and none added by
training: the perturbed-GSM8K gap is flat (10.1 base, 11.1 after GRPO; a names-only control moves
nothing; P1-03, P1-04), and partial-prompt verbatim completion is unchanged (X-02). Their clipping
mechanism cannot operate in our runs (clip fraction 0). But our main result invites their control:
if the on-policy gain is termination, a reward for finishing alone, with no correctness signal, might
reproduce much of it. **We did not run a random-, format- or finished-only-reward GRPO control**;
it is the cheapest experiment that would test whether our reward's correctness term taught anything,
and it belongs next to EXP-018 in what comes next.

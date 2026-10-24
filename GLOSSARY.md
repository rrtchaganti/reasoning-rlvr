# Glossary of abbreviations and terms used in this project

Short definitions, each with the one line you would say in an interview. Grouped by where they appear.

---

## Evaluation and statistics

| Term | Meaning |
|---|---|
| **pass@1** | Accuracy when the model gives one answer per problem. With greedy decoding (temperature 0) it is just "fraction of problems answered correctly". |
| **pass@k** | Probability that at least one of k sampled answers is correct. Measures how much capability is latent in sampling. Computed with the unbiased estimator from the Codex paper: draw n ≥ k samples, count c correct, `pass@k = 1 - C(n-c, k) / C(n, k)`. The naive estimate (take exactly k samples, check if any is right) has high variance; the estimator uses all n samples. |
| **maj@k** (majority vote, self-consistency) | Sample k answers, normalise them, return the most frequent one. Accuracy of that single returned answer. Unlike pass@k it does not need the ground truth to pick a winner, so it is a usable inference method. |
| **best-of-N** | Sample N answers, score each with a reward model, return the highest-scoring one. Accuracy of that answer. |
| **ORM** | Outcome Reward Model: scores a whole response as likely correct or not. Trained with labels from the final answer only. |
| **PRM** | Process Reward Model: scores each reasoning step. Needs step-level labels; expensive, often gamed. Used here only for a test-time comparison if an open one runs. |
| **greedy decoding** | Always pick the most probable next token (temperature 0). Deterministic up to batching effects. |
| **T, top-p, top-k** | Sampling knobs. Temperature T scales logits (T=1 is the model's own distribution, lower is sharper). top-p keeps the smallest set of tokens whose probability mass is at least p (nucleus sampling). top-k keeps the k most probable tokens. |
| **CI** (confidence interval) | Range that would contain the true value in 95 % of repeated experiments. Reported with every accuracy so that "A beats B by 2 points" can be checked against the noise. |
| **bootstrap** | Resample the problems with replacement many times (2000 here), recompute the metric each time, take the 2.5th and 97.5th percentiles as the CI. Needs no distributional assumption. |
| **paired bootstrap / paired comparison** | When two models are scored on the same problems, bootstrap the per-problem difference (correct_A − correct_B) instead of each accuracy separately. Much tighter CI because problem difficulty cancels out. |
| **standard error (SE)** | Standard deviation of an estimate. For an accuracy p on N problems, `SE ≈ sqrt(p(1-p)/N)`; at p=0.5 and N=500 that is 2.2 points, which is why MATH-500 differences under about 4 points need pairing or seeds. |
| **seed** | Random-number starting point. Two seeds of the same config means two independent training runs; their difference is run-to-run noise. This project runs one seed per ablation and repeats only the main GRPO config with a second seed, so that noise is measured once and every ablation delta is read against it. |
| **dev split / held-out** | Problems used to choose checkpoints and hyperparameters. Never the test set, otherwise the reported test number is optimistically biased. |
| **OOD** (out of distribution) | Evaluation data unlike the training data (here Countdown). Tests generalisation rather than recall. |
| **contamination** | Test problems (or near copies) present in the training data. Inflates benchmark numbers. Checked with n-gram overlap and fuzzy matching. |
| **13-gram overlap / 8-gram** | Contamination check: does any run of 13 (or 8) consecutive words in a training item also appear in a test item. |
| **partial-prompt completion test** | Contamination probe: show the model the first 60 % of a test problem and measure how often it reproduces the rest verbatim. Memorised items get completed. |
| **perturbation test (GSM-Symbolic, GSM-Plus style)** | Rewrite test problems with different numbers or names but the same logic. The gap between original and perturbed accuracy is a memorisation signal. |
| **truncation rate** | Fraction of responses cut off at the max token limit before producing an answer. Must be reported, because a longer model looks "worse" if truncation is silently counted as wrong. |
| **extraction failure** | The scorer could not find an answer in the response. Scored wrong in the accuracy, and reported as its own rate next to it so a format problem can be told from a math problem. |
| **format rate** | Fraction of responses with well-formed `<think>…</think><answer>…</answer>` tags. |
| **math-verify** | Library that parses and compares mathematical answers so that `1/2`, `0.5` and `\frac{1}{2}` count as equal. |

## Training stages

| Term | Meaning |
|---|---|
| **base model** | Pre-trained only (next-token prediction on documents). No chat tuning, no RL. Qwen3-1.7B-Base here. |
| **SFT** (supervised fine-tuning) | Train on (prompt, response) pairs with cross-entropy on the response tokens. Teaches format and style. |
| **distillation (SFT distillation)** | SFT on responses generated by a stronger model (the teacher). Phase 2 uses DeepSeek-R1 traces; Phase 7 uses our own `grpo` model as teacher for the 0.6B student. |
| **RFT / STaR / expert iteration / ReST** | Rejection-sampling fine-tuning: sample many answers, keep the correct ones, fine-tune on them, repeat. RL with a binary reward but no policy-gradient machinery. |
| **RLVR** (reinforcement learning with verifiable rewards) | RL where the reward is checked programmatically (answer matches, code passes tests) instead of by a learned reward model or humans. |
| **GRPO** (Group Relative Policy Optimization) | Policy-gradient method from DeepSeekMath. Samples a group of G responses per prompt and uses the group's mean reward as the baseline, so no value network is needed. |
| **PPO** (Proximal Policy Optimization) | The classical RLHF algorithm. Clips the probability ratio to keep each update small. Needs a value model for the baseline. GRPO keeps the clipping and replaces the value model with the group mean. |
| **DAPO** | A GRPO variant: asymmetric clipping (clip-higher), dynamic sampling, token-level loss, overlong penalty. |
| **Dr. GRPO** | "GRPO Done Right": removes the per-sequence length normalisation and the std normalisation that bias toward longer wrong answers. |
| **DPO** (Direct Preference Optimization) | Offline method: trains directly on (chosen, rejected) pairs with a logistic loss on the log-probability margin relative to a reference model. No sampling during training. |
| **on-policy / off-policy** | On-policy: training data is sampled from the current model (GRPO, RFT). Off-policy: data came from an older or different model (DPO on stored pairs). Off-policy pairs go stale as the policy moves. |
| **reference model (π_ref)** | Frozen copy of the starting policy. The KL penalty keeps the trained policy from drifting too far from it. With LoRA, it is the same network with the adapters switched off. |
| **LoRA** (Low-Rank Adaptation) | Train small low-rank matrices added to the weight matrices instead of the full weights. `r` is the rank, `α` a scaling factor. Cheaper memory, slightly lower ceiling than full fine-tuning. |
| **full FT** | Fine-tune every parameter. |
| **AWQ** | Activation-aware weight quantisation to 4-bit for faster, smaller inference. "Group size 128" means one scale per 128 weights. |
| **vLLM** | Fast inference engine used for evaluation sampling and for generating rollouts inside the RL loop. "Colocate" mode runs it on the same GPU as training. |
| **budget forcing** | Test-time trick from the s1 paper: cap the number of thinking tokens, or force the model to keep thinking by appending "Wait" before it stops. |
| **test-time compute (TTC)** | Spending more inference compute per problem (more samples, longer thinking) instead of more training compute. Parallel TTC = more samples (maj@k, best-of-N); sequential TTC = longer single chain (budget forcing). |

## Loss and optimisation

| Term | Meaning |
|---|---|
| **CE** (cross-entropy) | The standard next-token loss: `-log p(correct token)` averaged over tokens. Used in SFT, RFT and distillation. |
| **loss masking (assistant-only loss)** | Compute CE only on response tokens, not on the prompt, so the model is not trained to predict the question. |
| **packing** | Concatenate several short training sequences into one long one to avoid padding waste. Needs a block-diagonal attention mask so sequences cannot see each other ("cross-contamination" of attention). |
| **PG** (policy gradient) | Loss whose gradient pushes up the log-probability of actions with positive advantage and down those with negative advantage. |
| **advantage (A)** | How much better a response was than expected. In GRPO: `(reward − group mean) / (group std + ε)`. |
| **group-normalised** | The advantage is computed within the group of G samples for the same prompt. |
| **zero-variance group / zero-advantage group** | All G samples got the same reward (all right or all wrong). Every advantage is 0, so the group contributes no gradient. Wasted compute; dynamic sampling resamples to avoid it. |
| **clipping, ε** | PPO/GRPO limit on how far the probability ratio new/old may move per update: ratio kept in `[1−ε, 1+ε]`. |
| **clip-higher (ε_high)** | DAPO's larger upper bound (0.28 vs 0.2) so low-probability tokens can grow, which keeps entropy from collapsing. |
| **ratio (importance ratio)** | `π_new(token) / π_old(token)`. Exactly 1 if you take one gradient step per batch of rollouts (μ = 1), which makes the clip inert. |
| **μ** | Number of optimisation epochs over one batch of rollouts. μ = 1 here. |
| **KL (Kullback-Leibler divergence)** | Measure of how far the trained policy has moved from the reference. Added to the loss with weight β. |
| **k1 / k2 / k3 estimators** | Three per-token ways of estimating KL from samples. k1 = log-ratio (unbiased, high variance, can be negative); k2 = half the squared log-ratio; k3 = `exp(−d) − 1 + d` with d the log-ratio (unbiased, low variance, always ≥ 0). GRPO papers use k3. |
| **β (KL coefficient)** | Weight of the KL penalty. 0.04 default; the A1 ablation sets it to 0. |
| **entropy (policy entropy)** | `−Σ p log p` over the vocabulary at each generated position, averaged. Low entropy = the model is very confident / less diverse. A sudden fall ("entropy collapse") usually precedes stalled RL. |
| **entropy collapse** | Policy entropy drops near zero, sampling becomes deterministic, groups become zero-variance, learning stops. |
| **token-level vs sequence-level loss** | Token-level: average the loss over all tokens in the batch (long responses weigh more). Sequence-level: average per response first, then over responses (each response weighs the same, which penalises long correct answers less and long wrong answers less too). Dr. GRPO argues for token-level without length normalisation. |
| **length normalisation** | Dividing a sequence's loss by its length. Dr. GRPO shows it biases toward longer incorrect responses. |
| **reward hacking** | The policy raises reward without doing the intended thing: gaming the format bonus, repeating answers, padding with "Wait". |
| **LR** (learning rate) | Step size of the optimiser. Typical here: 1e-5 for SFT, 1e-6 for GRPO, 5e-7 for DPO. |
| **schedule: cosine, constant, warmup** | How LR changes over training. Warmup: ramp up from 0 for the first few steps. Cosine: decay following a half cosine to a floor. Constant: fixed after warmup (used for GRPO). |
| **AdamW, β=(0.9, 0.95)** | The optimiser and its two momentum coefficients. **wd** = weight decay. |
| **gradient clipping (clip 1.0)** | Rescale the gradient if its norm exceeds 1.0, for stability. |
| **gradient norm** | Length of the gradient vector before clipping. Spikes mean instability. |
| **grad ckpt (gradient checkpointing)** | Recompute activations in the backward pass instead of storing them, to save memory at the cost of compute. |
| **bf16** | 16-bit brain-float precision for weights and activations; standard for training on modern GPUs. |
| **effective batch** | Sequences per optimiser step after gradient accumulation across micro-batches. |
| **epoch** | One pass over the training set. |
| **rollout** | One sampled response from the policy during RL. 512 rollouts per step = 64 prompts × G 8. |
| **G** (group size) | Number of responses sampled per prompt in GRPO. |
| **BCE** (binary cross-entropy) | Loss for a yes/no classifier, used to train the ORM. |
| **σ** (sigmoid) | `1 / (1 + e^−x)`; appears in the DPO loss. |
| **GPU-hour** | One GPU running for one hour; the unit of the compute ledger. |

## Datasets and benchmarks

| Term | Meaning |
|---|---|
| **GSM8K** | 8.5k grade-school math word problems (7.5k train, 1319 test). Numeric answers. |
| **MATH / MATH-500** | 12.5k competition problems with LaTeX answers and difficulty levels 1-5. MATH-500 is the 500-problem test subset popularised by the PRM800K paper. |
| **AIME24** | 30 problems from the 2024 American Invitational Mathematics Examination. Too few and too hard for a 1.7B model; sanity row only. |
| **Countdown** | Number game: combine given numbers with + − × ÷ to hit a target. Programmatically verifiable; used as the OOD task (as in TinyZero). |
| **OpenR1-Math-220k** | 220k math problems with DeepSeek-R1 reasoning traces, built from NuminaMath 1.5. SFT source after filtering and decontamination. |
| **CoT** (chain of thought) | Step-by-step reasoning written before the answer. "4-shot-CoT" = four worked examples in the prompt. |
| **few-shot / zero-shot** | Number of solved examples placed in the prompt before the question. |
| **think tags** | `<think>…</think>` for reasoning, `<answer>…</answer>` for the final answer; the format all trained checkpoints use. |

## Infrastructure and process

| Term | Meaning |
|---|---|
| **HF** | Hugging Face: model and dataset hub; also the `transformers` library. |
| **TRL** | Hugging Face library with SFTTrainer, GRPOTrainer, DPOTrainer. |
| **verl** | Alternative RL framework (ByteDance); the fallback if TRL's vLLM path fails. |
| **W&B** | Weights & Biases, experiment tracking and plots. |
| **model card** | README on a Hugging Face model page: data, config, compute, eval table, known failures. |
| **pre-registration** | Writing the hypothesis and the falsifying evidence in `experiments.md` before running the experiment, so the conclusion is not fitted to the result. |
| **ablation** | Remove or change one component and measure the effect, keeping everything else fixed. |
| **compute-matched** | Two methods compared at equal GPU-hours, not at "whatever each needed". |
| **ruff / pytest / pre-commit** | Linter, test runner, and the hook that runs them before each commit. |

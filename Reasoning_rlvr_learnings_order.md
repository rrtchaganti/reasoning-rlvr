Here is the order the project uses, which is also the natural learning order since each method builds on the previous one's checkpoint and on a question the previous one raised.

0. Evaluation and the from-scratch references (Phase 0)
Before any training: a deterministic eval harness (batch-invariant vLLM, bootstrap CIs, paired comparisons), and reference implementations under src/rlvr/ref/: the SFT cross-entropy loss, the GRPO loss with its three aggregations, group advantages, the k1/k2/k3 KL estimators, the unbiased pass@k estimator, and a minimal sampler with KV cache. Learn these first because every later method is read through them.

1. Prompting baselines and the pass@k ceiling (Phase 1)
Zero-shot, 4-shot, 4-shot-CoT and think-tag prompting of the base model; pass@k at n = 64 to measure latent capability (the sharpening-versus-capability question from Yue et al.); contamination and memorisation probes (n-gram overlap, perturbed GSM8K, partial-prompt completion). Not a training method, but it is the reference point every training method is judged against.

2. SFT on distilled traces (Phase 2)
Supervised fine-tuning on OpenR1-Math-220k chain-of-thought traces in the think-tag format. Teaches the format and a style. Sub-topics learned here: learning-rate sweep, data-scaling curve, packing with flash attention, overlap filtering of training data against evals.

3. Rejection-sampling fine-tuning, RFT or expert iteration (Phase 3)
Sample the SFT policy on training prompts, keep correct finished traces (shortest two per problem), fine-tune on them. The cheapest form of on-policy learning. Sub-topics: solve-rate distributions and "learnable" prompts, token-matched and step-matched controls, ablating the correctness filter versus the length filter, the per-problem decomposition into termination versus reasoning.

4. GRPO with verifiable rewards (Phase 4)
Group-relative policy optimisation from the RFT checkpoint: binary reward (finished and well-formed and correct), group-mean baseline, clipped ratio, KL penalty to the reference, reward-hacking watch. The ablations are where the related algorithms enter:
    - KL weight off (is the reference model needed),
    - DAPO's pieces: dynamic sampling (refill zero-variance groups) and clip-higher,
    - Dr. GRPO: removing the per-sequence length normalisation and the std scaling,
    - the 2,048-token cap run (the reward-shaping effect of the cap).

5. DPO (Phase 5, planned)
Direct preference optimisation on pairs, now reframed as a mechanism test: finished-correct versus looping pairs, off-policy, no rollouts. Learn it as the contrast to GRPO: offline pairwise versus online group-relative.
6. Test-time compute (Phase 6, planned)
    Majority vote, best-of-N with a trained outcome reward model, budget forcing in the style of s1. The accuracy-versus-tokens chart that compares inference-time methods with the trained policies.

7. Distillation, quantisation, serving (Phase 7, planned)
    Distilling the 1.7B teacher into Qwen3-0.6B, AWQ 4-bit quantisation, vLLM throughput and latency. Engineering deliverables rather than research.

The through-line to hold while learning: 2 is off-policy on someone else's traces, 3 is on-policy without a gradient-based policy objective, 4 is on-policy with one, 5 is off-policy pairwise, 6 spends compute at inference instead of training. The project's running question across 2 to 4 is whether each step bought reasoning, sharpening, or termination.

## Resources
What I can give you now, from the project context, is the mapping I would start from, so the main session has less to check. The plan's section 10.1 already names five papers the project tests directly; the rest are the standard companions for each step of the learning order:
- Step 0, references and evaluation. Chen et al. 2021 (Codex) for the unbiased pass@k estimator; Schulman's "Approximating KL Divergence" note for k1/k2/k3; vLLM's batch-invariance write-up (Thinking Machines, "Defeating Nondeterminism in LLM Inference") for why greedy evals drift.
- Step 1, baselines and the ceiling. Qwen3 technical report (the calibration numbers); Yue et al. 2025, "Does RL Really Incentivize Reasoning Capacity Beyond the Base Model" (the pass@k framing); GSM-Symbolic (Mirzadeh et al.) for the perturbed-GSM8K control.
- Step 2, SFT. The OpenR1 / DeepSeek-R1 distillation section (R1 paper, the distilled-model part); "LIMA" or "s1" for the small-data-SFT view; the Minerva paper for the 4-shot-CoT prompt convention.
- Step 3, RFT. Yuan et al. 2023, "Scaling Relationship on Learning Mathematical Reasoning with LLMs" (the RFT paper); STaR (Zelikman et al. 2022); ReST-EM (Singh et al. 2023) for expert iteration.
- Step 4, GRPO and ablations. DeepSeekMath (Shao et al. 2024, GRPO's definition); PPO (Schulman et al. 2017); DAPO (Yu et al. 2025: clip-higher, dynamic sampling, token-level loss, overlong shaping); Dr. GRPO (Liu et al. 2025, "Understanding R1-Zero-Like Training"); DeepSeek-R1 for the reward design and the "aha" claims; for the reward-hacking discussion, the Open-Reasoner-Zero or SimpleRL-Zoo reports and the "Spurious Rewards" paper (Shao et al. 2025) are the ones that bear on a verifiable-reward setting.
- Step 5, DPO. Rafailov et al. 2023 (DPO); the "Iterative Reasoning Preference Optimization" paper (Pang et al. 2024) for DPO on reasoning chains, which is closest to the finished-versus-looping pairs reframing.
- Step 6, test-time compute. s1 (Muennighoff et al. 2025, budget forcing); Snell et al. 2024, "Scaling LLM Test-Time Compute Optimally"; Lightman et al. 2023 ("Let's Verify Step by Step") for the ORM/PRM distinction; Wang et al. 2022 for self-consistency.
- Step 7, distillation and serving. Hinton et al. 2015 (knowledge distillation); the AWQ paper (Lin et al. 2023); the vLLM paper (Kwon et al. 2023, PagedAttention).
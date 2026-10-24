# Derivations, each tied to what this project measured

`PLAN.md` section 10.3's list. Each item gives the argument in a few lines, then the figure, table or
test in this repo that shows it, or says plainly that the project did not measure it.

**1. From the policy gradient to GRPO.** The policy gradient is E[R grad log pi(y|x)]. Subtracting a
baseline b that does not depend on y leaves it unbiased, because E[grad log pi] = 0, and lowers its
variance. PPO replaces log pi with the ratio r = pi / pi_old and clips it to [1 - eps, 1 + eps] so
that a step on stale samples cannot move the policy far. GRPO drops the value model: the baseline
is the mean reward of the G samples drawn for the same prompt, scaled by their std. Because the
mean includes the sample's own reward, the expected gradient is scaled by (G - 1) / G, not turned
(a leave-one-out mean would remove even that); the division by the group std, which also depends on
the sample, is what makes the estimator biased (item 3). *Here:* `tests/test_ref_grpo_loss.py::test_gradient_equals_reinforce_with_baseline_when_ratio_is_one`.

**2. The ratio is 1 after one update per batch.** When each batch of completions is used for exactly
one optimiser step, pi_old equals pi at the moment the loss is computed: r = 1, nothing is clipped,
and the clipped objective's gradient is the REINFORCE-with-baseline gradient. *Here:* the clip
fraction was 0 throughout every GRPO run (`outputs/reports/figures/phase-4-grpo-curves.png`), and
`test_single_update_ratio_is_one_and_nothing_clipped`. The consequence for item 4 is direct.

**3. Length and std normalisation (Dr. GRPO).** Averaging the loss per sequence before averaging
over sequences gives each token of a short completion more weight than a token of a long one: a
wrong long answer is penalised less per token, a pressure towards length. Dividing advantages by
the group std up-weights groups whose rewards barely differ, which are the easiest or hardest
prompts. *Here:* both biases are pinned by tests (`test_sequence_agg_weights_short_completions_more`,
`test_std_normalisation_inflates_low_variance_groups`); the main run used both, and its mean
completion fell from 1,406 to 1,201 tokens rather than growing. The ablation that would isolate the
loss (EXP-014c, Dr. GRPO) is pre-registered and was not run.

**4. Asymmetric clipping and entropy.** Raising only the upper clip bound (DAPO's clip-higher) lets
low-probability tokens with positive advantage grow, which slows entropy collapse. It can only act
when the ratio leaves [1 - eps, 1 + eps]. *Here:* it never did (item 2), and the entropy at step 200
was 0.427 for the main run against 0.425 with clip-higher and dynamic sampling
(`outputs/reports/phase-4.md`, session 2): the null result of EXP-014b is what item 2 predicts.

**5. Zero-advantage groups.** A group whose G rewards are all equal has zero advantage everywhere
and contributes no gradient, only compute. *Here:* about 30 % of groups had zero variance throughout
the main run (`phase-4-grpo-curves.png`); dynamic sampling (EXP-014b) refilled them at 1.5x the
compute and changed nothing, because the learnable-prompt filter had already left 32 informative
groups in 199 steps of 200.

**6. The k1, k2 and k3 KL estimators.** With samples from pi and r = pi_ref / pi: k1 = -log r is
unbiased but can be negative and has high variance; k2 = (log r)^2 / 2 is biased but low-variance;
k3 = (r - 1) - log r is unbiased, always non-negative and lower-variance than k1, which is why GRPO
puts k3 in the loss. *Here:* `tests/test_ref_advantages_kl.py::test_k1_and_k3_unbiased_k2_biased_on_discrete_toy`
and `test_k3_nonnegative_and_lower_variance_than_k1`. Measured: the main run's KL stayed near 0.006
per token; with the KL term off the offline KL to RFT was 0.098, 15 times larger (EXP-014a).

**7. Capability or sharpening.** If RL only re-weights solutions the base already samples, pass@1
rises and pass@k at large k does not (it can fall). *Here:* `phase-4-pass-at-k-grpo.png` and the
per-problem counts: on GSM8K, GRPO over RFT at k = 16 is -0.3 [-0.9, 0.3], no problem newly solved in
32 tries; on MATH-500 +1.9 [0.4, 3.5], 10 newly solved and 2 lost. Mostly sharpening, a small
widening on the harder set.

**8. Parallel against sequential test-time compute; verifiers against votes.** Parallel: sample k
answers and aggregate (vote, best-of-k, weighted vote). Sequential: let one answer run longer or
nudge it ("Wait"). A weak verifier's arg-max over k samples is over-optimised as k grows (the more
candidates, the likelier a wrong one outscores every right one); its scores as vote weights are not.
*Here:* the final chart (`outputs/reports/figures/final-accuracy-vs-tokens.png`); "Wait" equals the
plain stop at the same tokens (EXP-021); best-of-32 by the 0.6B verifier is 12-13 points below
majority-of-32 on MATH-500 (EXP-022). No process reward model was trained.

**9. Distillation against RL at 0.6B; quantisation; serving.** Not measured: dropped on 22 Oct 2026
(`docs/plan-after-phase-4.md` section 4). The one non-generic version, whether the termination
behaviour distils, is recorded there as an open question.

**10. What it cost, and what ten times the compute would buy.** About 55 H100-hours and 11 L4-hours
(USD 260) plus about 40 laptop GPU-hours (`COMPUTE.md`). A GRPO run cost 3.1 H100-hours, RFT's
training 12 minutes, the stop rule nothing. Ten times the compute would go to GRPO on harder prompts
(EXP-018) with two seeds, the Dr. GRPO loss (EXP-014c) and a matched-incorrect RFT set (EXP-010i); not
to more steps of the same configuration, whose reward plateaued by step 50.

# Experiment registry

Pre-registered before each run; closed with a conclusion after. One entry per run or per
family of runs that share a hypothesis. Config hashes come from `run_info.json`.

Template:

```
## EXP-NNN  <title>                                               [phase N] [status: planned|running|done]
**Question.** (link to PLAN.md section 6 Q-id if any)
**Hypothesis.** What we expect and why.
**Metric.** Exactly what number decides it, on which eval, with what CI.
**Falsifier.** What result would make us reject the hypothesis.
**Config.** path(s) + config hash(es), seed(s), commit.
**Result.** The number(s), with CI. Link to report / W&B.
**Conclusion.** "At this scale and budget, X; this does not tell us Y."
```

---

## EXP-000  Harness self-consistency: three identical greedy evals agree      [phase 0] [status: done]
**Question.** Section 4.1: what is the run-to-run variation of a temperature-0 vLLM eval?
**Hypothesis.** Greedy decoding with a fixed seed is deterministic up to vLLM batching effects;
pass@1 on GSM8K-test varies by well under 0.5 points across three identical runs.
**Metric.** pass@1 on GSM8K-test and MATH-500 for Qwen3-0.6B-Base, few_shot_cot, three runs.
Report max minus min.
**Falsifier.** Spread above 1 point: the harness is not reproducible and every later
comparison needs this noise added to its CI.
**Config.** `configs/eval/phase0_smoke.yaml`; seeds 0, 0, 0; default vLLM mode at commit 1366950, batch-invariant mode at commit f4d1360. Local RTX 4060, vllm 0.31.0, torch 2.13.0.
**Attempt 1 (commit 9a2edfa, discarded).** One run only. It found three harness bugs rather than measuring noise: MATH-500 61 % extraction failures (prompt ended on the wrong token boundary, model emitted end-of-text), Countdown 100 % failures (pre-filled `<think>` as the last token; stop string stripped). GSM8K 67.0 [58.0, 76.0] on its 100-problem subset looked sane and 12 of 12 inspected wrong samples were genuine model errors. Fixed in the following commit; see FAILURES_AND_LEARNINGS.md (14 Oct). Output kept locally as `run1_before_prompt_fixes`.
**Result.** Default vLLM mode (1366950), pass@1 % on 100-problem subsets: GSM8K 68 / 70 / 68 (spread 2), MATH-500 25 / 27 / 31 (spread 6), Countdown 1 / 2 / 2 (spread 1). The completion *text* differed across runs on 16 / 55 / 69 of 100 problems, sometimes from the first character. Hypothesis rejected: the falsifier (spread above 1 point) fired on two of three evals.
With `VLLM_BATCH_INVARIANT=1` (two runs at 1366950, then three runs at f4d1360 where it is the default): GSM8K 68 / 68 / 68, MATH-500 33 / 33 / 33, Countdown 2 / 2 / 2; text identical on 300 / 300 problems. Cost: sampling 68 s vs ~60 s per run, engine start-up ~80 s vs ~70 s. Report: `outputs/reports/phase-0.md`.
**Conclusion.** vLLM greedy decoding is not reproducible run-to-run in its default mode at this scale; on 100-problem subsets the noise is 1-6 points, which would swamp most ablation effects. Batch-invariant mode removes it entirely and is now the harness default, so every reported number is a function of (commit, config, seed) only. This does not tell us anything about accuracy: batch-invariant mode is a different fixed reduction order (MATH-500 33 vs 25-31 here), so comparisons are only valid within one mode.

---

## EXP-001  Base-model greedy baselines against the Qwen3 technical report      [phase 1] [status: done]
**Question.** Section 6 (feeds Q1, Q5, Q7, Q9, Q10): what can Qwen3-1.7B-Base (and the 0.6B student) do before any training, and is the harness calibrated?
**Hypothesis.** With the 4-shot-CoT prompt, greedy pass@1 on GSM8K-test lands within 3 points of the report's 75.44 (1.7B) and 59.59 (0.6B). MATH-500 lands within about 5 points of the report's full-MATH 43.50 / 32.44 (MATH-500 is a representative subset, so the gap can be a few points either way). Zero-shot is well below 4-shot; the think-tag prompt is the weakest on GSM8K/MATH for a base model; Countdown is near zero for both models.
**Metric.** pass@1 with 95 % bootstrap CI over problems, four templates x {GSM8K-test 1319, MATH-500 (and by level), Countdown 500, AIME24 30}, both models, batch-invariant vLLM, local RTX 4060. Extraction-failure and truncation rates reported next to every number.
**Falsifier.** 4-shot-CoT GSM8K more than 3 points from the report on either model means the harness or prompt is wrong, not the model; nothing downstream runs until it is explained. MATH-500 more than 8 points off triggers the same check.
**Config.** `configs/eval/base_{1.7b,0.6b}_{zero_shot,few_shot,few_shot_cot,think_tag}.yaml`; seed 0; commit ffd3018; local RTX 4060.
**Result.** 4-shot-CoT GSM8K 74.5 [72.0, 77.0] (1.7B; report 75.44) and 61.9 [59.3, 64.6] (0.6B; report 59.59); MATH-500 41.4 [37.0, 45.8] and 28.0 [24.0, 31.8] (report 43.50 / 32.44 on full MATH). Zero-shot with the boxed instruction is higher on both models: 77.6 / 53.6 (1.7B) and 64.9 / 37.6 (0.6B). Answer-only 4-shot collapses (20.5 / 9.8). Think-tag: GSM8K 61.3 with 36 % truncation, Countdown 1.2 % (1.7B) and 2.0 % (0.6B). MATH-500 by level (1.7B, 4-shot-CoT): 67.4 / 64.4 / 51.4 / 35.2 / 15.7. Full tables: `outputs/reports/phase-1-baselines.md`; report `outputs/reports/phase-1.md`.
**Conclusion.** Calibrated: the harness reproduces the report's 4-shot-CoT numbers within 1-2 points on GSM8K and 2-4 on MATH-500 for both models. The hypothesis that think-tag is the weakest prompt held for GSM8K but not for MATH-500 (43.0 vs 41.4 on 1.7B, within CI). The unexpected result is that zero-shot beats 4-shot-CoT by 10-12 points on MATH-500; both rows are kept for every later checkpoint. This does not tell us anything about sampling headroom (EXP-002).

## EXP-002  The pass@k ceiling: how much is latent in sampling?      [phase 1] [status: deferred, runs in the Phase 2 rental session]
**Question.** Q5 reference point (Yue et al.): the base model's pass@64 is the bar every later RL checkpoint's pass@1 is compared against.
**Hypothesis.** At T = 0.6, top-p 0.95, n = 64, 4-shot-CoT: pass@64 exceeds greedy pass@1 by more than 15 points on GSM8K and more than 25 points on MATH-500 for the 1.7B base; maj@64 sits between pass@1 and pass@64 and is above greedy by at least 5 points on both. Countdown pass@64 is above 10 % even though pass@1 is near zero (the format is reachable by chance).
**Metric.** Unbiased pass@k and maj@k curves for k in {1, 2, 4, 8, 16, 32, 64} with bootstrap CIs; mean sample accuracy vs greedy pass@1 as the sampling-noise reference; tokens per sample. Every sample saved to `outputs/samples/base/` and uploaded to the project's HF dataset repo (reused for DPO pairs and the sharpening analysis).
**Falsifier.** pass@64 minus pass@1 under 10 points on MATH-500: little latent capability, so a later RL gain cannot be "sharpening" and the Q5 framing changes.
**Config.** `configs/sample/base_1.7b_n64.yaml` (max 4096 new tokens, seed 0, `--no-batch-invariant` because reproducibility comes from the stored samples); rented A100; commit TBD.
**Status note (14 Oct 2026).** Not run in Phase 1: no rented GPU yet. Decision: merge Phase 1 with this criterion marked "not met, because no rented GPU yet" and run EXP-002 first thing in the Phase 2 rental session (`docs/rental-setup.md`, section 3). Hypothesis, metric, falsifier and config above stay as pre-registered; the result goes here, the figure and the HF link go into `outputs/reports/phase-1.md` and `docs/phase-1-engineering-notes.md` by an addendum commit on the Phase 2 branch.
**Result.** TBD
**Conclusion.** TBD

## EXP-003  Contamination and memorisation baselines of the base model      [phase 1] [status: done]
**Question.** Q9 baseline (section 4.3): how much of the base model's GSM8K/MATH performance is recall?
**Hypothesis.** (a) Partial-prompt completion: given the first 60 % of a test problem's text, the base model continues it near-verbatim (normalised Indel similarity >= 0.9 on the held-out 40 %) on fewer than 2 % of GSM8K-test and MATH-500 problems. (b) Perturbation gap: original minus perturbed GSM8K (numbers resampled, names swapped, answers recomputed) is between 2 and 6 points for 4-shot-CoT greedy, in line with GSM-Symbolic-style results for models of this size. (c) 13-gram overlap between every eval set and the training files to be used in Phases 2-4 (GSM8K-train, MATH-train, any filtered prompt file) is at most a handful of hits, all templates or boilerplate.
**Metric.** (a) completion rate with Wilson CI; (b) paired bootstrap CI of the per-problem difference on 1319 matched pairs; (c) hit counts and the ids to drop.
**Falsifier.** Completion rate above 5 % or perturbation gap above 8 points: the base has memorised the test set in part, GSM8K numbers are treated as recall-contaminated and the perturbed set becomes the primary GSM8K metric for every later checkpoint.
**Config.** `configs/probe/partial_prompt_1.7b.yaml`, `rlvr-perturb-gsm8k --seed 0` (manifest hash b5afb83e6aacafcf, 830 / 1319), `scripts/perturbation_gap.py`, `scripts/probe_report.py`; commit ffd3018; local.
**Result.** (a) Partial-prompt completion at similarity >= 0.9: GSM8K 2.1 % [1.5, 3.1], MATH-500 16.3 % [13.3, 19.8]; restricted to held-out parts of >= 12 words: 1.1 % [0.6, 1.9] and 11.6 % [7.9, 16.7]; chance 0. (b) Perturbation gap (numbers + names), paired on 830: 10.1 [7.0, 13.3] for 1.7B 4-shot-CoT, 10.7 [8.0, 13.6] zero-shot, 8.8 [5.8, 11.9] for 0.6B 4-shot-CoT; flips 131 / 47. (c) Training-data overlap deferred to Phase 2 (no training files yet).
**Conclusion.** Falsifiers fired twice: MATH-500 completion is above 5 % (genuine verbatim recall of specific problem text, read by eye), and the GSM8K gap is above 8 points. The GSM8K part is resolved by EXP-003b below as arithmetic brittleness, not recall; GSM8K is not treated as memorised, the perturbed set stays the primary reasoning-vs-recall GSM8K metric. MATH-500 gains in later phases are read next to the level breakdown and Countdown transfer, never alone.

## EXP-003b  Names-only perturbation: is the GSM8K gap recall or arithmetic?      [phase 1] [status: done]
**Question.** Follow-up to EXP-003 (b). The number-and-name perturbation gap on the 1.7B base came out at 10.1 points
[7.0, 13.3] (paired, 830 problems), above the 8-point falsifier. Two explanations: the base model has memorised
test problems (recall), or it is brittle to larger numbers (the resampled values are drawn from [x/2, 2x], so the
arithmetic gets harder on average). A names-only variant separates them: names carry no arithmetic.
**Hypothesis.** Names-only gap is below 2 points (the model does not recall problems by their characters), so most
of the 10-point gap is arithmetic brittleness, which is a reasoning finding, not a contamination finding.
**Metric.** Paired gap original minus names-only on the problems that have a swappable name, 4-shot-CoT greedy, 1.7B base,
with the paired bootstrap CI and the flip table; the number-only gap (`--no-names`) as the third arm if time allows.
**Falsifier.** Names-only gap above 5 points with a CI excluding 2: recall is a large part of the gap and GSM8K is treated as
contaminated for this model, as EXP-003 said.
**Config.** `rlvr-perturb-gsm8k --names-only --seed 0` (516 / 1319, manifest hash 4ae8f836dfe6960f); `configs/eval/base_1.7b_few_shot_cot_names_only.yaml`; commit 585a586; local, 3 GPU-minutes.
**Result.** Names-only gap -0.4 [-3.3, 2.5] (original 73.8, names-only 74.2, flips 28 / 30 = 11 %). Numbers + names on the same 371 problems: 10.8 [6.2, 15.4] (flips 61 / 21).
**Conclusion.** Hypothesis confirmed: the whole gap comes from the numbers. The 1.7B base is brittle to resampled (on average larger) numbers and does not recall GSM8K problems by their surface form. Side result: 11 % of problems flip under a null change, the per-problem noise floor of a single greedy eval on this model.

## EXP-004  Training-data overlap between OpenR1-Math-220k and every eval set      [phase 2] [status: planned]
**Question.** D3 part (a), PLAN.md section 4.3 probe 1: how many eval problems appear in the SFT source, and which training items must be dropped before any training run.
**Hypothesis.** The `default` split of OpenR1-Math-220k (93,733 problems, NuminaMath-1.5 sources) contains 20-40 MATH-500 problems by 13-gram or fuzzy match (published audits found 20-30), under 5 GSM8K-test problems, and 0 AIME24 problems; it also contains a sizeable part of MATH-train (the `math` source), which matters because the dev split is drawn from MATH-train. Countdown is generated and cannot overlap.
**Metric.** Per eval set: number of eval problems hit by a 13-gram, number hit by normalised Indel similarity >= 0.9, the union, and the number of training items dropped; the ten worst hits printed for reading by eye. Evals: gsm8k, math500, aime24, gsm8k_perturbed, gsm8k_names_only, gsm8k_dev, math_dev.
**Falsifier.** More than 5 % of MATH-500 (25 problems) hit by the *fuzzy* detector after the 13-gram drop: the dedup is not catching paraphrases and the filter must be tightened before training. More than 1 % of GSM8K-test hit: GSM8K is in the SFT source and the perturbed set carries more weight.
**Config.** `rlvr-build-sft` (overlap stage writes `outputs/reports/phase-2-overlap/*.json`), source revision pinned in `data/manifests/sft_openr1.json`; CPU only.
**Result.** TBD
**Conclusion.** TBD

## EXP-005  SFT learning-rate sweep on a 10 % subset      [phase 2] [status: planned]
**Question.** PLAN.md Phase 2 step 2: which peak LR for the full fine-tune, chosen on the dev split and never on a test set.
**Hypothesis.** Over {5e-6, 1e-5, 2e-5} (AdamW 0.9/0.95, wd 0.1, cosine to 10 %, 3 % warmup, 1 epoch on a seeded 10 % subset of the filtered set), 1e-5 gives the best dev accuracy; 2e-5 is within noise on GSM8K-dev but loses on MATH-dev levels 1-2 (the "LR too high erases base knowledge" failure mode from PLAN.md), and 5e-6 under-fits the format (higher tag-format failure rate).
**Metric.** Greedy think-tag accuracy on the dev split (gsm8k_dev 250 + math_dev 250, max 4096 new tokens) with 95 % CIs, tag-format failure rate, final training loss. The LR with the highest pooled dev accuracy is used for EXP-006, 007 and 008; ties within 2 points go to 1e-5 as pre-registered.
**Falsifier.** The three dev accuracies are within one CI of each other: LR is not a sensitive knob at this scale and data size, and the sweep is reported as a null result (still 1e-5 for the main run).
**Config.** `configs/train/sft_1p7b_lr{5e-6,1e-5,2e-5}.yaml` (subset_fraction 0.10, 1 epoch), `configs/eval/ckpt_think_tag_dev.yaml`; rented A100, about 3 x 20 min plus evals; commit TBD.
**Result.** TBD
**Conclusion.** TBD

## EXP-006  SFT on distilled traces: what the format and a competent style buy      [phase 2] [status: planned]
**Question.** Q1 (the SFT term), Q7, Q9, Q10. PLAN.md Phase 2 steps 1-2.
**Hypothesis.** Full fine-tune of Qwen3-1.7B-Base for 2 epochs on the filtered, decontaminated OpenR1 traces rewritten into the think/answer format (LR from EXP-005) raises think-tag greedy pass@1 on GSM8K from 61.3 to at least 75 (meeting the base's own 4-shot-CoT row), on MATH-500 from 43.0 to at least 50, with the gain concentrated on levels 3-5, and on Countdown from 1.2 to under 10 (the format transfers, the arithmetic search does not). Tag-format failures fall below 1 %. The perturbation gap on GSM8K stays within 3 points of the base's 12.3 (think-tag); the partial-prompt completion rates do not rise (SFT on a MATH-containing source is exactly where recall could enter, which is why the overlap drop in EXP-004 runs first). pass@64 (n = 64, T 0.6) rises by less than pass@1 does: SFT mostly sharpens toward the teacher's style; the ceiling moves little.
**Metric.** Paired bootstrap difference SFT minus base on every eval, same problems (think-tag at max 4096 new tokens for both, so the base think-tag row is re-run at 4096 locally as the paired reference and the 1024-token Phase 1 row is not reused); MATH-500 by level; truncation and format-failure rates; pass@k and maj@k curves k <= 64 from the same n = 64 sampling as EXP-002; both the 4-shot-CoT and zero-shot rows of the checkpoint for comparability with Phase 1.
**Falsifier.** GSM8K think-tag pass@1 below 70 or MATH-500 below 46 after 2 epochs with the chosen LR: SFT on this source at this length cap does not deliver the R1-distillation gain at 1.7B, and the recipe comparison in Phase 4 starts from a weaker policy. MATH-500 levels 1-2 falling by more than 3 points with a CI excluding 0: knowledge erasure, LR is lowered and the run repeated once. Perturbation gap widening by more than 3 points or completion rate rising above the Phase 1 numbers: recall entered through the SFT data (Q9).
**Config.** `configs/train/sft_1p7b.yaml`, data manifest `data/manifests/sft_openr1.json` (content hash recorded), `configs/eval/sft_1.7b_{think_tag,few_shot_cot,zero_shot}.yaml`, `configs/sample/sft_1.7b_n64*.yaml`; rented A100 (training about 3-5 h depending on the kept-token count, which the data report fixes before the run); commit TBD.
**Result.** TBD
**Conclusion.** TBD

## EXP-007  LoRA r = 64 versus full fine-tune at the same data and epochs      [phase 2] [status: planned]
**Question.** PLAN.md Phase 2 step 3: the LoRA-versus-full number, needed to justify where LoRA is used later (0.6B jobs locally) and where it is not.
**Hypothesis.** LoRA (r 64, alpha 128, dropout 0.05, all linear layers, LR 2e-4, same data and 2 epochs) lands within 3 points of the full fine-tune on GSM8K think-tag and at least 3 points below it on MATH-500 (LoRA learns less on the harder, more knowledge-heavy eval; Biderman et al. 2024), at a lower peak memory and a similar wall clock.
**Metric.** Paired difference full minus LoRA on GSM8K, perturbed GSM8K, MATH-500 (by level) and Countdown, think-tag greedy, 95 % paired CIs; peak GPU memory and tokens per second from the trainer logs.
**Falsifier.** LoRA within 1 point of full on MATH-500 with a CI including 0: at this scale and data size the adapter is enough, and full fine-tunes later in the project are not justified by this evidence.
**Config.** `configs/train/sft_1p7b_lora.yaml`; rented A100; commit TBD.
**Result.** TBD
**Conclusion.** TBD

## EXP-008  SFT data-scaling curve: 25 % / 50 % / 100 % of the filtered set      [phase 2] [status: planned]
**Question.** PLAN.md Phase 2 step 4: is data the bottleneck of SFT at 1.7B, and where does the curve bend.
**Hypothesis.** With the same LR and 2 epochs, think-tag accuracy rises roughly log-linearly in the number of traces: from 25 % to 100 % MATH-500 gains at least 5 points while GSM8K gains fewer than 3 (GSM8K-style problems saturate early); Countdown stays flat (no Countdown-like data in the source).
**Metric.** The three points (25, 50, 100 % = EXP-006 run) on GSM8K, perturbed GSM8K, MATH-500, Countdown, think-tag greedy, with CIs; one figure (`scripts/plot_data_scaling.py`), log-x; the 25 and 50 % subsets are seeded, nested samples of the full set so each is a subset of the next.
**Falsifier.** MATH-500 difference between 25 % and 100 % under 2 points with a paired CI including 0: more of this data does not help at this scale, and the Phase 3/4 budget should go to RL steps rather than to more SFT traces.
**Config.** `configs/train/sft_1p7b_data25.yaml`, `sft_1p7b_data50.yaml` (subset_fraction 0.25 / 0.50, same seed); rented A100, about 0.75 x the EXP-006 wall clock together; commit TBD.
**Result.** TBD
**Conclusion.** TBD

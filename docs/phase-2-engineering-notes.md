# Phase 2 engineering notes: SFT on distilled traces

Companion to `outputs/reports/phase-2.md` (results), `docs/phase-2-journal.md` (the chronological
story) and the 15 Oct entries in `FAILURES_AND_LEARNINGS.md`. What was built, how it is tested, why it
is designed this way, and the questions it invites. Written when the code was complete (15 Oct 2026);
the sections marked *after the rental* are filled in when the training runs have been done.

---

## 1. What was built

| piece | file | tests |
|---|---|---|
| SFT data pipeline from OpenR1-Math-220k | `src/rlvr/data/sft_data.py` (`rlvr-build-sft`) | `tests/test_sft_data.py` (12) |
| Coverage-based n-gram contamination | `src/rlvr/data/contamination.py` (`ngram_overlap(min_coverage=)`) | `tests/test_contamination_and_runinfo.py` (+2) |
| Paired pass@k of two n-sample runs (perturbed vs original; base vs SFT curves) | `rlvr.eval.metrics.paired_pass_at_k`; `scripts/perturbation_gap.py --all-samples`; `scripts/compare_checkpoints.py --all-samples` | `tests/test_eval_metrics.py`, `tests/test_perturbation_gap.py`, `tests/test_phase2_scripts.py` (+3) |
| Sampling-run projection from the vLLM progress line (the ten-minute rule) | `scripts/project_sampling.py` | `tests/test_project_sampling.py` (7) |
| Dev split and MATH-train loader | `src/rlvr/eval/datasets.py` (`gsm8k_dev`, `math_dev`, `math_train`) | `tests/test_sft_data.py` |
| Reference SFT loss and packing masks | `src/rlvr/ref/sft_loss.py` | `tests/test_ref_sft_loss.py` (6) |
| Trainer, incl. the untrained-tag-row handling | `src/rlvr/train/sft.py` (`rlvr-sft`) | `tests/test_train_sft.py` (8 CPU + 2 GPU + 2 slow) |
| Paired checkpoint comparison | `scripts/compare_checkpoints.py` | `tests/test_phase2_scripts.py` |
| LR choice, data-scaling figure, model card and uploads | `scripts/pick_lr.py`, `scripts/plot_data_scaling.py`, `scripts/push_to_hub.py` | `tests/test_phase2_scripts.py` |
| Context-budget guard in the eval harness | `src/rlvr/eval/sampler.py` (`context_budget`), `run_eval.py` | `tests/test_eval_harness_end_to_end.py` (+1, +1 slow) |
| Rental session | `scripts/phase2_rental.sh` (resumable, 8 steps + `speed`) | `bash -n`; exercised on the box |
| E2E volume routine and the cheaper-card rule | `scripts/e2e_attach.sh`, `scripts/e2e_env.sh`, `scripts/project_run_time.py` | `tests/test_phase2_scripts.py` (+2); shell scripts exercised on the box |
| Configs | `configs/train/*.yaml` (8), `configs/eval/{sft_*,ckpt_think_tag_dev,base_1.7b_think_tag_4096}.yaml`, `configs/sample/sft_*.yaml` | config loading tested |

Suite at the end of the code phase: 155 tests on CPU (about 40 s), 2 GPU-marked and 3 slow tests
(the trainer's four pass on the laptop's RTX 4060).

## 2. The data pipeline: what each stage is for

**Source.** `open-r1/OpenR1-Math-220k`, `default` config (93,733 problems, the curated split), revision
pinned in the manifest. Each row carries several DeepSeek-R1 traces with a per-trace math-verify
verdict.

**Stages, with the drop counts of the final build.**

1. *Select* the first trace math-verify judged correct (28,627 rows have none).
2. *Parse* it: `<think> reasoning </think> summary`, the summary ending in exactly one `\boxed{}`.
   Multi-box answers (5,965) are multi-part answers our grader cannot score; no box, no closing tag,
   unbalanced braces are malformed (about 700).
3. *Rewrite* into the project's format. Prompt = the think-tag template used by `rlvr-eval` and later
   by RL, byte for byte (a test asserts it), ending `Assistant:⏎<think>⏎`. Completion = R1 reasoning,
   then the clean summary, then `</think>`, then `<answer> value </answer>`.
4. *Dedup* on the normalised problem text (721).
5. *Cap* at 4096 Qwen3 tokens for prompt + completion, counted with the real tokeniser in batches
   (34,650 dropped: the median R1 trace is about 5k tokens).
6. *Decontaminate* against every eval set and the dev split (57 dropped).

Result: 23,000 examples, 61.8 M tokens, content hash `45b3fa4862b11e2a`, manifest committed.

**Why the summary goes inside the think block.** The answer block is what math-verify grades in
Phase 4; keeping it to the bare value makes the reward unambiguous. The summary is still worth
teaching (a wrap-up before answering), so it closes the think block. Alternative rejected: summary in
the answer block, which would make the RL reward depend on parsing prose.

**Why ids are built, not copied.** The source's `uuid` has 22 "NaN" rows and 11 repeats; the trainer
keys by id. `unique_id` adds the row index on a missing or repeated uuid; a test asserts uniqueness.

**Why a 4096 cap and not longer.** PLAN fixes 4k for SFT; the eval protocol allows 4096 new tokens;
the KV cache of a 1.7B model at 8k would halve the rental's throughput; and the cap keeps nothing
truncated mid-answer at train time.

## 3. Contamination: from "any 13-gram" to coverage

The Phase 0 detector follows the GPT-3 convention: an eval problem is hit if any of its 13-token
windows appears in a training item. On this source that reports AIME24 as 27 % contaminated through
one template sentence ("where m and n are relatively prime positive integers, find m + n"), in many
variants. A document-frequency cutoff was tried first and failed: each variant is individually rare
among the 23k kept items. The rule that holds is **coverage**: a pair counts when the shared windows
cover at least half of the eval problem's words. A copy is covered end to end; a template covers one
line. Every hit item is still dropped from training (conservative); only the reported count changes.
Final: MATH-500 3 / 500 (one verbatim, one 0.91 paraphrase, one n-gram-covered), AIME24 0, GSM8K 0
everywhere, MATH-dev 2 / 250.

**Tests.** The template-versus-copy case (19-word template inside 45-word problems: template not
counted, copy counted, both dropped); the coverage arithmetic on an 8-token toy; the earlier
verbatim-copy and number-swap cases unchanged.

## 4. The reference loss and why packing is refused without flash attention

`rlvr.ref.sft_loss` writes the objective out: labels of −100 on prompt and padding, token-mean
cross-entropy with the one-position shift, first-fit packing with positions restarting per sequence,
and a block-diagonal causal mask. The test runs a tiny random Qwen3 on three sequences separately and
packed and demands equal per-token losses to 1e-4; then it feeds the same packed row with a plain
causal mask and shows the later sequences' losses change. That is the cross-contamination PLAN warns
about, made concrete.

The test found a real trap on the way: a boolean 4D mask is *added* to the scores by transformers
5.17's eager path (True → +1.0), not applied. `additive_mask` converts to 0 / −inf first. Consequence
for the trainer: TRL only warns when best-fit packing runs on SDPA or eager, and the leak is silent,
so `rlvr-sft` raises unless the attention backend is a flash variant or `allow_unsafe_packing` is set
on purpose. The local 0.6B smoke therefore runs with `packing: false`.

## 5. The trainer: what the wrapper adds to TRL

- A YAML config with `key=value` overrides, hashed into `run_info.json` with the commit, the data
  hash and the subset size, so every checkpoint says exactly what it was trained on.
- The manifest check: the training file's content hash must equal the committed manifest's, else the
  run refuses to start (the Phase 1 "silently different dataset" lesson, enforced).
- Nested subsets: one seeded shuffle, prefixes of it, so the 25 % set is inside the 50 % set which is
  inside the full set (the data-scaling points differ only in how much data, not which).
- `completion_only_loss` on the prompt/completion format; `packing_strategy="bfd"` with
  `padding_free` when packing is on; AdamW (0.9, 0.95), weight decay 0.1, cosine to 10 % of peak,
  3 % warmup (`warmup_steps` takes a float ratio in transformers 5; `warmup_ratio` is gone).
- LoRA through PEFT, merged into the base weights at the end so vLLM evaluates a plain checkpoint;
  the adapter is saved too.
- The trainer's log history, wall clock, peak memory and tokens per epoch saved next to the model.
- **Untrained format tokens.** `<think>` and `</think>` are single added tokens whose embedding rows
  in Qwen3-*Base* are the untouched filler vector shared by every unused row (94 identical rows in
  the 1.7B, 291 in the 0.6B; found by the local smoke, whose LoRA checkpoint never emitted
  `</think>`). Before training, `rlvr-sft` detects such a row (bit-identical to other rows) and sets
  it to the mean of its plain-text pieces (`</`, `think`, `>`). Full fine-tunes then train it like
  any weight. LoRA runs train exactly those rows next to the adapters: the embedding matrix gets
  gradients, a hook zeroes every other row, the matrix leaves the weight-decay group, and the rows
  are held in fp32 during the run. PEFT's `trainable_token_indices` was tried first and is bypassed
  by TRL's fused LM head (it reads `lm_head.weight` directly). `run_info.json` records duplicates,
  norms and how far each row moved; a LoRA run that would leave an untrained tag frozen is refused.
  The second local smoke confirmed the mechanism: on the same 400 examples the checkpoint wrote
  `</think>` in 5 of 100 outputs (0 before), every one in the training position, and `run_info.json`
  showed both rows moved (journal, section 13).
- **Precision.** Full fine-tunes load fp32 master weights (`load_dtype: float32`) with bf16
  autocast; pure-bf16 AdamW at LR 1e-5 rounds most updates away (bf16 spacing at 0.02 is 1.2e-4).
  LoRA configs keep the frozen base in bf16; PEFT holds the adapters in fp32.

**Tests.** Config defaults, overrides and unknown keys; the nested-subset property; the manifest check
accepting and rejecting; subset-then-limit; the packing guard; tag-row detection and initialisation
on a tiny tied Qwen3 with filler rows; masked-row gradients touching only the tag rows; two-step
training smokes on that tiny model (GPU only: TRL 1.15's SFT loss is a fused Triton LM-head kernel
without a CPU path) for the full fine-tune in fp32 + autocast and for LoRA, where the tag rows must
move and the tie must survive merge and save; the refusals of unsafe packing and of frozen untrained
tags.

## 6. The dev split and the LR rule

250 GSM8K-train + 250 MATH-train problems, fixed seed, sampled over sorted ids so the split is
identical on every machine and immune to row order. It exists so that the learning rate (EXP-005) and
the checkpoint are chosen on held-out *training* problems and never on a test set. The SFT data is
decontaminated against it. The LR rule is pre-registered and mechanical (`pick_lr.py`): highest
pooled dev accuracy; ties within 2 points go to 1e-5.

## 7. Eval protocol for checkpoints

Think-tag at 4096 new tokens (the plan's protocol) for every checkpoint from now on; the base's
think-tag row is re-run at 4096 as the paired reference and the Phase 1 1024-token rows are not
reused. Both the 4-shot-CoT and zero-shot rows of each checkpoint are kept for comparability with
Phase 1. Every comparison is paired on shared problem ids (`compare_checkpoints.py`): accuracy of both,
the paired bootstrap difference, the flip counts, MATH-500 by level.

As run, the budget is a 4096-token *window* (`max_model_len: 4096`, chosen for the laptop's KV cache):
4096 minus the prompt, 3,865-3,994 new tokens. The base reference and the three checkpoint configs
share the window, so the cap is identical per problem on both sides of every pair; the dev-split and
n = 64 configs use 6144. `rlvr-eval` computes the budget per eval from the prompt lengths, prints a
warning when the window caps it and writes `context_budget` into `metrics.json`. The base reference
itself (journal, section 14) showed that the base's think-tag loops never end: every GSM8K output
truncated at 1024 was still running at 4096.

## 8. Budget

61.8 M tokens per epoch, two epochs, 1.7B parameters: about 1.3e18 FLOPs for the main run, 3-5
A100-hours with gradient checkpointing. The whole session, including the deferred EXP-002 and the
n = 64 sampling of the SFT checkpoint, is about 15 A100-hours (USD 25-40). The plan's line for SFT
was 3 hours; the difference is the ablations and the deferred sampling, and it fits the total budget.
Order of sacrifice if money runs short: the 50 % scaling point, then LoRA; never the main run.

*Actual (19 Oct 2026).* 17.75 H100-hours (about 32 A100-equivalents), INR 6,730, for the Phase 2
part of the TIR session: the main run took 3 h 02 min on the H100 with sdpa and no packing (the
plan's 3-5 A100-hours assumed packing), and the two n = 64 sampling runs of the SFT checkpoint took
6 h 50 min because its completions are 2-4x the base's. LoRA was the sacrifice, not the 50 % point:
at 1.26x the full fine-tune's step time it was the more expensive and less informative of the two.
Rows in `COMPUTE.md`.

## 9. Known limitations and debts

- The length cap removes 60 % of the correct traces; the kept set skews toward shorter problems. The
  data-scaling curve measures "more of the same kind of trace", not "longer traces".
- Multi-part answers are dropped rather than reformatted; the SFT set has no "a and b" style answers.
- No flash-attention wheel exists for torch 2.13 (checked 17 Oct: Dao-AILab 2.8.3.post2 stops at torch
  2.8). The L4 image's 12.6 toolkit could not build one; the TIR H100 image (nvcc 12.9) did, in 13
  minutes, but only after every Phase 2 training run had finished, so Phase 2 trained with sdpa and
  `packing: false` (exact attention, padded batches, 16 x 4 micro-batch) while Phase 3 trained packed
  with flash attention. The loss is the same (section 4); the composition of a micro-batch is not, and
  every cross-phase comparison says so. The wheel lives on the notebook disk and `setup_env.sh`
  installs it on later starts (`docs/rental-setup.md`, 2b).
- The LR sweep is 36 optimiser steps per point (10 % of the data, one epoch). It separated 5e-6 from
  the others and not 1e-5 from 2e-5; at this step count the checkpoints mostly do not close the think
  block, so the sweep measured early format acquisition as much as accuracy. A sweep at the main
  run's step count would cost 3x the main run and was never in the plan.
- The think-tag SFT checkpoint does not terminate a quarter of its generations (28 % truncation on
  GSM8K, 52 % on MATH-500 at 4,096 tokens; the same at 8,192). The trained format scores 7-13 points
  below zero-shot on the same checkpoint for this reason. Nothing in the Phase 2 recipe penalises
  length or rewards stopping; the data's own length distribution (mean 2,689, p90 3,790 tokens, from
  a 4,096 cap) is the likely teacher. A length-aware filter or a stop-token loss weight would be the
  Phase 2 fixes; Phase 3's selection of finished traces is the one this project tested.
- The model card's eval table prints one truncation and one extraction-failure column per template
  (the last eval's values), which misreads as a per-template rate; the per-eval rates are in the
  `metrics.json` files and the report. Fix in `push_to_hub.py` before the next card.
- `rlvr-eval` treated a flag equal to its parser default as absent (a config's value won), and a
  `--limit` on a derived set drew an independent subset. Both bit the Phase 3 part of the session
  and are fixed with tests on `phase-3-rft` (3e055ad); Phase 2's runs used no `--limit` and passed
  every cap through configs, so its numbers are unaffected.
- TRL warned on some SFT rows that the tokenised prompt is not a prefix of the tokenised
  prompt + completion (tokens merging across the boundary). TRL falls back to its own boundary
  handling; how many rows and whether the label mask shifts by one token is uncounted.
- The two n = 64 sampling runs of the SFT checkpoint cost 6 h 50 min of H100 time (285 M generated
  tokens) because its completions are 2-4x the base's; the ten-minute rule caught the length but the
  projection script assumed the greedy mean length, half the sampled one. The script now needs the
  sampled mean from the first finished completions, not the greedy eval's.
- The local trainer smoke runs on a random tiny model; the first real end-to-end check (data →
  trainer → merged checkpoint → `rlvr-eval`) is the 0.6B LoRA smoke on the laptop.
- The contamination coverage threshold (50 %) is a judgement; the per-eval JSON keeps every hit so the
  threshold can be re-read later.
- The base model's think-tag rows (Phase 1 and the 4096-token reference) measure a base that reads an
  untrained token after `Assistant:`; the paired think-tag comparison therefore flatters SFT by
  construction, and the fair base row is 4-shot-CoT (EXP-006 amendment). Every trained checkpoint has
  trained rows, so comparisons between checkpoints are unaffected.
- The mean-of-pieces initialisation is a heuristic (Hewitt-style mean-plus-noise would be the other
  common choice); the full fine-tune trains the rows for 46k examples afterwards, so the choice matters
  mostly for LoRA, where it should be read as part of the EXP-007 recipe.
- A LoRA adapter directory alone does not reproduce the checkpoint: the two tag rows live in the merged
  `final/`, not in the adapter.
- The tag rows train at the run's learning rate, and the masked-row mechanism keeps an fp32 copy, a
  gradient and two Adam moments for the whole embedding matrix (about 2.5 GB on the 0.6B, 5 GB on the
  1.7B) in order to move two rows. A separate, higher learning rate for new rows is the usual recipe
  and a sparse optimiser state the usual saving; neither was added before the rental. Revisit if the
  EXP-007 format rate lags the full fine-tune's.
- Countdown at n = 64 under the 4096 cap costs about 87 M generated tokens per checkpoint for a model
  whose completions loop (the base: 2,716 tokens a sample on average). Pre-registered and kept; the
  options if the A100 rate makes it unaffordable are in `docs/e2e-session-handoff.md`, section 1.
- The think-tag window is 4096 tokens in total, not 4096 new tokens (section 7). A true 4096 would
  need `max_model_len: 6144` in the four paired configs and a rerun of the base reference (about half
  an A100-hour); not done because the base's loops never end and the pair shares the cap.

## 10. Results and what they mean (19 Oct 2026; full tables in `outputs/reports/phase-2.md`)

- **EXP-002 complete.** pass@64 of the base: 98.8 GSM8K, 90.2 perturbed, 87.8 MATH-500, 20.0 AIME24,
  63.8 Countdown; maj@k saturates by k = 16 on the math sets and is useless on Countdown (11 %);
  sampling at T = 0.6 costs 3-5 points against greedy.
- **EXP-005.** 1e-5 wins the sweep (pooled dev 53.8 against 44.0 and 50.6); 5e-6 is separated, 1e-5
  and 2e-5 are not. After 36 steps every checkpoint mostly fails to close the think block.
- **EXP-006.** SFT lifts think-tag greedy accuracy by +13.0 [10.0, 15.8] on GSM8K, +14.6 on perturbed
  GSM8K, +5.8 [0.6, 11.0] on MATH-500, +2.6 on Countdown (paired, shared 4096 window); 4-shot-CoT
  +5.8 / +3.7 / +16.6 and zero-shot +4.3 / +2.8 / +9.4 against the base at the same cap. The bars
  (75, 50, under 10) are met in substance; no falsifier fires. Two parts of the hypothesis failed:
  the MATH-500 gain under the trained format sits on levels 1-2, and tag-format failures equal the
  truncation rate (28 / 52 %) instead of falling under 1 %, so the trained format is the worst of the
  three on its own checkpoint (74.8 against 82.2 zero-shot on GSM8K). Phase 3's runs on the same
  checkpoint show the deficit is looping, not the cap, and mostly a greedy-decoding degeneration.
  Perturbation gap unchanged (11.3 against 10.4); partial-prompt completion flat (no recall entered);
  pass@64 unchanged on GSM8K, +5.2 [2.4, 8.0] on MATH-500 with +15.3 at pass@1 (mostly sharpening);
  Countdown pass@64 62.0 against 63.8 with the greedy row at 3.8 % (a length failure).
- **EXP-007 not run** (stopped at step 20 for budget; the one fact: LoRA with fp32 tag rows is 1.26x
  the full fine-tune's step time on this card).
- **EXP-008.** GSM8K rises with data at every size (+4.0 [1.4, 6.7] from 25 to 100 %, paired); 9 of
  the 13 base-to-SFT points are there at 25 %. MATH-500 is flat within noise (+2.8 [-1.4, 7.0]):
  inconclusive, small if present; neither the hypothesis nor the falsifier is met. MATH-500 truncation
  jumps to 52-58 % at the smallest size and stays. More traces of this kind are not where the next
  MATH-500 point comes from.
- **What it means.** SFT on distilled traces buys the format and 4-17 points under prompts the model
  can finish in, and installs a habit of not stopping that costs 7-13 points at greedy in the trained
  format. The ceiling barely moves. The lever Phase 3 pulls is the obvious one: keep the traces that
  finish and are right.

## 11. Interview question bank for this phase

- *How did you turn a 220k-row distillation set into training data you can defend?* Section 2: one
  verified trace per problem, the format shared with eval and RL, the cap, dedup, decontamination with
  the counts, the manifest hash enforced by the trainer.
- *Your overlap check said AIME24 was 27 % contaminated. Was it?* No: every hit was a template
  sentence through one unrelated training item; a frequency cutoff failed because template variants are
  rare; coverage (half the problem reproduced) separates copies from templates. Read the pairs.
- *What does "loss on assistant tokens only" change and how do you know your packing is right?*
  Section 4: the labels, the shift, and the test that packed equals unpacked to 1e-4 while naive packing
  does not; plus the boolean-mask trap.
- *Why refuse to pack without flash attention when TRL only warns?* Because the leak is silent, the
  test shows it is real, and a warning in a log nobody reads is not a safeguard.
- *How are the LR and the checkpoint chosen?* On a fixed dev split of training problems, by a rule
  written before the runs; never on a test set.
- *Why nested subsets for the scaling curve?* So the points differ only in quantity; a fresh sample per
  point would confound quantity with composition.
- *Your SFT smoke scored 53 % on GSM8K and never closed the think block. What happened?* The format
  rate (0 %) said what the accuracy column hid. Qwen3-Base has never trained the `<think>`/`</think>`
  rows (bit-identical to 90-odd filler rows), LoRA cannot move an embedding row, and the tied output
  head made `</think>` unreachable. Fix: initialise the rows from their plain-text pieces, train
  exactly those rows next to the adapters, record the movement. Found for 20 GPU-minutes locally
  rather than after the rented run. The rerun with the fix wrote `</think>` in 5 of 100 outputs, each in the training position,
  with accuracy unchanged within the paired CI; the low count is 50 LoRA steps on 400 examples, not
  the mechanism.
- *Your protocol says 4096 new tokens. Was it?* As run, no: `max_model_len: 4096` makes vLLM stop at
  the window minus the prompt, 3,865-3,994 new tokens, and the harness said nothing. Caught on the
  local base reference from the lengths of the truncated outputs. Kept, because every paired row
  shares the window and the base loops never end anyway; stated in every table; the harness now
  prints and records the effective budget per eval.
- *Why fp32 master weights when everyone says "train in bf16"?* Because recipes that say so run
  DeepSpeed or FSDP with an fp32 master copy. On a single GPU with the plain Trainer, bf16 weights are
  the master copy, and an Adam step at LR 1e-5 is below the bf16 spacing of a typical weight.
- *What would you do differently?* Look at the embedding rows of every token in the format before
  designing the prompt (Phase 1's `<think>` garbage was the same bug, diagnosed one layer too shallow);
  verify the flash-attention wheel before the rental; build the coverage rule into the contamination
  module from the start; keep multi-part answers with a structured answer block.
- *Your base model has pass@64 of 99 % on GSM8K. What is left for RL to do?* Separate the two
  curves: maj@64 is 85.7 and flattens by k = 16, so the model already prefers the right answer on
  most problems and sharpening alone has about 13 points of room on GSM8K, 30 on MATH-500. The
  perturbed set and AIME24 say where capability, not preference, is the limit: 10 % of perturbed
  problems are never solved in 64 tries, and on AIME24 the right answer is reachable (pass@64 20 %)
  but never the plurality (maj@64 0).
- *Why is your sampled pass@1 below your greedy pass@1?* Temperature 0.6 spreads mass away from the
  argmax; mean sample accuracy is 3-5 points under greedy on this model. The pre-registered metric
  names it as the noise reference, so every later sampled row is read against the sampled base, not
  the greedy one.
- *You estimated the sampling run twice and were wrong twice. What changed?* The first miss was the
  context window (throughput falls with the window through KV-cache concurrency); the second was the
  completion length within one window (3,134 tokens/s at 154 tokens a sample, 816 at 2,700). The
  rule is now per set: tokens from the nearest run's mean length, rate from a set of similar length,
  most expensive set first when the budget is tight.
- *Your trained format scores below a plain zero-shot prompt on the same checkpoint. Why?* Because
  a quarter of the think-tag generations never terminate (28 % truncation on GSM8K, 52 % on MATH-500)
  and a truncated output has no answer. It is not the budget: at an 8,192-token cap the subset
  truncation is still 27 % and the Countdown outputs run to 7,900 tokens. It is mostly greedy
  decoding: at T = 0.6 truncation falls to 13 % and sampled pass@1 rises to 80.3. The checkpoint
  knows the answer (zero-shot 82.2); the format's stopping behaviour was never trained. The honest
  table reports both rows and says which one measures what.
- *Was the learning rate a sensitive knob?* At the low end: 5e-6 lost by 10 points of pooled dev
  accuracy; 1e-5 and 2e-5 were within one CI. At 36 steps the sweep also measured how fast each LR
  learns to close the think block (format failure 100 / 79 / 58 %), which is the quantity that
  actually separated them.
- *Your data-scaling curve is flat on MATH-500. Does more data not help?* The unpaired numbers
  (47.8 / 44.0 / 50.6) look flat; the paired difference between 25 and 100 % is +2.8 [-1.4, 7.0],
  which is "inconclusive, small if present", not "flat". GSM8K rises at every size (+4.0 [1.4, 6.7]).
  The shape is set by truncation: MATH-500 truncation jumps to 52-58 % at 5,750 examples and stays,
  so each size learns the long style and none learns to finish. The question "is data the
  bottleneck" has the answer "not for this kind of data"; a different kind (shorter, or finished)
  is Phase 3's.
- *You trained without flash attention. Does that change the result?* Not the loss: sdpa is exact
  attention and the packed and unpacked losses agree to 1e-4 (section 4). It changed the cost (1.5-2x
  per step) and the micro-batch composition, and because Phase 3 trained packed with flash, any
  SFT-versus-RFT comparison carries that backend difference in its caveat. The control run of Phase 3
  shares RFT's backend, so the causal comparison there is backend-matched.
- *Why did you cut the LoRA ablation and not the 50 % scaling point, when your own order of sacrifice
  said the opposite?* The order was written before the step times were measured. LoRA with the tag
  rows trained in fp32 ran at 1.26x the full fine-tune's step time (3 h 50 min for one point); the
  two scaling runs together were 2.6 h for two points on a three-point curve. Cheaper and more
  informative won; the pre-registration and the amendment stand for a later session, and the exit
  criterion is marked not met with the reason.
- *What did the session cost against the estimate, and why?* 17.75 H100-hours (INR 6,730) for the
  Phase 2 part against 10-12 planned. Two n = 64 sampling runs of the SFT checkpoint took 6 h 50 min
  because its completions are 2-4x the base's; the projection script assumed the greedy mean length.
  1.7 h was idle or debugging: a tmux name-prefix match, a stale secret in a shell copy, a YAML
  string. Each has a fix in the repo or the next-session guide.

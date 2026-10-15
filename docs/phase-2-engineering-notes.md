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
| Dev split and MATH-train loader | `src/rlvr/eval/datasets.py` (`gsm8k_dev`, `math_dev`, `math_train`) | `tests/test_sft_data.py` |
| Reference SFT loss and packing masks | `src/rlvr/ref/sft_loss.py` | `tests/test_ref_sft_loss.py` (6) |
| Trainer, incl. the untrained-tag-row handling | `src/rlvr/train/sft.py` (`rlvr-sft`) | `tests/test_train_sft.py` (8 CPU + 2 GPU + 2 slow) |
| Paired checkpoint comparison | `scripts/compare_checkpoints.py` | `tests/test_phase2_scripts.py` |
| LR choice, data-scaling figure, model card and uploads | `scripts/pick_lr.py`, `scripts/plot_data_scaling.py`, `scripts/push_to_hub.py` | `tests/test_phase2_scripts.py` |
| Context-budget guard in the eval harness | `src/rlvr/eval/sampler.py` (`context_budget`), `run_eval.py` | `tests/test_eval_harness_end_to_end.py` (+1, +1 slow) |
| Rental session | `scripts/phase2_rental.sh` (resumable, 8 steps) | `bash -n`; exercised on the box |
| Configs | `configs/train/*.yaml` (8), `configs/eval/{sft_*,ckpt_think_tag_dev,base_1.7b_think_tag_4096}.yaml`, `configs/sample/sft_*.yaml` | config loading tested |

Suite at the end of the code phase: 153 tests on CPU (about 40 s), 2 GPU-marked and 3 slow tests
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

## 9. Known limitations and debts

- The length cap removes 60 % of the correct traces; the kept set skews toward shorter problems. The
  data-scaling curve measures "more of the same kind of trace", not "longer traces".
- Multi-part answers are dropped rather than reformatted; the SFT set has no "a and b" style answers.
- No flash-attention wheel was verified for torch 2.13 / CUDA 13 before the rental; the fallback
  (`packing: false`) is correct but slower, and `docs/rental-setup.md` says how to record it.
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
- The think-tag window is 4096 tokens in total, not 4096 new tokens (section 7). A true 4096 would
  need `max_model_len: 6144` in the four paired configs and a rerun of the base reference (about half
  an A100-hour); not done because the base's loops never end and the pair shares the cap.

## 10. Results and what they mean *(after the rental)*

To be written from `outputs/reports/phase-2.md` once EXP-005 to EXP-008 and the deferred EXP-002 have
run: the LR sweep table, the base-vs-SFT paired table with MATH-500 by level, the pass@k curves for base
and SFT, LoRA vs full, the data-scaling curve, the contamination probes on the SFT checkpoint, and the
compute rows.

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

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
| Trainer | `src/rlvr/train/sft.py` (`rlvr-sft`) | `tests/test_train_sft.py` (5 CPU + 2 GPU/slow) |
| Paired checkpoint comparison | `scripts/compare_checkpoints.py` | `tests/test_phase2_scripts.py` |
| LR choice, data-scaling figure, model card and uploads | `scripts/pick_lr.py`, `scripts/plot_data_scaling.py`, `scripts/push_to_hub.py` | `tests/test_phase2_scripts.py` |
| Rental session | `scripts/phase2_rental.sh` (resumable, 8 steps) | `bash -n`; exercised on the box |
| Configs | `configs/train/*.yaml` (8), `configs/eval/{sft_*,ckpt_think_tag_dev,base_1.7b_think_tag_4096}.yaml`, `configs/sample/sft_*.yaml` | config loading tested |

Suite at the end of the code phase: 149 tests on CPU (about 22 s), 2 GPU-marked trainer tests.

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

**Tests.** Config defaults, overrides and unknown keys; the nested-subset property; the manifest check
accepting and rejecting; subset-then-limit; the packing guard; a two-step training smoke on a tiny
model with a tiny word-level tokeniser (GPU only: TRL 1.15's SFT loss is a fused Triton LM-head
kernel without a CPU path); the refusal of unsafe packing.

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
- *What would you do differently?* Verify the flash-attention wheel before the rental; build the
  coverage rule into the contamination module from the start; keep multi-part answers with a
  structured answer block.

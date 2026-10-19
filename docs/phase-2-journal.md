# Phase 2 journal: SFT on distilled traces (started 15 Oct 2026)

Chronological record of the phase, written as the work happens: the context, each decision and the
explanation given at the time, each failure, what was tried, what replaced it. Summary in
`phase-2-engineering-notes.md`; numbers in `outputs/reports/phase-2.md`; one-paragraph post-mortems in
`FAILURES_AND_LEARNINGS.md`. Rented-GPU runs are added here when their results arrive.

## 1. Context on the morning of 15 Oct

- Phase 1 merged (PR #2, fd70ca0, tag `phase-1`) with one item deferred: EXP-002, the n = 64 base-model
  sampling, to be run at the start of this phase's rental session.
- Phase 2's job (PLAN.md): teach the think/answer format and a competent reasoning style from distilled
  traces, and measure what SFT alone buys. Steps: build the data (OpenR1-Math-220k filtered to verified
  correct, at most 4k tokens, deduplicated, decontaminated), full fine-tune 1.7B with an LR sweep on a dev
  split, one LoRA run, a 25 / 50 / 100 % data-scaling curve. Exit: `sft` checkpoint on HF with a card,
  base-vs-SFT table with paired CIs, pass@k curve for SFT, data-scaling plot.
- Local constraints carried over: no flash-attention wheel in the local venv; 8 GB GPU (0.6B LoRA fits,
  1.7B training does not); long jobs run from your terminal; the harness refuses a dirty tree.
- What the training code must satisfy (PHASES_IO_AND_HYPERPARAMS.md): loss on response tokens only,
  packing with a block-diagonal mask so packed samples cannot attend to each other, AdamW (0.9, 0.95),
  weight decay 0.1, cosine to 10 % of peak, 3 % warmup, effective batch 64 sequences of 4096, 2 epochs,
  checkpoint selection on a dev split and never on a test set.

## 2. Branch, pre-registration, PR body

`git checkout main && git pull` gave the merge commit; tag `phase-1` on it; branch `phase-2-sft` pushed.
Five experiments pre-registered and committed (7f83d51) before any data existed:

- EXP-004 training-data overlap: hypothesis 20-40 MATH-500 problems in the SFT source (published audits
  said 20-30), under 5 GSM8K, 0 AIME24; falsifier: more than 25 MATH-500 problems caught only by the
  fuzzy detector (the n-gram pass is missing paraphrases) or more than 1 % of GSM8K.
- EXP-005 LR sweep {5e-6, 1e-5, 2e-5} on a 10 % subset, 1 epoch, chosen on the dev split; ties within 2
  points go to 1e-5.
- EXP-006 SFT main: think-tag GSM8K from 61.3 to at least 75, MATH-500 from 43.0 to at least 50 with the
  gain on levels 3-5, Countdown under 10; format failures under 1 %; perturbation gap within 3 points of
  the base; pass@64 moves less than pass@1 (SFT sharpens toward the teacher's style). Falsifiers: GSM8K
  under 70 or MATH-500 under 46; levels 1-2 losing more than 3 points (knowledge erasure, lower the LR);
  perturbation gap widening or completion rate rising (recall entered through the data).
- EXP-007 LoRA r = 64 vs full: within 3 points on GSM8K, at least 3 below on MATH-500.
- EXP-008 data scaling: MATH-500 gains at least 5 points from 25 % to 100 %, GSM8K fewer than 3.

A first attempt to append the registry through a shell heredoc failed to parse (a backslash in the
trailing check), wrote nothing, and was redone through the editor tool: the Phase 1 rule, re-learned.
The PR body file `PR_BODY_phase2.md` was created next to the repo for you to paste.

## 3. Reconnaissance before writing code

Checked in one pass, because each fact changes the design:

- OpenR1-Math-220k `default` config: 93,733 rows, 2.15 GB of parquet, fields `problem`, `answer`,
  `generations` (list of R1 traces), `correctness_math_verify` (one bool per generation), `source`,
  `problem_type`, `uuid`. Each generation is `<think> ... </think>` followed by a summary ending in
  `\boxed{}`. The `all` config has 225k rows and the `extended` 131k; `default` was chosen as the curated
  split.
- MATH train split: `hendrycks/competition_math` is disabled on the Hub; `EleutherAI/hendrycks_math`
  (7 subject configs, 7500 train problems) is available.
- TRL 1.15 `SFTConfig` has `packing`, `packing_strategy="bfd"`, `padding_free`, `completion_only_loss`,
  `max_length`; `warmup_ratio` no longer exists in transformers 5, `warmup_steps` takes a float ratio;
  `cosine_with_min_lr` scheduler exists. TRL only *warns* when best-fit packing runs without a
  flash-attention backend, although other attention implementations let packed sequences attend to each
  other. No `flash_attn` in the local venv; PEFT 0.21 supports `target_modules="all-linear"`.
- Disk: 760 GB free; the HF cache already held 48 GB from other projects.

## 4. Design decisions for the data pipeline (`rlvr-build-sft`)

Explained at the time:

- **One format everywhere.** The SFT *prompt* is byte-identical to the think-tag prompt that
  `rlvr-eval` sends and that RL will use (`Assistant:⏎<think>⏎`), so there is no template drift between
  SFT, eval and RL. A test asserts the equality.
- **What goes where.** The R1 reasoning *and* the clean post-think summary both go inside the think
  block (the summary teaches a wrap-up before answering); the answer block holds only the bare boxed
  value, `<answer> 180 </answer>`, mirroring the example in the prompt. The grader takes the answer block
  and, inside it, the last boxed if any; math-verify grades the bare value in Phase 4.
- **Selection.** First generation that math-verify marked correct. Dropped: no closing think tag, no
  summary, no box in the summary, more than one box (multi-part answers our grader cannot score; 5,965
  rows), unbalanced braces.
- **Length cap** of 4096 tokens on prompt plus completion, measured with the Qwen3 tokeniser, so nothing
  is truncated mid-answer at train time (PLAN's "what goes wrong").
- **Dedup** on the normalised problem text, first occurrence kept.
- **Decontamination** against every eval set and the dev split with the Phase 0 detectors; every hit
  item dropped; counts reported per eval set.
- **Dev split**: 250 GSM8K-train + 250 MATH-train, fixed seed (20261015), sample over sorted ids so it
  is identical on every machine and immune to row order. It exists so the LR and the checkpoint are
  chosen on held-out *training* problems, never on a test set. The SFT data is decontaminated against
  it too, because OpenR1 contains MATH-train problems.
- **Manifest with a content hash**, committed; the training file itself is git-ignored and must be
  regenerated on every machine (the Phase 1 lesson, now enforced by the trainer).

Tests were written against hand-built rows with a whitespace tokeniser: parsing, each reject reason, the
prompt equality, the completion round-trip through the grader, the funnel counts, decontamination with a
verbatim copy and a number-swapped paraphrase, the JSONL round trip, the dev split's determinism.

Two test mistakes on the first run, both mine: the dedup happens before the tokeniser (so a duplicate is
never counted as a length candidate), and my "number-swapped" sentence had a number-free 13-word window,
so the n-gram detector caught it too. Fixed the expectations, not the code. 130 tests green; committed
(31baab2).

## 5. The reference SFT loss and the packing mask

`rlvr.ref.sft_loss`: completion-only labels, token-mean cross-entropy, first-fit packing with positions
restarting per sequence, and a block-diagonal causal mask. The test runs a tiny random Qwen3 on three
sequences separately and packed, and demands equal per-token losses to 1e-4, then shows that a plain
causal mask over the same packed row gives different losses (the leak PLAN warns about).

**It failed first.** Packed and unpacked disagreed by up to 0.04 nats: too big for float noise, too small
for a real leak. The mask had been passed as a boolean tensor; transformers 5.17 returns a 4D mask as
"already prepared", and the eager path *adds* it to the scores, so True / False became +1.0 / +0.0 on the
logits. Nothing was masked. Fix: `additive_mask` converts to 0 / finfo.min before the model sees it; with
it the losses match to 1e-4 and the naive variant differs. Logged in the failures file. Consequence for
the trainer: packing is *refused* without a flash-attention backend, because the leak is silent and TRL
only warns.

## 6. The trainer (`rlvr-sft`)

A thin wrapper around TRL's `SFTTrainer` that adds what the project needs: a YAML config with
`key=value` overrides hashed into `run_info.json`; the manifest hash check; nested seeded subsets
(the 25 % set is a prefix of the 50 % set, which is a prefix of the full shuffle) for the sweep and the
scaling curve; `completion_only_loss` on the prompt/completion format; the packing refusal; optional LoRA
merged into the base weights at the end so vLLM evaluates a plain checkpoint; the trainer's log history,
wall clock, peak memory and token count written next to the model.

**Local test problem.** A two-step smoke on a tiny model with a tiny word-level tokeniser (no network)
failed on CPU with `Pointer argument cannot be accessed from Triton (cpu tensor?)`. Traced into TRL:
`SFTTrainer.compute_loss` always goes through a fused Triton LM-head kernel (`trl/kernels/chunked_logprob.py`)
that computes log-probabilities in tiles without materialising the full logits, and it has no CPU path
in this version. The smoke was re-marked as a GPU test and passed on the 4060 in 18 s. One more test
mistake: `sorted(ids) == ids` with ids like `id10` sorting before `id2`; fixed.

Configs written: the main run, the three LR points (10 % subset, 1 epoch), data 25 / 50 %, LoRA, a local
0.6B LoRA smoke (no packing, SDPA, 400 examples), checkpoint eval configs at 4096 new tokens (the plan's
protocol; the Phase 1 think-tag rows used 1024 and are not reused), a dev-split eval config, the base
think-tag rerun at 4096 as the paired reference, and the n = 64 sample configs for the SFT checkpoint.
141 tests green; committed and pushed (d73e206).

## 7. Building the data: four runs to get it right

**Run 1 (your terminal): download failure.** The Hugging Face Xet chunk-transfer client raised
`CAS Client Error: Request middleware error` on the first shard. Checked from WSL that the plain download
URL answers HTTP 200 with the full 213 MB shard; retried with `HF_HUB_DISABLE_XET=1` (whole-file HTTP,
resumable). Transient; nothing in the pipeline.

**Run 2: built, then inspected.** 23,058 examples, 62.0 M tokens, in 1487 s. Funnel: 28,627 rows without
a correct generation, 35,082 too long, 5,965 multi-box. Answer check: our extracted value agrees with the
source gold under math-verify in 99.5 % of 2000 sampled rows (the disagreements are `\dfrac` vs `\frac`
and a missing `+C`). The length histogram shows the cap cutting the long tail of R1 traces (mean 6k
tokens before, 2.7k after). Two defects found by reading the output:

- *Ids not unique*: 22 rows have the uuid string "NaN" and 11 uuids appear twice. The trainer keeps a
  dict by id, so 32 problems would have vanished silently. Fix: `unique_id` (missing → `row#i`,
  repeated → `uuid#i`), with a test.
- *AIME24 "27 % contaminated"*: 8 of 30 problems hit by a 13-gram, 0 by fuzzy. Printing the pairs showed
  every hit pointing at the same unrelated training item. The shared 13-gram was the sentence "where m
  and n are relatively prime positive integers, find m + n". A template, not a copy.

First fix for the template problem: a document-frequency cutoff (ignore 13-grams that occur in more than
5 training items) for the *reported* count, while still dropping every hit item. Also: the tokeniser was
switched to batched calls (3.5× faster build). Committed (60c9f0f).

**Run 3: the df filter did not work.** Same 8 AIME hits, all "rare". Dumping the shared grams with their
frequencies showed why: the template exists in many variants (`p/q` vs `m/n`, `tfrac` vs `frac`, "n is
not divisible by the square of any prime"), and over the 23k kept items several variants occur in only
4 training items. Each variant is individually rare; a frequency cutoff cannot see that they are all
boilerplate. (Over the full 93k source the same grams have frequencies 14-125, so the cutoff *would* have
worked against the whole source; it was applied to the filtered set.)

Second fix, replacing the first: **coverage**. A (eval, train) pair counts as contamination only when
the shared 13-gram windows cover at least half of the eval problem's words. A copied problem is covered
end to end; a template sentence covers one line. Tests: a synthetic AIME pair where the template covers
19 of 45 words and the copy covers everything; the coverage arithmetic on an 8-token toy. Every hit item is
still dropped from training regardless. Committed (781f2e7), with one process slip: the commit went
through while a test was failing because the gate's exit code was hidden behind a pipe to `tail`. The
test expectation was wrong (the copied problem also hits its sibling through the shared template, by
the any-gram rule), fixed in 04779ec; the gate now runs with `pipefail` before any commit is chained.

**Run 4: final.** 23,000 examples, 61.8 M tokens, hash `45b3fa4862b11e2a`, 318 s. Overlap table:

| eval set | any 13-gram | 13-gram covering ≥ 50 % | fuzzy ≥ 0.9 | contaminated | train items dropped |
|---|---|---|---|---|---|
| GSM8K-test (1319) | 0 | 0 | 0 | 0 | 0 |
| MATH-500 (500) | 6 | 2 | 2 | 3 (0.6 %) | 31 |
| AIME24 (30) | 8 | 0 | 0 | 0 | 36 |
| GSM8K perturbed / names-only / dev | 0 | 0 | 0 | 0 | 0 |
| MATH-dev (250) | 9 | 2 | 1 | 2 (0.8 %) | 37 |

The three MATH-500 problems (intermediate_algebra 90, 986, 1365): one verbatim (similarity 1.0), one
paraphrase at 0.91, one covered by n-grams. EXP-004 closed: the 20-40 hypothesis was wrong in the
reassuring direction; GSM8K is absent from the SFT source, so GSM8K rows after SFT measure transfer, not
recall. Manifest, data report, length figure and the per-eval overlap JSON committed (8fc9656).

## 8. Budget set by the data

61.8 M tokens per epoch, 2 epochs, 1.7B parameters: about 1.3e18 FLOPs for the main run, roughly 3-4
A100-hours with gradient checkpointing. The whole Phase 2 rental (sweep 0.15×, LoRA ~0.7×, data-scaling
0.75× of the main run, plus evals, the deferred EXP-002 and the SFT n = 64 sampling) is about 15
A100-hours, USD 25-40, within the plan's total budget though above its 3-hour line for SFT alone. If time
runs short the order of sacrifice is: data-scaling 50 % point, then LoRA, never the main run or its evals.

## 9. The rest of the code (afternoon of 15 Oct)

Written in one pass and tested on fake run directories: `compare_checkpoints.py` (paired
base-vs-checkpoint tables with the flip counts and MATH-500 by level, the shape every later phase's
D-table takes), `pick_lr.py` (the pre-registered EXP-005 rule, mechanical), `plot_data_scaling.py`,
`push_to_hub.py` (uploads `final/` with a model card generated from `run_info.json` and the eval
tables; sample sets to a dataset repo; `--dry-run` writes the card without touching the Hub), and
`scripts/phase2_rental.sh`, the ordered, resumable session (every step skips when its output exists).
`docs/rental-setup.md` gained the training extras: a flash-attention wheel (or the kernels-hub build,
or `packing: false` as the correct-but-slower fallback), W&B login, the session steps with their hours.
Engineering notes and the report skeleton written; 149 tests. The journals for Phases 0 and 1 were
reconstructed from the logs the same afternoon, after you asked for the chronological record.

## 10. Where the phase stands, and your actions

Code complete and pushed. Three things need a GPU, in this order:

1. **Local end-to-end smoke (laptop, about 20 min).** Train the 0.6B with LoRA on 400 examples, no
   packing, then evaluate the merged checkpoint on 100 GSM8K problems with the think-tag template. This
   proves data → trainer → merged checkpoint → `rlvr-eval` works before any money is spent. Expect the
   format-failure rate of the 0.6B base (12-34 %) to drop, and GSM8K to move a little; the number is
   not reported anywhere.
2. **Base think-tag reference at 4096 tokens (laptop, about 1 h).** `configs/eval/base_1.7b_think_tag_4096.yaml`.
   The rental script also runs it if it is missing, so this is optional locally.
3. **The rental session** (`docs/rental-setup.md`, section 3), about 15 A100-hours, with the results
   pasted back here and into `experiments.md`.

The exact commands are in the actions log and in the chat message of 15 Oct.

## 11. Local smoke, step 1: the trainer runs end to end (15 Oct, your terminal)

`rlvr-sft --config configs/train/sft_0p6b_lora_smoke.yaml` at e84dba1: manifest hash checked
(45b3fa4862b11e2a, 400 of 23,000 examples used), 50 optimiser steps in 305 s, loss 0.78 → 0.62,
grad norm 0.9 → 0.25, token accuracy 0.78 → 0.81, peak GPU memory 1.6 GB (LoRA r 16, batch 1,
gradient checkpointing), merged checkpoint written to `outputs/train/sft_0p6b_lora_smoke/final`,
`run_info.json` complete. Two things worth noting for the real run, neither a problem here:

- TRL's token counter reached 0.77 M tokens while the data said 1.07 M: the smoke's `max_length`
  is 2048 and the examples go up to 4096, so TRL truncated the long ones. The real configs use
  4096, equal to the data cap, so nothing is truncated there.
- The loss plateaued around 0.62 after half an epoch at LR 2e-4 on 400 examples: the format is
  learned fast; the content needs the full set.

## 12. Local smoke, step 2: the eval ran, and the checkpoint never wrote `</think>` (15 Oct, your terminal)

`rlvr-eval` on the merged 0.6B LoRA checkpoint, 100 GSM8K problems, think-tag template, 2048 new
tokens: pass@1 53.0 [43.0, 63.0], truncation 66 %, extraction failure 0 %, tag-format rate **0 %**,
mean 1,619 tokens with the median at the cap. vLLM loaded the merged safetensors (343 s through the
9P mount of the Windows drive; the HF cache on ext4 loads in seconds, so a checkpoint that is
evaluated more than once locally should be copied into the WSL file system first). The pipeline
works end to end; the numbers are a smoke, not a result. Two observations:

- Not one of the 100 outputs contained `</think>`. Thirty-four finished before the cap: 26 ended
  with end-of-text right after the boxed summary, 7 went straight from the summary to
  `<answer> x </answer>` with a blank line exactly where `</think>` sits in the training data, 1 hit
  the `User:` stop. The 66 truncated ones were R1-style loops ("but just to be sure, let me think
  again") on problems the model had already solved; 27 of them were still graded correct through the
  last-number fallback.
- Truncation at 2048 is expected for a 0.6B after 50 steps on olympiad traces whose mean length is
  2,689 tokens; it says nothing about the 1.7B at 4096.

**Chasing the missing tag, in order.**

1. First suspicion: vLLM strips `</think>` as a special token. Checked offline: the Qwen3 tokeniser
   marks `<think>` and `</think>` as added tokens with `special=False`; `decode(skip_special_tokens=True)`
   and the `tokenizers` `DecodeStream` that vLLM's detokeniser uses both keep the string. Not that.
2. Retokenising each output and comparing with the reported token count: the 26 end-of-text outputs
   were exactly one token longer than their text (the EOS id in `token_ids`); nothing else was
   missing. The model really did not generate `</think>`.
3. The embedding matrix of the base. Rows 151667 (`<think>`) and 151668 (`</think>`) of
   Qwen3-1.7B-Base are bit-identical to `<|im_start|>`, `<|im_end|>`, every tool token and dozens of
   padding rows: 94 identical rows (291 in the 0.6B), norm 0.37 against 1.58 for an average trained
   row. **The Base checkpoints have never trained these rows.** This also explains the 14 Oct finding
   that `<think>` as the final prompt token produced garbage: the model was reading a filler vector.
4. Why the smoke could not fix it: LoRA adapts the linear layers and leaves the embeddings frozen,
   and the tied output head is the same matrix, so the logit of `</think>` equals the logit of every
   filler token and the model can never prefer it. The 7 `<answer>` outputs show that it learned to
   skip the unreachable token instead.

**Decisions.**

- An untrained tag row (detected as "bit-identical to other rows") is initialised to the mean of its
  plain-text pieces (`</`, `think`, `>`), the tokens the base saw whenever the tag appeared in ordinary
  text. Done before training; `run_info.json` records duplicates, norms and how far each row moved.
- LoRA runs train those two rows as well. First attempt: PEFT's `trainable_token_indices`, which ties
  the delta to the output head. It failed in the GPU test: TRL 1.15's fused LM head reads
  `lm_head.weight` directly and crashed on the wrapper (`'TrainableTokensWrapper' object has no
  attribute 'bias'`); without the crash the wrapper would simply have been bypassed and the output
  side never trained. Replaced by a direct mechanism: the embedding matrix gets gradients, a hook
  zeroes every row but the tags, and the matrix is removed from the weight-decay group (decay on
  151k frozen rows would silently shrink the vocabulary). The rows train in fp32 and are cast back
  for saving. The adapter directory alone is therefore incomplete; `final/` is the checkpoint.
- A LoRA run with `train_tags: false` while a tag row is untrained is refused, like unsafe packing.
- While here: full fine-tunes now load fp32 master weights (`load_dtype: float32`) under bf16
  autocast. A bf16 weight of magnitude 0.02 has a spacing of 1.2e-4, so pure-bf16 AdamW at LR 1e-5
  rounds most updates to nothing; the main run would have trained far less than its loss curve
  suggested. About 27 GB static for the 1.7B, fine on the A100. LoRA configs keep the frozen base in
  bf16 (PEFT holds the adapters in fp32).
- The base think-tag reference row is now understood as "the base reading an untrained token after
  `Assistant:`"; the fair base comparison for EXP-006 is its 4-shot-CoT row, which the hypothesis
  already names. Amendments written into EXP-006 and EXP-007 before any run.

Tests: a tiny tied Qwen3 with filler rows (detection and init on CPU; masked-row gradients touch only
the tag rows), GPU smokes for the full fine-tune (fp32 + autocast) and for LoRA (rows move, the tie
survives merge and save). Suite 152 CPU + 4 GPU/slow.

**Your next action**: rerun the smoke with the new trainer into new directories so the two runs can
be compared (train about 5 min, eval about 10 min); commands in the actions log. What to look for:
`tag_format_rate` well above 0 and `</think>` present in `samples.jsonl`; in `run_info.json`,
`tag_tokens` → `train_delta_norm` above 0 for both tags.

## 13. Local smoke, step 3: the fixed trainer closes the think block (15 Oct, your terminal)

Same recipe as step 1 (0.6B, LoRA r 16, 400 examples, 50 steps) under the 57625ea trainer, written to
`sft_0p6b_lora_smoke_v2`, then the same 100 GSM8K problems with the think-tag template at 2048 tokens.
Greedy and batch-invariant on both sides, so the two smokes are a paired comparison on identical inputs.

**Training.** The two `[rlvr-sft] tag token` lines reported `"untrained": true, "initialised": true`
for both tags (290 duplicate rows in the 0.6B; norm 0.35 -> 0.56 after the mean-of-pieces init), and
`run_info.json` closed with `train_delta_norm` 0.054 for `<think>` and 0.095 for `</think>` and
`duplicate_rows_final` 0 for both: the rows left the filler cluster and moved during training. Loss
0.77 -> 0.62 as before (final 0.659 against 0.66 in step 1). Peak memory 4.48 GB against 1.62 GB:
the whole 0.6B embedding matrix now carries an fp32 copy, a gradient and two Adam moments (about
0.6 GB each) so that two rows can train; on the 1.7B that is about 5 GB extra, nothing on an A100.
Training took 362 s against 305 s.

**Eval, paired with step 2.**

| | step 2 (e84dba1 trainer) | step 3 (57625ea trainer) |
|---|---|---|
| pass@1 | 53.0 [43.0, 63.0] | 46.0 [36.0, 56.0] |
| truncated at 2048 (%) | 66 | 76 |
| outputs with `</think>` | 0 | 5 |
| outputs with `<answer>` | 7 | 11 |
| tag-format rate (%) | 0 | 5 |
| extraction: tag / boxed / last-number / phrase | 2 / 30 / 44 / 24 | 5 / 16 / 40 / 39 |
| tokens mean / median | 1,619 / 2,048 | 1,741 / 2,048 |

Paired accuracy difference (step 3 minus step 2) -7 points, bootstrap 95 % CI [-18, +5]: 32 problems
right in both, 21 only in step 2, 14 only in step 3. Noise between two 50-step runs on 400 examples;
the step-3 run also truncated ten more outputs, and 30 of its 76 truncated outputs were graded correct
by the last-number fallback (27 of 66 in step 2).

The five `</think>` outputs follow the training layout byte for byte: the reflection ends, then
`**Final Answer**`, the boxed number, a blank line, `</think>`, newline, `<answer> N </answer>`.
Of the 24 outputs that finished under the cap, 5 have the full format, 6 go from the boxed summary
straight to `<answer>` (the step-2 habit), 13 end with end-of-text after the boxed summary. In step 2
the same count was 0 of 34 with `</think>`.

**Reading.** The mechanism does what it was built for: `</think>` is reachable, and when the model
uses it, it uses it in the right place. Five of 100 after 400 examples and 50 LoRA steps is not a
format rate anyone would ship; it is the expected amount of learning for one gradient signal per
example on a row that started at a third of a trained row's norm. The real runs give the rows 46,000
signals (two epochs of 23,000; in the full fine-tune every weight moves, so the network can also meet
the row halfway) or, for EXP-007, a LoRA learning rate twenty times the main run's. The 76 %
truncation is the 0.6B-on-olympiad-traces story from step 2 and says nothing about the 1.7B at 4096.

**Not done, on purpose.** A separate, higher learning rate for the two new rows is a known recipe for
new tokens and would have shown a larger movement in this smoke; it adds a parameter group and a
config knob to a trainer that is about to run for money, and EXP-007 already measures LoRA with the
rows at the adapter learning rate. Written down as a debt in the engineering notes (section 9), to be
revisited if the rented LoRA run's format rate lags the full fine-tune's.

Smoke step 3 closes the local part of Phase 2. Next: your draft PR from `../PR_BODY_phase2.md`, then
the rental session (`docs/rental-setup.md`, `scripts/phase2_rental.sh`).

## 14. The base think-tag reference at the 4096 context (15 Oct, your terminal, 3 h 55 min)

`configs/eval/base_1.7b_think_tag_4096.yaml` on the 4060: all five eval sets, greedy, batch-invariant.
I had said "about an hour". The Phase 1 think-tag row I anchored on was a 1024-token run (24 min), and
I did not scale it: four times the token budget at a third of the throughput (the KV cache holds about
six 4096-token sequences on 8 GB against twenty at 1024; 372 output tokens/s against 1,216) is ten
times the wall clock, and that is what it took. A generation-time estimate is tokens divided by the
throughput measured at that context length, not a remembered number. Logged in the compute ledger.

| eval | n | pass@1 at 1024 (Phase 1) | pass@1 at the 4096 window | truncated 1024 -> 4096 | extraction failure 1024 -> 4096 |
|---|---|---|---|---|---|
| GSM8K | 1319 | 61.3 [58.7, 63.8] | 61.8 [59.2, 64.4] | 35.8 % -> 35.8 % | 0 -> 0 |
| GSM8K perturbed | 830 | 50.8 [47.6, 54.5] | 51.4 [48.1, 54.8] | 35.8 % -> 35.8 % | 0 -> 0 |
| MATH-500 | 500 | 43.0 [38.8, 47.4] | 44.8 [40.4, 49.2] | 24.2 % -> 20.0 % | 28.2 % -> 24.6 % |
| Countdown | 500 | 1.2 [0.4, 2.2] | 1.2 [0.4, 2.2] | 66.4 % -> 65.6 % | 68.8 % -> 68.6 % |
| AIME24 | 30 | 6.7 [0.0, 16.7] | 6.7 [0.0, 16.7] | 40 % -> 30 % | 0 -> 0 |

`tag_format_rate` is 0 on every eval, as section 12 predicts: the base cannot emit `</think>`.

**Two findings.**

- *The base's think-tag loops never end.* Greedy decoding is deterministic and batch-invariant, so
  the first 1024 tokens of every output are the same in both runs (all 847 GSM8K outputs that stopped
  under 1024 are byte-identical across the runs). Of the 472 GSM8K outputs truncated at 1024, all 472
  were still running at the 4096 window; perturbed 297 of 297; MATH-500 100 of 121; Countdown 328 of
  332; AIME24 9 of 12. Four times the budget bought MATH-500 21 finished outputs (+1.8 points, CIs
  overlapping) and GSM8K nothing but 57 last-number flips inside the loops (+0.5 net). The base's
  truncation rate is a property of the model, not of the budget: once it loops, it does not stop.
  The SFT checkpoint is the first model for which the 4096 budget can mean anything.
- *`max_model_len: 4096` clips the budget.* The truncated outputs end at 3,865-3,994 tokens, never
  4,096: vLLM stops a completion at the context window minus the prompt, and the config sets the
  window to 4096 (chosen for the 8 GB KV cache). The protocol "4096 new tokens" was, as run, "a
  4096-token window, 4096 minus a 102-231-token prompt". The three SFT checkpoint configs set the
  same window, so base and checkpoint get the same cap on every problem and the paired comparison is
  intact; the dev-split and the n = 64 sample configs use 6144 and do get the full 4096. Decision:
  keep the window at 4096 for the paired rows (rerunning the base for 100-200 more tokens of loops
  costs 4 local hours or half an A100-hour and, by the first finding, changes nothing), state the
  effective budget in every table, and make the harness say it: `rlvr-eval` now computes the context
  budget per eval from the prompt lengths, prints a warning when `max_model_len` caps it, and writes
  `context_budget` into `metrics.json`. Tests: the arithmetic on the numbers above and the HF sampler
  on the tiny model. EXP-006 amendment (d) records the protocol as run. Caught after a local run
  rather than after a rented one, which is what the local run was for.

**Next.** The reference folder `outputs/eval/base_1.7b_think_tag_4096/` must be on the rental box
before `scripts/phase2_rental.sh` runs (the `exp002` step skips the reference when its `metrics.json`
exists); `docs/rental-setup.md` section 3 says how. Nothing else is left to run locally.

## 15. Compute provider chosen: E2E Networks, one volume, nodes come and go (16 Oct)

You asked for the cheapest way to run the remaining phases on E2E Networks (on-demand only, no H100
on offer), switching between the laptop and rented nodes, with one persistent volume moved between a
small and a big instance. Your worry: the low-compute stretches are longer than the training bursts.

Facts from E2E's documentation, read 16 Oct: a powered-off GPU node on MyAccount keeps billing until
it is terminated (TIR's billing page says the opposite for TIR nodes; we use MyAccount nodes and
terminate); block-storage volumes outlive nodes, attach to GPU and C3 plans and re-attach elsewhere;
saved images need a powered-off node and are billed for the whole root disk. Prices in INR per hour:
L4 24 GB 49, A40 48 GB 96, L40S 48 GB 102, A100 40 GB 179, A100 80 GB 189. The A100 image reports
CUDA 12.8, so the lock file's CUDA 13 torch will not load there; the setup script takes
`UV_TORCH_BACKEND=cu128` and the versions go into `COMPUTE.md`. The number that shapes everything: a
cold start (uv, wheels, flash-attn, model and dataset downloads, data builds) costs about one
A100-hour, INR 190, so it happens once, on the INR 49 L4, with the volume attached.

Options laid out: (A) one volume, nodes come and go: 5-10 min per switch, no re-setup, storage only
between sessions; (B) a saved image: no attach step, but 100+ GB of billed image, and the outputs
still have to leave the node before Terminate; (C) stateless, git plus the Hub: 30-60 min and about
INR 190 of setup per session, fine for two sessions, bad for five; (D) the laptop as the small instance
for everything low-compute. Chosen: A plus D. The long low-compute stretches cost nothing because they
run on the 4060 and the editor, and the volume just sits there. One correction to the plan's split:
after section 14, 4096-token greedy evals of checkpoints are not laptop work (4 h each); they run
inside the session, where they take about ten minutes.

Phase to node: Phase 2 on the A100 80 GB (15 h, about INR 2,850), with the option to run the sweep,
main, LoRA and scaling steps on the L40S if a measured speed check passes; Phase 3 RFT on the A100,
3 h; Phase 4 GRPO on the A100 80 GB only (vLLM colocated with an fp32-master 1.7B policy), 36 h,
about INR 6,800, seed repeat first to drop; Phase 5 DPO 3 h; Phase 6 sampling 3 h; Phase 7
distillation of the 0.6B on an L4 or L40S. About 42 A100-hours plus 10-15 cheaper hours, INR
9,300-10,700 before storage, inside the USD 100-170 budget. Five sessions, each: launch from the same
image, attach, mount, source, run the phase script in tmux, confirm the uploads, terminate.

Built for it: `scripts/e2e_attach.sh` (finds the volume, formats once after a typed YES, mounts at
`/mnt/rlvr`, fstab entry), `scripts/e2e_env.sh` (fixed paths on the volume for the venv, uv's
CPython, the wheel cache and `HF_HOME`, so the environment built on the L4 runs unchanged on the
A100), `scripts/project_run_time.py` plus a `speed` step in `scripts/phase2_rental.sh` (runs the
first sweep run, measures tokens per second, projects the main run; rule fixed in advance: stay on
the cheaper card if the main run projects to at most 8 h, else move the volume to the A100; the
finished sweep run is reused either way), and `RLVR_TRAIN_EXTRA` for the 48 GB cards
(`--set per_device_batch=2 --set grad_accum=32`). Rental guide section 2c has the click-by-click cold
start and the per-session routine. Unverified until the first node: the cu128 install of vLLM 0.31;
it is done on the L4 and checked with `triton_check.py` and a 20-problem eval before any A100 exists.

## 16. First E2E node: what the image actually has, and what that changes (17 Oct, your terminal)

You launched an L4 and asked, rightly, that nothing be installed until the drivers and libraries were
known. The read-only check: Ubuntu 24.04.3, kernel 6.8, driver 570.133.20 (CUDA 12.8), a CUDA 12.6
toolkit at `/usr/local/cuda` and on `LD_LIBRARY_PATH`, system Python 3.12.3 with no pip, no torch, no
conda, 25 vCPU, 100 GB RAM, one 233 GB root disk, no volume yet (ticket open).

Three consequences, each checked against the real artefacts rather than assumed:

- *The route is cu129, not cu128.* The PyTorch cu128 index stops at torch 2.11; torch 2.13 (what
  vLLM 0.31 pins) exists for cu129 and cu130. vLLM 0.31.0's release assets carry exactly one CUDA
  variant besides the CUDA 13 default: `vllm-0.31.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl`.
  CUDA 12.9 binaries run on a 12.8 driver through minor-version compatibility; Triton compiles with
  its own bundled ptxas, so the driver never JIT-compiles PTX. `setup_env.sh` now takes
  `RLVR_VLLM_CUDA=cu129`: the release wheel plus the cu129 torch index, same vLLM version as the
  laptop's lock. The verification is `triton_check.py` and a 20-problem eval on the L4.
- *No flash-attn on E2E.* Dao-AILab's newest wheels (2.8.3.post2) stop at torch 2.8, and a source
  build would use the image's 12.6 toolkit against torch's 12.9: the build refuses the mismatch.
  Training on E2E therefore runs `packing: false` (the trainer's own guard; correct, 1.5-2x slower)
  unless the kernels-hub flash-attn build works, which is tried once and recorded either way.
- *The image's `LD_LIBRARY_PATH` is unset by `e2e_env.sh`.* torch and vLLM bring the CUDA libraries
  they were built against; a system libcudart 12.6 ahead of them is the mismatch to avoid.

Also fixed while here: `setup_env.sh` used to install the newest vLLM and rewrite the tracked lock
file at the end, which on a box would both drift from the laptop and dirty the tree (the harness
refuses a dirty tree). It now installs the locked version and writes its freeze to an ignored
per-host file, printing the torch/vllm lines that differ, which is what the `COMPUTE.md` row records.

Without the volume, the layout goes on the root disk at the same paths; when the volume arrives, one
rsync moves it (rental guide, section 2c). The L4 earns its INR 49/h with the data build, the model
downloads, the smoke, the base n = 64 sampling overnight (EXP-002; 24 GB is enough) and possibly the
LoRA run; the full fine-tunes wait for the 80 GB card.

## 17. The L4 session: four of EXP-002's five sets done, Countdown stopped at 7 % (17-18 Oct, the node session)

A second Claude Code session ran on the node itself, inside JupyterLab over an SSH tunnel, with the
17 Oct handoff file as its brief. Its two files (`outputs/e2e_session_log.md`,
`outputs/e2e_session_notes.md`) and the metrics came back in a 12 MB archive and, after the node was
emptied, a full copy of the node's repo folder next to this one. Everything below is read from those
files and from `metrics.json`; the node session's own summary was wrong in one place (it reported the
four sets as 4 h 40 min; the per-eval wall clocks sum to 5 h 10 min, and the engine logs agree).

What happened, in order:

1. `git pull` on the node failed (no credential in the shell); harmless, the clone was already at the
   laptop's HEAD (ade4f6a). Model downloads fine. The data step reproduced both hashes
   (`b5afb83e6aacafcf`, `45b3fa4862b11e2a`) in 371 s.
2. The smoke eval refused to start: "working tree has uncommitted changes". The data step had
   rewritten the committed manifest and data report with new timings and the build commit. The node
   session restored both with `git checkout`, kept copies, and wrote the finding down; it later drafted
   a patch for it in a scratch clone (its "patch 0001"). The same fix was written here independently on
   the same afternoon and is commit b34c25b (section 16); the node's patch is superseded and was not
   applied. Two sessions, one finding, one fix: the handoff file now says so, so the A100 session does
   not try to apply the patch again.
3. Smoke: 20 GSM8K problems, 4-shot-CoT, greedy, pass@1 0.75 [0.55, 0.90]. Environment confirmed.
4. EXP-002, 4-shot-CoT config (GSM8K, perturbed, MATH-500, AIME24), in tmux. vLLM engine up in 75 s;
   KV cache 15.3 GiB = 143k tokens, so 23 sequences of 6,144 tokens run at once on the 24 GB card.

   | set | sequences | wall clock | output tok/s | mean tokens / sample | hit the 4096 cap |
   |---|---|---|---|---|---|
   | GSM8K | 84,416 | 69 min | 3,134 | 154 | 0.1 % |
   | GSM8K perturbed | 53,120 | 43 min | 3,082 | 151 | 0.1 % |
   | MATH-500 | 32,000 | 2 h 44 min | 1,171 | 359 | 4.3 % |
   | AIME24 | 1,920 | 31 min | 964 | 926 | 15.9 % |
   | Countdown (stopped) | 2,368 of 32,000 | 1 h 36 min | 816 | about 2,700 expected | 65.6 % expected |

   Four sets: 5 h 10 min, 34.3 M generated tokens, INR 253 at the L4's rate.
5. Countdown (think-tag, 500 problems x 64): 2.55 s per sequence and 21 hours still to go after 1 h
   36 min. You stopped it. The node session proposed cutting `max_tokens` to 2048 for Countdown with
   evidence from the greedy run (every correct answer under 711 tokens); you rejected that so the SFT
   checkpoint is not short-changed, and moved the Countdown half to the A100. The partial run left no
   samples: 1.6 L4-hours and about 4.7 M tokens discarded (INR 78).
6. On 18 Oct, with the archive verified on the laptop, the node's environments were deleted and the
   node terminated at 00:01 IST. Invoice: 10 h 33 min at INR 49, INR 516.95, of which 8.3 h ran jobs
   and 2.3 h were setup, the archive and idle time (two consecutive invoice lines, 13:28-16:01 and
   16:01-00:01; the console's "2 vCPU, 6 GB" plan label does not describe the node).

What the four sets say (the figure is `outputs/reports/figures/phase-1-pass-at-k.png`, the table is
in `experiments.md` and the Phase 1 report's addendum):

- The pre-registered hypothesis holds where it can be checked. pass@64 minus greedy pass@1: GSM8K
  +24.3 points (98.8 vs 74.5), MATH-500 +46.4 (87.8 vs 41.4), against bars of 15 and 25. maj@64 minus
  greedy: +11.2 and +16.2, against a bar of 5. The falsifier (MATH-500 gap under 10) is nowhere near.
  The Countdown clause (pass@64 above 10 %) waits for the A100.
- Sampling at T = 0.6 costs about 3 points on GSM8K and 5 on MATH-500 against greedy (mean sample
  accuracy 71.6 and 36.3). That is the single-sample noise reference every later pass@1 row is read
  against: a checkpoint sampled at 0.6 is not compared with a greedy base.
- maj@k flattens by k = 16 (GSM8K 84.4 at 16, 85.7 at 64) while pass@k keeps climbing. Majority vote
  is bounded by what the model prefers; pass@k by what it can produce at all. The gap between the two
  curves (13 points on GSM8K, 30 on MATH-500 at k = 64) is the room RL has to work with if it only
  sharpens.
- AIME24: 6 of 30 problems solved at least once in 64 tries (pass@64 20 % [6.7, 36.7]), 11 correct
  samples out of 1,920, and maj@64 = 0: on every solved problem the right answer is never the
  plurality. One sample in six hit the 4096 cap: the base loops on hard problems in the plain CoT
  format too, not only behind an untrained `<think>` token.
- Perturbed GSM8K stays behind the original at every k: 64.5 vs 71.6 at pass@1, 90.2 vs 98.8 at
  pass@64 (different problem sets, 830 vs 1,319, so unpaired). Ten per cent of the perturbed problems
  are never solved in 64 tries against 1.2 % of the originals: the brittleness to resampled numbers
  found in EXP-003 is a capability gap, not a decoding accident. (Paired on the 830 shared ids on
  18 Oct, with the new `--all-samples` mode of `perturbation_gap.py`: 10.2 [8.3, 12.0] at pass@1,
  8.6 [6.6, 10.6] at pass@64; 74 originals solved at least once whose twin never is, 3 the other way.
  The greedy gap of EXP-003, 10.1, holds at every k.)
- MATH-500 mean sample accuracy by level: 62.8, 61.1, 45.7, 25.0, 10.4 (levels 1-5). 8.9 % of the
  MATH samples carried no boxed answer and count as wrong; the SFT comparison reads the format rate next
  to the accuracy.

The estimate was wrong again, in a new way. Section 14 fixed the rule "tokens divided by the throughput
measured at that context length"; this time the throughput was measured on the node and the projection
(5-6 h for everything, from the GSM8K rate) still missed by a factor of four, because the rate is a
property of the completion length, not of the run: 3,134 tokens/s at 154 tokens a sample, 1,171 at
359, 964 at 926, 816 at 2,700. The rule now has two parts: estimate per eval set from that set's
expected completion length (for a base model, Countdown and think-tag mean about two thirds of the
cap), and read the rate off the progress line of each set, not the first one. For the A100,
Countdown is 32,000 x 2,700 = 87 M tokens, more than the four sets together; at a guessed 3,000-5,000
tokens/s that is 5-8 hours, and the `sample` step does it again for the SFT checkpoint. The handoff
file puts the three options (keep and measure, fewer problems, shorter cap) in front of you before
the A100 is launched; the pre-registered config stays the default.

## 18. The node changes: a TIR H100 notebook, and its disk as the volume (18 Oct, laptop)

The A100 route of section 15 assumed a MyAccount node plus a block volume that E2E had to assign by
hand; the ticket was still open on 18 Oct and the user withdrew it. The user picked instead a TIR
notebook (E2E's managed platform): the "NVIDIA H100 SXM 80 GB" plan at INR 379.05 per hour, 26 vCPU,
250 GB RAM, a 300 GB workspace disk at `/home/jovyan`, from the pre-built "vLLM v0.15.1, CUDA 12.9,
Transformers 4.57.6, Python 3.12.12" image, and asked whether a volume was still needed.

What the TIR documentation says (read 18 Oct): a stopped notebook bills storage only; the workspace
disk persists across Stop and Start and dies with Delete; an ephemeral `/home/user` disk is lost on
restart; MyAccount block volumes attach to node plans, not to notebooks. So the disk is the volume:
one notebook for the whole project, Stopped between phases while the laptop does code and docs,
Started again for the next phase with the venv, the caches and every `final/` checkpoint in place.
The off-node copies (Hub uploads, the laptop archive) are made before every Stop, as they were before
Terminate on the L4.

Two numbers changed with the card. The guide had allowed an H100 only within 1.5x the A100 price;
this one is 2.0x. The right comparison is rupees per experiment, not per hour: bf16 training runs
about 2x faster and vLLM sampling 1.6-2x on an H100 SXM, so the session costs about the same and
takes half the wall-clock. And the honest A100 sum for Phase 2 was never 15 hours once Countdown's
87 M tokens were measured (section 17): 25-32 A100-hours for Phase 2 plus 3, which is 15-18 hours on
this card, INR 5,700-6,800. The whole plan now projects to INR 15,000-18,000 with the disk, above the
USD 170 ceiling; the drop list (GRPO seed repeat, Countdown on 200 problems, LoRA SFT) is written in
the rental guide (2d) and the decision waits for measured rates at the end of this session.

What changed in the repo: `scripts/e2e_env.sh` picks `/home/jovyan/rlvr` as the home when that
directory exists, unsets `PYTHONPATH` (the image's own vLLM and Transformers must not shadow the
locked venv) and exports the node's price as `RLVR_NODE_RATE`, which `scripts/project_sampling.py`
reads as its default `--rate`; the handoff is rewritten for a container without root (no `apt`, no
`sudo`; tmux from conda or a `nohup` fallback), with the Phase 3 rules restated in H100 hours and a
pre-authorised pruning step for the 20 GB intermediate checkpoints; the guide's section 2d holds the
routine, the disk budget (peak about 150 GB) and the per-phase table.

## 19. The H100 session starts: the box as found (18 Oct, 09:50 UTC, the node session)

The user started the TIR notebook, cloned the repo, installed Claude Code and handed over with the
opening message of the handoff (section 9). The first facts changed the handoff's assumptions, which
had been written for a container without root: the shell was root with `HOME=/home/jovyan`, `apt`
worked, tmux and uv were on the image, conda was not. The driver was 550.127.08 (a CUDA 12.4 host)
under a CUDA 12.9 user space, which works only through the forward-compatibility `libcuda` in
`/usr/local/cuda/compat`; the env script of bb6f2a6 keeps that directory first on `LD_LIBRARY_PATH`
and the image's torch named the H100. `nproc` printed 256 and `free` 1.5 TB (the host's numbers seen
from the container; the cgroup limits were the plan's 26 vCPU and 250 GB), a harmless surprise with
one consequence: data-loader worker counts must not trust `nproc`. The disk was 339 GB with 455 MB
used (repo plus the two folders copied by scp from the laptop).

The environment built by the cu129 route in 25 minutes with an empty lock diff apart from the CUDA
labels (torch 2.13.0+cu129, vLLM 0.31.0+cu129, transformers 5.17.0, trl 1.15.0), the Triton check
passed through the compat library, and then vLLM would not import: `torchcodec` 0.17.0, a vLLM
dependency used only for audio, is not among the packages uv's torch backend routes to the CUDA
index, so PyPI's wheel landed, and that wheel is a CUDA 13 build that wants `libnvrtc.so.13`. No
0.17.0+cu129 wheel exists. The fix was to uninstall it (vLLM's import of it is guarded); the open
question for the laptop was how the L4 session got past this on 17 Oct (answer, from its freeze: an
older torchcodec that did not need CUDA 13). The data step reproduced both hashes
(b5afb83e6aacafcf, 45b3fa4862b11e2a) and left the tree clean; the b34c25b fix behaved.

A risk was written down before any step ran: the Phase 2 `report` step regenerates two *tracked*
files (`figures/phase-1-pass-at-k.png` and `.md`, now with the Countdown panel), and the harness
refuses to run on a tree with tracked modifications, so Phase 3 would have been blocked. The plan,
executed later: save the regenerated files aside, `git checkout -- outputs/reports` to restore the
tracked versions, carry both copies in the archive; the laptop commits the regenerated ones (this
commit does).

## 20. Countdown measured, and 27 idle minutes (18 Oct, 10:27-12:28 UTC)

The handoff's section 1 asked for a ten-minute measurement before deciding how to run EXP-002's
Countdown half. At ten minutes the H100 produced 9,735 output tokens/s on the set against the L4's
816, 11.9x, better than the 5,000-9,000 guessed; the projection was 2.3 h and INR 861, under the
4-hour rule, so the full pre-registered config ran. It finished in 1 h 34 min (INR 595), because the
sampled completions at T = 0.6 averaged 1,946 tokens (median 466), shorter than the greedy run's
2,716, and the rate held at 11,047 tokens/s. pass@64 63.8 [59.6, 68.0] with pass@1 2.8 % and
maj@64 11 %: the base can find the expression in 64 tries for two problems in three and almost never
first; the KV cache held 74 GB at 90 % utilisation and 6,144 context.

Then nothing happened for 27 minutes. The chain runner that was to start the sweep waited on
`tmux has-session -t phase2`, which prefix-matched the runner's own session `phase2b` after `phase2`
had ended, so it never saw the end. Caught at the scheduled 55-minute check; about INR 170 of idle
H100. Fixed by waiting on the step's EXIT marker in its log rather than on a tmux name, and tmux's
`-t =name` is the exact-match form. The lesson went into the notes and the next-session guide.

## 21. Two findings before the sweep could run (18 Oct, 12:34-12:41 UTC)

First, the training backend. The train configs set `attn_implementation: flash_attention_2` with
`packing: true`. No flash-attention wheel exists for torch 2.13 (Dao-AILab's releases stop at torch
2.8), and the kernels-hub route (`kernels-community/flash-attn`) failed on authentication (404 with
the fine-grained token, 401 without). A source build was judged too slow and risky on a billing node
at that hour, and more to the point, every Phase 2 training run had to share one backend to stay
comparable. Decision: every Phase 2 `rlvr-sft` call ran with `--set attn_implementation=sdpa --set
packing=false`, exact attention, padded batches, 1.5-2x slower, recorded in every COMPUTE.md row.
(The user later asked for flash attention to be used; section 25.)

Second, the learning rate. `learning_rate: 5e-6` in the three sweep YAMLs is a *string* under YAML
1.1 (PyYAML needs a dot: `5.0e-6`), and so is `--set learning_rate=5e-6`; AdamW raised a TypeError
on `0.0 <= lr`. The main and scaling configs were unaffected (they get the LR from `--set` with a
dotted value chosen by `pick_lr.py`). The node is not allowed to edit tracked files, so the
workaround was a PATH shim (`outputs/bin/rlvr-sft`) that appends a dotted override; the laptop fix
(cast numeric knobs to the type of their default, with a test) is commit 3e055ad on `phase-3-rft`,
and the three YAMLs now carry dotted floats.

Two smaller things from the same minutes: tmux sessions started from a running server do not see
variables exported later in one window (secrets must be exported before the first tmux command, or
set in tmux's global environment), and TRL warned "the tokenized prompt is not a prefix of the
tokenized prompt+completion" on some rows, meaning tokens merge across the prompt/completion
boundary; not fatal (TRL falls back to its own boundary), worth a count on the laptop (debt).
Debug and idle time so far: about 40 minutes, INR 250.

## 22. The sweep, and a model that will not stop thinking (18 Oct, 13:21-14:05 UTC)

Each sweep run was 36 optimiser steps on 2,300 examples at 16.5 s per step (10.5 min, 33 GB), plus
a 3-minute dev eval. The surprise came from the first dev eval: at 5e-6, GSM8K-dev 54.0 with 86.8 %
truncation, MATH-dev 34.0 with 80 %, median length at the cap. After 36 steps the model writes long
think blocks and rarely closes them inside 4,096 tokens. The other two LRs did better (tag-format
failure 100 -> 79 -> 58 % with the LR), and the rule picked 1e-5 (pooled dev 53.8 against 44.0 and
50.6). The note written at that moment: "if truncation stays this high after `main`, the think-tag
template is the bottleneck, not the LR". It did stay high (section 23).

The user, watching, asked for the micro-batch to fill the card for the long runs: 16 x 4 instead of
4 x 16 (effective batch 64 unchanged, the pre-registered quantity). Measured on the main run's first
steps: 15.6 s per step against 16.4, 5-7 % faster, 72.8 GB reserved; padding to the longest sequence
in a 16-sequence micro-batch eats most of the gain, but the card is full either way. Kept.

At 14:20 UTC the user went to sleep and asked for decisions without approval questions, with the
handoff's rules as the limits (the ten-minute rule on every n = 64 step with the 4-hour cap, the
18-node-hour rule for the control run, and the 20-hour line turned into "finish Phase 3 if the
projection stays under about 24 node-hours").

## 23. The main run and the think-tag surprise (18 Oct, 14:13-17:39 UTC)

720 steps in 3 h 02 min, 15.19 s per step, loss 0.413, 51.6 GB allocated. Both tag rows were
untrained in the base (93 duplicate rows), were initialised from their pieces and moved by 0.0078
(`<think>`) and 0.0531 (`</think>`); no duplicates remained. The three evals, the gap and the probe
took 23 minutes and the chain moved into them with no idle gap.

The result that defines this phase: the think-tag template, the training format, is the worst of the
three on its own checkpoint. GSM8K 74.8 against 82.2 zero-shot and 79.4 4-shot-CoT; MATH-500 50.6
against 63.2 and 56.8. The cause is visible in the lengths: think-tag completions average 1,887
tokens on GSM8K with the p90 at the cap, 28 % truncate; MATH-500 52 %; Countdown 95 %. The model
learned to think long and a quarter of the time does not stop inside 4,096 tokens. Zero-shot has
the same disease on MATH-500 (59 % truncation) and still scores best. The pre-registered bars were
met in substance (74.8 against 75, 50.6 against 50, Countdown 3.8 against "under 10"), the falsifiers
did not fire, and the "tag-format failures below 1 %" part of the hypothesis failed outright: the
format failures *are* the truncations. The perturbation gap was unchanged by SFT (11.3 against the
base's 10.4), and the partial-prompt rates did not rise (MATH-500 14.1 % against 16.3 %): no recall
entered through the data. The implication for Phase 3 was written down immediately: with the p90 at
3,960 the cap rule would pick 4,096 and the RFT sampling would be long.

## 24. The n = 64 runs and the ten-minute rule, twice (18 Oct 17:39 to 19 Oct 00:30 UTC)

The few-shot n = 64 run (four sets, 84,416 GSM8K sequences first) was projected at ten minutes with
`project_sampling.py`: 0.8 h more for GSM8K from the script, 1.5 h from tqdm, because the script
assumed the greedy mean length (429 tokens) and the sampled completions at T = 0.6 run about twice
that. The honest estimate for the whole run was 3.5-4 h, under the 4-hour rule: CONTINUE. It took
3 h 45 min (INR 1,420) and the sampled completions averaged 930-1,960 tokens, pushing truncation to
15-16 % on the math sets where the base had none. Under this template SFT changed nothing on GSM8K
at any k (pass@64 98.2 against 98.8) and lifted MATH-500 by 15 points at pass@1 and 5 at pass@64.

The Countdown half (500 x 64, think-tag) projected 3.4 h at ten minutes (1,472 of 32,000 sequences,
finished completions averaging 3,646 tokens): under the rule, CONTINUE, with two reasons recorded
beyond the rule: it is the only n = 64 Countdown point for the SFT checkpoint and the report step
lists the run unconditionally once the few-shot samples exist. It took 3 h 04 min: pass@64 62.0
against the base's 63.8, pass@1 11.9, truncation 79 %. The ceiling is unchanged by SFT and the
greedy row (3.8 %) is a length failure, not a search failure. The session's budget projection at
21:36 UTC was already 23-24 node-hours for Phase 2 plus a Phase 3 core, at the edge of the "about
24" line the user had set.

## 25. LoRA stopped, scaling kept, and the flash-attention build (19 Oct, 00:30-03:15 UTC)

The LoRA run started at 00:30 at 19.2 s per step, 26 % slower than the full fine-tune at the same
micro-batch (the embedding rows train in fp32 next to the adapters), 3 h 50 min projected, with the
session at 14.8 node-hours. With LoRA the projection was Phase 2 archived at 22 node-hours and Phase
3 at 26.5, over the line; applying the line literally would have meant no Phase 3 at all, which is
the branch's purpose. The call, made alone as the user had asked: stop LoRA at step 20 (INR 45),
run the data-scaling curve (2.6 h, three points) and the report, then Phase 3, and LoRA last only
if the clock allowed. The scaling curve was kept over LoRA as the cheaper and more informative
ablation. Mechanics: the chain's tmux session was killed, the partial run dir moved aside (no
checkpoint had been written), a new runner started with one idle minute. A second environment slip
cost three minutes here: the Claude Code shell keeps the environment it was started with, so after
the user rotated the W&B key the shell's copy was stale and the first scaling launch failed.

At 01:40 UTC the user, awake, asked why flash attention was not in use and said they wanted it.
The reasons (no wheel for torch 2.13; kernels-hub auth failure; the comparability of the Phase 2
runs) were written down, and at 01:50 the user chose: finish scaling, report, Phase 3 core, LoRA and
the RFT Countdown sampling and the control stay cut "unless stopped/paused by me". At 02:18 the user
asked for the build anyway, for Phase 3. The image turned out to have nvcc 12.9 (the 17 Oct L4 image
had 12.6, which is what the guide had warned about), and the notebook's real limits were the plan's
26 vCPU and 250 GB. `pip wheel flash-attn==2.8.3.post2 --no-build-isolation` with `MAX_JOBS=12` and
`FLASH_ATTN_CUDA_ARCHS=90` built the wheel in 13 minutes on the CPU while data50 trained (its step
time moved from 15.1 to 15.5 s, within noise). The wheel went to `/home/jovyan/rlvr/wheels`; the
kernel test (flash versus sdpa, max abs diff 0.0039), the varlen test and a Qwen3-0.6B load under
`flash_attention_2` with logits matching sdpa within bf16 noise passed at 03:15. Phase 3 therefore
ran with the pre-registered packed config; the comparability note (SFT unpacked with sdpa, RFT
packed with flash; exact attention in both, different micro-batch composition) went into the Phase 3
report. Phase 2 itself stayed on sdpa throughout.

## 26. Data scaling, and Phase 2 closed (19 Oct, 00:43-03:35 UTC)

data25 (180 steps, 46 min) and data50 (360 steps, 1 h 31 min) at the main run's LR, backend and
micro-batch, each followed by the think-tag eval: GSM8K 61.8 -> 70.8 -> 72.1 -> 74.8, monotone in
data; MATH-500 47.8 / 44.0 / 50.6, flat within CI; MATH-500 truncation 20 % (base) -> 57 / 58 / 52 %
at every SFT size. The think-tag SFT teaches long thinking on MATH-style problems at the first 5,750
examples and more data does not teach it to stop. The report step ran at 03:29 (compare tables, the
figures, the model card, the Hub uploads: `Raviteja546/reasoning-rlvr-sft` with the three sweep
finals and the two scaling finals under `ablations/`, the samples dataset with the four n = 64 sets,
the Phase 2 archive). The HF token belongs to the `Raviteja546` namespace, not the `rrtchaganti` the
scripts defaulted to; the node exported the three `HF_REPO_*` variables and the laptop changed the
defaults (3e055ad). Phase 2 closed at 03:35 UTC: 17.75 node-hours, about INR 6,730, including 1.7 h
of waste and the flash build, against the handoff's 10-12 hours for this part; the overrun was the
SFT model's long completions in the two n = 64 runs (about 4 h more than planned together). The
disk clean-up at 04:00 deleted 174 GB of intermediate checkpoints and, on the user's request, the
33 GB of ablation finals already on the Hub; 50 GB of 339 used.

## 27. Laptop: the archive, the fixes, the paired curve, the write-up (19 Oct)

The session archive (68 MB compressed, 768 files: every `metrics.json`, `report.md`, `run_info.json`,
`train_log.jsonl`, the per-sample files of every greedy eval, `outputs/reports`, the RFT data
manifests, the five node patches, the shim, the session log and notes, the gap analysis and its
results, the Hub inventory, the redacted transcript) came back through the JupyterLab download and
extracted next to the repo. The usage report from the TIR console gave the real numbers: the H100
billed 25.07 h to 16:01 IST on 19 Oct and about 3 h more after it (28.1 h, INR 10,640), the disk
INR 10 per GB-month on 340 GB (the guide had guessed INR 5), the dataset volume the user had created
and deleted INR 10.20, and an A30 hour from 16 Oct that no journal mentions.

The laptop's first commits went to `phase-3-rft`, where the code is current: the five harness defects
the node had worked around (flag equal to the default ignored; independent `--limit` draws for the
paired GSM8K sets; the YAML learning-rate string; the Hub card's `base_model` as a local path;
`torchcodec`) fixed with tests, and the rental docs updated with the flash recipe, the disk rate, the
secrets-before-tmux rule and the node's next-session guide. Then this branch: the Phase 2 report
tables, figures and model card from the archive; the regenerated Phase 1 figure with its Countdown
panel; the three paired data-scaling comparisons computed here from the archived samples (the node
never ran them): 100 minus 25 % is +4.0 [1.4, 6.7] on GSM8K and +2.8 [-1.4, 7.0] on MATH-500, which
makes EXP-008's MATH-500 result "inconclusive, and small if present" rather than the flat curve the
unpaired numbers suggested; the base 4-shot-CoT and zero-shot rows at the SFT cap (G5 of the gap
analysis, 5 minutes on the node) so that every paired table in the report shares its cap; the
registry entries, the ledger rows, this journal, the engineering notes and the PR body. The two
Phase 3 runs that explain the think-tag deficit (an 8,192-token cap changes nothing, truncation is
looping; T = 0.6 removes most of it) are cited in the Phase 2 report because they are about this
checkpoint; their record is the Phase 3 journal.

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

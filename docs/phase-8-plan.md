# Phase 8 plan: a process reward model on the same samples (addendum)

Written 3 Nov 2026 on branch `phase-8-prm`, after the project closed at Phase 7. **Status: plan for the
user's edits; nothing is downloaded, built or run.** Laptop only, INR 0, no training.

## 1. Why

EXP-022 trained an outcome reward model (ORM: one score per solution) and found it useful at k = 2
and over-optimised as an arg-max at k = 32 (best-of-32 12-13 points below majority-of-32 on
MATH-500). Current maths work compares that with a process reward model (PRM: one score per step),
whose labels are now produced automatically (Math-Shepherd's completion-based labels, OmegaPRM's
tree search, FoVer's formal verifiers) and whose known weaknesses are documented (Qwen's "Lessons"
paper, arXiv 2501.07301: completion-based step labels are noisy, and PRMs tuned only for best-of-N
drift towards ORMs). Two questions follow, both answerable from data already on the laptop:

- **EXP-025 (option 1).** On the same 57,600 samples, does an open PRM pick better than our ORM and
  than majority vote, especially at large k where the ORM collapsed?
- **EXP-026 (option 2).** Phase 6 located where each loop starts. Does the PRM's step score fall at
  the loop onset, so that a step-level verifier "sees" the project's central failure?

Option 3 (training our own PRM from completion-based labels) is not in this plan; its decision rule
is in section 6.

## 2. What was verified before writing this (3 Nov 2026)

| item | checked how | result |
|---|---|---|
| The samples | counted the six `samples.jsonl` files (`../hub_samples/{sft,rft}_1.7b_n32_think_tag_t06`, `../h100_p4_session2_archive/outputs/samples/grpo_1.7b_n32_think_tag_t06`; GSM8K and MATH-500) | 9,600 each = 300 problems x 32; full completion text in `text`; `correct`, `finish_reason`, `num_tokens` per sample |
| Same problems across checkpoints | compared the problem-id sets | identical for `sft`, `rft`, `grpo` on both sets |
| The questions | `rlvr.eval.datasets.load_eval_set` offline (HF cache) | every one of the 300 ids per set found; the samples do not store the question, the loader supplies it |
| The ORM's scores | `scores.jsonl` beside each sample file | 9,600 lines each (EXP-022); same ids |
| The ORM's input | `src/rlvr/train/orm.py` | inputs longer than 2,048 tokens kept their *tail* (the answer end) |
| The PRM | Hugging Face model card, `config.json`, file listing of `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B` | a Qwen2.5-Math-1.5B-Instruct derivative with a one-unit linear value head on the last hidden state; per-step reward = sigmoid of the value at each step's newline token; steps separated by `"\n"`; question formatted as BOS + problem + `"\n"`, no chat template; the card scores a solution by the mean of its step rewards for best-of-N |
| PRM context length | `config.json` | `max_position_embeddings` 4,096, no RoPE scaling |
| PRM weights | file listing | `pytorch_model.bin`, 3.09 GB, a PyTorch pickle (no safetensors); custom `modeling_qwen2_rm.py` (72 KB); `config.json` names `Qwen2ForPrmModel` while its `auto_map` names `Qwen2ForRewardModel` (inconsistent) |
| PRM inference code | `github.com/SkyworkAI/skywork-o1-prm-inference` (Apache-2.0), read `model_utils/prm_model.py` and `io_utils.py` | a value-head wrapper over `AutoModelForCausalLM`; no truncation or length handling; the vLLM route pins vLLM 0.6.4.post1 (ours is 0.31.0) |
| PRM licence | model card and files | "Skywork Community License" (PDF in the repo); the card says commercial use is supported under its terms; research use here is within them, **to be read by the user** |
| How many samples fit the window | completion tokens + a question allowance against 4,096 (completions were counted with the Qwen3 tokenizer) | GSM8K: 86.5 / 94.3 / 98.5 % of samples fit whole (`sft` / `rft` / `grpo`); MATH-500: 58.3 / 69.7 / 86.4 %. The rest are mostly truncated (looping) samples; on MATH-500, 82 / 47 / 8 problems have fewer than 8 of 32 samples that fit |
| Loops in these samples | `finish_reason == "length"` counts | GSM8K 1,245 / 519 / 139 truncated; MATH-500 3,784 / 2,694 / 1,197 |
| Loop onset per completion | `scripts/analyse_traces.py` | `first_repeat(words, ngram)` gives the onset word per completion; importable, so the onset maps onto a character offset and a PRM step |
| Reusing the analysis code | `scripts/plot_accuracy_vs_tokens.py` and `scripts/evidence.py` | both read the score file name `scores.jsonl` hard-coded; a PRM score file needs a name option |

## 3. Assumptions still open (each checked in step 0, before any scoring)

1. **The weights load under our stack.** Plan: load `pytorch_model.bin` with `torch.load(weights_only=True)`,
   never `trust_remote_code`; build the base with `AutoModelForCausalLM` (Qwen2 is native in
   transformers 5.17) and our own `nn.Linear(hidden, 1)` head; list the state dict's non-layer keys to
   find the head's weight names. Check: the card's demo example reproduces its per-step rewards with our
   loader to 1e-3. If not, stop and report.
2. **Tokenizer counts.** The fit figures above use Qwen3 token counts; the PRM uses Qwen2.5's tokenizer.
   Check: recount a 500-sample slice with the PRM tokenizer.
3. **What a step is in our text.** The card splits on single newlines. Our completions are R1-style
   monologues whose LaTeX display blocks span several lines, so a single-newline split cuts formulas
   into fragments (median 14-42 blank-line paragraphs per completion; more single lines). Primary rule:
   **one step = one paragraph** (blank-line separated), inner newlines replaced by spaces, steps joined
   with `"\n"`. Sensitivity: the raw single-newline split on a 1,000-sample slice.
4. **The think-tag wrapper.** The PRM never saw `</think>` or `<answer>` tags. Primary: drop the tag
   strings and keep their content (the boxed answer usually sits in the summary before `</think>`, and
   the answer block's content follows). Sensitivity on the same slice: tags kept.
5. **The window.** A sample longer than the window keeps the question and the *head* of its response
   (the first steps that fit, about 3,800 tokens), and records `steps_scored`, `steps_total` and
   `complete`. For a truncated sample the head holds the derivation and the start of the loop (onset
   near word 500 / 800 in Phase 6), so the PRM still sees the loop begin. All samples stay in every
   draw, as in EXP-022; results are also reported on the samples that fit whole.
6. **The PRM transfers to our traces.** It was trained on another model's concise solutions. Check
   first, as EXP-022 did: solution-level AUROC against `correct` on every set. If the AUROC is under
   0.70 on a set, that set's best-of-k results are reported as uninformative, not as a finding.
7. **Time.** Unmeasured: the ORM (0.6B, inputs capped at 2,048) took 38-55 minutes per set of 9,600;
   the PRM is 2.5x the parameters at up to twice the input, so a guess is 2.5-5 hours per set, 15-30
   hours for all six. Step 0 measures it on 200 samples and projects.

## 4. Deliverables

| id | what | done when |
|---|---|---|
| P0 | step 0: download (3.1 GB to the HF cache), load check, tokenizer recount, step-split and tag sensitivity on a 1,000-sample slice, throughput on 200 samples | a short report in `outputs/reports/phase-8-step0.md`; any failed check stops the phase |
| P1 | `src/rlvr/eval/prm.py` (or `scripts/prm_score.py`): the loader, the step builder (rules 3-5), batched scoring with right padding, one line per sample `{problem_id, sample_idx, step_rewards, mean, min, last, steps_scored, steps_total, complete}` into `prm_scores.jsonl` beside `samples.jsonl` | tests on a tiny model: the step builder, the window rule, reward positions, the weights-only load |
| P2 | scoring of the six sets, resumable, detached in WSL | six `prm_scores.jsonl` files, 9,600 lines each |
| P3 | `--scores-name` in `scripts/plot_accuracy_vs_tokens.py` and a score-file parameter in the evidence rule `best_of_k` | tests; EXP-022's numbers re-derive unchanged |
| P4 | EXP-025 analysis: AUROC per set (PRM mean, min, last; ORM), best-of-k and score-weighted vote for k = 1-32 for PRM and ORM against majority vote, paired on the same seeded draws; all samples and fitting-only | a table and a figure in `outputs/reports/`, claims in the evidence index |
| P5 | EXP-026 analysis on truncated samples with a loop whose onset is inside the scored window: per sample, mean step reward over the 5 steps before the onset step and the 5 after, paired difference with a bootstrap CI; the same positions on finished correct samples as the control; per set and checkpoint | a table and one figure of step reward against step position relative to onset |
| P6 | write-up: report section, registry closes, narrative, journal, notes, README lines, COMPUTE rows | as every phase (`WORKFLOW.md`), `bash scripts/ci_local.sh` before the PR body |

## 5. Pre-registration drafts (go into `experiments.md` and the narrative after the user's approval)

**EXP-025 hypothesis.** The PRM's solution-level AUROC is at or above 0.80 on every set. With the mean of
step rewards as the score, PRM best-of-k is at or above majority vote at every k on GSM8K and at k up
to 8 on MATH-500, and at k = 32 on MATH-500 it falls less below majority-of-32 than the ORM did
(ORM: -11.7 to -13.3). The PRM-weighted vote is at or above majority vote at every k.
**Falsifier.** PRM best-of-32 on MATH-500 as far or further below majority-of-32 as the ORM's (within the
paired CI): step-level scoring does not cure arg-max over-optimisation at this scale.

**EXP-026 hypothesis.** On truncated samples, the mean step reward over the 5 steps after the loop
onset is lower than over the 5 before, by at least 0.10 (paired, CI excluding 0), while finished
correct samples show no such drop at the matched positions.
**Falsifier.** No drop, or the same drop on the controls: the PRM does not see the loop as an error,
and a PRM is not a loop detector here.

## 6. The decision rule for option 3 (written now, read after EXP-025)

- **Option 3 earns its cost** (our own PRM from completion-based step labels, same 0.6B base and the
  same traces as our ORM) if the off-the-shelf PRM's best-of-32 on MATH-500 is above our ORM's by more
  than the paired CI and within 3 points of majority-of-32 or above it. Then step-level scoring
  helps and the confounds (1.5B against 0.6B, foreign against in-distribution training) need removing.
- **Option 3 adds little** if the PRM is level with the ORM or also collapses, or its AUROC is under
  0.70: the write-up says so and the project ends there.

## 7. Order and time

Step 0 first (about 1-2 hours including the download); the user reads its report before P1-P2.
Then P1 and P3 (code and tests, laptop CPU), P2 detached overnight (15-30 hours of GPU, measured in
step 0; if the projection exceeds 20 hours, MATH-500 for all three checkpoints goes first, GSM8K
after), P4 and P5 (CPU), P6. No tracked file is edited while P2 runs.

## 8. Decisions for the user

1. **The licence.** Read the Skywork Community License (`misc/misc_Skywork Community License.pdf` in the
   model repo) and confirm research use here is fine.
2. **The step rule and tag handling** of section 3 (paragraph steps, tags dropped) as primary, with the
   single-newline split and kept tags as sensitivities. Recommended as written.
3. **GPU time.** Up to 30 laptop hours for all six sets, or MATH-500 only (the cell where the ORM
   collapsed, about half). Recommended: all six if step 0 projects under 20 hours, else MATH-500 first.
4. **Branch and naming.** This is an addendum after the project closed; it is numbered Phase 8 and
   tagged `phase-8` after its merge. Recommended as written.

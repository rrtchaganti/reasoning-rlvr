# Phase 7 plan: the write-up and the interview packet

Written 24 Oct 2026 on branch `phase-7-write-up`, after Phases 5B and 6 merged (PR #8, tags
`phase-5b`, `phase-6`). This is `PLAN.md`'s Phase 8, renumbered (`docs/plan-after-phase-4.md`
section 3). No rental, no new training. The one GPU job is the reproduction check (W8), about 1.2
laptop hours. **Status (24 Oct 2026): the user took the recommendations of section 5: a slim README,
the repo public after the merge (the user's action), a fresh environment for EXP-024, the model
cards extended (pushed with the user's token at the end); the blog post and the short post are
paused.**

## 1. Goal and exit

The reader is a research engineer or hiring manager on a post-training team. In ten minutes they
should know the question, the answer with its numbers, and why to trust them. In the hour after,
every claim should lead them to the problems and completions behind it. (`PLAN.md` section 0 lists
the six things such a loop probes; each deliverable below names the one it serves.)

Exit, from `PLAN.md` Phase 8: every deliverable of `PLAN.md` section 1 is checked off or marked
dropped with its reason, and someone else can clone the repo, run one script and reproduce the
base-model baseline table.

## 2. Deliverables

| id | deliverable | what it holds | done when |
|---|---|---|---|
| W1 | README final pass | the layout in section 3; the opening still says "21 Oct 2026, four phases run" | every number in it is in the evidence index or a committed table; under about 220 lines |
| W2 | the one chart (D6) | accuracy against generated tokens on the pinned subsets: each checkpoint's greedy point, the forced-stop curves of `sft`, `rft`, `grpo`, maj@k, best-of-k and the weighted vote with the reward model; marker shape by machine (H100 or laptop), because pairs only hold within one | `scripts/plot_final_chart.py` with a test; the figure and its table in `outputs/reports/figures/` |
| W3 | the two-page interview packet | `docs/interview-packet.md`: the study in six lines; five results with CIs; the falsifiers that fired and what each taught; one line per paper (agree or disagree, with our number); cost and tradeoffs; what 10x compute would buy and what it would not | fits two printed pages; every number links to a claim id |
| W4 | paper-defence notes | `docs/paper-notes.md`: GRPO/DeepSeekMath, DAPO, Dr. GRPO, Yue et al., s1, then R1, STaR and "Spurious rewards" for context; per paper five short paragraphs (contribution, method, evidence, limits, what we measured) | each paper's claims checked against the paper itself (arXiv ids cited), not from memory |
| W5 | drills | `docs/drills.md`: `PLAN.md` 10.2 mapped to the from-scratch code and tests in `src/rlvr/ref` and `tests/`; the drills with no code here listed as such | a table, one row per drill: file, test, status |
| W6 | derivations tied to plots | `docs/derivations.md`: `PLAN.md` 10.3, each item with the figure from this project that shows it, or "no plot here" where the project did not measure it (for example asymmetric clipping, which was not ablated) | no derivation points at a plot that does not exist |
| W7 | review of the registry, ledger and incident log | `experiments.md` (every status final, an index at the top), `COMPUTE.md` (final totals: H100 hours, INR, laptop GPU hours), `FAILURES_AND_LEARNINGS.md` (an index at the top), `METHODS_AND_METRICS.md` and `GLOSSARY.md` read against the 22-24 Oct corrections (the user's edit of line 18 lands here) | the totals add up from the rows; no sentence contradicts a later correction |
| W8 | the reproduction check (EXP-024; done 24 Oct 2026: 11,216 of 11,216 completions identical) | `scripts/reproduce_baselines.sh`: fresh clone, `setup_env.sh`, the 1.7B EXP-001 configs, `collect_baselines.py`, then `scripts/check_reproduction.py`, which pairs the new per-sample files with the archived ones (laptop archive on the Hub) per problem: same completion text, same grade | the script runs end to end; the result is written up as EXP-024, whatever it says |
| W9 | `PLAN.md` deliverables table | D1-D9 with their state: D4 three of four checkpoints and D7 dropped, each with the reason (`docs/plan-after-phase-4.md` section 4) | one table in the README's status section |
| W10 | resume entry | `../resume_reasoning_rlvr.md` (outside the repo): the same-machine numbers, the machine term (10-23 % of problems flip across GPUs, net -1.8 to +1.8), Phases 5B and 6 | every number matches the README |
| W11 | Phase 7 journal, engineering notes, PR, tag `phase-7` | as for every phase (`WORKFLOW.md`), with `bash scripts/ci_local.sh` before the PR body | merged and tagged |

Optional, the user's call (section 5): a blog post, a short post with the chart, the model cards
extended with the Phase 5 results.

## 3. The README, proposed layout

1. **Title and the question** (three sentences): what each stage of the recipe buys at 1.7B, and
   whether it is reasoning, sharpening or termination.
2. **The answer in five findings**, one sentence each with its number and CI and a claim id:
   (a) the on-policy gains sit on problems the previous policy did not finish; on problems both
   finished, accuracy does not move; (b) a decoding rule recovers RFT's gain on SFT and, on one
   machine, puts RFT and GRPO level on MATH-500 when both stop at the same budget; (c) voting, or
   a small verifier at k = 2, reaches the GRPO policy's greedy number; (d) the loops are one step
   repeated from a fixed point, and training changed their frequency, not their shape; (e) greedy
   decoding is reproducible per machine, not across GPUs, and one result flipped once that was
   controlled.
3. **The one table**: today's stage decomposition, plus three rows: the stop rule on SFT (same
   machine), RFT and GRPO both stopped at 2,048, and the base model's best prompt.
4. **The one chart** (W2).
5. **What failed**: the falsifiers that fired and the predictions that were wrong, one line each,
   linked to the registry. Kept near the top on purpose.
6. **What every number carries**: the rigour list, condensed, with the machine term and the
   evidence index.
7. **What comes next**: EXP-018 (GRPO on harder prompts) as the experiment that could change the
   conclusion; what 10x compute would buy.
8. **Reproduce** (W8) and the repository map.

The phase-by-phase story and the full result tables move to `docs/results.md`; the research
narrative stays as is and is linked.

## 4. EXP-024, the reproduction check (draft pre-registration)

Under the 21 Oct rule, this goes into `experiments.md` and `docs/research-narrative.md` before the
run.

- **Known.** EXP-000: greedy decoding with the batch-invariant harness is bit-identical on one
  machine. EXP-020b: it is not across GPUs. The Phase 1 base runs were made on this laptop at
  ffd3018 with the same package versions as today's lock (vllm 0.31.0, torch 2.13.0, transformers
  5.17.0). `src/rlvr/eval/extraction.py` is unchanged since ffd3018; `report.py` and `metrics.py`
  gained 59 lines, additions only (forced-stop and Wait fields, paired pass@k); the 22 Oct
  correction changed wording, not code.
- **Question.** Does a fresh clone, set up by `setup_env.sh` and run on the same laptop, reproduce
  the 1.7B base-model table?
- **Prediction.** Every completion text is identical to the archived one (0 flips in decoding),
  every grade is identical, and every cell of the table reproduces to the printed digit.
- **Falsifier.** Any completion text differs, on the same machine with the same versions: then
  determinism depends on something the run record does not capture, and every same-machine claim
  needs a second look. Identical texts with a different grade would mean the grading path changed
  through a dependency (math-verify), and the diff names the problems.
- **Cost.** About 1.2 laptop GPU-hours (the eight EXP-001 configs took 1.19 h on 14 Oct; the 1.7B
  half is about 0.8 h), plus the environment build.

## 5. Decisions for the user

1. **README length.** A slim README (about 200 lines) with the long sections moved to
   `docs/results.md`, or keep today's 305 lines and rewrite in place. Recommended: slim.
2. **Make the GitHub repo public at the end?** The exit says "someone else can clone the repo";
   while it is private, the README and the packet can only be shared as files. The Hub repos are
   already public. Recommended: public after the Phase 7 merge, once you have read it as a
   stranger would.
3. **Reproduction environment.** A fresh `setup_env.sh` into a new venv (about 8 GB of downloads,
   the honest test of the "one script" claim), or the fresh clone with today's venv (minutes, tests
   only the code path). Recommended: fresh venv.
4. **Blog post and a short post with the chart** (`PLAN.md` Phase 8 lists both). Yes or no, and
   where.
5. **Model cards.** Extend the three cards with the Phase 5 results (the stop rule, the machine
   term). This needs a write token once more.

## 6. Order and time

About two working days. W7 and W9 first (they fix the facts the rest quotes), then W8 (it runs on
the GPU while the writing goes on), then W2, W1, W3, W4, W5, W6, then W10 and W11. Each document
goes to the user for reading before the next depends on it.

## 7. Not in this phase

No new experiment beyond EXP-024. EXP-018, EXP-010i and EXP-014c stay pre-registered and unrun.
DPO, distillation and serving stay dropped (`docs/plan-after-phase-4.md` section 4).

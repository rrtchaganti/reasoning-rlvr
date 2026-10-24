# Phase 7 engineering notes: the write-up and the reproduction check

Results: `README.md`, `docs/results.md`; the chronological story: `docs/phase-7-journal.md`; the plan:
`docs/phase-7-plan.md`.

## 1. What was built

- **`scripts/reproduce_baselines.sh`**: the "one script" of the project's exit. Clones the repo at a
  revision, builds a fresh venv with `setup_env.sh` (the locked vLLM, torch and transformers),
  downloads the laptop archive from the Hub and checks its SHA-256, regenerates the perturbed GSM8K
  set and stops if its tracked manifest changes, reruns the four Qwen3-1.7B-Base greedy configs,
  rebuilds the table and pairs every completion with the archived one. Each step is skipped when its
  output exists, so a dropped run resumes with the same command.
- **`scripts/check_reproduction.py`**: pairs two eval runs per (problem, sample) and counts identical
  texts, extracted answers and grades, with the character where the first differing texts part.
  Tests on synthetic runs: identical runs pass, a changed text or grade is named, a missing run fails
  under `--strict`.
- **`scripts/plot_final_chart.py`**: the D6 chart from the EXP-022 analysis JSON (sampled curves) and
  the laptop runs restricted to the pinned subset ids, with a companion table carrying the CIs.
  Tests: subset restriction (a problem outside the subset is ignored, one missing from a run is an
  error), the table, the PNG.
- **Documents**: the slim README, `docs/results.md`, `docs/interview-packet.md`,
  `docs/paper-notes.md`, `docs/drills.md`, `docs/derivations.md`; indexes for the registry and the
  incident log; header rows for the ledger's tables; the card sections on the stop rule.

## 2. Design decisions

- **The reproduction compares completions, not only the table.** A table can reproduce to the digit
  while individual completions differ (flips cancel); a per-problem comparison of texts is the test
  of determinism, and the table is the test of the claim a reader sees.
- **The reference comes from the Hub, hash-checked**, not from the laptop's own files, so the script
  is the one a stranger runs; the clone comes from the local repo only while GitHub is private.
- **Same machine first.** EXP-020b showed decoding is reproducible per machine, not across GPUs, so the
  pre-registered prediction is bit-identity on the laptop; on another GPU the README tells the reader
  to compare CIs, not digits.
- **One chart, three encodings.** Colour for the checkpoint, line style for the method, marker for the
  machine, because a reader must not pair a laptop point with an H100 curve; the verifier's weighted
  vote only for RFT, to keep each panel readable (the verifier's own figure stays in Phase 5).
- **The README cites claim ids**, so every headline number leads to the problems behind it in one hop.
- **The paper facts were gathered by a separate agent told to mark what it could not verify**, and
  each note separates the paper's claim from what this project measured.

**The result.** EXP-024 held in full: 11,216 of 11,216 completions identical in text, extracted
answer and grade, the table identical to the digit (`outputs/reports/reproduction-check.md`).

## 3. Known limitations

- The reproduction ran on the machine that produced the reference; it does not test another GPU (the
  machine term is measured separately, EXP-020b) and it shares the Hugging Face cache (weights and
  datasets) with the existing environment.
- The palette was not run through the validator (Node.js is not installed on the laptop).
- The model cards' new sections are hand-written; `scripts/push_to_hub.py` does not generate them, so a
  regenerated card would drop them.
- The finished-only-reward control and EXP-018 remain unrun.

## 4. Interview question bank for this phase

*How do you know your results reproduce?* A fresh clone and a fresh environment on the same machine
regenerate the base-model table; every completion is compared with the archived one by text and
grade (EXP-024). Across GPUs they do not reproduce bit for bit, and we measured by how much.

*Why does your README cite claim ids?* Each id is a YAML entry with the runs and a selection rule;
a script re-derives the quoted number from the per-sample files and lists the problems. A sentence
whose number disagrees with its files is a bug, and the index shows it.

*What would you have done differently?* Pair within one machine from the first laptop run (the
machine term cost two reversals); read CI after the first push of each phase; check a control's data
against the treatment's by id before training.

*Why did clip-higher do nothing?* Because with one optimiser step per batch of completions the
importance ratio is exactly 1 when the loss is computed, so no ratio ever reaches a clip bound; the
clip fraction was 0 throughout. Clip-higher needs several updates per batch to matter.

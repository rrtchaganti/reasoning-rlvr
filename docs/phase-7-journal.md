# Phase 7 journal: the write-up, the packet, and one check of the whole (chronological)

Context first, then what happened in order. Plan: `docs/phase-7-plan.md`. Pre-registration:
`experiments.md` EXP-024.

## 1. Context (24 Oct 2026, after PR #8)

Phases 5B and 6 merged as PR #8 (ac607e2, tags `phase-5b`, `phase-6`). Before Phase 7 started, the
laptop-only files went to the Hub: the archive of every laptop run (142 MB, sha256 8cbe1bdd...06f1),
the verifier's scores next to the sample sets, a dataset card and the three model cards with the 22
Oct correction. The first push failed with a 403: the laptop's cached Hugging Face login is a
read-only token from another project; the user ran the staged script with a write token, and every
file was checked against its local copy by hash. The worktree, the WSL clone and the scratch branch
were removed. Branch `phase-7-write-up` from `main`.

The plan (`docs/phase-7-plan.md`, 200436a) listed W1-W11 and five decisions. The user took the
recommendations: a slim README, the repo public after the merge, a fresh environment for the
reproduction check, the cards extended; the blog post paused ("go with your recommendations, pause
blog post work for now").

## 2. What happened, in order

- **The plan's own error.** The draft said the grading code had changed since the Phase 1 commit;
  `git log` showed `extraction.py` unchanged and `report.py`, `metrics.py` with additions only. The
  pre-registration was corrected before it was committed: the prediction became "identical texts,
  identical grades, identical digits".
- **EXP-024 pre-registered and launched** (4548c00): `scripts/reproduce_baselines.sh` (clone, fresh
  venv via `setup_env.sh`, the archived reference from the Hub with its hash checked, the four 1.7B
  base configs, the table, the per-problem check) and `scripts/check_reproduction.py` (tested). The
  clone came from the local repo because GitHub is still private. Two checks passed on the way: the
  fresh environment differed from the lock in nothing outside the CUDA libraries, and regenerating
  the perturbed GSM8K set left its tracked manifest byte-identical (the script stops if it does not).
- **The review (W7),** 5725e11. The registry got an index of all 36 entries and an entry for EXP-011
  (dropped in Phase 3 and never registered). The incident log got an index and four incidents of
  18-20 Oct that lived only in journals and notes (the YAML float, the flag-equals-default harness bug
  behind EXP-009's 2,048 cap, the controls trained on seen data, the GRPO out-of-memory runs). Every
  table block of `COMPUTE.md` after the first had no header row and rendered as text on GitHub; each
  got one, and a sentence that inserted rows had split in two was rejoined. The user's edit of
  `METHODS_AND_METRICS.md` (extraction failures are scored wrong, with the measured 20.8 %) was
  committed with a full stop added, and `GLOSSARY.md` got the same correction.
- **The chart (W2),** 46b8efe. One figure, two panels, every method on the pinned 300-problem
  subsets: colour the checkpoint, line style the method, marker the machine. The palette validator
  needs Node.js, which is not installed; the reference palette's first three slots are documented as
  validated and identity is also carried by labels and the legend. Two renders were fixed by eye: the
  log-2 ticks became token counts, and the "grpo, no stop" label, which sat on its own line, moved to
  empty space with a leader line.
- **The README (W1),** 8754da1: 305 lines to 201, five findings each citing its evidence claim, one
  table (the decomposition plus the stop-rule rows), the chart, what failed, the deliverables. The
  long sections moved to `docs/results.md`, which also answers the plan's ten pre-registered
  questions with their numbers (none had been answered in one place). Thirty-one claim locations
  pointed at old README section numbers and were re-pointed; the evidence index was rebuilt with 0
  disagreements.
- **Drills and derivations (W5, W6),** 571f88f. Writing item 4 found a mechanism for an old null
  result: clip-higher can only act when the probability ratio leaves the clip range, and with one
  optimiser step per batch the ratio was 1 and the clip fraction 0 throughout, so EXP-014b's
  clip-higher could not have done anything. Two sentences were corrected before the commit: the
  masking bug of 15 Oct was caught by the reference test (nothing leaked into training), and the
  group-mean baseline scales the expected gradient by (G - 1) / G rather than biasing its direction.
- **Cards,** f384223: each gets the stop rule as measured for that checkpoint and the machine term.
- **Paper notes and packet (W3, W4),** 4a3ef98. A research agent read the eight papers on arXiv and
  marked what it could not verify; the notes tie each paper to our numbers. They surfaced the
  cheapest missing control: if the on-policy gain is termination, a GRPO run rewarded for finishing
  alone would show whether the correctness term taught anything. It went into the README's next
  steps as proposed, not pre-registered.
- **EXP-024's result,** 05:59 UTC (WSL clock). Every one of the 11,216 completions in the 17
  run-and-set cells is identical to the archived one in text, extracted answer and grade, and the
  regenerated table's 29 rows equal the committed ones: on the machine that made them, the numbers
  are a function of the commit and the lock file. Evals 0.82 GPU-hours; the reports went into
  `outputs/reports/` and the claim P7-01 points at them. The ledger got its final totals: about INR
  21,750 rented, about 42 laptop GPU-hours.
- **The resume** (`../resume_reasoning_rlvr.md`, outside the repo) is being updated by the user's
  forked session, so this session left it alone.

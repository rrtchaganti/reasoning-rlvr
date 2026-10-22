# Engineering notes, one file per phase

Results live in `outputs/reports/phase-N.md`; incidents in `FAILURES_AND_LEARNINGS.md`; these files hold the
reasoning: what was built, how each piece is tested and why that test is the right one, the design decisions
and the alternatives rejected, known debts, and the questions a reviewer or interviewer is likely to ask.
Written at the end of each phase, before its PR is marked ready (WORKFLOW.md section 7, step 6).

Two files per phase. The *engineering notes* are the summary (what was built, how tested, why, debts, interview
questions). The *journal* is the chronological story: context first, then every decision, failure, attempt and
replacement in the order they happened, including the explanations given in chat and the rented-GPU runs as their
results arrive. Read the journal to learn the phase hands-on; read the notes to revise it.

| Phase | Notes | Journal |
|---|---|---|
| 0 Scaffold, harness, from-scratch references | [`phase-0-engineering-notes.md`](phase-0-engineering-notes.md) | [`phase-0-journal.md`](phase-0-journal.md) |
| 1 Baselines and pass@k ceiling | [`phase-1-engineering-notes.md`](phase-1-engineering-notes.md) | [`phase-1-journal.md`](phase-1-journal.md) |
| 2 SFT on distilled traces | [`phase-2-engineering-notes.md`](phase-2-engineering-notes.md) (results section after the rental) | [`phase-2-journal.md`](phase-2-journal.md) |
| 3 Rejection-sampling fine-tuning | [`phase-3-engineering-notes.md`](phase-3-engineering-notes.md) | [`phase-3-journal.md`](phase-3-journal.md) |
| 4 GRPO and ablations | [`phase-4-engineering-notes.md`](phase-4-engineering-notes.md) | [`phase-4-journal.md`](phase-4-journal.md) |
| 5 The cost of termination (budget forcing, accuracy against tokens) | [`phase-5-engineering-notes.md`](phase-5-engineering-notes.md) | [`phase-5-journal.md`](phase-5-journal.md) |

Also here: [`evidence-index.md`](evidence-index.md) (how every quoted claim maps to the problems and completions behind it; the index itself is `outputs/reports/evidence/README.md`), [`research-narrative.md`](research-narrative.md), the chain of questions across every phase (what was
known, the intuition, the question, the prediction, what the run said, what it raised; no run is scheduled
without its link there), and [`rental-setup.md`](rental-setup.md), the rented-GPU checklist used from Phase 1 on; the plan
pages [`phase-3-plan.md`](phase-3-plan.md), [`phase-4-plan.md`](phase-4-plan.md) and
[`phase-5-plan.md`](phase-5-plan.md) (the next phases, awaiting the user's signal).

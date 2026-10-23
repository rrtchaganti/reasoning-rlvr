# The evidence index: from every claim to the problems behind it

Built 22 Oct 2026 at the user's request: a reader of any sentence in the reports or the README
should be able to reach, in one hop, the problems and completions the sentence rests on, and see
the number re-derived from them. The index is `outputs/reports/evidence/README.md`; one file per
claim sits next to it (`outputs/reports/evidence/<id>.md`).

## What it is

`configs/evidence/claims.yaml` lists the claims (47 on 24 Oct 2026, Phases 1 to 6; the first version
of this page said 37 for the 39 of 22 Oct, a miscount), each with its
experiment id, the sentence as quoted and where, the runs it rests on (a named root plus a path), the
eval set, a selection rule and the number as quoted. `scripts/evidence.py` re-derives the number from
the per-sample files, compares it with the quoted one (tolerance 0.2 points unless the claim says
otherwise), lists every problem the rule selects, and writes short excerpts of the first few
completions with their gold, extracted answer, token count and finish reason. The full texts stay in
the sample files; each root in the YAML says where they live on the Hub.

The rules: `paired_flips` (two greedy runs: the paired difference and the problems only one side got
right), `decomposition` (the Phase 4 split by the reference's truncation, with the problems in each
cell), `truncation_rate`, `truncated_right` (what a cut completion is worth and how the extractor read
it), `forced` (EXP-016's forced answers), `maj_k` (EXP-017's votes: per problem, the seeded draw of
sample indices, their answers, the vote and the hit; the generator is seeded per problem with the
same seed, as in `scripts/plot_accuracy_vs_tokens.py`, so the first draw uses the same indices for
every problem and the twenty draws differ), `pass_k_newly_solved`, `perturbation_gap`
(originals paired with rewrites by `meta.original_id`), `reward_log` (a training run's per-step
statistics), `file_pointer` (claims that are not per-problem: the files, with a note).

## How to read a claim file

The header gives the experiment, where the sentence is quoted, the sentence, and the runs. Each
eval section starts with the check ("Quoted 4.1, re-derived 4.1, agree: yes"), the counts behind the
number, the list of problem ids the rule selected (truncated at 60-80 with a count), and the
examples: for a paired claim, the same problem under both runs, with the tail of each completion. A
`**NO**` in the agree column is a sentence to fix, never to leave.

## What is not in it, and what should be

- The sample files of the Phase 1 base runs and probes (`outputs/eval/base_1.7b_*`, `outputs/probe`)
  and of the Phase 5 forced-stop runs (`outputs/eval/*force*`) are on the laptop only; the roots
  `P1` and `LAP` say so. They should go to the Hub dataset as an archive with a hash, like the
  session archives, when a token is available.
- Claims about training dynamics point at the reward logs and completion logs rather than listing
  problems; the completion logs (one parquet per step, 256 completions with reward and advantage) let
  any step be read.
- Per-claim excerpts show the tail of a completion (240 characters); a loop is recognisable from its
  tail, a derivation is not, so the full text in the sample file is the evidence and the excerpt the
  pointer.

## Rebuilding

```bash
python scripts/evidence.py --claims configs/evidence/claims.yaml \
  --root P1=outputs --root P23=../h100_session_archive/outputs --root P4=../h100_p4_session2_archive/outputs \
  --root A2=../h100_p4_session2_addendum2/outputs --root HUB=../hub_samples --root LAP=outputs \
  --out outputs/reports/evidence
```

Adding a claim: one YAML entry with its rule, runs, eval and the number as quoted; the index then
checks it. A claim whose files are missing is listed as such, not skipped silently.

# Engineering workflow for `reasoning-rlvr`

How this repository is built, phase by phase. These rules hold for the whole project and are the same rules a reviewer at a lab would expect to see in the git history. Adopted 14 Oct 2026 before Phase 0.

---

## 1. Repository shape

- **One repository**, rooted at this folder. GitHub name `reasoning-rlvr`. The plan documents (`PLAN.md`, `WORKFLOW.md`, `GLOSSARY.md`, `METHODS_AND_METRICS.md`, `PHASES_IO_AND_HYPERPARAMS.md`) live in it from the first commit, so the design history is reviewable.
- **Code lives in the Windows folder, runs in WSL2.** The repo is at `/mnt/c/Users/chaga/Documents/Job_shift/Independent_research_projects/reasoning_rlvr` from Ubuntu. The Python environment, the Hugging Face cache and vLLM's cache stay on the Linux disk (`~/venvs/rlvr`, `HF_HOME=~/.cache/huggingface`) because virtual environments and large caches on `/mnt/c` are slow and break file permissions. `outputs/` is git-ignored and can point anywhere via `RLVR_OUTPUT_DIR`.
- **What is never committed**: datasets, model weights, checkpoints, LoRA adapters, tokenizer dumps, sampled generations, W&B run directories, logs, caches, virtual environments, secrets. Enforced by `.gitignore`; see section 4 for where each of these goes instead.
- **What is committed**: source, tests, configs, scripts, documentation (including `docs/phase-N-engineering-notes.md` per phase), Markdown reports with tables, small JSON summaries under 1 MB (per-run metrics, CIs), figures as PNG or SVG under 1 MB, data manifests (URL, revision, content hash, row count), and the compute ledger.

## 2. Branching: one branch per phase

```
main ──●──────────────●─────────────────●──────────────▶   always green, tagged per phase
        \            /  \              /
         phase-0 ──●──●      phase-1 ──●──●──●
```

- `main` is protected in spirit: nothing lands on it except through a pull request, and it must pass CI at every commit. It is always in a state where someone can clone it and reproduce every result reported in the README.
- Each phase is developed on its own branch, named `phase-N-short-slug`:
  `phase-0-scaffold-harness`, `phase-1-baselines`, `phase-2-sft`, `phase-3-rft`, `phase-4-grpo`, `phase-5-dpo`, `phase-6-test-time-compute`, `phase-7-distil-serve`, `phase-8-writeup`.
- A phase branch is opened from the tip of `main` when the phase starts and merged back when the phase's exit criterion in `PLAN.md` is met. The merge is a **merge commit** (`--no-ff`), so the branch history survives, and `main` is tagged `phase-N` at the merge. Model cards and the write-up cite these tags.
- Fixes to something already on `main` (a harness bug found during a later phase) go on a `fix/short-slug` branch, get their own pull request, and are merged before the phase branch rebases onto `main`. A harness fix that changes any reported number triggers a re-run of the affected evals and a `FAILURES_AND_LEARNINGS.md` entry.
- Long-running experiments inside a phase (ablations, sweeps) do not get branches. They get a config file, a row in `experiments.md`, and a W&B run whose config records the commit hash. Branches are for code, not for runs.
- Rebase a phase branch onto `main` only when `main` has moved (after a `fix/` merge). Never rebase or force-push `main`.

## 3. Commits and pull requests

- **Small commits, each one green.** `pre-commit` runs `ruff check`, `ruff format`, and the fast unit tests before every commit when committing from inside WSL (`pre-commit install` there). Commits made from the Windows side run the same gate by hand first: `bash scripts/check.sh` in WSL. A commit that breaks tests is not made.
- **Conventional-commit prefixes** so the history reads as a changelog:
  `feat:` new capability, `fix:` bug fix (say which number it changed, if any), `test:` tests only, `docs:` documentation, `refactor:` no behaviour change, `chore:` tooling, `exp:` an experiment registered in `experiments.md` or a result committed to `outputs/reports/`, `data:` a manifest or filtering script change.
  Subject in the imperative, under 72 characters, body explains *why* when it is not obvious.
- **Attribution.** Commits made with Claude Code end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- **One pull request per phase**, opened as a draft on the day the branch is opened, so the diff and the checklist accumulate in public. The PR body follows `.github/pull_request_template.md`: objective, what changed, exit-criteria checklist copied from `PLAN.md`, result table or link to the report, compute ledger rows added, failures logged, what the next phase needs from this one.
- **Merge when** CI is green, every exit criterion is ticked or explicitly marked "not met, because", `README.md` has the phase's status row and tables updated, and `COMPUTE.md` has the phase's rows.
- Self-review before merging: read the full diff once as if someone else wrote it. Interviewers do look at PR descriptions.

## 4. Where large things go

| Artefact | Lives in | Repo keeps |
|---|---|---|
| Eval sets (GSM8K, MATH-500, AIME24) | Hugging Face Hub, loaded by `datasets` with a pinned revision | `data/manifests/*.json`: dataset id, revision, split, row count, content hash |
| Generated sets (perturbed GSM8K, Countdown) | regenerated by script, seeded | the generator, the seed, the manifest hash; the generated file is ignored |
| Filtered SFT / RFT / GRPO prompt files | `outputs/data/` (ignored) and the project's HF dataset repo | the filtering script, its config, and the manifest hash of its output |
| Checkpoints and adapters | HF Hub model repos (`sft`, `rft`, `grpo`, `distilled-0.6b`) | model card source under `cards/`, the config that produced it, the commit hash and `phase-N` tag |
| Sampled generations (n = 64 per problem) | `outputs/samples/` (ignored), uploaded to the HF dataset repo at phase end | the per-run `metrics.json` summary and the report Markdown |
| Training curves and logs | W&B project `reasoning-rlvr` | exported PNG figures under `outputs/reports/figures/` (committed) |
| Secrets (`HF_TOKEN`, `WANDB_API_KEY`) | `.env` (ignored), environment on the rental box | `.env.example` with variable names only |

## 5. Reproducibility hooks that the code enforces

- Every entry point writes `run_info.json` into its output directory: git commit hash, whether the tree was dirty, the resolved config, the seed, `pip freeze`, the vLLM and torch versions, hostname and GPU name, start and end time.
- A dirty working tree is allowed for local debugging and refused for any run whose numbers will be reported (`--allow-dirty` must be passed explicitly, and the report then shows `-dirty`).
- Configs are YAML under `configs/`; the config file path and its content hash go into W&B and `run_info.json`. Changing a config is a commit, not an edit on the box.
- `COMPUTE.md` is appended by hand at the end of each rented session, from the W&B wall-clock and the provider's invoice line, one row per run.

## 6. Continuous integration

`.github/workflows/ci.yml` runs on every pull request and every push to `main` or a `phase-*` branch, on a CPU runner: install the package, `ruff check`, `ruff format --check`, `pytest -m "not gpu"`. GPU-marked tests (`@pytest.mark.gpu`) run locally before a PR is marked ready. CI stays under five minutes; anything slower is a GPU test.

## 6a. Local actions log

`GIT_ACTIONS_LOG.md` (git-ignored, never pushed) records every git, GitHub and environment
command run for this project: where, why, the exact command, the result, and how to undo it.
It is the personal audit trail for the repository and doubles as a cheat sheet.

## 7. The per-phase routine

1. `git switch main && git pull && git switch -c phase-N-slug`.
2. Open a draft PR with the template; paste the phase's objective and exit criteria from `PLAN.md`.
3. Pre-register the phase's experiments in `experiments.md` (hypothesis, metric, falsifier) and commit that before any run.
4. Build, test, run. Commit small. Log every diagnosis in `FAILURES_AND_LEARNINGS.md` as it happens, not at the end.
5. Write results into `outputs/reports/phase-N.md`, update `README.md` tables and the status row, append `COMPUTE.md`.
6. Write `docs/phase-N-engineering-notes.md`: what was built, how each piece is tested and why that test, the design decisions and the alternatives rejected, known debts, and the interview questions the phase invites. Results go in the report; reasoning goes here.
   Alongside it, keep `docs/phase-N-journal.md` *as the phase happens*: the chronological story (context, each decision with the explanation given at the time, each failure, what was tried, what replaced it, rented-GPU results as they arrive). The notes are the summary; the journal is how to learn the phase hands-on.
7. Tick the checklist, mark the PR ready, merge with a merge commit, tag `phase-N`, push tags.
8. Push checkpoints and sample sets to the Hub with the tag in their cards.

## 8. Remote setup (done once)

GitHub CLI is not installed on this machine. Either install it and let it create the repo:

```powershell
winget install GitHub.cli
gh auth login
gh repo create reasoning-rlvr --private --source . --remote origin --push
```

or create an empty repository named `reasoning-rlvr` on github.com (no README, no .gitignore, no licence, all of those are already here) and run:

```bash
git remote add origin git@github.com:<user>/reasoning-rlvr.git
git push -u origin main
git push -u origin phase-0-scaffold-harness
```

Flip the repository to public when the README has its first results table, or earlier if you want the commit history visible from day one. Enable "require a pull request before merging" on `main` under Settings → Branches once the remote exists.

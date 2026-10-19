# reasoning-rlvr: working rules for Claude Code

Read first: `WORKFLOW.md` (branch, PR and phase rules), `PLAN.md` section 4 (rigour standards),
`docs/rental-setup.md` (rented boxes; section 2c for E2E Networks). On an E2E node, read
`docs/e2e-session-handoff.md` next: it holds the current state and the exact next steps.

Rules that bite:

- The harness (`rlvr-eval`, `rlvr-sft`, `rlvr-build-sft`) refuses to run on a dirty tree. Never edit a
  tracked file on a box; notes go to `outputs/` (git-ignored). Docs, journal and commits are the
  laptop session's job.
- Secrets only as environment variables (`HF_TOKEN`, `WANDB_API_KEY`) typed into the shell of the
  session. Never in a file in the repo, in a config, or on the volume.
- Long jobs run inside `tmux`. The session scripts (`scripts/phase*_rental.sh`) are resumable: after a
  drop, rerun the same command and finished steps are skipped.
- A node bills while idle, and a powered-off E2E node still bills. A node is either running a step or
  being terminated. Before terminating: Hub uploads done, `metrics.json`, `report.md`, `run_info.json`
  and `train_log.jsonl` copied to the laptop.
- Terminating a node, deleting outputs, or starting anything that costs more than about one GPU-hour
  beyond the handoff's listed steps: ask the user first.
- Log every git, environment and install action with where / why / ran / result / undo. On the laptop:
  `GIT_ACTIONS_LOG.md` (git-ignored). On a box: `outputs/e2e_session_log.md` (ignored); the user carries
  it back to the laptop session.
- Gate before any commit, laptop only: `set -o pipefail && bash scripts/check.sh`. Nodes hold a
  read-only GitHub token and never commit or push.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. PR bodies end
  with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Dates in documents are absolute (for example "17 Oct 2026"). Numbers in reports carry their CIs;
  comparisons between checkpoints are paired on shared problem ids.
- Report outcomes as they are: a failed step is reported with its output, a skipped step is named as
  skipped. The user is not fluent in Linux or git; prefer complete, paste-ready commands and say what
  each one will do before running it.

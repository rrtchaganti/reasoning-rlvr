# Phase 0 journal: scaffold, harness, references (14 Oct 2026)

This is the chronological story of the phase: what the situation was, what was tried, what failed, what
replaced it, and what was explained along the way. The engineering notes (`phase-0-engineering-notes.md`)
are the summary; the report (`outputs/reports/phase-0.md`) has the numbers; `FAILURES_AND_LEARNINGS.md`
has the one-paragraph post-mortems. Written retrospectively from the actions log and the failures log
on 15 Oct 2026; later phases' journals are written as the work happens.

## Context before the first commit

- Decisions were confirmed on 13 Oct: Qwen3-1.7B-Base as the one base model, Qwen3-0.6B-Base as the
  student, USD 100-170 of rented A100 time, one seed per ablation plus one seed-repeat of the main GRPO
  run, TRL for training with vLLM for sampling, Countdown as the out-of-distribution task, a perturbed
  GSM8K as the reasoning-versus-recall metric, and a contamination report run on every checkpoint.
- The machine: a laptop with an RTX 4060 (8 GB) and 15.7 GB of RAM, Windows 11, WSL2 Ubuntu 24.04. vLLM
  only runs on Linux, so everything executes in WSL while files are edited on Windows. The WSL user has no
  passwordless sudo, which decides several things below.
- The process rules (WORKFLOW.md): one branch and one pull request per phase, merged with a merge commit
  and tagged; pre-register experiments before running them; data and weights never enter git; every
  git and environment command is logged in a git-ignored actions log so it can be undone.

## Day 1, morning: repository and environment

**Repo.** `git init -b main`, first commit of the plan and workflow files, branch `phase-0-scaffold-harness`.
Git warned "LF will be replaced by CRLF" on every file: the repo is edited on Windows and run on Linux, and
CRLF breaks shebangs and shell scripts. Fixed with a `.gitattributes` (`* text=auto eol=lf`) and
`git add --renormalize`. Then the GitHub repo was created by hand (private), `origin` added, both branches
pushed, and the draft PR opened in the browser (it was first created as a normal PR by mistake and
converted to draft from the sidebar; `gh` CLI is not installed).

**WSL memory.** WSL defaulted to 7 GB; `.wslconfig` was raised to 12 GB for vLLM. This was a mistake that
showed up an hour later.

**Environment build, attempt 1.** Launched from Git Bash as `wsl.exe -- bash /mnt/c/.../install.sh`.
Exit 127: bash looked for `C:/Program Files/Git/mnt/c/...`. MSYS rewrites any argument that looks like
a POSIX path. Fix: `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -- bash -lc "..."` with the Linux path inside the
quoted string. Every WSL launch since uses this shape.

**Environment build, attempt 2.** Stock Ubuntu has no `pip`, no `venv` module, and `sudo apt` wants a
password. Solution without sudo: `uv` installed into `~/.local/bin`, `uv venv ~/venvs/rlvr --python 3.12`,
`uv pip install vllm` (brings its own torch), `uv pip install -e ".[dev,gpu]"`. The venv lives on the Linux
disk, not under `/mnt/c`, for speed.

**Killed.** The install was stopped by the assistant's memory guard: Windows was critically low on memory
because WSL held 12 GB of 15.7 GB and the wheel downloads filled its page cache. Left behind: an empty venv
and 7.2 GB of cached wheels. Fix: `.wslconfig` to 9 GB; the rerun was done by you from PowerShell, where
nothing reaps it. Rule established: long jobs run from your own terminal; the assistant prepares the
command.

**Rerun: packages fine, CUDA not.** All packages installed (vllm 0.31.0, torch 2.13.0+cu130, trl 1.15.0,
transformers 5.17.0, datasets 5.1.0, math-verify 0.9.0, peft 0.21.2). The verify step failed with "The
NVIDIA driver on your system is too old (found version 12070)". torch was built for CUDA 13.0; the
Windows driver (566.36) exposes CUDA 12.7, and WSL uses the Windows driver. You updated the driver to
617.42 (CUDA 13.4); `torch.cuda.is_available()` became True with no reinstall. Also found: WSL rejected
the `autoMemoryReclaim` key, removed from `.wslconfig`.

## Day 1, midday: the code

Four commits, each gated by `scripts/check.sh` (ruff check, ruff format, pytest on CPU):

1. Package definition, pre-commit config, lock file (`uv pip freeze`), the check gate.
2. `src/rlvr/ref/`: from-scratch reference implementations with tests against closed forms and
   Monte-Carlo: unbiased pass@k (Chen et al.), group advantages with the std switch, the k1/k2/k3 KL
   estimators, the GRPO loss with sequence / token / Dr. GRPO aggregation and clip-higher, and a
   KV-cache sampling loop checked against `generate`.
3. `src/rlvr/eval/`: eval-set loaders with content-hash manifests (GSM8K, MATH-500, AIME24, a Countdown
   generator with an AST verifier), four prompt templates, an extraction chain that records which rule
   fired, math-verify grading, bootstrap and paired-bootstrap CIs, maj@k, pass@k curves, vLLM and HF
   samplers, the `rlvr-eval` CLI, and `run_info.json` per run with a refusal to run on a dirty tree.
   `src/rlvr/data/contamination.py`: the 13-gram and fuzzy overlap probe.
4. Docs: failures log, experiment registry, compute ledger, smoke config.

One snag during commit 3: `git add src/rlvr/data` refused because the `.gitignore` rule `data/` matched
the source package of the same name at any depth. Anchored all top-level rules (`/data/`, `/outputs/`).
A repo that ignores its own package would not import itself; `git check-ignore -v` is now a habit.

## Day 1, afternoon: the smoke run that found three harness bugs

**Smoke run 1, attempt 1.** vLLM engine start on the 0.6B model died after model load:
`fatal error: Python.h: No such file or directory`. Triton compiles a tiny CPython extension on first use
and needs the headers, which Ubuntu puts in `python3.12-dev` (sudo). Options explained at the time:
(a) ask you for the sudo password once, or (b) rebuild the venv on a uv-managed CPython that ships its
own headers. (b) was chosen because it removes the dependency on the distro interpreter entirely and
transfers to the rental box. `uv python install 3.12`, new venv, `uv python pin 3.12` (`.python-version`
committed). The rebuild script's own Triton check failed for a silly reason (Triton refuses kernels
defined on stdin), rerun from a file: `TRITON_OK`. `scripts/setup_env.sh` and `scripts/triton_check.py`
were written from this.

**Attempt 2.** The harness refused in five seconds: "working tree has uncommitted changes". That is our
own guard; the new scripts were not committed yet. Two more snags while committing them: the shell
script had CRLF endings (written by Python on Windows; "syntax error: unexpected end of file" in WSL), and
`.gitignore` ignored `.python-version`. Both fixed; every file write since passes `newline="\n"`.

**Attempt 3 (4.5 min).** GSM8K 67 % (sane). MATH-500: 61 % extraction failures, 53 of them completions
that were empty. Countdown: 100 % failures. A probe script with the HF backend printed the top-5 next
tokens at the end of each prompt and found three separate bugs:

- With the MATH prompt ending in `Solution:`, the top next token was end-of-text with probability
  0.24-0.49. The Qwen3 tokeniser merges colon-plus-newline into one token, so the four shots contained
  `[Solution][:⏎]` while the prompt ended on a bare `[:]`, a state the shots never show. Fix: the prompt
  ends with the newline, so the last prompt token equals the shots'. A regression test pins it.
- The think-tag prompt ended on `<think>`, a single added token the base model has barely seen; output
  after it was garbage (`__(*`, `.linkedin .linkedin`). With a newline after it, normal text. Fix:
  `Assistant:⏎<think>⏎`, completion prefix `<think>⏎`.
- vLLM was configured with `include_stop_str_in_output=False`, so `</answer>` was stripped and the
  extractor, which requires the closing tag, could never match. Fix: keep the stop string; the HF
  sampler mirrors it with `truncate_at_stop`.

Also: `$228 000` (space-grouped thousands) was extracted as `228`; the number regex now accepts it.

The patch for these was first attempted through a Bash heredoc; it matched nothing because the heredoc
collapsed doubled backslashes. Applied through a script written with the editor tool instead. This
"no backslashes through heredocs" rule bit twice more in Phase 1.

## Day 1, evening: EXP-000, the reproducibility check

The pre-registered Phase 0 exit test: three identical greedy evals must agree. They did not. Same
commit, config, seed, temperature 0: GSM8K 68 / 70 / 68, MATH-500 25 / 27 / 31, Countdown 1 / 2 / 2 on
100-problem subsets. Comparing `samples.jsonl` problem by problem showed the generated text itself differed
on 16 / 55 / 69 of 100 problems, sometimes from the first token. That is numerical non-determinism of
batched bf16 decoding (reduction order depends on batch composition, chunked prefill, prefix caching),
not grading. vLLM 0.31 has `VLLM_BATCH_INVARIANT=1`; two runs with it set agreed on 300 / 300
completions. It became the `VLLMSampler` default (`--no-batch-invariant` to opt out), recorded in
`run_info.json`, at a cost of about 13 % slower sampling. Three final runs on the committed code: bit
identical.

A second finding from the same runs: `wall_clock_s` was 61 s while the timestamps were 135 s apart. The
timer started after the engine was built, so 75 s of model load and graph capture per run never reached
the compute ledger. Now `engine_startup_s` and `sampling_wall_clock_s` are recorded separately.

A third: `outputs/reports/phase-0.md` was not being tracked. Git cannot re-include a file under an
excluded directory; `/outputs/` had to become `/outputs/*` so that `!/outputs/reports/` could apply.

## Close-out

Report written, EXP-000 closed with its conclusion, README status row and the compute ledger (0.76 local
GPU-hours, USD 0) updated, the old system-Python venv deleted. You opened the PR body from the prepared
text, checked CI, pressed "Ready for review", merged with a merge commit (PR #1, 39a6e4b), and the tag
`phase-0` was pushed. 95 tests at that point.

## What Phase 0 taught that the later phases rely on

- Every reported number comes from batch-invariant decoding; the engine's own noise was measured first.
- Prompt endings are checked at the token level against the shots; base-model prompts never end on an
  added token.
- The harness refuses a dirty tree; nothing tracked is edited while a run is in progress.
- Long GPU jobs run from your terminal; the assistant prepares the exact command and the log path.
- Files are written with explicit LF; patches with backslashes go through the editor tool, not a heredoc.

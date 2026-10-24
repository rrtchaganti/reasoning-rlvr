# Failures and learnings

Every bug that changed a number, every diagnosis made from a plot or a log, every reward-hacking
incident, every design decision reversed. Written as it happens, not at the end. Same template as
the Spark-TTS project.

Template:

```
## YYYY-MM-DD  <one-line title>                                   [phase N] [tags]
**Symptom.** What was observed (number, plot, log line).
**Diagnosis.** What it turned out to be, and how it was found.
**Fix.** What changed (commit).
**Learning.** The general rule to carry forward.
```

---

## Index

| date | incident | phase, tags |
|---|---|---|
| 2026-10-14 | Git Bash converts `/mnt/c/...` paths before WSL sees them | phase 0, tooling |
| 2026-10-14 | Ubuntu in WSL has no pip, no venv module, no passwordless sudo | phase 0, env |
| 2026-10-14 | WSL given 12 GB starved Windows; install job killed | phase 0, env |
| 2026-10-14 | vLLM's torch needs CUDA 13 driver; laptop driver is CUDA 12.7 | phase 0, env |
| 2026-10-14 | `.gitignore` rule `data/` silently excluded the source package `src/rlvr/data` | phase 0, tooling |
| 2026-10-14 | Triton cannot compile: `Python.h: No such file or directory` (WSL, no sudo) |  |
| 2026-10-14 | Smoke run 1: 53 % of MATH-500 completions were empty (model emitted end-of-text at once) |  |
| 2026-10-14 | Smoke run 1: Countdown completions were garbage after the pre-filled `<think>` |  |
| 2026-10-14 | Smoke run 1: the `</answer>` stop string was stripped, so the tag rule could never fire |  |
| 2026-10-14 | Minor: `$228 000` was extracted as `228` |  |
| 2026-10-14 | Files written from Windows Python arrive in WSL with CRLF line endings |  |
| 2026-10-14 | EXP-000: three "identical" greedy vLLM runs disagreed by up to 6 points |  |
| 2026-10-14 | run_info.json wall clock excluded engine start-up |  |
| 2026-10-14 | `outputs/reports/phase-0.md` was ignored despite the `!/outputs/reports/` rule |  |
| 2026-10-14 | Perturbed GSM8K: two ways a "recomputed" gold can be silently wrong |  |
| 2026-10-14 | Background GPU job killed by the assistant's memory guard, second time |  |
| 2026-10-14 | A "better" extraction rule that made the grader worse (rejected, kept the fallback) |  |
| 2026-10-14 | Widening a name list silently changed the perturbed GSM8K set |  |
| 2026-10-15 | A boolean 4D attention mask is silently *added*, not applied | phase 2, training, masking |
| 2026-10-15 | 13-gram overlap fires on template sentences; AIME24 looked 27 % contaminated | phase 2, contamination |
| 2026-10-15 | Source uuids are not unique: 22 rows are "NaN", 11 uuids repeat | phase 2, data |
| 2026-10-15 | The SFT smoke checkpoint never emitted `</think>`: Qwen3-Base's think tokens are untrained rows, and LoRA cannot train them | phase 2, training, tokenizer |
| 2026-10-15 | Caught before running: pure-bf16 AdamW at LR 1e-5 cannot move most weights | phase 2, training, precision |
| 2026-10-15 | "4096 new tokens" was 4096 minus the prompt: `max_model_len` clips the budget silently | phase 2, eval, protocol |
| 2026-10-15 | Wall-clock estimate off by four: "about an hour" took 3 h 55 min | phase 2, eval, process |
| 2026-10-17 | EXP-002 projected at 5-6 h; four sets took 5 h and the fifth wanted 22 more | phase 2, eval, process |
| 2026-10-17 | Data steps rewrote tracked files, and the harness refused to run on the node | phase 2, 4, tooling, harness |
| 2026-10-18 | No flash-attention for the locked torch: SFT trained unpacked with sdpa, RFT packed with flash | phase 2, 3, training, comparability |
| 2026-10-18 | A learning rate written `5e-6` in YAML is a string; the sweep's AdamW refused it | phase 2, config, training |
| 2026-10-19 | A flag equal to its parser default lost to the config: EXP-009 sampled at 2,048 tokens, not 4,096 | phase 3, eval, harness |
| 2026-10-19 | The compute-matched controls trained on data the SFT checkpoint had already seen | phase 3, design, controls |
| 2026-10-20 | The first GRPO correctness check had no teeth, and the fp32 check compared two precisions | phase 4, testing, training |
| 2026-10-20 | The session script printed its exit line before the dev evals, so a watcher would have overlapped runs | phase 4, tooling, orchestration |
| 2026-10-20 | GRPO ran out of memory twice at the plan's micro-batch of 32 | phase 4, training, memory |
| 2026-10-21 | The hacking watch's repeated-answer flag fired on both healthy GRPO runs | phase 4, reward-hacking, monitoring |
| 2026-10-21 | "Mostly by learning to finish" was an inference, stated before it was computed | phase 4, analysis, claims |
| 2026-10-21 | The base model's zero-shot MATH-500 was quoted as 63.2, which is SFT's number | phase 4, docs, numbers |
| 2026-10-21 | CI red on the Phase 4 PR: one GRPO test imported TRL, which CI does not install | phase 4, ci, tests |
| 2026-10-21 | The first EXP-016 launch downloaded every checkpoint in the SFT repo, over a slow link | phase 5, tooling, hub |
| 2026-10-22 | Two false sentences about scoring, found by reading the code against the prose | phase 1-5, docs, extraction |
| 2026-10-23 | The evidence index's page counted 37 claims where the file held 39 | phase 5, docs, evidence |
| 2026-10-23 | The pre-registered "answer before the loop" measure misread the zero-shot completions | phase 6, analysis, design |
| 2026-10-23 | The GRPO control could not start: a dirty main checkout, and a worktree WSL's git could not read | phase 5, tooling, git |
| 2026-10-23 | Batch-invariant greedy decoding is not hardware-invariant: 10-23 % of problems flip between an H100 and the laptop | phase 5, eval, reproducibility |
| 2026-10-23 | A Hub lookup inside PEFT's save hung in the WSL kernel and lost 6.4 hours of reward-model training | phase 5, tooling, wsl |
| 2026-10-24 | CI red on every Phase 5B commit: the reward-model smoke test imported PEFT, which CI does not install | phase 5, ci, tests |
| 2026-10-24 | The Hub push was refused: the laptop's cached login was a read-only token | phase 7, tooling, hub |
| 2026-10-24 | Every table in the compute ledger after the first rendered as plain text | phase 7, docs |
| 2026-10-24 | `git add -A` committed the user's two untracked personal notes | phase 7, git, process |
| 2026-10-24 | `scripts/ci_local.sh` imported nothing: its venv pointed at a deleted worktree | phase 7, ci, tooling |
| 2026-10-24 | The Phase 7 journal and the PR #9 body said the write-up had found a mechanism that was already pre-registered | phase 7, docs, claims |

## 2026-10-14  Git Bash converts `/mnt/c/...` paths before WSL sees them        [phase 0] [tooling]
**Symptom.** `wsl.exe -d Ubuntu -- bash /mnt/c/.../install.sh` failed with exit 127: bash looked for
`C:/Program Files/Git/mnt/c/...`.
**Diagnosis.** MSYS path conversion rewrites any argument that looks like a POSIX absolute path.
**Fix.** Pass the Linux path inside a quoted `bash -lc "..."` string and set `MSYS_NO_PATHCONV=1`.
**Learning.** Anything launched from Git Bash into WSL gets its arguments rewritten; wrap them.

## 2026-10-14  Ubuntu in WSL has no pip, no venv module, no passwordless sudo     [phase 0] [env]
**Symptom.** `python3 -m pip` and `ensurepip` both missing; `sudo apt install` needs a password.
**Diagnosis.** Stock Ubuntu 24.04 image ships Python without pip; the WSL user is not NOPASSWD.
**Fix.** Use `uv` installed into `~/.local/bin`; `uv venv` and `uv pip` need neither pip nor sudo.
**Learning.** Build the environment with `uv` everywhere (local and rental) so the lock file is the
same artefact on both machines.

## 2026-10-14  WSL given 12 GB starved Windows; install job killed                  [phase 0] [env]
**Symptom.** Background environment build was stopped by Claude Code for critically low system memory
during the wheel-download stage; WSL reported 11.9 GB total, 7.2 GB landed in the uv cache.
**Diagnosis.** `.wslconfig memory=12GB` on a 15.7 GB laptop leaves under 4 GB for Windows, the IDE and
the browser; the download stage fills WSL's page cache and WSL2 returns it to Windows only slowly.
**Fix.** `memory=9GB` and `autoMemoryReclaim=gradual` in `.wslconfig`; rerun is user-triggered.
**Learning.** On a shared laptop, give WSL at most ~60 % of RAM and turn on gradual reclaim; the same
low-memory kill hit the Spark-TTS install on 8 Oct, so this is the machine's standing constraint.

## 2026-10-14  vLLM's torch needs CUDA 13 driver; laptop driver is CUDA 12.7                [phase 0] [env]
**Symptom.** `torch.cuda.get_device_name` -> "The NVIDIA driver on your system is too old (found version 12070)".
**Diagnosis.** vLLM 0.31 pins torch 2.13.0+cu130; the Windows driver 566.36 exposes CUDA 12.7, and WSL2 has no
driver of its own. The CUDA 12.6 toolkit directory seen earlier in WSL is irrelevant; only the driver matters.
**Fix.** Update the Windows NVIDIA driver to the 580 series or newer (user action). Lock file written afterwards.
**Learning.** Check `nvidia-smi` "CUDA Version" against the wheel's `+cuXXX` tag before a multi-GB install, and
pin the rental image to the same CUDA major so the lock file transfers.

## 2026-10-14  `.gitignore` rule `data/` silently excluded the source package `src/rlvr/data`   [phase 0] [tooling]
**Symptom.** `git add src/rlvr/data` refused: "paths are ignored by one of your .gitignore files".
**Diagnosis.** An unanchored directory pattern matches at every depth; the rule meant for the top-level
data folder also caught the Python package of the same name.
**Fix.** Anchor top-level-only rules with a leading slash (`/data/`, `/outputs/`, `/models/`, ...).
**Learning.** Always run `git check-ignore -v <path>` on a new package directory; an ignored source
folder would have shipped a repo that cannot import itself.

## 2026-10-14  Triton cannot compile: `Python.h: No such file or directory` (WSL, no sudo)
**Symptom.** First vLLM engine start on the 0.6B model died after model load with
`torch._inductor.exc.InductorError: CalledProcessError: ['/usr/bin/gcc', '.../cuda_utils.c', ...]`;
the first line of the gcc output was `fatal error: Python.h: No such file or directory`.
**Cause.** Triton builds a tiny CPython extension (`cuda_utils`) on first use with the include path
from `sysconfig` (`/usr/include/python3.12`). Ubuntu's `python3` package does not ship headers; they
live in `python3.12-dev`, which needs `sudo apt install`. Same class of problem as the missing `pip`
and `venv`: a distro Python is not a development Python.
**Fix.** Rebuilt the venv on a uv-managed CPython 3.12 (`uv python install 3.12`), which ships
`include/python3.12/Python.h`. Pinned with `uv python pin 3.12` (`.python-version` committed) so the
environment no longer depends on the distro interpreter at all. Verified with a 4-line Triton kernel
before paying for another vLLM startup.
**Lesson.** Check `python -c "import sysconfig,os;print(os.path.exists(sysconfig.get_paths()['include']+'/Python.h'))"`
as part of environment setup whenever Triton or torch.compile is in the stack. Added to the install
script's verify step.

## 2026-10-14  Smoke run 1: 53 % of MATH-500 completions were empty (model emitted end-of-text at once)
**Symptom.** `math500` extraction-failure rate 61 %, of which 53 completions had exactly one token
and empty text, finish reason "stop". GSM8K on the same model, same run: 0 % failures.
**Diagnosis.** Probed the next-token distribution with the HF backend. With the prompt ending in
`Solution:` the top token was `<|endoftext|>` with probability 0.24 to 0.49 on every empty case.
The Qwen3 tokeniser merges colon-plus-newline into one token, so the four shots contain
`[Solution][:⏎]` while the prompt ended on a bare `[:]`, a token state the shots never show.
GSM8K was fine because its shots read `Answer: The ...`, where the colon stands alone anyway.
**Fix.** MATH prompts (zero-shot and few-shot) now end with `Solution:` followed by a newline, so
the final prompt token is the same `[:⏎]` the shots use. Probe after the fix: top tokens
`The / To / We / Let`, normal solutions. Regression test pins the ending.
**Lesson.** Check the token boundary at the end of every prompt against the shots
(`tokenizer.encode` of the last few characters), not just the text. The classic "token healing"
bug; it costs nothing to test and silently halves an eval score.

## 2026-10-14  Smoke run 1: Countdown completions were garbage after the pre-filled `<think>`
**Symptom.** 100 % extraction failures on Countdown. Completions started with `__(*14, 2, 4, 5) = 15`
or `.linkedin .linkedin ...`, 36 % ran to the 1024-token limit repeating `Assistant: __(...)`.
**Diagnosis.** `<think>` is a single added token (id 151667) in the Qwen3 tokeniser. The base
model has barely seen it, so with `<think>` as the final prompt token the distribution is broken
whatever precedes it (tested `Assistant: <think>` and `Assistant: Let me solve this step by step.`
plus `<think>`, both garbage). With a newline after `<think>` the model writes ordinary
step-by-step text. TinyZero's prompt pre-filled `<think>` for Qwen2.5, where it is three BPE pieces.
**Fix.** think_tag prompt ends `Assistant:` newline `<think>` newline; completion prefix is
`<think>` plus newline. The later GRPO format reward is unaffected (it sees the full text).
**Lesson.** Special or added tokens at the end of a base-model prompt are a trap: a base model has
no training signal for what follows them. Prefill through to the first ordinary token.

## 2026-10-14  Smoke run 1: the `</answer>` stop string was stripped, so the tag rule could never fire
**Symptom.** Even a perfectly formatted `<answer> 20 + 21 + 7 + 18 </answer>` would have been an
extraction failure, because vLLM was configured with `include_stop_str_in_output=False` and the
stop string is `</answer>`, while the extractor requires the closing tag.
**Fix.** vLLM keeps the stop string; the HF sampler mirrors it through `truncate_at_stop` (unit
tested) so both backends hand identical text to the extractor.
**Lesson.** When the stop string is part of the answer format, keep it in the output. A smoke run
with a deliberately well-formatted fake completion through the full sampler path would have caught
this before the GPU did.

## 2026-10-14  Minor: `$228 000` was extracted as `228`
GSM8K questions sometimes write thousands with spaces and the model copies the style. The number
regex now accepts space-grouped thousands (groups of exactly three digits) and strips the spaces.
Did not change any grade in run 1 (the model's 228 000 was wrong anyway), logged for completeness.

## 2026-10-14  Files written from Windows Python arrive in WSL with CRLF line endings
`scripts/setup_env.sh` failed in WSL bash with "syntax error: unexpected end of file" because
`Path.write_text` on Windows used CRLF. Fixed with a CR-stripping `sed`; all later writes pass
`newline="\n"`. Check before committing: a recursive grep for carriage returns must be empty.

## 2026-10-14  EXP-000: three "identical" greedy vLLM runs disagreed by up to 6 points
**Symptom.** Same commit, config, seed, GPU; temperature 0; pass@1 on 100-problem subsets came out
68 / 70 / 68 (GSM8K), 25 / 27 / 31 (MATH-500), 1 / 2 / 2 (Countdown). The pre-registered falsifier
(spread above 1 point) fired.
**Diagnosis.** Compared `samples.jsonl` across runs problem by problem: the generated text itself
differed on 16 / 55 / 69 of 100 problems, in some cases from the first token. So this is numerical
non-determinism in batched bf16 decoding (reduction order changes with batch composition, chunked
prefill and prefix caching), not grading or sampling. vLLM 0.31 ships `VLLM_BATCH_INVARIANT=1`
(tuned batch-invariant matmuls, fixed-split attention, TF32 off); the RTX 4060 is in its tuned
"ada" family.
**Fix.** Two runs with the variable set agreed on 300 / 300 completions; three more after making it
the `VLLMSampler` default (commit f4d1360) also agreed exactly. Cost: about 13 % slower sampling
and 10-15 s longer start-up on this GPU. `--no-batch-invariant` remains for throughput-only work.
**Lesson.** "Temperature 0" is not "deterministic" on a batched inference engine. Measure the
run-to-run spread before measuring anything else, and if the engine offers a batch-invariant mode,
turn it on for every reported number. Also: compare the texts, not just the metric, when two runs
disagree; it separates numerics from grading in one step.

## 2026-10-14  run_info.json wall clock excluded engine start-up
`wall_clock_s` was 61 s while the start and finish timestamps were 135 s apart: the timer started
after `make_sampler`, so about 75 s of model load and graph capture per run never reached the
compute ledger. Fixed; `engine_startup_s` and `sampling_wall_clock_s` are recorded separately.
Lesson: cross-check every duration field against timestamps once before trusting it in a ledger.

## 2026-10-14  `outputs/reports/phase-0.md` was ignored despite the `!/outputs/reports/` rule
Git cannot re-include a file when its parent *directory* is excluded: `/outputs/` excluded the
directory itself, so the negation never applied. The rule is now `/outputs/*` (exclude the
contents), after which `!/outputs/reports/` works. Same change for `/data/*`. Check with
`git check-ignore -v <path>` before assuming a report will be committed.

## 2026-10-14  Perturbed GSM8K: two ways a "recomputed" gold can be silently wrong
**Context.** Phase 1 builds the perturbed GSM8K set from the calculator annotations: literals in the
`<<a*b=c>>` chain are classified as references to earlier results, perturbable question numbers, or
constants, and the chain is re-run with new numbers. Every kept problem passes a self-check (the chain
with the original numbers reproduces every annotation and the official gold). That check is necessary
but not sufficient; eyeballing random outputs against their originals found two holes.
**Hole 1: steps done in prose.** "Thrice as much" was computed in the text, not in an annotation, so
the chain was just `40 + 120 = 160`. The 40 moved, the 120 did not, the gold was wrong, and the
self-check passed because the single annotated step was consistent. Fix: every literal above 12 must
be explained (a reference, a perturbable number, a number written in the question as digits or as a
word such as "fifty" or "dozen", or a unit constant like 60 or 100); otherwise the problem is dropped
as `unexplained_constant`. 90 of 1319 test problems go that way.
**Hole 2: fractional intermediates where the story has none.** Money problems legitimately pass
through cents, so denominators of 2, 4, 5, 10, 20, 25, 50, 100 were allowed everywhere; a class of 17
students split into 2 groups then gave 8.5 students per group and the answer was still an integer.
Fix: fractional intermediates are allowed only when the question mentions money or already contains
a decimal. A unit test over 30 seeds now checks the division stays exact.
**Also.** Zero intermediates ("she had 0 notebooks") are rejected; a resampled value may not collide
with another number in the question; percentages are never perturbed because their decimal form hides
in the chain. Coverage after all rules: 830 / 1319 (63 %), reasons counted in the report JSON.
**Lesson.** A self-consistency check validates your parse of the *annotated* steps, not the problem.
Sample and read outputs against originals every time a generator changes; the two holes above were
found by reading seven examples each. Residual risk: semantically odd but arithmetically correct
stories ("2 quarters", "2/6 of the oranges"); these do not affect the gold and are documented, not fixed.

## 2026-10-14  Background GPU job killed by the assistant's memory guard, second time
**Symptom.** The first 1.7B baseline run (launched as a Claude Code background job inside WSL) was
stopped thirty seconds after the vLLM engine finished initialising, with the note that the Windows
host was critically low on memory. The 13 Oct install was stopped the same way.
**Cause.** WSL2 is capped at 9 GB of the laptop's 15.7 GB and holds it once used; the host side
(editor, browser, the assistant itself) then has about 6 GB, and the guard reaps the assistant's own
background shells first. The job itself was healthy: no OOM in WSL, GPU memory fine.
**Fix / rule.** Long GPU jobs on this machine are started from the user's own PowerShell window
(`wsl -d Ubuntu -- bash -lc "..."`), where no guard reaps them; the assistant prepares the exact
command and the log path, and reads the results afterwards. Background jobs from the assistant are
for short things (gate, lint, tests, generators). `.wslconfig` memory can also be lowered to 8 GB.
**Cost.** About 17 minutes (mostly the one-time 3.4 GB weight download, which did survive in the HF
cache) plus one engine start-up.

## 2026-10-14  A "better" extraction rule that made the grader worse (rejected, kept the fallback)
**Trigger.** On the 1.7B few-shot-CoT GSM8K run, 230 of 1319 answers came from the last-number fallback and a
read of the wrong ones found a real grader error: "there are 15 liters of water in the remaining 24 liters of
drink" was extracted as 24.
**Attempt.** A "conclusion sentence" rule: take the first number of the last sentence that starts with So /
Therefore / Thus / Hence / In total, placed before the last-number fallback. Evaluated on the stored samples
before touching the code: it changed 34 extractions, fixed 6 and broke 10 on GSM8K (net minus 0.30 points) and
fixed 3, broke 5 on the perturbed set. Rejected.
**Lesson.** Every grader change is measured on stored samples before it is adopted, and reported as a delta.
Intuition from six examples is not evidence; the base model restates givens in its conclusion sentences about as
often as it states answers. The fallback's share (17 % here) is in every table so the reader can judge.

## 2026-10-14  Widening a name list silently changed the perturbed GSM8K set
**Symptom.** After adding names to the swap pools for the names-only control, `rlvr-perturb-gsm8k --seed 0`
produced 825 problems with a new content hash instead of the committed 830 / `b5afb83e6aacafcf`, which eight
baseline runs had already used.
**Cause.** Name swaps and number resampling draw from the same seeded stream; a different pool size changes
how many draws the name step consumes (numpy's bounded integers reject-sample), so every later number
changes too. A seeded generator's output is a function of every draw it makes.
**Fix.** The default set keeps the original 30 + 30 pools byte-for-byte (verified by regenerating and comparing
the hash); the widened pools are used only by `--names-only`, a separate run. Rule: any change to any draw is
a new dataset version; either isolate the streams per component or bump the version and re-run.
**Also found while fixing it.** A patch written through a shell heredoc turned the regex `\b` into literal
backspace bytes, so the name pattern matched nothing and the set changed again (829). Exact-bytes patching
only, and `cat -A` on any regex line after editing it.

## 2026-10-15  A boolean 4D attention mask is silently *added*, not applied        [phase 2] [training] [masking]
**Symptom.** The reference packing test (one row holding three sequences, block-diagonal mask,
positions restarting per sequence) disagreed with the unpacked per-token losses by up to 0.04 nats,
far above float noise but far below what a real leak between sequences produces.
**Diagnosis.** The 4D mask was passed as a boolean tensor. transformers 5.17 returns an
"already prepared" 4D mask as-is, and the eager attention path adds it to the scores: True / False
became + 1.0 / + 0.0 on the logits instead of 0 / -inf. Nothing was masked; the scores were nudged.
**Fix.** `rlvr.ref.sft_loss.additive_mask` converts the boolean mask to 0 / finfo.min before it
reaches the model; with it the packed losses match the unpacked ones to 1e-4 and the plain-causal
variant of the same row provably differs (`tests/test_ref_sft_loss.py`).
**Learning.** A mask that is "accepted" is not a mask that is applied. Any hand-built attention
mask needs an equality test against the unmasked per-sequence computation, not just a shape check.
The same test is the reason `rlvr-sft` refuses `packing: true` without a flash-attention backend:
TRL only warns, and the leak is silent.

## 2026-10-15  13-gram overlap fires on template sentences; AIME24 looked 27 % contaminated        [phase 2] [contamination]
**Symptom.** First SFT data build: 8 of 30 AIME24 problems and 6 MATH-500 problems "hit" by a 13-gram
in OpenR1-Math-220k, 0 by the fuzzy detector on AIME. Printing the pairs showed every AIME hit pointing
at the *same* training item, an unrelated AIME problem.
**Diagnosis.** After punctuation stripping, "can be expressed as m n where m and n are relatively
prime positive integers find m n" is a 13-gram shared by most AIME-style problems. The GPT-3 n-gram
convention has no notion of boilerplate; one template sentence in the training set hits every eval
problem that uses it.
**Fix.** `ngram_overlap(max_df=5)`: a gram that occurs in more than 5 training items is a template
and is ignored for the *reported* count; the conservative drop set still removes every hit item.
Reports now show any-13-gram, rare-13-gram, fuzzy, and "contaminated = rare or fuzzy" side by side.
**Learning.** Read the hit pairs before believing an overlap number. A detector that cannot say
*which* training item matched is not a detector; the ten worst pairs belong in every overlap report.
**Update (same day).** The df cutoff did not work on the rebuilt set: the template exists in many variants
(p/q vs m/n, tfrac vs frac, "n is not divisible by the square of any prime"), and in the 23k kept items
several variants occur in only four training items, under the cutoff. Replaced by a coverage rule: a pair
counts as contamination only when the shared 13-gram windows cover at least half of the eval problem's
words (a copy is covered end to end, a template covers one line). AIME24 went to 0 contaminated, MATH-500
to 3 of 500. Second learning: a frequency cutoff is a property of the corpus it is computed on; the
quantity that actually distinguishes a copy from a template is how much of the problem is reproduced.

## 2026-10-15  Source uuids are not unique: 22 rows are "NaN", 11 uuids repeat        [phase 2] [data]
**Symptom.** 23,058 rows but 23,026 distinct ids in the built SFT file.
**Diagnosis.** OpenR1-Math-220k has rows whose `uuid` is the string "NaN" and a few uuids attached to
two different problems. `rlvr-sft` keeps a dict by id, so 32 problems would have silently vanished
from training while the manifest counted them.
**Fix.** `unique_id`: missing uuid -> `row#<index>`, repeated uuid -> `<uuid>#<index>`; rebuilt.
**Learning.** Assert id uniqueness in every generator (now in the tests) and never trust a source's
primary key.

## 2026-10-15  The SFT smoke checkpoint never emitted `</think>`: Qwen3-Base's think tokens are untrained rows, and LoRA cannot train them        [phase 2] [training] [tokenizer]
**Symptom.** Local end-to-end smoke (0.6B, LoRA r 16, 400 examples, then `rlvr-eval` on 100 GSM8K
problems): pass@1 53 %, but tag-format rate 0 %. No output of 100 contained `</think>`; 26 ended with
end-of-text right after the boxed summary and 7 jumped from the summary to `<answer>` with a blank
line where `</think>` sits in every training example.
**Diagnosis.** Not the detokeniser: the tags are added tokens with `special=False`, and both
`decode(skip_special_tokens=True)` and vLLM's `DecodeStream` keep them; retokenising the outputs
accounted for every token. The embedding matrix: rows 151667 (`<think>`) and 151668 (`</think>`) in
Qwen3-1.7B-Base are bit-identical to `<|im_start|>`, `<|im_end|>`, the tool tokens and the padding
rows (94 identical rows; 291 in the 0.6B), norm 0.37 against 1.58 for a trained row. The Base
checkpoints have never trained them. LoRA leaves embeddings frozen and the output head is tied to
the same matrix, so the logit of `</think>` equals that of every filler token: unreachable. Same root
cause as the 14 Oct entry "Countdown completions were garbage after the pre-filled `<think>`".
**Fix.** `rlvr-sft` detects a tag row that is identical to other rows and initialises it to the mean
of its plain-text pieces (`</`, `think`, `>`); LoRA runs train exactly those rows (gradient hook on
the embedding matrix, matrix excluded from weight decay, fp32 during training); a LoRA run that would
leave an untrained tag frozen is refused. PEFT's `trainable_token_indices` was tried first and failed:
TRL 1.15's fused LM head reads `lm_head.weight` directly and bypasses (and crashes on) the wrapper.
`run_info.json` records how far each tag row moved. Tests: tiny tied model with filler rows, CPU and
GPU; `tests/test_train_sft.py`.
**Verified.** Second smoke with the fixed trainer, same 400 examples and same 100 problems: both rows
initialised (norm 0.35 -> 0.56) and moved (delta norm 0.05 and 0.09); `</think>` present in 5 of 100
outputs (0 before), each one in the training position between the boxed summary and `<answer>`;
accuracy unchanged within the paired CI (-7 points, [-18, +5]). Journal section 13.
**Learning.** Before teaching a model a format built from tokens, look at the embedding rows of those
tokens; a never-trained row is bit-identical to the other never-trained rows. A tag that the
tokeniser merges into one token the base has not trained is a *new token*: it needs an
initialisation and a trainable row, and an adapter on the linear layers cannot supply either. The
20-GPU-minute smoke paid for itself: without it the rented LoRA run and every think-tag number of
the phase would have been wrong in a way the accuracy column hides (53 % looked plausible).

## 2026-10-15  Caught before running: pure-bf16 AdamW at LR 1e-5 cannot move most weights        [phase 2] [training] [precision]
**Symptom (anticipated).** The full fine-tune configs loaded the model in bf16 and trained it in
bf16 with no fp32 master copy (the plain HF Trainer does not keep one). A bf16 weight of magnitude
0.02 has a spacing of 1.2e-4; an Adam step at LR 1e-5 is a tenth of that and rounds to no change.
The loss curve would still fall (through the few small-magnitude weights that do move) while most of
the network stayed put.
**Fix.** `load_dtype: float32` for every full fine-tune, `bf16: true` as autocast only; about 27 GB
static for the 1.7B. LoRA configs keep the frozen base in bf16 because PEFT holds the adapters in
fp32, and the tag rows trained next to them are upcast to fp32 for the run. The GPU trainer test now
runs the fp32-plus-autocast combination through TRL's fused LM head.
**Learning.** "bf16 training" in a DeepSpeed recipe means bf16 compute with fp32 master weights; the
same flags on a single-GPU Trainer mean pure bf16. Check what holds the master copy before copying
a recipe's precision settings.

## 2026-10-15  "4096 new tokens" was 4096 minus the prompt: `max_model_len` clips the budget silently        [phase 2] [eval] [protocol]
**Symptom.** The base think-tag reference (`configs/eval/base_1.7b_think_tag_4096.yaml`, max_tokens
4096) reported truncated outputs of 3,865-3,994 tokens, never 4,096.
**Diagnosis.** vLLM stops a completion at `max_model_len - len(prompt)` when that is below
`max_tokens`; the config sets `max_model_len: 4096` (for the 8 GB KV cache) next to `max_tokens: 4096`.
The three SFT checkpoint configs do the same, so the paired base-vs-SFT cap is identical per problem;
the dev and n = 64 sample configs use 6144 and are unaffected; the Phase 1 rows (1024 new tokens)
always had room.
**Fix.** Window kept at 4096 for the paired rows (the base's loops never end, journal section 14, so
100-200 more tokens change nothing and a rerun costs 4 local hours); the effective budget is stated
in every table; `rlvr-eval` computes `context_budget` per eval from the prompt lengths, warns when
the window caps it and records it in `metrics.json` (`tests/test_eval_harness_end_to_end.py`).
EXP-006 amendment (d).
**Learning.** A generation budget has two knobs and the context window wins. Read the effective
budget off the run (the longest truncated output) and have the harness print it before sampling,
rather than a reviewer computing it afterwards.

## 2026-10-15  Wall-clock estimate off by four: "about an hour" took 3 h 55 min        [phase 2] [eval] [process]
**Symptom.** The base think-tag reference at the 4096 window was announced as about an hour on the
4060; it took 3 h 55 min (five eval sets, 3,179 problems, 5.3 M generated tokens).
**Diagnosis.** The estimate was anchored on the Phase 1 think-tag row (24 min) without scaling it:
that row had a 1024-token budget. Four times the tokens, and the throughput fell from 1,216 to 372
output tokens/s because the 8 GB KV cache holds about six 4096-token sequences instead of twenty.
**Fix.** None needed for the run; the estimate rule is now: tokens = problems x expected mean length
(assume the cap for a looping base), divided by the throughput measured at that context length on
that GPU, plus engine start-up. `COMPUTE.md` carries the measured throughput.
**Learning.** Throughput is not a constant of the GPU; it falls with the context length through the
KV-cache concurrency. Re-derive the estimate whenever the budget or the window changes.

## 2026-10-17  EXP-002 projected at 5-6 h; four sets took 5 h and the fifth wanted 22 more        [phase 2] [eval] [process]
**Symptom.** On the L4, EXP-002's GSM8K set ran at 3,134 output tokens/s and the whole experiment
(about 60 M tokens by the old count) was projected at 5-6 hours. The four 4-shot-CoT sets took 5 h
10 min; the Countdown set (think-tag, 500 problems x 64) then ran at 816 tokens/s with 21 hours left
after 1 h 36 min, and was stopped with nothing to keep (INR 78).
**Diagnosis.** Two errors. The throughput was measured on the right GPU at the right window (the
15 Oct rule) but on the wrong set: vLLM's rate falls with the completion length through KV-cache
concurrency, 3,134 tokens/s at 154 tokens a sample down to 816 at 2,700, so a single rate cannot
cover sets whose mean lengths differ by 18x. And the token count was anchored on a 600-token guess
for Countdown; the base's Countdown completions average 2,716 tokens (65.6 % hit the cap in the
greedy run, data that existed since 15 Oct), so the set is 87 M tokens, not 19 M.
**Fix.** Countdown moved to the A100 with three options recorded in `docs/e2e-session-handoff.md`
(keep and measure in the first ten minutes; fewer problems; shorter cap, rejected). Estimate rule
extended: per eval set, tokens = problems x n x the mean length seen for that set in the nearest run
(for a base model on Countdown or think-tag, two thirds of the cap), divided by the rate measured on
a set of similar length; read the rate off every set's progress line, not the first one.
**Learning.** The cheapest set finishes first and sets the mood. Before a multi-set run, write the
per-set token estimate down and find the most expensive set; run it first when the budget is tight,
so a stop costs the least.

## 2026-10-17  Data steps rewrote tracked files, and the harness refused to run on the node        [phase 2, 4] [tooling] [harness]
**Symptom.** On the first rented node (L4, 17 Oct) the smoke eval refused to start with "working tree has uncommitted changes" right after the data step; on the Phase 4 node (20 Oct) `setup` needed a relaunch for the same reason.
**Diagnosis.** The SFT data build rewrote the committed manifest and data report with new timings and the build commit even when it reproduced the same set; on 20 Oct the RFT builder rewrote the date line of the tracked Phase 3 data report. A node may not edit tracked files, and the harness refuses a dirty tree, so a rebuild blocked every later step.
**Fix.** A rebuild that reproduces the committed set leaves the tracked manifest, report and figure untouched (b34c25b); the RFT builder writes its report and figure next to the data, outside the tracked tree (fb856a8). The node restored the files with `git checkout` meanwhile.
**Learning.** A script that runs on a box must never write a tracked file; generated reports go next to the data, and the laptop commits them.

## 2026-10-18  No flash-attention for the locked torch: SFT trained unpacked with sdpa, RFT packed with flash        [phase 2, 3] [training] [comparability]
**Symptom.** The training configs set `flash_attention_2` with packing; on the H100 there was no wheel to install.
**Diagnosis.** Dao-AILab's released wheels stop at torch 2.8 and the lock pins torch 2.13; the kernels-hub build (`kernels-community/flash-attn`) failed on authentication (404 with the fine-grained token, 401 without). A source build looked too slow and risky on a billing node that evening, and every Phase 2 run had to share one backend to stay comparable.
**Fix.** Every Phase 2 run used `attn_implementation=sdpa` and `packing=false` (exact attention, padded batches, 1.5-2x slower), recorded in each ledger row. On 19 Oct, at the user's request, the wheel was built from source in 13 minutes (the image had nvcc 12.9) and checked against sdpa (kernel max difference 0.0039, logits within bf16 noise); Phase 3 trained packed with flash, and the wheel was kept for Phase 4. The comparability note (SFT unpacked with sdpa, RFT packed with flash; exact attention in both, different micro-batch composition) is in the Phase 3 report.
**Learning.** Check every compiled dependency against the locked torch before the session, and keep a built wheel with the run archive (the wheels went to the Hub for later sessions).

## 2026-10-18  A learning rate written `5e-6` in YAML is a string; the sweep's AdamW refused it        [phase 2] [config] [training]
**Symptom.** The three EXP-005 sweep runs died at optimiser construction on the H100 with a TypeError on `0.0 <= lr`.
**Diagnosis.** Under YAML 1.1 (PyYAML) `5e-6` without a dot is a string, and `--set learning_rate=5e-6` produced the same; the main and scaling configs were unaffected because their dotted LR came from `pick_lr.py`. The node may not edit tracked files, so it ran with a PATH shim that appended a dotted override (`docs/phase-2-journal.md`).
**Fix.** Numeric knobs are cast to the type of their default, with a test (3e055ad); the three YAMLs carry dotted floats.
**Learning.** A config value's type is part of the config; parse to the default's type and test the cast, rather than trusting the file's spelling.

## 2026-10-19  A flag equal to its parser default lost to the config: EXP-009 sampled at 2,048 tokens, not 4,096        [phase 3] [eval] [harness]
**Symptom.** The pre-registered cap rule chose 4,096 for the solve-rate pass and the script passed `--max-tokens 4096`, but the samples stopped at 2,048; an n = 32 run sampled five sets instead of two; `--limit` on the perturbed GSM8K set left 67 pairs of 300 for the gap.
**Diagnosis.** `rlvr-eval` merged flags over the config by "flag differs from the parser default", so a flag that happened to equal the default (4,096) was treated as absent and the config's 2,048 won; `--limit` on a derived set drew a subset independent of its source's.
**Fix.** Explicitly passed flags always win, and `--limit` on a derived set follows its source's ids; both with regression tests (3e055ad). EXP-009 was kept and reported as "solved within 2,048 tokens" (`experiments.md`, deviation), since every kept trace is under 2,048 tokens.
**Learning.** "Was this flag given?" is not "does it differ from the default"; argparse needs a sentinel default to tell them apart, and a run record should print the resolved values, which is what caught it.

## 2026-10-19  The compute-matched controls trained on data the SFT checkpoint had already seen        [phase 3] [design] [controls]
**Symptom.** Found on the laptop after the session: each control's `train_ids.json` is a subset of the SFT run's.
**Diagnosis.** `limit=11145, subset_seed=1` selects a seeded subset of `outputs/data/sft/train.jsonl`, the 23,000 traces SFT had trained on for two epochs; the pre-registration asked for fresh traces, and the funnel had kept every usable trace, so none were fresh. A design error, not a code bug.
**Fix.** EXP-010b and EXP-010e carry the deviation where their numbers are quoted; the claim they support is narrowed to "not more steps on the distilled distribution", not "not more data" (`experiments.md`; `docs/phase-3-engineering-notes.md`).
**Learning.** A control's data is checked against the treatment's by id before the run, not after; "fresh" is a property to test, not to assume.

## 2026-10-20  The first GRPO correctness check had no teeth, and the fp32 check compared two precisions        [phase 4] [testing] [training]
**Symptom.** The reference GRPO loss matched TRL's to ten decimals on the first full run; the first `--dtype float32` run then disagreed *more* than the bf16 one (gradient relative difference 5-11 % against 1.3 %).
**Diagnosis.** In the first run every real reward was 0 (the 0.6B base never writes a finished, correct think-tag answer in 256 tokens), so every advantage was zero and the policy term never ran; at ratio 1 the published loss's value is 0 for any centred advantages, so matching values proved nothing. In the fp32 run `--dtype float32` changed only the weights: TRL's forward still ran under bf16 autocast while the reference ran in fp32.
**Fix.** The check trains TRL on a deterministic mixed reward (a hash bit of the completion; the real reward's rate still reported) and compares the *gradients* of the two losses with respect to the LoRA weights; `--dtype float32` also turns autocast off (59b5e1e). In fp32 the gradients agree to 1.4e-5; the 1.3 % in bf16 was forward noise.
**Learning.** A check passes for the right reason only if the case it checks is exercised: confirm the signal is non-zero before reading agreement, compare gradients rather than loss values, and change one thing at a time.

## 2026-10-20  The session script printed its exit line before the dev evals, so a watcher would have overlapped runs        [phase 4] [tooling] [orchestration]
**Symptom.** Explaining the `EXIT=` line the chaining relied on, the night before session 1, found that the GRPO session script printed `EXIT=0 step=<run>` after training and *before* the dev evals of the checkpoints.
**Diagnosis.** A watcher waiting on `^EXIT=` would have launched the next run on top of the running evals: two processes on one GPU, which the session rules forbid and which would have out-of-memoried or skewed both.
**Fix.** The script prints no `EXIT=` line of its own and the watcher reads only the log's last line, anchored (a8b9f69); the fix was pulled on the node between `setup` and `smoke`, so every run of record is on it.
**Learning.** A completion marker must be the last thing a step writes; check what a watcher keys on before it runs unattended.

## 2026-10-20  GRPO ran out of memory twice at the plan's micro-batch of 32        [phase 4] [training] [memory]
**Symptom.** The debugging run hit CUDA OOM twice: first with 11 GiB fragmented, then with the expandable-segments allocator and vLLM at 0.30, 76.7 GiB allocated.
**Diagnosis.** The need was real, not fragmentation: 32 completions of about 1,700 tokens over a 151,936-token vocabulary in fp32 is about 33 GB of logits, on top of 58 GB of weights, optimiser state, reference model and vLLM (`docs/phase-4-journal.md`).
**Fix.** The user's decision at 15:23 UTC: micro-batch 16 with gradient accumulation 16 (the same 256 completions per step), vLLM at 0.30, expandable segments, for every GRPO run alike, with one fallback to 8 allowed without asking; pinned as the config default and checked by the preflight.
**Learning.** Estimate the logits tensor (batch x length x vocabulary x bytes) before choosing a micro-batch; at a 152k vocabulary it, not the weights, sets the limit.

## 2026-10-21  The hacking watch's repeated-answer flag fired on both healthy GRPO runs        [phase 4] [reward-hacking] [monitoring]
**Symptom.** After session 1, the repeated-answer flag (a group of samples agreeing on one answer) had fired on the main run and the seed repeat, neither of which showed any other sign of hacking.
**Diagnosis.** As the policy improves, more groups agree on the *correct* answer, so the plain repeated-answer rate rises with accuracy; the flag lacked a correctness term and measured improvement.
**Fix.** The flag now needs repeated *incorrect* answers, with a test of the new statistic and a fallback for older logs (fb856a8); with it no flag fired on any run.
**Learning.** A hacking signal must separate the bad behaviour from the good one that looks like it; test it on a healthy run before trusting a firing.

## 2026-10-21  "Mostly by learning to finish" was an inference, stated before it was computed        [phase 4] [analysis] [claims]
**Symptom.** The first recap of session 1 said GRPO beat RFT "mostly by learning to finish". The user asked how that was known and not assumed.
**Diagnosis.** It rested on four correlations (truncation fell where accuracy rose, other templates did not move, Countdown's gain tracked its truncation, the finished rate rose in training), and one curve pointed the other way (the correct rate among finished completions rose from 0.52 to 0.79 in training).
**Fix.** `scripts/decompose_gain.py` splits the paired gain per problem by whether the reference completion finished, with bootstrap CIs: the whole gain sits on problems RFT truncated (+10.2 of +7.0 on MATH-500) and the both-finished difference is 0.0 [-4.3, 3.9]. The report and README now state the measured split, with its caveat (truncation is an outcome of the policy, so the split is descriptive).
**Learning.** When the per-sample files can answer a question, compute the answer; a mechanism sentence carries the number that measures it.

## 2026-10-21  The base model's zero-shot MATH-500 was quoted as 63.2, which is SFT's number        [phase 4] [docs] [numbers]
**Symptom.** A recap and the Phase 4 report's section 4.1 gave the base model's best-prompt MATH-500 as 63.2 and concluded that the pipeline added about 1 point there.
**Diagnosis.** 63.2 is the SFT checkpoint's zero-shot number (Phase 2); the base model's is 53.8 (Phase 1). The figure was carried from memory of a table, not read from it.
**Fix.** Corrected where it appeared, with a note in the report (section 4.1): the pipeline adds 11 points on MATH-500 under the best prompt, and the claim that training "mostly repaired a format it broke" holds for GSM8K's trained-format row, not for MATH-500.
**Learning.** Quote a number from its file, every time; the evidence index (22 Oct) later made this a check.

## 2026-10-21  CI red on the Phase 4 PR: one GRPO test imported TRL, which CI does not install        [phase 4] [ci] [tests]
**Symptom.** The Phase 4 PR (opened as a draft on 21 Oct) showed two failed checks (the workflow runs
on the push and on the pull_request event). The laptop gate and the CI's own commands were green
locally (ruff clean, 198 tests including the slow ones).
**Diagnosis.** CI installs the core dependencies only (`pip install -e ".[dev]"`; the GPU extras
`trl`, `vllm`, `peft`, `wandb` are not installed). `tests/test_train_grpo.py::test_grpo_config_matches_shape_b`
calls `make_grpo_config` and `make_trainer_class`, which import TRL lazily, so on the runner it raised
`ModuleNotFoundError`. The other TRL-using tests (`test_train_sft.py`) guard with
`pytest.importorskip("trl")`; this one, added with the Phase 4 code on 20 Oct, did not, so every CI run
of the branch since then was red and nobody looked until the PR. Reproduced on the laptop by running
the suite with `sys.modules["trl"] = None` (and vllm, peft, wandb): the one test fails; with the guard
it skips and 194 pass.
**Fix.** `pytest.importorskip("trl")` at the top of the test (commit below). The guard means CI never
exercises that test; the laptop gate and the node, which have TRL, do.
**Learning.** Any test that reaches a GPU-extra import, even lazily, needs the `importorskip`, and the
branch's CI status should be read after the first push of a phase, not at the PR. A CI simulation with
the extras blocked (`scratchpad/ci_no_trl.sh`, the `sys.modules[...] = None` trick) is cheap enough to
run before each PR.

## 2026-10-21  The first EXP-016 launch downloaded every checkpoint in the SFT repo, over a slow link        [phase 5] [tooling] [hub]
**Symptom.** The laptop sweep's first launch spent its first hour downloading: the SFT repo's ablation and control checkpoints arrived along with the root model, at 0.4-0.65 MB/s.
**Diagnosis.** Passing a Hub repo id to vLLM makes its loader fetch every `model.safetensors` in the repo, including the ones under `ablations/` and `controls/`; the rental script had always used explicit excludes, the new laptop script did not.
**Fix.** A `download` step with the rental script's excludes, the runs reading local copies (e3f7e81); the sweep reordered checkpoint-outer so one checkpoint's five budgets run while the next downloads (1cf82a5).
**Learning.** Download a model to a local path with explicit includes and point the engine at the path; a repo id hands the file choice to the library.

## 2026-10-22  Two false sentences about scoring, found by reading the code against the prose        [phase 1-5] [docs] [extraction]
**Symptom.** The Phase 1 report and the three model cards said prose answers without a box are "counted as failures, never as wrong"; several explanations in chat said a truncated completion "scores as wrong" or "yields nothing". The user's learning session (22 Oct 2026) read `grade()` and the extractor and found both false.
**Diagnosis.** `grade(None, gold, type)` returns False: an extraction failure is scored wrong in every accuracy and additionally reported as a rate. The extractor falls back (final phrase, last number, boxed) on any completion, cut or not; measured on the archives (`outputs/reports/phase-4-truncated-accuracy.md`), truncated think-tag completions are right 31.1 % of the time on GSM8K and 13.5 % on MATH-500 for `sft` (91.9 / 90.8 % when finished), and 58.8 % under zero-shot on MATH-500 where the box precedes the loop. Only the GRPO training reward scores a cut completion 0.
**Fix.** The report sentence, the cards and `push_to_hub.py` reworded with the numbers; the table added next to the Phase 4 decomposition (section 4.5) and cited in the Phase 5 report and notes; the three Hub cards re-pushed on 24 Oct 2026 (checked against `cards/*.md`).
**Learning.** A mechanism sentence without its number is a guess dressed as a fact; every such sentence carries the measurement it rests on, and the code path is read before the prose is written.

## 2026-10-23  The evidence index's page counted 37 claims where the file held 39        [phase 5] [docs] [evidence]
**Symptom.** `docs/evidence-index.md` (22 Oct) said the claims file listed 37 claims; re-counting while adding the Phase 5B claims found 39.
**Diagnosis.** The number was written by hand from a scroll through the YAML, not counted by a command.
**Fix.** The page and the README corrected, with the miscount stated on the page (b3c6d67); later counts were taken with `grep -c "^  - id:"`.
**Learning.** A count quoted in a document is a measurement; take it with a command, like any other.

## 2026-10-23  The pre-registered "answer before the loop" measure misread the zero-shot completions        [phase 6] [analysis] [design]
**Symptom.** The first EXP-023 run put the loop onset of zero-shot completions in their first 2-10 % and found an answer before it in only 10-41 % of them, while truncated zero-shot completions were known to be right 59-63 % of the time.
**Diagnosis.** Under zero-shot the trained checkpoints write the whole solution, box the answer, then write the whole solution again: the repeating unit is the solution itself, so the first recurring window starts near the beginning and the answer sits inside the first cycle, after the measured onset.
**Fix.** A second measure, "answer by the end of the first cycle" (onset + unit + 50 words), added after the first run and recorded as a deviation in the EXP-023 entry; both measures are reported (`outputs/reports/phase-6.md`).
**Learning.** A text measure designed on one shape of output must be read against an example of every shape before its numbers are believed.

## 2026-10-23  The GRPO control could not start: a dirty main checkout, and a worktree WSL's git could not read        [phase 5] [tooling] [git]
**Symptom.** The queued EXP-020 control (unforced GRPO on the laptop) exited at once at 22:07 UTC; relaunched from the worktree, it failed again.
**Diagnosis.** The harness refuses a dirty tree, and the main checkout held the user's uncommitted one-line edit to `METHODS_AND_METRICS.md`, which was not to be touched. The worktree had been created from Windows, so its `.git` file points at a Windows path that WSL's git cannot follow.
**Fix.** A fresh clone inside WSL at the same commit (0d10994), the run launched there at 22:09 UTC and its outputs copied back; the clone was deleted after the merge.
**Learning.** A run that must not wait on someone else's uncommitted edit goes to a clean clone; worktrees made on one side of WSL are not usable from the other.

## 2026-10-23  Batch-invariant greedy decoding is not hardware-invariant: 10-23 % of problems flip between an H100 and the laptop        [phase 5] [eval] [reproducibility]
**Symptom.** EXP-016 paired forced-stop runs made on the laptop's RTX 4060 against unforced runs of the same checkpoints made on the H100 in Phases 2-4. The laptop control run (EXP-020b) re-ran the unforced `sft` and `rft` on the laptop.
**Diagnosis.** Same checkpoint, same harness, same batch-invariant vLLM settings, greedy: the laptop scores `sft` 76.0 / 52.4 against the H100's 74.8 / 50.6 (GSM8K / MATH-500), with 17.6 % and 23.4 % of problems flipping in one direction or the other; `rft` +0.8 / +0.6 with 14.9 % and 20.6 % flipping. EXP-000 showed batch invariance makes greedy decoding bit-identical across runs on one machine; it does not make two GPUs produce the same floating-point results, and a near-tie in the argmax early in a long completion changes everything after it. The net differences (0.6-1.8 points) sit inside their CIs but the flips are large.
**Fix.** Every comparison against an unforced checkpoint in EXP-016 re-paired against the laptop's own unforced run (`phase-5-compare-{sft,rft}-force*-vs-{sft,rft}-laptop.md`); the report, registry, README and evidence claims carry the corrected numbers (`sft`+force at 1,536: +4.6 / +7.2, first quoted +5.8 / +9.0). Comparisons against `grpo` (run on the H100 only) carry the machine term as a caveat.
**Learning.** "Deterministic" is per machine. A paired comparison is only paired when both sides ran on the same hardware and software build; a cross-machine pair needs a same-machine control before its number is believed, and the flip rate, not only the net difference, says how large the machine term is.

**Follow-up (23-24 Oct 2026).** The last cross-machine pair, GRPO under the stop rule, was re-paired against GRPO's own laptop run: the forced run at 2,048 is +2.0 [0.6, 3.6] on MATH-500, where the H100 pairing had read +0.2 [-3.4, 3.6] and the registry had recorded the falsifier as fired. GRPO's machine term (-1.8 on MATH-500, 18.6 % of problems flipping) was the size of the effect it hid.

## 2026-10-23  A Hub lookup inside PEFT's save hung in the WSL kernel and lost 6.4 hours of reward-model training        [phase 5] [tooling] [wsl]
**Symptom.** The first EXP-022 run trained for 6.4 hours, printed its held-out accuracy (87.9 %, AUROC 0.938), and then froze while saving: the adapter folder held only a README, the GPU went idle, the process stayed alive for an hour. Even `ip addr` inside WSL then hung and could not be timed out.
**Diagnosis.** PEFT's `save_pretrained` makes a Hugging Face Hub call after writing the model card. Its DNS lookup asks the kernel for the interface list over netlink, and the WSL kernel's netlink dump blocked in uninterruptible sleep (state D, wait channel `__netlink_dump_start`), as did every later netlink call. The weights existed only in that process's memory. A Wi-Fi toggle did not free it; `wsl --shutdown` waited on the stuck tasks; the user had to stop the VM's worker and restart the WSL and Hyper-V compute services from an elevated shell.
**Fix.** `rlvr.train.orm` writes the head and the adapter's state dict with `torch.save` before anything that can touch the network, saves before the held-out pass, and `load` falls back to the plain state dict (tested by deleting the PEFT folder and re-scoring); the laptop script runs the reward-model steps with `HF_HUB_OFFLINE=1` (commit 3bd2898). The run was repeated.
**Learning.** Write the result of an expensive computation to disk with the plainest call available before handing it to any library save that might do I/O beyond the file; and run steps that need no network in offline mode, so a network fault cannot reach them.

## 2026-10-24  CI red on every Phase 5B commit: the reward-model smoke test imported PEFT, which CI does not install        [phase 5] [ci] [tests]
**Symptom.** The `ci` workflow failed on all nine pushes of `phase-5b-dev` and on `phase-5b-test-time-compute` (d17eb42 to e0ffc04, 23-24 Oct). The laptop gate was green on every commit. Noticed only when the user looked at the Actions page before opening the PR.
**Diagnosis.** The same class as the 21 Oct entry: CI installs CPU torch and the dev extras, no PEFT, and runs the slow tests (`-m "not gpu"`), which the laptop gate deselects. `test_orm.py::test_train_and_score_smoke_on_tiny_model` is marked slow and builds a PEFT adapter, so it is skipped on the laptop and fails with `ModuleNotFoundError: No module named 'peft'` on CI. Reproduced in a CI-like venv (CPU torch + `.[dev]`): 1 failed, 215 passed. Blocking the import with stub modules instead misleads (transformers sees a module named `peft` and imports it), so the reliable reproduction is the clean venv.
**Fix.** `pytest.importorskip("peft")` at the top of the test; in the CI-like venv 215 passed, 5 skipped; with PEFT installed the test still runs and passes. `scripts/ci_local.sh` builds that venv once and runs CI's three commands.
**Learning.** The 21 Oct learning (read CI after the first push of a phase; simulate CI before the PR) was written and not applied: a learning that is a habit to remember fails, one that is a command to run holds. `bash scripts/ci_local.sh` runs before every PR body is written.

## 2026-10-24  The Hub push was refused: the laptop's cached login was a read-only token        [phase 7] [tooling] [hub]
**Symptom.** The upload of the laptop archive, the scores and the cards failed with HTTP 403 on the write step (`xet-write-token`); nothing was uploaded.
**Diagnosis.** `whoami` showed the WSL user logged in with a fine-grained token named for another project, scoped `repo.content.read` only. Reads had worked all along, so the gap only showed at the first write.
**Fix.** The user created a write token for the four repos, typed it into the shell with `read -s` and ran the staged script; every file was then checked against its local copy by hash.
**Learning.** Check the token's scope (`whoami`) before a push, not after; the cached login says who you are, not what you may do.

## 2026-10-24  Every table in the compute ledger after the first rendered as plain text        [phase 7] [docs]
**Symptom.** Reading `COMPUTE.md` during the Phase 7 review: on GitHub, eleven blocks of ledger rows showed as lines of pipes, not tables, and one paragraph was split in two by rows inserted into its middle.
**Diagnosis.** Rows were appended session by session after blank lines, each block without the header and separator rows a Markdown table needs; only the first block had them. The split sentence came from inserting rows at a line inside a paragraph. Both had been there since Phase 2.
**Fix.** A header row above every block and the sentence rejoined (5725e11).
**Learning.** Look at a document the way its reader will (rendered), not only as source.

## 2026-10-24  `git add -A` committed the user's two untracked personal notes        [phase 7] [git] [process]
**Symptom.** Commit 4c4d333 on `phase-7-write-up` added two Markdown files the user keeps untracked in the checkout; the commit was already pushed.
**Diagnosis.** `git add -A` stages everything untracked; the user's notes had been left out of every earlier commit deliberately.
**Fix.** The files untracked in the next commit (8bf108e), kept on disk; the PR merged with "Squash and merge" so `main`'s history never held them (checked with `git log origin/main -- <files>`); the remote branch deleted. PR #9's Commits tab still shows them, which GitHub keeps; the user accepted that.
**Learning.** Stage files by name in a checkout that holds anyone else's files, and read `git status` before committing.

## 2026-10-24  `scripts/ci_local.sh` imported nothing: its venv pointed at a deleted worktree        [phase 7] [ci] [tooling]
**Symptom.** The CI-like check stopped with 13 collection errors, `ModuleNotFoundError: No module named 'rlvr'`.
**Diagnosis.** The venv was built once, from the `rlvr_dev` worktree, with an editable install; its `.pth` file still pointed at `rlvr_dev/src`, which had been removed after the Phase 5B merge. The script reused any existing venv without checking where it imported from.
**Fix.** The script re-points the editable install at the current checkout on every run (`uv pip install --no-deps -e .`, seconds; 4c4d333); 219 passed, 5 skipped.
**Learning.** A cached environment carries the path it was built from; a reused one must be re-pointed, not trusted.

## 2026-10-24  The Phase 7 journal and the PR #9 body said the write-up had found a mechanism that was already pre-registered        [phase 7] [docs] [claims]
**Symptom.** Writing the project journey, the Phase 4 journal showed that EXP-014b's pre-registration (20 Oct) already stated that clip-higher is inert at one optimiser step per batch; the Phase 7 journal and the PR #9 description said the write-up had "found" it.
**Diagnosis.** The derivations page was written from the measured clip fraction and the entropy numbers, without re-reading the pre-registration, and the journal described the page's argument as a discovery.
**Fix.** The Phase 7 journal corrected with the date of the original statement (93d328e); the PR #10 body records the correction, since a merged PR's description is not changed.
**Learning.** Before calling something a finding, search the record for it: the registry is the first place to look.

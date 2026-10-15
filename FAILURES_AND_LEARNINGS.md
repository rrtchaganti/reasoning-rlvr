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

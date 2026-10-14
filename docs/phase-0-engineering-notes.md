# Phase 0 engineering notes: what was built, how it was tested, why it is organised this way

Companion to `outputs/reports/phase-0.md` (results) and `FAILURES_AND_LEARNINGS.md` (incidents).
This file is the one to read before an interview: it explains every design decision in the Phase 0
code, how each piece is tested and why that test is the right one, what was deliberately left out,
and the questions a reviewer is likely to ask. Written 14 Oct 2026 at tag `phase-0`.

---

## 1. The shape of the repository and why

```
reasoning_rlvr/
  src/rlvr/            the package (src layout, installed editable with `pip install -e .`)
    ref/               from-scratch reference implementations: formulas written to be defended
    eval/              the evaluation harness: datasets, prompts, sampler, extraction, metrics, report, CLI
    data/              data hygiene: contamination probe (later: perturbed GSM8K, filtering scripts)
    utils/             run metadata (commit, config hash, dirty flag, versions, GPU, timings)
  tests/               one test module per source module, plus one end-to-end module
  configs/             YAML for every run; a config change is a commit
  scripts/             thin shell/Python entry points: environment, gate, smoke, inspection
  outputs/             git-ignored except outputs/reports/ (Markdown + small JSON + figures)
  docs/                these notes, one file per phase
  PLAN.md, WORKFLOW.md, METHODS_AND_METRICS.md, GLOSSARY.md, PHASES_IO_AND_HYPERPARAMS.md
  experiments.md, FAILURES_AND_LEARNINGS.md, COMPUTE.md, README.md
```

**Why a `src/` layout.** The tests import `rlvr` the way a user would (from the installed package),
not from the working directory, so a missing `__init__.py` or a module that only works when run from
the repo root fails in CI rather than on the rental box. It also keeps `rlvr-eval` a real console
script (`pyproject.toml [project.scripts]`) instead of `python path/to/file.py`.

**Why `ref/` is separate from `eval/`.** Nothing in the harness depends on the reference
implementations except pass@k. `ref/` exists to be read: each file states the formula in its module
docstring, implements it in the plainest PyTorch, and is tested against a closed form or a brute
force. In Phase 4 the TRL/vLLM training loop is checked against these on a toy run; if the two
disagree, the reference is the one that is argued about line by line.

**Why both a vLLM and an HF sampler behind one interface.** vLLM is the production path (tens of
thousands of completions per eval). The HF sampler exists so the *same* harness code runs on CPU in
CI against a tiny random model, and so a disagreement between back-ends can be localised (both
return `text`, `num_tokens`, `finish_reason`, and both keep stop strings, see section 4). The
harness never touches a model directly.

**Why configs are YAML files and not CLI flags.** A reported number must be reproducible from
(commit, config, seed). `run_eval.py` merges the YAML into the argparse namespace, and the resolved
config plus its SHA-256 go into `run_info.json`. Flags override YAML for debugging only.

**Why `outputs/` is ignored but `outputs/reports/` is not.** Samples and metrics for one n = 64 run
are hundreds of MB; reports, CIs and figures are kilobytes. The ignore rule is `/outputs/*` with a
negation for `reports/`; the obvious `/outputs/` does not work because git cannot re-include a file
under an excluded directory (this bit us once; `FAILURES_AND_LEARNINGS.md`, 14 Oct).

**Why the project lives in WSL2 with code on the Windows disk.** vLLM is Linux-only. The code sits on
`/mnt/c` so the Windows editor and git client see it; the virtual environment and the Hugging Face
cache sit on the Linux disk because they are large, slow over the 9P bridge, and permission-fragile.
The Python interpreter is uv-managed (`.python-version` = 3.12) rather than the distro one, because
the distro Python has no pip, no venv and no C headers, and Triton needs the headers to build its
runtime extension. `scripts/setup_env.sh` rebuilds the whole environment with no sudo on any Linux
box, which is also how the rental box will be set up.

**Why one branch per phase with merge commits and a tag.** Each phase is one reviewable unit with
its own PR description (objective, exit criteria, results, compute, failures). `--no-ff` keeps the
branch history; the `phase-N` tag is what model cards and the write-up cite. Details in `WORKFLOW.md`.

---

## 2. The reference implementations and how each is tested

The rule for `ref/`: every function is tested against something that is *not* the function, i.e. a
closed form, a brute-force enumeration, a Monte-Carlo limit, or a different implementation.

### 2.1 `pass_at_k.py`: unbiased pass@k (Chen et al. 2021)

- **Formula.** With `n` samples and `c` correct, the probability that `k` samples drawn without
  replacement contain at least one correct one is `1 - C(n-c, k) / C(n, k)`, computed as
  `1 - prod_{i=n-c+1..n} (1 - k/i)` to avoid overflow. Returns 1.0 when `n - c < k` (every
  k-subset must contain a correct sample).
- **Tests.** (a) `test_matches_brute_force`: enumerate every k-subset with `itertools.combinations`
  and count those containing a correct index; the formula must match to 1e-12 for six (n, c, k)
  triples. (b) `test_naive_estimate_is_biased_downward`: the plug-in estimate `1 - (1 - c/n)^k` is
  shown to be *below* the exact one at (4, 1, 2) and, over 200 000 Bernoulli draws, the exact
  estimator's mean equals the true `1 - (1-p)^k` while the naive one sits below it (Jensen on the
  convex function `(1-p)^k`). (c) `hypothesis` property test: bounds in [0, 1], monotone in k.
- **Why this matters for the project.** The central question (Q5, Yue et al.) compares the RL
  policy's pass@1 against the base model's pass@64. A biased pass@k would bias that comparison in
  one direction; the brute-force test is the proof that it is unbiased.

### 2.2 `advantages.py`: group-relative advantages

- **Formula.** For a group of G rewards, `A_i = (r_i - mean) / (std + eps)`; with
  `normalize_std=False` (Dr. GRPO) only the mean is subtracted.
- **Tests.** Zero mean and unit variance per group; a constant-reward group gives exactly zero
  advantage (no learning signal, which is why dynamic sampling exists); the Dr. GRPO variant only
  centres; and `test_std_normalisation_inflates_low_variance_groups` shows the effect the Dr. GRPO
  paper criticises: a near-unanimous group gets its advantages scaled up relative to a balanced one
  with the same centred pattern. Shape validation rejects 1-D input and G = 1.
- **Interview angle.** "Why does GRPO not need a critic?" The group mean is the baseline; the test
  with a constant group is the concrete demonstration that the baseline removes all signal when the
  group agrees.

### 2.3 `kl.py`: k1, k2, k3 estimators (Schulman's blog post)

- **Formulas.** With `r = log pi_ref - log pi` on tokens sampled from `pi`: `k1 = -r`,
  `k2 = r^2 / 2`, `k3 = exp(r) - r - 1`.
- **Tests.** A 16-way discrete distribution pair with a known exact KL; 200 000 samples from `p`.
  k1 and k3 means match the exact KL within 0.02 (unbiased); k3 is non-negative sample-by-sample
  and has lower variance than k1 (the reason GRPO implementations use it); k2 is only checked to be
  finite and non-negative because it estimates a different quantity in general. `masked_mean`
  ignores padded positions.
- **Interview angle.** "Why k3 and not k1 in the loss?" k1 is unbiased but can be negative per
  token and is high-variance; k3 is unbiased, always non-negative and lower-variance. The test shows
  both facts numerically rather than quoting them.

### 2.4 `grpo_loss.py`: the clipped objective with every ablation as a switch

- **Formula.** Per token `ratio = exp(logp - logp_old)`; the clipped policy term
  `-min(ratio * A, clip(ratio, 1 - eps_low, 1 + eps_high) * A)`; optional `beta * k3` KL to the
  reference; aggregation `sequence` (mean per sequence, then mean over sequences, the original
  GRPO), `token` (mean over all unmasked tokens, DAPO), `drgrpo` (sum over tokens divided by a
  constant, Dr. GRPO). Returns loss and diagnostics (clip fraction, KL, ratio mean/max).
- **Tests.** Ratio is exactly one on the first update and nothing is clipped;
  `test_gradient_equals_reinforce_with_baseline_when_ratio_is_one` checks the analytic gradient
  (`-A` per token under token aggregation) with autograd, which pins the sign convention; sequence
  aggregation weights tokens of short completions more than token aggregation does (the length bias
  Dr. GRPO describes), demonstrated on a batch with unequal lengths; Dr. GRPO divides by a constant
  regardless of length; clipping bounds the update (clip fraction is 1.0 when every ratio is
  outside the band); clip-higher with `eps_high = 0.28` lets a ratio of 1.22 through while the
  symmetric 0.2 clips it; the KL term is zero when policy equals reference and requires `logp_ref`
  when `beta > 0`; shape errors are raised.
- **Interview angle.** Every Phase 4 ablation (A1 no-KL, A2 clip-higher, A3 Dr. GRPO loss) is one
  keyword argument here, so the ablation diff is a config line, not a code branch.

### 2.5 `sampling.py`: a KV-cache decoding loop

- **What it is.** Prefill the prompt once, then one forward per new token with
  `past_key_values`, with greedy or temperature/top-k/top-p sampling. Not for throughput; it exists
  so that the two facts of decoding (prefill is parallel, decode is serial and memory-bound) can be
  pointed at in code.
- **Tests.** On a tiny random Qwen2-architecture model (same cache plumbing as Qwen3, runs on CPU
  in milliseconds): greedy output equals `model.generate(do_sample=False)`; the cached loop equals
  recomputing the full sequence every step (the cache is exact, not approximate); generation stops
  at the first EOS; sampling is reproducible given a `torch.Generator`; `_filter_logits` keeps the
  right mass under top-k and top-p.
- **Interview angle.** "Why does vLLM get so much more throughput than this?" Batching of decode
  steps across sequences, paged KV memory, CUDA graphs, fused kernels, continuous batching; this
  loop does one sequence with Python overhead per token. Also: KV cache size for Qwen3-1.7B is
  112 KB per token (28 layers x 8 KV heads x 128 dims x 2 (K, V) x 2 bytes), which bounds batch
  size at a given context length on 8 GB.

---

## 3. The evaluation harness and how it is tested

### 3.1 Datasets (`eval/datasets.py`)

- Loaders pin a Hugging Face dataset id and revision, return `EvalSet(name, answer_type,
  problems, source, revision)`; `content_hash()` is a SHA-256 over the problems' ids, questions
  and answers; `manifest()` writes id, revision, count and hash. The hash goes in the report, so a
  number can be matched to the exact eval set rows.
- `gsm8k_gold` parses the `#### N` line and drops the `<<3+4=7>>` calculator annotations.
- **Countdown** is generated, not downloaded: `generate_countdown(n, seed, spec)` draws four
  numbers in [1, 25], builds an expression over them with `+ - * /` using exact `Fraction`
  arithmetic, and keeps targets in [10, 100]. Each problem stores a `reference_solution` so the set
  is known to be solvable. `verify_countdown` parses the model's equation with `ast`, allows only
  the four operators and parentheses, requires every given number to be used exactly once,
  evaluates with `Fraction` (so `6 / 3 + 2` is exact and `7 / 2 + 2 = 5` is not), tolerates a
  trailing `= target`, and rejects `**`, function calls, names, division by zero and empty input.
- **Tests.** Generation is seeded (two calls with the same seed give the same hash, a different
  seed a different one), every generated reference solution verifies, targets are in range; the
  verifier rules above are each exercised by a positive or negative case; `subset(limit, seed)`
  is deterministic and returns everything when the limit exceeds the size; an unknown eval name
  raises `KeyError`; the manifest has the expected fields.
- **Why an AST verifier rather than `eval`.** `eval` on model output is a code-execution hole and
  would accept `abs(-9)` or `3 ** 2`; the AST whitelist is the entire rule set in twenty lines.

### 3.2 Prompts (`eval/prompts.py`)

- Four templates fixed for the whole project: `zero_shot`, `few_shot` (4 shots, answer only),
  `few_shot_cot` (4 worked shots: the Wei et al. GSM8K shots, the Minerva MATH shots), `think_tag`
  (R1-Zero / TinyZero style `<think> ... </think> <answer> ... </answer>` with a one-line system
  preamble). Base models get plain text, no chat template. Each template defines its stop strings
  (`\n\nProblem:`, `\nQuestion:` ...), because a few-shot base model will otherwise write the next
  question itself.
- **Two regression tests came from the first smoke run** and are the most instructive in the
  suite. (a) MATH prompts must end `Solution:` plus newline, because the Qwen3 tokeniser merges
  colon-newline into one token: the shots contained `[Solution][:\n]` and the prompt ended on a bare
  `[:]`, a token state the shots never show; the 0.6B base put 24-49 % probability on end-of-text
  and 53 % of completions were empty. (b) The think-tag prompt must not end on the `<think>` token
  itself (a single added token the base model has barely seen; output after it was garbage), so it
  ends `Assistant:` newline `<think>` newline and the completion prefix is `<think>` newline.
- **Why fixed templates.** Every checkpoint in every phase is scored with the same four prompts,
  so a change in the number is a change in the model, not in the prompt. The template name is in
  every report row.

### 3.3 Sampler (`eval/sampler.py`)

- `SamplingConfig(n, temperature, top_p, top_k, max_tokens, seed, stop)`; `Completion(text,
  num_tokens, finish_reason)`. `VLLMSampler` builds one `LLM` and uses `SamplingParams` with
  `include_stop_str_in_output=True`; `HFSampler` batches `generate` and applies
  `truncate_at_stop`, which keeps the stop string, so both back-ends hand the extractor identical
  text (the `</answer>` tag is itself the stop string and must survive for the tag rule to fire).
- **Batch-invariant mode is on by default.** EXP-000 showed that vLLM's default mode is not
  bit-reproducible run to run (section 5 below); `VLLMSampler` sets `VLLM_BATCH_INVARIANT=1`
  before the engine is built, records it in `run_info.json`, and `--no-batch-invariant` turns it
  off for throughput-only work.
- **Tests.** `truncate_at_stop` unit test (keeps the stop string, earliest stop wins, no-stop case);
  the default flag and the CLI switch; and a `slow`-marked end-to-end test that runs `HFSampler` on
  a tiny random model through grading and summary. The vLLM path is exercised by the smoke runs
  (`scripts/phase0_smoke.sh`) on the GPU, not in CI.

### 3.4 Extraction and grading (`eval/extraction.py`)

- **Extraction is a chain, and the rule that fired is recorded.** Order: last `<answer>` block
  (inside it, a `\boxed{}` if present), last `\boxed{}` anywhere (balanced-brace scan, so nested
  fractions survive and a truncated box returns nothing), a final-answer phrase (`#### N`, "The
  answer is", "Final answer:"), and, for numeric sets only, the last number in the text. The source
  (`tag`, `boxed`, `final_phrase`, `last_number`, `none`) is in every sample record and counted in
  the report, so a checkpoint that drifts from `tag` to `last_number` is visible.
- **Extraction failures are a separate column, never silently wrong.** The report shows
  `extract-fail %` next to accuracy; a wrong answer and an unparseable answer are different
  failure modes with different fixes.
- **Grading.** Numeric: both sides normalised to an exact `Fraction` (strip `$`, commas, `%`,
  units, `x =`, convert `\frac{a}{b}` and `a/b`, accept space-grouped thousands), then compared.
  Otherwise `math-verify` symbolic equivalence (so `0.5`, `1/2`, `\frac12`, intervals, sets
  agree), with a string-normalisation fallback when the library cannot parse. Countdown: the AST
  verifier with the problem's numbers and target.
- **Tests.** Balanced-brace `\boxed` extraction incl. nested and truncated cases; the chain order
  on a crafted string for each rule; `#### 1,234` keeps the comma for display but grades equal to
  1234; `$228 000` extracts as 228000 and `3 1000` is two numbers; thirteen numeric normalisation
  cases (`$5.50` to 11/2, `50%`, `\dfrac{-3}{4}`, `x = 7`, `12 dollars`, `3 apples`); rejections
  (`[2,5)`, `1/0`, `x^2`); nine grading pairs incl. `None` pred; LaTeX equivalence via
  math-verify (skipped if not installed); Countdown grading through the verifier; `tag_format_ok`
  requires exactly one think block followed by exactly one answer block; vote normalisation
  (`\dfrac` vs `\frac`, spacing).

### 3.5 Metrics (`eval/metrics.py`)

- **Bootstrap over problems, not samples.** The unit of replication is the problem; the problems
  are fixed across checkpoints. 2000 resamples, percentile interval, seeded.
- **Paired bootstrap for comparisons.** Two models on the same problems: bootstrap the per-problem
  difference, so shared difficulty cancels. The test constructs two models that disagree on 10 %
  of 400 problems and shows the paired interval is narrower than the sum of the two unpaired ones,
  and exactly zero-width for identical models.
- `bootstrap_ci` test: on 500 Bernoulli(0.6) draws the interval covers 0.6 and its width is within
  25 % of `2 x 1.96 x binomial SE`, which is the sanity check that the bootstrap agrees with the
  closed form where one exists.
- **Majority vote and maj@k.** Votes are over normalised answers (`normalize_for_vote`), ties go
  to the first seen, extraction failures do not vote, a problem with no votes counts as wrong.
  maj@k for k < n averages over 20 random k-subsets. Tests: ties and `None`, the full-k and
  subset paths on a hand-built example.
- `pass_at_k_curve` wraps the unbiased estimator with a bootstrap CI per k and is tested monotone.

### 3.6 Report (`eval/report.py`) and CLI (`eval/run_eval.py`)

- `grade_completions` turns completions into `ProblemResult`s with per-sample `SampleRecord`s
  (text, tokens, finish reason, extracted answer, source, correct, format flag). `summarize`
  computes pass@1 (first sample; greedy when T = 0), mean sample accuracy, truncation rate,
  extraction-failure rate, tag-format rate, extraction-source counts, token statistics split by
  correctness, pass@k and maj@k when n > 1, and a per-level breakdown when the problems carry a
  `level`. It refuses ragged sample counts.
- The end-to-end test builds three toy problems with four fake completions each and checks every
  summary field by hand (pass@1 = 2/3, pass@4 = 1, maj@4 = 1, one extraction failure in twelve,
  one well-formed tag block, levels {1, 2}) and the Markdown row prefix.
- The CLI writes per eval `samples.jsonl` (ignored), `metrics.json`, `manifest.json`,
  `prompt_example.txt`, and per run `report.md`, `metrics.json`, `run_info.json`. It refuses to run
  on a dirty tree unless `--allow-dirty` (then the run tag ends in `-dirty`).

### 3.7 Contamination probe (`data/contamination.py`)

- Two detectors over normalised text: 13-gram hash-set overlap (catches verbatim copies) and
  rapidfuzz normalised Indel similarity >= 0.9 within a 30 % length window (catches paraphrases and
  number swaps). `overlap_report` returns hits per eval id and the set of training ids to drop.
- **Tests** are constructed so that each detector catches exactly what it should: a verbatim copy
  with a `Question:` prefix is caught by n-grams; a 14-token problem whose only change is one number
  is *not* caught by 13-grams (every window contains the number) but *is* caught by the fuzzy
  detector; a clean item is caught by neither; the report round-trips through JSON.

### 3.8 Run metadata (`utils/run_info.py`)

- Records commit, branch, dirty flag, resolved config and its hash, seed, package versions, GPU,
  argv, `pip freeze`, start/finish timestamps, wall clock, engine start-up and sampling time.
- Tests: config hash is key-order independent; `write_run_info` produces the fields; a dirty
  tree is refused when not allowed (monkeypatched) and tagged `-dirty` when allowed.

---

## 4. Test strategy as a whole

- **One test module per source module** plus one end-to-end module; 95 tests, about 20 s on CPU.
- **Markers.** `gpu` (needs CUDA; none yet, Phase 4 will add the TRL-vs-reference check) and
  `slow` (downloads or heavier CPU work; one test). CI runs `pytest -m "not gpu"`; the local gate
  `scripts/check.sh` runs `not gpu and not slow` before every commit; GPU tests run locally before
  a PR is marked ready.
- **What a test must be.** A comparison against something that is not the code under test: a
  brute force (pass@k), a closed form (bootstrap width, binomial SE), a Monte-Carlo limit (KL
  estimators, pass@k unbiasedness), an independent implementation (`generate` vs the KV loop,
  cached vs uncached), or a hand-computed expectation on a small example (summary fields).
- **Regression tests carry the incident in their name and docstring** (the two prompt tests), so
  the suite doubles as a record of what went wrong.
- **What is deliberately not unit-tested.** The vLLM path (covered by the smoke script on the
  GPU; a unit test would need a GPU and minutes of engine start-up), `math-verify` internals
  (a dependency, tested by its authors; our test checks the integration), dataset downloads
  (manifests pin them; a network test in CI is flaky by construction).
- **Lint.** `ruff` with `E, F, I, B, UP, W, N, SIM, RUF` and `ruff format`, line length 100, in
  CI and in the gate. Pre-commit hooks are installed for commits made inside WSL.

---

## 5. What EXP-000 taught about "deterministic" evaluation

The pre-registered hypothesis was that three identical greedy runs would agree to well under half a
point. They did not: pass@1 on 100-problem subsets moved by 2 (GSM8K), 6 (MATH-500) and 1
(Countdown) points, and the generated text differed on 16 / 55 / 69 of 100 problems. Comparing the
texts (not just the metric) separated numerics from grading in one step: it was the engine. Batched
bf16 decoding has a reduction order that depends on batch composition; vLLM 0.31 ships a
batch-invariant mode (`VLLM_BATCH_INVARIANT=1`: tuned batch-invariant matmuls, fixed-split
attention, TF32 off) under which two, then three, runs agreed on all 300 completions, at about
13 % more sampling time. It is now the harness default. Two consequences for the project: every
reported number is a function of (commit, config, seed), and comparisons are only valid within one
mode, because the mode changes the numbers themselves (MATH-500 33 vs 25-31), not toward a "truer"
value, just toward a fixed one.

---

## 6. Known limitations and debts at the end of Phase 0

- MATH answers stated in prose without `\boxed{}` (about 10 % on the 0.6B smoke) are reported as
  extraction failures, not graded. A "last `$...$` expression" fallback would grade more but also
  grade garbage; decision deferred until the 1.7B rows show how often it happens.
- The `think_tag` template is the only Countdown prompt; the 0.6B base nearly never closes
  `</think>` and half its completions hit the token limit. Expected for a sub-1B base, but it means
  Countdown pass@1 before RL is a floor, not a measurement.
- The HF sampler is slow (no paged cache, simple batching) and exists for tests and tiny probes.
- No multi-GPU, no quantised inference yet (Phase 7).
- The 0.6B smoke numbers are 100-problem subsets and are not comparable to any published number.

---

## 7. Interview question bank for this phase

**Scaffold and process**
- *How is the repo organised and why the src layout?* Section 1.
- *How do you make a number reproducible?* Commit + config hash + seed in `run_info.json`, dirty
  tree refused, pinned dataset revisions with content hashes, lock file, batch-invariant decoding.
- *Why merge commits and tags instead of squashing?* The phase history is the audit trail; tags are
  what cards and the write-up cite.
- *What does CI run and why is it CPU-only?* Lint, format, `pytest -m "not gpu"` in under five
  minutes; GPU tests are run locally before a PR is marked ready and their results pasted into it.

**Tests**
- *How do you know pass@k is unbiased?* Brute-force enumeration of k-subsets matches to 1e-12 and a
  200 000-draw Monte-Carlo shows the plug-in estimate biased low while ours hits the truth.
- *How did you test the GRPO loss without a training run?* Analytic gradient at ratio 1 equals
  REINFORCE with a baseline; clip fraction 1.0 when every ratio is out of band; the length bias of
  sequence aggregation demonstrated on unequal lengths; KL zero at the reference.
- *How do you test a decoding loop?* Against `generate` and against the uncached forward on a tiny
  random model; EOS and generator reproducibility.
- *Why bootstrap over problems and why paired?* Section 3.5; the test shows the paired interval is
  tighter and zero for identical models.
- *What is not tested and why?* Section 4.

**Harness design**
- *Why record which extraction rule fired?* A shift from `tag` to `last_number` between checkpoints
  is a format regression that accuracy alone hides.
- *Why count extraction failures separately?* Different failure mode, different fix; silently
  counting them wrong penalises verbose correct models and rewards terse wrong ones.
- *Why an AST verifier for Countdown?* `eval` is a code-execution hole and accepts operators the
  task forbids; the whitelist is the rule set.
- *Why did your MATH prompt produce empty completions and how did you find it?* Token-boundary
  mismatch between shots and prompt (`:` + newline is one token); found by comparing next-token
  distributions with the HF backend for the two prompt endings. Fix: end the prompt on the shots'
  token. Lesson: check prompt endings at the token level.
- *Why not end a base-model prompt on `<think>`?* An added token the base model has no training
  signal after; prefill through to the first ordinary token.
- *Why keep the stop string in the output?* When the stop string is part of the answer format,
  stripping it makes the format unparseable.
- *What did you learn about determinism?* Section 5.
- *Why Fraction arithmetic in grading?* Exact equality for `0.5` vs `1/2`, no float tolerance to
  tune, and exact division checks for Countdown.

**Environment**
- *Why uv and a uv-managed Python?* No sudo on the box; distro Python lacks pip, venv and headers;
  the managed interpreter is identical on every machine and pinned in the repo.
- *Why did the install fail twice before the smoke run?* CUDA 13 driver requirement for the torch
  that vLLM pins; missing `Python.h` for Triton. Both in `FAILURES_AND_LEARNINGS.md` with the exact
  error text and the check that now catches them in `setup_env.sh`.

---

## 8. Where the time went (for the honest version of "how long did this take")

Environment (driver, memory limits, uv, headers): about half of Phase 0. Writing the harness and
references with tests: a third. Debugging the three smoke-run bugs and EXP-000: the rest. The
reference implementations were the fastest part because each had a closed form to test against;
the environment was the slowest because each failure surfaced only after a multi-minute start-up.

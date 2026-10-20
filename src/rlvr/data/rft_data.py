"""Build the Phase 3 rejection-sampling set from an n-sample run of the ``sft`` checkpoint.

PLAN.md Phase 3 steps 1-2 (EXP-009, EXP-010). Input: ``rlvr-eval`` output directories with one
``samples.jsonl`` per training split (``gsm8k_train``, ``math_train`` levels 1-3, dev problems
excluded, think-tag template, n samples per problem at T = 1.0). Pipeline, every drop counted:

1. **Solve rates.** Per problem: n, c (correct samples), rate = c / n, mean tokens, truncated
   count. Written to ``solve_rate.jsonl`` (the Phase 4 prompt filter reads it) with a histogram.
2. **Select.** Per problem the correct samples that stopped on their own, carry exactly one
   closed think block and one answer block (``format_ok``), deduplicated on the normalised
   reasoning text (Yuan et al.: distinct paths, not raw count), the **shortest** ``per_problem``
   of them kept. Problems with rate 0 contribute nothing.
3. **Format.** The prompt is rebuilt with the think-tag template (byte-identical to what the
   sampler sent, and to the Phase 2 SFT prompt); the completion is the sample text as generated.
   Prompt + completion must fit ``max_tokens`` with the real tokenizer.
4. **Decontaminate** the kept prompts against every eval set with the Phase 2 code and the
   same coverage rule (D3 part a); dropped items counted.
5. **Learnable prompts** for Phase 4: ``0 < rate < 1`` (GRPO wants groups with variance),
   written with question and gold answer and their own content hash.

Outputs under ``outputs/data/rft/``: ``train.jsonl`` (SFT schema, git-ignored),
``manifest.json`` (the trainer checks its hash), ``solve_rate.jsonl``, ``learnable_prompts.jsonl``;
the figure ``outputs/reports/figures/phase-3-solve-rate.png`` and ``outputs/reports/phase-3-data.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from rlvr.data.contamination import normalize_text
from rlvr.data.sft_data import (
    DEFAULT_EVALS,
    DEFAULT_MAX_TOKENS,
    MIN_COVERAGE,
    TOKENIZER,
    SftExample,
    _eval_items,
    _real_token_counter,
    content_hash,
    decontaminate,
    token_stats,
    write_jsonl,
)
from rlvr.eval.datasets import EvalSet, Problem, output_dir
from rlvr.eval.prompts import build_prompt

TRAIN_SPLITS = ("gsm8k_train", "math_train")
THINK_CLOSE = "</think>"
ANSWER_CLOSE = "</answer>"
SOLVE_RATE_FIELDS = ("id", "source", "level", "n", "c", "rate", "mean_tokens", "truncated")


# --------------------------------------------------------------------------- reading samples


def read_samples(path: Path) -> dict[str, list[dict[str, Any]]]:
    """``{problem_id: [sample records in sample_idx order]}`` from one ``samples.jsonl``."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                groups[r["problem_id"]].append(r)
    for recs in groups.values():
        recs.sort(key=lambda r: r.get("sample_idx", 0))
    return dict(groups)


# --------------------------------------------------------------------------- step 1: solve rates


def solve_rates(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    source: str,
    level_by_id: Mapping[str, int | None] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for pid in sorted(groups):
        recs = groups[pid]
        n = len(recs)
        c = sum(1 for r in recs if r.get("correct"))
        rows.append(
            {
                "id": pid,
                "source": source,
                "level": (level_by_id or {}).get(pid),
                "n": n,
                "c": c,
                "rate": c / n if n else 0.0,
                "mean_tokens": round(sum(r.get("num_tokens", 0) for r in recs) / n, 1) if n else 0,
                "truncated": sum(1 for r in recs if r.get("finish_reason") == "length"),
            }
        )
    return rows


def solve_rate_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts per source (and per MATH level) of rate 0, rate 1 and in between."""
    out: dict[str, Any] = {}
    keyed: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        keyed[r["source"]].append(r)
        if r.get("level") is not None:
            keyed[f"{r['source']}/level{r['level']}"].append(r)
    for key, rs in sorted(keyed.items()):
        n = len(rs)
        zero = sum(1 for r in rs if r["c"] == 0)
        full = sum(1 for r in rs if r["c"] == r["n"] and r["n"] > 0)
        out[key] = {
            "problems": n,
            "rate_0": zero,
            "rate_1": full,
            "learnable": n - zero - full,
            "learnable_pct": round(100 * (n - zero - full) / max(1, n), 1),
            "mean_rate": round(sum(r["rate"] for r in rs) / max(1, n), 4),
            "histogram": dict(sorted(Counter(r["c"] for r in rs).items())),
        }
    return out


def learnable_prompts(
    rows: Sequence[Mapping[str, Any]], *, lo: float = 0.0, hi: float = 1.0
) -> list[dict[str, Any]]:
    return [dict(r) for r in rows if lo < r["rate"] < hi]


# --------------------------------------------------------------------------- step 2: selection


def rft_reject_reason(rec: Mapping[str, Any], *, keep_incorrect: bool = False) -> str | None:
    """Why a sample cannot become a training example (None = usable). ``keep_incorrect`` drops
    the correctness filter and nothing else (EXP-010g: selection or self-distillation?)."""
    if not rec.get("correct") and not keep_incorrect:
        return "incorrect"
    if rec.get("finish_reason") != "stop":
        return "truncated"
    text = str(rec.get("text", ""))
    if text.count(THINK_CLOSE) != 1:
        return "think_tags"
    if not rec.get("format_ok", False):
        return "format"
    if not text.rstrip().endswith(ANSWER_CLOSE):
        return "no_answer_close"
    if not str(rec.get("extracted") or "").strip():
        return "no_answer"
    return None


def reasoning_key(text: str) -> str:
    """Dedup key: the normalised think block (distinct reasoning paths, not raw sample count)."""
    return normalize_text(text.split(THINK_CLOSE, 1)[0])


def select_examples(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    problems: Mapping[str, Problem],
    *,
    eval_name: str,
    source: str,
    count_tokens: Callable[[Sequence[str]], list[int]],
    per_problem: int = 2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    drops: Counter | None = None,
    keep_incorrect: bool = False,
    select: str = "shortest",
    select_seed: int = 0,
) -> list[SftExample]:
    """The kept examples for one split: per problem the shortest ``per_problem`` distinct,
    usable samples that fit the cap, in the SFT schema with the think-tag prompt rebuilt.
    ``select='random'`` draws the ``per_problem`` samples at random (seeded per problem) instead
    of shortest first (EXP-010h: how much of the stopping effect is length selection?)."""
    if select not in ("shortest", "random"):
        raise ValueError(f"select must be 'shortest' or 'random', got {select!r}")
    drops = drops if drops is not None else Counter()
    out: list[SftExample] = []
    for pid in sorted(groups):
        if pid not in problems:
            drops["unknown_problem_id"] += len(groups[pid])
            continue
        problem = problems[pid]
        prompt = build_prompt(problem, "think_tag", eval_name).text
        prompt_tokens = count_tokens([prompt])[0]
        usable: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        for rec in groups[pid]:
            why = rft_reject_reason(rec, keep_incorrect=keep_incorrect)
            if why is not None:
                drops[why] += 1
                continue
            key = reasoning_key(str(rec["text"]))
            if key in seen:
                drops["duplicate_reasoning"] += 1
                continue
            seen.add(key)
            usable.append(rec)
        if select == "random":
            # a per-problem seed keeps the draw independent of how many problems precede it
            random.Random(f"{select_seed}:{pid}").shuffle(usable)
        else:
            usable.sort(key=lambda r: (int(r.get("num_tokens", 0)), int(r.get("sample_idx", 0))))
        kept = 0
        for rec in usable:
            if kept >= per_problem:
                drops["over_per_problem_cap"] += 1
                continue
            n_tokens = prompt_tokens + int(rec.get("num_tokens", 0))
            if n_tokens > max_tokens:
                drops["over_max_tokens"] += 1
                continue
            out.append(
                SftExample(
                    id=f"{pid}-s{int(rec.get('sample_idx', 0))}",
                    source=source,
                    problem_type=str(problem.meta.get("subject") or eval_name),
                    prompt=prompt,
                    completion=str(rec["text"]),
                    answer=str(rec.get("extracted") or problem.answer),
                    n_tokens=n_tokens,
                    meta={
                        "problem": problem.question,
                        "problem_id": pid,
                        "level": problem.meta.get("level"),
                        "sample_idx": int(rec.get("sample_idx", 0)),
                        "correct": bool(rec.get("correct")),  # always true unless keep_incorrect
                    },
                )
            )
            kept += 1
    return out


# --------------------------------------------------------------------------- outputs


def write_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def rows_hash(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> str:
    h = hashlib.sha256()
    for r in rows:
        for k in keys:
            h.update(str(r.get(k, "")).encode())
    return h.hexdigest()[:16]


def plot_solve_rate_hist(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = sorted({r["source"] for r in rows})
    n = max((r["n"] for r in rows), default=1)
    fig, axes = plt.subplots(1, max(1, len(sources)), figsize=(4.2 * max(1, len(sources)), 3.4))
    axes = [axes] if len(sources) <= 1 else list(axes)
    for ax, src in zip(axes, sources, strict=False):
        counts = Counter(r["c"] for r in rows if r["source"] == src)
        xs = list(range(n + 1))
        ax.bar(xs, [counts.get(x, 0) for x in xs], color="#4C72B0")
        ax.set_xlabel(f"correct samples of {n}")
        ax.set_ylabel("problems")
        ax.set_title(src)
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def rft_report_markdown(manifest: Mapping[str, Any]) -> str:
    lines = [
        f"# Phase 3 RFT data: {manifest['name']} (built {manifest.get('built_at', '')})",
        "",
        f"Samples from `{manifest['samples_dir']}` (run tag `{manifest.get('samples_run_tag')}`, "
        f"model `{manifest.get('model')}`, n = {manifest.get('n_per_problem')}, "
        f"T = {manifest.get('temperature')}, max_tokens {manifest.get('sampling_max_tokens')}).",
        "",
        "## Solve rates (EXP-009)",
        "",
        "| split | problems | rate 0 | rate 1 | learnable (0 < rate < 1) | mean rate |",
        "|---|---|---|---|---|---|",
    ]
    for key, s in manifest["solve_rates"].items():
        lines.append(
            f"| {key} | {s['problems']} | {s['rate_0']} | {s['rate_1']} | "
            f"{s['learnable']} ({s['learnable_pct']} %) | {s['mean_rate']:.3f} |"
        )
    lines += [
        "",
        f"Learnable prompts written for Phase 4: {manifest['num_learnable']} "
        f"(hash {manifest['learnable_hash']}).",
        "",
        "## Kept set (EXP-010 training data)",
        "",
        f"- examples: {manifest['num_examples']} (at most {manifest['per_problem']} per problem, "
        f"shortest distinct correct samples), content hash `{manifest['content_hash']}`",
        f"- by split: {manifest['by_source']}",
        f"- drops by reason: {manifest['drops']}",
        f"- prompt + completion tokens: {manifest['kept_token_stats']}",
        "",
        "## Decontamination (D3 a)",
        "",
        "| eval | hits (any n-gram) | covered | fuzzy | contaminated | train items dropped |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in manifest["overlap"].items():
        if name.startswith("_"):
            continue
        lines.append(
            f"| {name} | {s['eval_hit_ngram']} | {s['eval_hit_ngram_covered']} | {s['eval_hit_fuzzy']} | "
            f"{s['eval_contaminated']} | {s['train_items_dropped']} |"
        )
    total = manifest["overlap"].get("_total", {})
    lines += ["", f"Dropped in total: {total.get('train_items_dropped_union', 0)}."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- the build


def build(
    samples_dir: Path,
    *,
    splits: Sequence[str],
    problems_by_split: Mapping[str, Mapping[str, Problem]],
    count_tokens: Callable[[Sequence[str]], list[int]],
    eval_items: Mapping[str, Mapping[str, str]],
    out_dir: Path,
    figure: Path | None,
    report: Path | None,
    per_problem: int = 2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    run_fuzzy: bool = True,
    min_coverage: float | None = MIN_COVERAGE,
    name: str = "rft_iter1",
    keep_incorrect: bool = False,
    select: str = "shortest",
    select_seed: int = 0,
) -> dict[str, Any]:
    t0 = time.time()
    rate_rows: list[dict[str, Any]] = []
    examples: list[SftExample] = []
    drops: Counter = Counter()
    for split in splits:
        path = samples_dir / split / "samples.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path}: no samples for split {split!r}")
        groups = read_samples(path)
        problems = problems_by_split[split]
        levels = {pid: p.meta.get("level") for pid, p in problems.items()}
        rate_rows += solve_rates(groups, split, levels)
        examples += select_examples(
            groups,
            problems,
            eval_name=split,
            source=split,
            count_tokens=count_tokens,
            per_problem=per_problem,
            max_tokens=max_tokens,
            drops=drops,
            keep_incorrect=keep_incorrect,
            select=select,
            select_seed=select_seed,
        )
    if not examples:
        raise ValueError(f"no usable examples under {samples_dir}; drops: {dict(drops)}")
    examples, overlap = decontaminate(
        examples,
        eval_items,
        report_dir=out_dir / "overlap",
        run_fuzzy=run_fuzzy,
        min_coverage=min_coverage,
    )
    learnable = []
    for r in learnable_prompts(rate_rows):
        p = problems_by_split[r["source"]][r["id"]]
        learnable.append(
            {
                "id": r["id"],
                "source": r["source"],
                "question": p.question,
                "answer": p.answer,
                "level": r["level"],
                "rate": r["rate"],
            }
        )
    write_jsonl(examples, out_dir / "train.jsonl")
    write_rows(rate_rows, out_dir / "solve_rate.jsonl")
    write_rows(learnable, out_dir / "learnable_prompts.jsonl")
    if figure is not None:
        plot_solve_rate_hist(rate_rows, figure)
    run_info = _read_json(samples_dir / "run_info.json")
    cfg = run_info.get("config") or {}
    manifest: dict[str, Any] = {
        "name": name,
        "samples_dir": str(samples_dir),
        "samples_run_tag": run_info.get("run_tag"),
        "samples_git_commit": run_info.get("git_commit"),
        "model": cfg.get("model"),
        "n_per_problem": cfg.get("n"),
        "temperature": cfg.get("temperature"),
        "sampling_max_tokens": cfg.get("max_tokens"),
        "splits": list(splits),
        "per_problem": per_problem,
        "max_tokens": max_tokens,
        "evals_decontaminated": sorted(eval_items),
        "fuzzy": run_fuzzy,
        "min_coverage": min_coverage,
        "keep_incorrect": keep_incorrect,
        "select": select,
        "select_seed": select_seed,
        "fraction_incorrect": round(
            sum(1 for ex in examples if not ex.meta.get("correct", True)) / len(examples), 4
        ),
        "num_examples": len(examples),
        "content_hash": content_hash(examples),
        "kept_token_stats": token_stats([ex.n_tokens for ex in examples]),
        "by_source": dict(Counter(ex.source for ex in examples)),
        "drops": dict(drops),
        "solve_rates": solve_rate_summary(rate_rows),
        "num_problems": len(rate_rows),
        "num_learnable": len(learnable),
        "learnable_hash": rows_hash(learnable, ("id", "question", "answer")),
        "overlap": overlap,
        "built_in_s": round(time.time() - t0, 1),
        "built_at": time.strftime("%Y-%m-%d %H:%M"),
        "train_file": str(out_dir / "train.jsonl"),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\n"
    )
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(rft_report_markdown(manifest), encoding="utf-8", newline="\n")
    return manifest


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _problems_from_loaders(splits: Sequence[str]) -> dict[str, dict[str, Problem]]:
    """Full train splits by id (ids are stable, so the filtered sampling pool maps back)."""
    from rlvr.eval.datasets import load_eval_set

    out: dict[str, dict[str, Problem]] = {}
    for split in splits:
        es: EvalSet = load_eval_set(split)
        out[split] = {p.id: p for p in es.problems}
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build the Phase 3 RFT set from sampled traces.")
    p.add_argument("--samples", type=Path, default=Path("outputs/samples/sft_1.7b_rft_n8"))
    p.add_argument("--splits", nargs="+", default=list(TRAIN_SPLITS))
    p.add_argument("--per-problem", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--evals", nargs="+", default=list(DEFAULT_EVALS))
    p.add_argument("--no-fuzzy", action="store_true")
    p.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    p.add_argument("--name", default="rft_iter1")
    p.add_argument(
        "--keep-incorrect",
        action="store_true",
        help="EXP-010g: keep finished, well-formed samples whether or not they are correct",
    )
    p.add_argument(
        "--per-problem-select",
        choices=("shortest", "random"),
        default="shortest",
        help="EXP-010h: which usable samples to keep per problem (default: the shortest)",
    )
    p.add_argument("--select-seed", type=int, default=0, help="seed of --per-problem-select random")
    p.add_argument("--out", type=Path, default=None, help="default outputs/data/rft")
    p.add_argument(
        "--figure", type=Path, default=Path("outputs/reports/figures/phase-3-solve-rate.png")
    )
    p.add_argument("--report", type=Path, default=Path("outputs/reports/phase-3-data.md"))
    args = p.parse_args(argv)
    out_dir = args.out or (output_dir() / "data" / "rft")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rft] samples {args.samples}, splits {args.splits}")
    problems = _problems_from_loaders(args.splits)
    eval_items = _eval_items(args.evals)
    manifest = build(
        args.samples,
        splits=args.splits,
        problems_by_split=problems,
        count_tokens=_real_token_counter(args.tokenizer),
        eval_items=eval_items,
        out_dir=out_dir,
        figure=args.figure,
        report=args.report,
        per_problem=args.per_problem,
        max_tokens=args.max_tokens,
        run_fuzzy=not args.no_fuzzy,
        min_coverage=args.min_coverage,
        name=args.name,
        keep_incorrect=args.keep_incorrect,
        select=args.per_problem_select,
        select_seed=args.select_seed,
    )
    for key, s in manifest["solve_rates"].items():
        print(f"[rft] {key}: {s}")
    print(
        f"[done] {manifest['num_examples']} examples, hash {manifest['content_hash']}, "
        f"{manifest['num_learnable']} learnable prompts, drops {manifest['drops']}, "
        f"{manifest['built_in_s']} s -> {out_dir}"
    )


if __name__ == "__main__":
    main()

"""Build the Phase 2 SFT set from OpenR1-Math-220k (PLAN.md Phase 2 step 1; EXP-004, EXP-006).

Pipeline, in this order, every drop counted by reason:

1. **Select** one DeepSeek-R1 generation per problem: the first one that ``math-verify`` judged
   correct (the dataset's ``correctness_math_verify`` column).
2. **Parse** it: ``<think> reasoning </think> summary`` where the summary ends in exactly one
   ``\\boxed{}``. Generations without a closing think tag, without a box, or with several boxes
   (multi-part answers, which our grader cannot score) are dropped.
3. **Rewrite** into the project's format. The *prompt* is the think-tag template used by every
   eval and by RL later (``rlvr.eval.prompts``), so it ends ``Assistant:\\n<think>\\n``; the
   *completion* is ``reasoning + summary`` inside the think block and the bare boxed value in
   ``<answer> ... </answer>``. One format for SFT, eval and RL means no template drift.
4. **Length cap**: prompt + completion at most ``max_tokens`` Qwen3 tokens (4096), measured
   with the real tokenizer, so nothing is truncated mid-answer at train time.
5. **Dedup** on the normalised problem text (keep the first).
6. **Decontaminate** against every eval set and the dev split with the 13-gram and fuzzy
   detectors from :mod:`rlvr.data.contamination`; every hit item is dropped (conservative),
   while the *reported* contamination counts only 13-gram hits that cover at least
   ``MIN_COVERAGE`` of the eval problem, plus fuzzy matches, because template sentences such
   as "where m and n are relatively prime positive integers, find m + n" are 13-grams too
   (EXP-004 / D3 part a).

Ids: the source uuid, made unique (22 rows carry the uuid "NaN" and some uuids repeat), so
the trainer's id map and the manifest hash never collapse two problems into one.

Outputs: ``outputs/data/sft/train.jsonl`` (git-ignored), ``data/manifests/sft_openr1.json``
(committed: revision, filters, counts, token statistics, content hash), the per-eval overlap
reports under ``outputs/reports/phase-2-overlap/`` and a Markdown data report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rlvr.data.contamination import normalize_text, overlap_report, write_report
from rlvr.eval.datasets import Problem, output_dir
from rlvr.eval.extraction import last_boxed
from rlvr.eval.prompts import build_prompt

SOURCE = "open-r1/OpenR1-Math-220k"
SOURCE_CONFIG = "default"
DEFAULT_MAX_TOKENS = 4096
TOKENIZER = "Qwen/Qwen3-1.7B-Base"  # shared by the 0.6B student
MIN_COVERAGE = 0.5  # 13-gram hit counts as a copy when it covers half the eval problem
MISSING_IDS = {"", "nan", "none", "null"}
DEFAULT_EVALS = (
    "gsm8k",
    "math500",
    "aime24",
    "gsm8k_perturbed",
    "gsm8k_names_only",
    "gsm8k_dev",
    "math_dev",
)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_BOXED = re.compile(r"\\(?:boxed|fbox)\s*\{")


# --------------------------------------------------------------------------- parsing one trace


@dataclass(frozen=True)
class ParsedTrace:
    reasoning: str
    summary: str
    final: str


def count_boxed(text: str) -> int:
    return len(_BOXED.findall(text))


def parse_generation(text: str) -> ParsedTrace | None:
    """Split an R1 generation into reasoning, summary and the single boxed final answer.

    Returns None (caller counts the reason via :func:`reject_reason`) when the trace has no
    closing think tag, no summary, no box or more than one box in the summary.
    """
    reason = reject_reason(text)
    if reason is not None:
        return None
    body = text.strip()
    if body.startswith(_THINK_OPEN):
        body = body[len(_THINK_OPEN) :]
    reasoning, summary = body.split(_THINK_CLOSE, 1)
    final = last_boxed(summary)
    assert final is not None
    return ParsedTrace(reasoning.strip(), summary.strip(), final)


def reject_reason(text: str) -> str | None:
    body = text.strip()
    if body.count(_THINK_CLOSE) != 1:
        return "no_single_think_close"
    if body.count(_THINK_OPEN) > 1:
        return "nested_think"
    _, summary = body.split(_THINK_CLOSE, 1)
    if not summary.strip():
        return "empty_summary"
    n = count_boxed(summary)
    if n == 0:
        return "no_boxed_in_summary"
    if n > 1:
        return "multiple_boxed"
    final = last_boxed(summary)
    if final is None or not final.strip():
        return "unbalanced_boxed"
    return None


# --------------------------------------------------------------------------- one SFT example


@dataclass
class SftExample:
    id: str
    source: str
    problem_type: str
    prompt: str
    completion: str
    answer: str
    n_tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


def sft_prompt(problem_text: str) -> str:
    """The think-tag prompt, byte-identical to what ``rlvr-eval --template think_tag`` sends
    for a MATH-style problem, so SFT, evaluation and RL share one template."""
    return build_prompt(Problem("sft", problem_text, ""), "think_tag", "math500").text


def sft_completion(trace: ParsedTrace) -> str:
    think_body = trace.reasoning
    if trace.summary:
        think_body = f"{think_body}\n\n{trace.summary}"
    return f"{think_body}\n{_THINK_CLOSE}\n<answer> {trace.final} </answer>"


def build_example(
    *,
    uid: str,
    problem_text: str,
    trace: ParsedTrace,
    source: str,
    problem_type: str,
    n_tokens: int = 0,
) -> SftExample:
    prompt = sft_prompt(problem_text)
    completion = sft_completion(trace)
    return SftExample(
        id=uid,
        source=source,
        problem_type=problem_type,
        prompt=prompt,
        completion=completion,
        answer=trace.final,
        n_tokens=n_tokens,
    )


# --------------------------------------------------------------------------- selection pass


@dataclass
class BuildStats:
    rows_seen: int = 0
    kept: int = 0
    drops: Counter = field(default_factory=Counter)
    # token lengths of every parsed candidate *before* the cap, to show what the cap removes
    candidate_tokens: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = {"rows_seen": self.rows_seen, "kept": self.kept, "drops": dict(self.drops)}
        if self.candidate_tokens:
            d["candidate_token_stats"] = token_stats(self.candidate_tokens)
        return d


def token_stats(lengths: Sequence[int]) -> dict[str, float]:
    xs = sorted(lengths)
    if not xs:
        return {}
    q = lambda p: xs[min(len(xs) - 1, int(p * len(xs)))]  # noqa: E731
    return {
        "n": len(xs),
        "mean": round(statistics.fmean(xs), 1),
        "p50": q(0.50),
        "p90": q(0.90),
        "p99": q(0.99),
        "max": xs[-1],
        "total": int(sum(xs)),
    }


def pick_correct_generation(row: Mapping[str, Any]) -> tuple[int, str] | None:
    gens = row.get("generations") or []
    flags = row.get("correctness_math_verify") or []
    for i, (g, ok) in enumerate(zip(gens, flags, strict=False)):
        if ok:
            return i, g
    return None


def unique_id(raw: Any, row_index: int, seen: set[str]) -> str:
    """The source uuid when it is present and new; otherwise ``<uuid or row>#<row index>``."""
    uid = str(raw).strip() if raw is not None else ""
    if uid.lower() in MISSING_IDS:
        uid = f"row#{row_index}"
    elif uid in seen:
        uid = f"{uid}#{row_index}"
    seen.add(uid)
    return uid


def count_tokens_batched(
    texts: Sequence[str], count_many: Callable[[Sequence[str]], list[int]], batch_size: int
) -> list[int]:
    out: list[int] = []
    for i in range(0, len(texts), batch_size):
        out.extend(count_many(texts[i : i + batch_size]))
    assert len(out) == len(texts)
    return out


def select_examples(
    rows: Iterable[Mapping[str, Any]],
    *,
    count_tokens: Callable[[Sequence[str]], list[int]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    stats: BuildStats | None = None,
    batch_size: int = 512,
) -> list[SftExample]:
    """Steps 1-5: select, parse, rewrite, dedup, then count tokens in batches and cap.
    Deterministic in the row order. ``count_tokens`` maps a list of texts to their lengths."""
    stats = stats if stats is not None else BuildStats()
    seen_problems: set[str] = set()
    seen_ids: set[str] = set()
    candidates: list[SftExample] = []
    for row in rows:
        row_index = stats.rows_seen
        stats.rows_seen += 1
        problem = str(row.get("problem") or "").strip()
        if not problem:
            stats.drops["empty_problem"] += 1
            continue
        picked = pick_correct_generation(row)
        if picked is None:
            stats.drops["no_correct_generation"] += 1
            continue
        _, gen = picked
        reason = reject_reason(gen)
        if reason is not None:
            stats.drops[reason] += 1
            continue
        trace = parse_generation(gen)
        assert trace is not None
        key = normalize_text(problem)
        if key in seen_problems:
            stats.drops["duplicate_problem"] += 1
            continue
        seen_problems.add(key)
        ex = build_example(
            uid=unique_id(row.get("uuid"), row_index, seen_ids),
            problem_text=problem,
            trace=trace,
            source=str(row.get("source") or ""),
            problem_type=str(row.get("problem_type") or ""),
        )
        ex.meta = {"problem": problem}
        candidates.append(ex)

    prompt_lens = count_tokens_batched([ex.prompt for ex in candidates], count_tokens, batch_size)
    completion_lens = count_tokens_batched(
        [ex.completion for ex in candidates], count_tokens, batch_size
    )
    out: list[SftExample] = []
    for ex, lp, lc in zip(candidates, prompt_lens, completion_lens, strict=True):
        ex.n_tokens = lp + lc
        stats.candidate_tokens.append(ex.n_tokens)
        if ex.n_tokens > max_tokens:
            stats.drops["too_long"] += 1
            continue
        out.append(ex)
    stats.kept = len(out)
    return out


# --------------------------------------------------------------------------- decontamination


def decontaminate(
    examples: list[SftExample],
    eval_items_by_name: Mapping[str, Mapping[str, str]],
    *,
    report_dir: Path | None,
    n: int = 13,
    fuzzy_threshold: float = 0.9,
    run_fuzzy: bool = True,
    min_coverage: float | None = MIN_COVERAGE,
) -> tuple[list[SftExample], dict[str, Any]]:
    """Step 6 (EXP-004): drop every training item that hits any eval problem by any detector;
    report per eval set the n-gram hits (any, and those covering at least ``min_coverage`` of
    the problem), the fuzzy hits, and ``eval_contaminated`` = covered n-gram or fuzzy, the
    number reported as present in the training data."""
    train_items = {ex.id: ex.meta.get("problem", "") for ex in examples}
    summary: dict[str, Any] = {}
    to_drop: set[str] = set()
    for name, eval_items in eval_items_by_name.items():
        t0 = time.time()
        rep = overlap_report(
            name,
            dict(eval_items),
            train_items,
            n=n,
            fuzzy_threshold=fuzzy_threshold,
            run_fuzzy=run_fuzzy,
            min_coverage=min_coverage,
        )
        hit_ids = set(rep.ngram_hits) | set(rep.fuzzy_hits)
        contaminated = rep.contaminated_eval_ids
        fuzzy_only = set(rep.fuzzy_hits) - set(rep.ngram_hits)
        summary[name] = {
            "num_eval": rep.num_eval,
            "eval_hit_ngram": len(rep.ngram_hits),
            "eval_hit_ngram_covered": None
            if rep.ngram_hits_covered is None
            else len(rep.ngram_hits_covered),
            "eval_hit_fuzzy": len(rep.fuzzy_hits),
            "eval_hit_fuzzy_only": len(fuzzy_only),
            "eval_hit_union": len(hit_ids),
            "eval_hit_union_pct": round(100 * len(hit_ids) / max(1, rep.num_eval), 2),
            "eval_contaminated": len(contaminated),
            "eval_contaminated_pct": round(100 * len(contaminated) / max(1, rep.num_eval), 2),
            "train_items_dropped": len(rep.train_ids_to_drop),
            "seconds": round(time.time() - t0, 1),
        }
        to_drop |= rep.train_ids_to_drop
        if report_dir is not None:
            write_report(rep, report_dir / f"sft_openr1__{name}.json", examples=25)
    kept = [ex for ex in examples if ex.id not in to_drop]
    summary["_total"] = {
        "train_items_dropped_union": len(to_drop),
        "kept_after_decontamination": len(kept),
    }
    return kept, summary


# --------------------------------------------------------------------------- outputs


def content_hash(examples: Sequence[SftExample]) -> str:
    h = hashlib.sha256()
    for ex in examples:
        h.update(ex.id.encode())
        h.update(ex.prompt.encode())
        h.update(ex.completion.encode())
    return h.hexdigest()[:16]


def write_jsonl(examples: Sequence[SftExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for ex in examples:
            row = asdict(ex)
            row.pop("meta", None)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def data_report_markdown(manifest: Mapping[str, Any], overlap: Mapping[str, Any]) -> str:
    st = manifest["stats"]
    kept = manifest["kept_token_stats"]
    cand = st.get("candidate_token_stats", {})
    lines = [
        "# Phase 2 SFT data report",
        "",
        f"Source `{manifest['source']}` config `{manifest['config']}` revision "
        f"`{manifest['revision']}`; tokenizer `{manifest['tokenizer']}`; cap "
        f"{manifest['max_tokens']} tokens; built at commit `{manifest.get('git_commit')}`.",
        "",
        "## Funnel",
        "",
        "| stage | count |",
        "|---|---|",
        f"| rows in source | {st['rows_seen']} |",
    ]
    for k, v in sorted(st["drops"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| dropped: {k} | {v} |")
    lines.append(f"| kept before decontamination | {st['kept']} |")
    tot = overlap.get("_total", {})
    lines.append(f"| dropped: eval overlap (union) | {tot.get('train_items_dropped_union', 0)} |")
    lines.append(f"| **kept** | **{manifest['num_examples']}** |")
    lines += [
        "",
        "## Length (prompt + completion, Qwen3 tokens)",
        "",
        "| set | n | mean | p50 | p90 | p99 | max | total |",
        "|---|---|---|---|---|---|---|---|",
    ]
    if cand:
        lines.append(
            f"| all parsed candidates (before the cap) | {cand['n']} | {cand['mean']} | "
            f"{cand['p50']} | {cand['p90']} | {cand['p99']} | {cand['max']} | {cand['total']} |"
        )
    lines.append(
        f"| kept | {kept['n']} | {kept['mean']} | {kept['p50']} | {kept['p90']} | "
        f"{kept['p99']} | {kept['max']} | {kept['total']} |"
    )
    lines += [
        "",
        "## Training-data overlap (EXP-004, D3 part a)",
        "",
        "| eval set | problems | any 13-gram | 13-gram covering >= "
        f"{int(100 * MIN_COVERAGE)} % | fuzzy >= 0.9 | fuzzy only | **contaminated** (covered or fuzzy) | % | "
        "train items dropped (any hit) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in overlap.items():
        if name.startswith("_"):
            continue
        lines.append(
            f"| {name} | {s['num_eval']} | {s['eval_hit_ngram']} | {s['eval_hit_ngram_covered']} | "
            f"{s['eval_hit_fuzzy']} | {s['eval_hit_fuzzy_only']} | **{s['eval_contaminated']}** | "
            f"{s['eval_contaminated_pct']} | {s['train_items_dropped']} |"
        )
    lines += [
        "",
        "A 13-gram hit whose shared windows cover less than half of the eval problem is a template "
        "sentence (AIME's 'relatively prime positive integers, find m + n'), not a copy; those items "
        "are still dropped from training but not counted as contamination. Per-eval JSON with the "
        "hit ids: `outputs/reports/phase-2-overlap/`.",
        "",
        "## Sources and problem types kept",
        "",
        "| source | kept |",
        "|---|---|",
    ]
    for k, v in sorted(manifest["by_source"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    lines += ["", "| problem type | kept |", "|---|---|"]
    for k, v in sorted(manifest["by_problem_type"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    if "answer_check" in manifest:
        ac = manifest["answer_check"]
        lines += [
            "",
            "## Answer check",
            "",
            f"On a seeded sample of {ac['n']} kept examples, our extracted `<answer>` value agrees "
            f"with the source's `answer` field under math-verify in {ac['agree']} cases "
            f"({ac['agree_pct']} %); disagreements are listed in the manifest for reading by eye.",
        ]
    return "\n".join(lines) + "\n"


def plot_length_hist(lengths: Sequence[int], cap: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(lengths, bins=60, color="#4C72B0")
    ax.axvline(cap, color="#C44E52", ls="--", label=f"cap {cap}")
    ax.set_xlabel("prompt + completion tokens (all parsed candidates)")
    ax.set_ylabel("problems")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- CLI


def _real_token_counter(name: str) -> Callable[[Sequence[str]], list[int]]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    return lambda texts: [
        len(ids) for ids in tok(list(texts), add_special_tokens=False)["input_ids"]
    ]


def _eval_items(names: Sequence[str]) -> dict[str, dict[str, str]]:
    from rlvr.eval.datasets import load_eval_set

    out: dict[str, dict[str, str]] = {}
    for name in names:
        try:
            es = load_eval_set(name)
        except FileNotFoundError as e:  # perturbed / names-only files not generated on this box
            print(f"[overlap] skipping {name}: {e}")
            continue
        out[name] = {p.id: p.question for p in es.problems}
    return out


def _answer_check(examples: Sequence[SftExample], gold: Mapping[str, str], n: int, seed: int):
    from rlvr.eval.extraction import _math_verify_equal

    rng = random.Random(seed)
    sample = rng.sample(list(examples), min(n, len(examples)))
    agree, disagreements = 0, []
    for ex in sample:
        g = gold.get(ex.id, "")
        ok = _math_verify_equal(ex.answer, g)
        if ok:
            agree += 1
        elif len(disagreements) < 40:
            disagreements.append({"id": ex.id, "ours": ex.answer, "source_answer": g})
    return {
        "n": len(sample),
        "agree": agree,
        "agree_pct": round(100 * agree / max(1, len(sample)), 1),
        "disagreements": disagreements,
    }


def same_data_as(manifest_path: Path, manifest: dict[str, Any]) -> bool:
    """True when the manifest on disk describes the same examples (hash and count) as ``manifest``."""
    if not manifest_path.exists():
        return False
    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return existing.get("content_hash") == manifest.get("content_hash") and existing.get(
        "num_examples"
    ) == manifest.get("num_examples")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build the Phase 2 SFT set from OpenR1-Math-220k.")
    p.add_argument("--revision", default=None, help="dataset revision (commit sha) to pin")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--limit", type=int, default=None, help="rows to read (smoke runs)")
    p.add_argument("--evals", nargs="+", default=list(DEFAULT_EVALS))
    p.add_argument("--no-fuzzy", action="store_true")
    p.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    p.add_argument("--verify-sample", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="default outputs/data/sft")
    p.add_argument("--manifest", type=Path, default=Path("data/manifests/sft_openr1.json"))
    p.add_argument(
        "--report", type=Path, default=Path("outputs/reports/phase-2-data.md"), help="markdown"
    )
    args = p.parse_args(argv)

    from datasets import load_dataset
    from huggingface_hub import dataset_info

    from rlvr.utils.run_info import git_commit

    t0 = time.time()
    out_dir = args.out or (output_dir() / "data" / "sft")
    revision = args.revision or dataset_info(SOURCE).sha
    print(f"[load] {SOURCE}:{SOURCE_CONFIG} @ {revision}")
    ds = load_dataset(SOURCE, SOURCE_CONFIG, split="train", revision=revision)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    count_tokens = _real_token_counter(args.tokenizer)

    stats = BuildStats()
    print("[select] parsing, rewriting, counting tokens ...")
    examples = select_examples(
        ds, count_tokens=count_tokens, max_tokens=args.max_tokens, stats=stats
    )
    print(f"[select] kept {stats.kept} of {stats.rows_seen}; drops {dict(stats.drops)}")

    report_dir = output_dir() / "reports" / "phase-2-overlap"
    eval_items = _eval_items(args.evals)
    print(f"[overlap] {len(examples)} items vs {list(eval_items)}")
    examples, overlap = decontaminate(
        examples,
        eval_items,
        report_dir=report_dir,
        run_fuzzy=not args.no_fuzzy,
        min_coverage=args.min_coverage,
    )
    for name, s in overlap.items():
        print(f"[overlap] {name}: {s}")

    gold: dict[str, str] = {}
    if args.verify_sample:
        seen_ids: set[str] = set()
        for i, r in enumerate(ds):
            gold[unique_id(r.get("uuid"), i, seen_ids)] = str(r["answer"])
    answer_check = (
        _answer_check(examples, gold, args.verify_sample, args.seed) if args.verify_sample else None
    )

    write_jsonl(examples, out_dir / "train.jsonl")
    lengths = [ex.n_tokens for ex in examples]
    manifest: dict[str, Any] = {
        "name": "sft_openr1",
        "source": SOURCE,
        "config": SOURCE_CONFIG,
        "revision": revision,
        "tokenizer": args.tokenizer,
        "max_tokens": args.max_tokens,
        "limit": args.limit,
        "evals_decontaminated": sorted(eval_items),
        "fuzzy": not args.no_fuzzy,
        "min_coverage": args.min_coverage,
        "num_examples": len(examples),
        "content_hash": content_hash(examples),
        "kept_token_stats": token_stats(lengths),
        "stats": stats.as_dict(),
        "overlap": overlap,
        "by_source": dict(Counter(ex.source for ex in examples)),
        "by_problem_type": dict(Counter(ex.problem_type for ex in examples)),
        "git_commit": git_commit(),
        "built_in_s": round(time.time() - t0, 1),
        "train_file": str(out_dir / "train.jsonl"),
    }
    if answer_check:
        manifest["answer_check"] = answer_check
    if same_data_as(args.manifest, manifest):
        # A rebuild on a box that reproduces the committed set must not touch the tracked manifest,
        # report or figure: the timings and commit inside them differ, the tree goes dirty, and the
        # harness then refuses every run (first E2E node, 17 Oct 2026). This build's own manifest
        # lands next to the data instead.
        (out_dir / "rebuild_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8", newline="\n"
        )
        print(
            f"[rlvr-build-sft] {args.manifest} already describes this data "
            f"(hash {manifest['content_hash']}); tracked files left untouched, "
            f"this build's manifest at {out_dir / 'rebuild_manifest.json'}"
        )
    else:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            data_report_markdown(manifest, overlap), encoding="utf-8", newline="\n"
        )
        if stats.candidate_tokens:
            plot_length_hist(
                stats.candidate_tokens,
                args.max_tokens,
                args.report.parent / "figures" / "sft_length_hist.png",
            )
    print(
        f"[done] {len(examples)} examples, hash {manifest['content_hash']}, "
        f"{manifest['kept_token_stats'].get('total', 0)} tokens, {manifest['built_in_s']} s"
    )


if __name__ == "__main__":
    main()

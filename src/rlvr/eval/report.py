"""Turn graded samples into the D1 report: per-problem rows, dataset metrics with CIs,
breakdowns (MATH level, extraction source), and a Markdown table.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rlvr.eval.datasets import EvalSet
from rlvr.eval.extraction import (
    Extraction,
    extract_answer,
    grade,
    normalize_for_vote,
    tag_format_ok,
)
from rlvr.eval.metrics import bootstrap_ci, length_stats, maj_at_k, pass_at_k_curve
from rlvr.eval.sampler import Completion

DEFAULT_KS = (1, 2, 4, 8, 16, 32, 64)


@dataclass
class SampleRecord:
    problem_id: str
    sample_idx: int
    text: str
    num_tokens: int
    finish_reason: str
    extracted: str | None
    extraction_source: str
    correct: bool
    format_ok: bool
    forced_stop: bool = False  # budget forcing (EXP-016); absent in files written before it


@dataclass
class ProblemResult:
    problem_id: str
    gold: str
    meta: dict[str, Any]
    samples: list[SampleRecord] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def c(self) -> int:
        return sum(s.correct for s in self.samples)


def grade_completions(
    es: EvalSet, completions: list[list[Completion]], *, completion_prefix: str = ""
) -> list[ProblemResult]:
    results: list[ProblemResult] = []
    for problem, comps in zip(es.problems, completions, strict=True):
        pr = ProblemResult(problem.id, problem.answer, problem.meta)
        for j, comp in enumerate(comps):
            full_text = completion_prefix + comp.text
            ext: Extraction = extract_answer(full_text, es.answer_type)
            ok = grade(ext.answer, problem.answer, es.answer_type, meta=problem.meta)
            pr.samples.append(
                SampleRecord(
                    problem_id=problem.id,
                    sample_idx=j,
                    text=comp.text,
                    num_tokens=comp.num_tokens,
                    finish_reason=comp.finish_reason,
                    extracted=ext.answer,
                    extraction_source=ext.source,
                    correct=ok,
                    format_ok=tag_format_ok(full_text),
                    forced_stop=bool(getattr(comp, "forced", False)),
                )
            )
        results.append(pr)
    return results


def _breakdown(results: list[ProblemResult], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for pr in results:
        g = pr.meta.get(key)
        if g is None:
            continue
        groups[str(g)].append(pr.samples[0].correct if pr.samples else 0.0)
    return {g: bootstrap_ci(v).as_dict() for g, v in sorted(groups.items())}


def summarize(results: list[ProblemResult], *, ks=DEFAULT_KS) -> dict[str, Any]:
    """Dataset-level metrics. pass@1 here is the first sample's accuracy (greedy when T=0)."""
    if not results:
        raise ValueError("no results")
    n = results[0].n
    if any(pr.n != n for pr in results):
        raise ValueError("all problems must have the same number of samples")

    first_correct = [float(pr.samples[0].correct) for pr in results]
    all_samples = [s for pr in results for s in pr.samples]
    out: dict[str, Any] = {
        "num_problems": len(results),
        "num_samples_per_problem": n,
        "pass@1": bootstrap_ci(first_correct).as_dict(),
        "mean_sample_accuracy": bootstrap_ci([pr.c / pr.n for pr in results]).as_dict(),
        "truncation_rate": sum(s.finish_reason == "length" for s in all_samples) / len(all_samples),
        "extraction_failure_rate": sum(s.extracted is None for s in all_samples) / len(all_samples),
        "tag_format_rate": sum(s.format_ok for s in all_samples) / len(all_samples),
        "extraction_sources": dict(Counter(s.extraction_source for s in all_samples)),
        "tokens": length_stats([s.num_tokens for s in all_samples]),
        "tokens_correct": length_stats([s.num_tokens for s in all_samples if s.correct]),
        "tokens_incorrect": length_stats([s.num_tokens for s in all_samples if not s.correct]),
    }
    forced = [s for s in all_samples if getattr(s, "forced_stop", False)]
    out["forced_stop_rate"] = len(forced) / len(all_samples)
    if forced:
        out["forced_stop_accuracy"] = sum(s.correct for s in forced) / len(forced)
        out["forced_stop_finished_rate"] = sum(s.finish_reason != "length" for s in forced) / len(
            forced
        )
    if n > 1:
        curve = pass_at_k_curve([pr.c for pr in results], n, ks)
        out["pass@k"] = {str(k): v.as_dict() for k, v in curve.items()}
        answers = [[s.extracted for s in pr.samples] for pr in results]
        correct = [[s.correct for s in pr.samples] for pr in results]
        out["maj@k"] = {
            str(k): maj_at_k(answers, correct, k, normalize=normalize_for_vote)
            for k in ks
            if k <= n
        }
    level = _breakdown(results, "level")
    if level:
        out["by_level"] = level
    return out


def write_samples_jsonl(results: list[ProblemResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pr in results:
            for s in pr.samples:
                f.write(
                    json.dumps({**asdict(s), "gold": pr.gold, "meta": pr.meta}, ensure_ascii=False)
                    + "\n"
                )


def markdown_row(label: str, summary: dict[str, Any]) -> str:
    def fmt(est: dict[str, Any]) -> str:
        return f"{100 * est['mean']:.1f} [{100 * est['lo']:.1f}, {100 * est['hi']:.1f}]"

    p1 = fmt(summary["pass@1"])
    pk = summary.get("pass@k", {})
    top_k = max((int(k) for k in pk), default=None)
    pk_s = fmt(pk[str(top_k)]) if top_k else "-"
    mk = summary.get("maj@k", {})
    mk_s = f"{100 * mk[str(top_k)]:.1f}" if top_k and str(top_k) in mk else "-"
    return (
        f"| {label} | {summary['num_problems']} | {p1} | {pk_s} (k={top_k}) | {mk_s} | "
        f"{100 * summary['truncation_rate']:.1f} | {100 * summary['extraction_failure_rate']:.1f} | "
        f"{summary['tokens']['mean']:.0f} / {summary['tokens']['p90']:.0f} |"
    )


MARKDOWN_HEADER = (
    "| eval | n | pass@1 [95% CI] | pass@k [95% CI] | maj@k | trunc % | extract-fail % | tokens mean / p90 |\n"
    "|---|---|---|---|---|---|---|---|"
)

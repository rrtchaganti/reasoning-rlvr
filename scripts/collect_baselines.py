"""Collect EXP-001 greedy baseline runs into the D2 tables.

Reads ``outputs/eval/base_<model>_<template>/metrics.json`` (one file per run, keyed by eval) and
writes two Markdown tables plus a JSON summary:

1. pass@1 [95 % CI] per (model, template) x eval, with extraction-failure and truncation rates in
   a second table, because a low number with 30 % extraction failures is a different finding from
   a low number with 0 %.
2. MATH-500 accuracy by level (1-5) per (model, template).

Usage:
    python scripts/collect_baselines.py outputs/eval --out outputs/reports/phase-1-baselines.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RUN_RE = re.compile(
    r"^base_(?P<model>[0-9.]+b)_(?P<template>zero_shot|few_shot_cot|few_shot|think_tag)$"
)
EVAL_ORDER = ("gsm8k", "gsm8k_perturbed", "math500", "countdown", "aime24")
TEMPLATE_ORDER = ("zero_shot", "few_shot", "few_shot_cot", "think_tag")


def load_runs(eval_root: Path) -> dict[tuple[str, str], dict]:
    runs: dict[tuple[str, str], dict] = {}
    for d in sorted(Path(eval_root).iterdir()):
        m = RUN_RE.match(d.name)
        if not m or not (d / "metrics.json").exists():
            continue
        runs[(m["model"], m["template"])] = json.loads((d / "metrics.json").read_text())
    if not runs:
        raise FileNotFoundError(f"no base_<model>_<template>/metrics.json under {eval_root}")
    return runs


def _fmt(est: dict | None) -> str:
    if not est:
        return "-"
    return f"{100 * est['mean']:.1f} [{100 * est['lo']:.1f}, {100 * est['hi']:.1f}]"


def _sorted_keys(runs: dict[tuple[str, str], dict]) -> list[tuple[str, str]]:
    return sorted(runs, key=lambda k: (-float(k[0][:-1]), TEMPLATE_ORDER.index(k[1])))


def accuracy_table(runs: dict[tuple[str, str], dict]) -> str:
    evals = [e for e in EVAL_ORDER if any(e in r for r in runs.values())]
    head = "| model | template | " + " | ".join(evals) + " |"
    sep = "|---|---|" + "---|" * len(evals)
    rows = [head, sep]
    for model, template in _sorted_keys(runs):
        r = runs[(model, template)]
        cells = [_fmt(r[e]["pass@1"]) if e in r else "-" for e in evals]
        rows.append(f"| Qwen3-{model.upper()}-Base | {template} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def failure_table(runs: dict[tuple[str, str], dict]) -> str:
    evals = [e for e in EVAL_ORDER if any(e in r for r in runs.values())]
    head = (
        "| model | template | " + " | ".join(f"{e} fail% / trunc% / tokens" for e in evals) + " |"
    )
    sep = "|---|---|" + "---|" * len(evals)
    rows = [head, sep]
    for model, template in _sorted_keys(runs):
        r = runs[(model, template)]
        cells = []
        for e in evals:
            if e not in r:
                cells.append("-")
                continue
            m = r[e]
            cells.append(
                f"{100 * m['extraction_failure_rate']:.1f} / {100 * m['truncation_rate']:.1f} / "
                f"{m['tokens']['mean']:.0f}"
            )
        rows.append(f"| Qwen3-{model.upper()}-Base | {template} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def level_table(runs: dict[tuple[str, str], dict]) -> str:
    rows = [
        "| model | template | L1 | L2 | L3 | L4 | L5 | all |",
        "|---|---|---|---|---|---|---|---|",
    ]
    any_level = False
    for model, template in _sorted_keys(runs):
        r = runs[(model, template)]
        m = r.get("math500")
        if not m or "by_level" not in m:
            continue
        any_level = True
        cells = [_fmt(m["by_level"].get(str(lv))) for lv in range(1, 6)]
        rows.append(
            f"| Qwen3-{model.upper()}-Base | {template} | "
            + " | ".join(cells)
            + f" | {_fmt(m['pass@1'])} |"
        )
    return ("\n".join(rows) + "\n") if any_level else ""


def extraction_sources_table(runs: dict[tuple[str, str], dict]) -> str:
    rows = ["| model | template | eval | extraction rule counts |", "|---|---|---|---|"]
    for model, template in _sorted_keys(runs):
        for e, m in runs[(model, template)].items():
            src = m.get("extraction_sources")
            if src:
                counts = ", ".join(
                    f"{k} {v}" for k, v in sorted(src.items(), key=lambda kv: -kv[1])
                )
                rows.append(f"| Qwen3-{model.upper()}-Base | {template} | {e} | {counts} |")
    return "\n".join(rows) + "\n"


def summary_json(runs: dict[tuple[str, str], dict]) -> dict:
    out: dict = {}
    for (model, template), r in runs.items():
        out[f"{model}/{template}"] = {
            e: {
                "pass@1": m["pass@1"],
                "extraction_failure_rate": m["extraction_failure_rate"],
                "truncation_rate": m["truncation_rate"],
                "num_problems": m["num_problems"],
                "run_tag": m.get("run_tag"),
                "by_level": m.get("by_level"),
            }
            for e, m in r.items()
        }
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("eval_root", type=Path)
    ap.add_argument(
        "--out", type=Path, required=True, help="Markdown file; a .json twin is written too"
    )
    args = ap.parse_args(argv)
    runs = load_runs(args.eval_root)
    tags = sorted({m.get("run_tag", "?") for r in runs.values() for m in r.values()})
    md = (
        "# Phase 1 greedy baselines (EXP-001)\n\n"
        f"Commits: {', '.join(tags)}. Greedy (T = 0), batch-invariant vLLM, 1024 max new tokens. "
        "pass@1 in %, 95 % bootstrap CI over problems.\n\n"
        "## Accuracy\n\n" + accuracy_table(runs) + "\n"
        "## Extraction failures / truncation / mean tokens\n\n" + failure_table(runs) + "\n"
    )
    lv = level_table(runs)
    if lv:
        md += "## MATH-500 by level\n\n" + lv + "\n"
    md += "## Which extraction rule fired\n\n" + extraction_sources_table(runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md)
    args.out.with_suffix(".json").write_text(json.dumps(summary_json(runs), indent=2))
    print(md)


if __name__ == "__main__":
    main()

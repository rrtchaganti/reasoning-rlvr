"""Decompose a paired checkpoint-minus-reference gain by whether each side's completion finished.

Usage (repo root, venv active):
    python scripts/decompose_gain.py --ref outputs/eval/rft_1.7b_think_tag --new outputs/eval/grpo_1.7b_think_tag \
        --label-ref rft --label-new grpo --out outputs/reports/phase-4-gain-decomposition.md

For every shared problem the first sample of each run gives (finished?, correct?), where finished
means ``finish_reason != "length"``. The net gain in points of the set (only-new minus only-ref
over all problems) is split into the part on problems the reference truncated and the part on
problems it finished; the accuracy on problems where both runs finished is compared paired. Both
splits carry 95 % bootstrap CIs over problems (2,000 resamples, seed 0). This is a descriptive
decomposition: truncation is itself an outcome of each policy, so the cells are not fixed strata,
and "the gain on problems the reference truncated" says where the extra correct answers are, not
what the reference would have answered with more tokens.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load(d: Path) -> dict[str, tuple[bool, bool]]:
    out: dict[str, tuple[bool, bool]] = {}
    for line in (d / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("sample_idx", 0) != 0:
            continue
        out[r["problem_id"]] = (r.get("finish_reason") != "length", bool(r["correct"]))
    return out


def ci(values: list[float], fn, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    point = fn(values)
    boots = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(fn(sample))
    boots.sort()
    return point, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def decompose(
    ref_dir: Path, new_dir: Path, label_ref: str, label_new: str
) -> tuple[dict, list[str]]:
    ref, new = load(ref_dir), load(new_dir)
    ids = sorted(set(ref) & set(new))
    n = len(ids)
    rows = [(ref[i][0], ref[i][1], new[i][0], new[i][1]) for i in ids]

    # per-problem contributions, in points of the set
    def contrib(r, where):
        rf, rc, _nf, nc = r
        if where == "ref_trunc" and rf:
            return 0.0
        if where == "ref_fin" and not rf:
            return 0.0
        return (float(nc) - float(rc)) * 100.0

    total = ci(rows, lambda rs: sum(contrib(r, "all") for r in rs) / len(rs))
    on_trunc = ci(rows, lambda rs: sum(contrib(r, "ref_trunc") for r in rs) / len(rs))
    on_fin = ci(rows, lambda rs: sum(contrib(r, "ref_fin") for r in rs) / len(rs))
    both = [r for r in rows if r[0] and r[2]]
    both_diff = (
        ci(both, lambda rs: sum((float(r[3]) - float(r[1])) * 100.0 for r in rs) / max(len(rs), 1))
        if both
        else (0.0, 0.0, 0.0)
    )
    cells: dict[tuple[bool, bool], dict[str, int]] = {}
    for rf, rc, nf, nc in rows:
        c = cells.setdefault((rf, nf), {"n": 0, "only_ref": 0, "only_new": 0})
        c["n"] += 1
        if rc and not nc:
            c["only_ref"] += 1
        elif nc and not rc:
            c["only_new"] += 1
    result = {
        "pairs": n,
        "net_gain": total,
        "gain_on_ref_truncated": on_trunc,
        "gain_on_ref_finished": on_fin,
        "both_finished_n": len(both),
        "both_finished_acc_ref": sum(r[1] for r in both) / max(len(both), 1) * 100,
        "both_finished_acc_new": sum(r[3] for r in both) / max(len(both), 1) * 100,
        "both_finished_paired_diff": both_diff,
        "cells": {
            f"{'fin' if k[0] else 'trunc'}/{'fin' if k[1] else 'trunc'}": v
            for k, v in cells.items()
        },
    }
    f = lambda t: f"{t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}]"  # noqa: E731
    lines = [
        f"| {ref_dir.name} | {new_dir.name} | {n} | {f(total)} | {f(on_trunc)} | {f(on_fin)} | "
        f"{len(both)} | {result['both_finished_acc_ref']:.1f} -> {result['both_finished_acc_new']:.1f} | {f(both_diff)} |"
    ]
    return result, lines


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--label-ref", default="ref")
    ap.add_argument("--label-new", default="new")
    ap.add_argument(
        "--evals",
        nargs="*",
        default=None,
        help="default: every <eval>/samples.jsonl shared by both",
    )
    ap.add_argument("--out", type=Path, default=None, help="Markdown; JSON next to it")
    args = ap.parse_args(argv)
    evals = args.evals or sorted(
        p.parent.name
        for p in args.ref.glob("*/samples.jsonl")
        if (args.new / p.parent.name / "samples.jsonl").exists()
    )
    header = [
        f"Paired gain {args.label_new} - {args.label_ref}, in points of each set, split by whether the "
        f"{args.label_ref} completion finished (95 % bootstrap CIs over problems):",
        "",
        f"| eval | pairs | net gain | on problems {args.label_ref} truncated | on problems {args.label_ref} finished | both finished (n) | accuracy there, {args.label_ref} -> {args.label_new} | paired diff there |",
        "|---|---|---|---|---|---|---|---|",
    ]
    results, lines = {}, []
    for ev in evals:
        res, ln = decompose(args.ref / ev, args.new / ev, args.label_ref, args.label_new)
        results[ev] = res
        lines.append(ln[0].replace(f"| {ev} | {ev} |", f"| {ev} |", 1))
    text = "\n".join(header + lines) + "\n"
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8", newline="\n")
        args.out.with_suffix(".json").write_text(
            json.dumps(results, indent=2), encoding="utf-8", newline="\n"
        )


if __name__ == "__main__":
    main()

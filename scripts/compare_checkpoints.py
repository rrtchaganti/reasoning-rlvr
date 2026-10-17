"""Paired comparison of two ``rlvr-eval`` runs on the same problems (checkpoint minus reference).

With ``--all-samples`` the two runs are n-sample runs (``configs/sample/*``) and the comparison is
the paired pass@k curve: per k, both means and the paired CI of the difference, plus the
"ever solved in n tries" table. Without it, only the first sample per problem is used (greedy runs).

The D-tables of every phase from here on are "base vs checkpoint with paired CIs" (PLAN.md section
4.2): the problems are shared, so the per-problem correctness difference is bootstrapped, which is
much tighter than two independent CIs. Per eval set: accuracy of both runs, the paired difference with
its 95 % CI, the flip table (solved only by A / only by B), and for MATH-500 the same by level.

Usage:
    python scripts/compare_checkpoints.py --ref outputs/eval/base_1.7b_think_tag_4096 \
        --new outputs/eval/sft_1.7b_think_tag --label-ref base --label-new sft \
        --out outputs/reports/phase-2-compare-think_tag.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlvr.eval.metrics import bootstrap_ci, paired_bootstrap_diff, paired_pass_at_k

EVAL_ORDER = ("gsm8k", "gsm8k_perturbed", "math500", "countdown", "aime24")
DEFAULT_KS = (1, 4, 16, 64)


def first_sample(samples_path: Path) -> dict[str, tuple[bool, dict]]:
    out: dict[str, tuple[bool, dict]] = {}
    with Path(samples_path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("sample_idx", 0) == 0:
                out[r["problem_id"]] = (bool(r["correct"]), r.get("meta") or {})
    return out


def sample_counts(samples_path: Path) -> dict[str, tuple[int, int, dict]]:
    """Per problem id: (samples, correct samples, meta) over every sample in the file."""
    n: dict[str, int] = {}
    c: dict[str, int] = {}
    meta: dict[str, dict] = {}
    with Path(samples_path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pid = r["problem_id"]
            n[pid] = n.get(pid, 0) + 1
            c[pid] = c.get(pid, 0) + int(bool(r["correct"]))
            meta.setdefault(pid, r.get("meta") or {})
    return {pid: (n[pid], c[pid], meta[pid]) for pid in n}


def paired_samples(
    ref: dict[str, tuple[int, int, dict]],
    new: dict[str, tuple[int, int, dict]],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict:
    """Paired pass@k on the shared ids of two n-sample runs (same n on both sides)."""
    ids = sorted(set(ref) & set(new))
    if not ids:
        raise ValueError("no shared problem ids between the two runs")
    ns = {ref[i][0] for i in ids} | {new[i][0] for i in ids}
    if len(ns) != 1:
        raise ValueError(f"both runs must have the same n for every problem; found {sorted(ns)}")
    out = paired_pass_at_k([ref[i][1] for i in ids], [new[i][1] for i in ids], ns.pop(), ks)
    out["unpaired"] = len(set(ref) ^ set(new))
    return out


def paired(ref: dict[str, tuple[bool, dict]], new: dict[str, tuple[bool, dict]]) -> dict:
    ids = sorted(set(ref) & set(new))
    if not ids:
        raise ValueError("no shared problem ids between the two runs")
    a = [float(ref[i][0]) for i in ids]
    b = [float(new[i][0]) for i in ids]
    return {
        "num_pairs": len(ids),
        "unpaired": len(set(ref) ^ set(new)),
        "acc_ref": bootstrap_ci(a).as_dict(),
        "acc_new": bootstrap_ci(b).as_dict(),
        "diff_new_minus_ref": paired_bootstrap_diff(b, a).as_dict(),
        "only_ref": sum(1 for x, y in zip(a, b, strict=True) if x and not y),
        "only_new": sum(1 for x, y in zip(a, b, strict=True) if y and not x),
    }


def by_level(
    ref: dict[str, tuple[bool, dict]], new: dict[str, tuple[bool, dict]]
) -> dict[int, dict]:
    out: dict[int, dict] = {}
    levels = sorted({m.get("level") for _, m in ref.values() if m.get("level") is not None})
    for lv in levels:
        r = {i: v for i, v in ref.items() if v[1].get("level") == lv}
        n = {i: v for i, v in new.items() if i in r}
        if r and n:
            out[int(lv)] = paired(r, n)
    return out


def _f(e: dict) -> str:
    return f"{100 * e['mean']:.1f} [{100 * e['lo']:.1f}, {100 * e['hi']:.1f}]"


def markdown(
    results: dict[str, dict], levels: dict[int, dict], label_ref: str, label_new: str
) -> str:
    lines = [
        f"| eval | pairs | {label_ref} | {label_new} | {label_new} - {label_ref} [paired CI] | "
        f"only {label_ref} / only {label_new} |",
        "|---|---|---|---|---|---|",
    ]
    for ev in sorted(results, key=lambda e: EVAL_ORDER.index(e) if e in EVAL_ORDER else 99):
        g = results[ev]
        lines.append(
            f"| {ev} | {g['num_pairs']} | {_f(g['acc_ref'])} | {_f(g['acc_new'])} | "
            f"**{_f(g['diff_new_minus_ref'])}** | {g['only_ref']} / {g['only_new']} |"
        )
    if levels:
        lines += [
            "",
            f"MATH-500 by level, {label_new} - {label_ref}:",
            "",
            f"| level | n | {label_ref} | {label_new} | diff [paired CI] |",
            "|---|---|---|---|---|",
        ]
        for lv, g in sorted(levels.items()):
            lines.append(
                f"| {lv} | {g['num_pairs']} | {_f(g['acc_ref'])} | {_f(g['acc_new'])} | "
                f"{_f(g['diff_new_minus_ref'])} |"
            )
    return "\n".join(lines) + "\n"


def markdown_samples(results: dict[str, dict], label_ref: str, label_new: str) -> str:
    lines = [
        f"| eval | pairs | k | {label_ref} pass@k | {label_new} pass@k | "
        f"{label_new} - {label_ref} [paired CI] |",
        "|---|---|---|---|---|---|",
    ]
    for ev in sorted(results, key=lambda e: EVAL_ORDER.index(e) if e in EVAL_ORDER else 99):
        g = results[ev]
        for k, r in sorted(g["by_k"].items()):
            lines.append(
                f"| {ev} | {g['num_pairs']} | {k} | {_f(r['ref'])} | {_f(r['new'])} | "
                f"**{_f(r['diff_new_minus_ref'])}** |"
            )
    lines += ["", f"Solved at least once in n tries, {label_ref} / {label_new}:", ""]
    lines += [
        f"| eval | n | both | only {label_ref} | only {label_new} | neither |",
        "|---|---|---|---|---|---|",
    ]
    for ev in sorted(results, key=lambda e: EVAL_ORDER.index(e) if e in EVAL_ORDER else 99):
        g, t = results[ev], results[ev]["ever_solved"]
        lines.append(
            f"| {ev} | {g['n']} | {t['both']} | {t['only_ref']} | {t['only_new']} | {t['neither']} |"
        )
    return "\n".join(lines) + "\n"


def compare_runs(
    ref_dir: Path, new_dir: Path, *, all_samples: bool = False, ks: tuple[int, ...] = DEFAULT_KS
) -> tuple[dict[str, dict], dict[int, dict]]:
    results: dict[str, dict] = {}
    levels: dict[int, dict] = {}
    for sp in sorted(Path(ref_dir).glob("*/samples.jsonl")):
        ev = sp.parent.name
        other = Path(new_dir) / ev / "samples.jsonl"
        if not other.exists():
            continue
        if all_samples:
            results[ev] = paired_samples(sample_counts(sp), sample_counts(other), ks)
            continue
        ref, new = first_sample(sp), first_sample(other)
        results[ev] = paired(ref, new)
        if ev == "math500":
            levels = by_level(ref, new)
    if not results:
        raise FileNotFoundError(f"no shared <eval>/samples.jsonl between {ref_dir} and {new_dir}")
    return results, levels


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Paired comparison of two eval runs.")
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--label-ref", default="ref")
    ap.add_argument("--label-new", default="new")
    ap.add_argument("--out", type=Path, default=None, help="Markdown; JSON written next to it")
    ap.add_argument(
        "--all-samples",
        action="store_true",
        help="n-sample runs: paired pass@k curves instead of first-sample accuracy",
    )
    ap.add_argument(
        "--ks", default=",".join(map(str, DEFAULT_KS)), help="k values, comma-separated"
    )
    args = ap.parse_args(argv)
    ks = tuple(int(k) for k in args.ks.split(","))
    results, levels = compare_runs(args.ref, args.new, all_samples=args.all_samples, ks=ks)
    md = (
        markdown_samples(results, args.label_ref, args.label_new)
        if args.all_samples
        else markdown(results, levels, args.label_ref, args.label_new)
    )
    print(md)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8", newline="\n")
        args.out.with_suffix(".json").write_text(
            json.dumps({"evals": results, "math500_by_level": levels}, indent=2)
        )


if __name__ == "__main__":
    main()

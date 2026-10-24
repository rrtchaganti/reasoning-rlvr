"""EXP-024: pair a reproduced eval run with the archived one, per problem and sample.

For each run name and each eval set present in the reference, reads ``<root>/<run>/<eval>/samples.jsonl``
on both sides, pairs records on (problem_id, sample_idx), and counts: pairs present on both sides,
identical completion text, identical extracted answer, identical grade; pass@1 on each side. Writes a
Markdown table and a JSON file, and lists the first differing problems with where their texts part.

Usage:
    python scripts/check_reproduction.py --ref REF_EVAL_ROOT --new NEW_EVAL_ROOT \\
        --runs base_1.7b_zero_shot base_1.7b_think_tag --out outputs/reports/reproduction-check.md

Exit status 0 whatever the result (the result is the finding); ``--strict`` exits 1 on any difference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[(r["problem_id"], int(r.get("sample_idx", 0)))] = r
    return out


def first_divergence(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def compare_eval(ref_file: Path, new_file: Path) -> dict:
    ref, new = load(ref_file), load(new_file)
    shared = sorted(set(ref) & set(new))
    text_diff, extr_diff, grade_diff = [], [], []
    for k in shared:
        a, b = ref[k], new[k]
        if a.get("text") != b.get("text"):
            text_diff.append((k, first_divergence(a.get("text") or "", b.get("text") or "")))
        if a.get("extracted") != b.get("extracted"):
            extr_diff.append(k)
        if bool(a.get("correct")) != bool(b.get("correct")):
            grade_diff.append(k)

    def acc(d: dict) -> float:
        if not shared:
            return float("nan")
        return 100.0 * sum(bool(d[k].get("correct")) for k in shared) / len(shared)

    return {
        "n_ref": len(ref),
        "n_new": len(new),
        "pairs": len(shared),
        "text_same": len(shared) - len(text_diff),
        "extracted_same": len(shared) - len(extr_diff),
        "grade_same": len(shared) - len(grade_diff),
        "acc_ref": acc(ref),
        "acc_new": acc(new),
        "text_diff": [
            {"problem_id": k[0], "sample_idx": k[1], "first_diff_char": i} for k, i in text_diff
        ],
        "grade_diff": [{"problem_id": k[0], "sample_idx": k[1]} for k in grade_diff],
    }


def identical(r: dict) -> bool:
    return (
        r["n_ref"]
        == r["n_new"]
        == r["pairs"]
        == r["text_same"]
        == r["extracted_same"]
        == r["grade_same"]
    )


def run(ref_root: Path, new_root: Path, runs: list[str]) -> list[dict]:
    rows = []
    for name in runs:
        ref_dir, new_dir = ref_root / name, new_root / name
        evals = sorted(p.parent.name for p in ref_dir.glob("*/samples.jsonl"))
        if not evals:
            rows.append({"run": name, "eval": "-", "missing": f"no samples under {ref_dir}"})
            continue
        for ev in evals:
            nf = new_dir / ev / "samples.jsonl"
            if not nf.exists():
                rows.append({"run": name, "eval": ev, "missing": f"no {nf}"})
                continue
            r = compare_eval(ref_dir / ev / "samples.jsonl", nf)
            r.update(run=name, eval=ev)
            rows.append(r)
    return rows


def render(rows: list[dict]) -> str:
    out = [
        "| run | eval | pairs (ref / new) | same text | same extracted | same grade | pass@1 ref | pass@1 new | identical |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "missing" in r:
            out.append(f"| {r['run']} | {r['eval']} | {r['missing']} | | | | | | **NO** |")
            continue
        out.append(
            f"| {r['run']} | {r['eval']} | {r['pairs']} ({r['n_ref']} / {r['n_new']}) | {r['text_same']} | "
            f"{r['extracted_same']} | {r['grade_same']} | {r['acc_ref']:.1f} | {r['acc_new']:.1f} | "
            f"{'yes' if identical(r) else '**NO**'} |"
        )
    diffs = [
        (r["run"], r["eval"], d) for r in rows if "text_diff" in r for d in r["text_diff"][:10]
    ]
    if diffs:
        out += ["", "First differing completions (up to 10 per eval set):", ""]
        out += [
            f"- {a} / {b}: `{d['problem_id']}` sample {d['sample_idx']}, texts part at character {d['first_diff_char']}"
            for a, b, d in diffs
        ]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--ref",
        type=Path,
        required=True,
        help="archived eval root (holds <run>/<eval>/samples.jsonl)",
    )
    ap.add_argument("--new", type=Path, required=True, help="reproduced eval root")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument(
        "--out", type=Path, required=True, help="Markdown output; a .json is written next to it"
    )
    ap.add_argument("--strict", action="store_true", help="exit 1 on any difference")
    a = ap.parse_args(argv)
    rows = run(a.ref, a.new, a.runs)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    text = render(rows)
    a.out.write_text("# Reproduction check (EXP-024)\n\n" + text, encoding="utf-8")
    a.out.with_suffix(".json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(text)
    ok = all("missing" not in r and identical(r) for r in rows)
    print("ALL IDENTICAL" if ok else "DIFFERENCES FOUND")
    return 1 if (a.strict and not ok) else 0


if __name__ == "__main__":
    sys.exit(main())

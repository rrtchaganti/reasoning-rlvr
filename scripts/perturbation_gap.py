"""Perturbation gap: accuracy on original GSM8K minus accuracy on its perturbed twins, paired.

EXP-003 (b). Reads ``samples.jsonl`` from a run's ``gsm8k/`` and ``gsm8k_perturbed/`` folders, pairs
each perturbed problem with its original through ``meta.original_id``, and reports

- accuracy on the paired originals and on the perturbed set (both over the same problems),
- the gap with a paired bootstrap 95 % CI (per-problem difference resampled over problems),
- the 2 x 2 table (solved both / only original / only perturbed / neither), which is the honest
  picture: a gap of 3 points can be 3 % flipping one way or 20 % flipping each way.

Only the first sample per problem is used (greedy runs have one), unless ``--all-samples``: then
the run is an n-sample run and the gap is reported per k as a paired pass@k difference, with the
"ever solved in n tries" table (EXP-002 at n = 64; added 18 Oct 2026 after the L4 session). Usage:
    python scripts/perturbation_gap.py outputs/eval/base_1.7b_few_shot_cot [--out gap.json]
    python scripts/perturbation_gap.py outputs/samples/base_1.7b_n64_few_shot_cot --all-samples --out outputs/reports/phase-1-perturbation-gap-n64.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_checkpoints import DEFAULT_KS, sample_counts

from rlvr.eval.metrics import bootstrap_ci, paired_bootstrap_diff, paired_pass_at_k


def first_sample_correct(samples_path: Path) -> dict[str, tuple[bool, dict]]:
    out: dict[str, tuple[bool, dict]] = {}
    with Path(samples_path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("sample_idx", 0) == 0:
                out[r["problem_id"]] = (bool(r["correct"]), r.get("meta") or {})
    return out


def perturbation_gap(
    original: dict[str, tuple[bool, dict]], perturbed: dict[str, tuple[bool, dict]]
) -> dict:
    pairs: list[tuple[float, float]] = []
    missing = 0
    for _pid, (ok_p, meta) in perturbed.items():
        oid = meta.get("original_id")
        if oid not in original:
            missing += 1
            continue
        pairs.append((float(original[oid][0]), float(ok_p)))
    if not pairs:
        raise ValueError(
            "no pairs: does the perturbed set carry meta.original_id and was the original run on the same split?"
        )
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    both = sum(1 for x, y in pairs if x and y)
    only_orig = sum(1 for x, y in pairs if x and not y)
    only_pert = sum(1 for x, y in pairs if y and not x)
    neither = sum(1 for x, y in pairs if not x and not y)
    return {
        "num_pairs": len(pairs),
        "unpaired_perturbed": missing,
        "acc_original_on_pairs": bootstrap_ci(a).as_dict(),
        "acc_perturbed": bootstrap_ci(b).as_dict(),
        "gap_original_minus_perturbed": paired_bootstrap_diff(a, b).as_dict(),
        "table": {
            "both": both,
            "only_original": only_orig,
            "only_perturbed": only_pert,
            "neither": neither,
        },
        "flip_rate": (only_orig + only_pert) / len(pairs),
    }


def perturbation_gap_pass_at_k(
    original: dict[str, tuple[int, int, dict]],
    perturbed: dict[str, tuple[int, int, dict]],
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict:
    """Original minus perturbed pass@k, paired through ``meta.original_id``, at each k."""
    orig_counts: list[int] = []
    pert_counts: list[int] = []
    missing = 0
    ns: set[int] = set()
    for _pid, (n_p, c_p, meta) in perturbed.items():
        oid = meta.get("original_id")
        if oid not in original:
            missing += 1
            continue
        n_o, c_o, _ = original[oid]
        ns |= {n_o, n_p}
        orig_counts.append(c_o)
        pert_counts.append(c_p)
    if not orig_counts:
        raise ValueError(
            "no pairs: does the perturbed set carry meta.original_id and was the original run on the same split?"
        )
    if len(ns) != 1:
        raise ValueError(f"both sets must have the same n for every problem; found {sorted(ns)}")
    # ref = perturbed, new = original, so "new minus ref" is the gap (original minus perturbed)
    g = paired_pass_at_k(pert_counts, orig_counts, ns.pop(), ks)
    by_k = {
        k: {
            "pass_at_k_original": r["new"],
            "pass_at_k_perturbed": r["ref"],
            "gap_original_minus_perturbed": r["diff_new_minus_ref"],
        }
        for k, r in g["by_k"].items()
    }
    t = g["ever_solved"]
    return {
        "n": g["n"],
        "num_pairs": g["num_pairs"],
        "unpaired_perturbed": missing,
        "by_k": by_k,
        "ever_solved": {
            "both": t["both"],
            "only_original": t["only_new"],
            "only_perturbed": t["only_ref"],
            "neither": t["neither"],
        },
    }


def markdown_pass_at_k(label: str, g: dict) -> str:
    def f(e: dict) -> str:
        return f"{100 * e['mean']:.1f} [{100 * e['lo']:.1f}, {100 * e['hi']:.1f}]"

    lines = [
        "| run | pairs | k | original pass@k [CI] | perturbed pass@k [CI] | gap (orig - pert) [paired CI] |",
        "|---|---|---|---|---|---|",
    ]
    for k, r in sorted(g["by_k"].items()):
        lines.append(
            f"| {label} | {g['num_pairs']} | {k} | {f(r['pass_at_k_original'])} | "
            f"{f(r['pass_at_k_perturbed'])} | **{f(r['gap_original_minus_perturbed'])}** |"
        )
    t = g["ever_solved"]
    lines += [
        "",
        f"Solved at least once in {g['n']} tries: both {t['both']}, only original "
        f"{t['only_original']}, only perturbed {t['only_perturbed']}, neither {t['neither']} "
        f"(of {g['num_pairs']} pairs).",
    ]
    return "\n".join(lines) + "\n"


def markdown_row(label: str, g: dict) -> str:
    def f(e: dict) -> str:
        return f"{100 * e['mean']:.1f} [{100 * e['lo']:.1f}, {100 * e['hi']:.1f}]"

    t = g["table"]
    return (
        f"| {label} | {g['num_pairs']} | {f(g['acc_original_on_pairs'])} | {f(g['acc_perturbed'])} | "
        f"{f(g['gap_original_minus_perturbed'])} | {t['only_original']} / {t['only_perturbed']} | "
        f"{100 * g['flip_rate']:.1f} |"
    )


MARKDOWN_HEADER = (
    "| run | pairs | original acc [CI] | perturbed acc [CI] | gap (orig - pert) [paired CI] | "
    "flips orig-only / pert-only | flip % |\n|---|---|---|---|---|---|---|"
)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "run_dir", type=Path, help="an rlvr-eval output dir containing gsm8k/ and gsm8k_perturbed/"
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="default <run_dir>/perturbation_gap.json"
    )
    ap.add_argument(
        "--all-samples",
        action="store_true",
        help="n-sample run: paired pass@k gap per k (writes JSON and a Markdown table next to it)",
    )
    ap.add_argument(
        "--ks", default=",".join(map(str, DEFAULT_KS)), help="k values, comma-separated"
    )
    args = ap.parse_args(argv)
    o_path = args.run_dir / "gsm8k" / "samples.jsonl"
    p_path = args.run_dir / "gsm8k_perturbed" / "samples.jsonl"
    if args.all_samples:
        ks = tuple(int(k) for k in args.ks.split(","))
        g = perturbation_gap_pass_at_k(sample_counts(o_path), sample_counts(p_path), ks)
        out = args.out or args.run_dir / "perturbation_gap_pass_at_k.json"
        md = markdown_pass_at_k(args.run_dir.name, g)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(g, indent=2), encoding="utf-8", newline="\n")
        out.with_suffix(".md").write_text(md, encoding="utf-8", newline="\n")
        print(md)
        print(f"wrote {out} and {out.with_suffix('.md')}")
        return
    orig = first_sample_correct(o_path)
    pert = first_sample_correct(p_path)
    g = perturbation_gap(orig, pert)
    out = args.out or args.run_dir / "perturbation_gap.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(g, indent=2))
    print(MARKDOWN_HEADER)
    print(markdown_row(args.run_dir.name, g))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

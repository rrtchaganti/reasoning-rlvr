"""Perturbation gap: accuracy on original GSM8K minus accuracy on its perturbed twins, paired.

EXP-003 (b). Reads ``samples.jsonl`` from a run's ``gsm8k/`` and ``gsm8k_perturbed/`` folders, pairs
each perturbed problem with its original through ``meta.original_id``, and reports

- accuracy on the paired originals and on the perturbed set (both over the same problems),
- the gap with a paired bootstrap 95 % CI (per-problem difference resampled over problems),
- the 2 x 2 table (solved both / only original / only perturbed / neither), which is the honest
  picture: a gap of 3 points can be 3 % flipping one way or 20 % flipping each way.

Only the first sample per problem is used (greedy runs have one). Usage:
    python scripts/perturbation_gap.py outputs/eval/base_1.7b_few_shot_cot [--out gap.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlvr.eval.metrics import bootstrap_ci, paired_bootstrap_diff


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
    args = ap.parse_args(argv)
    orig = first_sample_correct(args.run_dir / "gsm8k" / "samples.jsonl")
    pert = first_sample_correct(args.run_dir / "gsm8k_perturbed" / "samples.jsonl")
    g = perturbation_gap(orig, pert)
    out = args.out or args.run_dir / "perturbation_gap.json"
    out.write_text(json.dumps(g, indent=2))
    print(MARKDOWN_HEADER)
    print(markdown_row(args.run_dir.name, g))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

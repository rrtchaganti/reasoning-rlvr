"""Data-scaling curve for SFT (EXP-008): pass@1 against the fraction of the filtered set.

Usage (repo root, venv active):
    python scripts/plot_data_scaling.py \
        0.25=outputs/eval/sft_1.7b_data25_think_tag 0.5=outputs/eval/sft_1.7b_data50_think_tag \
        1.0=outputs/eval/sft_1.7b_think_tag --base outputs/eval/base_1.7b_think_tag_4096 \
        --out outputs/reports/figures/phase-2-data-scaling.png

Each run directory holds ``metrics.json`` keyed by eval with ``pass@1`` (mean, lo, hi). One panel per
eval, log-x, 95 % bootstrap error bars; the base model (fraction 0) as a horizontal reference when
given. Also writes a Markdown table next to the figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EVAL_ORDER = ("gsm8k", "gsm8k_perturbed", "math500", "countdown", "aime24")


def parse_points(items: list[str]) -> list[tuple[float, Path]]:
    points = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected FRACTION=RUN_DIR, got {item!r}")
        frac, path = item.split("=", 1)
        points.append((float(frac), Path(path)))
    points.sort(key=lambda fp: fp[0])
    if not points:
        raise ValueError("no points given")
    return points


def load_metrics(run_dir: Path) -> dict[str, dict]:
    path = Path(run_dir) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def collect(points: list[tuple[float, Path]]) -> dict[str, list[tuple[float, dict]]]:
    """``{eval: [(fraction, pass@1 estimate), ...]}`` in fraction order."""
    out: dict[str, list[tuple[float, dict]]] = {}
    for frac, d in points:
        for ev, m in load_metrics(d).items():
            if "pass@1" in m:
                out.setdefault(ev, []).append((frac, m["pass@1"]))
    if not out:
        raise FileNotFoundError("no pass@1 in any metrics.json")
    return out


def markdown_table(
    curves: dict[str, list[tuple[float, dict]]], base: dict[str, dict] | None
) -> str:
    fracs = sorted({f for pts in curves.values() for f, _ in pts})
    head = (
        "| eval | " + ("base | " if base else "") + " | ".join(f"{int(100 * f)} %" for f in fracs)
    )
    sep = "|---|" + ("---|" if base else "") + "---|" * len(fracs)
    rows = [head + " |", sep]
    for ev in sorted(curves, key=lambda e: EVAL_ORDER.index(e) if e in EVAL_ORDER else 99):
        by_frac = dict(curves[ev])
        cells = []
        if base:
            b = base.get(ev, {}).get("pass@1")
            cells.append(
                f"{100 * b['mean']:.1f} [{100 * b['lo']:.1f}, {100 * b['hi']:.1f}]" if b else "-"
            )
        for f in fracs:
            e = by_frac.get(f)
            cells.append(
                f"{100 * e['mean']:.1f} [{100 * e['lo']:.1f}, {100 * e['hi']:.1f}]" if e else "-"
            )
        rows.append(f"| {ev} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def plot(
    curves: dict[str, list[tuple[float, dict]]], base: dict[str, dict] | None, out: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evals = sorted(curves, key=lambda e: EVAL_ORDER.index(e) if e in EVAL_ORDER else 99)
    fig, axes = plt.subplots(1, len(evals), figsize=(3.6 * len(evals), 3.2), squeeze=False)
    for ax, ev in zip(axes[0], evals, strict=True):
        pts = curves[ev]
        xs = [f for f, _ in pts]
        ys = [100 * e["mean"] for _, e in pts]
        lo = [100 * (e["mean"] - e["lo"]) for _, e in pts]
        hi = [100 * (e["hi"] - e["mean"]) for _, e in pts]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, color="#4C72B0", label="SFT")
        if base and ev in base and "pass@1" in base[ev]:
            ax.axhline(100 * base[ev]["pass@1"]["mean"], color="#C44E52", ls="--", label="base")
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{int(100 * x)} %" for x in xs])
        ax.set_title(ev)
        ax.set_xlabel("fraction of the SFT set")
        ax.set_ylabel("pass@1 (%)")
        ax.grid(alpha=0.3)
    axes[0][0].legend(loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="SFT data-scaling figure (EXP-008).")
    ap.add_argument("points", nargs="+", help="FRACTION=RUN_DIR, e.g. 0.25=outputs/eval/x")
    ap.add_argument("--base", type=Path, default=None, help="base-model run dir (reference line)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    curves = collect(parse_points(args.points))
    base = load_metrics(args.base) if args.base else None
    plot(curves, base, args.out)
    table = markdown_table(curves, base)
    args.out.with_suffix(".md").write_text(table, encoding="utf-8", newline="\n")
    print(table)
    print(f"figure: {args.out}")


if __name__ == "__main__":
    main()

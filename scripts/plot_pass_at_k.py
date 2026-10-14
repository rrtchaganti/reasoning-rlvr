"""Plot pass@k and maj@k against k from one or more ``rlvr-eval`` runs with n > 1 (EXP-002, D2 figure).

Usage (repo root, venv active):
    python scripts/plot_pass_at_k.py outputs/samples/base_1.7b_n64_few_shot_cot \
        outputs/samples/base_1.7b_n64_countdown --out outputs/reports/figures/phase-1-pass-at-k.png

Each run directory holds ``<eval>/metrics.json`` with ``pass@k`` (mean, lo, hi per k) and ``maj@k``.
One panel per eval; pass@k drawn with its 95 % bootstrap band, maj@k as a dashed line, greedy pass@1
(if a ``--greedy`` run directory is given) as a horizontal reference. Also writes a Markdown table
next to the figure with the k = 1, 4, 16, 64 columns, because a reviewer wants the numbers too.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_runs(run_dirs: list[Path]) -> dict[str, dict]:
    """``{eval_name: metrics}`` across run directories (later directories win on a clash)."""
    out: dict[str, dict] = {}
    for d in run_dirs:
        for mpath in sorted(Path(d).glob("*/metrics.json")):
            m = json.loads(mpath.read_text())
            if "pass@k" in m:
                out[m.get("eval", mpath.parent.name)] = m
    if not out:
        raise FileNotFoundError(f"no <eval>/metrics.json with pass@k under {run_dirs}")
    return out


def greedy_pass1(run_dir: Path | None) -> dict[str, float]:
    if run_dir is None:
        return {}
    out = {}
    for mpath in Path(run_dir).glob("*/metrics.json"):
        m = json.loads(mpath.read_text())
        out[m.get("eval", mpath.parent.name)] = m["pass@1"]["mean"]
    return out


def markdown_table(runs: dict[str, dict], ks=(1, 4, 16, 64)) -> str:
    head = (
        "| eval | n | "
        + " | ".join(f"pass@{k}" for k in ks)
        + " | "
        + " | ".join(f"maj@{k}" for k in ks)
        + " |"
    )
    sep = "|---|---|" + "---|" * (2 * len(ks))
    rows = [head, sep]
    for name, m in runs.items():
        pk = m["pass@k"]
        mk = m.get("maj@k", {})
        cells = []
        for k in ks:
            e = pk.get(str(k))
            cells.append(
                f"{100 * e['mean']:.1f} [{100 * e['lo']:.1f}, {100 * e['hi']:.1f}]" if e else "-"
            )
        for k in ks:
            v = mk.get(str(k))
            cells.append(f"{100 * v:.1f}" if v is not None else "-")
        rows.append(f"| {name} | {m['num_samples_per_problem']} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def plot(
    runs: dict[str, dict], out: Path, *, greedy: dict[str, float] | None = None, title: str = ""
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(runs)
    fig, axes = plt.subplots(
        1, len(names), figsize=(4.2 * len(names), 3.6), squeeze=False, sharey=False
    )
    for ax, name in zip(axes[0], names, strict=True):
        m = runs[name]
        ks = sorted(int(k) for k in m["pass@k"])
        mean = [100 * m["pass@k"][str(k)]["mean"] for k in ks]
        lo = [100 * m["pass@k"][str(k)]["lo"] for k in ks]
        hi = [100 * m["pass@k"][str(k)]["hi"] for k in ks]
        ax.plot(ks, mean, marker="o", label="pass@k")
        ax.fill_between(ks, lo, hi, alpha=0.2)
        mk = m.get("maj@k", {})
        if mk:
            mks = sorted(int(k) for k in mk)
            ax.plot(mks, [100 * mk[str(k)] for k in mks], marker="s", linestyle="--", label="maj@k")
        if greedy and name in greedy:
            ax.axhline(100 * greedy[name], color="grey", linestyle=":", label="greedy pass@1")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("k (samples)")
        ax.set_ylabel("accuracy (%)")
        ax.set_title(f"{name} (n={m['num_samples_per_problem']})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument(
        "--greedy", type=Path, default=None, help="a greedy run dir for the pass@1 reference line"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args(argv)
    runs = load_runs(args.run_dirs)
    path = plot(runs, args.out, greedy=greedy_pass1(args.greedy), title=args.title)
    table = markdown_table(runs)
    args.out.with_suffix(".md").write_text(table)
    print(f"wrote {path} and {args.out.with_suffix('.md')}")
    print(table)


if __name__ == "__main__":
    main()

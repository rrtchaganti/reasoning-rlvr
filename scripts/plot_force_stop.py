"""Budget forcing (EXP-016): accuracy against generated tokens per budget, with the unforced
greedy runs of each checkpoint as reference points.

Usage (repo root, venv active):
    python scripts/plot_force_stop.py --runs sft=outputs/eval/sft_force{B}_think_tag rft=outputs/eval/rft_force{B}_think_tag \
        --budgets 768 1024 1536 2048 3072 \
        --greedy sft=outputs/eval/sft_1.7b_think_tag rft=outputs/eval/rft_1.7b_think_tag grpo=outputs/eval/grpo_1.7b_think_tag \
        --out outputs/reports/figures/phase-5-force-stop.png

``--runs`` takes LABEL=PATTERN with ``{B}`` standing for the budget; each resolved directory is an
``rlvr-eval`` run with ``<eval>/metrics.json`` (pass@1 with its CI, mean tokens, and the forced-stop
fields the harness writes: ``forced_stop_rate``, ``forced_stop_accuracy``). Greedy references are
plain runs. One panel per eval: x = mean generated tokens per problem, y = pass@1 with its CI band;
one line per forced checkpoint across budgets (points labelled with the budget), stars for the
greedy references. A Markdown table with every number is written next to the figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EVALS = ("gsm8k", "math500")


def parse_labelled(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"expected LABEL=PATH, got {item!r}")
        label, p = item.split("=", 1)
        out[label] = p
    return out


def read_metrics(run_dir: Path, ev: str) -> dict[str, Any] | None:
    p = Path(run_dir) / ev / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    return {
        "acc": m["pass@1"]["mean"],
        "lo": m["pass@1"]["lo"],
        "hi": m["pass@1"]["hi"],
        "tokens": m["tokens"]["mean"],
        "truncation_rate": m.get("truncation_rate"),
        "forced_stop_rate": m.get("forced_stop_rate"),
        "forced_stop_accuracy": m.get("forced_stop_accuracy"),
        "forced_stop_finished_rate": m.get("forced_stop_finished_rate"),
    }


def collect(
    runs: dict[str, str], budgets: list[int], greedy: dict[str, str], evals=EVALS
) -> dict[str, Any]:
    out: dict[str, Any] = {"evals": {}}
    for ev in evals:
        block: dict[str, Any] = {"forced": {}, "greedy": {}}
        for label, pattern in runs.items():
            pts = []
            for b in budgets:
                m = read_metrics(Path(pattern.format(B=b)), ev)
                if m:
                    pts.append({"budget": b, **m})
            if pts:
                block["forced"][label] = pts
        for label, d in greedy.items():
            m = read_metrics(Path(d), ev)
            if m:
                block["greedy"][label] = m
        if block["forced"] or block["greedy"]:
            out["evals"][ev] = block
    if not out["evals"]:
        raise SystemExit("no metrics found")
    return out


def _ci(m: dict[str, Any]) -> str:
    return f"{100 * m['acc']:.1f} [{100 * m['lo']:.1f}, {100 * m['hi']:.1f}]"


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def markdown(result: dict[str, Any]) -> str:
    lines = []
    for ev, block in result["evals"].items():
        lines.append(f"**{ev}**")
        lines.append("")
        lines.append(
            "| run | budget | pass@1 | tokens / problem | forced % | forced answers right % | truncated % |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for label, m in block["greedy"].items():
            lines.append(
                f"| {label} greedy (unforced) | 4096 | {_ci(m)} | {m['tokens']:.0f} | - | - | "
                f"{_pct(m['truncation_rate'])} |"
            )
        for label, pts in block["forced"].items():
            for m in pts:
                lines.append(
                    f"| {label} + force | {m['budget']} | {_ci(m)} | {m['tokens']:.0f} | "
                    f"{_pct(m['forced_stop_rate'])} | {_pct(m['forced_stop_accuracy'])} | "
                    f"{_pct(m['truncation_rate'])} |"
                )
        lines.append("")
    return "\n".join(lines)


def plot(result: dict[str, Any], out: Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evals = list(result["evals"])
    fig, axes = plt.subplots(1, len(evals), figsize=(6.2 * len(evals), 4.6), squeeze=False)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, ev in zip(axes[0], evals, strict=True):
        block = result["evals"][ev]
        colours: dict[str, str] = {}
        for i, lab in enumerate([*block["forced"], *block["greedy"]]):
            colours.setdefault(lab, cycle[i % len(cycle)])
        for lab, pts in block["forced"].items():
            x = [p["tokens"] for p in pts]
            y = [100 * p["acc"] for p in pts]
            lo = [100 * p["lo"] for p in pts]
            hi = [100 * p["hi"] for p in pts]
            ax.plot(x, y, "-o", color=colours[lab], label=f"{lab} + forced stop")
            ax.fill_between(x, lo, hi, color=colours[lab], alpha=0.12, linewidth=0)
            for p in pts:
                ax.annotate(
                    str(p["budget"]),
                    (p["tokens"], 100 * p["acc"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=7,
                    color=colours[lab],
                )
        for lab, m in block["greedy"].items():
            ax.errorbar(
                [m["tokens"]],
                [100 * m["acc"]],
                yerr=[[100 * (m["acc"] - m["lo"])], [100 * (m["hi"] - m["acc"])]],
                fmt="*",
                markersize=13,
                color=colours[lab],
                capsize=3,
                label=f"{lab} greedy, 4096",
            )
        ax.set_xlabel("mean generated tokens per problem")
        ax.set_ylabel("pass@1 (%)")
        ax.set_title(ev)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--runs", nargs="+", required=True, help="LABEL=PATTERN with {B} for the budget"
    )
    ap.add_argument("--budgets", nargs="+", type=int, required=True)
    ap.add_argument("--greedy", nargs="*", default=[], help="LABEL=DIR of unforced greedy runs")
    ap.add_argument("--evals", nargs="+", default=list(EVALS))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args(argv)
    result = collect(
        parse_labelled(args.runs), args.budgets, parse_labelled(args.greedy), args.evals
    )
    plot(result, args.out, args.title)
    md = markdown(result)
    args.out.with_suffix(".md").write_text(md, encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(md)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

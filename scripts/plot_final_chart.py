"""The project's one chart (D6): accuracy against generated tokens, every way of spending them.

On the pinned 300-problem subsets of GSM8K and MATH-500 (the problems of the n = 32 sample sets):

- greedy decoding with the stop forced at a budget (laptop runs), ending at the unforced greedy run of
  the same checkpoint on the same machine ("no stop");
- majority vote over k samples (T = 0.6, H100 sample sets), k = 1..32;
- the reward-weighted vote with the 0.6B verifier over `rft`'s samples (EXP-022).

Colour is the checkpoint; line style is the method; marker shape is the machine (circle laptop, square
H100). Points are levels with CIs (in the table), not paired differences: a pair is only read within
one machine (EXP-020b). Sampled curves come from the EXP-022 analysis JSON; laptop points are computed
here on the subset ids.

Usage:
    python scripts/plot_final_chart.py \\
        --curves outputs/reports/figures/phase-5-accuracy-vs-tokens-orm.json \\
        --subset-from ../hub_samples/sft_1.7b_n32_think_tag_t06 \\
        --laptop-root outputs/eval --out outputs/reports/figures/final-accuracy-vs-tokens.png
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

CKPTS = ("sft", "rft", "grpo")
BUDGETS = {
    "sft": (768, 1024, 1536, 2048, 3072),
    "rft": (768, 1024, 1536, 2048, 3072),
    "grpo": (1536, 2048),
}
EVALS = (("gsm8k", "GSM8K"), ("math500", "MATH-500"))
# reference palette slots 1-3 (dataviz references/palette.md), chrome from the same file
COLOR = {"sft": "#2a78d6", "rft": "#eb6834", "grpo": "#1baf7a"}
# label beside each "no stop" marker, placed off the lines that pass near it
LABEL_OFFSET = {"sft": (9, -4), "rft": (9, -4)}
# grpo's marker sits inside the other curves; its label goes to an empty spot (data coords) with a leader
LABEL_AT = {("gsm8k", "grpo"): (600, 86.3), ("math500", "grpo"): (680, 70.5)}
INK, INK2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    "#fcfcfb",
)


def subset_ids(run_dir: Path, ev: str) -> list[str]:
    ids = []
    with (run_dir / ev / "samples.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.append(json.loads(line)["problem_id"])
    return sorted(set(ids))


def bootstrap(xs: list[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def greedy_on_subset(run_dir: Path, ev: str, ids: list[str]) -> dict[str, Any] | None:
    f = run_dir / ev / "samples.jsonl"
    if not f.exists():
        return None
    recs: dict[str, dict] = {}
    with f.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if int(r.get("sample_idx", 0)) == 0:
                    recs[r["problem_id"]] = r
    missing = [i for i in ids if i not in recs]
    if missing:
        raise ValueError(f"{run_dir}/{ev}: {len(missing)} subset problems missing")
    acc = [1.0 if recs[i].get("correct") else 0.0 for i in ids]
    lo, hi = bootstrap(acc)
    return {
        "acc": 100 * sum(acc) / len(acc),
        "lo": 100 * lo,
        "hi": 100 * hi,
        "tokens": sum(recs[i]["num_tokens"] for i in ids) / len(ids),
        "n": len(ids),
    }


def laptop_points(root: Path, ev: str, ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for c in CKPTS:
        pts = []
        for b in BUDGETS[c]:
            p = greedy_on_subset(root / f"{c}_force{b}_think_tag", ev, ids)
            if p:
                pts.append(dict(p, label=f"stop at {b:,}", budget=b))
        p = greedy_on_subset(root / f"{c}_laptop_think_tag", ev, ids)
        if p:
            pts.append(dict(p, label="no stop", budget=None))
        out[c] = pts
    return out


def sampled_curves(curves: dict[str, Any], ev: str) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for c, rows in curves["evals"][ev]["sampled"].items():
        out[c] = [
            {
                "k": r["k"],
                "tokens": r["tokens_per_problem"],
                "maj": 100 * r["maj_at_k"]["mean"],
                "maj_lo": 100 * r["maj_at_k"]["lo"],
                "maj_hi": 100 * r["maj_at_k"]["hi"],
                "wvote": 100 * r["wvote_at_k"]["mean"] if r.get("wvote_at_k") else None,
            }
            for r in rows
        ]
    return out


def build(curves_json: Path, subset_from: Path, laptop_root: Path) -> dict[str, Any]:
    curves = json.loads(curves_json.read_text(encoding="utf-8"))
    res: dict[str, Any] = {}
    for ev, _ in EVALS:
        ids = subset_ids(subset_from, ev)
        n_json = curves["evals"][ev]["num_problems"]
        if len(ids) != n_json:
            raise ValueError(f"{ev}: subset has {len(ids)} ids, the curves JSON {n_json}")
        res[ev] = {
            "ids": len(ids),
            "laptop": laptop_points(laptop_root, ev, ids),
            "sampled": sampled_curves(curves, ev),
        }
    return res


def markdown(res: dict[str, Any]) -> str:
    out = [
        "# The final chart: accuracy against generated tokens (pinned 300-problem subsets)",
        "",
        "Greedy points are laptop runs (forced stop at a budget, or none); vote rows are the H100 n = 32 sample sets",
        "(T = 0.6). Accuracy in %, 95 % bootstrap CI over problems; tokens are mean generated tokens per problem.",
        "",
    ]
    for ev, name in EVALS:
        out += [
            f"## {name}",
            "",
            "| checkpoint | method | accuracy [95 % CI] | tokens |",
            "|---|---|---|---|",
        ]
        for c in CKPTS:
            for p in res[ev]["laptop"][c]:
                out.append(
                    f"| {c} | greedy, {p['label']} (laptop) | {p['acc']:.1f} [{p['lo']:.1f}, {p['hi']:.1f}] | {p['tokens']:,.0f} |"
                )
            for r in res[ev]["sampled"][c]:
                out.append(
                    f"| {c} | maj@{r['k']} (H100) | {r['maj']:.1f} [{r['maj_lo']:.1f}, {r['maj_hi']:.1f}] | {r['tokens']:,.0f} |"
                )
            for r in res[ev]["sampled"][c]:
                if c == "rft" and r["wvote"] is not None:
                    out.append(
                        f"| {c} | verifier-weighted vote @{r['k']} (H100) | {r['wvote']:.1f} | {r['tokens']:,.0f} |"
                    )
        out.append("")
    return "\n".join(out)


def plot(res: dict[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for ax, (ev, name) in zip(axes, EVALS, strict=True):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for c in CKPTS:
            col = COLOR[c]
            samp = res[ev]["sampled"][c]
            ax.plot(
                [r["tokens"] for r in samp],
                [r["maj"] for r in samp],
                linestyle=(0, (4, 2)),
                color=col,
                linewidth=1.4,
                marker="s",
                markersize=4,
                alpha=0.9,
            )
            if c == "rft":
                wv = [r for r in samp if r["wvote"] is not None]
                ax.plot(
                    [r["tokens"] for r in wv],
                    [r["wvote"] for r in wv],
                    linestyle=(0, (1, 1.5)),
                    color=col,
                    linewidth=1.6,
                    marker="s",
                    markersize=4,
                )
            pts = res[ev]["laptop"][c]
            forced = [p for p in pts if p["budget"] is not None]
            ax.plot(
                [p["tokens"] for p in pts],
                [p["acc"] for p in pts],
                color=col,
                linewidth=2,
                marker="o",
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1,
            )
            nostop = [p for p in pts if p["budget"] is None]
            if nostop:
                p = nostop[0]
                ax.plot(
                    [p["tokens"]],
                    [p["acc"]],
                    marker="o",
                    markersize=10,
                    markerfacecolor=SURFACE,
                    markeredgecolor=col,
                    markeredgewidth=2,
                    linestyle="none",
                )
            if forced and nostop:
                p = nostop[0]
                xy = (p["tokens"], p["acc"])
                if (ev, c) in LABEL_AT:
                    ax.annotate(
                        f"{c}, no stop",
                        xy,
                        xytext=LABEL_AT[(ev, c)],
                        textcoords="data",
                        color=INK2,
                        fontsize=8,
                        arrowprops={
                            "arrowstyle": "-",
                            "color": MUTED,
                            "linewidth": 0.8,
                            "shrinkB": 6,
                        },
                    )
                else:
                    ax.annotate(
                        f"{c}, no stop",
                        xy,
                        xytext=LABEL_OFFSET[c],
                        textcoords="offset points",
                        color=INK2,
                        fontsize=8,
                    )
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{v / 1024:g}k" if v >= 1024 else f"{v:g}")
        )
        ax.set_xlabel("generated tokens per problem (log scale)")
        ax.set_title(name, color=INK, fontsize=11, loc="left")
    axes[0].set_ylabel("accuracy, % (pinned 300 problems)")
    style = [
        Line2D(
            [],
            [],
            color=INK2,
            linewidth=2,
            marker="o",
            markersize=6,
            label="greedy, stop forced at a budget (laptop)",
        ),
        Line2D(
            [],
            [],
            color=INK2,
            linewidth=0,
            marker="o",
            markersize=10,
            markerfacecolor=SURFACE,
            markeredgewidth=2,
            label="greedy, no stop (laptop)",
        ),
        Line2D(
            [],
            [],
            color=INK2,
            linestyle=(0, (4, 2)),
            linewidth=1.4,
            marker="s",
            markersize=4,
            label="majority vote, k = 1..32 (H100 samples)",
        ),
        Line2D(
            [],
            [],
            color=INK2,
            linestyle=(0, (1, 1.5)),
            linewidth=1.6,
            marker="s",
            markersize=4,
            label="verifier-weighted vote, rft only (H100 samples)",
        ),
    ]
    ckpt = [Line2D([], [], color=COLOR[c], linewidth=3, label=c) for c in CKPTS]
    fig.legend(
        handles=ckpt + style,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
        labelcolor=INK2,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Every way of spending tokens on the same problems: forcing the stop, voting, and a verifier",
        color=INK,
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--curves", type=Path, required=True, help="EXP-022 analysis JSON (sampled curves)"
    )
    ap.add_argument(
        "--subset-from",
        type=Path,
        required=True,
        help="an n = 32 sample run: its problem ids are the subset",
    )
    ap.add_argument(
        "--laptop-root", type=Path, required=True, help="eval root holding the laptop runs"
    )
    ap.add_argument("--out", type=Path, required=True, help="PNG; .md and .json written next to it")
    a = ap.parse_args(argv)
    res = build(a.curves, a.subset_from, a.laptop_root)
    plot(res, a.out)
    a.out.with_suffix(".md").write_text(markdown(res), encoding="utf-8")
    a.out.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"wrote {a.out} and its .md, .json")


if __name__ == "__main__":
    main()

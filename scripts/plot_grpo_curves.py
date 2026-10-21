"""Plot the per-step training statistics of one or more ``rlvr-grpo`` runs, and their dev curves.

Usage (repo root, venv active):
    python scripts/plot_grpo_curves.py outputs/train/grpo_1p7b outputs/train/grpo_1p7b_seed1 \
        --labels main seed1 --dev-ref outputs/eval/rft_1.7b_dev \
        --out outputs/reports/figures/phase-4-grpo-curves.png

Each run directory holds ``reward_log.jsonl`` (one row per reward call: reward, finished_rate,
correct_rate, mean_tokens, zero_variance_group_rate, repeated_answer_group_rate, ...) and
``train_log.jsonl`` (TRL's per-step log: kl, entropy, grad_norm, step_time). The dev curve of a run is
read from ``outputs/eval/<run name>_dev_checkpoint-<step>/metrics.json`` next to the run's output
root (the folders ``rlvr-grpo``'s session script writes), with ``--dev-ref`` giving the step-0 point.
Six panels: reward, finished rate, mean tokens, KL to the reference, zero-variance groups, dev
accuracy; the first five are smoothed with a centred moving average (``--window``) over the raw
per-step points drawn faintly. A Markdown table with the window means at steps 1, 50, 100, 150, 200
is written next to the figure.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = [
    ("reward", "reward (finished, well-formed, correct)", "reward"),
    ("finished_rate", "finished rate", "reward"),
    ("mean_tokens", "mean completion tokens", "reward"),
    ("kl", "KL to the reference (k3)", "train"),
    ("zero_variance_group_rate", "zero-variance groups", "reward"),
]
MARKS = (1, 50, 100, 150, 200)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def smooth(ys: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(ys)
    half = window // 2
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - half), min(len(ys), i + half + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out


def series(run: Path, key: str, source: str) -> tuple[list[int], list[float]]:
    rows = read_jsonl(run / ("reward_log.jsonl" if source == "reward" else "train_log.jsonl"))
    xs, ys = [], []
    for r in rows:
        if key not in r or r[key] is None:
            continue
        step = r.get("global_step", r.get("step"))
        if step is None:
            continue
        # reward_log counts from 0, train_log from 1: both become 1-based here
        xs.append(int(step) + (1 if source == "reward" else 0))
        ys.append(float(r[key]))
    return xs, ys


def dev_curve(run: Path, eval_root: Path) -> dict[str, list[tuple[int, float]]]:
    """``<eval_root>/<run name>_dev_checkpoint-<step>/metrics.json`` -> {eval: [(step, acc)]}."""
    out: dict[str, list[tuple[int, float]]] = {}
    pat = re.compile(re.escape(run.name) + r"_dev_checkpoint-(\d+)$")
    for d in sorted(eval_root.glob(run.name + "_dev_checkpoint-*")):
        m = pat.match(d.name)
        if not m or not (d / "metrics.json").exists():
            continue
        metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        for ev, v in metrics.items():
            if isinstance(v, dict) and "pass@1" in v:
                out.setdefault(ev, []).append((int(m.group(1)), float(v["pass@1"]["mean"])))
    for ev in out:
        out[ev].sort()
    return out


def window_mean(xs: list[int], ys: list[float], at: int, window: int) -> float | None:
    vals = [y for x, y in zip(xs, ys, strict=True) if abs(x - at) <= window // 2]
    return sum(vals) / len(vals) if vals else None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--eval-root", type=Path, default=None, help="default: <run>/../../eval")
    ap.add_argument(
        "--dev-ref", type=Path, default=None, help="the step-0 dev eval (the start checkpoint)"
    )
    ap.add_argument("--window", type=int, default=9)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args(argv)
    labels = args.labels or [r.name for r in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("--labels must match the runs")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    table: dict[str, dict[str, list[float | None]]] = {}
    for i, (run, label) in enumerate(zip(args.runs, labels, strict=True)):
        c = colors[i % len(colors)]
        table[label] = {}
        for ax, (key, title, source) in zip(axes, PANELS, strict=False):
            xs, ys = series(run, key, source)
            if not xs:
                continue
            ax.plot(xs, ys, color=c, alpha=0.18, linewidth=0.8)
            ax.plot(xs, smooth(ys, args.window), color=c, linewidth=1.8, label=label)
            ax.set_title(title)
            ax.set_xlabel("step")
            table[label][key] = [window_mean(xs, ys, m, args.window) for m in MARKS]
        eval_root = args.eval_root or (run.resolve().parent.parent / "eval")
        curve = dev_curve(run, eval_root)
        ref = {}
        if args.dev_ref and (args.dev_ref / "metrics.json").exists():
            metrics = json.loads((args.dev_ref / "metrics.json").read_text(encoding="utf-8"))
            ref = {
                ev: float(v["pass@1"]["mean"])
                for ev, v in metrics.items()
                if isinstance(v, dict) and "pass@1" in v
            }
        for j, (ev, pts) in enumerate(sorted(curve.items())):
            xs = [0] + [p[0] for p in pts] if ev in ref else [p[0] for p in pts]
            ys = [ref[ev]] + [p[1] for p in pts] if ev in ref else [p[1] for p in pts]
            axes[5].plot(
                xs,
                ys,
                marker="o",
                color=c,
                linestyle="-" if j == 0 else "--",
                label=f"{label} {ev}",
            )
    axes[5].set_title("dev accuracy (greedy think-tag, 250 + 250)")
    axes[5].set_xlabel("step")
    for ax in axes:
        ax.grid(alpha=0.3)
        if ax.has_data():
            ax.legend(fontsize=8)
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)

    lines = [
        f"Window means (centred, {args.window} steps) at steps {', '.join(map(str, MARKS))}:",
        "",
    ]
    lines.append("| run | statistic | " + " | ".join(f"step {m}" for m in MARKS) + " |")
    lines.append("|---|---|" + "---|" * len(MARKS))
    for label, stats in table.items():
        for key, _, _ in PANELS:
            vals = stats.get(key)
            if not vals:
                continue
            cells = [
                "-"
                if v is None
                else (
                    f"{v:.0f}"
                    if key == "mean_tokens"
                    else f"{v:.4f}"
                    if key == "kl"
                    else f"{v:.3f}"
                )
                for v in vals
            ]
            lines.append(f"| {label} | {key} | " + " | ".join(cells) + " |")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(args.out, args.out.with_suffix(".md"))


if __name__ == "__main__":
    main()

"""Accuracy against generated tokens: greedy, majority vote and pass@k per checkpoint (EXP-017, D6).

Usage (repo root, venv active):
    python scripts/plot_accuracy_vs_tokens.py \
        --samples sft=outputs/samples/sft_1.7b_n32_think_tag_t06 rft=... grpo=... \
        --greedy sft=outputs/eval/sft_1.7b_think_tag rft=... grpo=... \
        --reference-greedy grpo --evals gsm8k math500 \
        --out outputs/reports/figures/phase-5-accuracy-vs-tokens.png

Each ``--samples`` directory is an ``rlvr-eval`` run with n > 1 (``<eval>/samples.jsonl``: one row
per sample with ``problem_id``, ``sample_idx``, ``extracted``, ``correct``, ``num_tokens``); each
``--greedy`` directory a greedy run of the same checkpoint. Per eval the runs are restricted to the
problems every given run shares (the n = 32 sets are pinned 300-problem subsets; the greedy runs
are full sets), and that count is printed and written. For every sampled run and k in ``--ks``
(k <= n): maj@k and pass@k with 95 % bootstrap CIs over problems, and the expected generated tokens
per problem at k samples (k x the problem's mean sample length, averaged over problems). Greedy
runs contribute one point (pass@1, mean tokens). With ``--reference-greedy LABEL`` every maj@k and
pass@k gets a paired CI against that run's greedy correctness on the same problems, which is the
EXP-017 question: at which k does voting over an earlier checkpoint match the RL policy's greedy
number, and at what token cost.

One panel per eval: x = generated tokens per problem (log scale), y = accuracy; maj@k solid with
markers, pass@k dashed, greedy as a star. A Markdown table and a JSON with every number go next to
the figure.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from rlvr.eval.extraction import normalize_for_vote
from rlvr.eval.metrics import bootstrap_ci, majority_vote, paired_bootstrap_diff, pass_at_k

DEFAULT_KS = (1, 2, 4, 8, 16, 32, 64)


def parse_labelled(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"expected LABEL=DIR, got {item!r}")
        label, d = item.split("=", 1)
        out[label] = Path(d)
    return out


def load_samples(run_dir: Path, eval_name: str) -> dict[str, list[dict[str, Any]]]:
    """``{problem_id: [sample rows sorted by sample_idx]}``; empty when the eval is absent."""
    path = run_dir / eval_name / "samples.jsonl"
    if not path.exists():
        return {}
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["problem_id"]].append(r)
    for pid in rows:
        rows[pid].sort(key=lambda r: r["sample_idx"])
    return dict(rows)


def load_scores(run_dir: Path, eval_name: str) -> dict[str, dict[int, float]] | None:
    """``{problem_id: {sample_idx: score}}`` from ``<eval>/scores.jsonl`` (an outcome reward model's
    logits, EXP-022), or None when the run has none."""
    path = run_dir / eval_name / "scores.jsonl"
    if not path.exists():
        return None
    out: dict[str, dict[int, float]] = defaultdict(dict)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["problem_id"]][int(r["sample_idx"])] = float(r["score"])
    return dict(out)


def scored_at_k_per_problem(
    rows: list[dict[str, Any]],
    scores: dict[int, float],
    k: int,
    *,
    repeats: int = 20,
    seed: int = 0,
) -> tuple[float, float]:
    """(best-of-k hit rate, reward-weighted-vote hit rate) over the same seeded draws as maj@k:
    best-of-k takes the drawn sample with the highest score; the weighted vote sums each drawn
    sample's sigmoid(score) on its normalised answer and takes the heaviest key."""
    n = len(rows)
    rng = np.random.default_rng(seed)
    correct_keys = {
        normalize_for_vote(r["extracted"]) for r in rows if r["correct"] and r["extracted"]
    }
    n_rep = 1 if k >= n else repeats
    bon_hits = wv_hits = 0.0
    for _ in range(n_rep):
        idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)
        best = max(idx, key=lambda j: scores.get(int(rows[j]["sample_idx"]), float("-inf")))
        bon_hits += float(bool(rows[best]["correct"]))
        weights: dict[str, float] = {}
        for j in idx:
            a = rows[j]["extracted"]
            if a is None:
                continue
            w = 1.0 / (1.0 + np.exp(-scores.get(int(rows[j]["sample_idx"]), 0.0)))
            key = normalize_for_vote(a)
            weights[key] = weights.get(key, 0.0) + w
        if weights:
            heaviest = max(weights, key=lambda key: weights[key])
            wv_hits += float(heaviest in correct_keys)
    return bon_hits / n_rep, wv_hits / n_rep


def maj_at_k_per_problem(
    answers: list[str | None], correct: list[bool], k: int, *, repeats: int = 20, seed: int = 0
) -> float:
    """Majority-vote hit rate over random k-subsets of one problem's n samples (1.0 or 0.0 at k = n)."""
    n = len(answers)
    rng = np.random.default_rng(seed)
    correct_keys = {normalize_for_vote(a) for a, c in zip(answers, correct, strict=True) if c and a}
    n_rep = 1 if k >= n else repeats
    hits = 0.0
    for _ in range(n_rep):
        idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)
        voted = majority_vote([answers[i] for i in idx], normalize=normalize_for_vote)
        hits += float(voted is not None and voted in correct_keys)
    return hits / n_rep


def curves_for_run(
    samples: dict[str, list[dict[str, Any]]],
    ids: list[str],
    ks: tuple[int, ...],
    ref_greedy: dict[str, bool] | None,
    scores: dict[str, dict[int, float]] | None = None,
) -> list[dict[str, Any]]:
    """One record per k: tokens, maj@k, pass@k (with CIs) and the paired differences; with
    ``scores``, best-of-k and the reward-weighted vote too, paired against maj@k."""
    n = len(samples[ids[0]])
    if any(len(samples[i]) != n for i in ids):
        raise ValueError("ragged sample counts across problems")
    out = []
    ref = [float(ref_greedy[i]) for i in ids] if ref_greedy else None
    for k in ks:
        if k > n:
            continue
        maj = [
            maj_at_k_per_problem(
                [r["extracted"] for r in samples[i]], [bool(r["correct"]) for r in samples[i]], k
            )
            for i in ids
        ]
        pk = [pass_at_k(n, sum(bool(r["correct"]) for r in samples[i]), k) for i in ids]
        tokens = [k * float(np.mean([r["num_tokens"] for r in samples[i]])) for i in ids]
        rec: dict[str, Any] = {
            "k": k,
            "n": n,
            "tokens_per_problem": float(np.mean(tokens)),
            "maj_at_k": bootstrap_ci(maj, seed=k).as_dict(),
            "pass_at_k": bootstrap_ci(pk, seed=k).as_dict(),
        }
        if ref is not None:
            rec["maj_minus_ref_greedy"] = paired_bootstrap_diff(maj, ref, seed=k).as_dict()
            rec["pass_minus_ref_greedy"] = paired_bootstrap_diff(pk, ref, seed=k).as_dict()
        if scores is not None:
            pairs = [scored_at_k_per_problem(samples[i], scores.get(i, {}), k) for i in ids]
            bon = [a for a, _ in pairs]
            wv = [b for _, b in pairs]
            rec["bon_at_k"] = bootstrap_ci(bon, seed=k).as_dict()
            rec["wvote_at_k"] = bootstrap_ci(wv, seed=k).as_dict()
            rec["bon_minus_maj"] = paired_bootstrap_diff(bon, maj, seed=k).as_dict()
            rec["wvote_minus_maj"] = paired_bootstrap_diff(wv, maj, seed=k).as_dict()
            if ref is not None:
                rec["bon_minus_ref_greedy"] = paired_bootstrap_diff(bon, ref, seed=k).as_dict()
                rec["wvote_minus_ref_greedy"] = paired_bootstrap_diff(wv, ref, seed=k).as_dict()
        out.append(rec)
    return out


def greedy_point(samples: dict[str, list[dict[str, Any]]], ids: list[str]) -> dict[str, Any]:
    correct = [float(samples[i][0]["correct"]) for i in ids]
    tokens = [float(samples[i][0]["num_tokens"]) for i in ids]
    return {
        "k": 1,
        "n": 1,
        "tokens_per_problem": float(np.mean(tokens)),
        "pass_at_1": bootstrap_ci(correct).as_dict(),
    }


def analyse(
    sample_runs: dict[str, Path],
    greedy_runs: dict[str, Path],
    evals: list[str],
    ks: tuple[int, ...],
    reference_greedy: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"evals": {}}
    for ev in evals:
        loaded_s = {lab: load_samples(d, ev) for lab, d in sample_runs.items()}
        loaded_g = {lab: load_samples(d, ev) for lab, d in greedy_runs.items()}
        loaded_s = {k: v for k, v in loaded_s.items() if v}
        loaded_g = {k: v for k, v in loaded_g.items() if v}
        if not loaded_s and not loaded_g:
            continue
        id_sets = [set(v) for v in (*loaded_s.values(), *loaded_g.values())]
        ids = sorted(set.intersection(*id_sets))
        if not ids:
            raise SystemExit(f"{ev}: the runs share no problem ids")
        ref = None
        if reference_greedy:
            if reference_greedy not in loaded_g:
                raise SystemExit(f"{ev}: --reference-greedy {reference_greedy!r} has no greedy run")
            ref = {i: bool(loaded_g[reference_greedy][i][0]["correct"]) for i in ids}
        result["evals"][ev] = {
            "num_problems": len(ids),
            "reference_greedy": reference_greedy,
            "sampled": {
                lab: curves_for_run(s, ids, ks, ref, load_scores(sample_runs[lab], ev))
                for lab, s in loaded_s.items()
            },
            "scored_runs": [
                lab for lab in loaded_s if load_scores(sample_runs[lab], ev) is not None
            ],
            "greedy": {lab: greedy_point(g, ids) for lab, g in loaded_g.items()},
        }
        print(f"[accuracy-vs-tokens] {ev}: {len(ids)} shared problems", flush=True)
    if not result["evals"]:
        raise SystemExit("no eval found in any run directory")
    return result


def _fmt(e: dict[str, Any] | None) -> str:
    if not e:
        return "-"
    return f"{100 * e['mean']:.1f} [{100 * e['lo']:.1f}, {100 * e['hi']:.1f}]"


def _fmt_diff(e: dict[str, Any] | None) -> str:
    if not e:
        return "-"
    return f"{100 * e['mean']:+.1f} [{100 * e['lo']:+.1f}, {100 * e['hi']:+.1f}]"


def markdown(result: dict[str, Any]) -> str:
    lines = []
    for ev, block in result["evals"].items():
        ref = block["reference_greedy"]
        lines.append(
            f"**{ev}** ({block['num_problems']} shared problems"
            + (f"; paired against `{ref}` greedy)" if ref else ")")
        )
        lines.append("")
        scored = bool(block.get("scored_runs"))
        lines.append(
            "| run | k | tokens / problem | maj@k | pass@k | maj@k - ref greedy | pass@k - ref greedy |"
            + (
                " best-of-k | weighted vote | best-of-k - maj@k | weighted vote - maj@k |"
                if scored
                else ""
            )
        )
        lines.append("|---|---|---|---|---|---|---|" + ("---|---|---|---|" if scored else ""))
        for lab, g in block["greedy"].items():
            lines.append(
                f"| {lab} greedy | 1 | {g['tokens_per_problem']:.0f} | {_fmt(g['pass_at_1'])} | "
                f"{_fmt(g['pass_at_1'])} | - | - |" + (" - | - | - | - |" if scored else "")
            )
        for lab, recs in block["sampled"].items():
            for r in recs:
                lines.append(
                    f"| {lab} sampled | {r['k']} | {r['tokens_per_problem']:.0f} | "
                    f"{_fmt(r['maj_at_k'])} | {_fmt(r['pass_at_k'])} | "
                    f"{_fmt_diff(r.get('maj_minus_ref_greedy'))} | "
                    f"{_fmt_diff(r.get('pass_minus_ref_greedy'))} |"
                    + (
                        f" {_fmt(r.get('bon_at_k'))} | {_fmt(r.get('wvote_at_k'))} | "
                        f"{_fmt_diff(r.get('bon_minus_maj'))} | {_fmt_diff(r.get('wvote_minus_maj'))} |"
                        if scored
                        else ""
                    )
                )
        lines.append("")
    return "\n".join(lines)


def plot(result: dict[str, Any], out: Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evals = list(result["evals"])
    fig, axes = plt.subplots(1, len(evals), figsize=(6.2 * len(evals), 4.6), squeeze=False)
    for ax, ev in zip(axes[0], evals, strict=True):
        block = result["evals"][ev]
        colours = {}
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, lab in enumerate([*block["sampled"], *block["greedy"]]):
            colours.setdefault(lab, cycle[i % len(cycle)])
        for lab, recs in block["sampled"].items():
            x = [r["tokens_per_problem"] for r in recs]
            maj = [100 * r["maj_at_k"]["mean"] for r in recs]
            lo = [100 * r["maj_at_k"]["lo"] for r in recs]
            hi = [100 * r["maj_at_k"]["hi"] for r in recs]
            pk = [100 * r["pass_at_k"]["mean"] for r in recs]
            ax.plot(x, maj, "-o", color=colours[lab], label=f"{lab} maj@k")
            ax.fill_between(x, lo, hi, color=colours[lab], alpha=0.12, linewidth=0)
            ax.plot(x, pk, "--", color=colours[lab], alpha=0.7, label=f"{lab} pass@k")
            if recs and "bon_at_k" in recs[0]:
                ax.plot(
                    x,
                    [100 * r["bon_at_k"]["mean"] for r in recs],
                    ":",
                    marker="s",
                    markersize=4,
                    color=colours[lab],
                    label=f"{lab} best-of-k (ORM)",
                )
                ax.plot(
                    x,
                    [100 * r["wvote_at_k"]["mean"] for r in recs],
                    "-.",
                    marker="^",
                    markersize=4,
                    color=colours[lab],
                    label=f"{lab} weighted vote (ORM)",
                )
        for lab, g in block["greedy"].items():
            ax.plot(
                [g["tokens_per_problem"]],
                [100 * g["pass_at_1"]["mean"]],
                marker="*",
                markersize=14,
                linestyle="none",
                color=colours[lab],
                label=f"{lab} greedy",
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("generated tokens per problem (expected, log scale)")
        ax.set_ylabel("accuracy (%)")
        ax.set_title(f"{ev} ({block['num_problems']} problems)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
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
    ap.add_argument("--samples", nargs="*", default=[], help="LABEL=DIR of n>1 runs")
    ap.add_argument("--greedy", nargs="*", default=[], help="LABEL=DIR of greedy runs")
    ap.add_argument(
        "--reference-greedy", default=None, help="label of the greedy run to pair against"
    )
    ap.add_argument("--evals", nargs="+", default=["gsm8k", "math500"])
    ap.add_argument("--ks", nargs="*", type=int, default=list(DEFAULT_KS))
    ap.add_argument("--out", type=Path, required=True, help="PNG; .md and .json written next to it")
    ap.add_argument("--title", default="")
    args = ap.parse_args(argv)
    result = analyse(
        parse_labelled(args.samples),
        parse_labelled(args.greedy),
        args.evals,
        tuple(args.ks),
        args.reference_greedy,
    )
    plot(result, args.out, args.title)
    md = markdown(result)
    args.out.with_suffix(".md").write_text(md, encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(md)
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()

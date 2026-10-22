"""The traces (EXP-023, Phase 6): what a loop is, where it starts, what precedes it, and what the
training changed in the text. From first-sample files of greedy runs; no GPU.

Usage (repo root, venv active):
    python scripts/analyse_traces.py --runs sft=../h100_session_archive/outputs/eval/sft_1.7b_think_tag \\
        grpo=../h100_p4_session2_archive/outputs/eval/grpo_1.7b_think_tag --evals gsm8k math500 \\
        --pairs sft:grpo --out outputs/reports/phase-6-traces.md

Per completion: word count; the think block's share of the text; the *loop onset*, the position
(as a fraction of the completion's words) of the first ``--ngram``-word window that recurs later,
and the *repeat unit*, the distance in words to its next occurrence; whether a boxed answer or an
answer tag precedes the loop onset; counts of "Wait" and of verification phrases. Per run and eval:
quantiles of those over finished and truncated completions, the share of truncated completions with
a loop, the answer-before-loop rate among them; for each ``--pairs A:B`` the two-sample
Kolmogorov-Smirnov distance between A's and B's repeat-unit lengths and loop onsets among truncated
completions (the pre-registered test of "GRPO's loops have the same structure as SFT's"). A Markdown
table, a JSON with every number, and a figure (onset and repeat-unit histograms) next to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

WAIT_RE = re.compile(r"\bWait\b")
VERIFY_RE = re.compile(
    r"\b(let me (?:check|verify|double-check|re-check|recheck|confirm)|double-check|verify|"
    r"let's check|let me make sure|to confirm)\b",
    re.IGNORECASE,
)
ANSWER_RE = re.compile(r"\\boxed\{|<answer>|the (?:final )?answer is", re.IGNORECASE)
THINK_CLOSE = "</think>"


def first_repeat(words: list[str], n: int = 50) -> tuple[int | None, int | None]:
    """(index of the first n-gram that recurs later, distance in words to its next occurrence), or
    (None, None) when no n-gram recurs."""
    if len(words) <= n:
        return None, None
    seen: dict[tuple[str, ...], int] = {}
    best_i, best_d = None, None
    for j in range(len(words) - n + 1):
        key = tuple(words[j : j + n])
        if key in seen:
            i = seen[key]
            if best_i is None or i < best_i:
                best_i, best_d = i, j - i
        else:
            seen[key] = j
    return best_i, best_d


def analyse_completion(text: str, *, truncated: bool, ngram: int) -> dict[str, Any]:
    words = text.split()
    n_words = len(words)
    close = text.find(THINK_CLOSE)
    think_share = (close / len(text)) if (close >= 0 and text) else (1.0 if text else 0.0)
    onset_i, unit = first_repeat(words, ngram)
    onset_frac = (onset_i / n_words) if (onset_i is not None and n_words) else None
    answer_before_loop = None
    if onset_i is not None:
        onset_char = len(" ".join(words[:onset_i]))
        m = list(ANSWER_RE.finditer(text[:onset_char]))
        answer_before_loop = bool(m)
    return {
        "words": n_words,
        "truncated": truncated,
        "think_share": think_share,
        "has_loop": onset_i is not None,
        "loop_onset_frac": onset_frac,
        "repeat_unit_words": unit,
        "answer_before_loop": answer_before_loop,
        "wait": len(WAIT_RE.findall(text)),
        "verify": len(VERIFY_RE.findall(text)),
    }


def load_first(run_dir: Path, ev: str) -> list[dict[str, Any]]:
    p = Path(run_dir) / ev / "samples.jsonl"
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("sample_idx", 0) == 0:
                    out.append(r)
    return out


def q(xs: list[float], p: float) -> float | None:
    return float(np.quantile(xs, p)) if xs else None


def summarise(per: list[dict[str, Any]]) -> dict[str, Any]:
    tr = [c for c in per if c["truncated"]]
    fi = [c for c in per if not c["truncated"]]
    loops_tr = [c for c in tr if c["has_loop"]]
    loops_fi = [c for c in fi if c["has_loop"]]

    def block(cs: list[dict[str, Any]]) -> dict[str, Any]:
        loops = [c for c in cs if c["has_loop"]]
        return {
            "n": len(cs),
            "words_median": q([c["words"] for c in cs], 0.5),
            "has_loop_rate": (len(loops) / len(cs)) if cs else None,
            "loop_onset_median": q([c["loop_onset_frac"] for c in loops], 0.5),
            "loop_onset_q25": q([c["loop_onset_frac"] for c in loops], 0.25),
            "repeat_unit_median": q([c["repeat_unit_words"] for c in loops], 0.5),
            "repeat_unit_q75": q([c["repeat_unit_words"] for c in loops], 0.75),
            "answer_before_loop_rate": (
                sum(bool(c["answer_before_loop"]) for c in loops) / len(loops)
            )
            if loops
            else None,
            "wait_mean": float(np.mean([c["wait"] for c in cs])) if cs else None,
            "verify_mean": float(np.mean([c["verify"] for c in cs])) if cs else None,
        }

    return {
        "n": len(per),
        "truncated_rate": len(tr) / len(per) if per else None,
        "truncated": block(tr),
        "finished": block(fi),
        "_units_truncated": [c["repeat_unit_words"] for c in loops_tr],
        "_onsets_truncated": [c["loop_onset_frac"] for c in loops_tr],
        "_units_finished": [c["repeat_unit_words"] for c in loops_fi],
    }


def ks_distance(a: list[float], b: list[float]) -> float | None:
    if not a or not b:
        return None
    from scipy.stats import ks_2samp

    return float(ks_2samp(a, b).statistic)


def analyse(
    runs: dict[str, Path], evals: list[str], pairs: list[tuple[str, str]], ngram: int
) -> dict[str, Any]:
    out: dict[str, Any] = {"ngram": ngram, "evals": {}}
    for ev in evals:
        block: dict[str, Any] = {"runs": {}, "pairs": {}}
        for label, d in runs.items():
            if not (Path(d) / ev / "samples.jsonl").exists():
                continue
            recs = load_first(Path(d), ev)
            per = [
                analyse_completion(
                    r.get("text", ""), truncated=(r.get("finish_reason") == "length"), ngram=ngram
                )
                for r in recs
            ]
            block["runs"][label] = summarise(per)
        for a, b in pairs:
            if a in block["runs"] and b in block["runs"]:
                ra, rb = block["runs"][a], block["runs"][b]
                block["pairs"][f"{a}:{b}"] = {
                    "ks_repeat_unit_truncated": ks_distance(
                        ra["_units_truncated"], rb["_units_truncated"]
                    ),
                    "ks_loop_onset_truncated": ks_distance(
                        ra["_onsets_truncated"], rb["_onsets_truncated"]
                    ),
                    "n_a": len(ra["_units_truncated"]),
                    "n_b": len(rb["_units_truncated"]),
                }
        if block["runs"]:
            out["evals"][ev] = block
    if not out["evals"]:
        raise SystemExit("no run has any of the evals")
    return out


def _f(x: float | None, pct: bool = False, nd: int = 1) -> str:
    if x is None:
        return "-"
    return f"{100 * x:.{nd}f}" if pct else f"{x:.{nd}f}"


def markdown(result: dict[str, Any]) -> str:
    lines = [
        f"Loop = the first {result['ngram']}-word window that recurs later in the completion; onset = its position as a fraction of the completion's words; unit = words to its next occurrence. First samples, greedy.",
        "",
    ]
    for ev, block in result["evals"].items():
        lines += [
            f"**{ev}**",
            "",
            "| run | n | truncated % | truncated: words (median) | with a loop % | loop onset (median) | repeat unit words (median) | answer before loop % | Wait / completion (trunc. / fin.) | verify phrases (trunc. / fin.) | finished with a loop % |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for label, r in block["runs"].items():
            t, f = r["truncated"], r["finished"]
            lines.append(
                f"| {label} | {r['n']} | {_f(r['truncated_rate'], True)} | {_f(t['words_median'], nd=0)} | {_f(t['has_loop_rate'], True)} | "
                f"{_f(t['loop_onset_median'], nd=2)} | {_f(t['repeat_unit_median'], nd=0)} | {_f(t['answer_before_loop_rate'], True)} | "
                f"{_f(t['wait_mean'])} / {_f(f['wait_mean'])} | {_f(t['verify_mean'])} / {_f(f['verify_mean'])} | {_f(f['has_loop_rate'], True)} |"
            )
        if block["pairs"]:
            lines += [
                "",
                "| pair | KS distance, repeat-unit length (truncated) | KS distance, loop onset (truncated) | n |",
                "|---|---|---|---|",
            ]
            for key, p in block["pairs"].items():
                lines.append(
                    f"| {key} | {_f(p['ks_repeat_unit_truncated'], nd=2)} | {_f(p['ks_loop_onset_truncated'], nd=2)} | {p['n_a']} / {p['n_b']} |"
                )
        lines.append("")
    return "\n".join(lines)


def plot(result: dict[str, Any], out: Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evals = list(result["evals"])
    fig, axes = plt.subplots(2, len(evals), figsize=(6 * len(evals), 7.5), squeeze=False)
    for col, ev in enumerate(evals):
        block = result["evals"][ev]
        ax_on, ax_unit = axes[0][col], axes[1][col]
        for label, r in block["runs"].items():
            if r["_onsets_truncated"]:
                ax_on.hist(
                    r["_onsets_truncated"],
                    bins=20,
                    range=(0, 1),
                    histtype="step",
                    linewidth=1.5,
                    label=f"{label} (n={len(r['_onsets_truncated'])})",
                )
            if r["_units_truncated"]:
                ax_unit.hist(
                    np.clip(r["_units_truncated"], 0, 400),
                    bins=40,
                    range=(0, 400),
                    histtype="step",
                    linewidth=1.5,
                    label=label,
                )
        ax_on.set_title(f"{ev}: loop onset in truncated completions")
        ax_on.set_xlabel("onset (fraction of the completion's words)")
        ax_unit.set_title(f"{ev}: repeat-unit length in truncated completions")
        ax_unit.set_xlabel("words between a window and its next occurrence (clipped at 400)")
        for ax in (ax_on, ax_unit):
            ax.set_ylabel("completions")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def strip_private(result: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(result))
    for block in out["evals"].values():
        for r in block["runs"].values():
            for k in list(r):
                if k.startswith("_"):
                    del r[k]
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs", nargs="+", required=True, help="LABEL=DIR of greedy runs")
    ap.add_argument("--evals", nargs="+", default=["gsm8k", "math500"])
    ap.add_argument("--pairs", nargs="*", default=[], help="A:B pairs for the KS distances")
    ap.add_argument("--ngram", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True, help="Markdown; .json and .png next to it")
    ap.add_argument("--title", default="")
    args = ap.parse_args(argv)
    runs = {}
    for item in args.runs:
        label, d = item.split("=", 1)
        runs[label] = Path(d)
    pairs = [tuple(p.split(":", 1)) for p in args.pairs]
    result = analyse(runs, args.evals, pairs, args.ngram)
    md = markdown(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8", newline="\n")
    args.out.with_suffix(".json").write_text(
        json.dumps(strip_private(result), indent=2), encoding="utf-8", newline="\n"
    )
    plot(result, args.out.with_suffix(".png"), args.title)
    print(md)
    print(f"wrote {args.out}, .json, .png")


if __name__ == "__main__":
    main()

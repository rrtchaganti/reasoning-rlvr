"""Project the remaining wall clock of a running ``rlvr-eval`` sampling job from its log.

Why: the L4 session of 17 Oct 2026 projected EXP-002 at 5-6 hours from the GSM8K rate and the
Countdown set alone wanted 22 (journal, section 17). vLLM's output rate falls with the completion
length, and tqdm's ETA is anchored on the sequences that finished first, which are the short ones.
This script applies the rule from the handoff instead: remaining sequences times the mean completion
length seen for that set in the nearest run, divided by the output rate on this set's own progress
line. The verdict says whether the step stays under ``--max-hours``.

    python scripts/project_sampling.py outputs/a100_exp002.log --from-metrics outputs/eval/base_1.7b_think_tag_4096/countdown/metrics.json --rate 189 --max-hours 8
    python scripts/project_sampling.py outputs/a100_exp002.log --mean-tokens 2700 --rate 189

The log is whatever ``tee`` wrote; the last ``Processed prompts`` line is used (tqdm separates
updates with carriage returns, which count as line breaks here).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROGRESS = re.compile(
    r"Processed prompts:\s*\d+%\|[^|]*\|\s*(?P<done>\d+)/(?P<total>\d+)\s*"
    r"\[(?P<elapsed>[\d:]+)<(?P<eta>[^,\]]*)"
    r"(?:,.*?output:\s*(?P<output>[\d.]+)\s*toks/s)?"
)  # the speed part is absent on the first (0 %) update


def _seconds(clock: str) -> float:
    parts = [float(p) for p in clock.split(":")]
    out = 0.0
    for p in parts:
        out = out * 60 + p
    return out


def parse_progress(text: str) -> dict | None:
    """The last ``Processed prompts`` update in ``text``, or None when there is none yet."""
    last = None
    for line in re.split(r"[\r\n]+", text):
        m = PROGRESS.search(line)
        if m:
            last = m
    if last is None:
        return None
    eta = last["eta"].strip()
    return {
        "done": int(last["done"]),
        "total": int(last["total"]),
        "elapsed_s": _seconds(last["elapsed"]),
        "tqdm_eta_s": None if "?" in eta else _seconds(eta),
        "output_tps": float(last["output"] or 0.0),
    }


def mean_tokens_from_metrics(path: Path, eval_name: str | None = None) -> float:
    """``tokens.mean`` of a finished run: one eval's ``metrics.json`` or a run's with ``eval_name``."""
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    if "tokens" not in m:
        if eval_name is None or eval_name not in m:
            raise KeyError(f"{path}: pass the eval name; found {sorted(m)}")
        m = m[eval_name]
    return float(m["tokens"]["mean"])


def project(progress: dict, *, mean_tokens: float, rate: float, max_hours: float) -> dict:
    """Remaining hours and cost; ``stay`` when the projection is at most ``max_hours``."""
    if progress["output_tps"] <= 0:
        raise ValueError("output rate must be positive; wait for the first progress update")
    remaining = progress["total"] - progress["done"]
    remaining_tokens = remaining * mean_tokens
    hours = remaining_tokens / progress["output_tps"] / 3600
    done = progress["done"]
    tokens_so_far = progress["output_tps"] * progress["elapsed_s"]
    return {
        "done": done,
        "total": progress["total"],
        "elapsed_h": progress["elapsed_s"] / 3600,
        "output_tps": progress["output_tps"],
        "mean_tokens_finished_so_far": tokens_so_far / done if done else None,
        "mean_tokens_assumed": mean_tokens,
        "remaining_sequences": remaining,
        "remaining_tokens": remaining_tokens,
        "projected_hours": hours,
        "tqdm_eta_hours": (
            None if progress.get("tqdm_eta_s") is None else progress["tqdm_eta_s"] / 3600
        ),
        "rate_per_hour": rate,
        "projected_cost": hours * rate,
        "max_hours": max_hours,
        "stay": hours <= max_hours,
    }


def render(p: dict, currency: str = "INR") -> str:
    verdict = (
        "CONTINUE: under the limit"
        if p["stay"]
        else "STOP and ask: the projection is above the limit"
    )
    finished = p["mean_tokens_finished_so_far"]
    tqdm_eta = p["tqdm_eta_hours"]
    return "\n".join(
        [
            "# Sampling projection",
            "",
            f"- progress: {p['done']:,} / {p['total']:,} sequences after {p['elapsed_h']:.2f} h "
            f"at {p['output_tps']:,.0f} output tokens/s",
            f"- completions finished so far average {finished:,.0f} tokens"
            if finished
            else "- no completions finished yet",
            f"- assumed mean for the rest: {p['mean_tokens_assumed']:,.0f} tokens "
            f"(the short ones finish first, so the rest run longer than the average so far)",
            f"- remaining: {p['remaining_sequences']:,} sequences, "
            f"{p['remaining_tokens'] / 1e6:.1f} M tokens",
            f"- projected: {p['projected_hours']:.1f} h more, about {currency} "
            f"{p['projected_cost']:.0f} at {currency} {p['rate_per_hour']:.0f}/h"
            + (f" (tqdm says {tqdm_eta:.1f} h; it underestimates)" if tqdm_eta is not None else ""),
            f"- rule: continue if at most {p['max_hours']:g} h more",
            "",
            f"verdict: {verdict}",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log", type=Path, help="the log tee'd from the running rlvr-eval")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--mean-tokens", type=float, help="expected mean completion length")
    src.add_argument(
        "--from-metrics", type=Path, help="metrics.json of the nearest finished run on this set"
    )
    ap.add_argument("--eval", default=None, help="eval name inside a run-level metrics.json")
    ap.add_argument("--rate", type=float, default=189.0, help="price of this node per hour")
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--currency", default="INR")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    progress = parse_progress(args.log.read_text(encoding="utf-8", errors="replace"))
    if progress is None:
        raise SystemExit(f"{args.log}: no 'Processed prompts' line yet; wait for the first update")
    mean_tokens = (
        args.mean_tokens
        if args.mean_tokens is not None
        else mean_tokens_from_metrics(args.from_metrics, args.eval)
    )
    p = project(progress, mean_tokens=mean_tokens, rate=args.rate, max_hours=args.max_hours)
    print(json.dumps(p, indent=2) if args.json else render(p, args.currency))


if __name__ == "__main__":
    main()

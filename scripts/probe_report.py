"""Re-report a partial-prompt probe run from its stored samples, with a boilerplate-resistant variant.

The headline completion rate (similarity >= 0.9 on the held-out 40 %) counts a problem as "completed"
even when the held-out text is a four-word formula such as "as a common fraction." that any model
would produce. This script adds the same rates restricted to problems whose held-out part has at
least ``--min-words`` words (default 12), where a verbatim continuation is hard to explain by
boilerplate, and prints both side by side with Wilson CIs. Usage:

    python scripts/probe_report.py outputs/probe/partial_prompt_1.7b [--min-words 12]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rlvr.data.partial_prompt import wilson_ci


def rates(rows: list[dict], thresholds=(0.9, 0.75)) -> dict:
    n = len(rows)
    out: dict = {"n": n}
    for t in thresholds:
        k = sum(1 for r in rows if r["similarity"] >= t)
        lo, hi = wilson_ci(k, n)
        out[str(t)] = {"count": k, "rate": k / n if n else 0.0, "lo": lo, "hi": hi}
    return out


def report(run_dir: Path, min_words: int = 12) -> dict:
    out: dict = {}
    for samples in sorted(Path(run_dir).glob("*/samples.jsonl")):
        rows = [json.loads(line) for line in samples.open(encoding="utf-8") if line.strip()]
        long_rows = [r for r in rows if len(r["holdout"].split()) >= min_words]
        out[samples.parent.name] = {"all": rates(rows), f"holdout>={min_words}w": rates(long_rows)}
    if not out:
        raise FileNotFoundError(f"no */samples.jsonl under {run_dir}")
    return out


def markdown(rep: dict) -> str:
    rows = [
        "| eval | problems | completed@0.9 [Wilson] | completed@0.75 | long-holdout problems | completed@0.9 long [Wilson] | completed@0.75 long |",
        "|---|---|---|---|---|---|---|",
    ]
    for ev, r in rep.items():
        a = r["all"]
        key = next(k for k in r if k != "all")
        lg = r[key]

        def f(x: dict) -> str:
            return f"{100 * x['rate']:.1f} [{100 * x['lo']:.1f}, {100 * x['hi']:.1f}]"

        rows.append(
            f"| {ev} | {a['n']} | {f(a['0.9'])} | {100 * a['0.75']['rate']:.1f} | {lg['n']} | "
            f"{f(lg['0.9'])} | {100 * lg['0.75']['rate']:.1f} |"
        )
    return "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--min-words", type=int, default=12)
    ap.add_argument("--out", type=Path, default=None, help="default <run_dir>/probe_report.json")
    args = ap.parse_args(argv)
    rep = report(args.run_dir, args.min_words)
    out = args.out or args.run_dir / "probe_report.json"
    out.write_text(json.dumps(rep, indent=2))
    print(markdown(rep))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

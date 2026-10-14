"""Print graded samples for inspection by eye.

PLAN.md Phase 0: after the first harness run, read ~50 "wrong" samples and check that the
grader, not the model, is the one that is wrong as rarely as possible. Each sample shows the
gold answer, what was extracted and by which rule, the finish reason, and the tail of the text.

Usage (venv active, repo root):
    python scripts/inspect_samples.py outputs/eval/<run>/gsm8k/samples.jsonl --wrong -n 25
    python scripts/inspect_samples.py <samples.jsonl> --source last_number --tail 400
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("samples", type=Path)
    ap.add_argument("-n", type=int, default=25, help="how many samples to print")
    ap.add_argument("--wrong", action="store_true", help="only samples graded incorrect")
    ap.add_argument("--correct", action="store_true", help="only samples graded correct")
    ap.add_argument("--source", help="only samples whose extraction rule is this (tag/boxed/...)")
    ap.add_argument("--finish", help="only samples with this finish_reason (stop/length)")
    ap.add_argument(
        "--tail", type=int, default=600, help="characters of the completion tail to show"
    )
    ap.add_argument("--full", action="store_true", help="print the whole completion")
    args = ap.parse_args()

    rows = [
        json.loads(line) for line in args.samples.read_text(encoding="utf-8").splitlines() if line
    ]
    sel = rows
    if args.wrong:
        sel = [r for r in sel if not r["correct"]]
    if args.correct:
        sel = [r for r in sel if r["correct"]]
    if args.source:
        sel = [r for r in sel if r["extraction_source"] == args.source]
    if args.finish:
        sel = [r for r in sel if r["finish_reason"] == args.finish]

    print(f"{len(rows)} samples in file, {len(sel)} selected")
    print("extraction sources (selected):", dict(Counter(r["extraction_source"] for r in sel)))
    print("finish reasons    (selected):", dict(Counter(r["finish_reason"] for r in sel)))
    print("=" * 100)
    for r in sel[: args.n]:
        text = r["text"] if args.full else r["text"][-args.tail :]
        if not args.full and len(r["text"]) > args.tail:
            text = "..." + text
        print(
            f"[{r['problem_id']} #{r['sample_idx']}]  gold={r['gold']!r}  extracted={r['extracted']!r}  "
            f"via={r['extraction_source']}  finish={r['finish_reason']}  tokens={r['num_tokens']}  "
            f"correct={r['correct']}"
        )
        if r.get("meta"):
            keep = {
                k: v for k, v in r["meta"].items() if k in ("level", "subject", "numbers", "target")
            }
            if keep:
                print(f"  meta={keep}")
        print(text)
        print("-" * 100)


if __name__ == "__main__":
    main()

"""Paired MATH-500 differences with and without the test problems whose near-duplicates were in a
checkpoint's training prompts.

`rlvr-build-rft` decontaminates the RFT *traces* against every eval set but writes the Phase 4
prompt file (`learnable_prompts.jsonl`) from the solve-rate rows, before that filter. So GRPO was
trained on MATH-train prompts whose near-duplicates are MATH-500 (and MATH-dev) problems. Found on
5 Nov 2026 while checking the model cards. This script measures what that does to the paired
numbers: the same comparison on the full set and on the set without the flagged problems.

    python scripts/contamination_check.py --overlap <rft data dir>/overlap/sft_openr1__math500.json \
        --prompts <rft data dir>/learnable_prompts.jsonl \
        --pair rft=<eval>/rft_1.7b_think_tag/math500/samples.jsonl \
        --pair grpo=<eval>/grpo_1.7b_think_tag/math500/samples.jsonl --out outputs/reports/x
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "compare_checkpoints", Path(__file__).with_name("compare_checkpoints.py")
)
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


def flagged(overlap_path: Path, prompts_path: Path) -> tuple[set[str], set[str]]:
    """(eval ids flagged by the overlap report, train prompt ids of theirs present in the prompt file)."""
    o = json.loads(Path(overlap_path).read_text())
    train = {re.sub(r"-s\d+$", "", t) for t in o["train_ids_to_drop"]}
    prompts = {
        json.loads(line)["id"] for line in Path(prompts_path).read_text().splitlines() if line
    }
    return set(o["contaminated_eval_ids"]), train & prompts


def _pct(e: dict) -> str:
    return f"{100 * e['mean']:+.1f} [{100 * e['lo']:+.1f}, {100 * e['hi']:+.1f}]"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--overlap", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--pair", action="append", required=True, help="label=samples.jsonl, ref first")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    eval_ids, train_hits = flagged(a.overlap, a.prompts)
    runs = [(p.split("=", 1)[0], cc.first_sample(Path(p.split("=", 1)[1]))) for p in a.pair]
    (lr, ref), rows = runs[0], []
    for ln, new in runs[1:]:
        full = cc.paired(ref, new)
        keep = cc.paired(
            {k: v for k, v in ref.items() if k not in eval_ids},
            {k: v for k, v in new.items() if k not in eval_ids},
        )
        only = cc.paired(
            {k: v for k, v in ref.items() if k in eval_ids},
            {k: v for k, v in new.items() if k in eval_ids},
        )
        rows.append({"ref": lr, "new": ln, "full": full, "clean": keep, "flagged": only})
    md = [
        f"Flagged MATH-500 problems: {len(eval_ids)}; their near-duplicate train prompts present in "
        f"`{a.prompts.name}`: {len(train_hits)}.",
        "",
        "| pair | full set (n) | without flagged (n) | flagged only (n) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['new']} - {r['ref']} | {_pct(r['full']['diff_new_minus_ref'])} ({r['full']['num_pairs']}) | "
            f"{_pct(r['clean']['diff_new_minus_ref'])} ({r['clean']['num_pairs']}) | "
            f"{_pct(r['flagged']['diff_new_minus_ref'])} ({r['flagged']['num_pairs']}) |"
        )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    a.out.with_suffix(".json").write_text(
        json.dumps(
            {"eval_ids": sorted(eval_ids), "train_prompt_hits": sorted(train_hits), "rows": rows},
            indent=2,
        )
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()

"""Pick the SFT learning rate from the dev-split evals (EXP-005), by the pre-registered rule.

Rule (experiments.md, EXP-005): the LR with the highest *pooled* dev accuracy (gsm8k_dev + math_dev,
500 problems) wins; ties within 2 points go to the default 1e-5. Also prints the per-set numbers,
the tag-format failure rate and the final training loss so the choice can be read, not just obeyed.

Usage:
    python scripts/pick_lr.py 5e-6=outputs/eval/sft_1p7b_lr5e-6_dev 1e-5=outputs/eval/sft_1p7b_lr1e-5_dev \
        2e-5=outputs/eval/sft_1p7b_lr2e-5_dev --train-root outputs/train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_LR = "1e-5"
TIE_POINTS = 2.0


def pooled_accuracy(metrics: dict[str, dict]) -> tuple[float, int]:
    """Pooled accuracy over every eval in the run (correct count / problem count)."""
    correct, total = 0.0, 0
    for m in metrics.values():
        n = int(m.get("num_problems", 0))
        correct += m["pass@1"]["mean"] * n
        total += n
    return (100 * correct / total if total else 0.0), total


def choose(pooled: dict[str, float], default: str = DEFAULT_LR, tie: float = TIE_POINTS) -> str:
    best = max(pooled, key=pooled.get)
    if default in pooled and pooled[best] - pooled[default] < tie:
        return default
    return best


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Choose the SFT LR from dev-split evals.")
    ap.add_argument("runs", nargs="+", help="LR=EVAL_RUN_DIR")
    ap.add_argument("--train-root", type=Path, default=None, help="to read final train loss")
    args = ap.parse_args(argv)
    pooled: dict[str, float] = {}
    rows = []
    for item in args.runs:
        lr, d = item.split("=", 1)
        metrics = json.loads((Path(d) / "metrics.json").read_text())
        acc, n = pooled_accuracy(metrics)
        pooled[lr] = acc
        fmt_fail = {
            ev: round(100 * (1 - m.get("tag_format_rate", 0.0)), 1) for ev, m in metrics.items()
        }
        loss = None
        if args.train_root:
            log = args.train_root / f"sft_1p7b_lr{lr}" / "train_log.jsonl"
            if log.exists():
                losses = [json.loads(line).get("loss") for line in log.open() if line.strip()]
                losses = [x for x in losses if x is not None]
                loss = losses[-1] if losses else None
        per = {ev: round(100 * m["pass@1"]["mean"], 1) for ev, m in metrics.items()}
        rows.append((lr, acc, n, per, fmt_fail, loss))
    print("| LR | pooled dev acc (%) | n | per set | tag-format failure (%) | final train loss |")
    print("|---|---|---|---|---|---|")
    for lr, acc, n, per, ff, loss in rows:
        print(f"| {lr} | {acc:.1f} | {n} | {per} | {ff} | {loss} |")
    winner = choose(pooled)
    print(
        f"\nwinner: {winner}  (rule: highest pooled; ties within {TIE_POINTS} points -> {DEFAULT_LR})"
    )


if __name__ == "__main__":
    main()

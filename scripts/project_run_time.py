"""Project the main SFT run's wall clock from a finished short run on the same node.

Used by the E2E session (docs/rental-setup.md, section 2c) to decide whether a cheaper card keeps
the main run or whether the volume moves to the A100. The first LR-sweep run (10 % of the data)
measures tokens per second on this node; the main run is the whole set for ``--main-epochs``
epochs. Rule: stay on this node if the projected wall clock is at most ``--max-hours``.

    python scripts/project_run_time.py outputs/train/sft_1p7b_lr5e-6/run_info.json --rate 102
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def project(info: dict, *, main_epochs: float, max_hours: float, rate: float) -> dict:
    cfg = info.get("config") or {}
    if cfg.get("limit"):
        raise ValueError("the run used --limit; its token count does not scale to the full set")
    runtime = float(info["train_runtime_s"])
    if runtime <= 0:
        raise ValueError("train_runtime_s must be positive")
    tokens_per_epoch = float(info["tokens_per_epoch"])
    epochs_done = float(info.get("epochs_done") or cfg.get("epochs") or 1.0)
    fraction = float(cfg.get("subset_fraction") or 1.0)
    trained = tokens_per_epoch * epochs_done
    tps = trained / runtime
    main_tokens = tokens_per_epoch / fraction * main_epochs
    hours = main_tokens / tps / 3600
    return {
        "gpu": info.get("gpu"),
        "tokens_trained": trained,
        "train_runtime_s": runtime,
        "tokens_per_s": tps,
        "subset_fraction": fraction,
        "main_epochs": main_epochs,
        "main_tokens": main_tokens,
        "projected_hours": hours,
        "rate_per_hour": rate,
        "projected_cost": hours * rate,
        "max_hours": max_hours,
        "stay": hours <= max_hours,
    }


def render(p: dict, currency: str = "INR") -> str:
    verdict = (
        "STAY on this node for the main run"
        if p["stay"]
        else "MOVE the volume to the A100 for the main run"
    )
    return "\n".join(
        [
            "# Speed check",
            "",
            f"- node: {p['gpu']}",
            f"- measured: {p['tokens_trained'] / 1e6:.2f} M tokens in {p['train_runtime_s'] / 60:.1f} min "
            f"= {p['tokens_per_s']:.0f} tokens/s",
            f"- main run: {p['main_tokens'] / 1e6:.1f} M tokens "
            f"({p['main_epochs']:g} epochs of the full set; the short run used "
            f"{100 * p['subset_fraction']:.0f} %)",
            f"- projected: {p['projected_hours']:.1f} h, about {currency} "
            f"{p['projected_cost']:.0f} at {currency} {p['rate_per_hour']:.0f}/h",
            f"- rule: stay if at most {p['max_hours']:g} h",
            "",
            f"verdict: {verdict}",
        ]
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_info", type=Path, help="run_info.json of the finished short run")
    ap.add_argument("--main-epochs", type=float, default=2.0)
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--rate", type=float, default=102.0, help="price of this node per hour")
    ap.add_argument("--currency", default="INR")
    ap.add_argument("--json", action="store_true", help="print the numbers as JSON instead")
    args = ap.parse_args(argv)
    info = json.loads(args.run_info.read_text())
    p = project(info, main_epochs=args.main_epochs, max_hours=args.max_hours, rate=args.rate)
    print(json.dumps(p, indent=2) if args.json else render(p, args.currency))


if __name__ == "__main__":
    main()

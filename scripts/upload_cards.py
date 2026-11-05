"""Upload the committed model cards (cards/*.md) as the README.md of their Hub repos, and nothing else.

`push_to_hub.py model` uploads a whole checkpoint folder with a freshly rendered card; once the
weights are on the Hub, a card correction only needs this file. Needs HF_TOKEN (write) in the
environment of the shell, typed there, never stored.

    python scripts/upload_cards.py --dry-run     # prints what would be uploaded
    python scripts/upload_cards.py               # uploads
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

CARDS = {
    "sft": "Raviteja546/reasoning-rlvr-sft",
    "rft": "Raviteja546/reasoning-rlvr-rft",
    "grpo": "Raviteja546/reasoning-rlvr-grpo",
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cards", type=Path, default=Path(__file__).resolve().parents[1] / "cards")
    ap.add_argument("--only", choices=sorted(CARDS), nargs="*", default=sorted(CARDS))
    ap.add_argument("--message", default="Model card: per-set rates, significance bold, fixes")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    for name in a.only:
        path = a.cards / f"{name}.md"
        if not path.exists():
            raise SystemExit(f"missing card {path}")
        print(f"{path} -> {CARDS[name]}:README.md ({len(path.read_bytes())} bytes)")
    if a.dry_run:
        return
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN is not set in this shell")
    from huggingface_hub import HfApi

    api = HfApi()
    for name in a.only:
        info = api.upload_file(
            path_or_fileobj=str(a.cards / f"{name}.md"),
            path_in_repo="README.md",
            repo_id=CARDS[name],
            repo_type="model",
            commit_message=a.message,
        )
        print(f"{CARDS[name]}: {info}")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Phase 0 exit criterion: harness runs end to end on Qwen3-0.6B-Base, three times, greedy.
# Usage (inside WSL2, venv active):  bash scripts/phase0_smoke.sh [--allow-dirty]
set -euo pipefail
cd "$(dirname "$0")/.."
for i in 1 2 3; do
  rlvr-eval --config configs/eval/phase0_smoke.yaml \
    --out "outputs/eval/phase0_smoke_qwen3-0.6b-base/run$i" "$@"
done
python - <<'PY'
import json, pathlib
root = pathlib.Path("outputs/eval/phase0_smoke_qwen3-0.6b-base")
for ev in ("gsm8k", "math500", "countdown"):
    vals = [json.loads((root / f"run{i}" / ev / "metrics.json").read_text())["pass@1"]["mean"] for i in (1, 2, 3)]
    print(f"{ev:10s} pass@1 over 3 identical runs: {[round(100*v,1) for v in vals]}  spread={100*(max(vals)-min(vals)):.2f} pts")
PY

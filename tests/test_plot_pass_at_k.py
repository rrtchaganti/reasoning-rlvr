import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_pass_at_k.py"
spec = importlib.util.spec_from_file_location("plot_pass_at_k", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["plot_pass_at_k"] = mod
spec.loader.exec_module(mod)


def _fake_metrics(name: str, n: int = 8) -> dict:
    ks = [k for k in (1, 2, 4, 8) if k <= n]
    return {
        "eval": name,
        "num_problems": 10,
        "num_samples_per_problem": n,
        "pass@1": {"mean": 0.3, "lo": 0.1, "hi": 0.5, "n": 10},
        "pass@k": {
            str(k): {"mean": min(1.0, 0.3 + 0.1 * k), "lo": 0.2, "hi": 0.9, "n": 10} for k in ks
        },
        "maj@k": {str(k): min(1.0, 0.3 + 0.05 * k) for k in ks},
    }


def test_plot_and_table(tmp_path):
    run = tmp_path / "run"
    for name in ("gsm8k", "math500"):
        (run / name).mkdir(parents=True)
        (run / name / "metrics.json").write_text(json.dumps(_fake_metrics(name)))
    greedy = tmp_path / "greedy"
    (greedy / "gsm8k").mkdir(parents=True)
    (greedy / "gsm8k" / "metrics.json").write_text(
        json.dumps({"eval": "gsm8k", "pass@1": {"mean": 0.25}})
    )

    runs = mod.load_runs([run])
    assert set(runs) == {"gsm8k", "math500"}
    out = mod.plot(runs, tmp_path / "fig" / "p.png", greedy=mod.greedy_pass1(greedy), title="t")
    assert out.exists() and out.stat().st_size > 1000
    table = mod.markdown_table(runs, ks=(1, 4, 8, 64))
    assert "| gsm8k | 8 |" in table and table.count("-") >= 2  # k = 64 is absent -> "-"
    with pytest.raises(FileNotFoundError):
        mod.load_runs([tmp_path / "nothing"])

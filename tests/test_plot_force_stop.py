"""EXP-016's budget-forcing figure script on synthetic metrics files."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_force_stop.py"
spec = importlib.util.spec_from_file_location("plot_force_stop", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["plot_force_stop"] = mod
spec.loader.exec_module(mod)


def _metrics(d: Path, ev: str, acc: float, tokens: float, forced: float | None = None) -> None:
    (d / ev).mkdir(parents=True, exist_ok=True)
    m = {
        "pass@1": {"mean": acc, "lo": acc - 0.02, "hi": acc + 0.02, "n": 100},
        "tokens": {"mean": tokens, "p90": tokens * 1.3},
        "truncation_rate": 0.01,
    }
    if forced is not None:
        m.update(forced_stop_rate=forced, forced_stop_accuracy=0.5, forced_stop_finished_rate=0.99)
    (d / ev / "metrics.json").write_text(json.dumps(m))


def test_collect_markdown_and_plot(tmp_path):
    for b, acc, tok in ((768, 0.70, 740), (1536, 0.78, 1100)):
        _metrics(tmp_path / f"sft_force{b}", "gsm8k", acc, tok, forced=0.5)
        _metrics(tmp_path / f"sft_force{b}", "math500", acc - 0.2, tok + 300, forced=0.8)
    _metrics(tmp_path / "sft_greedy", "gsm8k", 0.72, 1800)
    _metrics(tmp_path / "grpo_greedy", "gsm8k", 0.82, 1000)
    res = mod.collect(
        {"sft": str(tmp_path / "sft_force{B}")},
        [768, 1536, 2048],  # 2048 is absent and must be skipped
        {"sft": str(tmp_path / "sft_greedy"), "grpo": str(tmp_path / "grpo_greedy")},
    )
    g = res["evals"]["gsm8k"]
    assert [p["budget"] for p in g["forced"]["sft"]] == [768, 1536]
    assert g["forced"]["sft"][1]["forced_stop_rate"] == 0.5
    assert set(g["greedy"]) == {"sft", "grpo"}
    assert "grpo" not in res["evals"]["math500"]["greedy"]  # no math500 greedy given
    md = mod.markdown(res)
    assert "| sft + force | 1536 | 78.0 [76.0, 80.0] | 1100 | 50.0 | 50.0 | 1.0 |" in md
    assert "| grpo greedy (unforced) | 4096 | 82.0 [80.0, 84.0] | 1000 | - | - | 1.0 |" in md
    out = tmp_path / "fig.png"
    mod.plot(res, out, "t")
    assert out.exists()
    mod.main(
        [
            "--runs",
            f"sft={tmp_path / 'sft_force{B}'}",
            "--budgets",
            "768",
            "1536",
            "--greedy",
            f"sft={tmp_path / 'sft_greedy'}",
            "--out",
            str(tmp_path / "cli.png"),
        ]
    )
    assert (tmp_path / "cli.md").exists() and (tmp_path / "cli.json").exists()

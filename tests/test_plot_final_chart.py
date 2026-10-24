"""The final chart on synthetic runs: subset restriction, the laptop points, the table and the PNG."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_final_chart.py"
spec = importlib.util.spec_from_file_location("plot_final_chart", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["plot_final_chart"] = mod
spec.loader.exec_module(mod)


def _write(d: Path, ev: str, rows: list[dict]) -> None:
    (d / ev).mkdir(parents=True, exist_ok=True)
    with (d / ev / "samples.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _greedy(ids_correct, tokens):
    return [
        {"problem_id": p, "sample_idx": 0, "correct": c, "num_tokens": tokens}
        for p, c in ids_correct
    ]


def test_points_are_restricted_to_the_subset_and_plotted(tmp_path):
    subset = tmp_path / "subset"
    lap = tmp_path / "eval"
    curves = {"evals": {}}
    for ev, _ in mod.EVALS:
        _write(
            subset, ev, [{"problem_id": p, "sample_idx": j} for p in ("a", "b") for j in range(2)]
        )
        for c in mod.CKPTS:
            # a third problem outside the subset must be ignored
            _write(
                lap / f"{c}_laptop_think_tag",
                ev,
                _greedy([("a", True), ("b", False), ("z", True)], 1000),
            )
            for b in mod.BUDGETS[c]:
                _write(
                    lap / f"{c}_force{b}_think_tag",
                    ev,
                    _greedy([("a", True), ("b", True), ("z", False)], b // 2),
                )
        row = {
            "k": 1,
            "tokens_per_problem": 900.0,
            "maj_at_k": {"mean": 0.5, "lo": 0.0, "hi": 1.0},
            "wvote_at_k": {"mean": 0.5, "lo": 0.0, "hi": 1.0},
        }
        curves["evals"][ev] = {
            "num_problems": 2,
            "sampled": {c: [row, dict(row, k=2, tokens_per_problem=1800.0)] for c in mod.CKPTS},
        }
    cj = tmp_path / "curves.json"
    cj.write_text(json.dumps(curves))
    res = mod.build(cj, subset, lap)
    nostop = res["gsm8k"]["laptop"]["sft"][-1]
    assert nostop["label"] == "no stop" and nostop["acc"] == 50.0 and nostop["n"] == 2
    assert (
        res["math500"]["laptop"]["grpo"][0]["acc"] == 100.0
        and res["math500"]["laptop"]["grpo"][0]["tokens"] == 768
    )
    out = tmp_path / "fig" / "final.png"
    mod.main(
        [
            "--curves",
            str(cj),
            "--subset-from",
            str(subset),
            "--laptop-root",
            str(lap),
            "--out",
            str(out),
        ]
    )
    assert out.exists() and out.stat().st_size > 10_000
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "| sft | greedy, no stop (laptop) | 50.0" in md and "verifier-weighted vote @2" in md


def test_a_subset_problem_missing_from_a_run_is_an_error(tmp_path):
    lap = tmp_path / "eval" / "r"
    _write(lap, "gsm8k", _greedy([("a", True)], 10))
    with pytest.raises(ValueError, match="missing"):
        mod.greedy_on_subset(lap, "gsm8k", ["a", "b"])

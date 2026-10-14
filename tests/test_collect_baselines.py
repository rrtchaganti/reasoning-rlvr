import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "collect_baselines.py"
spec = importlib.util.spec_from_file_location("collect_baselines", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["collect_baselines"] = mod
spec.loader.exec_module(mod)


def _metrics(acc: float, *, level: bool = False) -> dict:
    m = {
        "pass@1": {"mean": acc, "lo": acc - 0.05, "hi": acc + 0.05, "n": 100},
        "extraction_failure_rate": 0.1,
        "truncation_rate": 0.02,
        "tokens": {"mean": 150.0, "median": 140.0, "p90": 300.0},
        "num_problems": 100,
        "run_tag": "abc1234",
        "extraction_sources": {"final_phrase": 80, "last_number": 20},
    }
    if level:
        m["by_level"] = {
            str(lv): {"mean": acc - 0.1 * lv, "lo": 0.0, "hi": 1.0, "n": 20} for lv in range(1, 6)
        }
    return m


def test_tables_and_ordering(tmp_path):
    root = tmp_path / "eval"
    for model, template, acc in (
        ("0.6b", "few_shot_cot", 0.6),
        ("1.7b", "few_shot_cot", 0.75),
        ("1.7b", "zero_shot", 0.5),
    ):
        d = root / f"base_{model}_{template}"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(
            json.dumps({"gsm8k": _metrics(acc), "math500": _metrics(acc - 0.3, level=True)})
        )
    (root / "phase0_smoke_qwen3-0.6b-base").mkdir()  # ignored: not a base_ run
    runs = mod.load_runs(root)
    assert set(runs) == {("0.6b", "few_shot_cot"), ("1.7b", "few_shot_cot"), ("1.7b", "zero_shot")}
    keys = mod._sorted_keys(runs)
    assert (
        keys[0][0] == "1.7b" and keys[0][1] == "zero_shot"
    )  # bigger model first, templates in order
    acc = mod.accuracy_table(runs)
    assert "| Qwen3-1.7B-Base | few_shot_cot | 75.0 [70.0, 80.0] | 45.0 [40.0, 50.0] |" in acc
    fail = mod.failure_table(runs)
    assert "10.0 / 2.0 / 150" in fail
    lv = mod.level_table(runs)
    assert lv.startswith("| model | template | L1 |") and "| L5 |" in lv
    src = mod.extraction_sources_table(runs)
    assert "final_phrase 80, last_number 20" in src
    with pytest.raises(FileNotFoundError):
        mod.load_runs(tmp_path / "empty")

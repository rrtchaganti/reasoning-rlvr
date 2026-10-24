"""EXP-024's comparison on synthetic runs: identical runs pass, a changed text or grade is caught."""

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_reproduction.py"
spec = importlib.util.spec_from_file_location("check_reproduction", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["check_reproduction"] = mod
spec.loader.exec_module(mod)


def _write(root: Path, run: str, ev: str, rows: list[dict]) -> None:
    d = root / run / ev
    d.mkdir(parents=True, exist_ok=True)
    with (d / "samples.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _rows(texts, correct):
    return [
        {"problem_id": f"p{i}", "sample_idx": 0, "text": t, "extracted": str(i), "correct": c}
        for i, (t, c) in enumerate(zip(texts, correct, strict=True))
    ]


def test_identical_runs_pass_and_differences_are_named(tmp_path):
    ref, new = tmp_path / "ref", tmp_path / "new"
    _write(ref, "r", "gsm8k", _rows(["a b c", "d e f"], [True, False]))
    _write(new, "r", "gsm8k", _rows(["a b c", "d e f"], [True, False]))
    rows = mod.run(ref, new, ["r"])
    assert mod.identical(rows[0]) and rows[0]["acc_ref"] == rows[0]["acc_new"] == 50.0

    _write(new, "r", "gsm8k", _rows(["a b c", "d X f"], [True, True]))
    r = mod.run(ref, new, ["r"])[0]
    assert not mod.identical(r)
    assert r["text_same"] == 1 and r["grade_same"] == 1
    assert r["text_diff"] == [{"problem_id": "p1", "sample_idx": 0, "first_diff_char": 2}]
    assert "**NO**" in mod.render([r]) and "`p1`" in mod.render([r])


def test_missing_new_run_is_reported_and_strict_exits_1(tmp_path):
    ref, new = tmp_path / "ref", tmp_path / "new"
    _write(ref, "r", "math500", _rows(["x"], [True]))
    out = tmp_path / "out" / "check.md"
    rc = mod.main(
        ["--ref", str(ref), "--new", str(new), "--runs", "r", "--out", str(out), "--strict"]
    )
    assert rc == 1 and "no " in out.read_text(encoding="utf-8")
    assert json.loads(out.with_suffix(".json").read_text())[0]["missing"]

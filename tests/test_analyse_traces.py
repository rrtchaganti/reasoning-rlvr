"""The trace analysis (EXP-023) on synthetic completions."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyse_traces.py"
spec = importlib.util.spec_from_file_location("analyse_traces", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["analyse_traces"] = mod
spec.loader.exec_module(mod)


def test_first_repeat_finds_the_earliest_recurring_window():
    words = [f"w{i}" for i in range(30)] + [
        f"w{i}" for i in range(10, 30)
    ]  # words 10-29 repeat once
    i, d = mod.first_repeat(words, n=5)
    assert i == 10 and d == 20
    assert mod.first_repeat([f"u{i}" for i in range(40)], n=5) == (None, None)
    assert mod.first_repeat(["a"] * 3, n=5) == (None, None)


def test_analyse_completion_fields():
    unit = " ".join(f"x{i}" for i in range(60))
    loop = "Let me check the sum. \\boxed{7} Wait, " + unit + " " + unit + " " + unit
    c = mod.analyse_completion(loop, truncated=True, ngram=50)
    assert c["has_loop"] and c["truncated"] and c["wait"] == 1 and c["verify"] == 1
    assert c["answer_before_loop"] is True and 0 < c["loop_onset_frac"] < 0.2
    assert c["repeat_unit_words"] == 60
    fin = "2 + 2 = 4\n</think>\n<answer> 4 </answer>"
    f = mod.analyse_completion(fin, truncated=False, ngram=50)
    assert not f["has_loop"] and f["loop_onset_frac"] is None and 0 < f["think_share"] < 1
    assert f["answer_before_loop"] is None


def _write(run: Path, ev: str, rows):
    (run / ev).mkdir(parents=True, exist_ok=True)
    with (run / ev / "samples.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_analyse_markdown_plot_and_ks(tmp_path):
    unit = " ".join(f"x{i}" for i in range(60))
    loop_text = "step one \\boxed{3} " + unit + " " + unit + " " + unit
    fin_text = "done\n</think>\n<answer> 3 </answer>"
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write(
        a,
        "gsm8k",
        [
            {"problem_id": "p1", "sample_idx": 0, "text": loop_text, "finish_reason": "length"},
            {"problem_id": "p2", "sample_idx": 0, "text": fin_text, "finish_reason": "stop"},
        ],
    )
    _write(
        b,
        "gsm8k",
        [
            {
                "problem_id": "p1",
                "sample_idx": 0,
                "text": loop_text + " " + unit,
                "finish_reason": "length",
            },
            {"problem_id": "p2", "sample_idx": 0, "text": fin_text, "finish_reason": "stop"},
        ],
    )
    res = mod.analyse({"a": a, "b": b}, ["gsm8k", "math500"], [("a", "b")], 50)
    ra = res["evals"]["gsm8k"]["runs"]["a"]
    assert ra["truncated_rate"] == 0.5 and ra["truncated"]["has_loop_rate"] == 1.0
    assert (
        ra["truncated"]["answer_before_loop_rate"] == 1.0
        and ra["truncated"]["repeat_unit_median"] == 60
    )
    assert "math500" not in res["evals"]
    ks = res["evals"]["gsm8k"]["pairs"]["a:b"]
    assert ks["ks_repeat_unit_truncated"] == 0.0 and ks["n_a"] == 1
    md = mod.markdown(res)
    assert "| a | 2 | 50.0 |" in md and "| a:b |" in md
    out = tmp_path / "t.md"
    mod.main(
        ["--runs", f"a={a}", f"b={b}", "--evals", "gsm8k", "--pairs", "a:b", "--out", str(out)]
    )
    assert out.exists() and out.with_suffix(".json").exists() and out.with_suffix(".png").exists()
    j = json.loads(out.with_suffix(".json").read_text())
    assert "_units_truncated" not in j["evals"]["gsm8k"]["runs"]["a"]

"""Best-of-k and the reward-weighted vote in the accuracy-against-tokens script (EXP-022)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_accuracy_vs_tokens.py"
spec = importlib.util.spec_from_file_location("plot_accuracy_vs_tokens_scored", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["plot_accuracy_vs_tokens_scored"] = mod
spec.loader.exec_module(mod)


def test_scored_series_pick_the_highest_scored_sample(tmp_path):
    run = tmp_path / "run"
    (run / "gsm8k").mkdir(parents=True)
    # p1: three wrong "5" and one right "4"; the ORM scores the right one highest -> best-of-4 hits,
    # the plain majority ("5") misses, the weighted vote weighs 4 at sigmoid(3) ~ 0.95 against three
    # at sigmoid(-3) ~ 0.05 each -> hits
    rows = [
        {
            "problem_id": "p1",
            "sample_idx": j,
            "extracted": a,
            "correct": a == "4",
            "num_tokens": 10,
            "finish_reason": "stop",
        }
        for j, a in enumerate(["5", "5", "4", "5"])
    ]
    with (run / "gsm8k" / "samples.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with (run / "gsm8k" / "scores.jsonl").open("w") as f:
        for j, s in enumerate([-3.0, -3.0, 3.0, -3.0]):
            f.write(json.dumps({"problem_id": "p1", "sample_idx": j, "score": s}) + "\n")
    scores = mod.load_scores(run, "gsm8k")
    assert scores == {"p1": {0: -3.0, 1: -3.0, 2: 3.0, 3: -3.0}}
    bon, wv = mod.scored_at_k_per_problem(rows, scores["p1"], 4)
    assert bon == 1.0 and wv == 1.0
    assert (
        mod.maj_at_k_per_problem([r["extracted"] for r in rows], [r["correct"] for r in rows], 4)
        == 0.0
    )
    greedy = tmp_path / "greedy"
    (greedy / "gsm8k").mkdir(parents=True)
    (greedy / "gsm8k" / "samples.jsonl").write_text(
        json.dumps(
            {
                "problem_id": "p1",
                "sample_idx": 0,
                "extracted": "5",
                "correct": False,
                "num_tokens": 8,
                "finish_reason": "stop",
            }
        )
        + "\n"
    )
    res = mod.analyse({"r": run}, {"g": greedy}, ["gsm8k"], (1, 4), "g")
    rec4 = next(r for r in res["evals"]["gsm8k"]["sampled"]["r"] if r["k"] == 4)
    assert rec4["bon_at_k"]["mean"] == 1.0 and rec4["wvote_at_k"]["mean"] == 1.0
    assert rec4["bon_minus_maj"]["mean"] == 1.0 and rec4["bon_minus_ref_greedy"]["mean"] == 1.0
    assert res["evals"]["gsm8k"]["scored_runs"] == ["r"]
    md = mod.markdown(res)
    assert "best-of-k" in md and "| r sampled | 4 |" in md
    out = tmp_path / "f.png"
    mod.plot(res, out)
    assert out.exists()
    # a run without scores keeps the old table shape
    assert mod.load_scores(greedy, "gsm8k") is None

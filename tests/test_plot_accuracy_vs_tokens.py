"""EXP-017's accuracy-against-tokens script on synthetic sample files."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_accuracy_vs_tokens.py"
spec = importlib.util.spec_from_file_location("plot_accuracy_vs_tokens", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["plot_accuracy_vs_tokens"] = mod
spec.loader.exec_module(mod)


def _write(run: Path, ev: str, rows: list[dict]) -> None:
    (run / ev).mkdir(parents=True, exist_ok=True)
    with (run / ev / "samples.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _sampled(pid: str, answers: list[str | None], gold: str, tokens: int) -> list[dict]:
    return [
        {
            "problem_id": pid,
            "sample_idx": j,
            "extracted": a,
            "correct": a == gold,
            "num_tokens": tokens,
            "finish_reason": "stop",
        }
        for j, a in enumerate(answers)
    ]


def test_maj_at_k_per_problem_votes_and_ignores_none():
    # three votes for the right answer, one wrong, one failure: maj@5 is a hit, pass@5 too
    assert (
        mod.maj_at_k_per_problem(["4", "4", "5", None, "4"], [True, True, False, False, True], 5)
        == 1.0
    )
    # all wrong: never a hit
    assert mod.maj_at_k_per_problem(["5", "5"], [False, False], 2) == 0.0
    # subsets of size 1 hit with probability 2/3 on average (repeats make it close to that)
    v = mod.maj_at_k_per_problem(["4", "4", "5"], [True, True, False], 1, repeats=300)
    assert 0.55 < v < 0.8


def test_analyse_restricts_to_shared_problems_and_pairs_against_the_reference(tmp_path):
    weak = tmp_path / "weak"
    strong = tmp_path / "strong"
    greedy_ref = tmp_path / "greedy_ref"
    # the sampled runs cover p1-p3 (weak) and p1-p4 (strong); the greedy reference covers p1-p5
    _write(
        weak,
        "gsm8k",
        _sampled("p1", ["4", "4", "5", "4"], "4", 100)
        + _sampled("p2", ["1", "2", "2", "2"], "2", 100)
        + _sampled("p3", ["9", "9", "9", "9"], "8", 100),
    )
    _write(
        strong,
        "gsm8k",
        _sampled("p1", ["4"] * 4, "4", 50)
        + _sampled("p2", ["2"] * 4, "2", 50)
        + _sampled("p3", ["8", "8", "9", "9"], "8", 50)
        + _sampled("p4", ["7"] * 4, "7", 50),
    )
    _write(
        greedy_ref,
        "gsm8k",
        [
            {
                "problem_id": p,
                "sample_idx": 0,
                "extracted": "x",
                "correct": c,
                "num_tokens": 80,
                "finish_reason": "stop",
            }
            for p, c in (("p1", True), ("p2", True), ("p3", False), ("p4", True), ("p5", True))
        ],
    )
    res = mod.analyse(
        {"weak": weak, "strong": strong},
        {"ref": greedy_ref},
        ["gsm8k", "math500"],
        (1, 2, 4),
        "ref",
    )
    block = res["evals"]["gsm8k"]
    assert block["num_problems"] == 3  # p1-p3 only
    assert "math500" not in res["evals"]
    w4 = next(r for r in block["sampled"]["weak"] if r["k"] == 4)
    assert w4["tokens_per_problem"] == 400 and w4["maj_at_k"]["mean"] == pytest.approx(2 / 3)
    assert w4["pass_at_k"]["mean"] == pytest.approx(2 / 3)
    # the reference greedy is right on p1, p2 of the shared three: weak's maj@4 ties it
    assert w4["maj_minus_ref_greedy"]["mean"] == pytest.approx(0.0)
    s4 = next(r for r in block["sampled"]["strong"] if r["k"] == 4)
    assert s4["maj_at_k"]["mean"] == pytest.approx(1.0)  # p3's tie goes to the first-seen "8"
    assert s4["maj_minus_ref_greedy"]["mean"] == pytest.approx(1 / 3)
    g = block["greedy"]["ref"]
    assert g["pass_at_1"]["mean"] == pytest.approx(2 / 3) and g["tokens_per_problem"] == 80
    md = mod.markdown(res)
    assert "| ref greedy | 1 | 80 |" in md and "| strong sampled | 4 | 200 |" in md
    out = tmp_path / "fig.png"
    mod.plot(res, out, "t")
    assert out.exists()
    mod.main(
        [
            "--samples",
            f"weak={weak}",
            f"strong={strong}",
            "--greedy",
            f"ref={greedy_ref}",
            "--reference-greedy",
            "ref",
            "--evals",
            "gsm8k",
            "--ks",
            "1",
            "2",
            "4",
            "--out",
            str(tmp_path / "cli.png"),
        ]
    )
    assert (tmp_path / "cli.md").exists() and (tmp_path / "cli.json").exists()

"""The evidence index on synthetic sample files: every rule re-derives its number and lists the problems."""

import importlib.util
import json
import sys
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evidence.py"
spec = importlib.util.spec_from_file_location("evidence", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["evidence"] = mod
spec.loader.exec_module(mod)


def _write(run: Path, ev: str, rows: list[dict]) -> None:
    (run / ev).mkdir(parents=True, exist_ok=True)
    with (run / ev / "samples.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _rec(
    pid,
    correct,
    finish="stop",
    extracted="4",
    gold="4",
    tokens=100,
    idx=0,
    forced=False,
    src="tag",
    meta=None,
):
    return {
        "problem_id": pid,
        "sample_idx": idx,
        "text": f"think about {pid} ... answer {extracted}",
        "num_tokens": tokens,
        "finish_reason": finish,
        "extracted": extracted,
        "extraction_source": src,
        "correct": correct,
        "format_ok": True,
        "gold": gold,
        "meta": meta or {},
        "forced_stop": forced,
    }


def test_rules_rederive_numbers_and_list_problems(tmp_path):
    root = tmp_path / "root"
    # reference: p1 right finished, p2 wrong truncated, p3 right finished, p4 wrong truncated
    _write(
        root / "ref",
        "gsm8k",
        [
            _rec("p1", True),
            _rec("p2", False, "length", "9"),
            _rec("p3", True),
            _rec("p4", False, "length", None, src="none"),
        ],
    )
    # new: p1 right, p2 right (was truncated), p3 wrong (both finished, flipped back), p4 right forced
    _write(
        root / "new",
        "gsm8k",
        [
            _rec("p1", True),
            _rec("p2", True),
            _rec("p3", False, extracted="5"),
            _rec("p4", True, forced=True),
        ],
    )
    roots = {"R": root}
    c = {
        "id": "T-1",
        "title": "t",
        "statement": "s",
        "rule": "paired_flips",
        "ref": {"root": "R", "run": "ref", "label": "a"},
        "new": {"root": "R", "run": "new", "label": "b"},
        "evals": {"gsm8k": 25.0},
    }
    res = mod.run_claim(c, roots, {})[0]
    assert (
        res["derived"] == 25.0
        and res["agree"] == "yes"
        and res["res"]["only_new"] == 2
        and res["res"]["only_ref"] == 1
    )
    c2 = dict(c, rule="decomposition", evals={"gsm8k": {"value": 50.0, "both_finished": -50.0}})
    res = mod.run_claim(c2, roots, {})[0]
    assert (
        res["derived"] == 50.0
        and res["res"]["derived_both_finished"] == -50.0
        and res["agree"] == "yes"
    )
    c3 = {
        "id": "T-3",
        "title": "t",
        "statement": "s",
        "rule": "truncation_rate",
        "run": {"root": "R", "run": "ref"},
        "evals": {"gsm8k": 50.0},
    }
    assert mod.run_claim(c3, roots, {})[0]["derived"] == 50.0
    c4 = dict(c3, rule="forced", run={"root": "R", "run": "new"}, evals={"gsm8k": 100.0})
    r4 = mod.run_claim(c4, roots, {})[0]
    assert r4["derived"] == 100.0 and r4["res"]["derived_forced_rate"] == 25.0
    # a wrong quoted number is reported, not hidden
    c5 = dict(c, evals={"gsm8k": 30.0})
    assert mod.run_claim(c5, roots, {})[0]["agree"] == "**NO**"


def test_maj_k_pass_k_and_perturbation(tmp_path):
    root = tmp_path / "root"
    # sampled run: p1 three right of four, p2 none; greedy ref: p1 wrong, p2 right
    rows = [_rec("p1", a == "4", extracted=a, idx=j) for j, a in enumerate(["4", "4", "5", "4"])]
    rows += [_rec("p2", False, extracted="7", gold="2", idx=j) for j in range(4)]
    _write(root / "samp", "math500", rows)
    _write(
        root / "greedy",
        "math500",
        [_rec("p1", False, extracted="5"), _rec("p2", True, extracted="2", gold="2")],
    )
    roots = {"R": root}
    c = {
        "id": "T-m",
        "title": "t",
        "statement": "s",
        "rule": "maj_k",
        "run": {"root": "R", "run": "samp"},
        "ref": {"root": "R", "run": "greedy", "label": "g"},
        "k": 4,
        "evals": {"math500": 0.0},
    }
    r = mod.run_claim(c, roots, {})[0]
    assert r["res"]["derived_maj"] == 50.0 and r["derived"] == 0.0 and r["agree"] == "yes"
    _write(
        root / "samp2",
        "math500",
        [_rec("p1", False, extracted="9", idx=j) for j in range(4)]
        + [_rec("p2", j == 0, extracted="2" if j == 0 else "1", gold="2", idx=j) for j in range(4)],
    )
    c2 = {
        "id": "T-p",
        "title": "t",
        "statement": "s",
        "rule": "pass_k_newly_solved",
        "ref": {"root": "R", "run": "samp", "label": "a"},
        "new": {"root": "R", "run": "samp2", "label": "b"},
        "k": 4,
        "evals": {"math500": None},
    }
    r = mod.run_claim(c2, roots, {})[0]
    assert r["res"]["only_new"] == 1 and r["res"]["only_ref"] == 1
    # perturbation: originals o1 right, o2 right; rewrites of o1 wrong, o2 right -> gap +50
    _write(root / "pert", "gsm8k", [_rec("o1", True), _rec("o2", True)])
    _write(
        root / "pert",
        "gsm8k_perturbed",
        [
            _rec(
                "o1-p",
                False,
                extracted="3",
                meta={"original_id": "o1", "substitutions": {"2": "9"}},
            ),
            _rec("o2-p", True, meta={"original_id": "o2"}),
        ],
    )
    c3 = {
        "id": "T-g",
        "title": "t",
        "statement": "s",
        "rule": "perturbation_gap",
        "run": {"root": "R", "run": "pert"},
        "original_eval": "gsm8k",
        "evals": {"gsm8k_perturbed": 50.0},
    }
    assert mod.run_claim(c3, roots, {})[0]["agree"] == "yes"


def test_main_writes_index_and_claim_files(tmp_path):
    root = tmp_path / "root"
    _write(root / "ref", "gsm8k", [_rec("p1", True), _rec("p2", False, "length", "9")])
    _write(root / "new", "gsm8k", [_rec("p1", True), _rec("p2", True)])
    (root / "train").mkdir()
    with (root / "train" / "reward_log.jsonl").open("w") as f:
        for s in range(5):
            f.write(
                json.dumps(
                    {
                        "step": s,
                        "reward": 0.1 * s,
                        "finished_rate": 0.5,
                        "mean_tokens": 100,
                        "zero_variance_group_rate": 0.3,
                        "format_only_rate": 0.0,
                    }
                )
                + "\n"
            )
    spec_yaml = {
        "roots": {"R": {"desc": "test root", "hub": "nowhere"}},
        "claims": [
            {
                "id": "A-1",
                "title": "flip",
                "experiment": "EXP-T",
                "where": "here",
                "statement": "s",
                "rule": "paired_flips",
                "ref": {"root": "R", "run": "ref"},
                "new": {"root": "R", "run": "new"},
                "evals": {"gsm8k": 50.0},
            },
            {
                "id": "A-2",
                "title": "log",
                "statement": "s",
                "rule": "reward_log",
                "run": {"root": "R", "run": "train"},
                "expect_key": "reward",
                "expect": 30.0,
                "tolerance": 1.0,
                "window": 2,
            },
            {
                "id": "A-3",
                "title": "ptr",
                "statement": "s",
                "rule": "file_pointer",
                "note": "n",
                "files": [{"root": "R", "path": "train", "what": "w"}],
            },
            {
                "id": "A-4",
                "title": "missing",
                "statement": "s",
                "rule": "paired_flips",
                "ref": {"root": "R", "run": "nope"},
                "new": {"root": "R", "run": "new"},
                "evals": {"gsm8k": 1.0},
            },
        ],
    }
    (tmp_path / "claims.yaml").write_text(yaml.safe_dump(spec_yaml))
    out = tmp_path / "ev"
    mod.main(["--claims", str(tmp_path / "claims.yaml"), "--root", f"R={root}", "--out", str(out)])
    idx = (out / "README.md").read_text(encoding="utf-8")
    assert "| A-1 | EXP-T | gsm8k | flip | 50.0 | 50.0 | yes |" in idx
    assert "| A-2 |" in idx and "| A-3 |" in idx and "file missing" in idx
    assert (out / "A-1.md").exists() and "`p2`" in (out / "A-1.md").read_text(encoding="utf-8")


def test_best_of_k_picks_the_highest_scored_draw(tmp_path):
    root = tmp_path / "root"
    rows = [_rec("p1", a == "4", extracted=a, idx=j) for j, a in enumerate(["5", "4", "5", "5"])]
    _write(root / "samp", "math500", rows)
    with (root / "samp" / "math500" / "scores.jsonl").open("w") as f:
        for j, sc in enumerate([-2.0, 3.0, -1.0, -2.5]):
            f.write(json.dumps({"problem_id": "p1", "sample_idx": j, "score": sc}) + "\n")
    _write(root / "greedy", "math500", [_rec("p1", False, extracted="5")])
    c = {
        "id": "T-b",
        "title": "t",
        "statement": "s",
        "rule": "best_of_k",
        "run": {"root": "R", "run": "samp"},
        "ref": {"root": "R", "run": "greedy", "label": "g"},
        "k": 4,
        "evals": {"math500": 100.0},
    }
    r = mod.run_claim(c, {"R": root}, {})[0]
    assert r["derived"] == 100.0 and r["agree"] == "yes"
    assert "| `p1` |" in "\n".join(r["res"]["lines"])

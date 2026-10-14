import json

import pytest

from rlvr.eval.datasets import (
    CountdownSpec,
    EvalSet,
    Problem,
    generate_countdown,
    gsm8k_gold,
    load_eval_set,
    verify_countdown,
)
from rlvr.eval.prompts import TEMPLATES, build_prompt


def test_gsm8k_gold_parsing():
    assert gsm8k_gold("She has 3 + 4 = <<3+4=7>>7 left.\n#### 7") == "7"
    assert gsm8k_gold("...\n#### 1,200") == "1200"
    assert gsm8k_gold("#### $45") == "45"


def test_countdown_generation_is_seeded_and_solvable():
    a = generate_countdown(20, seed=3)
    b = generate_countdown(20, seed=3)
    assert [p.answer for p in a.problems] == [p.answer for p in b.problems]
    assert a.content_hash() == b.content_hash()
    for p in a.problems:
        meta = p.meta
        assert verify_countdown(meta["reference_solution"], meta["numbers"], meta["target"])
        assert CountdownSpec().target_min <= meta["target"] <= CountdownSpec().target_max
        assert len(meta["numbers"]) == 4
    c = generate_countdown(5, seed=4)
    assert c.content_hash() != a.content_hash()


def test_verify_countdown_rules():
    nums, target = [1, 2, 3], 9
    assert verify_countdown("(1 + 2) * 3", nums, target)
    assert verify_countdown("3 * (2 + 1) = 9", nums, target)
    assert not verify_countdown("3 * 3", nums, target)  # number reused, one missing
    assert not verify_countdown("1 + 2 + 3", nums, target)  # equals 6
    assert not verify_countdown("(1 + 2) * 3 + 0", nums, target)
    assert not verify_countdown("3 ** 2", nums, target)
    assert not verify_countdown("abs(-9)", nums, target)
    assert not verify_countdown("", nums, target)
    assert not verify_countdown("1 / 0 + 2 + 3", nums, target)
    assert verify_countdown("6 / 3 + 2", [6, 3, 2], 4)  # exact division allowed
    assert not verify_countdown(
        "7 / 2 + 2", [7, 2, 2], 5
    )  # non-integer intermediate is fine only if exact


def test_subset_is_deterministic_and_full_when_limit_large():
    es = generate_countdown(10, seed=0)
    s1, s2 = es.subset(4, seed=1), es.subset(4, seed=1)
    assert [p.id for p in s1.problems] == [p.id for p in s2.problems]
    assert es.subset(100) is es
    assert load_eval_set("countdown", limit=3, countdown_n=10).problems.__len__() == 3


def test_unknown_eval_set():
    with pytest.raises(KeyError):
        load_eval_set("nope")


def _problem(fam: str) -> tuple[Problem, str]:
    if fam == "countdown":
        return Problem(
            "c",
            json.dumps({"numbers": [2, 3, 5, 7], "target": 29}),
            "29",
            {"numbers": [2, 3, 5, 7], "target": 29},
        ), "countdown"
    if fam == "gsm8k":
        return Problem("g", "Janet has 3 ducks. How many legs?", "6"), "gsm8k"
    return Problem("m", "Compute $1+1$.", "2", {"level": 1}), "math500"


@pytest.mark.parametrize("template", TEMPLATES)
@pytest.mark.parametrize("fam", ["gsm8k", "math500"])
def test_prompts_end_where_the_model_continues(template, fam):
    problem, name = _problem(fam)
    pr = build_prompt(problem, template, name)
    assert problem.question in pr.text
    assert pr.text.rstrip().endswith(("Answer:", "Solution:", "<think>"))
    assert len(pr.stop) > 0
    if template.startswith("few_shot"):
        assert pr.text.count("Question:" if fam == "gsm8k" else "Problem:") == 5
    if template == "few_shot_cot":
        assert ("21 - 15 = 6" in pr.text) or ("\\boxed{[2,5)}" in pr.text)
    if template == "few_shot":
        assert "21 - 15" not in pr.text


def test_countdown_only_think_tag():
    problem, name = _problem("countdown")
    pr = build_prompt(problem, "think_tag", name)
    assert "[2, 3, 5, 7]" in pr.text and "29" in pr.text and pr.text.endswith("<think>")
    assert pr.completion_prefix == "<think>" and "</answer>" in pr.stop
    with pytest.raises(ValueError):
        build_prompt(problem, "few_shot_cot", name)


def test_manifest_fields():
    es = EvalSet("x", "numeric", [Problem("1", "q", "a")], "src", "rev")
    m = es.manifest()
    assert m["num_problems"] == 1 and m["revision"] == "rev" and len(m["content_hash"]) == 16

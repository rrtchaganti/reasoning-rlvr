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
    assert pr.text.endswith(("Answer:", "Solution:\n", "<think>\n"))
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
    assert "[2, 3, 5, 7]" in pr.text and "29" in pr.text and pr.text.endswith("<think>\n")
    assert pr.completion_prefix == "<think>\n" and "</answer>" in pr.stop
    with pytest.raises(ValueError):
        build_prompt(problem, "few_shot_cot", name)


def test_math_prompt_ends_on_the_shots_token_boundary():
    """Regression: ":\n" is one Qwen3 token. A prompt ending in a bare "Solution:" made the
    0.6B base emit <|endoftext|> on 53 % of MATH-500 problems (smoke run 1, 14 Oct 2026)."""
    problem, name = _problem("math500")
    for template in ("zero_shot", "few_shot", "few_shot_cot"):
        pr = build_prompt(problem, template, name)
        assert pr.text.endswith("Solution:\n")
    gsm, gname = _problem("gsm8k")
    assert build_prompt(gsm, "few_shot_cot", gname).text.endswith("Answer:")


def test_think_tag_prompt_never_ends_on_the_think_token():
    """Regression: <think> as the last prompt token makes Qwen3-0.6B-Base produce garbage."""
    for fam in ("gsm8k", "math500", "countdown"):
        problem, name = _problem(fam)
        pr = build_prompt(problem, "think_tag", name)
        assert pr.text.endswith("<think>\n") and pr.completion_prefix == "<think>\n"


def test_manifest_fields():
    es = EvalSet("x", "numeric", [Problem("1", "q", "a")], "src", "rev")
    m = es.manifest()
    assert m["num_problems"] == 1 and m["revision"] == "rev" and len(m["content_hash"]) == 16


def _train_set(n=300, name="math_train"):
    from rlvr.eval.datasets import EvalSet, Problem

    return EvalSet(
        name,
        "latex",
        [Problem(f"{name}-{i}", f"q{i}", str(i), {"level": 1 + i % 5}) for i in range(n)],
        "fake",
    )


def test_filter_problems_levels_exclusions_and_pool_cap_record_themselves():
    from rlvr.eval.datasets import DEV_SIZE_PER_SOURCE, dev_problem_ids, filter_problems

    es = _train_set()
    dev = dev_problem_ids(es)
    assert len(dev) == DEV_SIZE_PER_SOURCE and dev == dev_problem_ids(es)  # deterministic
    pool = filter_problems(es, levels=[1, 2, 3], exclude_ids=dev)
    assert all(p.meta["level"] in (1, 2, 3) for p in pool.problems)
    assert not any(p.id in dev for p in pool.problems)
    assert len(pool) == sum(1 for p in es.problems if p.meta["level"] <= 3 and p.id not in dev)
    assert pool.source == "fake#levels=1,2,3#excluded=250"
    capped = filter_problems(pool, pool_limit=10, seed=1)
    assert len(capped) == 10 and capped.source.endswith("#pool=10,seed=1")
    assert [p.id for p in capped.problems] == [
        p.id for p in filter_problems(pool, pool_limit=10, seed=1).problems
    ]
    assert filter_problems(pool, pool_limit=10_000) is not None and len(
        filter_problems(pool, pool_limit=10_000)
    ) == len(pool)
    with pytest.raises(ValueError):
        filter_problems(
            _train_set(name="gsm8k_train").__class__("x", "numeric", [Problem("a", "q", "1")], "s"),
            levels=[1],
        )


def test_run_eval_parses_pool_flags():
    from rlvr.eval.run_eval import parse_args, parse_levels, parse_pool_limits

    assert parse_levels("1,2,3") == [1, 2, 3] and parse_levels([1, 2]) == [1, 2]
    assert parse_levels(None) is None and parse_levels("") is None
    assert parse_pool_limits(["gsm8k_train=4000", "math_train=10"]) == {
        "gsm8k_train": 4000,
        "math_train": 10,
    }
    assert parse_pool_limits({"gsm8k_train": "4000"}) == {"gsm8k_train": 4000}
    with pytest.raises(ValueError):
        parse_pool_limits(["nope"])
    a = parse_args(
        [
            "--model",
            "m",
            "--out",
            "o",
            "--levels",
            "1,2",
            "--exclude-dev",
            "--pool-limit",
            "gsm8k_train=5",
        ]
    )
    assert a.levels == "1,2" and a.exclude_dev and a.pool_limit == ["gsm8k_train=5"]


def test_derived_set_subset_follows_the_parent_draw(monkeypatch):
    """`--limit` on gsm8k_perturbed keeps the problems whose original is in gsm8k's seeded subset,
    so the two sets stay paired (the H100 session drew them independently: 67 pairs of 300)."""
    from rlvr.eval import datasets

    parent = EvalSet(
        "gsm8k", "numeric", [Problem(f"gsm8k-test-{i}", f"q{i}", str(i)) for i in range(40)], "fake"
    )
    derived = EvalSet(
        "gsm8k_perturbed",
        "numeric",
        [
            Problem(
                f"gsm8k-test-{i}-perturbed", f"p{i}", str(i), {"original_id": f"gsm8k-test-{i}"}
            )
            for i in range(0, 40, 2)  # the perturbation kept every second problem
        ],
        "file:fake",
    )
    monkeypatch.setitem(datasets.LOADERS, "gsm8k", lambda revision=None: parent)
    monkeypatch.setitem(datasets.LOADERS, "gsm8k_perturbed", lambda revision=None: derived)
    sub_parent = load_eval_set("gsm8k", limit=10, seed=3)
    sub_derived = load_eval_set("gsm8k_perturbed", limit=10, seed=3)
    parent_ids = {p.id for p in sub_parent.problems}
    assert len(sub_parent) == 10
    assert sub_derived.problems and all(
        p.meta["original_id"] in parent_ids for p in sub_derived.problems
    )
    assert len(sub_derived) == sum(
        1 for p in derived.problems if p.meta["original_id"] in parent_ids
    )
    assert sub_derived.source.endswith("#paired=gsm8k:limit=10,seed=3")
    # a limit at or above the parent's size keeps every pair; no limit leaves the set untouched
    assert len(load_eval_set("gsm8k_perturbed", limit=100, seed=3)) == len(derived)
    assert load_eval_set("gsm8k_perturbed") is derived

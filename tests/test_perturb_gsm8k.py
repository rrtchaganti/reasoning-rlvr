import numpy as np
import pytest

from rlvr.data.perturb_gsm8k import (
    Program,
    eval_arith,
    perturb_many,
    perturb_problem,
    perturbable_values,
    question_numbers,
    swap_names,
)

Q = (
    "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins with 4. "
    "She sells the rest at $2 each. How much does Janet make every day?"
)
SOL = (
    "Janet has 16 - 3 - 4 = <<16-3-4=9>>9 eggs left.\n"
    "She makes 9 * 2 = $<<9*2=18>>18 every day.\n#### 18"
)


def test_eval_arith_exact_and_safe():
    assert eval_arith("16-3-4") == 9
    assert eval_arith("(1/3)*3") == 1
    assert eval_arith("2.5*4") == 10
    with pytest.raises(ValueError):
        eval_arith("2**3")
    with pytest.raises(ValueError):
        eval_arith("abs(-3)")
    with pytest.raises(ZeroDivisionError):
        eval_arith("1/0")


def test_program_self_check_reproduces_gold():
    prog = Program.parse(SOL)
    prog.check_original()
    assert [s.result for s in prog.steps] == [9, 18]
    assert prog.gold == 18
    # references chain: with identity symbols the program reproduces itself
    assert prog.run({}) == [9, 18]


def test_program_rejects_bad_annotations():
    with pytest.raises(ValueError, match="step_mismatch"):
        Program.parse("x = <<16-3=14>>14\n#### 14").check_original()
    with pytest.raises(ValueError, match="last_step_not_gold"):
        Program.parse("x = <<16-3=13>>13 then 13-4 = 9\n#### 9").check_original()
    with pytest.raises(ValueError, match="no_steps"):
        Program.parse("no annotations here\n#### 9")


def test_question_numbers_and_perturbable_set():
    nums = question_numbers("She has 1,200 apples, 2 bags, 7 days and $16.50 left, 3 kids.")
    assert [t[2] for t in nums] == ["1,200", "2", "7", "16.50", "3"]
    vals = perturbable_values(nums)
    # 2 is below the floor, 7 is a unit constant, 16.50 is not an integer
    assert vals == [1200, 3]


def test_perturb_recomputes_and_rewrites_consistently():
    rng = np.random.default_rng(0)
    p = perturb_problem("g1", Q, SOL, rng, swap_names_too=False)
    subs = {int(k): int(v) for k, v in p.meta["substitutions"].items()}
    assert set(subs) == {16, 3, 4}  # 2 (dollars each) is below the floor and stays
    new_eggs = subs[16] - subs[3] - subs[4]
    assert int(p.answer) == new_eggs * 2
    assert p.answer != "18" and p.meta["original_id"] == "g1" and p.meta["original_answer"] == "18"
    for old, new in subs.items():
        assert f" {old} " not in f" {p.question} " or old == new
        assert str(new) in p.question
    assert "$2 each" in p.question  # constant kept


def test_perturb_keeps_intermediates_integral_over_many_seeds():
    q = "A class of 28 students is split into 4 equal groups. Each group gets 5 pencils. How many pencils?"
    sol = "Groups: 28/4 = <<28/4=7>>7 students each. Pencils: 4*5 = <<4*5=20>>20.\n#### 20"
    for seed in range(30):
        p = perturb_problem("x", q, sol, np.random.default_rng(seed), swap_names_too=False)
        assert int(p.answer) >= 0
        s = {int(k): int(v) for k, v in p.meta["substitutions"].items()}
        assert s[28] % s[4] == 0  # the division stayed exact
        assert len(set(s.values())) == len(s)  # no two numbers collided


def test_ambiguous_literal_is_rejected():
    # question has 5; step 1 produces 5; step 2 uses 5: which one is it? refuse
    q = "Tom buys 10 apples and eats 5. He then buys 5 more. How many now?"
    sol = "10 - 5 = <<10-5=5>>5 left. 5 + 5 = <<5+5=10>>10.\n#### 10"
    with pytest.raises(ValueError, match="ambiguous_literal"):
        perturb_problem("x", q, sol, np.random.default_rng(0), swap_names_too=False)


def test_unexplained_constant_is_rejected_but_word_numbers_are_explained():
    # "thrice as much" computed in prose: 120 appears from nowhere -> refuse
    q = "You pay $40 for a fish sub and thrice as much for a combo sub. Total?"
    sol = "The combo is 120. Total 40 + 120 = <<40+120=160>>160\n#### 160"
    with pytest.raises(ValueError, match="unexplained_constant"):
        perturb_problem("x", q, sol, np.random.default_rng(0), swap_names_too=False)
    # the same constant written as a word in the question is explained
    q2 = "You pay $40 for a fish sub and fifty for a combo sub. Total?"
    sol2 = "Total 40 + 50 = <<40+50=90>>90\n#### 90"
    p = perturb_problem("x", q2, sol2, np.random.default_rng(0), swap_names_too=False)
    assert int(p.answer) == int(p.meta["substitutions"]["40"]) + 50
    # and a fully annotated chain handles it: 40 -> new, 120 -> reference
    sol3 = "Combo: 40*3 = <<40*3=120>>120. Total 40 + 120 = <<40+120=160>>160\n#### 160"
    p = perturb_problem("x", q, sol3, np.random.default_rng(0), swap_names_too=False)
    assert int(p.answer) == 4 * int(p.meta["substitutions"]["40"])


def test_percentages_are_not_perturbed():
    from rlvr.data.perturb_gsm8k import word_numbers

    q = "A $80 jacket is 25% off and a twenty-five dollar hat is 10 percent off."
    vals = perturbable_values(question_numbers(q), q)
    assert vals == [80]
    assert word_numbers(q) == {25}


def test_comma_formatting_preserved():
    q = "A car costs $12,000 and loses 3,000 in value. What is it worth?"
    sol = "12000 - 3000 = <<12000-3000=9000>>9000\n#### 9000"
    p = perturb_problem("x", q, sol, np.random.default_rng(1), swap_names_too=False)
    subs = {int(k): int(v) for k, v in p.meta["substitutions"].items()}
    assert f"${subs[12000]:,}" in p.question and f"{subs[3000]:,}" in p.question
    assert int(p.answer) == subs[12000] - subs[3000]


def test_name_swap_is_consistent_and_gendered():
    q = "Janet gave Tom 3 apples. Janet kept 2. Tom's sister took one."
    out, mapping = swap_names(q, np.random.default_rng(0))
    assert set(mapping) == {"Janet", "Tom"}
    assert mapping["Janet"] != "Janet" and mapping["Tom"] != "Tom"
    assert "Janet" not in out and "Tom" not in out
    assert out.count(mapping["Janet"]) == 2 and out.count(mapping["Tom"]) == 2
    from rlvr.data.perturb_gsm8k import FEMALE, MALE

    assert mapping["Janet"] in FEMALE and mapping["Tom"] in MALE


def test_perturb_many_counts_reasons():
    rows = [
        ("a", Q, SOL),
        ("b", "No numbers here at all.", "nothing <<1+1=2>>2\n#### 2"),
        ("c", "Tom has 5 cats.", "no annotations\n#### 5"),
    ]
    kept, reasons = perturb_many(rows, seed=0)
    assert [p.id for p in kept] == ["a-perturbed"]
    assert reasons["kept"] == 1 and reasons["nothing_to_perturb"] == 1 and reasons["no_steps"] == 1


@pytest.mark.slow
def test_real_gsm8k_test_coverage():
    """Downloads GSM8K; at least half of the test set must survive every check."""
    from rlvr.eval.datasets import _hf_load

    ds = _hf_load("openai/gsm8k", "main", "test", None)
    rows = [(f"t{i}", r["question"], r["answer"]) for i, r in enumerate(ds)]
    kept, reasons = perturb_many(rows, seed=0)
    assert len(kept) / len(rows) > 0.5, dict(reasons)

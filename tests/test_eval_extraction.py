import pytest

from rlvr.eval.extraction import (
    extract_answer,
    grade,
    last_boxed,
    normalize_for_vote,
    normalize_numeric,
    tag_format_ok,
)


def test_last_boxed_nested_and_unbalanced():
    assert last_boxed(r"so \boxed{\frac{1}{2}} and then \boxed{\dfrac{3}{4}}") == r"\dfrac{3}{4}"
    assert last_boxed(r"\boxed{a{b}c}") == "a{b}c"
    assert last_boxed(r"\boxed{\frac{1}{2") is None  # truncated mid-answer
    assert last_boxed(r"\boxed 7 is the result") == "7"
    assert last_boxed("no box") is None


def test_extraction_order_tag_then_boxed_then_phrase_then_number():
    t = "<think>5+5</think><answer>10</answer>"
    assert extract_answer(t, "numeric") == extract_answer(t, "numeric")
    assert (
        extract_answer(t, "numeric").answer == "10" and extract_answer(t, "numeric").source == "tag"
    )
    t = r"<answer>The result is \boxed{12}</answer>"
    assert extract_answer(t, "latex").answer == "12"
    t = r"We get $x = \boxed{\frac{1}{3}}$. Done."
    e = extract_answer(t, "latex")
    assert e.answer == r"\frac{1}{3}" and e.source == "boxed"
    e = extract_answer("so 3 + 4 = 7. The answer is 7.", "numeric")
    assert e.answer == "7" and e.source == "final_phrase"
    e = extract_answer("Step 1: 12 apples. Step 2: 12 - 4 = 8 remain", "numeric")
    assert e.answer == "8" and e.source == "last_number"
    e = extract_answer("#### 1,234", "numeric")
    assert e.answer == "1,234" and e.source == "final_phrase"
    # space-grouped thousands, as GSM8K questions sometimes write them (smoke run 1)
    e = extract_answer("The answer is $228 000.", "numeric")
    assert e.answer == "228000" and e.source == "final_phrase"
    e = extract_answer("so 1 000 000 people", "numeric")
    assert e.answer == "1000000" and e.source == "last_number"
    e = extract_answer("we get 3 1000 left", "numeric")
    assert e.answer == "1000"  # "3 1000" is two numbers, not a grouped one
    assert extract_answer("", "numeric").failed
    assert extract_answer("no digits here", "numeric").failed
    assert extract_answer("no box here", "latex").failed


def test_tag_format():
    assert tag_format_ok("<think>a</think><answer>b</answer>")
    assert tag_format_ok("<think>a</think>\n<answer>b</answer>\n")
    assert not tag_format_ok("<answer>b</answer><think>a</think>")
    assert not tag_format_ok("<think>a</think><answer>b</answer><answer>c</answer>")
    assert not tag_format_ok("<think>a<answer>b</answer>")


@pytest.mark.parametrize(
    "s,expected",
    [
        ("18", "18"),
        ("1,234", "1234"),
        ("$5.50", "11/2"),
        ("50%", "50"),
        ("-3", "-3"),
        ("1/2", "1/2"),
        ("0.5", "1/2"),
        (r"\frac{3}{4}", "3/4"),
        (r"\dfrac{-3}{4}", "-3/4"),
        ("x = 7", "7"),
        ("12 dollars", "12"),
        ("12.", "12"),
        ("3 apples", "3"),
    ],
)
def test_normalize_numeric(s, expected):
    assert str(normalize_numeric(s)) == expected


def test_normalize_numeric_rejects_non_numbers():
    assert normalize_numeric("[2,5)") is None
    assert normalize_numeric("1/0") is None
    assert normalize_numeric("x^2") is None


@pytest.mark.parametrize(
    "pred,gold,ok",
    [
        ("18", "18", True),
        ("18.0", "18", True),
        ("1,800", "1800", True),
        ("$18", "18", True),
        ("18 dollars", "18", True),
        ("0.5", "1/2", True),
        ("19", "18", False),
        ("-2/3", "-2/3", True),
        (None, "18", False),
    ],
)
def test_grade_numeric(pred, gold, ok):
    assert grade(pred, gold, "numeric") is ok


def test_grade_latex_equivalence():
    pytest.importorskip("math_verify")
    assert grade(r"\frac{1}{2}", "0.5", "latex")
    assert grade(r"\dfrac{2}{4}", r"\frac{1}{2}", "latex")
    assert grade("[2,5)", "[2,5)", "latex")
    assert grade(r"-\frac{2}{3}", r"-\frac{2}{3}", "latex")
    assert not grade(r"\frac{1}{3}", r"\frac{1}{2}", "latex")
    assert grade("x+1", "1+x", "latex")


def test_grade_countdown():
    meta = {"numbers": [2, 3, 5, 7], "target": 29}
    assert grade("(2 * 7) + (3 * 5)", "29", "countdown", meta=meta)
    assert grade("2*7+3*5 = 29", "29", "countdown", meta=meta)
    assert not grade("2*7+3*5+0", "29", "countdown", meta=meta)  # extra number
    assert not grade("7*5-3-2", "29", "countdown", meta=meta)  # 30
    assert not grade("__import__('os').system('ls')", "29", "countdown", meta=meta)
    assert not grade("2**7", "29", "countdown", meta=meta)


def test_normalize_for_vote():
    assert (
        normalize_for_vote("0.5") == normalize_for_vote("1/2") == normalize_for_vote(r"\frac{1}{2}")
    )
    assert normalize_for_vote(r"\dfrac{a}{b}") == normalize_for_vote(r"\frac{a}{ b }")
    assert normalize_for_vote("[2, 5)") == normalize_for_vote("[2,5)")

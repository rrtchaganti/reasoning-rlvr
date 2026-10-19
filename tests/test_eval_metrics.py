import numpy as np
import pytest

from rlvr.eval.metrics import (
    binomial_se,
    bootstrap_ci,
    length_stats,
    maj_at_k,
    majority_vote,
    paired_bootstrap_diff,
    pass_at_k_curve,
)


def test_bootstrap_ci_covers_and_is_ordered():
    rng = np.random.default_rng(0)
    x = (rng.random(500) < 0.6).astype(float)
    est = bootstrap_ci(x, seed=1)
    assert est.lo <= est.mean <= est.hi
    assert est.lo < 0.6 < est.hi
    # width close to 2 * 1.96 * binomial SE
    assert (est.hi - est.lo) == pytest.approx(2 * 1.96 * binomial_se(0.6, 500), rel=0.25)


def test_bootstrap_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_paired_diff_is_tighter_than_unpaired_and_zero_for_identical():
    rng = np.random.default_rng(0)
    difficulty = rng.random(400)
    a = (rng.random(400) < difficulty).astype(float)
    b = a.copy()
    flip = rng.choice(400, 40, replace=False)
    b[flip] = 1 - b[flip]  # models disagree on 10 % of problems
    same = paired_bootstrap_diff(a, a)
    assert same.mean == 0.0 and same.lo == 0.0 and same.hi == 0.0
    paired = paired_bootstrap_diff(a, b)
    ua, ub = bootstrap_ci(a), bootstrap_ci(b)
    unpaired_width = (ua.hi - ua.lo) + (ub.hi - ub.lo)
    assert (paired.hi - paired.lo) < unpaired_width
    with pytest.raises(ValueError):
        paired_bootstrap_diff(a, b[:10])


def test_majority_vote_ties_and_none():
    assert majority_vote(["3", "4", "3"]) == "3"
    assert majority_vote(["4", "3", "3", "4"]) == "4"  # tie -> first seen
    assert majority_vote([None, None]) is None
    # the vote is over normalised keys and the winning *key* is returned
    assert (
        majority_vote(["1/2", "0.5", "2"], normalize=lambda s: "half" if s in ("1/2", "0.5") else s)
        == "half"
    )


def test_maj_at_k_full_and_subset():
    answers = [["3", "3", "4", "5"], ["7", "8", "8", "8"], [None, None, "1", "2"]]
    correct = [[True, True, False, False], [True, False, False, False], [False, False, True, False]]
    assert maj_at_k(answers, correct, 4) == pytest.approx((1 + 0 + 1) / 3)
    v = maj_at_k(answers, correct, 2, repeats=50, seed=0)
    assert 0.0 <= v <= 1.0
    with pytest.raises(ValueError):
        maj_at_k(answers, correct, 5)


def test_pass_at_k_curve_monotone():
    num_correct = [0, 1, 2, 8, 4]
    curve = pass_at_k_curve(num_correct, n=8, ks=[1, 2, 4, 8, 16])
    assert set(curve) == {1, 2, 4, 8}
    means = [curve[k].mean for k in (1, 2, 4, 8)]
    assert means == sorted(means)
    assert curve[1].mean == pytest.approx(np.mean([c / 8 for c in num_correct]))


def test_length_stats():
    s = length_stats([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert s["mean"] == 5.5 and s["median"] == 5.5 and s["p90"] == pytest.approx(9.1)


def test_paired_pass_at_k_pairs_by_position_and_counts_ever_solved():
    from rlvr.eval.metrics import paired_pass_at_k

    n = 8
    ref = [0, 1, 4, 8, 0, 2]
    new = [0, 2, 8, 8, 3, 0]
    g = paired_pass_at_k(ref, new, n, ks=(1, 4, 8, 16))
    assert g["n"] == 8 and g["num_pairs"] == 6 and set(g["by_k"]) == {1, 4, 8}  # 16 > n dropped
    # pass@1 is the mean sample accuracy; paired diff is the mean of per-problem differences
    assert g["by_k"][1]["ref"]["mean"] == pytest.approx(sum(ref) / n / 6)
    assert g["by_k"][1]["new"]["mean"] == pytest.approx(sum(new) / n / 6)
    assert g["by_k"][1]["diff_new_minus_ref"]["mean"] == pytest.approx(
        (sum(new) - sum(ref)) / n / 6
    )
    # pass@n is "solved at least once"
    assert g["by_k"][8]["ref"]["mean"] == pytest.approx(4 / 6)
    assert g["by_k"][8]["new"]["mean"] == pytest.approx(4 / 6)
    assert g["ever_solved"] == {"both": 3, "only_ref": 1, "only_new": 1, "neither": 1}
    with pytest.raises(ValueError):
        paired_pass_at_k([1, 2], [1], n, ks=(1,))
    with pytest.raises(ValueError):
        paired_pass_at_k([], [], n, ks=(1,))

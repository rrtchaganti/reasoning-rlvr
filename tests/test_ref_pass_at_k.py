from itertools import combinations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rlvr.ref.pass_at_k import pass_at_k, pass_at_k_mean


def brute_force(n: int, c: int, k: int) -> float:
    """Enumerate every k-subset of n samples (first c correct) and count those with a hit."""
    correct = set(range(c))
    subsets = list(combinations(range(n), k))
    hits = sum(1 for s in subsets if correct & set(s))
    return hits / len(subsets)


@pytest.mark.parametrize(
    "n,c,k", [(4, 1, 1), (4, 1, 2), (6, 2, 3), (8, 3, 4), (8, 0, 8), (8, 8, 1)]
)
def test_matches_brute_force(n, c, k):
    assert pass_at_k(n, c, k) == pytest.approx(brute_force(n, c, k), abs=1e-12)


def test_edge_cases():
    assert pass_at_k(10, 0, 5) == 0.0
    assert pass_at_k(10, 10, 1) == 1.0
    assert pass_at_k(10, 1, 10) == 1.0  # k == n and at least one correct
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)  # pass@1 is the plain accuracy


def test_naive_estimate_is_biased_downward():
    # plug-in 1 - (1 - c/n)^k under-estimates for k > 1 (Jensen on the convex (1-p)^k);
    # the unbiased hypergeometric estimate is never below it and equals it at k = 1
    n, c, k = 4, 1, 2
    naive = 1 - (1 - c / n) ** k
    assert pass_at_k(n, c, k) > naive
    assert pass_at_k(n, c, 1) == pytest.approx(1 - (1 - c / n) ** 1)
    # averaged over many (n, c) draws from a Bernoulli(p) policy, the exact estimator is unbiased
    # and the naive one sits below the true 1 - (1-p)^k
    import numpy as np

    rng = np.random.default_rng(0)
    p, n, k = 0.3, 4, 3
    truth = 1 - (1 - p) ** k
    cs = rng.binomial(n, p, size=200_000)
    exact = np.mean([pass_at_k(n, int(c), k) for c in cs])
    naive = np.mean(1 - (1 - cs / n) ** k)
    assert exact == pytest.approx(truth, abs=0.005)
    assert naive < truth - 0.01


@given(st.integers(1, 30), st.integers(0, 30), st.integers(1, 30))
def test_bounds_and_monotone_in_k(n, c, k):
    if c > n or k > n:
        with pytest.raises(ValueError):
            pass_at_k(n, c, k)
        return
    v = pass_at_k(n, c, k)
    assert 0.0 <= v <= 1.0
    if k + 1 <= n:
        assert pass_at_k(n, c, k + 1) >= v - 1e-12


def test_dataset_mean_and_validation():
    assert pass_at_k_mean([4, 4], [0, 4], 1) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        pass_at_k_mean([4], [0, 1], 1)
    with pytest.raises(ValueError):
        pass_at_k_mean([2, 4], [1, 1], 4)  # first problem has fewer than k samples

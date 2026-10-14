"""Unbiased pass@k (Chen et al. 2021, "Evaluating Large Language Models Trained on Code").

Given n samples for a problem of which c are correct, the probability that at least one of
k samples drawn *without replacement* from those n is correct is

    pass@k = 1 - C(n - c, k) / C(n, k).

Why not the naive estimate ``1 - (1 - c/n) ** k``? It plugs the sample success rate p_hat = c/n
into the population formula 1 - (1 - p)^k. Because (1 - p)^k is convex in p, Jensen gives
E[(1 - p_hat)^k] >= (1 - p)^k, so the naive estimate is biased *downward* for k > 1 (and has
high variance for small n). The hypergeometric form above is exactly unbiased for the
population pass@k when the n samples are an i.i.d. draw from the policy, and it is never
smaller than the naive value. Example: n=4, c=1, k=2 gives 0.5 exact vs 0.4375 naive.
We compute the ratio as a product to avoid overflow in the binomial coefficients:
C(n-c,k)/C(n,k) = prod_{i=n-c+1}^{n} (1 - k / i).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of pass@k for one problem from n samples with c correct.

    Args:
        n: number of samples drawn for this problem.
        c: number of those samples that were correct (0 <= c <= n).
        k: the k in pass@k (1 <= k <= n).
    """
    if not 0 <= c <= n:
        raise ValueError(f"need 0 <= c <= n, got c={c}, n={n}")
    if not 1 <= k <= n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    if n - c < k:
        # fewer wrong samples than k: any k-subset must contain a correct one
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1, dtype=np.float64)))


def pass_at_k_mean(num_samples: Iterable[int], num_correct: Iterable[int], k: int) -> float:
    """Dataset-level pass@k: the mean of per-problem unbiased estimates.

    Problems with fewer than k samples are an error rather than silently skipped, because
    skipping them would bias the dataset number toward the problems that got more samples.
    """
    ns = list(num_samples)
    cs = list(num_correct)
    if len(ns) != len(cs):
        raise ValueError("num_samples and num_correct must have the same length")
    if not ns:
        raise ValueError("no problems")
    return float(np.mean([pass_at_k(n, c, k) for n, c in zip(ns, cs, strict=True)]))

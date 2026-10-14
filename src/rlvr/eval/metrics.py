"""Dataset-level metrics with uncertainty: accuracy with bootstrap CIs, paired comparisons,
majority vote, pass@k curves.

Why bootstrap over problems: the unit of replication is the problem, not the token, and the
problems are fixed across checkpoints. Resampling problems with replacement gives a CI for
"how would this number move on a different draw of problems of this kind". For two models on
the *same* problems the right comparison is paired: bootstrap the per-problem difference, so
that shared problem difficulty cancels and the CI reflects only where the two models disagree.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from rlvr.ref.pass_at_k import pass_at_k


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a 95 % (by default) bootstrap interval."""

    mean: float
    lo: float
    hi: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{100 * self.mean:.1f} [{100 * self.lo:.1f}, {100 * self.hi:.1f}] (n={self.n})"


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap CI for the mean of ``values`` (one entry per problem)."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        raise ValueError("empty values")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return Estimate(mean=float(x.mean()), lo=float(lo), hi=float(hi), n=int(x.size))


def paired_bootstrap_diff(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """CI for mean(a - b) when ``a`` and ``b`` are per-problem scores on the same problems.

    The interval excluding zero is the paired evidence that the models differ; its width is
    typically much narrower than the difference of two unpaired CIs.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired comparison needs equal-length, aligned score vectors")
    return bootstrap_ci(a - b, n_boot=n_boot, alpha=alpha, seed=seed)


def binomial_se(p: float, n: int) -> float:
    """Standard error of a proportion; the floor on how precise an accuracy can be."""
    return float(np.sqrt(p * (1 - p) / n))


def majority_vote(
    answers: Sequence[str | None],
    *,
    normalize: Callable[[str], str] = lambda s: s,
) -> str | None:
    """Most common answer after ``normalize``; returns the winning normalised key.

    Ties go to the key that appeared first. ``None`` answers (extraction failures) do not vote,
    and the result is ``None`` when nothing voted.
    """
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    for i, ans in enumerate(answers):
        if ans is None:
            continue
        key = normalize(ans)
        counts[key] += 1
        first_seen.setdefault(key, i)
    if not counts:
        return None
    best = max(counts, key=lambda k: (counts[k], -first_seen[k]))
    return best


def maj_at_k(
    answers_per_problem: Sequence[Sequence[str | None]],
    correct_per_problem: Sequence[Sequence[bool]],
    k: int,
    *,
    normalize: Callable[[str], str] = lambda s: s,
    repeats: int = 20,
    seed: int = 0,
) -> float:
    """Majority-vote accuracy at k samples, averaged over random k-subsets of the n samples.

    A voted answer is counted correct if any sample carrying that normalised answer was
    graded correct (the grader is deterministic in the answer, so all such samples agree).
    Extraction failures (None) do not vote; a problem with no votes counts as wrong.
    """
    rng = np.random.default_rng(seed)
    per_problem: list[float] = []
    for answers, correct in zip(answers_per_problem, correct_per_problem, strict=True):
        n = len(answers)
        if n < k:
            raise ValueError(f"problem has {n} samples, fewer than k={k}")
        if len(correct) != n:
            raise ValueError("answers and correct flags must align")
        correct_keys = {normalize(a) for a, c in zip(answers, correct, strict=True) if c and a}
        hits = 0.0
        n_rep = 1 if k == n else repeats
        for _ in range(n_rep):
            idx = rng.choice(n, size=k, replace=False) if k < n else np.arange(n)
            voted = majority_vote([answers[i] for i in idx], normalize=normalize)
            hits += float(voted is not None and voted in correct_keys)
        per_problem.append(hits / n_rep)
    return float(np.mean(per_problem))


def pass_at_k_curve(num_correct: Sequence[int], n: int, ks: Sequence[int]) -> dict[int, Estimate]:
    """pass@k with a bootstrap CI over problems, for each k in ``ks`` (all k <= n)."""
    out: dict[int, Estimate] = {}
    for k in ks:
        if k > n:
            continue
        per_problem = [pass_at_k(n, c, k) for c in num_correct]
        out[k] = bootstrap_ci(per_problem, seed=k)
    return out


def length_stats(lengths: Sequence[int]) -> dict[str, float]:
    x = np.asarray(lengths, dtype=np.float64)
    if x.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p90": float(np.quantile(x, 0.9)),
    }

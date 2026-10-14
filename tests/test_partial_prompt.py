import pytest

from rlvr.data.partial_prompt import (
    completion_report,
    continuation_similarity,
    split_prefix,
    wilson_ci,
)


def test_split_prefix_at_word_boundary():
    text = "one two three four five six seven eight nine ten"
    pre, hold = split_prefix(text, 0.6)
    assert pre == "one two three four five six" and hold == "seven eight nine ten"
    pre, hold = split_prefix("a b c d", 0.99)
    assert hold == "d"  # at least one word is always held out
    with pytest.raises(ValueError):
        split_prefix("too short", 0.6)


def test_similarity_identity_and_unrelated():
    hold = "She sells the remainder at the farmers' market daily for $2 per fresh duck egg."
    assert continuation_similarity(hold, hold) == 1.0
    assert continuation_similarity(hold + " And then some extra text the model added.", hold) == 1.0
    assert continuation_similarity("The train leaves at noon and arrives by three.", hold) < 0.6
    assert continuation_similarity("", hold) == 0.0


def test_wilson_ci_basic():
    lo, hi = wilson_ci(0, 100)
    assert lo == 0.0 and 0.0 < hi < 0.05
    lo, hi = wilson_ci(50, 100)
    assert 0.40 < lo < 0.5 < hi < 0.60
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_completion_report_separates_verbatim_from_chance():
    # un-templated held-out texts: random word sequences, so chance similarity is low
    import numpy as np

    rng = np.random.default_rng(0)
    vocab = [
        "apple",
        "train",
        "river",
        "stone",
        "yellow",
        "quick",
        "seven",
        "garden",
        "window",
        "paper",
        "cloud",
        "music",
        "bread",
    ]
    holdouts = [" ".join(rng.choice(vocab, size=12)) for _ in range(20)]
    verbatim = holdouts[:5]
    novel = [f"a totally different sentence about trains number {i}" for i in range(5, 20)]
    rep = completion_report(holdouts, verbatim + novel, seed=0)
    r = rep["rates"]["0.9"]
    assert r["count"] == 5 and r["rate"] == pytest.approx(0.25)
    assert r["wilson_lo"] < 0.25 < r["wilson_hi"]
    assert r["chance_count"] < r["count"]
    assert rep["mean_similarity"] > rep["chance_mean_similarity"]
    with pytest.raises(ValueError):
        completion_report(holdouts, verbatim)

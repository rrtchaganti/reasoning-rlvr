import json

import pytest

from rlvr.data.contamination import (
    fuzzy_overlap,
    ngram_overlap,
    ngrams,
    normalize_text,
    overlap_report,
    write_report,
)
from rlvr.utils.run_info import config_hash, git_commit, write_run_info

EVAL = {
    "e1": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    # 14 tokens: every 13-gram window contains the number, so a number swap defeats n-grams
    "e2": "A robe takes 2 bolts of blue fiber and half that much white fiber.",
}
TRAIN = {
    "t_copy": "Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market? Answer: 18",
    "t_paraphrase": "A robe takes 3 bolts of blue fiber and half that much white fiber.",
    "t_clean": "Tom has 5 marbles and buys 7 more at the store before giving 2 away to his sister on the bus home.",
}


def test_normalize_text():
    assert normalize_text("Hello,  World!\nNew") == "hello world new"


def test_ngram_catches_verbatim_copy_only():
    hits = ngram_overlap(EVAL, TRAIN, n=13)
    assert hits == {"e1": ["t_copy"]}


def test_fuzzy_catches_number_swap():
    hits = fuzzy_overlap(EVAL, TRAIN, threshold=0.9)
    assert "e2" in hits and hits["e2"][0][0] == "t_paraphrase"
    assert "e1" in hits  # the verbatim copy also scores high
    assert all(tid != "t_clean" for v in hits.values() for tid, _ in v)


def test_report_and_drop_set(tmp_path):
    rep = overlap_report("toy", EVAL, TRAIN)
    assert rep.train_ids_to_drop == {"t_copy", "t_paraphrase"}
    s = rep.summary()
    assert s["eval_hit_any"] == 2 and s["train_to_drop"] == 2
    write_report(rep, tmp_path / "r.json")
    payload = json.loads((tmp_path / "r.json").read_text())
    assert payload["summary"]["eval"] == "toy"


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_write_run_info(tmp_path):
    info = write_run_info(tmp_path, config={"x": 1}, seed=3, allow_dirty=True)
    on_disk = json.loads((tmp_path / "run_info.json").read_text())
    assert on_disk["seed"] == 3 and on_disk["config_hash"] == info["config_hash"]
    assert "versions" in on_disk and "python" in on_disk["versions"]
    if git_commit() is not None:
        assert on_disk["run_tag"].split("-")[0] == git_commit(short=True)


def test_dirty_tree_refused_when_not_allowed(tmp_path, monkeypatch):
    import rlvr.utils.run_info as ri

    monkeypatch.setattr(ri, "git_is_dirty", lambda: True)
    with pytest.raises(RuntimeError):
        write_run_info(tmp_path, config={}, seed=0, allow_dirty=False)
    info = write_run_info(tmp_path, config={}, seed=0, allow_dirty=True)
    assert info["run_tag"].endswith("-dirty")


def test_ngram_coverage_filter_ignores_template_sentences():
    """A 13-gram shared through a template sentence covers one line of the problem; a copy
    covers all of it (the AIME 'relatively prime positive integers, find m + n' case, 15 Oct
    2026). A document-frequency cutoff cannot separate them because template variants are
    individually rare, so the report uses coverage."""
    tail = "can be written as m over n where m and n are relatively prime positive integers find m plus n"
    evals = {
        "aime_a": "Eight circles of radius thirty four are sequentially tangent and two of the "
        "circles are tangent to the two sides of the triangle respectively so the inradius " + tail,
        "aime_b": "Jen enters a lottery by picking four distinct numbers from the set of the "
        "first ten positive integers and she wins the grand prize with probability " + tail,
    }
    train = {"template": "Some other problem whose answer " + tail, "copy": evals["aime_a"]}
    every = ngram_overlap(evals, train, n=13)
    # by the any-gram rule the copy of aime_a also "hits" aime_b through the shared template
    assert set(every) == {"aime_a", "aime_b"} and set(every["aime_b"]) == {"template", "copy"}
    covered = ngram_overlap(evals, train, n=13, min_coverage=0.5)
    # the 19-token template covers under half of a 45-token problem; the copy covers all of it
    assert covered == {"aime_a": ["copy"]}
    rep = overlap_report("aime", evals, train, run_fuzzy=False, min_coverage=0.5)
    s = rep.summary()
    assert s["eval_hit_ngram"] == 2 and s["eval_hit_ngram_covered"] == 1
    assert rep.contaminated_eval_ids == {"aime_a"}
    assert rep.train_ids_to_drop == {"template", "copy"}  # conservative drop keeps every hit


def test_ngram_coverage_arithmetic():
    from rlvr.data.contamination import ngram_coverage

    toks = ["a", "b", "c", "d", "e", "f", "g", "h"]  # 8 tokens, n = 3: 6 windows
    pos = {}
    for i, g in enumerate(ngrams(toks, 3)):
        pos.setdefault(g, []).append(i)
    assert ngram_coverage({("a", "b", "c")}, pos, 3, 8) == 3 / 8
    assert ngram_coverage({("a", "b", "c"), ("b", "c", "d")}, pos, 3, 8) == 4 / 8
    assert ngram_coverage(set(pos), pos, 3, 8) == 1.0

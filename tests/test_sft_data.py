"""Phase 2 SFT data pipeline: parsing R1 traces, the shared format, the funnel, decontamination
and the dev split. No network: rows are hand-built, the tokenizer is whitespace."""

import json

import pytest

from rlvr.data.sft_data import (
    BuildStats,
    SftExample,
    build_example,
    content_hash,
    data_report_markdown,
    decontaminate,
    parse_generation,
    read_jsonl,
    reject_reason,
    select_examples,
    sft_completion,
    sft_prompt,
    token_stats,
    write_jsonl,
)
from rlvr.eval.datasets import DEV_SEED, DEV_SIZE_PER_SOURCE, EvalSet, Problem, _dev_split
from rlvr.eval.extraction import extract_answer, tag_format_ok
from rlvr.eval.prompts import build_prompt


def words(s: str) -> int:
    return len(s.split())


GEN_OK = (
    "<think>\nOkay, the tunnel. Let T be the time.\n\n12T/36 + 25T/36 = 185 so T = 180.\n"
    "</think>\n\nThus the original plan would have taken \\(\\boxed{180}\\) days."
)
GEN_TWO_BOXES = (
    "<think>\nspeeds.\n</think>\n\nThe ship is \\(\\boxed{10}\\) km/h and the river "
    "\\(\\boxed{4}\\) km/h."
)
GEN_NO_CLOSE = "<think>\nI keep thinking and never stop"
GEN_NO_BOX = "<think>\nsome reasoning\n</think>\n\nThe answer is 7."


def test_parse_generation_splits_reasoning_summary_and_final():
    t = parse_generation(GEN_OK)
    assert t is not None
    assert t.reasoning.startswith("Okay, the tunnel") and "T = 180" in t.reasoning
    assert t.summary.startswith("Thus") and t.final == "180"


@pytest.mark.parametrize(
    "gen, reason",
    [
        (GEN_TWO_BOXES, "multiple_boxed"),
        (GEN_NO_CLOSE, "no_single_think_close"),
        (GEN_NO_BOX, "no_boxed_in_summary"),
        ("<think>a</think>", "empty_summary"),
        ("<think>a</think> b \\boxed{1 ", "unbalanced_boxed"),
        (GEN_OK, None),
    ],
)
def test_reject_reasons(gen, reason):
    assert reject_reason(gen) == reason
    assert (parse_generation(gen) is None) == (reason is not None)


def test_prompt_is_the_eval_think_tag_template():
    q = "What is 2 + 2?"
    ours = sft_prompt(q)
    theirs = build_prompt(Problem("x", q, "4"), "think_tag", "math500")
    assert ours == theirs.text
    assert ours.endswith("Assistant:\n<think>\n")
    # the completion continues *inside* the think block the prompt opened
    assert theirs.completion_prefix == "<think>\n"


def test_completion_round_trips_through_the_grader():
    t = parse_generation(GEN_OK)
    comp = sft_completion(t)
    full = "<think>\n" + comp  # what the eval harness re-attaches before grading
    assert tag_format_ok(full)
    assert comp.count("</think>") == 1 and comp.count("<answer>") == 1
    assert comp.endswith("<answer> 180 </answer>")
    assert extract_answer(full, "latex").answer == "180"
    # the clean summary stays with the reasoning, not in the answer block
    assert "Thus the original plan" in comp.split("</think>")[0]


def _row(uuid, problem, gens, flags, source="olympiads", ptype="Algebra"):
    return {
        "uuid": uuid,
        "problem": problem,
        "generations": gens,
        "correctness_math_verify": flags,
        "source": source,
        "problem_type": ptype,
        "answer": "180",
    }


def test_select_examples_funnel_counts_every_drop():
    long_gen = "<think>\n" + "word " * 300 + "\n</think>\nSo \\boxed{3}."
    rows = [
        _row("a", "P1 about tunnels.", [GEN_NO_BOX, GEN_OK], [False, True]),  # second gen is used
        _row("b", "P2 wrong only.", [GEN_OK], [False]),
        _row("c", "P3 two boxes.", [GEN_TWO_BOXES], [True]),
        _row("d", "P1  about tunnels.", [GEN_OK], [True]),  # duplicate after normalisation
        _row("e", "P4 long.", [long_gen], [True]),
        _row("f", "", [GEN_OK], [True]),
    ]
    stats = BuildStats()
    out = select_examples(rows, count_tokens=words, max_tokens=200, stats=stats)
    assert [ex.id for ex in out] == ["a"]
    assert stats.rows_seen == 6 and stats.kept == 1
    assert stats.drops == {
        "no_correct_generation": 1,
        "multiple_boxed": 1,
        "duplicate_problem": 1,
        "too_long": 1,
        "empty_problem": 1,
    }
    # the long candidate is counted in the pre-cap length distribution; the duplicate is
    # dropped before tokenising (dedup is cheap, the tokenizer is not), so two candidates
    assert len(stats.candidate_tokens) == 2 and max(stats.candidate_tokens) > 200
    assert out[0].answer == "180" and out[0].n_tokens == words(out[0].prompt) + words(
        out[0].completion
    )


def test_decontaminate_drops_hits_and_reports_per_eval(tmp_path):
    copy = (
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes "
        "muffins for her friends every day with four. She sells the remainder at the farmers' "
        "market daily for $2 per fresh duck egg. How much in dollars does she make every day?"
    )
    # 14 words: every 13-gram window contains the swapped number, so only fuzzy can catch it
    swapped = "A robe takes 3 bolts of blue fiber and half that much white fiber."
    clean = "Tom has 5 marbles and buys 7 more at the store before giving 2 away to his sister."
    exs = []
    for uid, prob in [("copy", copy), ("swap", swapped), ("clean", clean)]:
        t = parse_generation(GEN_OK)
        ex = build_example(
            uid=uid, problem_text=prob, trace=t, source="s", problem_type="t", count_tokens=words
        )
        ex.meta = {"problem": prob}
        exs.append(ex)
    evals = {
        "gsm8k": {
            "g1": copy,
            "g2": "A robe takes 2 bolts of blue fiber and half that much white fiber.",
        },
        "math500": {"m1": "Find all real x with x^2 = 4."},
    }
    kept, summary = decontaminate(exs, evals, report_dir=tmp_path)
    assert [ex.id for ex in kept] == ["clean"]
    g = summary["gsm8k"]
    assert g["eval_hit_ngram"] == 1 and g["eval_hit_fuzzy"] == 2 and g["eval_hit_union"] == 2
    assert g["eval_hit_fuzzy_only"] == 1 and g["train_items_dropped"] == 2
    assert summary["math500"]["eval_hit_union"] == 0
    assert summary["_total"]["train_items_dropped_union"] == 2
    rep = json.loads((tmp_path / "sft_openr1__gsm8k.json").read_text())
    assert set(rep["train_ids_to_drop"]) == {"copy", "swap"}


def test_jsonl_round_trip_and_hash_is_content_based(tmp_path):
    t = parse_generation(GEN_OK)
    ex = build_example(
        uid="a", problem_text="P", trace=t, source="s", problem_type="t", count_tokens=words
    )
    ex.meta = {"problem": "P"}
    write_jsonl([ex], tmp_path / "train.jsonl")
    rows = read_jsonl(tmp_path / "train.jsonl")
    assert rows[0]["prompt"] == ex.prompt and rows[0]["completion"] == ex.completion
    assert "meta" not in rows[0]
    h1 = content_hash([ex])
    ex2 = SftExample(**{**rows[0]})
    assert content_hash([ex2]) == h1
    ex2.completion += " "
    assert content_hash([ex2]) != h1


def test_token_stats_and_report_render():
    st = token_stats([10, 20, 30, 40])
    assert st["n"] == 4 and st["mean"] == 25.0 and st["max"] == 40 and st["total"] == 100
    manifest = {
        "source": "s",
        "config": "c",
        "revision": "r",
        "tokenizer": "t",
        "max_tokens": 4096,
        "git_commit": "abc",
        "num_examples": 1,
        "kept_token_stats": st,
        "stats": {"rows_seen": 6, "kept": 2, "drops": {"too_long": 4}},
        "by_source": {"olympiads": 1},
        "by_problem_type": {"Algebra": 1},
    }
    overlap = {
        "gsm8k": {
            "num_eval": 2,
            "eval_hit_ngram": 1,
            "eval_hit_fuzzy": 1,
            "eval_hit_fuzzy_only": 0,
            "eval_hit_union": 1,
            "eval_hit_union_pct": 50.0,
            "train_items_dropped": 1,
        },
        "_total": {"train_items_dropped_union": 1, "kept_after_decontamination": 1},
    }
    md = data_report_markdown(manifest, overlap)
    assert "| dropped: too_long | 4 |" in md and "| gsm8k | 2 | 1 | 1 | 0 | 1 | 50.0 | 1 |" in md


def test_dev_split_is_deterministic_and_disjoint_from_the_rest():
    probs = [Problem(f"p-{i:04d}", f"q{i}", str(i)) for i in range(1000)]
    es = EvalSet("x_train", "numeric", probs, "src")
    a = _dev_split(es, "x_dev")
    b = _dev_split(EvalSet("x_train", "numeric", list(reversed(probs)), "src"), "x_dev")
    assert len(a) == DEV_SIZE_PER_SOURCE
    assert [p.id for p in a.problems] == [p.id for p in b.problems]  # order-independent
    assert a.content_hash() == b.content_hash()
    assert DEV_SEED == 20261015  # changing the seed changes the dev split: a new dataset version

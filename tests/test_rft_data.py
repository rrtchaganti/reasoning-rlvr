"""Phase 3 RFT data pipeline on hand-built samples: solve rates, selection rules, the shared
format, decontamination and the learnable-prompt file. No network, whitespace tokenizer."""

import json
from collections import Counter

import pytest

from rlvr.data import rft_data as rd
from rlvr.eval.datasets import Problem
from rlvr.eval.prompts import build_prompt


def words(texts):
    return [len(t.split()) for t in texts]


def _rec(pid, idx, correct, text=None, finish="stop", tokens=None, fmt=True, extracted="7"):
    text = (
        text
        if text is not None
        else f"step {idx} of {pid}\n</think>\n<answer> {extracted} </answer>"
    )
    return {
        "problem_id": pid,
        "sample_idx": idx,
        "text": text,
        "num_tokens": tokens if tokens is not None else len(text.split()),
        "finish_reason": finish,
        "extracted": extracted,
        "extraction_source": "tag",
        "correct": correct,
        "format_ok": fmt,
    }


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_solve_rates_and_summary_count_zero_full_and_learnable():
    groups = {
        "a": [_rec("a", i, i < 4) for i in range(4)],  # 4/4
        "b": [_rec("b", i, i < 1) for i in range(4)],  # 1/4
        "c": [_rec("c", i, False, finish="length") for i in range(4)],  # 0/4, all truncated
    }
    rows = rd.solve_rates(groups, "math_train", {"a": 1, "b": 3, "c": 3})
    by_id = {r["id"]: r for r in rows}
    assert by_id["a"]["rate"] == 1.0 and by_id["b"]["c"] == 1 and by_id["c"]["truncated"] == 4
    assert list(by_id) == ["a", "b", "c"] and set(rows[0]) == set(rd.SOLVE_RATE_FIELDS)
    s = rd.solve_rate_summary(rows)
    assert s["math_train"]["rate_0"] == 1 and s["math_train"]["rate_1"] == 1
    assert s["math_train"]["learnable"] == 1 and s["math_train/level3"]["problems"] == 2
    assert s["math_train"]["histogram"] == {0: 1, 1: 1, 4: 1}
    assert [r["id"] for r in rd.learnable_prompts(rows)] == ["b"]


@pytest.mark.parametrize(
    "rec, reason",
    [
        (_rec("p", 0, False), "incorrect"),
        (_rec("p", 0, True, finish="length"), "truncated"),
        (_rec("p", 0, True, text="no close tag <answer> 7 </answer>"), "think_tags"),
        (_rec("p", 0, True, fmt=False), "format"),
        (_rec("p", 0, True, text="x\n</think>\n<answer> 7 </answer> and more"), "no_answer_close"),
        (_rec("p", 0, True, extracted=""), "no_answer"),
        (_rec("p", 0, True), None),
    ],
)
def test_reject_reasons(rec, reason):
    assert rd.rft_reject_reason(rec) == reason


def test_select_keeps_shortest_two_distinct_usable_samples_in_the_shared_format():
    problems = {
        "gsm8k-train-1": Problem("gsm8k-train-1", "How many eggs?", "7"),
        "gsm8k-train-2": Problem("gsm8k-train-2", "Unsolved one", "9"),
    }
    long = "a very long reasoning " * 20 + "\n</think>\n<answer> 7 </answer>"
    groups = {
        "gsm8k-train-1": [
            _rec("gsm8k-train-1", 0, True, tokens=30),
            _rec("gsm8k-train-1", 1, True, tokens=10),
            _rec("gsm8k-train-1", 2, True, text=long, tokens=5),  # shortest by count, but it
            _rec(
                "gsm8k-train-1",
                3,
                True,
                tokens=10,
                text="step 1 of gsm8k-train-1\n</think>\n<answer> 7 </answer>",
            ),
            _rec("gsm8k-train-1", 4, False),
            _rec("gsm8k-train-1", 5, True, finish="length"),
        ],
        "gsm8k-train-2": [_rec("gsm8k-train-2", i, False) for i in range(3)],
        "ghost": [_rec("ghost", 0, True)],
    }
    drops = Counter()
    # cap of 40 "tokens": the prompt is ~60 words with the whitespace tokenizer, so use a big cap
    out = rd.select_examples(
        groups,
        problems,
        eval_name="gsm8k_train",
        source="gsm8k_train",
        count_tokens=words,
        per_problem=2,
        max_tokens=10_000,
        drops=drops,
    )
    ids = [ex.id for ex in out]
    # sample 3 duplicates sample 1's reasoning ("step 1 of ...") and is dropped; the two shortest
    # distinct usable ones are idx 2 (5 tokens) and idx 1 (10 tokens)
    assert ids == ["gsm8k-train-1-s2", "gsm8k-train-1-s1"]
    assert drops == Counter(
        {
            "incorrect": 4,
            "truncated": 1,
            "duplicate_reasoning": 1,
            "over_per_problem_cap": 1,
            "unknown_problem_id": 1,
        }
    )
    ex = out[1]
    assert ex.prompt == build_prompt(problems["gsm8k-train-1"], "think_tag", "gsm8k_train").text
    assert ex.prompt.endswith("Assistant:\n<think>\n") and ex.completion.endswith("</answer>")
    assert (
        ex.answer == "7" and ex.meta["problem"] == "How many eggs?" and ex.meta["sample_idx"] == 1
    )
    assert ex.n_tokens == words([ex.prompt])[0] + 10
    # the token cap drops a kept candidate without promoting the next one past the per-problem cap
    drops2 = Counter()
    out2 = rd.select_examples(
        groups,
        problems,
        eval_name="gsm8k_train",
        source="gsm8k_train",
        count_tokens=words,
        per_problem=2,
        max_tokens=words([ex.prompt])[0] + 6,
        drops=drops2,
    )
    assert [ex.id for ex in out2] == ["gsm8k-train-1-s2"] and drops2[
        "over_max_tokens"
    ] == 2  # idx1 and idx0 both over


MATH_Q = [
    "Find the sum of the first four primes.",
    "A regular hexagon has side length 3. What is its area?",
    "How many positive divisors does 360 have?",
    "Solve for x: 2x + 7 = 19.",
]


def test_build_end_to_end_writes_every_artefact(tmp_path):
    samples = tmp_path / "samples"
    problems = {
        "gsm8k_train": {
            f"g{i}": Problem(f"g{i}", f"gsm question number {i} about eggs", str(i))
            for i in range(4)
        },
        "math_train": {
            f"m{i}": Problem(
                f"m{i}",
                MATH_Q[i],
                str(i),
                {"level": 1 + i % 3, "subject": "algebra"},
            )
            for i in range(4)
        },
    }
    g_rows, m_rows = [], []
    for i in range(4):
        g_rows += [
            _rec(f"g{i}", j, j < i, extracted=str(i)) for j in range(4)
        ]  # rates 0, .25, .5, .75
        m_rows += [
            _rec(f"m{i}", j, j < 4 - i, extracted=str(i)) for j in range(4)
        ]  # rates 1, .75, .5, .25
    _write(samples / "gsm8k_train" / "samples.jsonl", g_rows)
    _write(samples / "math_train" / "samples.jsonl", m_rows)
    (samples / "run_info.json").write_text(
        json.dumps(
            {
                "run_tag": "abc1234",
                "git_commit": "abc",
                "config": {"model": "sft", "n": 4, "temperature": 1.0, "max_tokens": 2048},
            }
        )
    )
    # one eval item copies m1's question verbatim: it must be dropped from the kept set
    eval_items = {"math500": {"e1": MATH_Q[1], "e2": "unrelated"}}
    out = tmp_path / "rft"
    man = rd.build(
        samples,
        splits=["gsm8k_train", "math_train"],
        problems_by_split=problems,
        count_tokens=words,
        eval_items=eval_items,
        out_dir=out,
        figure=tmp_path / "fig.png",
        report=tmp_path / "rep.md",
        per_problem=2,
        max_tokens=10_000,
        run_fuzzy=True,
        min_coverage=0.5,
    )
    assert man["num_problems"] == 8 and man["n_per_problem"] == 4 and man["model"] == "sft"
    sr = man["solve_rates"]
    assert sr["gsm8k_train"]["rate_0"] == 1 and sr["math_train"]["rate_1"] == 1
    assert sr["gsm8k_train"]["learnable"] == 3 and sr["math_train"]["learnable"] == 3
    assert man["num_learnable"] == 6 and len(man["learnable_hash"]) == 16
    # kept: g1 (1), g2 (2), g3 (2), m0 (2), m2 (2), m3 (1); m1 (2) dropped by decontamination
    assert man["by_source"] == {"gsm8k_train": 5, "math_train": 5}
    assert man["overlap"]["math500"]["train_items_dropped"] == 2 and man["num_examples"] == 10
    rows = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 10 and not any(r["id"].startswith("m1-") for r in rows)
    assert rows[0]["prompt"].endswith("Assistant:\n<think>\n") and "meta" not in rows[0]
    learn = [
        json.loads(line) for line in (out / "learnable_prompts.jsonl").read_text().splitlines()
    ]
    assert {r["id"] for r in learn} == {"g1", "g2", "g3", "m1", "m2", "m3"}
    assert learn[0]["question"] and learn[0]["answer"]
    rates = [json.loads(line) for line in (out / "solve_rate.jsonl").read_text().splitlines()]
    assert len(rates) == 8 and {r["id"]: r["rate"] for r in rates}["m0"] == 1.0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["content_hash"] == man["content_hash"] and manifest["num_examples"] == 10
    assert (tmp_path / "fig.png").stat().st_size > 1000
    rep = (tmp_path / "rep.md").read_text()
    assert "| gsm8k_train | 4 | 1 | 0 | 3 (75.0 %)" in rep and "Dropped in total: 2." in rep


def test_build_refuses_an_empty_kept_set(tmp_path):
    samples = tmp_path / "s"
    _write(samples / "gsm8k_train" / "samples.jsonl", [_rec("g0", 0, False)])
    with pytest.raises(ValueError, match="no usable examples"):
        rd.build(
            samples,
            splits=["gsm8k_train"],
            problems_by_split={"gsm8k_train": {"g0": Problem("g0", "q", "1")}},
            count_tokens=words,
            eval_items={},
            out_dir=tmp_path / "o",
            figure=None,
            report=None,
        )
    with pytest.raises(FileNotFoundError):
        rd.build(
            samples,
            splits=["math_train"],
            problems_by_split={"math_train": {}},
            count_tokens=words,
            eval_items={},
            out_dir=tmp_path / "o",
            figure=None,
            report=None,
        )

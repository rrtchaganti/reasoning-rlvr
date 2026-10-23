"""The outcome reward model (EXP-022): the pair builder, AUROC, and a two-step train/score smoke on
the tiny model (CPU)."""

import json
from pathlib import Path

import pytest

from rlvr.train import orm


def _rows(pid, flags, tokens=5):
    return [
        {
            "problem_id": pid,
            "sample_idx": j,
            "text": f"t {pid} {j} {'ok' if c else 'bad'}",
            "correct": c,
            "num_tokens": tokens,
            "finish_reason": "stop",
        }
        for j, c in enumerate(flags)
    ]


def test_build_examples_balances_per_problem_and_skips_unknown_problems():
    rows = (
        _rows("p1", [True, True, True, False, False])
        + _rows("p2", [False] * 4)
        + _rows("zz", [True])
    )
    prompts = {"p1": "Q1 <think>\n", "p2": "Q2 <think>\n"}
    ex = orm.build_examples(rows, prompts.get, eval_name="gsm8k_train", per_problem=2, seed=0)
    by = {}
    for e in ex:
        by.setdefault((e["problem_id"], e["label"]), 0)
        by[(e["problem_id"], e["label"])] += 1
    assert by == {("p1", 1): 2, ("p1", 0): 2, ("p2", 0): 2}  # zz has no prompt
    assert all(e["prompt"].endswith("<think>\n") for e in ex)
    tr, he = orm.split_heldout(ex, 0.5, 0)
    assert {e["problem_id"] for e in tr}.isdisjoint({e["problem_id"] for e in he}) and tr and he
    assert orm.ids_hash(ex) == orm.ids_hash(list(ex)) and len(orm.ids_hash(ex)) == 16


def test_auroc():
    assert orm.auroc([0.1, 0.9, 0.8, 0.2], [0, 1, 1, 0]) == 1.0
    assert orm.auroc([0.9, 0.1], [0, 1]) == 0.0
    assert orm.auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    assert orm.auroc([0.3], [1]) != orm.auroc([0.3], [1])  # nan when one class is absent


@pytest.mark.slow
def test_train_and_score_smoke_on_tiny_model(tmp_path, monkeypatch):
    from test_train_sft import _tiny_model_and_tokenizer

    mdl = tmp_path / "tiny"
    _tiny_model_and_tokenizer(mdl, tie=True, filler_tags=True)
    data = tmp_path / "data"
    rows = [
        {
            "problem_id": f"p{i}",
            "eval": "gsm8k_train",
            "sample_idx": 0,
            "prompt": "q r <think>",
            "completion": f"a b {i % 7} </think> <answer> {i % 7} </answer>",
            "label": i % 2,
            "num_tokens": 8,
            "finish_reason": "stop",
        }
        for i in range(12)
    ]
    orm.write_jsonl(rows[:8], data / "train.jsonl")
    orm.write_jsonl(rows[8:], data / "heldout.jsonl")
    out = tmp_path / "orm"
    orm.main(
        [
            "train",
            "--data",
            str(data),
            "--model",
            str(mdl),
            "--out",
            str(out),
            "--cpu",
            "--batch",
            "2",
            "--grad-accum",
            "2",
            "--epochs",
            "1",
            "--max-len",
            "32",
            "--lora-r",
            "2",
            "--lora-alpha",
            "4",
            "--log-every",
            "1",
            "--allow-dirty",
        ]
    )
    assert (
        (out / "adapter").exists()
        and (out / "head.pt").exists()
        and (out / "orm_config.json").exists()
    )
    m = json.loads((out / "metrics.json").read_text())
    assert m["steps"] == 2 and 0.0 <= m["heldout"]["accuracy"] <= 1.0
    # scoring: a fake sample set whose problem ids the loader must know -> monkeypatch the loader
    from rlvr.eval.datasets import EvalSet, Problem

    es = EvalSet(
        name="gsm8k",
        answer_type="numeric",
        source="t",
        problems=[Problem(id="s1", question="q", answer="1", meta={})],
    )
    monkeypatch.setattr("rlvr.eval.datasets.load_eval_set", lambda name, **kw: es)
    samp = tmp_path / "samp"
    (samp / "gsm8k").mkdir(parents=True)
    with (samp / "gsm8k" / "samples.jsonl").open("w") as f:
        for j in range(3):
            f.write(
                json.dumps(
                    {
                        "problem_id": "s1",
                        "sample_idx": j,
                        "text": f"a {j} </think> <answer> {j} </answer>",
                        "correct": j == 1,
                        "num_tokens": 6,
                    }
                )
                + "\n"
            )
    orm.main(
        [
            "score",
            "--orm",
            str(out),
            "--samples",
            str(samp),
            "--evals",
            "gsm8k",
            "--cpu",
            "--batch",
            "2",
        ]
    )
    scored = [json.loads(line) for line in (samp / "gsm8k" / "scores.jsonl").open()]
    assert [s["sample_idx"] for s in scored] == [0, 1, 2] and all(0 < s["prob"] < 1 for s in scored)
    assert Path(out / "run_info.json").exists()
    # the plain state dict written before save_pretrained is enough to load and score
    first = [s["score"] for s in scored]
    import shutil

    assert (out / "adapter_state.pt").exists()
    shutil.rmtree(out / "adapter")
    (samp / "gsm8k" / "scores.jsonl").unlink()
    orm.main(
        [
            "score",
            "--orm",
            str(out),
            "--samples",
            str(samp),
            "--evals",
            "gsm8k",
            "--cpu",
            "--batch",
            "2",
        ]
    )
    again = [json.loads(line)["score"] for line in (samp / "gsm8k" / "scores.jsonl").open()]
    assert again == pytest.approx(first, abs=1e-5)

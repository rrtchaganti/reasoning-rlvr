"""End-to-end harness test on a tiny random model (CPU): the plumbing, not the accuracy."""

import json

import pytest
import torch

from rlvr.eval.datasets import EvalSet, Problem
from rlvr.eval.prompts import build_prompt
from rlvr.eval.report import grade_completions, markdown_row, summarize
from rlvr.eval.sampler import Completion, HFSampler, SamplingConfig

transformers = pytest.importorskip("transformers")


def _fake_completions(texts: list[list[str]]) -> list[list[Completion]]:
    return [[Completion(t, len(t.split()), "stop") for t in group] for group in texts]


def test_grade_and_summarize_with_fake_completions():
    es = EvalSet(
        "toy",
        "numeric",
        [
            Problem("a", "1+1?", "2", {"level": 1}),
            Problem("b", "2+2?", "4", {"level": 2}),
            Problem("c", "3+3?", "6", {"level": 1}),
        ],
        "unit",
    )
    comps = _fake_completions(
        [
            ["<think>x</think><answer>2</answer>", "The answer is 3.", "2", "nothing"],
            ["4", "4", "4", "4"],
            ["7", "5", "6", "The answer is 6."],
        ]
    )
    results = grade_completions(es, comps)
    s = summarize(results, ks=(1, 2, 4))
    assert s["num_problems"] == 3 and s["num_samples_per_problem"] == 4
    assert s["pass@1"]["mean"] == pytest.approx(2 / 3)  # first samples: 2 ok, 4 ok, 7 wrong
    assert s["pass@k"]["4"]["mean"] == pytest.approx(1.0)
    assert s["maj@k"]["4"] == pytest.approx((1 + 1 + 1) / 3)  # 2 wins 2-1, 4 unanimous, 6 wins 2-1
    assert s["extraction_failure_rate"] == pytest.approx(1 / 12)
    assert s["tag_format_rate"] == pytest.approx(1 / 12)
    assert set(s["by_level"]) == {"1", "2"}
    row = markdown_row("toy", s)
    assert row.startswith("| toy | 3 |")


def test_summarize_rejects_ragged_samples():
    es = EvalSet("toy", "numeric", [Problem("a", "q", "1"), Problem("b", "q", "1")], "unit")
    comps = _fake_completions([["1", "1"], ["1"]])
    with pytest.raises(ValueError):
        summarize(grade_completions(es, comps))


@pytest.mark.slow
def test_hf_sampler_runs_on_tiny_model(tmp_path):
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    try:
        tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    except Exception as e:  # offline CI
        pytest.skip(f"tokenizer download unavailable: {e}")
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        eos_token_id=tok.eos_token_id,
        bos_token_id=tok.bos_token_id,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    sampler = HFSampler(model, tok, batch_size=2)
    es = EvalSet(
        "toy",
        "numeric",
        [Problem(str(i), f"What is {i}+{i}?", str(2 * i)) for i in range(3)],
        "unit",
    )
    prompts = [build_prompt(p, "zero_shot", "gsm8k") for p in es.problems]
    out = sampler.generate(
        [p.text for p in prompts],
        SamplingConfig(n=2, temperature=1.0, max_tokens=8, stop=prompts[0].stop),
    )
    assert len(out) == 3 and all(len(o) == 2 for o in out)
    assert all(c.num_tokens <= 8 for o in out for c in o)
    results = grade_completions(es, out)
    s = summarize(results, ks=(1, 2))
    json.dumps(s)  # serialisable

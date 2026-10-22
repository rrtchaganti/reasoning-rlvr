"""Budget forcing (EXP-016): the second pass, its bookkeeping, and the harness plumbing."""

import json

from rlvr.eval.datasets import EvalSet, Problem
from rlvr.eval.force_stop import force_stop, needs_forcing
from rlvr.eval.prompts import build_prompt
from rlvr.eval.report import grade_completions, summarize
from rlvr.eval.run_eval import parse_args
from rlvr.eval.sampler import Completion, Sampler, SamplingConfig


class ScriptedSampler(Sampler):
    """Returns the scripted completion for each prompt; records what it was asked."""

    name = "scripted"

    def __init__(self, script):
        self.script = script  # prompt text -> Completion
        self.calls = []

    def generate(self, prompts, cfg):
        self.calls.append((list(prompts), cfg))
        return [[self.script[p]] for p in prompts]


def test_needs_forcing_only_for_budget_hits_without_a_closed_think_block():
    assert needs_forcing(Completion("still thinking", 10, "length"))
    assert not needs_forcing(Completion("done\n</think>\n<answer> 4 </answer>", 10, "stop"))
    # hit the cap after the answer was written: nothing to force
    assert not needs_forcing(Completion("x\n</think>\n<answer> 4 </answer> trailing", 10, "length"))


def test_force_stop_appends_the_suffix_and_merges_the_continuation():
    prompts = ["P1", "P2", "P3"]
    first = [
        [Completion("loop loop", 8, "length")],
        [Completion("2+2=4\n</think>\n<answer> 4 </answer>", 9, "stop")],
        [Completion("more loop", 8, "length")],
    ]
    suffix = "\n</think>\n<answer>"
    sampler = ScriptedSampler(
        {
            "P1loop loop" + suffix: Completion(" 7 </answer>", 3, "stop"),
            "P3more loop" + suffix: Completion(" 1 2 3 4 5", 5, "length"),
        }
    )
    cfg = SamplingConfig(n=1, temperature=0.0, max_tokens=8, stop=("</answer>",))
    out, stats = force_stop(sampler, prompts, first, cfg, suffix=suffix, continue_tokens=5)
    # the second pass asked only for the two that needed it, with n=1 and the short budget
    asked, cont_cfg = sampler.calls[0]
    assert asked == ["P1loop loop" + suffix, "P3more loop" + suffix]
    assert cont_cfg.n == 1 and cont_cfg.max_tokens == 5 and cont_cfg.stop == ("</answer>",)
    # the finished one is untouched; the forced ones carry both parts and the continuation's reason
    assert out[1][0] == first[1][0] and not out[1][0].forced
    assert out[0][0].text == "loop loop" + suffix + " 7 </answer>"
    assert out[0][0].num_tokens == 11 and out[0][0].finish_reason == "stop" and out[0][0].forced
    assert out[2][0].finish_reason == "length" and out[2][0].forced
    assert stats == {
        "budget": 8,
        "suffix": suffix,
        "continue_tokens": 5,
        "num_completions": 3,
        "num_forced": 2,
        "num_forced_finished": 1,
    }
    # nothing to force: no second call
    out2, stats2 = force_stop(sampler, ["P2"], [first[1]], cfg, suffix=suffix)
    assert out2 == [first[1]] and stats2["num_forced"] == 0 and len(sampler.calls) == 1


def test_forced_answers_are_graded_and_counted_in_the_summary():
    es = EvalSet(
        name="gsm8k",
        answer_type="numeric",
        source="test",
        problems=[Problem(id="a", question="1+1?", answer="2", meta={})],
    )
    prompt = build_prompt(es.problems[0], "think_tag", "gsm8k")
    assert prompt.force_suffix == "\n</think>\n<answer>"
    forced = Completion("1 plus 1" + prompt.force_suffix + " 2 </answer>", 12, "stop", forced=True)
    results = grade_completions(es, [[forced]], completion_prefix=prompt.completion_prefix)
    rec = results[0].samples[0]
    assert rec.correct and rec.format_ok and rec.forced_stop and rec.extraction_source == "tag"
    s = summarize(results)
    assert s["forced_stop_rate"] == 1.0 and s["forced_stop_accuracy"] == 1.0
    assert s["forced_stop_finished_rate"] == 1.0
    # a file written before the field existed still loads: the default is False
    assert json.loads(json.dumps({"forced_stop": rec.forced_stop}))["forced_stop"] is True


def test_cli_flags_and_the_template_guard():
    base = ["--model", "m", "--out", "o"]
    a = parse_args(base)
    assert a.force_stop_at is None and a.force_stop_continue == 128
    a = parse_args([*base, "--force-stop-at", "1536", "--force-stop-continue", "64"])
    assert a.force_stop_at == 1536 and a.force_stop_continue == 64
    # the few-shot templates have no closing tags to force
    assert (
        build_prompt(
            Problem(id="a", question="q", answer="1", meta={}), "zero_shot", "gsm8k"
        ).force_suffix
        == ""
    )

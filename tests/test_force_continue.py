"""The Wait pass (EXP-021): continuation before the forced stop, and its bookkeeping."""

from rlvr.eval.datasets import EvalSet, Problem
from rlvr.eval.force_stop import force_continue, force_stop
from rlvr.eval.prompts import build_prompt
from rlvr.eval.report import grade_completions, summarize
from rlvr.eval.run_eval import parse_args
from rlvr.eval.sampler import Completion, Sampler, SamplingConfig


class ScriptedSampler(Sampler):
    name = "scripted"

    def __init__(self, script):
        self.script, self.calls = script, []

    def generate(self, prompts, cfg):
        self.calls.append((list(prompts), cfg))
        return [[self.script[p]] for p in prompts]


def test_force_continue_then_force_stop():
    suffix = "\n</think>\n<answer>"
    prompts = ["P1", "P2", "P3"]
    first = [
        [Completion("loop", 8, "length")],
        [Completion("done\n</think>\n<answer> 4 </answer>", 9, "stop")],
        [Completion("stuck", 8, "length")],
    ]
    sampler = ScriptedSampler(
        {
            # P1: the nudge makes it close the think block and answer on its own
            "P1loop\nWait": Completion(" no, 7\n</think>\n<answer> 7 </answer>", 6, "stop"),
            # P3: the nudge produces more loop; then the forced stop writes the answer
            "P3stuck\nWait": Completion(" still stuck", 4, "length"),
            "P3stuck\nWait still stuck" + suffix: Completion(" 1 </answer>", 3, "stop"),
        }
    )
    cfg = SamplingConfig(n=1, temperature=0.0, max_tokens=8, stop=("</answer>",))
    out, cstats = force_continue(sampler, prompts, first, cfg, text="\nWait", continue_tokens=4)
    assert (
        cstats["num_continued"] == 2
        and cstats["num_closed_think"] == 1
        and cstats["num_finished"] == 1
    )
    asked, ccfg = sampler.calls[0]
    assert asked == ["P1loop\nWait", "P3stuck\nWait"] and ccfg.max_tokens == 4 and ccfg.n == 1
    assert out[0][0].continued and out[0][0].finish_reason == "stop" and not out[0][0].forced
    assert out[1][0] == first[1][0]
    assert out[2][0].continued and out[2][0].finish_reason == "length"
    out2, fstats = force_stop(sampler, prompts, out, cfg, suffix=suffix, continue_tokens=3)
    # only P3 still needed the stop; it keeps the continued flag and gains the forced one
    assert fstats["num_forced"] == 1
    assert out2[2][0].forced and out2[2][0].continued and out2[2][0].finish_reason == "stop"
    assert out2[2][0].text == "stuck\nWait still stuck" + suffix + " 1 </answer>"
    assert not out2[0][0].forced


def test_records_and_summary_carry_the_continue_flag():
    es = EvalSet(
        name="gsm8k",
        answer_type="numeric",
        source="t",
        problems=[Problem(id="a", question="q", answer="7", meta={})],
    )
    prompt = build_prompt(es.problems[0], "think_tag", "gsm8k")
    comp = Completion("x\nWait y\n</think>\n<answer> 7 </answer>", 10, "stop", continued=True)
    res = grade_completions(es, [[comp]], completion_prefix=prompt.completion_prefix)
    rec = res[0].samples[0]
    assert rec.correct and rec.forced_continue and not rec.forced_stop
    s = summarize(res)
    assert s["forced_continue_rate"] == 1.0 and s["forced_continue_accuracy"] == 1.0
    assert s["forced_stop_rate"] == 0.0


def test_cli_flags():
    a = parse_args(["--model", "m", "--out", "o"])
    assert a.force_continue is None and a.force_continue_tokens == 512
    a = parse_args(
        [
            "--model",
            "m",
            "--out",
            "o",
            "--force-stop-at",
            "1536",
            "--force-continue",
            "\\nWait",
            "--force-continue-tokens",
            "64",
        ]
    )
    assert a.force_continue == "\\nWait" and a.force_continue_tokens == 64

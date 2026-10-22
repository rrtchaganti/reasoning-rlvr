"""Budget forcing for the think-tag format (EXP-016, Phase 5).

Phase 4 found that every post-SFT gain in the trained format sits on problems the previous policy
failed to terminate (the loops do not finish at 8,192 tokens either). If the policies learnt
"stop and answer", a decoding rule that forces the stop may buy the same points with no training:
generate at most ``B`` tokens; every completion that hit ``B`` without closing its think block gets
the template's closing tags appended (``Prompt.force_suffix``, ``"\\n</think>\\n<answer>"``) and up
to ``continue_tokens`` more tokens to write the answer. The forced completion's text and token count
include both parts, its ``finish_reason`` is the continuation's, and ``forced=True`` marks it so the
per-sample file and the summary can say how many answers were forced and how many were right.

This is the s1 "budget forcing" move applied as a stop (not as a "wait"): it reads what the loop
already holds rather than asking the model to go on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from rlvr.eval.sampler import Completion, Sampler, SamplingConfig

THINK_CLOSE = "</think>"


def needs_forcing(comp: Completion, close_tag: str = THINK_CLOSE) -> bool:
    """Hit the budget and never closed the think block."""
    return comp.finish_reason == "length" and close_tag not in comp.text


def force_stop(
    sampler: Sampler,
    prompts: Sequence[str],
    completions: list[list[Completion]],
    cfg: SamplingConfig,
    *,
    suffix: str,
    continue_tokens: int = 128,
) -> tuple[list[list[Completion]], dict[str, Any]]:
    """Second pass over the completions that need forcing; returns the new completions and stats.

    ``cfg`` is the first pass's config (its ``max_tokens`` is the budget B); the continuation uses
    the same sampling parameters with ``n=1`` and ``max_tokens=continue_tokens``.
    """
    if not suffix:
        raise ValueError("force_stop needs the template's closing suffix (think_tag only)")
    if len(prompts) != len(completions):
        raise ValueError("one completion list per prompt")
    targets = [
        (i, j)
        for i, comps in enumerate(completions)
        for j, comp in enumerate(comps)
        if needs_forcing(comp)
    ]
    stats: dict[str, Any] = {
        "budget": cfg.max_tokens,
        "suffix": suffix,
        "continue_tokens": continue_tokens,
        "num_completions": sum(len(c) for c in completions),
        "num_forced": len(targets),
        "num_forced_finished": 0,
    }
    if not targets:
        return completions, stats
    cont_prompts = [prompts[i] + completions[i][j].text + suffix for i, j in targets]
    cont_cfg = replace(cfg, n=1, max_tokens=continue_tokens)
    conts = sampler.generate(cont_prompts, cont_cfg)
    out = [list(comps) for comps in completions]
    for (i, j), cont in zip(targets, conts, strict=True):
        tail = cont[0]
        orig = completions[i][j]
        out[i][j] = Completion(
            text=orig.text + suffix + tail.text,
            num_tokens=orig.num_tokens + tail.num_tokens,
            finish_reason=tail.finish_reason,
            forced=True,
        )
        stats["num_forced_finished"] += int(tail.finish_reason != "length")
    return out, stats

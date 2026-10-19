"""Generation back-ends behind one interface: vLLM for real runs, HF for CPU tests.

The harness never calls a model directly; it asks a sampler for ``n`` completions per prompt
with fixed sampling parameters and gets back text, token count and the finish reason, which
is what the truncation-rate metric needs.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SamplingConfig:
    n: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 4096
    seed: int = 0
    stop: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "stop": list(self.stop),
        }


@dataclass
class Completion:
    text: str
    num_tokens: int
    finish_reason: str  # "stop" | "length" | other backend-specific values

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def truncate_at_stop(text: str, stops: Sequence[str]) -> tuple[str, bool]:
    """Cut ``text`` after the earliest stop string, keeping the stop string itself.

    Mirrors vLLM with ``include_stop_str_in_output=True`` so that both backends hand identical
    text to the extractor (the ``</answer>`` tag must survive for the tag rule to fire).
    """
    best = -1
    best_len = 0
    for s in stops:
        k = text.find(s)
        if k >= 0 and (best < 0 or k < best):
            best, best_len = k, len(s)
    if best < 0:
        return text, False
    return text[: best + best_len], True


def vllm_env_overrides(*, batch_invariant: bool) -> dict[str, str]:
    """Environment vLLM must see before its engine is built.

    ``VLLM_BATCH_INVARIANT=1`` makes greedy decoding bit-identical across runs. EXP-000
    (14 Oct 2026, Qwen3-0.6B-Base, RTX 4060): without it three identical T=0 runs produced
    different text on 16 / 55 / 69 of 100 problems (GSM8K / MATH-500 / Countdown) and pass@1
    moved by up to 6 points; with it, two runs agreed on all 300 completions. Cost: about
    13 % more sampling time and one extra minute of engine start-up.
    """
    return {"VLLM_BATCH_INVARIANT": "1" if batch_invariant else "0"}


def context_budget(
    prompt_lens: Sequence[int], max_tokens: int, max_model_len: int | None
) -> dict | None:
    """What the context window leaves for new tokens, per eval set.

    vLLM stops a completion at ``max_model_len - len(prompt)`` tokens when that is below
    ``max_tokens``, silently. A config that says ``max_tokens: 4096`` with ``max_model_len: 4096``
    therefore gives 4096 minus the prompt: 3,865-3,994 new tokens on the Phase 2 base reference
    (journal, section 14). The harness prints this before sampling and writes it into
    ``metrics.json`` so the effective budget is read off the run, not guessed from the config.
    """
    if max_model_len is None or not prompt_lens:
        return None
    longest, shortest = max(prompt_lens), min(prompt_lens)
    return {
        "max_model_len": int(max_model_len),
        "max_tokens": int(max_tokens),
        "prompt_tokens_min": int(shortest),
        "prompt_tokens_max": int(longest),
        "effective_max_tokens_min": int(min(max_tokens, max_model_len - longest)),
        "effective_max_tokens_max": int(min(max_tokens, max_model_len - shortest)),
        "capped_by_context": bool(max_model_len - longest < max_tokens),
    }


class Sampler:
    """Interface. ``generate`` returns one list of ``n`` completions per prompt, in order."""

    name = "base"
    max_model_len: int | None = None

    def generate(self, prompts: Sequence[str], cfg: SamplingConfig) -> list[list[Completion]]:
        raise NotImplementedError

    def prompt_token_counts(self, prompts: Sequence[str]) -> list[int]:
        """Prompt lengths in the backend's own tokenisation; empty when the backend cannot say."""
        return []

    def context_budget(self, prompts: Sequence[str], max_tokens: int) -> dict | None:
        return context_budget(self.prompt_token_counts(prompts), max_tokens, self.max_model_len)

    def describe(self) -> dict:
        return {"backend": self.name}


class VLLMSampler(Sampler):
    name = "vllm"

    def __init__(
        self,
        model: str,
        *,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 6144,
        tensor_parallel_size: int = 1,
        enable_prefix_caching: bool = True,
        revision: str | None = None,
        seed: int = 0,
        batch_invariant: bool = True,
    ) -> None:
        # vllm.envs reads os.environ lazily, so this works even if vllm was imported earlier,
        # but it must happen before the engine (and its worker processes) is created.
        os.environ.update(vllm_env_overrides(batch_invariant=batch_invariant))
        from vllm import LLM  # heavy import kept local

        self.model = model
        self.revision = revision
        self.batch_invariant = batch_invariant
        self.max_model_len = max_model_len
        self.llm = LLM(
            model=model,
            revision=revision,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=enable_prefix_caching,
            seed=seed,
        )

    def generate(self, prompts: Sequence[str], cfg: SamplingConfig) -> list[list[Completion]]:
        from vllm import SamplingParams

        params = SamplingParams(
            n=cfg.n,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            max_tokens=cfg.max_tokens,
            seed=cfg.seed,
            stop=list(cfg.stop) or None,
            # Keep the stop string: the think_tag template stops on "</answer>", and the
            # extraction regex needs the closing tag to be present in the text.
            include_stop_str_in_output=True,
        )
        outputs = self.llm.generate(list(prompts), params, use_tqdm=True)
        results: list[list[Completion]] = []
        for out in outputs:
            results.append(
                [
                    Completion(
                        text=o.text, num_tokens=len(o.token_ids), finish_reason=str(o.finish_reason)
                    )
                    for o in out.outputs
                ]
            )
        return results

    def prompt_token_counts(self, prompts: Sequence[str]) -> list[int]:
        tok = self.llm.get_tokenizer()
        return [len(tok.encode(p)) for p in prompts]

    def describe(self) -> dict:
        import vllm

        return {
            "backend": self.name,
            "model": self.model,
            "revision": self.revision,
            "vllm": vllm.__version__,
            "batch_invariant": self.batch_invariant,
        }


class HFSampler(Sampler):
    """Plain ``transformers`` generation. Slow; used for CPU tests and tiny debugging runs."""

    name = "hf"

    def __init__(
        self,
        model,
        tokenizer,
        *,
        device: str = "cpu",
        batch_size: int = 8,
        max_model_len: int | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.tok = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.max_model_len = max_model_len  # informational only; HF generation is not clipped by it
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

    def generate(self, prompts: Sequence[str], cfg: SamplingConfig) -> list[list[Completion]]:
        import torch

        results: list[list[Completion]] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = list(prompts[start : start + self.batch_size])
            enc = self.tok(batch, return_tensors="pt", padding=True).to(self.device)
            torch.manual_seed(cfg.seed + start)
            gen_kwargs = {
                "max_new_tokens": cfg.max_tokens,
                "do_sample": cfg.temperature > 0,
                "num_return_sequences": cfg.n,
                "pad_token_id": self.tok.pad_token_id,
            }
            if cfg.temperature > 0:
                gen_kwargs.update(temperature=cfg.temperature, top_p=cfg.top_p)
                if cfg.top_k > 0:
                    gen_kwargs["top_k"] = cfg.top_k
            with torch.no_grad():
                out = self.model.generate(**enc, **gen_kwargs)
            new = out[:, enc["input_ids"].shape[1] :]
            for i in range(len(batch)):
                comps = []
                for j in range(cfg.n):
                    ids = new[i * cfg.n + j]
                    text = self.tok.decode(ids, skip_special_tokens=True)
                    hit_eos = bool((ids == self.tok.eos_token_id).any())
                    finish = "stop" if hit_eos else "length"
                    text, hit_stop = truncate_at_stop(text, cfg.stop)
                    if hit_stop:
                        finish = "stop"
                    n_tok = int((ids != self.tok.pad_token_id).sum())
                    comps.append(Completion(text=text, num_tokens=n_tok, finish_reason=finish))
                results.append(comps)
        return results

    def prompt_token_counts(self, prompts: Sequence[str]) -> list[int]:
        return [len(self.tok(p).input_ids) for p in prompts]

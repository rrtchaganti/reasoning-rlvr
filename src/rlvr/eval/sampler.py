"""Generation back-ends behind one interface: vLLM for real runs, HF for CPU tests.

The harness never calls a model directly; it asks a sampler for ``n`` completions per prompt
with fixed sampling parameters and gets back text, token count and the finish reason, which
is what the truncation-rate metric needs.
"""

from __future__ import annotations

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


class Sampler:
    """Interface. ``generate`` returns one list of ``n`` completions per prompt, in order."""

    name = "base"

    def generate(self, prompts: Sequence[str], cfg: SamplingConfig) -> list[list[Completion]]:
        raise NotImplementedError

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
    ) -> None:
        from vllm import LLM  # heavy import kept local

        self.model = model
        self.revision = revision
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
            include_stop_str_in_output=False,
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

    def describe(self) -> dict:
        import vllm

        return {
            "backend": self.name,
            "model": self.model,
            "revision": self.revision,
            "vllm": vllm.__version__,
        }


class HFSampler(Sampler):
    """Plain ``transformers`` generation. Slow; used for CPU tests and tiny debugging runs."""

    name = "hf"

    def __init__(self, model, tokenizer, *, device: str = "cpu", batch_size: int = 8) -> None:
        self.model = model.to(device).eval()
        self.tok = tokenizer
        self.device = device
        self.batch_size = batch_size
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
                    for s in cfg.stop:
                        k = text.find(s)
                        if k >= 0:
                            text, finish = text[:k], "stop"
                    n_tok = int((ids != self.tok.pad_token_id).sum())
                    comps.append(Completion(text=text, num_tokens=n_tok, finish_reason=finish))
                results.append(comps)
        return results

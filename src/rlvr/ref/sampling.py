"""A minimal single-sequence decoding loop over a Hugging Face causal LM with a KV cache.

This exists to be read and explained (PLAN.md section 10.2 drill), not for throughput; the
harness uses vLLM. It makes the two facts of autoregressive decoding visible:

1. **Prefill** runs the whole prompt once and stores keys/values for every layer.
2. **Decode** feeds one token per step and only computes attention for that token against
   the cached keys/values, which is why decode is memory-bandwidth bound and prefill is
   compute bound.

Sampling: temperature scales logits before softmax; top-k keeps the k largest logits; top-p
keeps the smallest prefix of the sorted distribution whose mass reaches p (nucleus). With
``temperature == 0`` the loop is greedy and must agree token-for-token with
``model.generate(do_sample=False)``; the test checks exactly that.
"""

from __future__ import annotations

import torch


def _filter_logits(logits: torch.Tensor, *, top_k: int | None, top_p: float | None) -> torch.Tensor:
    """Set logits outside the top-k / nucleus to -inf. Operates on a [V] vector."""
    if top_k is not None and 0 < top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # keep the first token whose cumulative mass crosses p, drop everything after it
        drop = cum - torch.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits = torch.where(
            drop, torch.full_like(sorted_logits, float("-inf")), sorted_logits
        )
        logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)
    return logits


@torch.no_grad()
def sample_sequence(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | list[int] | None,
    temperature: float = 0.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Generate a continuation for one prompt. Returns the new tokens only, shape [L].

    Args:
        model: any HF causal LM supporting ``past_key_values`` and ``use_cache``.
        input_ids: [1, P] prompt token ids on the model's device.
        max_new_tokens: hard cap; generation also stops at EOS.
        eos_token_id: one id or a list of ids that end generation.
        temperature: 0 means greedy (argmax); otherwise logits are divided by it.
        top_k, top_p: optional truncations applied after temperature.
        generator: torch RNG for reproducible sampling.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("input_ids must be [1, P]; this loop handles one sequence")
    eos = set()
    if eos_token_id is not None:
        eos = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id)

    # --- prefill: whole prompt in one forward pass, cache every layer's K/V ---
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_logits = out.logits[0, -1, :]

    generated: list[int] = []
    for _ in range(max_new_tokens):
        if temperature <= 0.0:
            next_id = int(torch.argmax(next_logits))
        else:
            logits = _filter_logits(next_logits / temperature, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits.float(), dim=-1)
            next_id = int(torch.multinomial(probs, 1, generator=generator))
        generated.append(next_id)
        if next_id in eos:
            break
        # --- decode: one token in, attend over the cache, cache grows by one position ---
        step = torch.tensor([[next_id]], device=input_ids.device, dtype=input_ids.dtype)
        out = model(input_ids=step, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[0, -1, :]

    return torch.tensor(generated, dtype=input_ids.dtype, device=input_ids.device)

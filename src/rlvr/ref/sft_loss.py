"""From-scratch reference for the SFT objective (PLAN.md Phase 2; PHASES_IO_AND_HYPERPARAMS.md).

    L = -(1 / N_resp) * sum_{t in response} log p(y_t | y_<t, x)

Three pieces, each small enough to test exactly against a tiny model:

- :func:`completion_labels` masks the prompt (and padding) with -100 so only response tokens
  carry loss ("assistant-only" / "completion-only" masking).
- :func:`masked_cross_entropy` is the token-level mean over the labelled positions, i.e. what
  TRL's ``SFTTrainer`` computes with ``completion_only_loss=True`` and ``loss_type="nll"``.
- :func:`pack_sequences` + :func:`block_diagonal_causal_mask` implement packing *without*
  cross-contamination: several sequences share one row, positions restart at 0 for each
  sequence, and the attention mask is causal *within* each sequence and zero across them.
  ``tests/test_ref_sft_loss.py`` shows that this reproduces the unpacked per-token losses and
  that a plain causal mask over the packed row does not (the "packing without proper masks
  inflates loss" failure from PLAN.md, made concrete).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812

IGNORE = -100


def completion_labels(
    input_ids: torch.Tensor, prompt_lens: Sequence[int], attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Labels equal to ``input_ids`` on response tokens and -100 on prompt and padding tokens.

    ``prompt_lens[i]`` is the number of leading tokens of row ``i`` that belong to the prompt.
    The shift by one (predict token t from positions < t) is done inside the loss, not here, so
    ``labels[i, t] == input_ids[i, t]`` for every response position t.
    """
    labels = input_ids.clone()
    for i, n in enumerate(prompt_lens):
        labels[i, :n] = IGNORE
    if attention_mask is not None:
        labels[attention_mask == 0] = IGNORE
    return labels


def masked_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, *, reduction: str = "mean"
) -> torch.Tensor:
    """Causal-LM loss: ``logits[:, t]`` predicts ``labels[:, t + 1]``; positions whose shifted
    label is -100 contribute nothing. ``reduction="mean"`` divides by the number of labelled
    tokens (token-level mean, the SFT definition above); ``"sum"`` returns the total NLL;
    ``"none"`` returns the per-token NLL at the labelled positions, flattened, in order."""
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    nll = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=IGNORE,
        reduction="none",
    )
    keep = shift_labels.reshape(-1) != IGNORE
    if reduction == "none":
        return nll[keep]
    if reduction == "sum":
        return nll[keep].sum()
    n = keep.sum().clamp(min=1)
    return nll[keep].sum() / n


# --------------------------------------------------------------------------- packing


@dataclass
class PackedBatch:
    input_ids: torch.Tensor  # (1, T)
    labels: torch.Tensor  # (1, T)
    position_ids: torch.Tensor  # (1, T), restarting at 0 for every packed sequence
    seq_lens: list[int]  # lengths of the packed sequences, in row order
    attention_mask: torch.Tensor  # (1, 1, T, T) boolean: True where attention is allowed


def block_diagonal_causal_mask(seq_lens: Sequence[int], device=None) -> torch.Tensor:
    """(1, 1, T, T) boolean mask: query t may attend key s iff both are in the same packed
    sequence and s <= t. This is what flash-attention's varlen kernel does implicitly from
    ``cu_seqlens``; written out so it can be checked."""
    total = int(sum(seq_lens))
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    start = 0
    for n in seq_lens:
        block = torch.tril(torch.ones(n, n, dtype=torch.bool, device=device))
        mask[start : start + n, start : start + n] = block
        start += n
    return mask[None, None]


def additive_mask(allowed: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Boolean "may attend" mask -> additive float mask (0 where allowed, the dtype's most
    negative value elsewhere), the form the eager / SDPA attention paths add to the scores.
    Passing the boolean tensor itself to a model is not safe: an eager path that *adds* the
    mask would add 1.0 / 0.0 instead of masking (found by the packing test, 15 Oct 2026)."""
    out = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    return out.masked_fill(~allowed, torch.finfo(dtype).min)


def pack_sequences(
    sequences: Sequence[Sequence[int]],
    prompt_lens: Sequence[int],
    *,
    max_length: int,
) -> list[PackedBatch]:
    """First-fit packing in the given order (the test does not need best-fit-decreasing; TRL's
    ``bfd`` differs only in which sequences share a row, not in the masking). A sequence longer
    than ``max_length`` raises: the data pipeline must filter by length first (PLAN.md Phase 2,
    "traces longer than the max length truncated mid-answer")."""
    rows: list[tuple[list[list[int]], list[int]]] = []
    for seq, pl in zip(sequences, prompt_lens, strict=True):
        if len(seq) > max_length:
            raise ValueError(f"sequence of length {len(seq)} exceeds max_length {max_length}")
        for seqs, pls in rows:
            if sum(len(s) for s in seqs) + len(seq) <= max_length:
                seqs.append(list(seq))
                pls.append(pl)
                break
        else:
            rows.append(([list(seq)], [pl]))
    out: list[PackedBatch] = []
    for seqs, pls in rows:
        ids = torch.tensor([t for s in seqs for t in s])[None]
        lens = [len(s) for s in seqs]
        pos = torch.cat([torch.arange(n) for n in lens])[None]
        labels = ids.clone()
        start = 0
        for n, pl in zip(lens, pls, strict=True):
            labels[0, start : start + pl] = IGNORE
            start += n
        out.append(PackedBatch(ids, labels, pos, lens, block_diagonal_causal_mask(lens)))
    return out


def packed_cross_entropy(logits: torch.Tensor, packed: PackedBatch) -> torch.Tensor:
    """Per-token NLL at the labelled positions of a packed row. The shift must not cross a
    sequence boundary: the last token of sequence i must not be asked to predict the first
    token of sequence i + 1, so the label right after each boundary is dropped by making the
    boundary position's *next* label ignored for the shift."""
    labels = packed.labels.clone()
    start = 0
    for n in packed.seq_lens[:-1]:
        start += n
        labels[0, start] = IGNORE  # first token of the next sequence is never a target
    return masked_cross_entropy(logits, labels, reduction="none")
